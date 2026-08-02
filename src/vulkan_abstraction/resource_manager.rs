use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use ash::vk;

use crate::error::{ErrorSource, SrError};
use crate::render_graph::graph::RenderGraph;
use crate::render_graph::resource::Handle;
use crate::vulkan_abstraction::image::sampler::SamplerDesc;
use crate::vulkan_abstraction::{AccelerationStructure, ArenaBuffer, AsBuildJob, Buffer, EntityGpuData, Material, RawBuffer};
use crate::{error::SrResult, vulkan_abstraction};
use vk_sync_fork as vk_sync;

const ARENA_CAPACITY: vk::DeviceSize = 4096 * 16;

//TODO handle growable

/// Which arena a queued staging→GPU copy targets. Copies are queued at asset-load
/// time but recorded in a *later* frame's graph, and a `Handle` is only valid for
/// one graph build — so the queue stores this selector and resolves it to the
/// current build's handle in [`ResourceManager::take_queued_copies`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
enum ArenaId {
    MeshInfo,
    EmissiveTriangles,
}

/// Deferred work executed at the start of a specific absolute frame (see
/// [`ResourceManager::start_of_frame`]).
type FrameCallback<K> = Box<dyn FnOnce(&mut ResourceManager<K>) -> SrResult<()>>;

/// The raw per-frame data resolved from the caller's `(key, transforms)`
/// instance list. The renderer uploads these into CpuToGpu buffers created on
/// the spot each frame (and deferred-freed through the end-of-frame
/// callbacks) — nothing per-frame is retained in the manager.
pub(crate) struct FrameInstanceData {
    pub as_instances: Vec<vk::AccelerationStructureInstanceKHR>,
    /// Flat per-instance transforms in instance order;
    /// `EmissiveIndirectionEntry::entity_id` indexes into this list.
    pub transforms: Vec<vk::TransformMatrixKHR>,
    /// Dense `(emissive triangle slot, instance index)` table for NEE sampling.
    pub emissive_entries: Vec<vulkan_abstraction::gltf::EmissiveIndirectionEntry>,
}

//TODO there is structural decision to make on what to save and how cause raster needs a different way to store the geometry data probabibly with a features set constable?
/// Owns the *stable* GPU-side scene resources, keyed by the caller-provided
/// `K` (one key per BLAS / image): geometry + material data lives in arena
/// buffers whose slots survive across frames, plus the TLAS (rebuilt per frame
/// from the renderer's local instance buffer). Per-frame data only passes
/// through [`Self::frame_instance_data`] — it is never stored here.
pub(crate) struct ResourceManager<K: Hash + Eq + Copy> {
    tlas: vulkan_abstraction::Tlas,

    // ── Stable per-asset GPU data, keyed by `K`.
    /// Per-BLAS mesh info (vertex/index BDA + material). The slot index is the
    /// `gl_InstanceCustomIndexEXT` every instance of that BLAS uses.
    meshes_info: vulkan_abstraction::ArenaGpuBuffer<EntityGpuData>,
    blas_emissive_triangles: vulkan_abstraction::ArenaGpuBuffer<vulkan_abstraction::gltf::EmissiveTriangle>,
    blases: HashMap<K, vulkan_abstraction::Blas>,
    /// Key → slot in `meshes_info`.
    mesh_info_slots: HashMap<K, u32>,
    /// Key → slots of the BLAS's triangles in `blas_emissive_triangles`.
    emissive_triangle_slots: HashMap<K, Vec<u32>>,
    images: HashMap<K, vulkan_abstraction::Image>,
    /// Finite set of samplers, deduplicated by their description: glTF samplers
    /// don't need to be unique per texture. Never shrinks.
    samplers: HashMap<SamplerDesc, vulkan_abstraction::Sampler>,
    default_sampler: vulkan_abstraction::Sampler,

    /// Pending staging→GPU copy regions for the arena buffers, produced by asset
    /// loads. Drained each frame by [`Self::take_queued_copies`] and recorded as a
    /// transfer prologue in the render graph (`RenderGraph::add_prologue_buffer_copies`),
    /// so the copy rides the frame's submission ordered before the shader reads.
    buffer_copies_queued: Vec<(ArenaId, vk::BufferCopy)>,
    /// Deferred work keyed by the absolute frame at whose start it must run
    /// (currently just deferred arena slot frees scheduled by `remove`). Drained
    /// by [`Self::start_of_frame`] — nothing runs unconditionally every frame.
    //TODO the free callbacks should key off GPU completion (the frame timeline)
    //     rather than the CPU-side frame counter; the renderer's per-frame
    //     slot-reuse wait currently makes the CPU-counter timing safe.
    start_of_frame_callbacks: Vec<(u64, FrameCallback<K>)>,
    /// BLAS build jobs produced by [`vulkan_abstraction::Blas::new_deferred`] at
    /// asset-load time, waiting to be recorded into the next frame's graph by
    /// [`Self::queue_blas_builds`]. Each entry's BLAS is already registered in
    /// `blases` (so its device address is valid); only the build recording is
    /// pending.
    pending_blas_builds: Vec<(K, AsBuildJob)>,

    core: Rc<vulkan_abstraction::Core>,
}

// `K: 'static` because deferred frame work is stored as boxed `FnOnce(&mut Self)`.
impl<K: Hash + Eq + Copy + 'static> ResourceManager<K> {
    pub fn new_empty(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        // SHADER_DEVICE_ADDRESS so the heap path can compute the buffer's BDA
        // when allocating a storage-buffer descriptor (`Buffer::storage_slot`
        // internally calls `vkGetBufferDeviceAddress`).
        let meshes_info = vulkan_abstraction::ArenaGpuBuffer::new(
            core.clone(),
            ARENA_CAPACITY,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "Meshes info GPU buffer",
        )?;

        let blas_emissive_triangles = vulkan_abstraction::ArenaGpuBuffer::new(
            core.clone(),
            ARENA_CAPACITY,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "blas emissive triangles",
        )?;

        // Temporary one-element buffer for the initial empty TLAS build (the
        // build is synchronous, so dropping it right after is fine). Per-frame
        // instance buffers are created by the renderer each frame.
        let empty_instances_buffer = vulkan_abstraction::StagingBuffer::<vk::AccelerationStructureInstanceKHR>::new(
            Rc::clone(&core),
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "empty TLAS build instances",
        )?;

        // Build over 0 instances (the dummy buffer just keeps the build input
        // non-null). `SometimesChanges` == the pre-rework
        // `PREFER_FAST_TRACE | ALLOW_UPDATE` TLAS flags.
        let tlas = vulkan_abstraction::Tlas::new(
            Rc::clone(&core),
            &empty_instances_buffer,
            0,
            vulkan_abstraction::BuildType::SometimesChanges,
        )?;

        let default_sampler = vulkan_abstraction::Sampler::new(
            Rc::clone(&core),
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerMipmapMode::LINEAR,
        )?;

        Ok(Self {
            tlas,

            meshes_info,
            blas_emissive_triangles,
            blases: HashMap::new(),
            mesh_info_slots: HashMap::new(),
            emissive_triangle_slots: HashMap::new(),
            images: HashMap::new(),
            samplers: HashMap::new(),
            default_sampler,

            buffer_copies_queued: vec![],
            start_of_frame_callbacks: vec![],
            pending_blas_builds: vec![],
            core,
        })
    }

    // ─── Start-of-frame scheduling ───────────────────────────────────────────

    /// Absolute frame number the next rendered frame will carry.
    fn next_frame(&self) -> u64 {
        *self.core.absolute_frame_count.borrow() as u64 + 1
    }

    /// Schedule `callback` to run at the start of frame `frame` (or the first
    /// `start_of_frame` call at/after it).
    fn schedule_at_frame(&mut self, frame: u64, callback: impl FnOnce(&mut Self) -> SrResult<()> + 'static) {
        self.start_of_frame_callbacks.push((frame, Box::new(callback)));
    }

    /// Run the deferred work due at the start of `upcoming_frame` (the frame
    /// about to be rendered): currently the arena slot-free processing scheduled
    /// by `remove`. A callback may schedule further callbacks; ones due this same
    /// frame run before this returns. (Arena copies now ride the render graph — see
    /// [`Self::take_queued_copies`].)
    pub fn start_of_frame(&mut self, upcoming_frame: u64) -> SrResult<()> {
        let mut i = 0;
        while i < self.start_of_frame_callbacks.len() {
            if self.start_of_frame_callbacks[i].0 <= upcoming_frame {
                let (_, callback) = self.start_of_frame_callbacks.remove(i);
                callback(self)?;
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Import the arena buffers into the *current* graph build, returning their
    /// fresh handles. Graph handles die with `RenderGraph::reset`, so this must run
    /// once per rebuild, right after `reset` and before [`Self::take_queued_copies`].
    /// The returned handles are what consumers (the RT passes) declare reads on, so
    /// the prologue copy is ordered before — and barriered against — the trace.
    pub fn import_to_graph(&mut self, rg: &mut RenderGraph) -> [Handle<RawBuffer>; 2] {
        [self.meshes_info.import_into(rg), self.blas_emissive_triangles.import_into(rg)]
    }

    /// Drain the queued arena staging→GPU copies, collapsing redundant writes to
    /// the same destination region (keeping the latest) so the graph never records
    /// two unordered copies to one slot. The renderer records the result as a
    /// transfer prologue at the head of the frame's graph submission (ordered
    /// before the shader reads by [`RenderGraph::add_prologue_buffer_copies`]),
    /// replacing the old synchronous flush + device-wide idle wait.
    pub fn take_queued_copies(&mut self) -> SrResult<Vec<(&RawBuffer, Handle<RawBuffer>, vk::BufferCopy)>> {
        let copies = std::mem::take(&mut self.buffer_copies_queued);
        if copies.is_empty() {
            return Ok(vec![]);
        }

        // Keep only the newest write per destination region, so the graph never
        // records two unordered copies into the same slot.
        let mut latest_indices = std::collections::HashMap::new();
        for (i, (arena, region)) in copies.iter().enumerate() {
            latest_indices.insert((*arena, region.dst_offset, region.size), i);
        }
        // Sort the retained indices to preserve their relative submission order
        let mut kept_indices: Vec<usize> = latest_indices.into_values().collect();
        kept_indices.sort_unstable();

        let mut result = Vec::with_capacity(kept_indices.len());
        for i in kept_indices {
            let (arena, region) = copies[i];
            let (src, handle): (&RawBuffer, _) = match arena {
                ArenaId::MeshInfo => (self.meshes_info.staging().raw(), self.meshes_info.handle()),
                ArenaId::EmissiveTriangles => (self.blas_emissive_triangles.staging().raw(), self.blas_emissive_triangles.handle()),
            };
            let handle = handle.ok_or_else(|| {
                SrError::new(
                    ErrorSource::RenderGraph(crate::render_graph::error::GraphError::InvalidResourceRef),
                    format!("take_queued_copies: arena {arena:?} was not imported into this graph build"),
                )
            })?;
            result.push((src, handle.clone(), region));
        }

        Ok(result)
    }

    // ─── Per-frame data ──────────────────────────────────────────────────────

    /// Resolve the caller's per-frame `(key, transforms)` instance list into
    /// the raw arrays the frame needs: TLAS instances (custom index = the
    /// BLAS's stable mesh-info slot), the flat transform list (instance
    /// order), and the emissive indirection entries. Pure resolution — the
    /// renderer uploads the results into frame-local CpuToGpu buffers; nothing
    /// is stored here.
    pub fn frame_instance_data(&self, instances: &[(K, Vec<vk::TransformMatrixKHR>)]) -> SrResult<FrameInstanceData> {
        let mut as_instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::new();
        let mut transforms: Vec<vk::TransformMatrixKHR> = Vec::new();
        let mut emissive_entries: Vec<vulkan_abstraction::gltf::EmissiveIndirectionEntry> = Vec::new();
        let no_emissive: Vec<u32> = Vec::new();

        for (key, instance_transforms) in instances {
            // Validate the key is registered: `InstanceDesc::lower` resolves the
            // BLAS address through `blas_device_address`, which would otherwise
            // panic on an unknown key.
            if !self.blases.contains_key(key) {
                return Err(crate::error::SrError::new_custom(
                    "frame_instance_data: instance references a BLAS key that was never loaded".to_string(),
                ));
            }
            let mesh_info_slot = self.mesh_info_slots[key];
            let emissive_slots = self.emissive_triangle_slots.get(key).unwrap_or(&no_emissive);

            for transform in instance_transforms {
                let instance_index = transforms.len();

                transforms.push(*transform);

                // Plain-data instance description; the BLAS device address is
                // resolved late, inside `lower` (custom index = the BLAS's
                // stable mesh-info slot; hit_group_offset = 0, same hit group
                // for the whole scene; face culling disabled for simplicity).
                let desc = vulkan_abstraction::InstanceDesc {
                    blas: *key,
                    transform: *transform,
                    custom_index: mesh_info_slot,
                    mask: 0xFF,
                    sbt_offset: 0,
                    flags: vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
                };
                as_instances.push(desc.lower(self));

                for &tri_slot in emissive_slots {
                    emissive_entries.push(vulkan_abstraction::gltf::EmissiveIndirectionEntry {
                        blas_tri_index: tri_slot,
                        entity_id: instance_index as u32,
                    });
                }
            }
        }

        Ok(FrameInstanceData {
            as_instances,
            transforms,
            emissive_entries,
        })
    }

    /// Rebuild the TLAS from instances already written into `instances_buffer`
    /// (the renderer's frame-local buffer). Synchronous.
    #[allow(dead_code)]
    pub fn rebuild_tlas(&mut self, instance_count: u32, instances_buffer: &impl Buffer) -> SrResult<()> {
        self.tlas.rebuild_from_buffer(instance_count, instances_buffer)
    }

    // ─── Render-graph AS build queuing ───────────────────────────────────────

    /// Stash a deferred BLAS build job (from [`vulkan_abstraction::Blas::new_deferred`])
    /// to be recorded into the next frame's graph by [`Self::queue_blas_builds`].
    /// The BLAS must already be registered via [`Self::add_blas`].
    pub fn queue_blas_build_job(&mut self, key: K, job: AsBuildJob) {
        self.pending_blas_builds.push((key, job));
    }

    /// Record every pending BLAS build (from `new_deferred` at load time) into
    /// `rg` as a build pass — importing each BLAS so the TLAS build orders itself
    /// after them — and re-evaluate the rebuild/update heuristic for any built
    /// BLAS with no operation in flight. Returns, for each BLAS built this frame,
    /// its `(key, imported handle)`: the handles feed [`Self::queue_tlas_build`] as
    /// its dependencies, and the caller schedules each key's [`Self::mark_blas_built`]
    /// for when this frame's GPU work completes (the graph records the build but
    /// can't mutate this CPU-side heuristic state).
    pub fn queue_blas_builds(&mut self, rg: &mut RenderGraph) -> SrResult<Vec<(K, Handle<AccelerationStructure>)>> {
        let mut built = Vec::new();

        // 1. Pending initial builds. Their `op` was set by `new_deferred`; record
        //    the job now, and the caller folds the op in once this frame completes.
        let pending = std::mem::take(&mut self.pending_blas_builds);
        for (key, job) in pending {
            let Some(blas) = self.blases.get(&key) else {
                // Removed before its build could be recorded — drop the job.
                continue;
            };
            let handle = rg.import(blas.accel_arc());
            rg.add_as_build_pass("blas_build", &handle, &[], job)?;
            built.push((key, handle));
        }

        // 2. Steady-state heuristic for BLASes with nothing in flight. The renderer
        //    doesn't track per-BLAS geometry mutation yet (its meshes are static
        //    once loaded), so `inputs_changed = false` and `plan_op` yields `None`
        //    — nothing is queued. This loop is the seam a future animated/skinned
        //    mesh path drives an update/rebuild through.
        for blas in self.blases.values_mut() {
            if blas.op().is_none() {
                let _ = blas.plan_op(false);
            }
        }

        Ok(built)
    }

    /// Fold a recorded BLAS build's chosen op back into its heuristic state, once
    /// the frame that recorded it has completed on the GPU. No-op if the key was
    /// removed in the meantime. Scheduled by the renderer's end-of-frame drain.
    pub fn mark_blas_built(&mut self, key: K) {
        if let Some(blas) = self.blases.get_mut(&key) {
            blas.mark_built();
        }
    }

    /// Fold the recorded TLAS build's chosen op back into its heuristic state, once
    /// the frame that recorded it has completed on the GPU. Scheduled by the
    /// renderer's end-of-frame drain.
    pub fn mark_tlas_built(&mut self) {
        self.tlas.mark_built();
    }

    /// Record the TLAS build/update for this frame into `rg` (see
    /// [`vulkan_abstraction::Tlas::queue_build`] for the update-vs-rebuild choice),
    /// importing the structure so the build pass writes it, the BLAS builds in
    /// `blas_deps` are ordered before it, and the RT pass can declare a read after
    /// it. Returns the imported handle (for the RT pass's read) and the TLAS device
    /// address (for the RT push constant). The caller schedules [`Self::mark_tlas_built`]
    /// for when this frame's GPU work completes.
    pub fn queue_tlas_build(
        &mut self,
        rg: &mut RenderGraph,
        instance_count: u32,
        instances_buffer: &impl Buffer,
        blas_deps: &[Handle<AccelerationStructure>],
    ) -> SrResult<(Handle<AccelerationStructure>, vk::DeviceAddress)> {
        let (accel, address, job) = self.tlas.queue_build(instance_count, instances_buffer)?;
        // Import carrying the access the previous frame's RT trace left the TLAS in:
        // an in-place UPDATE writes the same structure the last frame read, so
        // `compile` needs this to emit the cross-frame read→build barrier (a full
        // rebuild lands on fresh memory, making the barrier harmlessly conservative).
        // This is what lets the build→trace hand-off be a barrier rather than a
        // device-wide idle once true frame overlap is enabled.
        let handle = rg.import_with_usage(accel, vk_sync::AccessType::RayTracingShaderReadAccelerationStructure);
        rg.add_as_build_pass("tlas_build", &handle, blas_deps, job)?;
        Ok((handle, address))
    }

    // ─── Asset management ────────────────────────────────────────────────────

    /// Register every asset of a loaded scene, assigning each BLAS and image a
    /// fresh key from `make_key`. Materials get their texture references
    /// resolved to descriptor-heap slots here (samplers are deduplicated into
    /// the manager's finite sampler set). Returns the BLAS keys, parallel to
    /// `blases`.
    pub fn add_scene_assets(
        &mut self,
        blases: Vec<crate::LoadedBlas>,
        textures: Vec<vulkan_abstraction::gltf::Texture>,
        sampler_descs: Vec<SamplerDesc>,
        images: Vec<vulkan_abstraction::Image>,
        make_key: &mut dyn FnMut() -> K,
    ) -> SrResult<Vec<K>> {
        let image_slots: Vec<u32> = images.iter().map(|image| image.sampled_slot()).collect();

        let mut sampler_slots = Vec::with_capacity(sampler_descs.len());
        for desc in &sampler_descs {
            sampler_slots.push(self.sampler_slot(desc)?);
        }
        let default_sampler_slot = self.default_sampler.slot();

        let resolve = |texture_index: Option<usize>| -> (u32, u32) {
            match texture_index {
                Some(i) => {
                    let texture = &textures[i];
                    let image_slot = image_slots[texture.source];
                    let sampler_slot = texture.sampler.map(|s| sampler_slots[s]).unwrap_or(default_sampler_slot);
                    (image_slot, sampler_slot)
                }
                None => (Material::NULL_TEXTURE_INDEX, Material::NULL_TEXTURE_INDEX),
            }
        };

        let mut keys = Vec::with_capacity(blases.len());
        for loaded in blases {
            let key = make_key();
            let material = Material::new(&loaded.material, &resolve);
            self.add_blas(key, loaded.blas, material, &loaded.emissive_triangles)?;
            keys.push(key);
        }

        for image in images {
            self.images.insert(make_key(), image);
        }

        Ok(keys)
    }

    /// Register a BLAS under `key`: uploads its mesh info (slot becomes the
    /// instance custom index) and its local-space emissive triangles.
    pub fn add_blas(
        &mut self,
        key: K,
        blas: vulkan_abstraction::Blas,
        material: Material,
        emissive_triangles: &[vulkan_abstraction::gltf::EmissiveTriangle],
    ) -> SrResult<()> {
        let gpu_data = EntityGpuData {
            vertex_buffer: blas.vertex_buffer().get_device_address(),
            index_buffer: blas.index_buffer().get_device_address(),
            material,
        };
        let (slot, copy_region) = self.meshes_info.allocate_and_update(&gpu_data)?;
        self.queue_copy(ArenaId::MeshInfo, copy_region);
        self.mesh_info_slots.insert(key, slot as u32);

        let mut tri_slots = Vec::with_capacity(emissive_triangles.len());
        for tri in emissive_triangles {
            let (tri_slot, tri_copy) = self.blas_emissive_triangles.allocate_and_update(tri)?;
            self.queue_copy(ArenaId::EmissiveTriangles, tri_copy);
            tri_slots.push(tri_slot as u32);
        }
        self.emissive_triangle_slots.insert(key, tri_slots);

        self.blases.insert(key, blas);
        Ok(())
    }

    /// Whether `key` currently has any asset (BLAS or image) registered.
    pub fn contains(&self, key: &K) -> bool {
        self.blases.contains_key(key) || self.images.contains_key(key)
    }

    /// Remove whatever asset `key` refers to (BLAS and/or image). Arena slots
    /// are deferred-freed (reclaimed by a start-of-frame callback scheduled for
    /// the frame at which no in-flight frame can still read them); the BLAS /
    /// image objects are dropped immediately, so the caller must guarantee the
    /// GPU is idle.
    pub fn remove(&mut self, key: &K) {
        let mut any_freed = false;
        if let Some(slot) = self.mesh_info_slots.remove(key) {
            self.meshes_info.free_index(slot as usize);
            any_freed = true;
        }
        if let Some(tri_slots) = self.emissive_triangle_slots.remove(key) {
            for slot in tri_slots {
                self.blas_emissive_triangles.free_index(slot as usize);
                any_freed = true;
            }
        }
        self.blases.remove(key);
        self.images.remove(key);

        if any_freed {
            // The arenas tag each free with the frame it happened on and only
            // reclaim once MAX_FRAMES_IN_FLIGHT frames have passed, so running
            // the callback at next_frame + MAX_FRAMES_IN_FLIGHT reclaims
            // exactly these slots (re-checked per slot — running early or late
            // is safe, just less precise).
            let due = self.next_frame() + crate::MAX_FRAMES_IN_FLIGHT as u64;
            self.schedule_at_frame(due, |rm| {
                rm.meshes_info.process_pending_frees();
                rm.blas_emissive_triangles.process_pending_frees();
                Ok(())
            });
        }
    }

    /// Slot of the (deduplicated) sampler matching `desc`, creating it on first
    /// use. The sampler set only grows — samplers are never removed.
    fn sampler_slot(&mut self, desc: &SamplerDesc) -> SrResult<u32> {
        if let Some(sampler) = self.samplers.get(desc) {
            return Ok(sampler.slot());
        }
        let sampler = vulkan_abstraction::Sampler::new_from_desc(Rc::clone(&self.core), desc)?;
        let slot = sampler.slot();
        self.samplers.insert(desc.clone(), sampler);
        Ok(slot)
    }

    // ─── Accessors for the heap-mode push constant ──────────────────────────

    #[allow(dead_code)]
    pub fn tlas(&self) -> &vulkan_abstraction::Tlas {
        &self.tlas
    }

    // Each call lazily allocates a `StorageBuffer` descriptor slot on first use.
    pub fn meshes_info_storage_slot(&self) -> u32 {
        self.meshes_info.raw().storage_slot()
    }

    pub fn emissive_triangles_storage_slot(&self) -> u32 {
        self.blas_emissive_triangles.raw().storage_slot()
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    fn queue_copy(&mut self, arena: ArenaId, region: vk::BufferCopy) {
        // Just enqueue: the renderer drains these via `take_queued_copies` each
        // frame and records them as a transfer prologue in the render graph.
        self.buffer_copies_queued.push((arena, region));
    }
}

// Key → address resolution, used by `InstanceDesc::lower` for late address
// resolution. Kept off the `'static` impl so it stays callable from `lower`'s
// non-`'static` bound.
impl<K: Hash + Eq + Copy> ResourceManager<K> {
    /// Cached device address of the BLAS registered under `key`, or `None` if no
    /// BLAS is registered there. Resolves key → device address (not key →
    /// handle), so a future handle-less cluster BLAS fits the same path.
    pub(crate) fn blas_device_address(&self, key: K) -> Option<vk::DeviceAddress> {
        self.blases.get(&key).map(|blas| blas.device_address())
    }
}
