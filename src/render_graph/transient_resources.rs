use crate::error::{SrError, SrResult};
use crate::render_graph::graph::{
    CachedPipeline, PassComponent, PipelineCache, PipelineHandle, ResourceBarrier, ResourceLifetimeUsage,
};
use crate::render_graph::resource::{GraphResourceDesc, GraphResourceImportInfo, GraphResourceInfo, Handle};
use crate::vulkan_abstraction::buffer::BufferDesc;
use crate::vulkan_abstraction::image::ImageDesc;
use crate::vulkan_abstraction::{
    AccelerationStructure, Buffer, ComputePipeline, Core, GraphicsPipeline, HeapComputePass, Image, RawBuffer,
    RayTracingPipeline, Sampler, ShaderBindingTable,
};
use ash::vk;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;
use vk_sync_fork as vk_sync;

#[derive(Default)]
pub struct TransientResources {
    pub(super) external_images: HashMap<u32, Arc<Image>>,
    pub(super) external_buffers: HashMap<u32, Arc<dyn Buffer>>,
    pub(super) external_samplers: HashMap<u32, Arc<Sampler>>,
    pub(super) external_raytracing_ac: HashMap<u32, Arc<AccelerationStructure>>,
    /// One wrapper per *resource id*, even when several resources share a memory
    /// slot. Each wrapper holds its own `vk::Image` handle + view; the underlying
    /// memory is owned by `slot_allocations` (Image::owns_memory == false).
    pub(super) transient_images: HashMap<u32, Image>,
    /// Same indirection as `transient_images` for buffers.
    pub(super) transient_buffers: HashMap<u32, RawBuffer>,
    /// Samplers are not memory-backed in the aliasable sense; one per resource id.
    pub(super) transient_samplers: HashMap<u32, Sampler>,
    /// One `gpu_allocator` allocation per memory slot. Resources sharing a slot
    /// bind to the same `Allocation` at offset 0. Indexed by slot id.
    pub(super) slot_allocations: Vec<gpu_allocator::vulkan::Allocation>,
    /// Maps each aliased transient resource id (images + buffers) to its slot id
    /// in `slot_allocations`. Samplers and AS are absent (they're not aliased).
    pub(super) resource_slots: HashMap<u32, u32>,
    /// Trace of the barriers that `compile` issued, in topological order, one
    /// entry per pass that needed at least one barrier. Populated by `compile`
    /// after `populate` has wired resources, cleared on `free_internal_state`.
    /// Purely informational — used by the `Debug` impl; the actual barrier
    /// commands are already recorded into the command buffer at this point.
    pub(crate) recorded_barriers: Vec<(usize, Vec<ResourceBarrier>)>,
    /// Cached for `Drop`. Set on first `populate`.
    core: Option<Rc<Core>>,
    /// Persistent pipeline cache. Unlike every other field here it is **not**
    /// cleared by `free_internal_state`: passes are rebuilt each frame but their
    /// pipelines are interned once and reused (see [`PipelineCache`]).
    pub(super) pipeline_cache: PipelineCache,
}

/// Aggregated memory requirements for a single transient alias slot — built up as
/// resources are folded in, then handed to `gpu_allocator` once per slot.
#[derive(Clone, Copy, Debug)]
struct SlotMemReqs {
    size: u64,
    alignment: u64,
    /// AND of every member's `memory_type_bits`. A slot whose intersection is
    /// empty cannot be allocated and signals a bad aliasing decision.
    memory_type_bits: u32,
    location: gpu_allocator::MemoryLocation,
}

/// Pre-built `vk::Image` / `vk::Buffer` handle with its memory requirements; held
/// during populate between the "create handles" pass and the "bind to slot memory"
/// pass.
enum PendingTransient {
    Image {
        handle: vk::Image,
        reqs: vk::MemoryRequirements,
        desc: ImageDesc,
    },
    Buffer {
        handle: vk::Buffer,
        reqs: vk::MemoryRequirements,
        desc: BufferDesc,
    },
}

impl PendingTransient {
    fn reqs(&self) -> vk::MemoryRequirements {
        match self {
            PendingTransient::Image { reqs, .. } => *reqs,
            PendingTransient::Buffer { reqs, .. } => *reqs,
        }
    }
    fn location(&self) -> gpu_allocator::MemoryLocation {
        match self {
            PendingTransient::Image { desc, .. } => desc.location,
            PendingTransient::Buffer { desc, .. } => desc.memory_location,
        }
    }
}

impl TransientResources {
    /// Allocate (or import) backing storage for every virtual resource.
    ///
    /// Slots are assigned by lifetime alone — a buffer's memory can later back an
    /// image and vice versa. Compatibility is enforced at slot creation: when a
    /// candidate slot is reused, its accumulated `memory_type_bits` must still
    /// intersect with the new member's; otherwise a fresh slot is opened. The
    /// resulting slot allocation's `requirements = (max size, max alignment, AND
    /// of memory_type_bits)` is what `gpu_allocator` actually sees.
    ///
    /// Lifetime + Drop:
    ///   - Slot allocations live on `Self`. `Drop` frees them via the cached `core`.
    ///   - Individual transient `Image` / `RawBuffer` wrappers are constructed with
    ///     `owns_memory == false` so their own `Drop` only destroys the `vk::Image`
    ///     / `vk::Buffer` handle (+ view), never `Allocator::free`.
    ///   - This lets `TransientResources` outlive a single frame: the same graph
    ///     can be replayed each frame without re-allocating.
    pub(crate) fn populate(
        &mut self,
        core: Rc<Core>,
        virtual_resources: &[GraphResourceInfo],
        components: &[PassComponent],
        usages: &BTreeMap<u32, ResourceLifetimeUsage>,
    ) -> SrResult<()> {
        // Drop previous frame's bindings + free their allocations. The graph is
        // designed to be re-populated; cross-frame reuse of the same allocations is
        // a future optimization.
        // TODO: detect that desc+lifetimes haven't changed and keep `slot_allocations`
        //       alive across populate calls so we don't churn the allocator each frame.
        self.free_internal_state();
        self.core = Some(Rc::clone(&core));

        // ---------- Phase 1: create unbound handles + collect memory requirements. ----------
        // Sampler / AS are handled separately because they don't participate in slot aliasing.
        let mut pending: HashMap<u32, PendingTransient> = HashMap::new();
        for (res_id, resource_info) in virtual_resources.iter().enumerate() {
            let res_id = res_id as u32;
            let desc = match resource_info {
                GraphResourceInfo::Created(desc) => desc,
                GraphResourceInfo::Imported(_) => continue,
            };
            match desc {
                GraphResourceDesc::Image(image_desc) => {
                    let (handle, reqs) = Image::create_unbound(
                        &core,
                        image_desc.extent,
                        image_desc.format,
                        image_desc.tiling,
                        image_desc.usage,
                    )?;
                    pending.insert(
                        res_id,
                        PendingTransient::Image {
                            handle,
                            reqs,
                            desc: image_desc.clone(),
                        },
                    );
                }
                GraphResourceDesc::Buffer(buffer_desc) => {
                    let (handle, reqs) = RawBuffer::create_unbound(&core, buffer_desc.byte_size, buffer_desc.usage)?;
                    pending.insert(
                        res_id,
                        PendingTransient::Buffer {
                            handle,
                            reqs,
                            desc: buffer_desc.clone(),
                        },
                    );
                }
                GraphResourceDesc::Sampler(sampler_desc) => {
                    // Samplers aren't aliased. Build the wrapper and (eagerly) reserve
                    // its descriptor heap slot.
                    let sampler = Sampler::new_from_desc(Rc::clone(&core), sampler_desc)?;
                    // TODO: this descriptor pre-assignment exists for the legacy single-
                    //       slot-per-resource model; the heap rework will replace it.
                    let _ = sampler.slot();
                    self.transient_samplers.insert(res_id, sampler);
                }
                GraphResourceDesc::RaytracingAS(_) => {
                    //TODO transient AS allocation is intentionally unimplemented and will
                    //     stay that way until the acceleration-structure module is
                    //     refactored — likely in tandem with introducing clustered BLAS
                    //     and partial TLAS updates, which change the lifetime/aliasing
                    //     model enough that designing transient AS now would be wasted.
                }
            }
        }

        // ---------- Phase 2: per-component slot assignment by lifetime, with mem compat. ----------
        // Slots are global (numbered across all components). Reuse condition: the
        // candidate slot's last_pass < this resource's first_pass AND its current
        // (memory_type_bits, location) is compatible with this resource's
        // (memory_type_bits, location). On reuse, the slot's accumulated requirements
        // grow to the max(size), max(alignment), AND(memory_type_bits).
        //TODO this conditions can be relaxed especially on the memory type side
        let mut next_slot: u32 = 0;
        let mut slot_reqs: HashMap<u32, SlotMemReqs> = HashMap::new();
        for component in components {
            // (last_pass_in_slot, slot_id) — kind isn't tracked here, slot_reqs drives compat.
            let mut active: Vec<(usize, u32)> = Vec::new();

            let mut transients: Vec<u32> = component
                .resources
                .iter()
                .copied()
                .filter(|res_id| pending.contains_key(res_id))
                .collect();
            transients.sort_by_key(|res_id| usages[res_id].first_pass);

            for res_id in transients {
                let lifetime = &usages[&res_id];
                let p = &pending[&res_id];
                let this_reqs = p.reqs();
                let this_loc = p.location();

                let candidate = active.iter().position(|(last_pass, slot)| {
                    if *last_pass >= lifetime.first_pass {
                        return false;
                    }
                    let s = &slot_reqs[slot];
                    s.location == this_loc && (s.memory_type_bits & this_reqs.memory_type_bits) != 0
                });

                let slot = if let Some(idx) = candidate {
                    let (_, slot) = active[idx];
                    let s = slot_reqs.get_mut(&slot).expect("slot reqs missing");
                    s.size = s.size.max(this_reqs.size);
                    s.alignment = s.alignment.max(this_reqs.alignment);
                    s.memory_type_bits &= this_reqs.memory_type_bits;
                    active[idx] = (lifetime.last_pass, slot);
                    slot
                } else {
                    let slot = next_slot;
                    next_slot += 1;
                    slot_reqs.insert(
                        slot,
                        SlotMemReqs {
                            size: this_reqs.size,
                            alignment: this_reqs.alignment,
                            memory_type_bits: this_reqs.memory_type_bits,
                            location: this_loc,
                        },
                    );
                    active.push((lifetime.last_pass, slot));
                    slot
                };
                self.resource_slots.insert(res_id, slot);
            }
        }

        // ---------- Phase 3: allocate one chunk of memory per slot. ----------
        // Iterate by slot id so `slot_allocations[i]` corresponds to slot `i`.
        let mut slot_allocations: Vec<gpu_allocator::vulkan::Allocation> = Vec::with_capacity(next_slot as usize);
        for slot in 0..next_slot {
            let s = slot_reqs[&slot];
            if s.memory_type_bits == 0 {
                // Should not be reachable: the compat check above guarantees a non-zero
                // intersection on every reuse.
                return Err(SrError::new_custom(format!(
                    "transient slot {slot}: empty memory_type_bits after aliasing"
                )));
            }
            let mem_reqs = vk::MemoryRequirements {
                size: s.size,
                alignment: s.alignment,
                memory_type_bits: s.memory_type_bits,
            };
            // linear: false — slots may host optimal-tiled images, and since members
            // never co-occupy the slot, bufferImageGranularity within the allocation
            // isn't an issue. Buffers placed in non-linear regions still work.
            let allocation = core.allocator_mut().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "render_graph_transient_slot",
                requirements: mem_reqs,
                location: s.location,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })?;
            slot_allocations.push(allocation);
        }

        // ---------- Phase 4: bind each handle into its slot's memory, wrap, pre-reserve descriptors. ----------
        let device = core.device().inner();
        let name_objects = core.debug_labels_enabled();
        for (res_id, p) in pending {
            let slot = self.resource_slots[&res_id];
            let alloc = &slot_allocations[slot as usize];
            match p {
                PendingTransient::Image { handle, reqs, desc } => {
                    unsafe { device.bind_image_memory(handle, alloc.memory(), alloc.offset()) }?;
                    if name_objects && let Ok(cname) = std::ffi::CString::new(desc.name) {
                        core.set_debug_object_name(handle, &cname);
                    }
                    let image = Image::from_aliased(Rc::clone(&core), handle, desc.extent, desc.format, reqs.size)?;
                    // TODO: this descriptor pre-assignment exists for the legacy single-
                    //       slot-per-resource model; the heap rework will replace it.
                    if desc.usage.contains(vk::ImageUsageFlags::STORAGE) {
                        let _ = image.storage_slot();
                    }
                    if desc.usage.contains(vk::ImageUsageFlags::SAMPLED) {
                        let _ = image.sampled_slot();
                    }
                    self.transient_images.insert(res_id, image);
                }
                PendingTransient::Buffer { handle, reqs: _, desc } => {
                    unsafe { device.bind_buffer_memory(handle, alloc.memory(), alloc.offset()) }?;
                    if name_objects && let Ok(cname) = std::ffi::CString::new(desc.name) {
                        core.set_debug_object_name(handle, &cname);
                    }
                    let buffer = RawBuffer::from_aliased(Rc::clone(&core), handle, desc.byte_size, desc.usage)?;
                    // TODO: same legacy-descriptor caveat as the image path above.
                    if desc.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
                        let _ = buffer.storage_slot();
                    }
                    if desc.usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
                        let _ = buffer.uniform_slot();
                    }
                    self.transient_buffers.insert(res_id, buffer);
                }
            }
        }
        self.slot_allocations = slot_allocations;

        // ---------- Phase 5: wire imported handles into external_* maps. ----------
        for (res_id, resource_info) in virtual_resources.iter().enumerate() {
            let res_id = res_id as u32;
            let import = match resource_info {
                GraphResourceInfo::Created(_) => continue,
                GraphResourceInfo::Imported(import) => import,
            };
            match import {
                GraphResourceImportInfo::Image { resource, .. } => {
                    self.external_images.insert(res_id, resource.clone());
                }
                GraphResourceImportInfo::Buffer { resource, .. } => {
                    self.external_buffers.insert(res_id, resource.clone());
                }
                GraphResourceImportInfo::Sampler { resource } => {
                    self.external_samplers.insert(res_id, resource.clone());
                }
                GraphResourceImportInfo::RayTracingAcceleration { resource, .. } => {
                    self.external_raytracing_ac.insert(res_id, resource.clone());
                }
            }
        }

        Ok(())
    }

    /// Drop all transient wrappers (which destroy their vk handles but won't free
    /// memory) and free every slot allocation. Used by both `populate` (rebuild)
    /// and `Drop`.
    pub(crate) fn free_internal_state(&mut self) {
        self.external_images.clear();
        self.external_buffers.clear();
        self.external_samplers.clear();
        self.external_raytracing_ac.clear();
        // Drop the wrappers first: their Drop destroys vk handles and skips
        // Allocator::free (owns_memory == false), so the underlying allocations are
        // still valid afterwards.
        self.transient_images.clear();
        self.transient_buffers.clear();
        self.transient_samplers.clear();
        self.resource_slots.clear();
        self.recorded_barriers.clear();

        if let Some(core) = self.core.as_ref() {
            let mut allocator = core.allocator_mut();
            for allocation in self.slot_allocations.drain(..) {
                if let Err(e) = allocator.free(allocation) {
                    log::error!("Allocator::free returned {e} in TransientResources::free_internal_state");
                }
            }
        } else {
            // No core cached: there cannot be allocations to free, but defensively clear.
            self.slot_allocations.clear();
        }
    }
}

impl Drop for TransientResources {
    fn drop(&mut self) {
        self.free_internal_state();
    }
}

impl TransientResources {
    /// Resolve a graph image handle to the concrete `Image` bound for this frame,
    /// whether it's a transient (created) resource or an imported one. Render
    /// closures call this to read an image's heap descriptor slots
    /// (`storage_slot()` / `sampled_slot()`) when the image is graph-managed
    /// rather than captured directly.
    pub fn image(&self, handle: &Handle<Image>) -> SrResult<&Image> {
        let id = handle.id;
        if let Some(img) = self.transient_images.get(&id) {
            return Ok(img);
        }
        if let Some(img) = self.external_images.get(&id) {
            return Ok(img.as_ref());
        }
        Err(SrError::new_custom(format!(
            "render graph: no image bound for resource id {id} (not created or imported as an image)"
        )))
    }

    /// The raw `vk::Buffer` bound to resource id `id`, transient or imported.
    /// Used by the graph to resolve a [`TransferPass`](crate::render_graph::pass_builder::TransferPass)'s
    /// declarative copy list at record time.
    pub(crate) fn buffer_by_id(&self, id: u32) -> Option<vk::Buffer> {
        self.transient_buffers
            .get(&id)
            .map(|buf| buf.inner())
            .or_else(|| self.external_buffers.get(&id).map(|buf| buf.inner()))
    }

    /// Resolve a [`PipelineHandle`] to its interned compute pipeline. Render
    /// closures installed by `ComputeRenderPassBuilder::generate_render` call this
    /// to bind the pipeline at record time.
    pub fn compute_pipeline(&self, handle: PipelineHandle) -> SrResult<&ComputePipeline<HeapComputePass>> {
        match self.pipeline_cache.get(handle) {
            Some(CachedPipeline::Compute(p)) => Ok(p),
            _ => Err(SrError::new_custom(format!(
                "render graph: pipeline handle {handle:?} is not a cached compute pipeline"
            ))),
        }
    }

    /// Resolve a [`PipelineHandle`] to its interned ray-tracing pipeline + shader
    /// binding table.
    pub fn raytracing_pipeline(&self, handle: PipelineHandle) -> SrResult<(&RayTracingPipeline, &ShaderBindingTable)> {
        match self.pipeline_cache.get(handle) {
            Some(CachedPipeline::RayTracing(p, sbt)) => Ok((p, sbt)),
            _ => Err(SrError::new_custom(format!(
                "render graph: pipeline handle {handle:?} is not a cached ray-tracing pipeline"
            ))),
        }
    }

    /// Resolve a [`PipelineHandle`] to its interned graphics pipeline.
    pub fn graphics_pipeline(&self, handle: PipelineHandle) -> SrResult<&GraphicsPipeline> {
        match self.pipeline_cache.get(handle) {
            Some(CachedPipeline::Graphics(p)) => Ok(p),
            _ => Err(SrError::new_custom(format!(
                "render graph: pipeline handle {handle:?} is not a cached graphics pipeline"
            ))),
        }
    }
}

impl TransientResources {
    /// Issue a `vkCmdPipelineBarrier` covering every `ResourceBarrier` in
    /// `barriers`. Each barrier is dispatched as an image barrier, buffer
    /// barrier, or global barrier depending on what kind of resource its
    /// `resource_id` resolves to here. Resources unknown to this struct (e.g.
    /// non-aliased samplers or AS-only ids that slipped through) collapse to a
    /// `GlobalBarrier` carrying just the access transition.
    ///
    /// Layouts: we use `vk_sync::ImageLayout::Optimal` for both sides, which
    /// tells vk_sync to pick the right `VK_IMAGE_LAYOUT_*` from the access
    /// types. Subresource range covers all mips / array layers — per-mip /
    /// per-layer barriers are a future optimization once passes can express
    /// subresource access.
    ///
    /// Queue family transfer is `IGNORED` on both sides — we're single-queue.
    ///
    /// TODO this is currently doing a 1 resource barriers to 1 actual barrier, this can be reduced to 1 barrier per image layout transition and a global barrier for the other stuff, this doesn't add any involuntary sync since I have already built the dependencies graph
    pub(crate) fn emit_barriers(&self, device: &ash::Device, cmd_buffer: vk::CommandBuffer, barriers: &[ResourceBarrier]) {
        if barriers.is_empty() {
            return;
        }
        let mut image_barriers: Vec<vk_sync::ImageBarrier> = Vec::new();
        let mut buffer_barriers: Vec<vk_sync::BufferBarrier> = Vec::new();
        let mut global_prev: Vec<vk_sync::AccessType> = Vec::new();
        let mut global_next: Vec<vk_sync::AccessType> = Vec::new();

        for b in barriers {
            let prev_slice = std::slice::from_ref(&b.prev_access);
            let next_slice = std::slice::from_ref(&b.next_access);

            // Image? (transient first, then imported — same resource id can never
            // appear in both maps so the order doesn't matter for correctness).
            let image_info = self
                .transient_images
                .get(&b.resource_id)
                .map(|img| (img.inner(), img.format()))
                .or_else(|| {
                    self.external_images
                        .get(&b.resource_id)
                        .map(|img| (img.inner(), img.format()))
                });

            if let Some((handle, format)) = image_info {
                image_barriers.push(vk_sync::ImageBarrier {
                    previous_accesses: prev_slice,
                    next_accesses: next_slice,
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: false,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: handle,
                    range: vk::ImageSubresourceRange {
                        aspect_mask: aspect_for(format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    },
                });
                continue;
            }

            let buffer_info = self
                .transient_buffers
                .get(&b.resource_id)
                .map(|buf| (buf.inner(), buf.byte_size()))
                .or_else(|| {
                    self.external_buffers
                        .get(&b.resource_id)
                        .map(|buf| (buf.inner(), buf.byte_size()))
                });

            if let Some((handle, size)) = buffer_info {
                buffer_barriers.push(vk_sync::BufferBarrier {
                    previous_accesses: prev_slice,
                    next_accesses: next_slice,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: handle,
                    offset: 0,
                    size: size as usize,
                });
                continue;
            }

            // Fallback: AS / sampler / anything else — collapse into a global
            // barrier so the access ordering is still expressed.
            global_prev.push(b.prev_access);
            global_next.push(b.next_access);
        }

        let global = if !global_prev.is_empty() {
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &global_prev,
                next_accesses: &global_next,
            })
        } else {
            None
        };

        vk_sync::cmd::pipeline_barrier(device, cmd_buffer, global, &buffer_barriers, &image_barriers);
    }
}

impl std::fmt::Debug for TransientResources {
    /// Renders the aliasing decisions in the same "report" layout used by the
    /// transient_aliasing_debug test: header, per-slot allocation table, per-resource
    /// slot assignment, and grouped aliasing sets. Lifetimes aren't
    /// stored on `self`, so they're omitted here — only what `populate` left
    /// behind is printed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let aliasable = self.resource_slots.len();
        let imported = self.external_images.len()
            + self.external_buffers.len()
            + self.external_samplers.len()
            + self.external_raytracing_ac.len();
        let total = self.transient_images.len() + self.transient_buffers.len() + self.transient_samplers.len() + imported;

        writeln!(f)?;
        writeln!(f, "=== TransientResources aliasing report ===")?;
        writeln!(f, "resources tracked           : {total}")?;
        writeln!(f, "  transient images          : {}", self.transient_images.len())?;
        writeln!(f, "  transient buffers         : {}", self.transient_buffers.len())?;
        writeln!(f, "  transient samplers        : {}", self.transient_samplers.len())?;
        writeln!(f, "  imported                  : {imported}")?;
        writeln!(f, "aliasable resources (img+buf): {aliasable}")?;
        writeln!(f, "slot allocations            : {}", self.slot_allocations.len())?;
        writeln!(f)?;

        writeln!(f, "Per-slot allocation:")?;
        for (i, alloc) in self.slot_allocations.iter().enumerate() {
            let mem = unsafe { alloc.memory() };
            writeln!(
                f,
                "  slot {i}: size={:>8} offset={:>8} memory={:?}",
                alloc.size(),
                alloc.offset(),
                mem,
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Per-resource slot assignment:")?;
        let mut assignments: Vec<(u32, u32)> = self.resource_slots.iter().map(|(r, s)| (*r, *s)).collect();
        assignments.sort();
        for (res_id, slot_id) in &assignments {
            let kind = if let Some(img) = self.transient_images.get(res_id) {
                let e = img.extent();
                format!("Image {}x{}x{}", e.width, e.height, e.depth)
            } else if let Some(buf) = self.transient_buffers.get(res_id) {
                format!("Buffer {} bytes", buf.byte_size())
            } else {
                "???".to_string()
            };
            writeln!(f, "  res {res_id} {kind:<32} -> slot {slot_id}")?;
        }
        writeln!(f)?;

        let mut by_slot: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (r, s) in &self.resource_slots {
            by_slot.entry(*s).or_default().push(*r);
        }
        writeln!(f, "Aliasing groups (slot -> resources sharing memory):")?;
        for (slot, members) in &by_slot {
            let mut m = members.clone();
            m.sort();
            let aliased = if m.len() > 1 { " (ALIASED)" } else { "" };
            writeln!(f, "  slot {slot} <- resources {m:?}{aliased}")?;
        }

        // Non-aliased extras
        if !self.transient_samplers.is_empty() {
            writeln!(f)?;
            let mut samplers: Vec<u32> = self.transient_samplers.keys().copied().collect();
            samplers.sort();
            writeln!(f, "Transient samplers (not slot-aliased): {samplers:?}")?;
        }
        if imported > 0 {
            writeln!(f)?;
            let mut imports: Vec<(u32, &'static str)> = Vec::new();
            imports.extend(self.external_images.keys().map(|k| (*k, "Image")));
            imports.extend(self.external_buffers.keys().map(|k| (*k, "Buffer")));
            imports.extend(self.external_samplers.keys().map(|k| (*k, "Sampler")));
            imports.extend(self.external_raytracing_ac.keys().map(|k| (*k, "AccelStruct")));
            imports.sort();
            writeln!(f, "Imported resources:")?;
            for (id, kind) in imports {
                writeln!(f, "  res {id} {kind}")?;
            }
        }

        // ---- Barriers recorded during compile ----
        writeln!(f)?;
        let total_barriers: usize = self.recorded_barriers.iter().map(|(_, b)| b.len()).sum();
        if self.recorded_barriers.is_empty() {
            writeln!(f, "Barriers recorded: none (graph not compiled or single-pass)")?;
        } else {
            writeln!(
                f,
                "Barriers recorded ({} total across {} pass(es), in topo order):",
                total_barriers,
                self.recorded_barriers.len(),
            )?;
            for (pass_id, barriers) in &self.recorded_barriers {
                writeln!(f, "  before pass {pass_id}:")?;
                for b in barriers {
                    let kind = if self.transient_images.contains_key(&b.resource_id)
                        || self.external_images.contains_key(&b.resource_id)
                    {
                        "Image"
                    } else if self.transient_buffers.contains_key(&b.resource_id)
                        || self.external_buffers.contains_key(&b.resource_id)
                    {
                        "Buffer"
                    } else if self.transient_samplers.contains_key(&b.resource_id)
                        || self.external_samplers.contains_key(&b.resource_id)
                    {
                        "Sampler"
                    } else if self.external_raytracing_ac.contains_key(&b.resource_id) {
                        "AccelStruct"
                    } else {
                        "Global"
                    };
                    writeln!(
                        f,
                        "    res {:>3} ({kind:<11}) {:?} -> {:?}",
                        b.resource_id, b.prev_access, b.next_access
                    )?;
                }
            }
        }
        writeln!(f, "==========================================")
    }
}

/// Pick the right `vk::ImageAspectFlags` for a given format. Used when building
/// image subresource ranges for layout transitions in `emit_barriers`.
pub(crate) fn aspect_for(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => vk::ImageAspectFlags::DEPTH,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}
