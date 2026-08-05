use crate::error::{SrError, SrResult};
use crate::render_graph::alias::{self, AliasResource, AliasStrategy, Placement};
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
    /// One `gpu_allocator` allocation per memory bucket. Resources sharing a bucket
    /// bind into the same `Allocation`, each at its own offset. Indexed by bucket id.
    pub(super) slot_allocations: Vec<gpu_allocator::vulkan::Allocation>,
    /// Where each aliased transient resource (images + buffers) landed: bucket id
    /// plus byte range. Samplers and AS are absent (they're not aliased).
    pub(super) placements: HashMap<u32, Placement>,
    /// Trace of the barriers that `compile` issued, in topological order, one
    /// entry per pass that needed at least one barrier. Populated by `compile`
    /// after `populate` has wired resources, cleared on `free_internal_state`.
    /// Purely informational — used by the `Debug` impl; the actual barrier
    /// commands are already recorded into the command buffer at this point.
    pub(crate) recorded_barriers: Vec<(usize, Vec<ResourceBarrier>)>,
    /// Cached for `Drop`. Set on first `populate`.
    core: Option<Arc<Core>>,
    /// Persistent pipeline cache. Unlike every other field here it is **not**
    /// cleared by `free_internal_state`: passes are rebuilt each frame but their
    /// pipelines are interned once and reused (see [`PipelineCache`]).
    pub(super) pipeline_cache: PipelineCache,
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
    /// Memory is assigned by lifetime alone — a buffer's memory can later back an
    /// image and vice versa — subject to `memory_type_bits` and heap compatibility.
    /// The policy is [`alias::plan`], selected by `SUNRAY_ALIAS_STRATEGY`; see that
    /// module for what the two strategies do and how they differ.
    ///
    /// Lifetime + Drop:
    ///   - Bucket allocations live on `Self`. `Drop` frees them via the cached `core`.
    ///   - Individual transient `Image` / `RawBuffer` wrappers are constructed with
    ///     `owns_memory == false` so their own `Drop` only destroys the `vk::Image`
    ///     / `vk::Buffer` handle (+ view), never `Allocator::free`.
    ///   - This lets `TransientResources` outlive a single frame: the same graph
    ///     can be replayed each frame without re-allocating.
    pub(crate) fn populate(
        &mut self,
        core: Arc<Core>,
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
        self.core = Some(Arc::clone(&core));

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
                    let sampler = Sampler::new_from_desc(Arc::clone(&core), sampler_desc)?;
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

        // ---------- Phase 2: assign every transient a bucket + offset. ----------
        // The policy itself lives in `alias` — Vulkan-free so it can be tested and
        // benchmarked without a device. Everything here is marshalling.
        //TODO the memory-type condition inside `alias` can be relaxed
        let granularity = core.device().properties().limits.buffer_image_granularity;
        let alias_resources: Vec<AliasResource> = pending
            .iter()
            .map(|(res_id, p)| {
                let reqs = p.reqs();
                let lifetime = &usages[res_id];
                AliasResource {
                    id: *res_id,
                    size: reqs.size,
                    alignment: reqs.alignment,
                    memory_type_bits: reqs.memory_type_bits,
                    location: p.location(),
                    first_pass: lifetime.first_pass,
                    last_pass: lifetime.last_pass,
                }
            })
            .collect();
        let alias_components: Vec<Vec<u32>> = components
            .iter()
            .map(|c| c.resources.iter().copied().filter(|id| pending.contains_key(id)).collect())
            .collect();

        let strategy = AliasStrategy::from_env();
        let (placements, buckets) = alias::plan(strategy, &alias_resources, &alias_components, granularity);
        self.placements = placements;

        // ---------- Phase 3: allocate one chunk of memory per bucket. ----------
        // Iterate by bucket id so `slot_allocations[i]` corresponds to bucket `i`.
        let mut slot_allocations: Vec<gpu_allocator::vulkan::Allocation> = Vec::with_capacity(buckets.len());
        for (bucket, reqs) in buckets.iter().enumerate() {
            if reqs.memory_type_bits == 0 {
                // Should not be reachable: `alias::plan` only folds a resource into a
                // bucket whose intersection with it is non-empty.
                return Err(SrError::new_custom(format!(
                    "transient bucket {bucket}: empty memory_type_bits after aliasing"
                )));
            }
            let mem_reqs = vk::MemoryRequirements {
                size: reqs.size,
                alignment: reqs.alignment,
                memory_type_bits: reqs.memory_type_bits,
            };
            // linear: false — buckets may host optimal-tiled images. Under `Bucket`
            // two members *can* co-occupy, so bufferImageGranularity matters within
            // the allocation; `alias::plan` was handed the limit and padded every
            // placement to it. Buffers placed in non-linear regions still work.
            let allocation = core.allocator_mut().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "render_graph_transient_slot",
                requirements: mem_reqs,
                location: reqs.location,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })?;
            slot_allocations.push(allocation);
        }

        // ---------- Phase 4: bind each handle into its slot's memory, wrap, pre-reserve descriptors. ----------
        let device = core.device().inner();
        let name_objects = core.debug_labels_enabled();
        for (res_id, p) in pending {
            let placement = self.placements[&res_id];
            let alloc = &slot_allocations[placement.bucket as usize];
            // The allocation's own offset into its `vk::DeviceMemory`, plus where
            // inside the bucket the placer put this resource.
            let base = alloc.offset() + placement.offset;
            match p {
                PendingTransient::Image { handle, reqs, desc } => {
                    unsafe { device.bind_image_memory(handle, alloc.memory(), base) }?;
                    if name_objects && let Ok(cname) = std::ffi::CString::new(desc.name) {
                        core.set_debug_object_name(handle, &cname);
                    }
                    let image = Image::from_aliased(Arc::clone(&core), handle, desc.extent, desc.format, reqs.size)?;
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
                    unsafe { device.bind_buffer_memory(handle, alloc.memory(), base) }?;
                    if name_objects && let Ok(cname) = std::ffi::CString::new(desc.name) {
                        core.set_debug_object_name(handle, &cname);
                    }
                    let buffer = RawBuffer::from_aliased(Arc::clone(&core), handle, desc.byte_size, desc.usage)?;
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
        self.placements.clear();
        self.recorded_barriers.clear();

        self.transient_images.clear();
        self.transient_buffers.clear();
        self.transient_samplers.clear();

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
        let mut global_prev: Vec<vk_sync::AccessType> = Vec::new();
        let mut global_next: Vec<vk_sync::AccessType> = Vec::new();

        for b in barriers {
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
                // Only a genuine layout change (or a discard, which forces
                // oldLayout to UNDEFINED) needs its own image barrier. Everything
                // else folds into the global one below.
                let changes_layout = epoch_layout(&b.prev) != epoch_layout(&b.next);
                if changes_layout || b.discard {
                    image_barriers.push(vk_sync::ImageBarrier {
                        previous_accesses: &b.prev,
                        next_accesses: &b.next,
                        previous_layout: vk_sync::ImageLayout::Optimal,
                        next_layout: vk_sync::ImageLayout::Optimal,
                        discard_contents: b.discard,
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
            }

            // Buffers, acceleration structures, samplers, and images whose layout
            // does not change: all of these need only availability/visibility, which
            // is exactly what one global memory barrier expresses. Folding them
            // costs nothing — they were already going into this same
            // `vkCmdPipelineBarrier2`, which is already a single sync point with one
            // pair of stage masks — and `vk_sync` ORs the accesses for us.
            //
            // ponytail: per-buffer `BufferBarrier`s are gone. They only beat a
            // global barrier for queue-family ownership transfer; re-add them when
            // multi-queue needs QFOT, which is the one case where the
            // buffer/offset/size fields carry information.
            global_prev.extend_from_slice(&b.prev);
            global_next.extend_from_slice(&b.next);
        }

        let global = if !global_prev.is_empty() {
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &global_prev,
                next_accesses: &global_next,
            })
        } else {
            None
        };

        vk_sync::cmd::pipeline_barrier(device, cmd_buffer, global, &[], &image_barriers);
    }
}

impl std::fmt::Debug for TransientResources {
    /// Renders the aliasing decisions in the same "report" layout used by the
    /// transient_aliasing_debug test: header, per-bucket allocation table,
    /// per-resource placement, and grouped aliasing sets. Lifetimes aren't
    /// stored on `self`, so they're omitted here — only what `populate` left
    /// behind is printed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let aliasable = self.placements.len();
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
        writeln!(f, "bucket allocations          : {}", self.slot_allocations.len())?;
        let total: u64 = self.slot_allocations.iter().map(|a| a.size()).sum();
        let requested: u64 = self.placements.values().map(|p| p.size).sum();
        writeln!(
            f,
            "allocated / requested bytes : {total} / {requested} ({:.0}%)",
            100.0 * total as f64 / requested.max(1) as f64
        )?;
        writeln!(f)?;

        writeln!(f, "Per-bucket allocation:")?;
        for (i, alloc) in self.slot_allocations.iter().enumerate() {
            let mem = unsafe { alloc.memory() };
            writeln!(
                f,
                "  bucket {i}: size={:>8} offset={:>8} memory={:?}",
                alloc.size(),
                alloc.offset(),
                mem,
            )?;
        }
        writeln!(f)?;

        // Grouped by bucket, ordered by offset — under `Bucket` the offsets are the
        // interesting part, since that is where the packing shows up.
        let mut by_bucket: BTreeMap<u32, Vec<(u32, Placement)>> = BTreeMap::new();
        for (r, p) in &self.placements {
            by_bucket.entry(p.bucket).or_default().push((*r, *p));
        }
        writeln!(f, "Aliasing groups (bucket -> resources sharing memory):")?;
        for (bucket, members) in &mut by_bucket {
            members.sort_by_key(|(r, p)| (p.offset, *r));
            let aliased = if members.len() > 1 { " (ALIASED)" } else { "" };
            let capacity = self.slot_allocations.get(*bucket as usize).map_or(0, |a| a.size());
            writeln!(f, "  bucket {bucket} ({capacity} bytes){aliased}")?;
            for (res_id, p) in members.iter() {
                let kind = if let Some(img) = self.transient_images.get(res_id) {
                    let e = img.extent();
                    format!("Image {}x{}x{}", e.width, e.height, e.depth)
                } else if let Some(buf) = self.transient_buffers.get(res_id) {
                    format!("Buffer {} bytes", buf.byte_size())
                } else {
                    "???".to_string()
                };
                writeln!(
                    f,
                    "    res {res_id:>3} {kind:<28} @ {:>9}..{:<9}",
                    p.offset,
                    p.offset + p.size
                )?;
            }
        }

        // Non-aliased extras
        if !self.transient_samplers.is_empty() {
            writeln!(f)?;
            let mut samplers: Vec<u32> = self.transient_samplers.keys().copied().collect();
            samplers.sort();
            writeln!(f, "Transient samplers (not aliased): {samplers:?}")?;
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
                    let folded = if kind == "Image" && epoch_layout(&b.prev) == epoch_layout(&b.next) && !b.discard {
                        "  [folded into global]"
                    } else {
                        ""
                    };
                    writeln!(
                        f,
                        "    res {:>3} ({kind:<11}) {:?} -> {:?}{}{}",
                        b.resource_id,
                        b.prev,
                        b.next,
                        if b.discard { "  [discard]" } else { "" },
                        folded
                    )?;
                }
            }
        }
        writeln!(f, "==========================================")
    }
}

/// The `VkImageLayout` an access type implies, as `vk_sync` would pick it for
/// `ImageLayout::Optimal`.
///
/// The graph needs this for two decisions it cannot make otherwise: whether a run
/// of reads can share one barrier (they must agree on layout — `vk_sync`'s
/// `get_image_memory_barrier` debug-asserts if they don't), and whether a barrier
/// changes layout at all, which is the only thing a `VkImageMemoryBarrier2` buys
/// over the global one.
pub(crate) fn image_layout_of(access: vk_sync::AccessType) -> vk::ImageLayout {
    vk_sync::get_access_info(access).image_layout
}

/// Layout implied by one side of a barrier. Every access in an epoch agrees on
/// layout by construction (`RenderGraph::plan_barriers` splits a read run when it
/// wouldn't), so the first one speaks for all.
fn epoch_layout(accesses: &[vk_sync::AccessType]) -> vk::ImageLayout {
    accesses.first().map_or(vk::ImageLayout::UNDEFINED, |a| image_layout_of(*a))
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
