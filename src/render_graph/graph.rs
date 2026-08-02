use crate::error::{ErrorSource, SrError, SrResult};
use crate::render_graph::error::GraphError;
use crate::render_graph::pass_builder::{
    ComputeRenderPass, PassCommonDataBuilder, RasterRenderPass, RaytracingRenderPass, TransferPass,
};
pub(crate) use crate::render_graph::resource::{
    GraphResourceDesc, GraphResourceImportInfo, GraphResourceInfo, Handle, Resource, ResourceDesc, RgImportable,
};
use crate::render_graph::transient_resources::TransientResources;
use crate::vulkan_abstraction::{
    AccelerationStructure, AsBuildJob, Buffer, CmdBuffer, ComputePipeline, Core, GpuOnlyBuffer, GraphicsPipeline,
    GraphicsPipelineShaders, HeapComputePass, Image, Pipeline, RawBuffer, RayTracingPipeline, RayTracingPipelineShaders,
    ShaderBindingTable, TimelineSemaphore,
};
use crate::MAX_FRAMES_IN_FLIGHT;
use ash::vk;
use petgraph::visit::EdgeRef;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use vk_sync_fork as vk_sync;
use vk_sync_fork::AccessType;

#[derive(Copy, Clone, Debug)]
pub enum PassResourceAccessSyncType {
    AlwaysSync,
    SkipSyncIfSameAccessType,
    NeverSync,
}

#[derive(Copy, Clone, Debug)]
pub struct PassResourceAccessType {
    pub(crate) access_type: vk_sync::AccessType,
    pub(crate) sync_type: PassResourceAccessSyncType,
}

pub(super) enum AnyRenderPass {
    Rt(RaytracingRenderPass),
    Raster(RasterRenderPass),
    Compute(ComputeRenderPass),
    Transfer(TransferPass),
}

/// Lightweight reference to a pipeline interned in the graph's [`PipelineCache`].
/// Render closures resolve it to the concrete pipeline at record time via
/// `TransientResources::{compute,raytracing,graphics}_pipeline`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineHandle(u32);

/// A heap-mode pipeline owned by the cache. RT additionally owns its shader
/// binding table (built alongside the pipeline).
pub(super) enum CachedPipeline {
    Compute(Rc<ComputePipeline<HeapComputePass>>),
    RayTracing(Rc<RayTracingPipeline>, Rc<ShaderBindingTable>),
    Graphics(Rc<GraphicsPipeline>),
}

/// Content-addressed cache of heap-mode pipelines, owned by the graph and kept
/// alive across the per-frame `reset()` / rebuild cycle. Passes describe their
/// shaders every frame, but the underlying `vk::Pipeline` (an expensive object)
/// is built exactly once per distinct shader set and reused — no per-frame
/// pipeline churn, and identical shaders shared by several passes are never
/// duplicated on the GPU.
///
/// Entries are addressed by [`PipelineHandle`] (an index into `entries`); the
/// `by_key` map dedups by a hash of the shader bytes. The cache is never cleared
/// by `free_internal_state` (that only frees per-frame transient resources), so
/// it lives for the whole graph; `core` is held so cached pipelines outlive the
/// device-owning `Core` no matter the surrounding drop order.
///
/// TODO: there is no eviction yet — an interned pipeline is kept until the graph
/// is dropped. Fine while the renderer uses a fixed, small shader set (every
/// entry is "still needed" every frame); add refcount/GC eviction once shaders
/// can come and go at runtime.
#[derive(Default)]
pub(super) struct PipelineCache {
    entries: Vec<CachedPipeline>,
    by_key: HashMap<u64, PipelineHandle>,
    core: Option<Rc<Core>>,
}

impl PipelineCache {
    pub(super) fn get(&self, handle: PipelineHandle) -> Option<&CachedPipeline> {
        self.entries.get(handle.0 as usize)
    }

    /// Return the handle for `key` if already interned, otherwise build the
    /// pipeline via `build`, store it, and return its fresh handle.
    pub(super) fn intern(
        &mut self,
        key: u64,
        core: &Rc<Core>,
        build: impl FnOnce() -> SrResult<CachedPipeline>,
    ) -> SrResult<PipelineHandle> {
        if let Some(handle) = self.by_key.get(&key) {
            return Ok(*handle);
        }
        let entry = build()?;
        if self.core.is_none() {
            self.core = Some(Rc::clone(core));
        }
        let handle = PipelineHandle(self.entries.len() as u32);
        self.entries.push(entry);
        self.by_key.insert(key, handle);
        Ok(handle)
    }
}

/// Hash a set of byte slices (plus a `kind` discriminant so a compute shader and
/// a same-bytes graphics shader never collide) into the cache key.
fn pipeline_cache_key(kind: u8, parts: &[&[u8]]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    kind.hash(&mut hasher);
    for part in parts {
        part.hash(&mut hasher);
    }
    hasher.finish()
}

/// Where a graph resource ends up once the graph's submission completes: the
/// last pass that touched it and the access it is left in.
///
/// TODO(temp impl): this is the seed of the cross-submission sync contract.
/// The goal is that whatever runs *after* the graph (the external blit, the
/// present transition, the next frame's graph) reads the end state of the
/// resources it shares with the graph and emits a precise pipeline barrier
/// from `end_access` instead of `vkDeviceWaitIdle`, and that `compile` itself
/// consumes the previous submission's end states as the initial access of
/// imported resources — so consecutive frames chain render-pass-to-render-pass
/// with no idle. Right now the states are only collected and exposed; nothing
/// consumes them yet.
#[derive(Clone, Debug)]
pub struct ResourceEndState {
    /// Pass id of the last pass that touched the resource, `None` if the
    /// resource was registered but never used by any pass.
    pub last_use_pass: Option<usize>,
    /// Access type the resource is left in when the submission completes
    /// (`Nothing` when never used).
    pub end_access: vk_sync::AccessType,
    /// Graph-created (transient) resource: its backing memory is recycled on
    /// `reset`, so its end state only matters for in-graph aliasing, never to
    /// the caller.
    pub internal: bool,
}

/// A single transition required before a destination pass can run, derived from
/// a read/write hazard on `resource_id` against an earlier producer or reader.
#[derive(Clone, Debug)]
pub(crate) struct ResourceBarrier {
    pub(crate) resource_id: u32,
    pub(crate) prev_access: vk_sync::AccessType,
    pub(crate) next_access: vk_sync::AccessType,
}

/// Edge weight on the pass dependency graph: all barriers that must be issued
/// before the destination pass runs because of the source pass.
#[derive(Clone, Debug, Default)]
pub(crate) struct PassDependency {
    pub(crate) barriers: Vec<ResourceBarrier>,
}

/// Per-resource lifetime + ordered list of (pass, access) touches. Lifetime is
/// inclusive: the resource must be live from `first_pass` through `last_pass`.
#[derive(Debug)]
pub(crate) struct ResourceLifetimeUsage {
    pub(crate) first_pass: usize,
    pub(crate) last_pass: usize,
    pub(crate) usages: Vec<(usize, PassResourceAccessType)>,
}

/// Hazard-tracking state for a single resource while scanning passes in order.
#[derive(Debug, Default)]
struct ResourceHazardState {
    last_writer: Option<(usize, vk_sync::AccessType)>,
    readers_since_write: Vec<(usize, vk_sync::AccessType)>,
}

/// A weakly-connected component of the dependency graph: a set of passes that
/// transitively share resources, plus the resource ids those passes touch.
/// Transient memory aliasing is computed independently per component.
#[derive(Debug)]
pub(crate) struct PassComponent {
    pub(crate) passes: Vec<usize>,
    pub(crate) resources: Vec<u32>,
}

fn record_usage(usages: &mut BTreeMap<u32, ResourceLifetimeUsage>, res_id: u32, pass_id: usize, access: PassResourceAccessType) {
    usages
        .entry(res_id)
        .and_modify(|u| {
            u.last_pass = pass_id;
            u.usages.push((pass_id, access));
        })
        .or_insert_with(|| ResourceLifetimeUsage {
            first_pass: pass_id,
            last_pass: pass_id,
            usages: vec![(pass_id, access)],
        });
}

fn add_dep_edge(
    graph: &mut petgraph::graph::DiGraph<usize, PassDependency>,
    nodes: &[petgraph::graph::NodeIndex],
    src: usize,
    dst: usize,
    barrier: ResourceBarrier,
) {
    // A pass that reads-then-writes its own resource produces a self-edge; the hazard
    // is already serialized by the pass itself, so skip it.
    if src == dst {
        return;
    }
    let s = nodes[src];
    let d = nodes[dst];
    if let Some(e) = graph.find_edge(s, d) {
        graph
            .edge_weight_mut(e)
            .expect("edge just found must have a weight")
            .barriers
            .push(barrier);
    } else {
        graph.add_edge(s, d, PassDependency { barriers: vec![barrier] });
    }
}

/// Graph-owned backing for one temporal (cross-frame) resource: one distinct,
/// persistent copy per frame in flight. Unlike transient `Created` resources
/// these are stored as ready-made *imports* and re-registered into the graph on
/// every rebuild, so the transient slot allocator never aliases them (only
/// `Created` resources are aliased) and `reset()` never recycles their memory —
/// each copy keeps its contents across frames, which is exactly what history /
/// ping-pong data needs (TAA accumulation, ReSTIR reservoirs, denoise).
struct TemporalResource {
    /// Absolute frame index when the backing was allocated. Lets the caller
    /// reason about how many frames of history have accumulated so far.
    frame_of_creation: usize,
    /// One persistent backing per frame in flight, kept as a clonable import so
    /// [`RenderGraph::register_temporal_resource`] can wire it into each rebuild.
    imports: [GraphResourceImportInfo; MAX_FRAMES_IN_FLIGHT],
}

/// Exported handle to a temporal resource. Returned by
/// [`RenderGraph::create_temporal_resource`] and kept by the caller across
/// frames: the backing it points at lives in the graph and survives
/// [`RenderGraph::reset`], so the same token re-binds the same GPU memory after a
/// graph rebuild via [`RenderGraph::register_temporal_resource`].
pub struct ExportedTemporalResource<R: Resource> {
    /// Index into [`RenderGraph::temporal_resources`].
    index: usize,
    desc: <R as Resource>::Desc,
    marker: PhantomData<R>,
}

// Manual `Clone` (mirrors `Handle`) so the token is cloneable regardless of
// whether `R` is `Clone` — only the `Desc` is stored.
impl<R: Resource> Clone for ExportedTemporalResource<R> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            desc: self.desc.clone(),
            marker: PhantomData,
        }
    }
}

pub struct RenderGraph {
    next_pass_id: u32,
    next_resource_id: u32,
    //TODO debug hooks and tools
    virtual_resources: Vec<GraphResourceInfo>,
    temporal_resources: Vec<TemporalResource>,
    passes: Vec<AnyRenderPass>,
    /// Frame-in-flight double buffering: one transient pool per slot
    /// (`frame % MAX_FRAMES_IN_FLIGHT`). Each pool frees + rebuilds only its own
    /// backing on `populate`, so recording frame N never frees the transient
    /// memory frame N-1's in-flight GPU work is still reading.
    /// ponytail: N independent pools means the pipeline cache is duplicated per
    /// slot (a handful of extra pipeline builds, then steady-state cached). A
    /// shared cache is a future rework — not worth the entanglement now.
    transient_resources: Vec<TransientResources>,
    /// Per-resource end state (last use + final access) collected by `compile`,
    /// keyed by resource id. See [`ResourceEndState`] — temp impl, exposed so a
    /// later stage can sync against the graph with a barrier instead of a
    /// device-wait-idle; nothing consumes it yet.
    resource_end_states: HashMap<u32, ResourceEndState>,
    /// `(temporal_index, copy_index, resource_id)` for every temporal backing
    /// registered into *this* frame's build. After `compile` computes each
    /// resource's end access, it threads that access back into the matching
    /// `temporal_resources[ti].imports[ci]` so *next* frame's compile emits the
    /// cross-frame barrier for the ping-pong write→read (mirrors what
    /// `Tlas::queue_build` does explicitly for the TLAS). Cleared on `reset`.
    registered_temporal: Vec<(usize, usize, u32)>,
    /// One primary command buffer per frame-in-flight slot, re-recorded when its
    /// slot comes around (reuse gated by [`Self::wait_for_slot_reuse`]).
    cmd_buffers: Vec<CmdBuffer>,
    /// This frame's passes are retired here after submission, kept per slot until
    /// the slot is reused N frames later. Passes own the AS-build scratch the GPU
    /// reads during the submission, so they must outlive it — see `run` / `reset`.
    retired_passes: Vec<Vec<AnyRenderPass>>,
    /// This frame's imported/created virtual resources are retired here after
    /// submission, kept per slot until the slot is reused N frames later. Imports
    /// hold the `Arc` that keeps a resource's backing alive (notably the TLAS,
    /// which `Tlas::queue_build` swaps for a freshly-allocated structure every
    /// frame); dropping them at the next `reset` — as the pre-overlap code did,
    /// when the previous frame was always already idle — would free memory the
    /// in-flight previous frame is still reading. See `run` / `reset`.
    retired_resources: Vec<Vec<GraphResourceInfo>>,
    /// Arena staging→GPU copies to record as a transfer prologue at the head of
    /// this frame's submission (handed over by the resource manager on asset
    /// load). Cleared on `reset`. See [`Self::add_prologue_buffer_copies`].
    prologue_copies: Vec<(vk::Buffer, Handle<RawBuffer>, vk::BufferCopy)>,
    /// Signaled with the absolute frame count when each frame's graph submission
    /// completes. Drives CPU slot-reuse gating, the cross-frame temporal
    /// ping-pong wait (frame F's graph waits F-1's), and the blit's wait on the
    /// graph — together replacing the old per-frame fence + device-wait-idle.
    /// TODO to be removed for the outside frame timeline semaphore
    graph_timeline: TimelineSemaphore,
    /// Interned `'static` checkpoint markers, keyed by pass name. Aftermath reads
    /// `p_checkpoint_marker` back *after* a DEVICE_LOST, so the string must
    /// outlive the frame — leaked once per unique pass name (a bounded set).
    /// Only populated when the Aftermath diagnostic tool is active.
    checkpoint_markers: HashMap<String, &'static std::ffi::CStr>,
    /// Cached so `run` can submit and `compile` can record without the caller
    /// having to re-thread `Core` through every call.
    core: Rc<Core>,
}
//TODO per frame global data uploaded each frame like transforms and the camera, these can then live in the descriptor heap based on kajiya DYNAMIC_CONSTANTS_BUFFER
//TODO Reintroduce the typestate of the render graph,as it is intended to work like this, the setup phase is where you can add stuff and so on, when you want to run it you compile it once done than you can return to the setup phase, this should empty out reset the cmdbuffer and allow to add again resources, this should make sure the resources in use are
//   not overwritten though while still allowing new resources to be added,also while on a built state it should be able to handle n frames in flight with internal sync to minimize the wait idle time and allow multiple frame to be run concurrently, this
impl RenderGraph {
    pub fn new(core: Rc<Core>) -> SrResult<Self> {
        let cmd_buffers = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| CmdBuffer::new(Rc::clone(&core)))
            .collect::<SrResult<Vec<_>>>()?;
        let transient_resources = (0..MAX_FRAMES_IN_FLIGHT).map(|_| TransientResources::default()).collect();
        let retired_passes = (0..MAX_FRAMES_IN_FLIGHT).map(|_| Vec::new()).collect();
        let retired_resources = (0..MAX_FRAMES_IN_FLIGHT).map(|_| Vec::new()).collect();
        let graph_timeline = TimelineSemaphore::new(Rc::clone(&core), 0)?;
        Ok(RenderGraph {
            next_pass_id: 0,
            next_resource_id: 0,
            passes: vec![],
            virtual_resources: vec![],
            temporal_resources: vec![],
            transient_resources,
            resource_end_states: HashMap::new(),
            registered_temporal: Vec::new(),
            checkpoint_markers: HashMap::new(),
            cmd_buffers,
            retired_passes,
            retired_resources,
            prologue_copies: vec![],
            graph_timeline,
            core,
        })
    }

    /// The frame-in-flight slot the current absolute frame maps to. `reset`,
    /// `compile` and `run` all run *after* `build_unified_graph` has incremented
    /// the absolute frame count, so this reads the frame being recorded.
    fn current_slot(&self) -> usize {
        *self.core.absolute_frame_count.borrow() % MAX_FRAMES_IN_FLIGHT
    }

    /// Block until the frame that last used the upcoming frame's slot
    /// (`frame - MAX_FRAMES_IN_FLIGHT`) has finished its graph submission, so this
    /// slot's command buffer, transient pool and retired passes can be safely
    /// re-recorded / freed on the CPU. Call once at the very top of a frame,
    /// before touching any slot state (including the resource manager's arena slot
    /// reclamation). Non-blocking in steady state — that frame completed long ago.
    /// Must be called *before* `build_unified_graph` increments the frame count.
    pub fn wait_for_slot_reuse(&self) -> SrResult<()> {
        let upcoming = *self.core.absolute_frame_count.borrow() as u64 + 1;
        if upcoming > MAX_FRAMES_IN_FLIGHT as u64 {
            self.graph_timeline.wait(upcoming - MAX_FRAMES_IN_FLIGHT as u64)?;
        }
        Ok(())
    }

    /// The graph completion timeline (signaled with the absolute frame count by
    /// each `run`). The caller makes its post-graph work (the blit) wait on this
    /// instead of a CPU fence, so it can be enqueued without stalling.
    pub fn graph_timeline_inner(&self) -> vk::Semaphore {
        self.graph_timeline.inner()
    }

    /// Block until the graph timeline reaches `value` (the absolute frame count a
    /// frame's submission signals on completion). This is now the single
    /// frame-completion timeline — the renderer's former separate `frame_timeline`
    /// was collapsed into it once the present blit moved inside the graph submit.
    pub fn wait_graph_timeline(&self, value: u64) -> SrResult<()> {
        self.graph_timeline.wait(value)
    }

    /// Hand the graph a batch of arena staging→GPU buffer copies to record as a
    /// transfer prologue pass. Each destination is an arena buffer imported into
    /// *this* build (see `ResourceManager::import_to_graph`).
    pub fn add_prologue_buffer_copies(
        &mut self,
        mut copies: Vec<(
            vk::Buffer,
            Handle<RawBuffer>,
            vk::BufferCopy,
        )>,
    ) {
        self.prologue_copies.append(
            &mut copies
        );
    }

    /// Clear all per-frame state (passes, virtual resources, transient bindings,
    /// swapchain import, recorded barrier trace) so the graph can be rebuilt
    /// from scratch on the next frame. The persistent `CmdBuffer` and cached
    /// `Core` survive: this is the entry point for "graph is an attribute of
    /// the renderer, rebuilt each frame, but the underlying primary command
    /// buffer is reused".
    pub fn reset(&mut self) {
        let slot = self.current_slot();
        self.next_pass_id = 0;
        self.next_resource_id = 0;
        self.passes.clear();
        self.virtual_resources.clear();
        self.prologue_copies.clear();
        self.resource_end_states.clear();
        self.registered_temporal.clear();
        // Free the previous occupant of this slot (frame N - MAX_FRAMES_IN_FLIGHT):
        // its passes own the AS-build scratch the GPU read, and `wait_for_slot_reuse`
        // proved that frame's submission is complete. This slot's transient pool is
        // freed + rebuilt by `populate` during `compile`.
        self.retired_passes[slot].clear();
        // Same reuse gate frees the previous occupant's retired imports/created
        // resources (the `Arc`s keeping their backings — e.g. that frame's TLAS —
        // alive); `run` parked them here after submission.
        self.retired_resources[slot].clear();
        // `transient_resources[slot]` (freed by `populate`), `temporal_resources`
        // and each pool's `pipeline_cache` intentionally persist across the rebuild
        // so history / ping-pong data and interned pipelines survive. Re-wire each
        // temporal resource with `register_temporal_resource` while rebuilding.
    }

    pub(super) fn next_pass_id(&mut self) -> u32 {
        let id = self.next_pass_id;
        self.next_pass_id += 1;
        id
    }
    pub(super) fn next_resource_id(&mut self) -> u32 {
        let id = self.next_resource_id;
        self.next_resource_id += 1;
        id
    }
    pub fn create_resource<Desc: ResourceDesc>(&mut self, desc: Desc) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        self.create_raw_resource(desc.clone().into());
        Handle {
            id: self.next_resource_id(),
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }

    /// Allocate a temporal (cross-frame) resource: `MAX_FRAMES_IN_FLIGHT`
    /// dedicated, persistent copies of `desc` that the graph owns for its whole
    /// lifetime. The backing is allocated once here; it is **not** aliased with
    /// transient resources and survives [`Self::reset`], so each copy preserves
    /// its contents from frame to frame (history buffers, ping-pong targets).
    ///
    /// Returns an [`ExportedTemporalResource`] the caller keeps across frames.
    /// Each frame (including the first), after `reset`, call
    /// [`Self::register_temporal_resource`] with this token to wire the copies
    /// into the rebuilt graph and obtain the per-frame [`Handle`]s.
    ///
    /// Only images and buffers can be temporal; samplers / acceleration
    /// structures return an error.
    pub fn create_temporal_resource<Desc: ResourceDesc>(
        &mut self,
        desc: Desc,
    ) -> SrResult<ExportedTemporalResource<<Desc as ResourceDesc>::Resource>>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        let graph_desc: GraphResourceDesc = desc.clone().into();

        let mut backings: Vec<GraphResourceImportInfo> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let backing = self.allocate_temporal_backing(&graph_desc)?;
            // Name each ping-pong copy for GPU captures, e.g.
            // "ReSTIR GI Reservoir Buffer[0]" (no-op without debug-utils).
            if self.core.debug_labels_enabled()
                && let Some(name) = graph_desc_name(&graph_desc)
                && let Ok(cname) = std::ffi::CString::new(format!("{name}[{i}]"))
            {
                name_import(&self.core, &backing, &cname);
            }
            backings.push(backing);
        }
        let imports: [GraphResourceImportInfo; MAX_FRAMES_IN_FLIGHT] = backings
            .try_into()
            .unwrap_or_else(|_| unreachable!("allocated exactly MAX_FRAMES_IN_FLIGHT backings"));

        let index = self.temporal_resources.len();
        self.temporal_resources.push(TemporalResource {
            frame_of_creation: *self.core.absolute_frame_count.borrow(),
            imports,
        });

        Ok(ExportedTemporalResource {
            index,
            desc: TypeEquals::same(desc),
            marker: PhantomData,
        })
    }

    /// Register the persistent backing of an exported temporal resource into the
    /// current graph build, returning a [`Handle`] for each frame-in-flight copy
    /// (index `i` is the copy for frame `i`; the caller selects current vs.
    /// history by frame parity). Call once per rebuild after [`Self::reset`].
    ///
    /// Copies are wired in as imports, so they bypass transient aliasing and are
    /// never recycled — see [`Self::create_temporal_resource`].
    pub fn register_temporal_resource<R: Resource>(
        &mut self,
        exported: &ExportedTemporalResource<R>,
    ) -> [Handle<R>; MAX_FRAMES_IN_FLIGHT] {
        let imports = self.temporal_resources[exported.index].imports.clone();
        std::array::from_fn(|i| {
            let id = self.next_resource_id();
            self.virtual_resources.push(GraphResourceInfo::Imported(imports[i].clone()));
            // Remember this backing's resource id so `compile` can thread its
            // end-of-frame access back into the stored import (cross-frame sync).
            self.registered_temporal.push((exported.index, i, id));
            Handle {
                id,
                desc: exported.desc.clone(),
                marker: PhantomData,
            }
        })
    }

    /// Absolute frame index at which this temporal resource's backing was
    /// allocated. The number of frames of history accumulated so far is
    /// `current_absolute_frame - temporal_frame_of_creation`.
    pub fn temporal_frame_of_creation<R: Resource>(&self, exported: &ExportedTemporalResource<R>) -> usize {
        self.temporal_resources[exported.index].frame_of_creation
    }

    /// The persistent per-frame backing images of a temporal *image* resource.
    /// The graph never transitions imported resources itself, so the caller uses
    /// these to drive a one-time layout transition right after (re)creation.
    pub fn temporal_image_backings(&self, exported: &ExportedTemporalResource<Image>) -> [Arc<Image>; MAX_FRAMES_IN_FLIGHT] {
        let imports = &self.temporal_resources[exported.index].imports;
        std::array::from_fn(|i| match &imports[i] {
            GraphResourceImportInfo::Image { resource, .. } => Arc::clone(resource),
            _ => unreachable!("temporal image resource backed by a non-image import"),
        })
    }

    /// The device addresses of the persistent per-frame backing buffers of a
    /// temporal *buffer* resource. Ping-pong buffers are still reached by device
    /// address in the shader (the graph import only governs synchronization), so
    /// the caller bakes these into its push constants.
    pub fn temporal_buffer_addresses(
        &self,
        exported: &ExportedTemporalResource<RawBuffer>,
    ) -> [vk::DeviceAddress; MAX_FRAMES_IN_FLIGHT] {
        let imports = &self.temporal_resources[exported.index].imports;
        std::array::from_fn(|i| match &imports[i] {
            GraphResourceImportInfo::Buffer { resource, .. } => resource.get_device_address(),
            _ => unreachable!("temporal buffer resource backed by a non-buffer import"),
        })
    }

    /// Drop every temporal resource's backing memory. Existing
    /// [`ExportedTemporalResource`] tokens dangle afterwards, so only call this
    /// when about to recreate them (e.g. a resize that changes their dimensions)
    /// and replace every token the caller holds. The caller must ensure the GPU
    /// is idle first — the backings may still be in use by an in-flight frame.
    pub fn clear_temporal_resources(&mut self) {
        self.temporal_resources.clear();
    }

    /// Allocate one owned, dedicated backing for a temporal resource and wrap it
    /// as an import ready to be registered each frame. The backing carries its
    /// own memory (so it is never aliased) and is reference-counted, so the graph
    /// can clone it into every rebuild while keeping it alive across resets.
    fn allocate_temporal_backing(&self, desc: &GraphResourceDesc) -> SrResult<GraphResourceImportInfo> {
        match desc {
            GraphResourceDesc::Image(image_desc) => {
                let image = Arc::new(Image::new_from_desc(self.core(), image_desc)?);
                Ok(GraphResourceImportInfo::Image {
                    resource: image,
                    access_type: vk_sync::AccessType::Nothing,
                })
            }
            GraphResourceDesc::Buffer(buffer_desc) => {
                let buffer = Arc::new(RawBuffer::new_from_desc(self.core(), buffer_desc)?);
                Ok(GraphResourceImportInfo::Buffer {
                    resource: buffer,
                    access_type: vk_sync::AccessType::Nothing,
                })
            }
            GraphResourceDesc::Sampler(_) | GraphResourceDesc::RaytracingAS(_) => Err(SrError::new_custom(
                "temporal resources are only supported for images and buffers".to_string(),
            )),
        }
    }

    fn create_raw_resource(&mut self, resource_desc: GraphResourceDesc) {
        self.virtual_resources.push(GraphResourceInfo::Created(resource_desc));
    }

    pub fn import<Desc: ResourceDesc>(
        &mut self,
        res: impl RgImportable<Desc> + Into<GraphResourceImportInfo>,
    ) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        let desc = res.import();
        self.virtual_resources.push(GraphResourceInfo::Imported(res.into()));
        Handle {
            id: self.next_resource_id(),
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }
    /// Like [`Self::import`], but overrides the access the resource is treated as
    /// carrying *into* this compile with `usage` — the state the previous frame's
    /// submission left it in. `compile` seeds a cross-frame init barrier from it
    /// (see `imported_initial_access`), so the caller can thread a resource's
    /// end-state back in and let the graph emit the hand-off barrier instead of a
    /// device-wide idle. Samplers / swapchain images carry no cross-frame access,
    /// so `usage` is ignored for them.
    pub fn import_with_usage<Desc: ResourceDesc>(
        &mut self,
        res: impl RgImportable<Desc> + Into<GraphResourceImportInfo>,
        usage: vk_sync::AccessType,
    ) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        let desc = res.import();
        let mut import = res.into();
        match &mut import {
            GraphResourceImportInfo::Image { access_type, .. }
            | GraphResourceImportInfo::Buffer { access_type, .. }
            | GraphResourceImportInfo::RayTracingAcceleration { access_type, .. } => *access_type = usage,
            GraphResourceImportInfo::Sampler { .. } => {}
        }
        self.virtual_resources.push(GraphResourceInfo::Imported(import));
        Handle {
            id: self.next_resource_id(),
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }

    pub fn add_render_pass(&mut self, render_pass: impl Into<AnyRenderPass>) {
        self.passes.push(render_pass.into())
    }

    /// Add a pass that records a deferred acceleration-structure build/update
    /// ([`AsBuildJob`]) into the graph's command buffer. The pass declares a write
    /// on `build_target` (so consumers — the TLAS build, the RT trace — are ordered
    /// after it) and a read on each of `deps` (a TLAS build reads the BLASes it
    /// references, so their builds are ordered before it). Scratch is allocated
    /// here, sized from the job, and kept alive by the pass closure until the next
    /// `reset` (past this frame's fence). See `ResourceManager::queue_*`.
    pub fn add_as_build_pass(
        &mut self,
        name: &str,
        build_target: &Handle<AccelerationStructure>,
        deps: &[Handle<AccelerationStructure>],
        job: AsBuildJob,
    ) -> SrResult<()> {
        let scratch = GpuOnlyBuffer::new_aligned::<u8>(
            Rc::clone(&self.core),
            job.scratch_size,
            job.scratch_alignment,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            "render graph AS build scratch",
        )?;

        let mut common = PassCommonDataBuilder::new(self, name);
        common.write(build_target, vk_sync::AccessType::AccelerationStructureBuildWrite)?;
        for dep in deps {
            common.read(dep, vk_sync::AccessType::AccelerationStructureBuildRead)?;
        }

        // The job is `FnOnce`; the render closure is `FnMut`, so take it out on the
        // first (only) invocation. `scratch` is owned by the closure and outlives
        // the submission (dropped at the next `reset`, guarded by the frame fence).
        let mut job = Some(job);
        common.render(move |cb, _tr| {
            if let Some(job) = job.take() {
                job.record(*cb, &scratch);
            }
            Ok(())
        });

        let pass = common.build_transfer();
        self.add_render_pass(pass);
        Ok(())
    }

    /// The graph's cached `Core`. Pass builders use this so callers no longer
    /// have to thread `Rc<Core>` into `generate_render` separately.
    pub(crate) fn core(&self) -> Rc<Core> {
        Rc::clone(&self.core)
    }

    /// Intern a heap-mode compute pipeline for `spirv`, returning a handle the
    /// render closure resolves at record time. Built once per distinct shader and
    /// reused across frame rebuilds (see [`PipelineCache`]).
    pub(crate) fn cache_compute_pipeline(&mut self, spirv: &[u8]) -> SrResult<PipelineHandle> {
        let key = pipeline_cache_key(0, &[spirv]);
        let core = Rc::clone(&self.core);
        let slot = self.current_slot();
        self.transient_resources[slot].pipeline_cache.intern(key, &core, || {
            let pipeline = ComputePipeline::<HeapComputePass>::new(core.clone_device(), spirv)?;
            Ok(CachedPipeline::Compute(Rc::new(pipeline)))
        })
    }

    /// Intern a heap-mode ray-tracing pipeline + its shader binding table.
    pub(crate) fn cache_raytracing_pipeline(&mut self, shaders: &RayTracingPipelineShaders) -> SrResult<PipelineHandle> {
        let key = pipeline_cache_key(1, &[&shaders.ray_gen, &shaders.miss, &shaders.closest_hit, &shaders.any_hit]);
        let core = Rc::clone(&self.core);
        let slot = self.current_slot();
        self.transient_resources[slot].pipeline_cache.intern(key, &core, || {
            let pipeline = Rc::new(RayTracingPipeline::new(Rc::clone(&core), shaders)?);
            let sbt = Rc::new(ShaderBindingTable::new(&core, &pipeline)?);
            Ok(CachedPipeline::RayTracing(pipeline, sbt))
        })
    }

    /// Intern a heap-mode graphics pipeline. The vertex layout is currently not
    /// part of the cache key (only vertex+fragment SPIR-V and color format are) —
    /// fine while the raster path is experimental and single-layout.
    pub(crate) fn cache_graphics_pipeline(&mut self, shaders: &GraphicsPipelineShaders) -> SrResult<PipelineHandle> {
        let key = pipeline_cache_key(
            2,
            &[
                &shaders.vertex,
                &shaders.fragment,
                &shaders.color_format.as_raw().to_ne_bytes(),
            ],
        );
        let core = Rc::clone(&self.core);
        let slot = self.current_slot();
        self.transient_resources[slot].pipeline_cache.intern(key, &core, || {
            let pipeline = GraphicsPipeline::new(Rc::clone(&core), shaders)?;
            Ok(CachedPipeline::Graphics(Rc::new(pipeline)))
        })
    }

    /// Bare `vkCmdBlitImage` from `src` (already in TRANSFER_SRC) to `dst` (already
    /// in TRANSFER_DST) — the caller owns the surrounding layout barriers. Scales if
    /// the extents differ (nearest). Used by [`Self::run_present`].
    fn record_present_blit(core: &Core, cb: vk::CommandBuffer, src: &Image, dst: &Image) {
        let device = core.device().inner();
        let src_ext = src.extent();
        let dst_ext = dst.extent();
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(0);
        let blit = vk::ImageBlit::default()
            .src_subresource(layers)
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: src_ext.width as i32,
                    y: src_ext.height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(layers)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: dst_ext.width as i32,
                    y: dst_ext.height as i32,
                    z: 1,
                },
            ]);
        unsafe {
            device.cmd_blit_image(
                cb,
                src.inner(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.inner(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::NEAREST,
            );
        }
    }

    pub fn compile(&mut self) -> SrResult<()> {
        //TODO force injection of previous temporal data on rebuild, the graph is incapable of understanding temporal dep across compilations
        //TODO mark the render pass goals as the result of the graph so anything unnecessary can be removed
        //TODO there are some complex optimizations as shown here https://www.youtube.com/watch?v=v9LaTFLhP38 and this is the site where it will be published the paper https://dl.acm.org/profile/99661091135
        //TODO respect PassResourceAccessSyncType (NeverSync / SkipSyncIfSameAccessType) when deciding whether to emit a barrier
        //TODO it currently returns a one time submit, but the cmd buffer can be reuse as long as the graph doesn't get rebuilt this requires the temporal stuff though and some rework on the sync side between each frame
        //TODO the render graph currently has no way to export the data, this is useful to synchronize across frames. The data should be released and reused basically each frame. This put a constraint, mutable data imported into the graph is hard to work with,
        // for example you could build a tlas the next frame if this is seen as an internal or created on the spot data structure, but exporting it would block the cpu on interacting with it until the previous frame has ended.
        // To further emphasise this there will need to be a dedicated way to handle multiple data based of frames in flight , transformation matrices and the camera should only live as long as a frame.





        if !self.prologue_copies.is_empty() {

            let mut hashset = HashSet::new();
            let mut prologue_copies_node_builder = PassCommonDataBuilder::new(self, "Internal prologue copy node");

            for dst in self.prologue_copies.iter().map(|(_, dst, _)| dst) {
                if hashset.insert(dst.id) {
                    prologue_copies_node_builder.write(&dst, AccessType::TransferWrite)?;
                }
            }

            let device_out = self.core.device().clone();
            let prologue_copies_for_closure = self.prologue_copies.clone();
            prologue_copies_node_builder.render(move |cmd_buffer, transient_resources| {
                let device = device_out.inner();

                unsafe {
                    for (src, dst, region) in prologue_copies_for_closure.iter() {
                        if let Some(dst_raw) = transient_resources.external_buffers.get(&dst.id) {
                            device.cmd_copy_buffer(*cmd_buffer, *src, dst_raw.inner(), std::slice::from_ref(region));

                        }else {
                            return Err(SrError::new( ErrorSource::RenderGraph(GraphError::InvalidResourceRef), format!("External Buffer with id not found {:?}" , &dst.id)));
                        }
                    }
                }
                SrResult::Ok(())
            });


            self.passes.insert(
                0,
                AnyRenderPass::Transfer(prologue_copies_node_builder.build_transfer())
            );
        }

        //From now on the graph passes should not be touched

        let slot = self.current_slot();
        let pass_count = self.passes.len();

        let mut resource_usages: BTreeMap<u32, ResourceLifetimeUsage> = BTreeMap::new();
        let mut hazard_states: HashMap<u32, ResourceHazardState> = HashMap::new();

        let mut dep_graph = petgraph::graph::DiGraph::<usize, PassDependency>::with_capacity(pass_count, pass_count * 2);

        //Note: acyclic graph ir with phi

        let pass_nodes: Vec<petgraph::graph::NodeIndex> = (0..pass_count).map(|i| dep_graph.add_node(i)).collect();

        //TODO let mut initialization_node = AnyRenderPass::Compute();

        for (pass_id, pass) in self.passes.iter().enumerate() {
            let common = match pass {
                AnyRenderPass::Rt(rt) => &rt.common,
                AnyRenderPass::Raster(raster) => &raster.common,
                AnyRenderPass::Compute(compute) => &compute.common,
                AnyRenderPass::Transfer(transfer) => &transfer.common,
            };

            for read in &common.read {
                let res_id = read.id;
                record_usage(&mut resource_usages, res_id, pass_id, read.access);
                let state = hazard_states.entry(res_id).or_default();
                if let Some((w_pass, w_access)) = state.last_writer {
                    add_dep_edge(
                        &mut dep_graph,
                        &pass_nodes,
                        w_pass,
                        pass_id,
                        ResourceBarrier {
                            resource_id: res_id,
                            prev_access: w_access,
                            next_access: read.access.access_type,
                        },
                    );
                }

                state.readers_since_write.push((pass_id, read.access.access_type));
            }

            for write in &common.write {
                let res_id = write.id;
                record_usage(&mut resource_usages, res_id, pass_id, write.access);
                let state = hazard_states.entry(res_id).or_default();
                if !state.readers_since_write.is_empty() {
                    for (r_pass, r_access) in &state.readers_since_write {
                        add_dep_edge(
                            &mut dep_graph,
                            &pass_nodes,
                            *r_pass,
                            pass_id,
                            ResourceBarrier {
                                resource_id: res_id,
                                prev_access: *r_access,
                                next_access: write.access.access_type,
                            },
                        );
                    }
                } else if let Some((w_pass, w_access)) = state.last_writer {
                    add_dep_edge(
                        &mut dep_graph,
                        &pass_nodes,
                        w_pass,
                        pass_id,
                        ResourceBarrier {
                            resource_id: res_id,
                            prev_access: w_access,
                            next_access: write.access.access_type,
                        },
                    );
                }
                state.last_writer = Some((pass_id, write.access.access_type));
                state.readers_since_write.clear();
            }
        }

        self.set_resources_end_states(&mut resource_usages);

        // Cross-frame sync for temporal (ping-pong / history) resources: thread
        // each backing's end access this frame back into its stored import, so
        // *next* frame's compile emits the read→write (or write→read) barrier for
        // the same physical backing across the frame boundary. Without this the
        // imports always re-enter as `Nothing` and the hazard graph — which only
        // orders passes *within* one compile — never synchronizes the ping-pong
        // reuse, leaving frame N's reservoir/accumulation write unordered against
        // frame N+1's read of the same memory. The TLAS gets this treatment
        // explicitly via `Tlas::queue_build`; temporal resources get it here.
        for &(ti, ci, rid) in &self.registered_temporal {
            if let Some(end) = self.resource_end_states.get(&rid) {
                set_import_access(&mut self.temporal_resources[ti].imports[ci], end.end_access);
            }
        }

        // Weakly-connected components via union-find over dependency edges. Any resource
        // shared by multiple passes already produced at least one hazard edge above, so
        // passes that share a resource end up in the same component.
        let mut uf = petgraph::unionfind::UnionFind::<usize>::new(pass_count);
        for edge in dep_graph.edge_indices() {
            let (a, b) = dep_graph.edge_endpoints(edge).expect("edge from iterator must exist");
            uf.union(a.index(), b.index());
        }
        let labels = uf.into_labeling();

        let mut components_by_root: HashMap<usize, PassComponent> = HashMap::new();
        for (pass_id, root) in labels.iter().enumerate() {
            components_by_root
                .entry(*root)
                .or_insert_with(|| PassComponent {
                    passes: vec![],
                    resources: vec![],
                })
                .passes
                .push(pass_id);
        }
        for (res_id, usage) in &resource_usages {
            let root = labels[usage.first_pass];
            components_by_root
                .get_mut(&root)
                .expect("pass component must exist for any resource that was touched")
                .resources
                .push(*res_id);
        }
        let components: Vec<PassComponent> = components_by_root.into_values().collect();

        self.transient_resources[slot].populate(Rc::clone(&self.core), &self.virtual_resources, &components, &resource_usages)?;

        // Topological order of passes. petgraph's toposort fails iff there is a
        // cycle, which would be a logic bug since hazards only ever produce
        // forward (lower pass_id → higher pass_id) edges by construction.
        let topo = petgraph::algo::toposort(&dep_graph, None)
            .map_err(|_| SrError::new_custom("render graph dependency graph contains a cycle".to_string()))?;

        // Pre-group all barriers by the pass that needs them issued *before* it
        // runs. Each dep edge contributes its barriers to the destination pass.
        let mut incoming: HashMap<usize, Vec<ResourceBarrier>> = HashMap::new();
        for edge in dep_graph.edge_references() {
            let dst = dep_graph[edge.target()];
            incoming
                .entry(dst)
                .or_default()
                .extend(edge.weight().barriers.iter().cloned());
        }

        let device = self.core.device().inner().clone();
        // This slot's command buffer was allocated in `RenderGraph::new`. Reset it
        // before re-recording — the pool was created with `RESET_COMMAND_BUFFER`,
        // so per-buffer reset is allowed. No `ONE_TIME_SUBMIT` flag since the buffer
        // is re-used every N frames; `wait_for_slot_reuse` guaranteed this slot's
        // previous submission is complete before we reset it.
        let raw_cb = self.cmd_buffers[slot].inner();
        unsafe {
            device.reset_command_buffer(raw_cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(raw_cb, &vk::CommandBufferBeginInfo::default())?;
        }

        // Initial layout transitions for created (transient) images. Their memory
        // is freshly bound this frame, so the image is in UNDEFINED; the first
        // pass that touches one accesses it through a storage/sampled descriptor
        // that requires GENERAL / SHADER_READ_ONLY. The hazard graph only emits
        // producer->consumer barriers, so a created image that is *written first*
        // (the common case: an RT/compute pass producing it) would otherwise be
        // accessed while still UNDEFINED. Discard-transition each one up front to
        // the layout implied by its first access. Imported resources are excluded:
        // they carry a layout from outside the graph (or across frames).
        let mut init_barriers: Vec<ResourceBarrier> = Vec::new();
        for (res_id, usage) in &resource_usages {
            let is_created_image = matches!(
                self.virtual_resources.get(*res_id as usize),
                Some(GraphResourceInfo::Created(GraphResourceDesc::Image(_)))
            );
            if !is_created_image {
                continue;
            }
            if let Some((_, first_access)) = usage.usages.first() {
                init_barriers.push(ResourceBarrier {
                    resource_id: *res_id,
                    prev_access: vk_sync::AccessType::Nothing,
                    next_access: first_access.access_type,
                });
            }
        }

        // Cross-frame init barriers for *imported* resources that come into this
        // frame carrying a non-`Nothing` access (the state the previous frame's
        // submission left them in, threaded back in by the caller — e.g. a TLAS an
        // in-place update inherits as `RayTracingShaderReadAccelerationStructure`).
        // The hazard graph only orders passes *within* this compile, so without
        // this a resource whose first in-graph use conflicts with its prior-frame
        // access (a build writing a just-traced TLAS) would be unsynchronized. This
        // is what lets the caller drop the device-wide idle wait between frames.
        for (res_id, usage) in &resource_usages {
            let prev_access = match self.virtual_resources.get(*res_id as usize) {
                Some(GraphResourceInfo::Imported(import)) => imported_initial_access(import),
                _ => continue,
            };
            if prev_access == vk_sync::AccessType::Nothing {
                continue;
            }
            if let Some((_, first_access)) = usage.usages.first() {
                init_barriers.push(ResourceBarrier {
                    resource_id: *res_id,
                    prev_access,
                    next_access: first_access.access_type,
                });
            }
        }

        if !init_barriers.is_empty() {
            self.transient_resources[slot].emit_barriers(&device, raw_cb, &init_barriers);
        }

        // Pass names, gathered up front (the loop borrows `self.passes` mutably):
        // used both for GPU-capture labels and the optional graph dump below.
        let pass_names: Vec<String> = self.passes.iter().map(pass_common_name).collect();
        // Pre-build nul-terminated labels only when a capture tool is active.
        let labels_on = self.core.debug_labels_enabled();
        let pass_clabels: Vec<Option<std::ffi::CString>> = if labels_on {
            pass_names.iter().map(|n| std::ffi::CString::new(n.as_str()).ok()).collect()
        } else {
            Vec::new()
        };
        // Per-pass Aftermath checkpoints: a completed checkpoint means the GPU
        // reached that pass, so after a DEVICE_LOST the last one logged names the
        // faulting pass. No-op unless Aftermath is active.
        let checkpoints_on =
            self.core.diagnostic_tool() == crate::vulkan_abstraction::diagnostics::DiagnosticTool::NvidiaAftermath;
        let pass_markers: Vec<&'static std::ffi::CStr> = if checkpoints_on {
            pass_names.iter().map(|n| self.intern_marker(n)).collect()
        } else {
            Vec::new()
        };

        // Drive each pass in topological order. We borrow `self.passes` mutably
        // (closures are FnMut) but only `self.transient_resources` immutably, so
        // the disjoint-field split borrow is fine.
        for node in &topo {
            let pass_id = dep_graph[*node];

            if let Some(barriers) = incoming.remove(&pass_id) {
                self.transient_resources[slot].emit_barriers(&device, raw_cb, &barriers);
                self.transient_resources[slot].recorded_barriers.push((pass_id, barriers));
            }

            if checkpoints_on {
                self.core.cmd_set_checkpoint(raw_cb, pass_markers[pass_id]);
            }

            // Bracket the pass in a debug-utils label so it shows as a named
            // scope in an Nsight Graphics / RenderDoc capture (no-op otherwise).
            let has_label = labels_on && matches!(pass_clabels.get(pass_id), Some(Some(_)));
            if has_label {
                self.core
                    .cmd_begin_debug_label(raw_cb, pass_clabels[pass_id].as_ref().unwrap());
            }

            let common = match &mut self.passes[pass_id] {
                AnyRenderPass::Rt(rt) => &mut rt.common,
                AnyRenderPass::Raster(raster) => &mut raster.common,
                AnyRenderPass::Compute(compute) => &mut compute.common,
                AnyRenderPass::Transfer(transfer) => &mut transfer.common,
            };
            if let Some(render) = common.render.as_mut() {
                let mut cb_handle = raw_cb;
                render(&mut cb_handle, &self.transient_resources[slot])?;
            }

            if has_label {
                self.core.cmd_end_debug_label(raw_cb);
            }
        }

        // The command buffer is left *open* on purpose: `run` (offscreen) or
        // `run_present` (swapchain) append their tail — nothing, or the
        // blit-to-swapchain + PRESENT_SRC transition — then end and submit it.

        // Optional per-frame graph dump (DOT + text) for offline visualization —
        // enabled by setting `SUNRAY_GRAPH_DUMP_DIR`. Cheap gate: only builds the
        // dump when the env var is present.
        if let Ok(dir) = std::env::var("SUNRAY_GRAPH_DUMP_DIR") {
            self.dump_graph(&dir, &pass_names, &dep_graph, &init_barriers, slot);
        }

        Ok(())
    }
    /// Export the end state of every resource: the last pass that touches it
    /// and the access it is left in when the submission completes. Usages are
    /// recorded in pass-id order, so the last entry is the latest pass.
    /// TODO(temp impl): nothing consumes these yet — see `ResourceEndState`
    /// for the intended pass-to-pass cross-submission sync.
    fn set_resources_end_states(&mut self, resource_usages: &mut BTreeMap<u32, ResourceLifetimeUsage>) {
        self.resource_end_states.clear();
        for (res_id, info) in self.virtual_resources.iter().enumerate() {
            let res_id = res_id as u32;
            let internal = matches!(info, GraphResourceInfo::Created(_));
            let (last_use_pass, end_access) = match resource_usages.get(&res_id).and_then(|u| u.usages.last()) {
                Some((pass, access)) => (Some(*pass), access.access_type),
                None => (None, vk_sync::AccessType::Nothing),
            };
            self.resource_end_states.insert(
                res_id,
                ResourceEndState {
                    last_use_pass,
                    end_access,
                    internal,
                },
            );
        }
    }

    /// Return a `'static` checkpoint marker for `name`, leaking a fresh
    /// `CString` the first time each name is seen (pass names are a bounded set,
    /// so this leaks a handful of strings total over the program's life).
    fn intern_marker(&mut self, name: &str) -> &'static std::ffi::CStr {
        if let Some(m) = self.checkpoint_markers.get(name) {
            return m;
        }
        let leaked: &'static std::ffi::CStr = Box::leak(std::ffi::CString::new(name).unwrap_or_default().into_boxed_c_str());
        self.checkpoint_markers.insert(name.to_owned(), leaked);
        leaked
    }

    /// Build and write a [`graph_debug::GraphDump`] for the just-compiled frame.
    /// Split out of `compile` to keep that function readable; only called when
    /// `SUNRAY_GRAPH_DUMP_DIR` is set.
    fn dump_graph(
        &self,
        dir: &str,
        pass_names: &[String],
        dep_graph: &petgraph::graph::DiGraph<usize, PassDependency>,
        init_barriers: &[ResourceBarrier],
        slot: usize,
    ) {
        use crate::render_graph::graph_debug::{GraphDump, ResourceDumpInfo};

        let transient = &self.transient_resources[slot];
        let resources: Vec<ResourceDumpInfo> = self
            .virtual_resources
            .iter()
            .enumerate()
            .map(|(id, info)| {
                let id = id as u32;
                let (kind, detail, import_access) = match info {
                    GraphResourceInfo::Created(GraphResourceDesc::Image(d)) => (
                        "created-image",
                        format!("{} {}x{}", d.name, d.extent.width, d.extent.height),
                        None,
                    ),
                    GraphResourceInfo::Created(GraphResourceDesc::Buffer(d)) => {
                        ("created-buffer", format!("{} {}B", d.name, d.byte_size), None)
                    }
                    GraphResourceInfo::Created(GraphResourceDesc::Sampler(_)) => ("created-sampler", String::new(), None),
                    GraphResourceInfo::Created(GraphResourceDesc::RaytracingAS(_)) => ("created-as", String::new(), None),
                    GraphResourceInfo::Imported(import) => {
                        let access = Some(imported_initial_access(import));
                        match import {
                            GraphResourceImportInfo::Image { resource, .. } => {
                                let e = resource.extent();
                                ("imported-image", format!("{}x{}", e.width, e.height), access)
                            }
                            GraphResourceImportInfo::Buffer { resource, .. } => {
                                ("imported-buffer", format!("{}B", resource.byte_size()), access)
                            }
                            GraphResourceImportInfo::Sampler { .. } => ("imported-sampler", String::new(), access),
                            GraphResourceImportInfo::RayTracingAcceleration { .. } => ("imported-as", String::new(), access),
                        }
                    }
                };
                ResourceDumpInfo {
                    id,
                    kind,
                    detail,
                    slot: transient.resource_slots.get(&id).copied(),
                    import_access,
                }
            })
            .collect();

        let edges: Vec<(usize, usize, &[ResourceBarrier])> = dep_graph
            .edge_references()
            .map(|e| (dep_graph[e.source()], dep_graph[e.target()], e.weight().barriers.as_slice()))
            .collect();

        let dump = GraphDump {
            frame: *self.core.absolute_frame_count.borrow() as u64,
            pass_names: pass_names.to_vec(),
            edges,
            resources,
            init_barriers,
            aliasing_report: format!("{transient:?}"),
        };
        dump.write_to(dir);
    }

    /// End states of every resource as collected by the last [`Self::compile`],
    /// keyed by resource id (cleared on [`Self::reset`]). For *imported*
    /// resources this is the state the resource is left in when the graph's
    /// submission completes — the caller can chain further GPU work with a
    /// plain pipeline barrier from `end_access` instead of waiting the device
    /// idle. Internal (created/transient) resources are reported too but die
    /// with the next `reset`. Temp impl, see [`ResourceEndState`].
    pub fn resource_end_states(&self) -> &HashMap<u32, ResourceEndState> {
        &self.resource_end_states
    }

    /// End state of one resource by handle, if the graph compiled it.
    pub fn end_state<R: Resource>(&self, handle: &Handle<R>) -> Option<&ResourceEndState> {
        self.resource_end_states.get(&handle.id)
    }

    /// Submit this frame's recorded command buffer to the graphics queue,
    /// signaling `graph_timeline` with the absolute frame count when it completes.
    ///
    /// The submission waits on `graph_timeline >= frame - 1`: the graph has no way
    /// to express cross-compile temporal dependencies (the accumulation / denoise /
    /// reservoir ping-pong buffers frame F reads were written by frame F-1), so it
    /// conservatively orders the whole graph after the previous frame's. This
    /// serializes GPU graph work but leaves the CPU free to record ahead — the
    /// overlap this buys is CPU-of-frame-F+1 against GPU-of-frame-F. Any binary
    /// `wait_semaphores` (async transfers) are added on top.
    ///
    /// Retires this frame's passes into the slot's bin afterwards: they own the
    /// AS-build scratch the GPU reads, so they must outlive the submission — the
    /// bin is cleared when the slot is reused N frames later (see `reset`).
    pub fn run(&mut self, wait_semaphores: &[vk::Semaphore], wait_stages: &[vk::PipelineStageFlags]) -> SrResult<()> {
        let slot = self.current_slot();
        let raw_cb = self.cmd_buffers[slot].inner();
        unsafe { self.core.device().inner().end_command_buffer(raw_cb)? };

        // Any binary transfer waits, converted to the timeline tuple form (value
        // ignored for binary semaphores).
        let extra_waits: Vec<(vk::Semaphore, u64, vk::PipelineStageFlags2)> = wait_semaphores
            .iter()
            .zip(wait_stages.iter())
            .map(|(sem, stage)| (*sem, 0, vk::PipelineStageFlags2::from_raw(stage.as_raw() as u64)))
            .collect();
        self.submit_current(&extra_waits, &[])
    }

    /// Blit-to-output variant of [`Self::run`]: append the blit tail to this
    /// frame's (still-open) command buffer, then end + submit it. Records, in order:
    ///   1. a barrier taking `source` from its graph end-access → TRANSFER_SRC and
    ///      the output `dst_image` from UNDEFINED → TRANSFER_DST,
    ///   2. `vkCmdBlitImage` source → dst (scales, nearest),
    ///   3. a barrier taking the dst image TRANSFER_DST → `dst_final`
    ///      (`Present` for a directly-presented swapchain image, `General` for an
    ///      offscreen readback target or a swapchain image an overlay finishes),
    ///      and `source` TRANSFER_SRC → back to its graph end-access.
    ///
    /// The output image never enters the graph as a resource — it is known only
    /// here, at run, as a borrowed non-owning [`Image`] (e.g. a swapchain image via
    /// [`Image::from_swapchain_image`]). `extra_signals` carries the binary present
    /// semaphore the caller's `queue_present` waits on (present path), if any.
    pub fn run_present(
        &mut self,
        source: &Handle<Image>,
        dst_image: &Image,
        dst_final: vk_sync::AccessType,
        extra_waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
        extra_signals: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
    ) -> SrResult<()> {
        let slot = self.current_slot();
        let raw_cb = self.cmd_buffers[slot].inner();
        let device = self.core.device().inner().clone();

        let src_end = self
            .end_state(source)
            .map(|e| e.end_access)
            .unwrap_or(vk_sync::AccessType::Nothing);
        let src_img = self.transient_resources[slot].image(source)?;
        let src_vk = src_img.inner();
        let src_fmt = src_img.format();
        let dst_vk = dst_image.inner();
        let dst_fmt = dst_image.format();

        let full_range = |fmt: vk::Format| vk::ImageSubresourceRange {
            aspect_mask: crate::render_graph::transient_resources::aspect_for(fmt),
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };
        // `previous_accesses` takes a slice, so `src_end` needs a local backing it.
        let src_prev = [src_end];

        // 1. source → TRANSFER_SRC, swapchain UNDEFINED → TRANSFER_DST (discard).
        let pre = [
            vk_sync::ImageBarrier {
                previous_accesses: &src_prev,
                next_accesses: &[vk_sync::AccessType::TransferRead],
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: false,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: src_vk,
                range: full_range(src_fmt),
            },
            vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::Nothing],
                next_accesses: &[vk_sync::AccessType::TransferWrite],
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: true,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: dst_vk,
                range: full_range(dst_fmt),
            },
        ];
        vk_sync::cmd::pipeline_barrier(&device, raw_cb, None, &[], &pre);

        // 2. the blit itself.
        Self::record_present_blit(&self.core, raw_cb, src_img, dst_image);

        // 3a. dst TRANSFER_DST → `dst_final` (PRESENT_SRC or GENERAL), and
        // 3b. source TRANSFER_SRC → back to its graph end-access (GENERAL storage).
        // The source is an *imported* storage image reused every frame; its heap
        // descriptor was written for GENERAL, so it must be handed back in GENERAL
        // or next frame's postprocess access hits a layout mismatch
        // (VUID-vkCmdDraw-None-09600). `src_prev` still holds `[src_end]`.
        let dst_final_arr = [dst_final];
        let post = [
            vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::TransferWrite],
                next_accesses: &dst_final_arr,
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: false,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: dst_vk,
                range: full_range(dst_fmt),
            },
            vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::TransferRead],
                next_accesses: &src_prev,
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: false,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: src_vk,
                range: full_range(src_fmt),
            },
        ];
        vk_sync::cmd::pipeline_barrier(&device, raw_cb, None, &[], &post);

        unsafe { device.end_command_buffer(raw_cb)? };
        self.submit_current(extra_waits, extra_signals)
    }

    /// End + submit this frame's recorded command buffer, waiting on
    /// `graph_timeline >= frame - 1` (the cross-compile temporal ping-pong order)
    /// plus `extra_waits`, and signaling `graph_timeline = frame` plus
    /// `extra_signals`. Retires this frame's passes / imports into the slot bin.
    fn submit_current(
        &mut self,
        extra_waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
        extra_signals: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
    ) -> SrResult<()> {
        let slot = self.current_slot();
        let frame = *self.core.absolute_frame_count.borrow() as u64;

        let mut waits: Vec<(vk::Semaphore, u64, vk::PipelineStageFlags2)> = vec![(
            self.graph_timeline.inner(),
            frame.saturating_sub(1),
            vk::PipelineStageFlags2::ALL_COMMANDS,
        )];
        waits.extend_from_slice(extra_waits);

        let mut signals = vec![(self.graph_timeline.inner(), frame, vk::PipelineStageFlags2::ALL_COMMANDS)];
        signals.extend_from_slice(extra_signals);

        self.core
            .graphics_queue()
            .submit_async_timelines(self.cmd_buffers[slot].inner(), &waits, &signals, vk::Fence::null())?;

        self.retired_passes[slot] = std::mem::take(&mut self.passes);
        // Park this frame's imported/created resources alongside its passes; the
        // `Arc`s they hold (the freshly-built TLAS in particular) must outlive the
        // submission the GPU is now running. Freed when this slot is reused (`reset`),
        // gated by `wait_for_slot_reuse`.
        self.retired_resources[slot] = std::mem::take(&mut self.virtual_resources);
        Ok(())
    }
}

/// The access an imported resource carries coming into a compile: the state the
/// previous frame's submission left it in, threaded back by the caller through the
/// import's `access_type`. Used to seed cross-frame init barriers (see `compile`).
/// Samplers and swapchain images carry no meaningful cross-frame access.
fn imported_initial_access(import: &GraphResourceImportInfo) -> vk_sync::AccessType {
    match import {
        GraphResourceImportInfo::Image { access_type, .. } => *access_type,
        GraphResourceImportInfo::Buffer { access_type, .. } => *access_type,
        GraphResourceImportInfo::RayTracingAcceleration { access_type, .. } => *access_type,
        GraphResourceImportInfo::Sampler { .. } => vk_sync::AccessType::Nothing,
    }
}

/// Overwrite the carried cross-frame access of an import (no-op for the variants
/// that don't track one). Used to thread a temporal backing's end-of-frame
/// access into next frame's compile — see the write-back loop in `compile`.
fn set_import_access(import: &mut GraphResourceImportInfo, access: vk_sync::AccessType) {
    match import {
        GraphResourceImportInfo::Image { access_type, .. } => *access_type = access,
        GraphResourceImportInfo::Buffer { access_type, .. } => *access_type = access,
        GraphResourceImportInfo::RayTracingAcceleration { access_type, .. } => *access_type = access,
        GraphResourceImportInfo::Sampler { .. } => {}
    }
}

/// The static name carried by an image/buffer resource desc (used for object
/// naming). `None` for descs that don't carry a name.
fn graph_desc_name(desc: &GraphResourceDesc) -> Option<&'static str> {
    match desc {
        GraphResourceDesc::Image(d) => Some(d.name),
        GraphResourceDesc::Buffer(d) => Some(d.name),
        GraphResourceDesc::Sampler(_) | GraphResourceDesc::RaytracingAS(_) => None,
    }
}

/// Attach a debug-utils name to whatever concrete vk handle an import wraps.
fn name_import(core: &Core, import: &GraphResourceImportInfo, name: &std::ffi::CStr) {
    match import {
        GraphResourceImportInfo::Image { resource, .. } => core.set_debug_object_name(resource.inner(), name),
        GraphResourceImportInfo::Buffer { resource, .. } => core.set_debug_object_name(resource.inner(), name),
        GraphResourceImportInfo::RayTracingAcceleration { resource, .. } => core.set_debug_object_name(resource.inner(), name),
        GraphResourceImportInfo::Sampler { .. } => {}
    }
}

/// The `PassCommonData::name` of any pass variant. Used for GPU-capture labels
/// and the graph dump.
fn pass_common_name(pass: &AnyRenderPass) -> String {
    match pass {
        AnyRenderPass::Rt(rt) => rt.common.name.clone(),
        AnyRenderPass::Raster(r) => r.common.name.clone(),
        AnyRenderPass::Compute(c) => c.common.name.clone(),
        AnyRenderPass::Transfer(t) => t.common.name.clone(),
    }
}

pub trait TypeEquals {
    type Other;
    fn same(value: Self) -> Self::Other;
}

impl<T: Sized> TypeEquals for T {
    type Other = Self;
    fn same(value: Self) -> Self::Other {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan_abstraction::buffer::BufferDesc;
    use crate::vulkan_abstraction::image::ImageDesc;
    use crate::vulkan_abstraction::image::sampler::SamplerDesc;
    use gpu_allocator::MemoryLocation;

    fn image(size: u32, name: &'static str) -> ImageDesc {
        ImageDesc {
            extent: vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            },
            format: vk::Format::R8G8B8A8_UNORM,
            tiling: vk::ImageTiling::OPTIMAL,
            location: MemoryLocation::GpuOnly,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            name,
        }
    }

    fn buffer(bytes: u64, name: &'static str) -> BufferDesc {
        BufferDesc {
            byte_size: bytes,
            alignment: 16,
            memory_location: MemoryLocation::GpuOnly,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            name,
        }
    }

    fn lifetime(first: usize, last: usize) -> ResourceLifetimeUsage {
        ResourceLifetimeUsage {
            first_pass: first,
            last_pass: last,
            usages: vec![],
        }
    }

    /// Exercises `TransientResources::populate` on a hand-built set of virtual
    /// resources whose lifetimes deliberately allow reuse. Prints the slot
    /// assignment + aliasing groups so the allocator decisions are inspectable
    /// with `cargo test transient_aliasing -- --nocapture`.
    ///
    /// Resource layout:
    ///   res 0: 256x256 image,   lifetime [0,1]  ┐ overlap → distinct slots
    ///   res 1: 512x512 image,   lifetime [0,1]  ┘
    ///   res 2: 128x128 image,   lifetime [2,3]  ┐ overlap, both 0/1 dead → reuse
    ///   res 3: 4096-byte buffer,lifetime [2,3]  ┘
    ///   res 4: 1024-byte buffer,lifetime [4,5]  → reuses earliest free slot
    ///   res 5: sampler                          (not aliased)
    #[test]
    fn transient_aliasing_debug() {
        let core = Rc::new(Core::new(false, false, vk::Format::R8G8B8A8_UNORM).expect("Core::new failed"));

        let virtual_resources = vec![
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(256, "img_256"))),
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(512, "img_512"))),
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(128, "img_128"))),
            GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(4096, "buf_4k"))),
            GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(1024, "buf_1k"))),
            GraphResourceInfo::Created(GraphResourceDesc::Sampler(SamplerDesc {
                min_filter: vk::Filter::LINEAR,
                mag_filter: vk::Filter::LINEAR,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                address_mode_w: vk::SamplerAddressMode::REPEAT,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            })),
        ];

        let mut usages: BTreeMap<u32, ResourceLifetimeUsage> = BTreeMap::new();
        usages.insert(0, lifetime(0, 1));
        usages.insert(1, lifetime(0, 1));
        usages.insert(2, lifetime(2, 3));
        usages.insert(3, lifetime(2, 3));
        usages.insert(4, lifetime(4, 5));
        // sampler: lifetime doesn't matter for slot assignment, but populate uses
        // `usages` only via `pending` keys, so we still record one.
        usages.insert(5, lifetime(0, 5));

        let components = vec![PassComponent {
            passes: (0..6).collect(),
            resources: vec![0, 1, 2, 3, 4, 5],
        }];

        let mut transient = TransientResources::default();
        transient
            .populate(Rc::clone(&core), &virtual_resources, &components, &usages)
            .expect("populate failed");

        println!("{transient:?}");

        // Sanity: overlapping-lifetime resources must NOT share a slot.
        assert_ne!(
            transient.resource_slots[&0], transient.resource_slots[&1],
            "res 0 and 1 overlap; must be in different slots"
        );
        assert_ne!(
            transient.resource_slots[&2], transient.resource_slots[&3],
            "res 2 and 3 overlap; must be in different slots"
        );
        // Total slot count must be strictly fewer than the number of aliasable
        // resources — otherwise no aliasing happened at all.
        assert!(
            transient.slot_allocations.len() < 5,
            "expected aliasing to reduce 5 resources to fewer slots; got {} slots",
            transient.slot_allocations.len()
        );
        // Sampler must have been materialized but not slot-aliased.
        assert_eq!(transient.transient_samplers.len(), 1);
        assert!(!transient.resource_slots.contains_key(&5));
    }

    /// End-to-end compile test: two compute passes with a producer→consumer
    /// dependency. Verifies that (a) the topological traversal invokes the
    /// passes in dependency order, (b) the render closure receives a non-null
    /// command buffer that's been begin-recorded, and (c) compile returns a
    /// `RenderGraph<Built>` carrying a real `CmdBuffer`.
    #[test]
    fn compile_runs_passes_in_topo_order() {
        use crate::render_graph::pass_builder::{ComputeRenderPassBuilder, PassCommonDataBuilder};
        use std::cell::RefCell;

        let core = Rc::new(Core::new(false, false, vk::Format::R8G8B8A8_UNORM).expect("Core::new failed"));
        let mut rg = RenderGraph::new(Rc::clone(&core)).expect("RenderGraph::new failed");
        // Simulate being on frame 1 so `run` signals a valid (>0) timeline value
        // and the slot index is well-defined.
        *core.absolute_frame_count.borrow_mut() += 1;
        let slot = rg.current_slot();

        let img_a = rg.create_resource(image(64, "img_a"));
        let img_b = rg.create_resource(image(64, "img_b"));

        // Shared trace: each render closure pushes its name.
        let trace: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));

        // Pass 0: write img_a.
        let mut common0 = PassCommonDataBuilder::new(&mut rg, "producer");
        common0
            .write(&img_a, vk_sync::AccessType::ComputeShaderWrite)
            .expect("producer write");
        {
            let trace = Rc::clone(&trace);
            common0.render(move |cb, _tr| {
                assert_ne!(*cb, vk::CommandBuffer::null(), "producer got null cmd buffer");
                trace.borrow_mut().push("producer");
                Ok(())
            });
        }
        let producer = ComputeRenderPassBuilder::default()
            .common(common0.build())
            .build()
            .expect("build producer pass");
        rg.add_render_pass(AnyRenderPass::Compute(producer));

        // Pass 1: read img_a, write img_b → depends on pass 0.
        let mut common1 = PassCommonDataBuilder::new(&mut rg, "consumer");
        common1
            .read(&img_a, vk_sync::AccessType::ComputeShaderReadOther)
            .expect("consumer read");
        common1
            .write(&img_b, vk_sync::AccessType::ComputeShaderWrite)
            .expect("consumer write");
        {
            let trace = Rc::clone(&trace);
            common1.render(move |cb, tr| {
                assert_ne!(*cb, vk::CommandBuffer::null(), "consumer got null cmd buffer");
                // img_a must be bound to a transient slot at this point.
                assert!(tr.resource_slots.contains_key(&0), "img_a not bound after populate");
                trace.borrow_mut().push("consumer");
                Ok(())
            });
        }
        let consumer = ComputeRenderPassBuilder::default()
            .common(common1.build())
            .build()
            .expect("build consumer pass");
        rg.add_render_pass(AnyRenderPass::Compute(consumer));

        rg.compile().expect("compile failed");

        // Print the transient state — the report now includes the barrier
        // trace recorded by compile.
        println!("{:?}", rg.transient_resources[slot]);

        // Both render closures must have fired, producer before consumer.
        {
            let trace = trace.borrow();
            assert_eq!(*trace, vec!["producer", "consumer"], "topo order violated");
        }
        // Persistent cmd buffer must be recorded.
        let recorded_cb = rg.cmd_buffers[slot].inner();
        assert_ne!(recorded_cb, vk::CommandBuffer::null());
        // At least one barrier must have been recorded (producer→consumer RAW on img_a).
        assert!(
            !rg.transient_resources[slot].recorded_barriers.is_empty(),
            "expected at least one recorded barrier between producer and consumer"
        );

        // Submit + wait. The graph stays usable for re-compile after run; the
        // submission's completion is tracked by the graph timeline, so wait the
        // queue idle here instead of a per-submit fence.
        rg.run(&[], &[]).expect("run failed");
        core.graphics_queue().wait_idle().expect("queue wait_idle failed");

        // The same primary command buffer persists; not reallocated by run().
        assert_eq!(
            rg.cmd_buffers[slot].inner(),
            recorded_cb,
            "cmd_buffer was reallocated across run()"
        );
    }
}
