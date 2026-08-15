use crate::MAX_FRAMES_IN_FLIGHT;
use crate::error::{ErrorSource, SrError, SrResult};
use crate::render_graph::alias::Placement;
use crate::render_graph::error::GraphError;
use crate::render_graph::pass_builder::{
    ComputeQueueAffinity, ComputeRenderPass, CopyEnd, InternalPass, PassCommonData, PassCommonDataBuilder, RasterRenderPass,
    RaytracingRenderPass, TransferPass, TransferPassBuilder, check_copy_bounds,
};
pub(crate) use crate::render_graph::resource::{
    GraphResourceDesc, GraphResourceImportInfo, GraphResourceInfo, Handle, Resource, ResourceDesc, ResourceRef, RgImportable,
};
use crate::render_graph::transient_resources::{TransientResources, image_layout_of};
use crate::vulkan_abstraction::{
    AccelerationStructure, AsBuildJob, Buffer, CmdBuffer, ComputePipeline, Core, GpuOnlyBuffer, GraphicsPipeline,
    GraphicsPipelineShaders, HeapComputePass, Image, Pipeline, QueueRole, RawBuffer, RayTracingPipeline,
    RayTracingPipelineShaders, ShaderBindingTable, TimelineSemaphore,
};
use ash::vk;
use petgraph::visit::EdgeRef;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use vk_sync_fork as vk_sync;
use vk_sync_fork::AccessType;

//TODO when I import previous usages, I should remove usages after FRAME_IN_FLIGHT so that I don't make a mask for reads that come from already executed frames

#[derive(Copy, Clone, Debug)]
//TODO this is basically unused or misused
pub enum PassResourceAccessSyncType {
    AlwaysSync,
    SkipSyncIfSameAccessType,
    NeverSync,
}

#[derive(Copy, Clone, Debug)]
pub struct PassResourceAccessType {
    pub(crate) access_type: vk_sync::AccessType,
    // Every construction site passes `AlwaysSync` and `plan_barriers` ignores it:
    // the knob is declared but not yet wired into the barrier planner (see the
    // TODO on the enum). Retained rather than deleted so wiring it up stays a
    // local change; delete both if the planner is never going to consult it.
    #[allow(dead_code)]
    pub(crate) sync_type: PassResourceAccessSyncType,
}

pub(crate) enum AnyRenderPass {
    Rt(RaytracingRenderPass),
    Raster(RasterRenderPass),
    Compute(ComputeRenderPass),
    Transfer(TransferPass),
    Internal(InternalPass),
}

impl AnyRenderPass {
    pub(super) fn common(&self) -> &PassCommonData {
        match self {
            AnyRenderPass::Rt(p) => &p.common,
            AnyRenderPass::Raster(p) => &p.common,
            AnyRenderPass::Compute(p) => &p.common,
            AnyRenderPass::Transfer(p) => &p.common,
            AnyRenderPass::Internal(p) => &p.common,
        }
    }

    pub(super) fn common_mut(&mut self) -> &mut PassCommonData {
        match self {
            AnyRenderPass::Rt(p) => &mut p.common,
            AnyRenderPass::Raster(p) => &mut p.common,
            AnyRenderPass::Compute(p) => &mut p.common,
            AnyRenderPass::Transfer(p) => &mut p.common,
            AnyRenderPass::Internal(p) => &mut p.common,
        }
    }

    /// Which queue this pass belongs on. Queue is a function of pass kind —
    /// raster and ray tracing need the universal queue, buffer copies belong on
    /// the DMA queue, and internal nodes stay universal (the present blit will
    /// live there, and presenting requires a present-capable queue). Compute is
    /// the only genuine choice, so it is the only kind carrying an affinity.
    ///
    /// TODO nothing consumes this yet: the graph still records every pass into
    /// one command buffer and submits it to the graphics queue
    /// (`submit_current`). It exists so multi-queue becomes a change to the
    /// scheduler rather than to every pass constructor.
    #[allow(dead_code)]
    pub(super) fn queue_role(&self) -> QueueRole {
        match self {
            AnyRenderPass::Rt(_) | AnyRenderPass::Raster(_) | AnyRenderPass::Internal(_) => QueueRole::Graphics,
            AnyRenderPass::Transfer(_) => QueueRole::Transfer,
            AnyRenderPass::Compute(p) => match p.queue_affinity {
                ComputeQueueAffinity::AsyncCompute => QueueRole::AsyncCompute,
                // `Inferred` resolves to the universal queue until the scheduler exists.
                ComputeQueueAffinity::Universal | ComputeQueueAffinity::Inferred => QueueRole::Graphics,
            },
        }
    }
}

/// Lightweight reference to a pipeline interned in the graph's [`PipelineCache`].
/// Render closures resolve it to the concrete pipeline at record time via
/// `TransientResources::{compute,raytracing,graphics}_pipeline`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineHandle(u32);

/// A heap-mode pipeline owned by the cache. RT additionally owns its shader
/// binding table (built alongside the pipeline).
pub(super) enum CachedPipeline {
    Compute(Arc<ComputePipeline<HeapComputePass>>),
    RayTracing(Arc<RayTracingPipeline>, Arc<ShaderBindingTable>),
    Graphics(Arc<GraphicsPipeline>),
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
    core: Option<Arc<Core>>,
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
        core: &Arc<Core>,
        build: impl FnOnce() -> SrResult<CachedPipeline>,
    ) -> SrResult<PipelineHandle> {
        if let Some(handle) = self.by_key.get(&key) {
            return Ok(*handle);
        }
        let entry = build()?;
        if self.core.is_none() {
            self.core = Some(Arc::clone(core));
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
    /// Every access of the resource's final epoch — the *whole* epoch, not just
    /// its last access.
    ///
    /// A resource ending in a run of several distinct reads is left in all of
    /// them. Recording only the last one makes the next frame's
    /// write-after-read barrier name a single source stage, so the write can
    /// begin while the other readers are still in flight. Empty means the
    /// resource was registered but never used.
    pub end_accesses: Vec<vk_sync::AccessType>,
    /// The write that produced the current contents, if any.
    ///
    /// When the final epoch is a read run, the cross-frame transition into
    /// another read emits no barrier at all, so nothing else names the producer.
    /// A queue change needs it for the ownership-transfer acquire's source
    /// access; it gains a queue role when multi-queue lands.
    pub last_write: Option<vk_sync::AccessType>,
    /// Graph-created (transient) resource: its backing memory is recycled on
    /// `reset`, so its end state only matters for in-graph aliasing, never to
    /// the caller.
    pub internal: bool,
}

/// The transition between two consecutive access epochs of one resource, to be
/// issued at the schedule position the epoch walk keyed it to.
///
/// `prev` / `next` carry *every* access of the epoch being left and entered, not
/// one apiece. That is what collapses N reader barriers into one: `vk_sync` ORs
/// the masks, so a single barrier covers the whole run. All accesses within one
/// side imply the same image layout by construction (see `plan_barriers`).
#[derive(Clone, Debug)]
pub(crate) struct ResourceBarrier {
    pub(crate) resource_id: u32,
    pub(crate) prev: Vec<vk_sync::AccessType>,
    pub(crate) next: Vec<vk_sync::AccessType>,
    /// Previous contents are undefined — freshly bound transient memory — so the
    /// transition may discard rather than preserve them.
    pub(crate) discard: bool,
}

/// Edge weight on the pass dependency graph: the resources whose hazards forced
/// this ordering.
///
/// Edges no longer carry barriers. A barrier belongs to a *position in the
/// schedule*, not to an edge — several edges can demand the same transition, and
/// one transition can cover readers spread over several edges. `plan_barriers`
/// computes them from the per-resource epoch walk instead; the resource ids are
/// kept only so the graph dump can label the edge.
#[derive(Clone, Debug, Default)]
pub(crate) struct PassDependency {
    pub(crate) resources: Vec<u32>,
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

/// One maximal run of mutually compatible accesses to a single resource: either a
/// single write, or a run of consecutive reads that all imply the same image
/// layout. Consecutive epochs are what barriers sit between.
#[derive(Debug)]
struct AccessEpoch {
    /// Every distinct access in the run. Unioned into one side of the barrier.
    accesses: Vec<vk_sync::AccessType>,
    /// Schedule position (not pass id) of the first and last pass in the run.
    first_pass: usize,
    last_pass: usize,
    is_write: bool,
    /// Contents entering this epoch are undefined. Only the synthetic seed epoch
    /// of a freshly-bound transient image sets it.
    discard: bool,
}

/// `Nothing` is neither a read nor a write access, but it means "contents are
/// undefined and a transition is required", so for epoch purposes it behaves as a
/// write — it must not merge into a neighbouring read run, and the first real use
/// must be ordered after it. `__imports` routes it to the write list for the same
/// reason (`PassCommonDataBuilder::declare_previous_imports`).
#[inline]
fn starts_write_epoch(access: vk_sync::AccessType) -> bool {
    access.is_write_access() || access == vk_sync::AccessType::Nothing
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

/// Everything the hazard scan derives from a pass list: per-resource lifetimes,
/// the pass dependency graph, and the weakly-connected components that aliasing is
/// computed within.
pub(crate) struct PassAnalysis {
    pub(crate) resource_usages: BTreeMap<u32, ResourceLifetimeUsage>,
    pub(crate) dep_graph: petgraph::graph::DiGraph<usize, PassDependency>,
    pub(crate) components: Vec<PassComponent>,
    /// Per pass, the passes that wrote the data it reads (read-after-write only).
    /// A strict subset of `dep_graph`'s incoming edges, which also carry WAR/WAW
    /// ordering — those point *dead reader → live writer*, so [`live_passes`]
    /// must not follow them or a dead pass resurrects itself.
    pub(crate) raw_producers: Vec<Vec<usize>>,
}

/// Single linear walk over the passes in declaration order, building lifetimes and
/// hazard edges, then the components those edges induce.
///
/// Split out of `compile` so it can be driven from plain `(read, write)` lists —
/// `bench_support::gen_graph` feeds it random pass declarations, which means the
/// random benchmarks exercise this scan rather than a reimplementation of it.
pub(crate) fn analyze_passes<'a>(passes: impl ExactSizeIterator<Item = (&'a [ResourceRef], &'a [ResourceRef])>) -> PassAnalysis {
    let pass_count = passes.len();
    //TODO possibile creazione con with size
    let mut resource_usages: BTreeMap<u32, ResourceLifetimeUsage> = BTreeMap::new();
    let mut hazard_states: HashMap<u32, ResourceHazardState> = HashMap::new();

    let mut dep_graph = petgraph::graph::DiGraph::<usize, PassDependency>::with_capacity(pass_count, pass_count * 2);
    let pass_nodes: Vec<petgraph::graph::NodeIndex> = (0..pass_count).map(|i| dep_graph.add_node(i)).collect();
    let mut raw_producers: Vec<Vec<usize>> = vec![Vec::new(); pass_count];

    for (pass_id, (read, write)) in passes.enumerate() {
        for read in read {
            let res_id = read.id;
            record_usage(&mut resource_usages, res_id, pass_id, read.access);
            let state = hazard_states.entry(res_id).or_default();
            if let Some((w_pass, _)) = state.last_writer {
                add_dep_edge(&mut dep_graph, &pass_nodes, w_pass, pass_id, res_id);
                // Same condition as the RAW edge above, minus the self-edge case
                // (a pass reading its own write depends on nothing external).
                if w_pass != pass_id && !raw_producers[pass_id].contains(&w_pass) {
                    raw_producers[pass_id].push(w_pass);
                }
            }

            state.readers_since_write.push((pass_id, read.access.access_type));
        }

        for write in write {
            let res_id = write.id;
            record_usage(&mut resource_usages, res_id, pass_id, write.access);
            let state = hazard_states.entry(res_id).or_default();
            if !state.readers_since_write.is_empty() {
                for (r_pass, _) in &state.readers_since_write {
                    add_dep_edge(&mut dep_graph, &pass_nodes, *r_pass, pass_id, res_id);
                }
            } else if let Some((w_pass, _)) = state.last_writer {
                add_dep_edge(&mut dep_graph, &pass_nodes, w_pass, pass_id, res_id);
            }
            state.last_writer = Some((pass_id, write.access.access_type));
            state.readers_since_write.clear();
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

    // `components_by_root` is a HashMap, so sort before handing the list on: bucket
    // ids in `alias::plan` are handed out in component order, and stable ids keep
    // graph dumps diffable across runs.
    let mut components: Vec<PassComponent> = components_by_root.into_values().collect();
    components.sort_unstable_by_key(|c| c.passes.first().copied().unwrap_or(usize::MAX));

    PassAnalysis {
        resource_usages,
        dep_graph,
        components,
        raw_producers,
    }
}

/// Backward reachability from the frame's declared results: `live[i]` is false for
/// every pass that cannot reach one and is therefore never recorded.
///
/// Roots are, in order of the checks below:
///   * the graph-synthesized prefix `0..internal_count` — `__imports` seeds every
///     import's incoming access, and `__prologue_copies` carries staging uploads
///     whose sources were already taken from the arena, so losing it loses the
///     upload for good (see `RenderGraph::build_internal_passes`);
///   * any pass writing a resource in `outputs` — `RenderGraph::mark_output`, plus
///     the writes of passes flagged by `PassCommonDataBuilder::mark_output`;
///   * any pass writing a temporal (history / ping-pong) backing. Those are read
///     by the *next* frame, which a single compile cannot see.
///
/// ponytail: temporal writes are rooted unconditionally rather than proven live
/// across frames — a cross-frame fixpoint would be the exact answer, and is worth
/// it only if history chains ever become optional.
pub(crate) fn live_passes(
    writes: impl ExactSizeIterator<Item = impl AsRef<[ResourceRef]>>,
    raw_producers: &[Vec<usize>],
    outputs: &HashSet<u32>,
    temporal: &HashSet<u32>,
    internal_count: usize,
) -> Vec<bool> {
    let mut live = vec![false; writes.len()];
    let mut worklist: Vec<usize> = Vec::new();

    for (pass_id, write) in writes.enumerate() {
        let is_root = pass_id < internal_count
            || write
                .as_ref()
                .iter()
                .any(|w| outputs.contains(&w.id) || temporal.contains(&w.id));
        if is_root {
            live[pass_id] = true;
            worklist.push(pass_id);
        }
    }

    while let Some(pass_id) = worklist.pop() {
        for &producer in &raw_producers[pass_id] {
            if !live[producer] {
                live[producer] = true;
                worklist.push(producer);
            }
        }
    }
    live
}

/// Is `[start, end)` entirely inside `covered`? [`cover_range`] keeps the list
/// sorted and merged, so a span crossing two entries is impossible — the only
/// entry that can contain `start` is the last one beginning at or before it, and
/// checking that single entry is exact.
fn range_covered(covered: &[(u64, u64)], start: u64, end: u64) -> bool {
    let after = covered.partition_point(|(s, _)| *s <= start);
    after > 0 && end <= covered[after - 1].1
}

/// Add `[start, end)` to `covered`, keeping it sorted and merged.
///
/// The list is already sorted and disjoint, so the entries this span touches are a
/// contiguous window: everything ending before `start` stays put, everything
/// beginning after `end` stays put, and the window between them collapses into one
/// entry. Both edges are binary searches, which is why this never re-sorts.
/// Touching counts as overlapping — `[0, 1024)` and `[1024, 2048)` merge — because
/// they leave no uncovered byte between them.
fn cover_range(covered: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    // Ends are ascending (the entries are disjoint), so the first entry reaching
    // `start` is a partition point, and likewise for the first start past `end`.
    let lo = covered.partition_point(|(_, e)| *e < start);
    let hi = covered.partition_point(|(s, _)| *s <= end);
    if lo == hi {
        covered.insert(lo, (start, end));
        return;
    }
    covered[lo] = (start.min(covered[lo].0), end.max(covered[hi - 1].1));
    covered.drain(lo + 1..hi);
}

fn add_dep_edge(
    graph: &mut petgraph::graph::DiGraph<usize, PassDependency>,
    nodes: &[petgraph::graph::NodeIndex],
    src: usize,
    dst: usize,
    res_id: u32,
) {
    // A pass that reads-then-writes its own resource produces a self-edge; the hazard
    // is already serialized by the pass itself, so skip it.
    if src == dst {
        return;
    }
    let s = nodes[src];
    let d = nodes[dst];
    if let Some(e) = graph.find_edge(s, d) {
        let w = graph.edge_weight_mut(e).expect("edge just found must have a weight");
        if !w.resources.contains(&res_id) {
            w.resources.push(res_id);
        }
    } else {
        graph.add_edge(s, d, PassDependency { resources: vec![res_id] });
    }
}

/// Topological order of the pass graph, as pass ids.
///
/// Kahn's algorithm with the ready set in a min-heap on pass id, so ties break
/// on declaration order instead of on petgraph's DFS. Determinism is load-bearing
/// here: `plan_barriers` walks each resource's usages in *this* linearization to
/// decide where barriers go, and the record loop replays the same order, so the
/// two must agree exactly and reproducibly.
///
/// The ready set is also the natural hook for multi-queue — assigning a pass to a
/// queue is a choice made at the moment it becomes ready.
pub(crate) fn kahn_toposort(dep_graph: &petgraph::graph::DiGraph<usize, PassDependency>) -> SrResult<Vec<usize>> {
    use petgraph::Direction;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = dep_graph.node_count();
    // `add_dep_edge` merges duplicates, so there are no parallel edges and each
    // neighbour is counted exactly once.
    let mut indegree: Vec<usize> = (0..n)
        .map(|i| {
            dep_graph
                .neighbors_directed(petgraph::graph::NodeIndex::new(i), Direction::Incoming)
                .count()
        })
        .collect();

    let mut ready: BinaryHeap<Reverse<usize>> = (0..n).filter(|i| indegree[*i] == 0).map(Reverse).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(Reverse(i)) = ready.pop() {
        let node = petgraph::graph::NodeIndex::new(i);
        order.push(dep_graph[node]);
        for succ in dep_graph.neighbors_directed(node, Direction::Outgoing) {
            indegree[succ.index()] -= 1;
            if indegree[succ.index()] == 0 {
                ready.push(Reverse(succ.index()));
            }
        }
    }

    // Fewer emitted than nodes ⇒ a cycle. Hazards only ever produce forward
    // (lower pass id → higher) edges by construction, so this is a logic bug.
    if order.len() != n {
        return Err(SrError::new_custom(
            "render graph dependency graph contains a cycle".to_string(),
        ));
    }
    Ok(order)
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
    /// Resources this frame declares as its results, via [`Self::mark_output`] or
    /// a pass marked with `PassCommonDataBuilder::mark_output`. Dead-pass culling
    /// keeps only what transitively feeds one of these (plus the temporal backings
    /// and the internal prefix — see [`live_passes`]). Cleared on `reset`.
    output_resources: HashSet<u32>,
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
    core: Arc<Core>,
}
//TODO per frame global data uploaded each frame like transforms and the camera, these can then live in the descriptor heap based on kajiya DYNAMIC_CONSTANTS_BUFFER
//TODO Reintroduce the typestate of the render graph,as it is intended to work like this, the setup phase is where you can add stuff and so on, when you want to run it you compile it once done than you can return to the setup phase, this should empty out reset the cmdbuffer and allow to add again resources, this should make sure the resources in use are
//   not overwritten though while still allowing new resources to be added,also while on a built state it should be able to handle n frames in flight with internal sync to minimize the wait idle time and allow multiple frame to be run concurrently, this
impl RenderGraph {
    pub fn new(core: Arc<Core>) -> SrResult<Self> {
        let cmd_buffers = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| CmdBuffer::new(Arc::clone(&core)))
            .collect::<SrResult<Vec<_>>>()?;
        let transient_resources = (0..MAX_FRAMES_IN_FLIGHT).map(|_| TransientResources::default()).collect();
        let retired_passes = (0..MAX_FRAMES_IN_FLIGHT).map(|_| Vec::new()).collect();
        let retired_resources = (0..MAX_FRAMES_IN_FLIGHT).map(|_| Vec::new()).collect();
        let graph_timeline = TimelineSemaphore::new(Arc::clone(&core), 0)?;
        Ok(RenderGraph {
            next_pass_id: 0,
            next_resource_id: 0,
            passes: vec![],
            virtual_resources: vec![],
            temporal_resources: vec![],
            transient_resources,
            resource_end_states: HashMap::new(),
            registered_temporal: Vec::new(),
            output_resources: HashSet::new(),
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
        self.core.absolute_frame_count() % MAX_FRAMES_IN_FLIGHT
    }

    /// Block until the frame that last used the upcoming frame's slot
    /// (`frame - MAX_FRAMES_IN_FLIGHT`) has finished its graph submission, so this
    /// slot's command buffer, transient pool and retired passes can be safely
    /// re-recorded / freed on the CPU. Call once at the very top of a frame,
    /// before touching any slot state (including the resource manager's arena slot
    /// reclamation). Non-blocking in steady state — that frame completed long ago.
    /// Must be called *before* `build_unified_graph` increments the frame count.
    pub fn wait_for_slot_reuse(&self) -> SrResult<()> {
        let upcoming = self.core.absolute_frame_count() as u64 + 1;
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
    /// frame-completion timeline
    pub fn wait_graph_timeline(&self, value: u64) -> SrResult<()> {
        self.graph_timeline.wait(value)
    }

    /// Hand the graph a batch of arena staging→GPU buffer copies to record as a
    /// transfer prologue pass. Each destination is an arena buffer imported into
    /// *this* build (see `ResourceManager::import_to_graph`).
    ///
    /// # Safety
    /// The sources are untracked by the graph: it declares nothing for them and
    /// emits no barriers on their behalf. Each must outlive this frame's graph
    /// submission, must not alias a graph-tracked resource, and is the caller's
    /// to synchronize against non-graph work.
    pub unsafe fn add_prologue_buffer_copies(
        &mut self,
        copies: Vec<(&impl Buffer, Handle<RawBuffer>, vk::BufferCopy)>,
    ) -> SrResult<()> {
        for (src, dst, region) in copies {
            check_copy_bounds("source", "arena staging", src.byte_size(), region.src_offset, &region)?;
            self.prologue_copies.push((src.inner(), dst, region));
        }
        Ok(())
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
        self.output_resources.clear();
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
    pub fn create_resource<Desc>(&mut self, desc: Desc) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: ResourceDesc + TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
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
    pub fn create_temporal_resource<Desc>(
        &mut self,
        desc: Desc,
    ) -> SrResult<ExportedTemporalResource<<Desc as ResourceDesc>::Resource>>
    where
        Desc: ResourceDesc + TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
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
            frame_of_creation: self.core.absolute_frame_count(),
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

    /// The storage-buffer heap slots of the persistent per-frame backing buffers
    /// of a temporal *buffer* resource. The graph import governs only
    /// synchronization; the shader reaches the buffer through the heap, so the
    /// caller bakes these slots into its push constants.
    ///
    /// The backings are stable for the lifetime of the temporal resource, and
    /// `storage_slot` caches on first call, so this does not churn the heap.
    pub fn temporal_buffer_storage_slots(&self, exported: &ExportedTemporalResource<RawBuffer>) -> [u32; MAX_FRAMES_IN_FLIGHT] {
        let imports = &self.temporal_resources[exported.index].imports;
        std::array::from_fn(|i| match &imports[i] {
            GraphResourceImportInfo::Buffer { resource, .. } => resource.storage_slot(),
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
                    access_types: vec![vk_sync::AccessType::Nothing],
                })
            }
            GraphResourceDesc::Buffer(buffer_desc) => {
                let buffer = Arc::new(RawBuffer::new_from_desc(self.core(), buffer_desc)?);
                Ok(GraphResourceImportInfo::Buffer {
                    resource: buffer,
                    access_types: vec![vk_sync::AccessType::Nothing],
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

    pub fn import<Desc>(
        &mut self,
        res: impl RgImportable<Desc> + Into<GraphResourceImportInfo>,
    ) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: ResourceDesc + TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
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
    pub fn import_with_usage<Desc>(
        &mut self,
        res: impl RgImportable<Desc> + Into<GraphResourceImportInfo>,
        usage: vk_sync::AccessType,
    ) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: ResourceDesc + TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        let desc = res.import();
        let mut import = res.into();
        set_import_access(&mut import, std::slice::from_ref(&usage));
        self.virtual_resources.push(GraphResourceInfo::Imported(import));
        Handle {
            id: self.next_resource_id(),
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }

    pub(crate) fn add_render_pass(&mut self, render_pass: impl Into<AnyRenderPass>) {
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
            Arc::clone(&self.core),
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
        // An AS build is compute work, not a buffer copy: it records
        // `cmd_build_acceleration_structures` and needs a COMPUTE-capable queue,
        // so it must not be a `TransferPass` (those are DMA-queue buffer copies,
        // and carry no render closure at all). `Inferred` because an AS build is
        // a genuine async-compute candidate once the scheduler can place it.

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

        self.add_render_pass(ComputeRenderPass {
            common: common.build(),
            shaders: None,
            queue_affinity: ComputeQueueAffinity::Inferred,
        });
        Ok(())
    }

    /// The graph's cached `Core`. Pass builders use this so callers no longer
    /// have to thread `Arc<Core>` into `generate_render` separately.
    pub(crate) fn core(&self) -> Arc<Core> {
        Arc::clone(&self.core)
    }

    /// Intern a heap-mode compute pipeline for `spirv`, returning a handle the
    /// render closure resolves at record time. Built once per distinct shader and
    /// reused across frame rebuilds (see [`PipelineCache`]).
    pub(crate) fn cache_compute_pipeline(&mut self, spirv: &[u8]) -> SrResult<PipelineHandle> {
        let key = pipeline_cache_key(0, &[spirv]);
        let core = Arc::clone(&self.core);
        let slot = self.current_slot();
        self.transient_resources[slot].pipeline_cache.intern(key, &core, || {
            let pipeline = ComputePipeline::<HeapComputePass>::new(core.clone_device(), spirv)?;
            Ok(CachedPipeline::Compute(Arc::new(pipeline)))
        })
    }

    /// Intern a heap-mode ray-tracing pipeline + its shader binding table.
    pub(crate) fn cache_raytracing_pipeline(&mut self, shaders: &RayTracingPipelineShaders) -> SrResult<PipelineHandle> {
        let key = pipeline_cache_key(1, &[&shaders.ray_gen, &shaders.miss, &shaders.closest_hit, &shaders.any_hit]);
        let core = Arc::clone(&self.core);
        let slot = self.current_slot();
        self.transient_resources[slot].pipeline_cache.intern(key, &core, || {
            let pipeline = Arc::new(RayTracingPipeline::new(Arc::clone(&core), shaders)?);
            let sbt = Arc::new(ShaderBindingTable::new(&core, &pipeline)?);
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
        let core = Arc::clone(&self.core);
        let slot = self.current_slot();
        self.transient_resources[slot].pipeline_cache.intern(key, &core, || {
            let pipeline = GraphicsPipeline::new(Arc::clone(&core), shaders)?;
            Ok(CachedPipeline::Graphics(Arc::new(pipeline)))
        })
    }

    /// Record a [`TransferPass`]'s copy list. `CopyEnd::Handle` endpoints are
    /// resolved through `tr`; `CopyEnd::Raw` endpoints are passed through as-is
    /// (their validity is the caller's `unsafe` obligation — see
    /// [`TransferPassBuilder`]). Barriers around these copies come from the
    /// declarations the builder made, not from here.
    fn record_buffer_copies(
        device: &ash::Device,
        cb: vk::CommandBuffer,
        copies: &[(CopyEnd, CopyEnd, vk::BufferCopy)],
        tr: &TransientResources,
    ) -> SrResult<()> {
        let resolve = |end: &CopyEnd| -> SrResult<vk::Buffer> {
            match end {
                CopyEnd::Raw(buffer) => Ok(*buffer),
                CopyEnd::Handle(id) => tr.buffer_by_id(*id).ok_or_else(|| {
                    SrError::new(
                        ErrorSource::RenderGraph(GraphError::InvalidResourceRef),
                        format!("transfer pass copies to/from unknown buffer resource id {id}"),
                    )
                }),
            }
        };
        for (src, dst, region) in copies {
            let (src, dst) = (resolve(src)?, resolve(dst)?);
            unsafe { device.cmd_copy_buffer(cb, src, dst, std::slice::from_ref(region)) };
        }
        Ok(())
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

    /// Build the nodes the graph synthesizes for itself and prepend them to
    /// `self.passes`, returning how many were prepended. Called once at the top of
    /// [`Self::compile`], before the hazard scan. The returned count is the live
    /// prefix dead-pass culling never touches — see [`live_passes`].
    ///
    /// Front placement is load-bearing: the hazard scan is a linear walk in
    /// insertion order that only ever adds edges from an already-seen pass, so
    /// an internal node placed at the back would be ordered *after* its
    /// consumers by WAR edges rather than before them.
    fn build_internal_passes(&mut self) -> SrResult<usize> {
        let mut internal: Vec<AnyRenderPass> = Vec::new();

        // `__imports`: declare the access every imported resource carries into
        // this frame — the state the previous frame's submission left it in,
        // threaded back by the caller (or by the temporal write-back below).
        //
        // As ordinary declarations these feed the hazard scan directly, so the
        // first consumer of each import is ordered against the previous frame
        // with no special case in `compile`. A declared *write* (or `Nothing`)
        // sets `last_writer`, so the first consumer gets a RAW/WAW barrier; a
        // declared *read* followed by another read produces no barrier at all,
        // which is correct and which the loop this replaced could not express —
        // it always emitted one.
        // Samplers are excluded: they have no memory contents, so they never need
        // a barrier — declaring one would only produce a global barrier for an id
        // that resolves to neither an image nor a buffer.
        // Each import contributes *every* access it carries in, so a resource the
        // previous frame left in a multi-reader epoch re-enters in all of them and
        // the first write this frame is ordered against all of them.
        let imported: Vec<(u32, Vec<AccessType>)> = self
            .virtual_resources
            .iter()
            .enumerate()
            .filter_map(|(id, info)| match info {
                GraphResourceInfo::Imported(GraphResourceImportInfo::Sampler { .. }) => None,
                GraphResourceInfo::Imported(import) => Some((id as u32, imported_initial_access(import).to_vec())),
                GraphResourceInfo::Created(_) => None,
            })
            .collect();

        if !imported.is_empty() {
            let mut builder = PassCommonDataBuilder::new(self, "__imports");
            for (id, accesses) in imported {
                for access in accesses {
                    builder.declare_previous_imports(id, access);
                }
            }
            internal.push(AnyRenderPass::Internal(builder.build_internal()));
        }

        // Arena staging copies: raw `vk::Buffer` sources (owned by the staging
        // arena, not the graph) into imported arena buffers.
        if !self.prologue_copies.is_empty() {
            let copies = std::mem::take(&mut self.prologue_copies);
            let mut builder = TransferPassBuilder::new(self, "__prologue_copies");
            for (src, dst, region) in copies {
                // SAFETY: upheld by the caller of `add_prologue_buffer_copies`,
                // which is itself `unsafe` for exactly this reason
                unsafe { builder.copy_from_prechecked_raw(src, &dst, region)? };
            }
            internal.push(AnyRenderPass::Transfer(builder.build()));
        }

        let internal_count = internal.len();
        if internal_count > 0 {
            internal.append(&mut self.passes);
            self.passes = internal;
        }
        Ok(internal_count)
    }

    /// Drop every pass that cannot reach a declared result, in place.
    ///
    /// Runs between `build_internal_passes` and the analysis whose output actually
    /// drives the frame, so the surviving passes stay densely indexed — every later
    /// stage (`schedule_pos`, the barrier map, the alias lifetimes, the record
    /// loop) keys off a pass's position in `self.passes`.
    ///
    /// Dropping is safe here: a culled pass was never recorded into a command
    /// buffer, so nothing in flight refers to it or to the scratch it owns.
    fn cull_dead_passes(&mut self, raw_producers: &[Vec<usize>], internal_count: usize) {
        // A pass marked as an output contributes its whole write list, so both tag
        // surfaces collapse into one set of output resources before rooting.
        let mut outputs = self.output_resources.clone();
        for pass in self.passes.iter().filter(|p| p.common().output) {
            outputs.extend(pass.common().write.iter().map(|w| w.id));
        }
        let temporal: HashSet<u32> = self.registered_temporal.iter().map(|(_, _, rid)| *rid).collect();

        // No declared result means nothing to cull against — an app that never calls
        // `mark_output` keeps today's behaviour (everything runs) instead of
        // compiling an empty frame.
        if outputs.is_empty() && temporal.is_empty() {
            log::error!(
                "render graph: There exists no output nodes, dead pass culling skipped. If not the graph would have been empty."
            );
            return;
        }

        for out in &outputs {
            if !self.passes.iter().any(|p| p.common().write.iter().any(|w| w.id == *out)) {
                log::warn!("render graph: resource {out} is marked as an output but no pass writes it");
            }
        }

        let live = live_passes(
            self.passes.iter().map(|p| p.common().write.as_slice()),
            raw_producers,
            &outputs,
            &temporal,
            internal_count,
        );
        let culled = live.iter().filter(|l| !**l).count();
        if culled == 0 {
            return;
        }

        log::info!(
            "render graph: culled {culled}/{} passes (kept {}): {}",
            live.len(),
            live.len() - culled,
            self.passes
                .iter()
                .zip(&live)
                .filter(|(_, keep)| !**keep)
                .map(|(p, _)| p.common().name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let mut keep = live.iter();
        self.passes.retain(|_| *keep.next().expect("one flag per pass"));
    }

    pub fn compile(&mut self) -> SrResult<()> {
        //TODO there are some complex optimizations as shown here https://www.youtube.com/watch?v=v9LaTFLhP38 and this is the site where it will be published the paper https://dl.acm.org/profile/99661091135
        //TODO respect PassResourceAccessSyncType (NeverSync / SkipSyncIfSameAccessType) when deciding whether to emit a barrier
        //TODO it currently returns a one time submit, but the cmd buffer can be reuse as long as the graph doesn't get rebuilt this requires the temporal stuff though and some rework on the sync side between each frame

        let internal_count = self.build_internal_passes()?;

        //From now on the graph passes should not be touched

        let slot = self.current_slot();

        let analyze = |passes: &[AnyRenderPass]| {
            analyze_passes(passes.iter().map(|pass| {
                let common = pass.common();
                (common.read.as_slice(), common.write.as_slice())
            }))
        };
        let mut analysis = analyze(&self.passes);
        self.cull_dead_passes(&analysis.raw_producers, internal_count);
        if self.passes.len() != analysis.raw_producers.len() {
            // Lifetimes, components and the dep graph are all keyed by position in
            // `self.passes`, so they have to be recomputed against the compacted list
            // rather than patched up.
            analysis = analyze(&self.passes);
        }
        let pass_count = self.passes.len();

        let PassAnalysis {
            resource_usages,
            dep_graph,
            components,
            raw_producers: _,
        } = analysis;

        self.transient_resources[slot].populate(
            Arc::clone(&self.core),
            &self.virtual_resources,
            &components,
            &resource_usages,
        )?;

        // Linearize: the epoch walk groups each resource's usages in *schedule*
        // order, and the record loop below replays the same order.
        let topo = kahn_toposort(&dep_graph)?;
        let mut schedule_pos = vec![0usize; pass_count];
        for (pos, pass_id) in topo.iter().enumerate() {
            schedule_pos[*pass_id] = pos;
        }

        // Epoch walk: every barrier this frame issues, keyed to the pass it must
        // precede, plus each resource's end state. Runs after `populate` because it
        // needs the memory-slot assignment — two transient resources sharing a slot
        // need a barrier between them (see `plan_barriers`).
        let (mut barriers_at, end_states) = Self::plan_barriers(
            &self.virtual_resources,
            &resource_usages,
            &schedule_pos,
            &self.transient_resources[slot].placements,
        );
        self.resource_end_states = end_states;

        // Cross-frame sync for temporal (ping-pong / history) resources: thread
        // each backing's end access this frame back into its stored import, so
        // *next* frame's compile emits the read→write (or write→read) barrier for
        // the same physical backing across the frame boundary. Without this the
        // imports always re-enter as `Nothing` and the hazard graph — which only
        // orders passes *within* one compile — never synchronizes the ping-pong
        // reuse, leaving frame N's reservoir/accumulation write unordered against
        // frame N+1's read of the same memory. The TLAS gets this treatment
        // explicitly via `Tlas::queue_build`; temporal resources get it here.
        //
        // Must run *after* `plan_barriers`: `reset` clears `resource_end_states`
        // every frame, so reading it before the epoch walk repopulates it always
        // found an empty map and the write-back silently never happened.
        for &(ti, ci, rid) in &self.registered_temporal {
            if let Some(end) = self.resource_end_states.get(&rid) {
                set_import_access(&mut self.temporal_resources[ti].imports[ci], &end.end_accesses);
            }
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

        // Pass names, gathered up front (the loop borrows `self.passes` mutably):
        // used both for GPU-capture labels and the optional graph dump below.
        let pass_names: Vec<String> = self.passes.iter().map(|p| p.common().name.clone()).collect();
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
        for &pass_id in &topo {
            if let Some(barriers) = barriers_at.remove(&pass_id) {
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

            match &mut self.passes[pass_id] {
                // Transfer passes are declarative — the graph records their
                // copies itself so they stay re-targetable to the DMA queue.
                AnyRenderPass::Transfer(transfer) => {
                    Self::record_buffer_copies(&device, raw_cb, &transfer.copies, &self.transient_resources[slot])?;
                }
                // Declaration-only: `__imports` exists to feed the hazard scan,
                // it records nothing.
                AnyRenderPass::Internal(_) => {}
                pass => {
                    if let Some(render) = pass.common_mut().render.as_mut() {
                        let mut cb_handle = raw_cb;
                        render(&mut cb_handle, &self.transient_resources[slot])?;
                    }
                }
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
        if let Some(dir) = crate::utils::graph_dump_dir() {
            self.dump_graph(&dir, &pass_names, &dep_graph, slot);
        }

        Ok(())
    }
    /// Group every resource's usages into access epochs and derive, from the
    /// transitions between them, the minimum set of barriers plus each resource's
    /// end state. This replaces emitting one barrier per hazard.
    ///
    /// Returns barriers keyed by the pass they must be issued *before* — a
    /// position in the schedule, never a graph edge. Two different resources whose
    /// transitions land on the same pass are merged into one
    /// `vkCmdPipelineBarrier2` by `TransientResources::emit_barriers`.
    ///
    /// `schedule_pos` maps pass id → index in the topological order. Usages are
    /// recorded in pass-id order, which is *not* necessarily schedule order, so
    /// they are re-sorted before grouping.
    ///
    /// Placement is the last legal point (immediately before the first pass of the
    /// epoch being entered). Moving a barrier earlier could merge more of them —
    /// see the interval-stabbing note in the plan — but it would add an ordering
    /// constraint that was not otherwise implied, so it is deliberately not done.
    /// `placements` maps a transient resource to the memory it was bound into;
    /// resources whose byte ranges overlap alias the same memory and need a barrier
    /// between them, which is why this runs after `populate`.
    pub(crate) fn plan_barriers(
        virtual_resources: &[GraphResourceInfo],
        resource_usages: &BTreeMap<u32, ResourceLifetimeUsage>,
        schedule_pos: &[usize],
        placements: &HashMap<u32, Placement>,
    ) -> (HashMap<usize, Vec<ResourceBarrier>>, HashMap<u32, ResourceEndState>) {
        let mut barriers_at: HashMap<usize, Vec<ResourceBarrier>> = HashMap::new();
        let mut end_states: HashMap<u32, ResourceEndState> = HashMap::new();

        // ── Phase 1: real epochs, per resource ──────────────────────────────
        let mut epochs_by_res: BTreeMap<u32, Vec<AccessEpoch>> = BTreeMap::new();

        for (res_id, info) in virtual_resources.iter().enumerate() {
            let res_id = res_id as u32;
            let is_image = matches!(
                info,
                GraphResourceInfo::Created(GraphResourceDesc::Image(_))
                    | GraphResourceInfo::Imported(GraphResourceImportInfo::Image { .. })
            );
            let mut epochs: Vec<AccessEpoch> = Vec::new();

            if let Some(usage) = resource_usages.get(&res_id) {
                // Stable sort by schedule position keeps a pass's reads ahead of its
                // writes (the hazard scan records them in that order).
                let mut ordered: Vec<(usize, vk_sync::AccessType)> =
                    usage.usages.iter().map(|(p, a)| (*p, a.access_type)).collect();
                ordered.sort_by_key(|(pass, _)| schedule_pos[*pass]);

                for (pass, access) in ordered {
                    if starts_write_epoch(access) {
                        epochs.push(AccessEpoch {
                            accesses: vec![access],
                            first_pass: pass,
                            last_pass: pass,
                            is_write: true,
                            discard: false,
                        });
                        continue;
                    }
                    // A read extends the current read run, but only while every
                    // access in the run implies the same image layout — `vk_sync`
                    // asserts on a barrier whose accesses disagree, and merging
                    // across a layout change is meaningless anyway.
                    let extends = epochs
                        .last()
                        .is_some_and(|e| !e.is_write && (!is_image || image_layout_of(e.accesses[0]) == image_layout_of(access)));
                    if extends {
                        let e = epochs.last_mut().expect("checked non-empty");
                        if !e.accesses.contains(&access) {
                            e.accesses.push(access);
                        }
                        e.last_pass = pass;
                    } else {
                        epochs.push(AccessEpoch {
                            accesses: vec![access],
                            first_pass: pass,
                            last_pass: pass,
                            is_write: false,
                            discard: false,
                        });
                    }
                }
            }
            epochs_by_res.insert(res_id, epochs);
        }

        // ── Phase 2: seed each created resource ─────────────────────────────
        // A created resource's memory is bound this frame, so whatever it contains
        // belongs to someone else. Two cases:
        //
        //   * nothing held those bytes earlier — the memory is freshly allocated, so
        //     only an image needs a transition (out of UNDEFINED) and a buffer needs
        //     nothing;
        //   * it *reuses* bytes — every earlier occupant's last access must complete
        //     and be made available before this one's first access, or the two
        //     overlap in the same memory. `Nothing` alone cannot express that: its
        //     stage mask is empty, so the barrier carries no execution dependency at
        //     all and the new resource's write can start while the old one is still
        //     being read.
        //
        // Aliased memory always discards: the previous occupant's layout says
        // nothing about this one, so the transition must come out of UNDEFINED.
        let alias_predecessors = Self::alias_predecessors(placements, resource_usages);

        for (res_id, info) in virtual_resources.iter().enumerate() {
            let res_id = res_id as u32;
            let is_created_image = matches!(info, GraphResourceInfo::Created(GraphResourceDesc::Image(_)));
            if !matches!(info, GraphResourceInfo::Created(_)) {
                // Imports need no seed — `__imports` declared their incoming access
                // as a real usage.
                continue;
            }

            let seed = match alias_predecessors.get(&res_id) {
                Some(prev_resources) => {
                    // Union of every earlier occupant of overlapping bytes. Under
                    // offset packing a resource can land on top of several smaller
                    // ones side by side, so a single predecessor is not enough.
                    //
                    // TODO: unioned without pruning. A predecessor whose bytes
                    // are fully covered by a later-ending one contributes a
                    // redundant source access — wider than necessary, never
                    // narrower. Prune by coverage if barriers get fat.
                    let mut prev_end: Vec<vk_sync::AccessType> = Vec::new();
                    for prev_res in prev_resources {
                        let Some(last) = epochs_by_res.get(prev_res).and_then(|e| e.last()) else {
                            continue;
                        };
                        for access in &last.accesses {
                            if !prev_end.contains(access) {
                                prev_end.push(*access);
                            }
                        }
                    }
                    if prev_end.is_empty() {
                        continue;
                    }
                    Some(prev_end)
                }
                None if is_created_image => Some(vec![vk_sync::AccessType::Nothing]),
                None => None,
            };

            if let Some(accesses) = seed {
                let epochs = epochs_by_res.get_mut(&res_id).expect("every resource has an entry");
                epochs.insert(
                    0,
                    AccessEpoch {
                        accesses,
                        // Sentinel: the seed precedes every real pass, so it can
                        // never collide with one in the same-pass check below, and
                        // the end-state scan can tell it apart from a real epoch.
                        first_pass: usize::MAX,
                        last_pass: usize::MAX,
                        is_write: true,
                        discard: true,
                    },
                );
            }
        }

        // ── Phase 3: emit transitions and end states ────────────────────────
        for (res_id, epochs) in &epochs_by_res {
            let res_id = *res_id;
            let internal = matches!(virtual_resources.get(res_id as usize), Some(GraphResourceInfo::Created(_)));

            for i in 1..epochs.len() {
                let (prev, next) = (&epochs[i - 1], &epochs[i]);
                // Both epochs wholly inside one pass: a pass that reads then writes its
                // own resource serializes that itself, exactly as the self-edge skip in
                // `add_dep_edge` assumes.
                //
                // `prev` must be *confined* to that pass, not merely end there. A read
                // run spanning several passes and ending at `next.first_pass` still has
                // earlier readers, and nothing serializes those against the write —
                // `add_dep_edge` records the ordering edge for them (the self-edge it
                // skips is only the pass against itself), but a schedule edge is not an
                // execution dependency. Skipping on `last_pass` alone drops that WAR.
                //
                // The barrier this emits sits *before* the writing pass, so its source
                // mask also names the accesses of that pass's own read. Harmless: masks
                // name stages, not commands, and the pass's read is recorded after the
                // barrier, so the pass still serializes itself.
                //
                //TODO a pass that reads and writes the same *image* at two different
                // layouts is unrepresentable either way — the transition would have to
                // land mid-pass, which the graph cannot express, and whichever layout the
                // barrier picks the other access is wrong. Reject it in the pass builder
                // rather than emitting something quietly incorrect.
                if prev.first_pass == prev.last_pass && prev.last_pass == next.first_pass {
                    continue;
                }
                barriers_at.entry(next.first_pass).or_default().push(ResourceBarrier {
                    resource_id: res_id,
                    prev: prev.accesses.clone(),
                    next: next.accesses.clone(),
                    discard: prev.discard,
                });
            }

            // End state comes from the last *real* epoch, carrying all of its
            // accesses — a final read run leaves the resource in every one of them.
            // A resource holding only the synthetic seed was never used.
            let is_real = |e: &AccessEpoch| e.last_pass != usize::MAX;
            let (last_use_pass, end_accesses) = match epochs.iter().rev().find(|e| is_real(e)) {
                Some(e) => (Some(e.last_pass), e.accesses.clone()),
                None => (None, Vec::new()),
            };
            let last_write = epochs
                .iter()
                .rev()
                .find(|e| e.is_write && is_real(e))
                .and_then(|e| e.accesses.last().copied());
            end_states.insert(
                res_id,
                ResourceEndState {
                    last_use_pass,
                    end_accesses,
                    last_write,
                    internal,
                },
            );
        }

        (barriers_at, end_states)
    }

    /// For every transient resource, every *earlier* resource that occupied bytes
    /// it now overlaps.
    ///
    /// "Earlier" is strict: `prev.last_pass < r.first_pass`. Two resources whose
    /// lifetimes overlap must never have been given overlapping bytes in the first
    /// place — [`alias::plan`](crate::render_graph::alias::plan) guarantees that,
    /// and the random invariant suite is what checks it — so a byte overlap here
    /// always means a sequential reuse that needs ordering.
    ///
    /// This is a set rather than a single predecessor because
    /// [`AliasStrategy::Bucket`](crate::render_graph::alias::AliasStrategy) packs at
    /// offsets: one resource can land on top of several smaller ones lying side by
    /// side, and it must wait for all of them.
    ///
    /// The set is pruned to the ones that actually matter. Candidates are walked
    /// latest-ending first and a candidate is dropped once the bytes it shares with
    /// `r` are already covered by ones kept so far. That is safe, not merely
    /// cheaper: a kept `k` covering dropped `p`'s bytes overlaps `p` in memory, so
    /// their lifetimes were disjoint, and `k` ends later than `p`, so `k` started
    /// after `p` finished — meaning `k`'s own alias barrier already waited on `p`.
    /// Waiting on `k` waits on `p` transitively.
    ///
    /// Under `Slot` every member sits at offset 0, so the largest late-ending
    /// occupant usually covers everything and this collapses to a single
    /// predecessor — the behaviour before offset packing existed.
    ///
    /// Two things keep the walk off its O(occupants²) worst case. Only strictly
    /// earlier occupants can be predecessors, and the list is sorted by descending
    /// `last_pass`, so they are a contiguous suffix — found by binary search instead
    /// of scanned from the front. And once `covered` spans the whole of `[lo, hi)`
    /// every remaining candidate would be pruned, so the scan stops there. Both are
    /// worth far more than the interval set `covered` is built on
    /// ([`cover_range`] / [`range_covered`]): those only lower the per-candidate
    /// constant, these remove candidates outright.
    ///
    /// ponytail: still O(occupants²) in the worst case. Offset packing deliberately
    /// makes buckets fat — total work is Σ(occupants²), dominated by the single
    /// largest bucket — so this is the first thing to feel a very large graph. Real
    /// frames put tens of resources in a bucket and land in the microseconds. Reach
    /// for an interval tree over the bucket's byte ranges only if graphs ever get
    /// big enough to care; see `docs/aliasing_benchmarks.md` for the scaling.
    fn alias_predecessors(
        placements: &HashMap<u32, Placement>,
        resource_usages: &BTreeMap<u32, ResourceLifetimeUsage>,
    ) -> HashMap<u32, Vec<u32>> {
        let mut by_bucket: HashMap<u32, Vec<u32>> = HashMap::new();
        for (res_id, placement) in placements {
            by_bucket.entry(placement.bucket).or_default().push(*res_id);
        }

        // The sort key's leading component, reused by the binary search below so the
        // two can never disagree about the order.
        let last_pass_of = |r: &u32| resource_usages.get(r).map_or(0, |u| u.last_pass);

        let mut predecessors: HashMap<u32, Vec<u32>> = HashMap::new();
        for occupants in by_bucket.values_mut() {
            if occupants.len() < 2 {
                continue;
            }
            // Latest-ending first; the id tiebreak keeps the walk deterministic.
            // The greedy prune below depends on this order.
            occupants.sort_unstable_by_key(|r| (std::cmp::Reverse(last_pass_of(r)), *r));

            let mut covered: Vec<(u64, u64)> = Vec::new();
            for res_id in occupants.iter() {
                let Some(first_pass) = resource_usages.get(res_id).map(|u| u.first_pass) else {
                    continue;
                };
                let mine = placements[res_id];
                let (lo, hi) = (mine.offset, mine.offset + mine.size);

                // Only strictly-earlier occupants can be predecessors, and the list is
                // sorted by descending `last_pass`, so they are exactly the suffix
                // starting here — the prefix would fail the filter on every element.
                // `res_id` itself is in the prefix (its own `last_pass >= first_pass`),
                // which is why the walk below needs no self-check.
                let earlier = occupants.partition_point(|other| last_pass_of(other) >= first_pass);

                covered.clear();
                let mut prev: Vec<u32> = Vec::new();
                for other in occupants[earlier..].iter() {
                    // The suffix is earlier by construction; this rejects only the
                    // occupants the usage map never saw (they sort to the very end).
                    if !resource_usages.get(other).is_some_and(|u| u.last_pass < first_pass) {
                        continue;
                    }
                    let theirs = placements[other];
                    let (start, end) = (lo.max(theirs.offset), hi.min(theirs.offset + theirs.size));
                    if start >= end || range_covered(&covered, start, end) {
                        continue;
                    }
                    cover_range(&mut covered, start, end);
                    prev.push(*other);
                    // Every clipped span lies inside `[lo, hi)`, so a merged list that
                    // is exactly that one interval means the range is fully covered and
                    // every remaining candidate would be pruned. Nothing left to find.
                    if covered.len() == 1 && covered[0] == (lo, hi) {
                        break;
                    }
                }
                if !prev.is_empty() {
                    prev.sort_unstable();
                    predecessors.insert(*res_id, prev);
                }
            }
        }
        predecessors
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
        dir: &std::path::Path,
        pass_names: &[String],
        dep_graph: &petgraph::graph::DiGraph<usize, PassDependency>,
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
                        let access = Some(imported_initial_access(import).to_vec());
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
                    placement: transient.placements.get(&id).copied(),
                    import_access,
                }
            })
            .collect();

        let edges: Vec<(usize, usize, &[u32])> = dep_graph
            .edge_references()
            .map(|e| (dep_graph[e.source()], dep_graph[e.target()], e.weight().resources.as_slice()))
            .collect();

        let dump = GraphDump {
            frame: self.core.absolute_frame_count() as u64,
            pass_names: pass_names.to_vec(),
            pass_uses: self
                .passes
                .iter()
                .map(|p| (p.common().read.as_slice(), p.common().write.as_slice()))
                .collect(),
            edges,
            resources,
            // The record loop already collected every barrier it issued, in
            // schedule order, so the dump reuses that rather than a second copy.
            barriers_at: &transient.recorded_barriers,
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

    /// Declare `handle` a result of this frame: dead-pass culling keeps whatever
    /// writes it, and everything that transitively feeds those writes. Anything
    /// that reaches no marked output (and no temporal backing) is dropped by
    /// [`Self::compile`] before the hazard scan — see [`live_passes`].
    ///
    /// Must be called before `compile`; `run_present`'s source is the canonical
    /// one. Cleared by `reset`, so mark again on every rebuild.
    pub fn mark_output<R: Resource>(&mut self, handle: &Handle<R>) {
        self.output_resources.insert(handle.id);
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

        // The whole final epoch: if the graph left `source` in several reads, all
        // of them have to be named as the source of the transition to TRANSFER_READ.
        let src_end: Vec<vk_sync::AccessType> = self
            .end_state(source)
            .map(|e| e.end_accesses.clone())
            .filter(|a| !a.is_empty())
            .unwrap_or_else(|| vec![vk_sync::AccessType::Nothing]);
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
        let src_prev = src_end.as_slice();

        // 1. source → TRANSFER_SRC, swapchain UNDEFINED → TRANSFER_DST (discard).
        let pre = [
            vk_sync::ImageBarrier {
                previous_accesses: src_prev,
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
                next_accesses: src_prev,
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
        let frame = self.core.absolute_frame_count() as u64;

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
fn imported_initial_access(import: &GraphResourceImportInfo) -> &[vk_sync::AccessType] {
    match import {
        GraphResourceImportInfo::Image { access_types, .. } => access_types,
        GraphResourceImportInfo::Buffer { access_types, .. } => access_types,
        GraphResourceImportInfo::RayTracingAcceleration { access_types, .. } => access_types,
        GraphResourceImportInfo::Sampler { .. } => &[],
    }
}

/// Overwrite the carried cross-frame access of an import (no-op for the variants
/// that don't track one). Used to thread a temporal backing's end-of-frame
/// access into next frame's compile — see the write-back loop in `compile`.
fn set_import_access(import: &mut GraphResourceImportInfo, accesses: &[vk_sync::AccessType]) {
    match import {
        GraphResourceImportInfo::Image { access_types, .. }
        | GraphResourceImportInfo::Buffer { access_types, .. }
        | GraphResourceImportInfo::RayTracingAcceleration { access_types, .. } => {
            access_types.clear();
            access_types.extend_from_slice(accesses);
        }
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

    // ── Epoch-walk tests ────────────────────────────────────────────────────
    // `plan_barriers` is a free function over plain data, so these need no
    // Vulkan device — descs are inert structs.

    /// Build a `ResourceLifetimeUsage` from `(pass, access)` pairs.
    fn usages(list: &[(usize, AccessType)]) -> ResourceLifetimeUsage {
        ResourceLifetimeUsage {
            first_pass: list.first().map_or(0, |(p, _)| *p),
            last_pass: list.last().map_or(0, |(p, _)| *p),
            usages: list
                .iter()
                .map(|(p, a)| {
                    (
                        *p,
                        PassResourceAccessType {
                            access_type: *a,
                            sync_type: PassResourceAccessSyncType::AlwaysSync,
                        },
                    )
                })
                .collect(),
        }
    }

    /// Identity schedule: pass id == schedule position.
    fn identity_schedule(n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    fn placed(bucket: u32, offset: u64, size: u64) -> Placement {
        Placement { bucket, offset, size }
    }

    /// A write followed by a run of three distinct reads followed by a write
    /// collapses to **two** barriers, not four: one RAW carrying the union of the
    /// three read accesses, one WAR carrying that same union as its source.
    ///
    /// The WAR source union is the correctness half — dropping readers from it is
    /// the race where a write starts while earlier readers are still in flight.
    #[test]
    fn epoch_walk_merges_a_reader_run() {
        let resources = vec![GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(1024, "buf")))];
        let mut u = BTreeMap::new();
        u.insert(
            0,
            usages(&[
                (1, AccessType::ComputeShaderWrite),
                (2, AccessType::ComputeShaderReadOther),
                (3, AccessType::RayTracingShaderReadOther),
                (4, AccessType::AnyShaderReadOther),
                (5, AccessType::ComputeShaderWrite),
            ]),
        );

        let (barriers, ends) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(6), &HashMap::new());

        let total: usize = barriers.values().map(|v| v.len()).sum();
        assert_eq!(total, 2, "W,R,R,R,W must produce 2 barriers, got {barriers:#?}");

        // RAW: before the *first* reader, carrying every reader's access.
        let raw = &barriers[&2][0];
        assert_eq!(raw.prev, vec![AccessType::ComputeShaderWrite]);
        assert_eq!(
            raw.next,
            vec![
                AccessType::ComputeShaderReadOther,
                AccessType::RayTracingShaderReadOther,
                AccessType::AnyShaderReadOther
            ],
            "RAW dst must cover the whole read run"
        );

        // WAR: before the writer, sourced from every reader.
        let war = &barriers[&5][0];
        assert_eq!(war.prev.len(), 3, "WAR src must cover all 3 readers, not just the last");
        assert_eq!(war.next, vec![AccessType::ComputeShaderWrite]);

        assert_eq!(ends[&0].last_use_pass, Some(5));
    }

    /// A pass that reads *and* writes a resource serializes itself, so the epoch
    /// pair it straddles needs no barrier — but only its own read. When the read run
    /// it closes started at an earlier pass, that earlier reader is still ordered
    /// against the write by nothing but the schedule, which is not an execution
    /// dependency. The skip must therefore key on the read run being confined to one
    /// pass, not on it merely ending there.
    #[test]
    fn war_survives_a_read_run_ending_at_the_writing_pass() {
        let resources = vec![GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(1024, "buf")))];
        let mut u = BTreeMap::new();
        // Pass 3 reads and writes; pass 2 read the same run.
        u.insert(
            0,
            usages(&[
                (1, AccessType::ComputeShaderWrite),
                (2, AccessType::ComputeShaderReadOther),
                (3, AccessType::ComputeShaderReadOther),
                (3, AccessType::ComputeShaderWrite),
            ]),
        );

        let (barriers, _) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(4), &HashMap::new());

        let war = barriers
            .get(&3)
            .and_then(|b| b.iter().find(|b| b.next == vec![AccessType::ComputeShaderWrite]))
            .expect("pass 2's read must be ordered against pass 3's write");
        assert_eq!(war.prev, vec![AccessType::ComputeShaderReadOther]);
    }

    /// The other half of the same rule: when the epoch being left *is* confined to
    /// the writing pass, the pass serializes it and no barrier is emitted. Without
    /// this the fix above would put a barrier between a pass and itself.
    #[test]
    fn self_contained_read_then_write_stays_barrier_free() {
        let resources = vec![GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(1024, "buf")))];
        let mut u = BTreeMap::new();
        u.insert(
            0,
            usages(&[
                (1, AccessType::ComputeShaderWrite),
                (3, AccessType::ComputeShaderReadOther),
                (3, AccessType::ComputeShaderWrite),
            ]),
        );

        let (barriers, _) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(4), &HashMap::new());

        let total: usize = barriers.values().map(|v| v.len()).sum();
        assert_eq!(total, 1, "only the RAW into pass 3's read belongs here, got {barriers:#?}");
        assert_eq!(barriers[&3][0].next, vec![AccessType::ComputeShaderReadOther]);
    }

    /// A read run only merges while the reads agree on image layout. Sampled and
    /// storage reads do not, so the run splits and a third barrier appears —
    /// without the split, `vk_sync` would debug-assert on conflicting layouts.
    #[test]
    fn read_run_splits_on_layout_change() {
        let resources = vec![GraphResourceInfo::Created(GraphResourceDesc::Image(image(64, "img")))];
        let mut u = BTreeMap::new();
        u.insert(
            0,
            usages(&[
                (1, AccessType::ComputeShaderWrite),
                (2, AccessType::ComputeShaderReadOther), // GENERAL
                (3, AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer), // SHADER_READ_ONLY
            ]),
        );

        let (barriers, _) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(4), &HashMap::new());

        // seed->write, write->read(GENERAL), read(GENERAL)->read(SHADER_READ_ONLY)
        let total: usize = barriers.values().map(|v| v.len()).sum();
        assert_eq!(total, 3, "layout-incompatible reads must not share an epoch: {barriers:#?}");
        assert_eq!(barriers[&3][0].prev, vec![AccessType::ComputeShaderReadOther]);
        assert_eq!(
            barriers[&3][0].next,
            vec![AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer]
        );
    }

    /// A created image is seeded UNDEFINED, so its first use gets a discarding
    /// transition placed before that use — not batched up front, and not emitted
    /// for imported resources (whose incoming access `__imports` declares).
    #[test]
    fn created_image_is_seeded_with_a_discard() {
        let resources = vec![GraphResourceInfo::Created(GraphResourceDesc::Image(image(64, "img")))];
        let mut u = BTreeMap::new();
        u.insert(0, usages(&[(3, AccessType::ComputeShaderWrite)]));

        let (barriers, _) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(4), &HashMap::new());

        assert_eq!(barriers.len(), 1);
        let b = &barriers[&3][0];
        assert!(b.discard, "freshly bound transient memory must discard");
        assert_eq!(b.prev, vec![AccessType::Nothing]);
    }

    /// Two resources whose transitions land on the same pass are keyed to the same
    /// position, which is the precondition for `emit_barriers` folding them into a
    /// single `vkCmdPipelineBarrier2`. This is the reduction that lowers the call
    /// count, so it is checked separately from the epoch merge above.
    #[test]
    fn transitions_on_the_same_pass_share_a_barrier_point() {
        let resources = vec![
            GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(1024, "a"))),
            GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(2048, "b"))),
        ];
        let mut u = BTreeMap::new();
        // Both written at 1 and 2 respectively, both read at 5.
        u.insert(
            0,
            usages(&[(1, AccessType::ComputeShaderWrite), (5, AccessType::ComputeShaderReadOther)]),
        );
        u.insert(
            1,
            usages(&[(2, AccessType::ComputeShaderWrite), (5, AccessType::ComputeShaderReadOther)]),
        );

        let (barriers, _) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(6), &HashMap::new());

        assert_eq!(barriers.len(), 1, "both transitions belong at pass 5");
        assert_eq!(barriers[&5].len(), 2, "one per resource, merged at emission");
    }

    /// A resource left in a run of three distinct reads must export all three, not
    /// just the last. Exporting one makes next frame's write-after-read barrier
    /// name a single source stage, so the write can start while the other two
    /// readers are still running — a real race, and the reason `end_accesses` is a
    /// set. `last_write` still names the producer, which a read-run epoch's
    /// barrier-free cross-frame transition would otherwise lose.
    #[test]
    fn end_state_exports_the_whole_final_read_run() {
        let resources = vec![GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(512, "buf")))];
        let mut u = BTreeMap::new();
        u.insert(
            0,
            usages(&[
                (1, AccessType::ComputeShaderWrite),
                (2, AccessType::ComputeShaderReadOther),
                (3, AccessType::RayTracingShaderReadOther),
                (4, AccessType::AnyShaderReadOther),
            ]),
        );

        let (_, ends) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(5), &HashMap::new());
        let end = &ends[&0];

        assert_eq!(end.end_accesses.len(), 3, "all three trailing readers must be exported");
        assert!(end.end_accesses.contains(&AccessType::ComputeShaderReadOther));
        assert!(end.end_accesses.contains(&AccessType::RayTracingShaderReadOther));
        assert!(end.end_accesses.contains(&AccessType::AnyShaderReadOther));
        assert_eq!(end.last_write, Some(AccessType::ComputeShaderWrite));
        assert_eq!(end.last_use_pass, Some(4));
    }

    /// Two transient resources sharing a memory slot alias the same bytes, so the
    /// second occupant's first access must be ordered against the first occupant's
    /// last one. Seeding it with `Nothing` is not enough: `Nothing` has an empty
    /// stage mask, so the barrier carries no execution dependency and the new
    /// resource's write can begin while the old one is still being read.
    ///
    /// The seed must therefore name the previous occupant's final accesses, and
    /// still discard — the old layout says nothing about the new resource.
    #[test]
    fn aliased_slot_reuse_is_ordered_against_the_previous_occupant() {
        let resources = vec![
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(64, "first"))),
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(64, "second"))),
        ];
        let mut u = BTreeMap::new();
        // res 0 lives over passes 0..1, res 1 over 2..3 — disjoint, so `populate`
        // hands them the same slot.
        u.insert(
            0,
            usages(&[(0, AccessType::ComputeShaderWrite), (1, AccessType::ComputeShaderReadOther)]),
        );
        u.insert(1, usages(&[(2, AccessType::ComputeShaderWrite)]));

        let mut slots = HashMap::new();
        slots.insert(0u32, placed(0, 0, 4096));
        slots.insert(1u32, placed(0, 0, 4096));

        let (barriers, _) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(4), &slots);

        let seed = barriers[&2]
            .iter()
            .find(|b| b.resource_id == 1)
            .expect("second occupant must get a barrier before its first use");
        assert_eq!(
            seed.prev,
            vec![AccessType::ComputeShaderReadOther],
            "the alias barrier must be sourced from the previous occupant's final access, not from Nothing"
        );
        assert!(seed.discard, "aliased memory carries no meaningful old layout");

        // Without a shared slot the same graph seeds from `Nothing` instead.
        let (plain, _) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(4), &HashMap::new());
        let unaliased = plain[&2].iter().find(|b| b.resource_id == 1).expect("still needs a seed");
        assert_eq!(unaliased.prev, vec![AccessType::Nothing]);
    }

    /// Sharing a *bucket* is not sharing *memory*. Under offset packing two
    /// resources sit in one allocation at disjoint byte ranges, and ordering them
    /// against each other would be a dependency the graph never implied.
    #[test]
    fn same_bucket_disjoint_bytes_needs_no_alias_barrier() {
        let resources = vec![
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(64, "low"))),
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(64, "high"))),
        ];
        let mut u = BTreeMap::new();
        u.insert(
            0,
            usages(&[(0, AccessType::ComputeShaderWrite), (1, AccessType::ComputeShaderReadOther)]),
        );
        u.insert(1, usages(&[(2, AccessType::ComputeShaderWrite)]));

        // Same bucket, adjacent-but-disjoint ranges: 0..1024 and 1024..2048.
        let mut slots = HashMap::new();
        slots.insert(0u32, placed(0, 0, 1024));
        slots.insert(1u32, placed(0, 1024, 1024));

        let (barriers, _) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(4), &slots);
        let seed = barriers[&2].iter().find(|b| b.resource_id == 1).expect("still needs a seed");
        assert_eq!(
            seed.prev,
            vec![AccessType::Nothing],
            "res 1 owns bytes res 0 never touched, so it seeds fresh, not from res 0"
        );
    }

    /// One resource landing on top of two smaller ones lying side by side must be
    /// ordered against *both*. This is the case the old "sort the slot's occupants
    /// and pair consecutive ones" rule could not express: it would have picked one
    /// predecessor and left the other unsynchronized.
    #[test]
    fn a_resource_covering_two_predecessors_waits_for_both() {
        let resources = vec![
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(64, "low"))),
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(64, "high"))),
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(64, "covering"))),
        ];
        let mut u = BTreeMap::new();
        // Two small resources live concurrently over passes 0..1, then a big one
        // covering both their ranges starts at pass 2.
        u.insert(
            0,
            usages(&[(0, AccessType::ComputeShaderWrite), (1, AccessType::TransferRead)]),
        );
        u.insert(
            1,
            usages(&[(0, AccessType::ComputeShaderWrite), (1, AccessType::ComputeShaderReadOther)]),
        );
        u.insert(2, usages(&[(2, AccessType::ComputeShaderWrite)]));

        let mut slots = HashMap::new();
        slots.insert(0u32, placed(0, 0, 1024));
        slots.insert(1u32, placed(0, 1024, 1024));
        slots.insert(2u32, placed(0, 0, 2048));

        let (barriers, _) = RenderGraph::plan_barriers(&resources, &u, &identity_schedule(4), &slots);
        let seed = barriers[&2]
            .iter()
            .find(|b| b.resource_id == 2)
            .expect("covering resource needs a seed");
        assert!(
            seed.prev.contains(&AccessType::TransferRead) && seed.prev.contains(&AccessType::ComputeShaderReadOther),
            "must wait on both occupants it overwrites, got {:?}",
            seed.prev
        );
        assert!(seed.discard);
    }

    /// The end-to-end tie between the two halves: whatever `alias::plan` decides,
    /// every sequential reuse of overlapping bytes it produces ends up ordered.
    /// Checked over seeded random graphs under both strategies — this is what says
    /// a placement change can't silently introduce an unsynchronized alias.
    ///
    /// Ordering may be transitive. `alias_predecessors` drops a predecessor whose
    /// bytes a later-ending one already covers, because that later one's own alias
    /// barrier already waited on the dropped one. So the property to check is
    /// *reachability* in the predecessor graph, not a direct edge — and separately,
    /// that every resource with predecessors actually receives a discarding
    /// barrier, since a chain of edges is worth nothing if no barrier is emitted.
    /// `cover_range` / `range_covered` are a sorted-disjoint interval set that
    /// `alias_predecessors` leans on twice: to prune predecessors, and to decide it
    /// can stop early. Both readers assume the list is sorted, merged and
    /// non-touching, and both now binary-search it, so a violated invariant would
    /// silently drop a predecessor rather than fail loudly.
    ///
    /// Checked against a naive byte-set model over a small universe: same covered
    /// bytes, same `range_covered` answer for every sub-range, invariant intact
    /// after every insertion.
    #[test]
    fn interval_set_matches_a_naive_byte_model() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};

        const N: u64 = 24;
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut covered: Vec<(u64, u64)> = Vec::new();
            let mut model = [false; N as usize];

            for _ in 0..12 {
                let start = rng.random_range(0..N);
                let end = rng.random_range(start + 1..=N);
                cover_range(&mut covered, start, end);
                for b in &mut model[start as usize..end as usize] {
                    *b = true;
                }

                // Sorted, non-empty, and separated by at least one uncovered byte —
                // touching entries must have been merged, not left adjacent.
                for w in covered.windows(2) {
                    assert!(w[0].1 < w[1].0, "seed {seed}: entries {:?} not merged/sorted", covered);
                }
                assert!(covered.iter().all(|(s, e)| s < e), "seed {seed}: empty entry in {covered:?}");

                // Same bytes as the model.
                for (byte, want) in model.iter().enumerate() {
                    let got = covered.iter().any(|(s, e)| *s <= byte as u64 && (byte as u64) < *e);
                    assert_eq!(got, *want, "seed {seed}: byte {byte} in {covered:?} vs model");
                }

                // `range_covered` agrees with the model on every sub-range.
                for a in 0..N {
                    for b in a + 1..=N {
                        let want = model[a as usize..b as usize].iter().all(|c| *c);
                        assert_eq!(
                            range_covered(&covered, a, b),
                            want,
                            "seed {seed}: range_covered({a},{b}) on {covered:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn random_graphs_order_every_byte_reuse() {
        use crate::render_graph::alias::{self, AliasStrategy};
        use crate::render_graph::bench_support::{GenParams, gen_graph};

        let params = GenParams::default();
        for seed in 0..16u64 {
            let fixture = gen_graph(seed, &params);
            let (res, components) = fixture.alias_input();

            for strategy in [AliasStrategy::Slot, AliasStrategy::Bucket] {
                let (placements, _) = alias::plan(strategy, res, components, 64);
                let usages = &fixture.analysis.resource_usages;
                let predecessors = RenderGraph::alias_predecessors(&placements, usages);
                let (barriers, _) =
                    RenderGraph::plan_barriers(&fixture.virtual_resources, usages, &fixture.schedule_pos, &placements);

                // Every resource that reuses someone's bytes must be entered
                // through a discarding barrier carrying a real source access.
                for res_id in predecessors.keys() {
                    let seed_barrier = barriers
                        .values()
                        .flatten()
                        .find(|b| b.resource_id == *res_id && b.discard)
                        .unwrap_or_else(|| panic!("seed {seed} {strategy:?}: res {res_id} aliases with no discard barrier"));
                    assert!(
                        !seed_barrier.prev.is_empty() && !seed_barrier.prev.contains(&AccessType::Nothing),
                        "seed {seed} {strategy:?}: res {res_id} enters aliased memory on {:?}, which carries \
                         no execution dependency",
                        seed_barrier.prev
                    );
                }

                for a in res {
                    for b in res {
                        // `b` vacated these bytes before `a` claimed them.
                        if a.id == b.id || b.last_pass >= a.first_pass || !placements[&a.id].overlaps(&placements[&b.id]) {
                            continue;
                        }
                        assert!(
                            waits_on(&predecessors, a.id, b.id),
                            "seed {seed} {strategy:?}: res {} reuses res {}'s bytes but is never ordered after it",
                            a.id,
                            b.id
                        );
                    }
                }
            }
        }
    }

    /// Is `target` reachable from `from` in the alias-predecessor graph? Edges point
    /// backwards in time, so this terminates.
    fn waits_on(predecessors: &HashMap<u32, Vec<u32>>, from: u32, target: u32) -> bool {
        let mut stack = vec![from];
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        while let Some(node) = stack.pop() {
            for prev in predecessors.get(&node).into_iter().flatten() {
                if *prev == target {
                    return true;
                }
                if seen.insert(*prev) {
                    stack.push(*prev);
                }
            }
        }
        false
    }

    /// The fold decision in `emit_barriers` is "does the layout actually change".
    /// Storage-image accesses all resolve to GENERAL and therefore fold into the
    /// global barrier; a sampled read does not.
    #[test]
    fn layout_query_drives_the_fold() {
        use crate::render_graph::transient_resources::image_layout_of;
        assert_eq!(
            image_layout_of(AccessType::General),
            image_layout_of(AccessType::ComputeShaderReadOther),
            "General->ComputeShaderReadOther changes no layout, so it must fold"
        );
        assert_ne!(
            image_layout_of(AccessType::General),
            image_layout_of(AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer),
            "a sampled read needs a real image barrier"
        );
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
    #[ignore = "needs an RT-capable GPU (Core::new); run with --include-ignored"]
    fn transient_aliasing_debug() {
        let core = Arc::new(Core::new(false, false, vk::Format::R8G8B8A8_UNORM).expect("Core::new failed"));

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
            .populate(Arc::clone(&core), &virtual_resources, &components, &usages)
            .expect("populate failed");

        println!("{transient:?}");

        // Sanity: overlapping-lifetime resources must NOT share bytes.
        assert!(
            !transient.placements[&0].overlaps(&transient.placements[&1]),
            "res 0 and 1 overlap in time; they must not overlap in memory"
        );
        assert!(
            !transient.placements[&2].overlaps(&transient.placements[&3]),
            "res 2 and 3 overlap in time; they must not overlap in memory"
        );
        // Total bucket count must be strictly fewer than the number of aliasable
        // resources — otherwise no aliasing happened at all.
        assert!(
            transient.slot_allocations.len() < 5,
            "expected aliasing to reduce 5 resources to fewer buckets; got {} buckets",
            transient.slot_allocations.len()
        );
        // Sampler must have been materialized but not aliased.
        assert_eq!(transient.transient_samplers.len(), 1);
        assert!(!transient.placements.contains_key(&5));
    }

    /// End-to-end compile test: two compute passes with a producer→consumer
    /// dependency. Verifies that (a) the topological traversal invokes the
    /// passes in dependency order, (b) the render closure receives a non-null
    /// command buffer that's been begin-recorded, and (c) compile returns a
    /// `RenderGraph<Built>` carrying a real `CmdBuffer`.
    #[test]
    #[ignore = "needs an RT-capable GPU (Core::new); run with --include-ignored"]
    fn compile_runs_passes_in_topo_order() {
        use crate::render_graph::pass_builder::{ComputeRenderPassBuilder, PassCommonDataBuilder};
        use parking_lot::Mutex;

        let core = Arc::new(Core::new(false, false, vk::Format::R8G8B8A8_UNORM).expect("Core::new failed"));
        let mut rg = RenderGraph::new(Arc::clone(&core)).expect("RenderGraph::new failed");
        // Simulate being on frame 1 so `run` signals a valid (>0) timeline value
        // and the slot index is well-defined.
        core.advance_frame();
        let slot = rg.current_slot();

        let img_a = rg.create_resource(image(64, "img_a"));
        let img_b = rg.create_resource(image(64, "img_b"));

        // Shared trace: each render closure pushes its name.
        let trace: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

        // Pass 0: write img_a.
        let mut common0 = PassCommonDataBuilder::new(&mut rg, "producer");
        common0
            .write(&img_a, vk_sync::AccessType::ComputeShaderWrite)
            .expect("producer write");
        {
            let trace = Arc::clone(&trace);
            common0.render(move |cb, _tr| {
                assert_ne!(*cb, vk::CommandBuffer::null(), "producer got null cmd buffer");
                trace.lock().push("producer");
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
            let trace = Arc::clone(&trace);
            common1.render(move |cb, tr| {
                assert_ne!(*cb, vk::CommandBuffer::null(), "consumer got null cmd buffer");
                // img_a must be bound to a transient slot at this point.
                assert!(tr.placements.contains_key(&0), "img_a not bound after populate");
                trace.lock().push("consumer");
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
            let trace = trace.lock();
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

    // ── Dead-pass culling ───────────────────────────────────────────────────
    // Driven from raw (read, write) declarations, the same shape `bench_support`
    // feeds `analyze_passes` — no device needed.

    fn refs(ids: &[u32], write: bool) -> Vec<ResourceRef> {
        let access_type = if write {
            AccessType::ComputeShaderWrite
        } else {
            AccessType::ComputeShaderReadOther
        };
        ids.iter()
            .map(|id| ResourceRef {
                id: *id,
                access: PassResourceAccessType {
                    access_type,
                    sync_type: PassResourceAccessSyncType::AlwaysSync,
                },
            })
            .collect()
    }

    /// `(reads, writes)` per pass → the liveness mask, with no internal prefix and
    /// no temporal resources unless the test says otherwise.
    fn cull(
        decls: &[(Vec<ResourceRef>, Vec<ResourceRef>)],
        outputs: &[u32],
        temporal: &[u32],
        internal_count: usize,
    ) -> Vec<bool> {
        let analysis = analyze_passes(decls.iter().map(|(r, w)| (r.as_slice(), w.as_slice())));
        live_passes(
            decls.iter().map(|(_, w)| w.as_slice()),
            &analysis.raw_producers,
            &outputs.iter().copied().collect(),
            &temporal.iter().copied().collect(),
            internal_count,
        )
    }

    #[test]
    fn cull_keeps_the_chain_feeding_an_output() {
        // A: -> 0 | B: 0 -> 1 | C: 1 -> 2 (the output)
        let decls = vec![
            (vec![], refs(&[0], true)),
            (refs(&[0], false), refs(&[1], true)),
            (refs(&[1], false), refs(&[2], true)),
        ];
        assert_eq!(cull(&decls, &[2], &[], 0), vec![true, true, true]);
    }

    #[test]
    fn cull_drops_a_pass_no_output_reaches() {
        // A: -> 0 | B: 0 -> 1 (output) | D: 0 -> 9, read by nobody.
        let decls = vec![
            (vec![], refs(&[0], true)),
            (refs(&[0], false), refs(&[1], true)),
            (refs(&[0], false), refs(&[9], true)),
        ];
        assert_eq!(cull(&decls, &[1], &[], 0), vec![true, true, false]);
        // Marking the dead pass's own output instead keeps it and drops B.
        assert_eq!(cull(&decls, &[9], &[], 0), vec![true, false, true]);
    }

    #[test]
    fn cull_does_not_follow_write_after_read_edges() {
        // E reads r0 and is dead; F overwrites r0 afterwards and is the output.
        // The dep graph has E -> F (WAR), which must not resurrect E.
        let decls = vec![
            (vec![], refs(&[0], true)),
            (refs(&[0], false), refs(&[9], true)),
            (vec![], refs(&[0], true)),
        ];
        assert_eq!(cull(&decls, &[0], &[], 0), vec![true, false, true]);
    }

    #[test]
    fn cull_roots_temporal_writes_and_the_internal_prefix() {
        // Pass 0 is the internal prefix; pass 1 writes only a temporal backing —
        // nothing reads it this frame, but next frame will.
        let decls = vec![
            (vec![], refs(&[0], true)),
            (refs(&[0], false), refs(&[7], true)),
            (vec![], refs(&[9], true)),
        ];
        assert_eq!(cull(&decls, &[], &[7], 1), vec![true, true, false]);
    }

    /// End-to-end: a pass whose output nothing reads and nothing marks is never
    /// recorded, and the transient image it alone wrote is never allocated.
    #[test]
    #[ignore = "needs an RT-capable GPU (Core::new); run with --include-ignored"]
    fn compile_culls_a_pass_that_reaches_no_output() {
        use crate::render_graph::pass_builder::{ComputeRenderPassBuilder, PassCommonDataBuilder};
        use parking_lot::Mutex;

        let core = Arc::new(Core::new(false, false, vk::Format::R8G8B8A8_UNORM).expect("Core::new failed"));
        let mut rg = RenderGraph::new(Arc::clone(&core)).expect("RenderGraph::new failed");
        core.advance_frame();
        let slot = rg.current_slot();

        let img_a = rg.create_resource(image(64, "img_a"));
        let img_dead = rg.create_resource(image(64, "img_dead"));

        let fired: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

        // Live: writes the frame's declared result.
        let mut live = PassCommonDataBuilder::new(&mut rg, "live");
        live.write(&img_a, vk_sync::AccessType::ComputeShaderWrite)
            .expect("live write");
        {
            let fired = Arc::clone(&fired);
            live.render(move |_cb, _tr| {
                fired.lock().push("live");
                Ok(())
            });
        }
        rg.add_render_pass(AnyRenderPass::Compute(
            ComputeRenderPassBuilder::default()
                .common(live.build())
                .build()
                .expect("build live pass"),
        ));

        // Dead: reads the live output, writes an image nobody ever reads.
        let mut dead = PassCommonDataBuilder::new(&mut rg, "dead");
        dead.read(&img_a, vk_sync::AccessType::ComputeShaderReadOther)
            .expect("dead read");
        dead.write(&img_dead, vk_sync::AccessType::ComputeShaderWrite)
            .expect("dead write");
        {
            let fired = Arc::clone(&fired);
            dead.render(move |_cb, _tr| {
                fired.lock().push("dead");
                Ok(())
            });
        }
        rg.add_render_pass(AnyRenderPass::Compute(
            ComputeRenderPassBuilder::default()
                .common(dead.build())
                .build()
                .expect("build dead pass"),
        ));

        rg.mark_output(&img_a);
        rg.compile().expect("compile failed");

        assert_eq!(*fired.lock(), vec!["live"], "the dead pass was recorded");
        assert!(
            !rg.passes.iter().any(|p| p.common().name == "dead"),
            "the dead pass survived the cull"
        );
        // No lifetime ⇒ no placement ⇒ no memory was ever allocated for it.
        assert!(
            !rg.transient_resources[slot].placements.contains_key(&img_dead.id),
            "img_dead was allocated despite its only writer being culled"
        );
        assert!(rg.transient_resources[slot].placements.contains_key(&img_a.id));
    }
}
