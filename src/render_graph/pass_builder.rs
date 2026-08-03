use crate::error::{SrError, SrResult};
use crate::render_graph::error::GraphError;
use crate::render_graph::graph::{AnyRenderPass, PassResourceAccessSyncType, PassResourceAccessType, RenderGraph};
use crate::render_graph::resource::{Handle, Resource, ResourceRef};
use crate::render_graph::transient_resources::TransientResources;
use crate::vulkan_abstraction::{Buffer, GraphicsPipelineShaders, Pipeline, RawBuffer, RayTracingPipelineShaders};
use ash::vk;
use ash::vk::CommandBuffer;
use derive_builder::Builder;
use std::path::PathBuf;

pub(crate) struct PassCommonData {
    pub(crate) read: Vec<ResourceRef>,
    pub(crate) write: Vec<ResourceRef>,

    pub(crate) name: String,
    #[allow(dead_code)]
    id: u32,
    /// The user-supplied recording function for this pass. Invoked by
    /// `RenderGraph::compile` once per pass in topological order, after the
    /// barriers required by that pass's incoming edges have already been issued
    /// into the command buffer. `None` means "nothing to record" (e.g.
    /// pass kept only to anchor an external resource transition).
    pub(crate) render: Option<Box<DynRenderFn>>,
}

pub struct PassCommonDataBuilder {
    pass_common_data: PassCommonData,
}

/// Append `(res_id, access_type)` to a pass's read or write list unless that
/// exact pair is already there.
///
///
/// TODO Real source of duplicates today: a transfer pass copying several regions into
/// the same destination buffer declares `TransferWrite` on it once per region.
fn push_unique(
    list: &mut Vec<ResourceRef>,
    res_id: u32,
    access_type: vk_sync_fork::AccessType,
    sync_type: PassResourceAccessSyncType,
) {
    if list.iter().any(|r| r.id == res_id && r.access.access_type == access_type) {
        return;
    }
    list.push(ResourceRef {
        id: res_id,
        access: PassResourceAccessType { access_type, sync_type },
    });
}

impl PassCommonDataBuilder {
    pub fn new(rg: &mut RenderGraph, name: impl Into<String>) -> Self {
        Self {
            pass_common_data: PassCommonData {
                read: vec![],
                write: vec![],
                name: name.into(),
                id: rg.next_pass_id(),
                render: None,
            },
        }
    }

    /// Attach the recording closure to this pass. Replaces any previous one.
    pub fn render<F>(&mut self, f: F) -> &mut Self
    where
        F: FnMut(&mut CommandBuffer, &TransientResources) -> SrResult<()> + 'static,
    {
        self.pass_common_data.render = Some(Box::new(f));
        self
    }

    /// Finalize the builder and consume it into the `PassCommonData` that the
    /// concrete pass builders embed.
    pub(crate) fn build(self) -> PassCommonData {
        self.pass_common_data
    }

    /// Finalize the builder as an [`InternalPass`] — a declaration-only node the
    /// graph synthesizes for itself (see [`AnyRenderPass::Internal`]).
    pub(super) fn build_internal(self) -> InternalPass {
        InternalPass {
            common: self.pass_common_data,
        }
    }

    /// Declare an access on resource id `res_id` without the read/write
    /// validation [`Self::read`] and [`Self::write`] apply, routing it to the
    /// read or write list by [`vk_sync_fork::AccessType::is_write_access`].
    ///
    /// This is the internal-node path: `__imports` declares whatever access each
    /// imported resource carries into the frame, which the caller-facing builders
    /// have no way to express. It takes a bare id rather than a `Handle` because
    /// the graph reaches these resources by id and has no `Desc` to fabricate.
    ///
    /// `AccessType::Nothing` is deliberately routed to the **write** list — it
    /// means "contents are undefined and a transition is required", and in the
    /// read list it would leave `last_writer` unset so the first real user of the
    /// resource would get no barrier at all.
    pub(super) fn declare_previous_imports(&mut self, res_id: u32, access_type: vk_sync_fork::AccessType) {
        let sync_type = PassResourceAccessSyncType::AlwaysSync;
        if access_type.is_write_access() || access_type == vk_sync_fork::AccessType::Nothing {
            push_unique(&mut self.pass_common_data.write, res_id, access_type, sync_type);
        } else {
            push_unique(&mut self.pass_common_data.read, res_id, access_type, sync_type);
        }
    }

    /// Declare an access on a resource routing it to the
    /// read or write list by [`vk_sync_fork::AccessType::is_write_access`].
    pub fn declare<Res: Resource>(&mut self, resource: &Handle<Res>, access_type: vk_sync_fork::AccessType) {
        let sync_type = PassResourceAccessSyncType::AlwaysSync;
        if access_type.is_write_access() {
            push_unique(&mut self.pass_common_data.write, resource.id, access_type, sync_type);
        } else {
            push_unique(&mut self.pass_common_data.read, resource.id, access_type, sync_type);
        }
    }

    pub fn read<Res: Resource>(&mut self, resource: &Handle<Res>, access_type: vk_sync_fork::AccessType) -> SrResult<()> {
        if !access_type.is_write_access() {
            push_unique(
                &mut self.pass_common_data.read,
                resource.id,
                access_type,
                PassResourceAccessSyncType::NeverSync,
            );
            Ok(())
        } else {
            Err(SrError::new(
                GraphError::IncorrectRenderAccessFlags.into(),
                format!("asked to read with such access: {access_type:?}"),
            ))
        }
    }

    pub fn write<Res: Resource>(&mut self, resource: &Handle<Res>, access_type: vk_sync_fork::AccessType) -> SrResult<()> {
        //TODO this needs to change the resource version
        //TODO more complex not always sync write+write and read+write and render graph state id lookup

        if access_type.is_write_access() {
            push_unique(
                &mut self.pass_common_data.write,
                resource.id,
                access_type,
                PassResourceAccessSyncType::AlwaysSync,
            );
            Ok(())
        } else {
            Err(SrError::new(
                GraphError::IncorrectRenderAccessFlags.into(),
                format!("asked to write with such access: {access_type:?}"),
            ))
        }
    }
}

/// Borrow the pre-compiled SPIR-V out of a `ShaderSource`. The heap-mode
/// pipeline helpers (`*RenderPassBuilder::generate_render`) only accept
/// already-compiled `Spirv`; `Glsl`/`Slang` sources must be compiled upstream
/// (e.g. via the Slang `ShaderCompiler`) before reaching the builder. `context`
/// is the caller label used in the error message.
fn extract_spirv<'a>(src: &'a ShaderSource, context: &str) -> SrResult<&'a [u8]> {
    match src {
        ShaderSource::Spirv(bytes) => Ok(bytes.as_slice()),
        ShaderSource::Glsl(path) => Err(SrError::new_custom(format!(
            "{context} only accepts ShaderSource::Spirv; got Glsl({path:?})"
        ))),
        ShaderSource::Slang(path) => Err(SrError::new_custom(format!(
            "{context} only accepts ShaderSource::Spirv; got Slang({path:?})"
        ))),
    }
}

impl From<RaytracingRenderPass> for AnyRenderPass {
    fn from(val: RaytracingRenderPass) -> Self {
        AnyRenderPass::Rt(val)
    }
}

impl From<RasterRenderPass> for AnyRenderPass {
    fn from(val: RasterRenderPass) -> Self {
        AnyRenderPass::Raster(val)
    }
}

impl From<ComputeRenderPass> for AnyRenderPass {
    fn from(val: ComputeRenderPass) -> Self {
        AnyRenderPass::Compute(val)
    }
}

impl From<TransferPass> for AnyRenderPass {
    fn from(val: TransferPass) -> Self {
        AnyRenderPass::Transfer(val)
    }
}

impl From<InternalPass> for AnyRenderPass {
    fn from(val: InternalPass) -> Self {
        AnyRenderPass::Internal(val)
    }
}

#[derive(Builder)]
#[builder(pattern = "owned")]
pub(crate) struct RaytracingRenderPass {
    pub(super) common: PassCommonData,
    /// The Slang/SPIR-V shaders + per-stage entry points this pass compiles into
    /// its pipeline. `None` for passes whose `common.render` closure binds a
    /// pre-built (persistent) pipeline directly instead of going through
    /// `RaytracingRenderPassBuilder::generate_render`.
    // Backs the generated `.shaders()` / `.trace_extent()` setters, which
    // `generate_render` reads off the *builder* before the pipeline is cached.
    // Nothing reads them off the built pass, but derive_builder needs the fields.
    #[builder(setter(strip_option), default)]
    #[allow(dead_code)]
    pub(super) shaders: Option<RayTracingShaders>,
    #[allow(dead_code)]
    pub(super) trace_extent: [u32; 3],
}

/// The shaders backing a single ray-tracing pass. Like [`ComputeShaders`], each
/// stage's entry point is the `(index, name)` of the owning module in `shaders`
/// so the builder picks it directly instead of scanning for it. The SBT/dispatch
/// currently assumes one raygen + one miss + one hit group (closest-hit +
/// any-hit).
pub struct RayTracingShaders {
    pub(super) shaders: Vec<ShaderSource>,
    pub(super) ray_gen: (usize, String),
    pub(super) miss: (usize, String),
    pub(super) closest_hit: (usize, String),
    pub(super) any_hit: (usize, String),
}

impl RayTracingShaders {
    pub fn new(
        shaders: Vec<ShaderSource>,
        ray_gen: (usize, impl Into<String>),
        miss: (usize, impl Into<String>),
        closest_hit: (usize, impl Into<String>),
        any_hit: (usize, impl Into<String>),
    ) -> Self {
        Self {
            shaders,
            ray_gen: (ray_gen.0, ray_gen.1.into()),
            miss: (miss.0, miss.1.into()),
            closest_hit: (closest_hit.0, closest_hit.1.into()),
            any_hit: (any_hit.0, any_hit.1.into()),
        }
    }
}

impl RaytracingRenderPassBuilder {
    /// Eagerly build the heap-mode raytracing pipeline + shader binding table
    /// for this pass and install a render closure on `common` that, when
    /// invoked by the graph at record time, binds the descriptor heap, binds
    /// the pipeline, and issues `cmd_trace_rays` with the configured
    /// `trace_extent`.
    ///
    /// Requires `common`, `shaders`, and `trace_extent` to already be set on the
    /// builder. The per-stage entry points are selected directly by their index
    /// into `RayTracingShaders::shaders`, and each selected module must be a
    /// `ShaderSource::Spirv` — heap-mode shaders are not compiled by this helper;
    /// callers compile them externally (e.g. via the Slang `ShaderCompiler`) and
    /// hand the bytes in.
    ///
    /// `push` is invoked at graph record time with the populated
    /// `TransientResources`, returning the raw push-constant bytes (e.g. a
    /// `RaytracingHeapPushConstant`). It can resolve graph image handles to heap
    /// slots via `tr.image(&h)?.storage_slot()` before assembling the bytes. If
    /// it returns an empty `Vec`, no `cmd_push_data` is issued.
    ///
    /// Like the compute builder, this lets the caller describe an RT pass with
    /// **only its shaders + a push-data closure**; the pipeline and SBT are built
    /// here so the caller never constructs a `RayTracingPipeline`/`ShaderBindingTable`
    /// itself.
    ///
    /// Note: this builds (interns) a single ray-tracing pipeline per pass. The
    /// app's two-pass RIS + final pipeline runs as two passes on this path, each
    /// interning its own pipeline + SBT in the cache; the RIS→final reservoir
    /// hand-off is expressed as a graph edge on the imported reservoir buffers
    /// (the RIS pass declares the reservoir writes, the final pass the reads), so
    /// the graph emits the barrier — no manual barrier in the closure.
    pub fn generate_render<F>(mut self, rg: &mut RenderGraph, push: F) -> SrResult<Self>
    where
        F: Fn(&TransientResources) -> SrResult<Vec<u8>> + 'static,
    {
        const CTX: &str = "RaytracingRenderPassBuilder::generate_render";

        let shaders = self
            .shaders
            .take()
            .flatten()
            .ok_or_else(|| SrError::new_custom(format!("{CTX}: no shaders set")))?;
        let trace_extent = self
            .trace_extent
            .ok_or_else(|| SrError::new_custom(format!("{CTX}: trace_extent not set")))?;

        // Pick each stage's module directly by its index — no scanning.
        let stage_spirv = |stage: &str, (idx, name): &(usize, String)| -> SrResult<Vec<u8>> {
            let src = shaders.shaders.get(*idx).ok_or_else(|| {
                SrError::new_custom(format!(
                    "{CTX}: {stage} entry_point index {idx} (\"{name}\") out of range for {} shader(s)",
                    shaders.shaders.len()
                ))
            })?;
            Ok(extract_spirv(src, CTX)?.to_vec())
        };

        // SBT currently hardcodes 1 raygen + 1 miss + 1 hit group.
        let rt_shaders = RayTracingPipelineShaders {
            ray_gen: stage_spirv("ray_gen", &shaders.ray_gen)?,
            miss: stage_spirv("miss", &shaders.miss)?,
            closest_hit: stage_spirv("closest_hit", &shaders.closest_hit)?,
            any_hit: stage_spirv("any_hit", &shaders.any_hit)?,
        };

        // Intern the pipeline + SBT in the graph's persistent cache (built once,
        // reused across frame rebuilds), resolved by handle in the closure.
        let handle = rg.cache_raytracing_pipeline(&rt_shaders)?;
        let core = rg.core();

        let mut common = self
            .common
            .take()
            .ok_or_else(|| SrError::new_custom(format!("{CTX}: common not set")))?;

        common.render = Some(Box::new(move |cb, tr| {
            let (pipeline, sbt) = tr.raytracing_pipeline(handle)?;
            let push_bytes = push(tr)?;
            let device = core.device().inner();
            unsafe {
                core.descriptor_heap().cmd_bind(*cb);
                device.cmd_bind_pipeline(*cb, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.inner());
                if !push_bytes.is_empty() {
                    let push_info = vk::PushDataInfoEXT::default().offset(0).data(vk::HostAddressRangeConstEXT {
                        address: push_bytes.as_ptr() as *const std::ffi::c_void,
                        size: push_bytes.len(),
                        _marker: Default::default(),
                    });
                    core.descriptor_heap_device().cmd_push_data(*cb, &push_info);
                }
                core.rt_pipeline_device().cmd_trace_rays(
                    *cb,
                    sbt.raygen_region(),
                    sbt.miss_region(),
                    sbt.hit_region(),
                    sbt.callable_region(),
                    trace_extent[0],
                    trace_extent[1],
                    trace_extent[2],
                );
            }
            Ok(())
        }));

        self.common = Some(common);
        Ok(self)
    }
}

// TODO EXPERIMENTAL, UNTESTED. The heap-mode raster path mirrors the compute and
// ray-tracing builders below, but it has never been driven end-to-end. The
// fixed-function state baked into `GraphicsPipeline::new_heap` is currently tuned
// for 2D alpha-blended overlays (the egui paint pass), and `generate_render`
// below does NOT yet wire up dynamic-rendering attachments, viewport/scissor,
// vertex/index buffer binding, or the draw call — verify all of that before
// relying on this.
#[derive(Builder)]
#[builder(pattern = "owned")]
pub(crate) struct RasterRenderPass {
    pub(super) common: PassCommonData,
    /// The vertex/fragment Slang/SPIR-V shaders + entry points this pass compiles
    /// into its graphics pipeline. `None` for passes whose `common.render`
    /// closure binds a pre-built pipeline directly.
    #[builder(setter(strip_option), default)]
    // See `RaytracingRenderPass::shaders` — backs the builder setter only.
    #[allow(dead_code)]
    pub(super) shaders: Option<RasterShaders>,
}

/// The shaders + fixed-function inputs backing a single raster pass. Like
/// [`ComputeShaders`], `vertex` / `fragment` are the `(index, name)` of the
/// owning module in `shaders` so the builder picks each stage directly.
pub struct RasterShaders {
    pub(super) shaders: Vec<ShaderSource>,
    pub(super) vertex: (usize, String),
    pub(super) fragment: (usize, String),
    pub(super) color_format: vk::Format,
    pub(super) vertex_binding: vk::VertexInputBindingDescription,
    pub(super) vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
}

impl RasterShaders {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shaders: Vec<ShaderSource>,
        vertex: (usize, impl Into<String>),
        fragment: (usize, impl Into<String>),
        color_format: vk::Format,
        vertex_binding: vk::VertexInputBindingDescription,
        vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
    ) -> Self {
        Self {
            shaders,
            vertex: (vertex.0, vertex.1.into()),
            fragment: (fragment.0, fragment.1.into()),
            color_format,
            vertex_binding,
            vertex_attributes,
        }
    }
}

impl RasterRenderPassBuilder {
    /// EXPERIMENTAL / UNTESTED heap-mode raster pass. Builds a [`GraphicsPipeline`]
    /// from this pass's `shaders` (vertex + fragment selected directly by index)
    /// and installs a render closure that binds the descriptor heap, binds the
    /// pipeline, and pushes the caller-provided bytes via `cmd_push_data`.
    ///
    /// The selected modules must be `ShaderSource::Spirv` — heap shaders are
    /// compiled by the caller (e.g. via the Slang `ShaderCompiler`).
    ///
    /// TODO: this does NOT yet begin/end dynamic rendering, set the
    /// viewport/scissor, bind vertex/index buffers, or issue a draw — those depend
    /// on a color target + geometry the graph doesn't model yet. Wire them up and
    /// test against a real attachment before using this for anything.
    #[allow(dead_code)]
    pub fn generate_render<F>(mut self, rg: &mut RenderGraph, push: F) -> SrResult<Self>
    where
        F: Fn(&TransientResources) -> SrResult<Vec<u8>> + 'static,
    {
        const CTX: &str = "RasterRenderPassBuilder::generate_render";

        let shaders = self
            .shaders
            .take()
            .flatten()
            .ok_or_else(|| SrError::new_custom(format!("{CTX}: no shaders set")))?;

        let stage_spirv = |stage: &str, (idx, name): &(usize, String)| -> SrResult<Vec<u8>> {
            let src = shaders.shaders.get(*idx).ok_or_else(|| {
                SrError::new_custom(format!(
                    "{CTX}: {stage} entry_point index {idx} (\"{name}\") out of range for {} shader(s)",
                    shaders.shaders.len()
                ))
            })?;
            Ok(extract_spirv(src, CTX)?.to_vec())
        };

        let graphics_shaders = GraphicsPipelineShaders {
            vertex: stage_spirv("vertex", &shaders.vertex)?,
            fragment: stage_spirv("fragment", &shaders.fragment)?,
            color_format: shaders.color_format,
            vertex_binding: shaders.vertex_binding,
            vertex_attributes: shaders.vertex_attributes.clone(),
        };

        // Intern in the graph's persistent cache (built once, reused across frame
        // rebuilds), resolved by handle in the closure.
        let handle = rg.cache_graphics_pipeline(&graphics_shaders)?;
        let core = rg.core();

        let mut common = self
            .common
            .take()
            .ok_or_else(|| SrError::new_custom(format!("{CTX}: common not set")))?;

        common.render = Some(Box::new(move |cb, tr| {
            let pipeline = tr.graphics_pipeline(handle)?;
            let push_bytes = push(tr)?;
            let device = core.device().inner();
            unsafe {
                core.descriptor_heap().cmd_bind(*cb);
                device.cmd_bind_pipeline(*cb, vk::PipelineBindPoint::GRAPHICS, pipeline.inner());
                if !push_bytes.is_empty() {
                    let push_info = vk::PushDataInfoEXT::default().offset(0).data(vk::HostAddressRangeConstEXT {
                        address: push_bytes.as_ptr() as *const std::ffi::c_void,
                        size: push_bytes.len(),
                        _marker: Default::default(),
                    });
                    core.descriptor_heap_device().cmd_push_data(*cb, &push_info);
                }
                // TODO EXPERIMENTAL: begin_rendering(color attachment), set dynamic
                // viewport/scissor, bind vertex/index buffers, cmd_draw*, end_rendering.
                // Without this the pass binds state but draws nothing.
            }
            Ok(())
        }));

        self.common = Some(common);
        Ok(self)
    }
}

/// Which queue a compute pass runs on.
///
/// Compute is the only pass kind whose queue is a real choice: raster and ray
/// tracing always need the universal queue and buffer copies always belong on
/// the DMA queue, so those are derived from the pass kind alone (see
/// [`AnyRenderPass::queue_role`](super::graph::AnyRenderPass::queue_role)).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ComputeQueueAffinity {
    /// Pin to the universal (graphics) queue.
    Universal,
    /// Pin to the async compute queue.
    AsyncCompute,
    /// TODO: let the scheduler pick, once the multi-queue heuristic exists.
    /// Resolves to the universal queue until then.
    #[default]
    Inferred,
}

#[derive(Builder)]
#[builder(pattern = "owned")]
/// Remember the push-constant type used in `generate_render` must be `#[repr(C)]`
/// or the gpu will get garbage data caused by rust reordering the fields of a struct.
pub(crate) struct ComputeRenderPass {
    pub(super) common: PassCommonData,
    /// The Slang/SPIR-V shaders + entry point this pass compiles into its
    /// pipeline. `None` for passes whose `common.render` closure binds a
    /// pre-built (persistent) pipeline directly instead of going through
    /// `ComputeRenderPassBuilder::generate_render`.
    #[builder(setter(strip_option), default)]
    // See `RaytracingRenderPass::shaders` — backs the builder setter only.
    #[allow(dead_code)]
    pub(super) shaders: Option<ComputeShaders>,
    /// Which queue this pass wants.
    #[builder(default)]
    pub(super) queue_affinity: ComputeQueueAffinity,
}

/// One endpoint of a buffer copy recorded by a [`TransferPass`].
///
/// `Handle` endpoints are graph resources: the pass declares a `TransferRead` /
/// `TransferWrite` on them, so the graph infers their lifetimes and emits the
/// surrounding barriers. `Raw` endpoints are untracked `vk::Buffer`s the graph
/// knows nothing about — it declares nothing and orders nothing for them, which
/// is why every builder method that takes one is `unsafe`.
#[derive(Copy, Clone, Debug)]
pub(crate) enum CopyEnd {
    Handle(u32),
    Raw(vk::Buffer),
}

/// A pass whose entire content is a list of buffer copies, described
/// declaratively rather than through a render closure.
/// When possible this runs on a queue with direct access to the DMA.
///
/// `common.render` is always `None`; the graph records the copies itself.
pub(crate) struct TransferPass {
    pub(super) common: PassCommonData,
    pub(super) copies: Vec<(CopyEnd, CopyEnd, vk::BufferCopy)>,
}

/// Check that `[offset, offset + region.size)` lies inside a buffer of
/// `byte_size` bytes, and that the copy moves at least one byte.
///
/// Only callable for endpoints the graph can see: a `Handle<RawBuffer>` carries
/// its `BufferDesc`, so the size is known at build time. Raw `vk::Buffer`
/// endpoints have no size the graph can reach, which is why the methods taking
/// one are `unsafe` — there the same bound becomes the caller's obligation.
///
/// Worth checking eagerly rather than leaving to validation layers: an
/// out-of-bounds `vkCmdCopyBuffer` (VUID-vkCmdCopyBuffer-srcOffset-00113 /
/// -dstOffset-00114) corrupts unrelated GPU memory or loses the device long
/// after the fact, inside a command buffer that no longer says who built it.
/// Failing here names the copy at the point it was declared.
pub(super) fn check_copy_bounds(
    which: &str,
    buffer_name: &str,
    byte_size: vk::DeviceSize,
    offset: vk::DeviceSize,
    region: &vk::BufferCopy,
) -> SrResult<()> {
    // VUID-VkBufferCopy-size-01988: a copy region must move at least one byte.
    if region.size == 0 {
        return Err(SrError::new(
            GraphError::CopyRegionOutOfBounds.into(),
            format!("transfer pass copy region has size 0 ({which} buffer \"{buffer_name}\")"),
        ));
    }
    // `checked_add`: offset and size are both caller-supplied `u64`, so a naive
    // `offset + size` can wrap and make an out-of-bounds copy look valid.
    if region.size.checked_add(offset).is_none_or(|end| end > byte_size) {
        return Err(SrError::new(
            GraphError::CopyRegionOutOfBounds.into(),
            format!(
                "transfer pass copy runs past the end of its {which} buffer \"{buffer_name}\": \
                 offset {offset} + size {} exceeds byte_size {byte_size}",
                region.size
            ),
        ));
    }
    Ok(())
}

/// Builder for [`TransferPass`]. Each `copy*` method declares the graph accesses
/// it can, and bounds-checks every endpoint whose size it can reach — see
/// [`CopyEnd`] for why three of the four are `unsafe`.
pub struct TransferPassBuilder {
    common: PassCommonDataBuilder,
    copies: Vec<(CopyEnd, CopyEnd, vk::BufferCopy)>,
}

impl TransferPassBuilder {
    pub fn new(rg: &mut RenderGraph, name: impl Into<String>) -> Self {
        Self {
            common: PassCommonDataBuilder::new(rg, name),
            copies: Vec::new(),
        }
    }

    /// Copy between two graph resources. Declares `TransferRead` on `src` and
    /// `TransferWrite` on `dst`, so lifetimes and barriers are inferred, and
    /// bounds-checks `region` against both buffers.
    pub fn copy(&mut self, src: &Handle<RawBuffer>, dst: &Handle<RawBuffer>, region: vk::BufferCopy) -> SrResult<()> {
        check_copy_bounds("source", src.desc.name, src.desc.byte_size, region.src_offset, &region)?;
        check_copy_bounds("destination", dst.desc.name, dst.desc.byte_size, region.dst_offset, &region)?;
        self.common.read(src, vk_sync_fork::AccessType::TransferRead)?;
        self.common.write(dst, vk_sync_fork::AccessType::TransferWrite)?;
        self.copies.push((CopyEnd::Handle(src.id), CopyEnd::Handle(dst.id), region));
        Ok(())
    }

    /// Copy from an untracked buffer into a graph resource. Declares
    /// `TransferWrite` on `dst`, and bounds-checks `region` against **both**
    /// ends — `src` is untracked by the graph but still a live `Buffer`, so its
    /// size is readable here.
    ///
    /// # Safety
    /// `src` must outlive this graph submission — the borrow ends when this call
    /// returns, but the GPU reads the buffer much later — must not alias any
    /// graph-tracked resource, and is the caller's to synchronize against
    /// non-graph work: the graph emits no barriers for it.
    pub unsafe fn copy_from_raw(&mut self, src: &impl Buffer, dst: &Handle<RawBuffer>, region: vk::BufferCopy) -> SrResult<()> {
        check_copy_bounds("source", "untracked", src.byte_size(), region.src_offset, &region)?;
        check_copy_bounds("destination", dst.desc.name, dst.desc.byte_size, region.dst_offset, &region)?;
        self.common.write(dst, vk_sync_fork::AccessType::TransferWrite)?;
        self.copies.push((CopyEnd::Raw(src.inner()), CopyEnd::Handle(dst.id), region));
        Ok(())
    }

    /// Copy from a graph resource into an untracked buffer. Declares
    /// `TransferRead` on `src`, and bounds-checks `region` against both ends.
    ///
    /// # Safety
    /// `dst` must outlive this graph submission — the borrow ends when this call
    /// returns, but the GPU writes the buffer much later — must not alias any
    /// graph-tracked resource, and is the caller's to synchronize against
    /// non-graph work: the graph emits no barriers for it.
    pub unsafe fn copy_to_raw(&mut self, src: &Handle<RawBuffer>, dst: &impl Buffer, region: vk::BufferCopy) -> SrResult<()> {
        check_copy_bounds("source", src.desc.name, src.desc.byte_size, region.src_offset, &region)?;
        check_copy_bounds("destination", "untracked", dst.byte_size(), region.dst_offset, &region)?;
        self.common.read(src, vk_sync_fork::AccessType::TransferRead)?;
        self.copies.push((CopyEnd::Handle(src.id), CopyEnd::Raw(dst.inner()), region));
        Ok(())
    }

    /// Copy between two untracked buffers. Both ends are bounds-checked, but the
    /// graph declares nothing and orders nothing — it only records the copy at
    /// this point in the stream.
    ///
    /// # Safety
    /// Both buffers must outlive this graph submission — the borrows end when
    /// this call returns, but the GPU runs the copy much later — neither may
    /// alias a graph-tracked resource, and both are the caller's to synchronize
    /// against all other work, in-graph included: the graph emits no barriers
    /// for either.
    pub unsafe fn copy_raw(&mut self, src: &impl Buffer, dst: &impl Buffer, region: vk::BufferCopy) -> SrResult<()> {
        check_copy_bounds("source", "untracked", src.byte_size(), region.src_offset, &region)?;
        check_copy_bounds("destination", "untracked", dst.byte_size(), region.dst_offset, &region)?;
        self.copies
            .push((CopyEnd::Raw(src.inner()), CopyEnd::Raw(dst.inner()), region));
        Ok(())
    }

    /// Prologue-copy path: `src` is a bare handle whose size the graph cannot
    /// reach, because [`RenderGraph::add_prologue_buffer_copies`] already
    /// bounds-checked it at queue time — the only point where the owning
    /// `impl Buffer` is still in scope. `RenderGraph` has no lifetime parameter,
    /// so the borrow cannot be parked in `prologue_copies` until compile.
    ///
    /// The `dst` side is still checked here.
    ///
    /// # Safety
    /// As [`Self::copy_from_raw`], plus: `src` must be the buffer whose
    /// `byte_size` was checked against `region` when the copy was queued.
    pub(super) unsafe fn copy_from_prechecked_raw(
        &mut self,
        src: vk::Buffer,
        dst: &Handle<RawBuffer>,
        region: vk::BufferCopy,
    ) -> SrResult<()> {
        check_copy_bounds("destination", dst.desc.name, dst.desc.byte_size, region.dst_offset, &region)?;
        self.common.write(dst, vk_sync_fork::AccessType::TransferWrite)?;
        self.copies.push((CopyEnd::Raw(src), CopyEnd::Handle(dst.id), region));
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }

    pub(crate) fn build(self) -> TransferPass {
        TransferPass {
            common: self.common.build(),
            copies: self.copies,
        }
    }
}

/// A node the graph synthesizes for itself rather than one the caller adds.
///
/// Declaration-only: `common.render` is `None` and the pass records nothing. It
/// exists so implicit per-frame work is expressed as ordinary graph
/// declarations — currently `__imports`, which declares the access every
/// imported resource carries into the frame so the hazard scan orders the first
/// consumer against the previous frame.
///
/// Pinned to the universal queue by
/// [`AnyRenderPass::queue_role`](super::graph::AnyRenderPass::queue_role).
pub(crate) struct InternalPass {
    pub(super) common: PassCommonData,
}

/// The shaders backing a single compute pass. Instead of carrying the entry
/// point as a bare name and scanning every shader for it, the entry point is the
/// `(index, name)` of the shader in `shaders` that owns it — so the builder can
/// pick the right module directly.
pub struct ComputeShaders {
    pub(super) shaders: Vec<ShaderSource>,
    /// `(index into `shaders` of the module containing the entry point, entry point name)`
    pub(super) entry_point: (usize, String),
}

impl ComputeShaders {
    pub fn new(shaders: Vec<ShaderSource>, entry_index: usize, entry_point: impl Into<String>) -> Self {
        Self {
            shaders,
            entry_point: (entry_index, entry_point.into()),
        }
    }
}

impl ComputeRenderPassBuilder {
    /// Build a heap-mode compute pipeline from this pass's `shaders` and install
    /// the render closure that binds the descriptor heap + pipeline, pushes the
    /// caller's push constant via `cmd_push_data`, and dispatches `dispatch`
    /// workgroups.
    ///
    /// This is the one-stop entry point that lets the caller (e.g. `lib.rs`)
    /// describe a graph pass with **only its shaders and a push-data closure** —
    /// pipeline creation lives here, so the caller never builds a
    /// `ComputePipeline` (or writes the bind/dispatch boilerplate) itself.
    ///
    /// `get_push_data` is invoked at record time with the populated
    /// `TransientResources`, so it can resolve graph image handles to heap slots
    /// (`tr.image(&h)?.storage_slot()`) before returning the `PushConstType`
    /// value pushed to the shader. A zero-sized `PushConstType` skips
    /// `cmd_push_data`.
    ///
    /// Requires `shaders` to be set (via `.shaders(ComputeShaders::new(..))`). The
    /// entry-point module is selected directly by `ComputeShaders::entry_point.0`
    /// (its index in the shader list) and must be a `ShaderSource::Spirv` — heap
    /// shaders are compiled by the caller, e.g. via the Slang `ShaderCompiler`.
    ///
    /// `PushConstType` must be `#[repr(C)]` plain data: it is copied verbatim into
    /// the push-constant range, so any field reordering/padding the GPU doesn't
    /// expect would corrupt it.
    pub fn generate_render<PushConstType: Copy + 'static>(
        mut self,
        rg: &mut RenderGraph,
        dispatch: [u32; 3],
        get_push_data: impl Fn(&TransientResources) -> SrResult<PushConstType> + 'static,
    ) -> SrResult<ComputeRenderPass> {
        const CTX: &str = "ComputeRenderPassBuilder::generate_render";

        let shaders = self
            .shaders
            .take()
            .flatten()
            .ok_or_else(|| SrError::new_custom(format!("{CTX}: no shaders set")))?;

        // Pick the entry-point module directly by its index — no scanning the
        // whole shader list to find which one owns the entry point.
        let (entry_index, entry_name) = &shaders.entry_point;
        let entry_shader = shaders.shaders.get(*entry_index).ok_or_else(|| {
            SrError::new_custom(format!(
                "{CTX}: entry_point index {entry_index} (\"{entry_name}\") out of range for {} shader(s)",
                shaders.shaders.len()
            ))
        })?;
        let spirv = extract_spirv(entry_shader, CTX)?;

        // Intern the pipeline in the graph's persistent cache: built once per
        // distinct shader, reused across the per-frame graph rebuilds, and
        // resolved by handle inside the render closure below.
        let handle = rg.cache_compute_pipeline(spirv)?;
        let core = rg.core();

        let mut common = self
            .common
            .take()
            .ok_or_else(|| SrError::new_custom(format!("{CTX}: common not set")))?;

        common.render = Some(Box::new(move |cb, tr| {
            let pipeline = tr.compute_pipeline(handle)?;
            let device = core.device().inner();

            let push_data: PushConstType = get_push_data(tr)?;

            let push_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(&push_data as *const PushConstType as *const u8, size_of::<PushConstType>())
            };

            unsafe {
                device.cmd_bind_pipeline(*cb, vk::PipelineBindPoint::COMPUTE, pipeline.inner());
                core.descriptor_heap().cmd_bind(*cb);
                if !push_bytes.is_empty() {
                    let push_info = vk::PushDataInfoEXT::default().offset(0).data(vk::HostAddressRangeConstEXT {
                        address: push_bytes.as_ptr() as *const std::ffi::c_void,
                        size: push_bytes.len(),
                        _marker: Default::default(),
                    });
                    core.descriptor_heap_device().cmd_push_data(*cb, &push_info);
                }
                device.cmd_dispatch(*cb, dispatch[0], dispatch[1], dispatch[2]);
            }
            Ok(())
        }));

        Ok(ComputeRenderPass {
            common,
            shaders: Some(shaders),
            queue_affinity: ComputeQueueAffinity::default(),
        })
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub enum RayTracingPipelineStage {
    RayGen,
    RayMiss,
    RayClosestHit,
}

pub trait ShaderDesc {}

#[derive(Clone, Debug)]
pub enum ShaderSource {
    //TODO supported shaders, for now pre-compiled SPIR-V is the only one supported
    //TODO the slang compiler and shaderc are to be moved inside the render_graph as he is the one allowed to compile at runtime,all other stuff is precompiled
    Glsl(PathBuf),
    Slang(PathBuf),
    /// Pre-compiled SPIR-V bytes — produced upstream (e.g. by the Slang
    /// `ShaderCompiler`) and consumed verbatim by heap-mode pipeline helpers
    /// like `RaytracingRenderPassBuilder::generate_render`.
    Spirv(Vec<u8>),
}

pub(crate) type DynRenderFn = dyn FnMut(&mut CommandBuffer, &TransientResources) -> SrResult<()>;
