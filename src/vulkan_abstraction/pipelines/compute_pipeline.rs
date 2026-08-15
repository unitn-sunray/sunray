use crate::error::SrResult;
use crate::vulkan_abstraction::{Core, Device};
use ash::vk;
use ash::vk::TaggedStructure;
use std::marker::PhantomData;
use std::{ffi::CStr, sync::Arc};

const SHADER_ENTRY_POINT: &CStr = c"main";

pub trait ComputeTypeDef {
    type PushConstant;
    fn spirv_bytes() -> &'static [u8];
}

pub struct DenoisePass;
pub struct TemporalPass;
pub struct PostprocessPass;

/// Neutral marker for heap-mode compute pipelines built directly from Slang
/// SPIR-V where no legacy descriptor-set layout or fixed push-constant type
/// applies — e.g. render-graph passes constructed via
/// `ComputeRenderPassBuilder::generate_render`. Only valid with `new_heap`
/// (which ignores `PushConstant` / `spirv_bytes`); the legacy `new` path is
/// meaningless for it.
pub struct HeapComputePass;

impl ComputeTypeDef for HeapComputePass {
    type PushConstant = ();
    fn spirv_bytes() -> &'static [u8] {
        &[]
    }
}

impl ComputeTypeDef for DenoisePass {
    type PushConstant = DenoisePushConstant;

    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/denoise.spirv"))
    }
}

impl ComputeTypeDef for TemporalPass {
    type PushConstant = TemporalAccumulationPushConstant;
    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/temporal_accumulation.spirv"))
    }
}

impl ComputeTypeDef for PostprocessPass {
    type PushConstant = PostprocessPushConstant;

    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/postprocess.spirv"))
    }
}

///Push Constant for the denoiser pass.
/// Frame count is self explicative.
/// Step width references the distance between each pixel used as a sample during the a-trous filtering.
#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct DenoisePushConstant {
    pub frame_count: u32,
    pub step_width: u32,
    pub width: u32,
    pub height: u32,
}

/// Heap-mode push constant for `shaders/denoise.slang`. Layout mirrors the
/// shader's `DenoisePC`: five heap slot indices followed by the same scalar tail
/// as `DenoisePushConstant`.
#[allow(dead_code)] // read by the gpu
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DenoiseHeapPushConstant {
    pub temporal_result: u32,
    pub depth: u32,
    pub normal: u32,
    pub diffuse: u32,
    pub spatial_output: u32,
    pub frame_count: u32,
    pub step_width: i32,
    pub width: u32,
    pub height: u32,
}

#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct TemporalAccumulationPushConstant {
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
}

/// Heap-mode push constant for `shaders/temporal_accumulation.slang`. Layout
/// mirrors the shader's `TemporalPC`: four heap slot indices followed by the
/// scalar tail. All four images are bound as STORAGE, so the accumulation
/// ping-pong stays in GENERAL the whole time.
#[allow(dead_code)] // read by the gpu
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TemporalAccumulationHeapPushConstant {
    pub raw_rt_color: u32,
    pub motion_vector: u32,
    pub history: u32,
    pub accum_output: u32,
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
}

#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct PostprocessPushConstant {
    pub input_idx: u32,
    pub output_idx: u32,
    pub exposure: f32,
}
pub struct ComputePipeline<PushConstType> {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    _marker: PhantomData<PushConstType>,
}

/// A heap-mode pipeline built from one or more Slang/SPIR-V shaders.
///
/// All three concrete pipelines (`ComputePipeline`, `RayTracingPipeline`,
/// `GraphicsPipeline`) are heap-mode: the `VkPipelineLayout` is null and the
/// pipeline carries `DESCRIPTOR_HEAP_EXT`; the push-constant interface lives in
/// the shader SPIR-V and is fed via `vkCmdPushDataEXT`. `Shaders` is the
/// per-pipeline bundle of SPIR-V blobs (plus, for graphics, the fixed-function
/// vertex/format inputs) the constructor needs.
///
/// `new` takes `Arc<Core>` rather than a bare `ash::Device` because the
/// ray-tracing pipeline needs the `VK_KHR_ray_tracing_pipeline` device wrapper
/// (and every pipeline holds a `Core`/`Device` for `Drop`); compute derives the
/// device from it.
pub trait Pipeline {
    type Shaders;

    fn new(device: Arc<Core>, shaders: &Self::Shaders) -> SrResult<Self>
    where
        Self: Sized;

    fn inner(&self) -> vk::Pipeline;

    fn layout(&self) -> vk::PipelineLayout;
}

/// SPIR-V for a heap-mode compute pipeline's single entry point.
pub struct ComputePipelineShaders {
    pub compute_spirv: Vec<u8>,
}

impl<PushConstType> ComputePipeline<PushConstType> {
    /// Heap-mode constructor: pipeline layout has no descriptor sets, only push constants;
    /// the pipeline itself is flagged `DESCRIPTOR_HEAP_EXT`. Caller supplies the SPIR-V
    /// directly (e.g. from the Slang `ShaderCompiler`) since heap-mode shaders are not
    /// the build-time-baked GLSL ones referenced by `T::spirv_bytes`.
    pub fn new(device: Arc<Device>, spirv_bytes: &[u8]) -> SrResult<Self> {
        let spirv_u32 = bytemuck::cast_slice(spirv_bytes);

        let module_create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32);
        let shader_module = unsafe { device.inner().create_shader_module(&module_create_info, None) }?;

        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .name(SHADER_ENTRY_POINT)
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE);

        // VK_EXT_descriptor_heap requires `layout = VK_NULL_HANDLE` when
        // `DESCRIPTOR_HEAP_BIT_EXT` is set; the push-constant interface lives in the
        // shader's SPIR-V interface block instead, and is fed via `vkCmdPushDataEXT`.
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default().flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage_create_info)
            .layout(vk::PipelineLayout::null())
            .push(&mut flags2);

        let pipelines = unsafe {
            device
                .inner()
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| {
                    device.inner().destroy_shader_module(shader_module, None);
                    err
                })?
        };
        let pipeline = pipelines[0];

        unsafe { device.inner().destroy_shader_module(shader_module, None) };

        Ok(Self {
            device,
            pipeline,
            pipeline_layout: vk::PipelineLayout::null(),
            _marker: PhantomData,
        })
    }
}

impl<PushConstType> Pipeline for ComputePipeline<PushConstType> {
    type Shaders = ComputePipelineShaders;

    fn new(core: Arc<Core>, shaders: &Self::Shaders) -> SrResult<Self> {
        Self::new(core.clone_device(), &shaders.compute_spirv)
    }

    fn inner(&self) -> vk::Pipeline {
        self.pipeline
    }

    fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl<PushConstType> Drop for ComputePipeline<PushConstType> {
    fn drop(&mut self) {
        unsafe {
            self.device.inner().destroy_pipeline(self.pipeline, None);
            // Heap-mode pipelines are constructed with a null layout; only legacy
            // descriptor-set pipelines own one.
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device.inner().destroy_pipeline_layout(self.pipeline_layout, None);
            }
        }
    }
}
