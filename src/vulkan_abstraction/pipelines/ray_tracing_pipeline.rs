use std::{ffi::CStr, sync::Arc};

use crate::error::SrResult;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::{Core, Pipeline};

use ash::vk;
use ash::vk::TaggedStructure;

// should match the one defined in build.rs
const SHADER_ENTRY_POINT: &CStr = c"main";

#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug)]
pub struct RaytracingPushConstant {
    pub frame_count: u32,
    pub use_srgb: bool,
    pub _padding: [u8; 3], //push constant size must be a multiple of 4
}

/// Push-constant layout for the heap-mode (Slang) raytracing pipeline. Mirrors
/// `shaders/rt_types.slang::RaytracingPC` field for field — 80 bytes,
/// against the 256 bytes this driver reports as `maxPushDataSize` (heap mode
/// feeds the block through `vkCmdPushDataEXT`, so `maxPushConstantsSize` is not
/// the limit that applies).
///
/// Every field but `tlas` is a bare heap slot index. Slang's `DescriptorHandle<T>` lowers
/// to a `uint2` whose high word is never read, so the shader declares plain
/// `uint`s and rebuilds the handle at the use site; see the comment on
/// `RaytracingPC` for why.
#[allow(dead_code)] // read by the gpu
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct RaytracingHeapPushConstant {
    /// TLAS device address, not a heap slot. The shader runs
    /// `OpConvertUToAccelerationStructureKHR` on it directly; see the `tlas`
    /// comment in `shaders/rt_types.slang` for why the heap-handle form is
    /// still unusable on this driver.
    pub tlas: u64,
    pub raw_color: u32,
    pub depth_img: u32,
    pub normal_img: u32,
    pub diffuse_img: u32,
    pub motion_vec_img: u32,
    pub matrices: u32,
    pub meshes_info: u32,
    pub emissive_triangles: u32,
    pub emissive_indirection: u32,
    pub entity_transforms: u32,
    pub blue_noise_tex: u32,
    pub blue_noise_sampler: u32,
    /// Storage-buffer heap slots for the ping-pong reservoir buffers; the shader
    /// picks current/history internally from `frame_count`.
    pub reservoirs: [u32; 2],
    pub reservoirs_gi: [u32; 2],
    pub frame_count: u32,
    pub use_srgb: u32,
}

// One u64 followed by 18 4-byte slots, no padding. The shader block is verified
// to land on the same offsets (0, then 8,12,…,56,64,72,76). A mismatch here is
// silent GPU garbage, not a validation error, so pin the size.
const _: () = assert!(
    size_of::<RaytracingHeapPushConstant>() == 80,
    "RaytracingHeapPushConstant must stay in lockstep with RaytracingPC in shaders/rt_types.slang"
);

/// The four SPIR-V blobs a heap-mode ray-tracing pipeline links together — one
/// per stage. The SBT/dispatch currently assumes exactly one raygen + one miss +
/// one hit group (closest-hit + any-hit).
pub struct RayTracingPipelineShaders {
    pub ray_gen: Vec<u8>,
    pub miss: Vec<u8>,
    pub closest_hit: Vec<u8>,
    pub any_hit: Vec<u8>,
}

pub struct RayTracingPipeline {
    core: Arc<vulkan_abstraction::Core>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl RayTracingPipeline {
    /// Heap-mode constructor: pipeline layout is `VK_NULL_HANDLE` and the
    /// pipeline is flagged `DESCRIPTOR_HEAP_EXT`. All descriptors and the
    /// push-constant block come from the Slang shaders' SPIR-V interface,
    /// driven at command time by `cmd_bind_resource/sampler_heap` and
    /// `cmd_push_data`. Caller supplies the four SPIR-V byte slices for
    /// ray-gen, miss, closest-hit, and any-hit.
    pub fn new_heap(
        core: Arc<vulkan_abstraction::Core>,
        ray_gen_spirv: &[u8],
        miss_spirv: &[u8],
        closest_hit_spirv: &[u8],
        any_hit_spirv: &[u8],
    ) -> SrResult<Self> {
        let device = core.device().inner();

        let make_stage = |stage: vk::ShaderStageFlags, spirv: &[u8]| -> SrResult<vk::PipelineShaderStageCreateInfo> {
            let spirv_u32 = bytemuck::cast_slice(spirv);
            let module_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32);
            let module = unsafe { device.create_shader_module(&module_info, None) }?;
            Ok(vk::PipelineShaderStageCreateInfo::default()
                .name(SHADER_ENTRY_POINT)
                .module(module)
                .stage(stage))
        };

        let stages = [
            make_stage(vk::ShaderStageFlags::RAYGEN_KHR, ray_gen_spirv)?,
            make_stage(vk::ShaderStageFlags::MISS_KHR, miss_spirv)?,
            make_stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, closest_hit_spirv)?,
            make_stage(vk::ShaderStageFlags::ANY_HIT_KHR, any_hit_spirv)?,
        ];

        let shader_groups = [
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(3)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        // Heap-mode requires `layout = VK_NULL_HANDLE` plus the
        // `DESCRIPTOR_HEAP_EXT` flag; the push-constant block lives in the
        // shader interface and is fed by `vkCmdPushDataEXT`.
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default().flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);

        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&shader_groups)
            .max_pipeline_ray_recursion_depth(2)
            .layout(vk::PipelineLayout::null())
            .push(&mut flags2);

        let pipelines = unsafe {
            core.rt_pipeline_device().create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, e)| e)?;
        let pipeline = pipelines[0];

        for stage in &stages {
            unsafe { device.destroy_shader_module(stage.module, None) };
        }

        Ok(Self {
            core,
            pipeline,
            pipeline_layout: vk::PipelineLayout::null(),
        })
    }

    pub fn inner(&self) -> vk::Pipeline {
        self.pipeline
    }
    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl Pipeline for RayTracingPipeline {
    type Shaders = RayTracingPipelineShaders;

    fn new(core: Arc<Core>, shaders: &Self::Shaders) -> SrResult<Self> {
        Self::new_heap(core, &shaders.ray_gen, &shaders.miss, &shaders.closest_hit, &shaders.any_hit)
    }

    fn inner(&self) -> vk::Pipeline {
        self.pipeline
    }

    fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        let device = self.core.device().inner();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            // Heap-mode pipelines own no `VkPipelineLayout`; only the legacy
            // descriptor-set constructor creates one.
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
        }
    }
}
