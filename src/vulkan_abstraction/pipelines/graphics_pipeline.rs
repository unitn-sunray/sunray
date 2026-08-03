use std::ffi::CStr;
use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure;

use crate::error::SrResult;
use crate::vulkan_abstraction::{Core, Pipeline};

// Slang emits every entry point as a SPIR-V "main" (see ray_tracing_pipeline.rs).
const ENTRY_POINT: &CStr = c"main";

/// Inputs for a heap-mode graphics pipeline: the vertex + fragment SPIR-V plus
/// the fixed-function vertex layout and color-attachment format the pipeline is
/// specialized for.
pub struct GraphicsPipelineShaders {
    pub vertex: Vec<u8>,
    pub fragment: Vec<u8>,
    pub color_format: vk::Format,
    pub vertex_binding: vk::VertexInputBindingDescription,
    pub vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
}

/// Heap-mode graphics (raster) pipeline.
///
/// Like [`crate::vulkan_abstraction::ComputePipeline::new`], the pipeline
/// layout is `VK_NULL_HANDLE` and the pipeline carries `DESCRIPTOR_HEAP_EXT`; the
/// push-constant interface lives in the shader's SPIR-V and is fed via
/// `vkCmdPushDataEXT`. Resources are addressed through `DescriptorHandle<>` in the
/// shader, not bound descriptor sets.
///
/// Fixed-function state is currently tuned for **2D alpha-blended overlays**
/// (the egui paint pass): triangle list, no depth/stencil, cull none,
/// premultiplied-alpha blend, dynamic viewport + scissor, and a single color
/// attachment supplied via dynamic rendering. The caller provides the vertex
/// layout and the color-attachment format.
pub struct GraphicsPipeline {
    core: Arc<Core>,
    pipeline: vk::Pipeline,
}

impl GraphicsPipeline {
    pub fn new_heap(
        core: Arc<Core>,
        vertex_spirv: &[u8],
        fragment_spirv: &[u8],
        color_format: vk::Format,
        vertex_binding: vk::VertexInputBindingDescription,
        vertex_attributes: &[vk::VertexInputAttributeDescription],
    ) -> SrResult<Self> {
        let device = core.device().inner();

        let make_module = |bytes: &[u8]| -> SrResult<vk::ShaderModule> {
            let code = bytemuck::cast_slice(bytes);
            let info = vk::ShaderModuleCreateInfo::default().code(code);
            Ok(unsafe { device.create_shader_module(&info, None) }?)
        };
        let vert_module = make_module(vertex_spirv)?;
        let frag_module = make_module(fragment_spirv)?;

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(ENTRY_POINT),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(ENTRY_POINT),
        ];

        let bindings = [vertex_binding];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(vertex_attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Premultiplied-alpha blending (egui convention).
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let blend_attachments = [blend_attachment];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = [color_format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        // Heap mode: null layout + DESCRIPTOR_HEAP_EXT (same as the compute path).
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default().flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(vk::PipelineLayout::null())
            .push(&mut rendering)
            .push(&mut flags2);

        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| {
                    device.destroy_shader_module(vert_module, None);
                    device.destroy_shader_module(frag_module, None);
                    err
                })?
        };
        let pipeline = pipelines[0];

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Ok(Self { core, pipeline })
    }

    pub fn inner(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl Pipeline for GraphicsPipeline {
    type Shaders = GraphicsPipelineShaders;

    fn new(core: Arc<Core>, shaders: &Self::Shaders) -> SrResult<Self> {
        Self::new_heap(
            core,
            &shaders.vertex,
            &shaders.fragment,
            shaders.color_format,
            shaders.vertex_binding,
            &shaders.vertex_attributes,
        )
    }

    fn inner(&self) -> vk::Pipeline {
        self.pipeline
    }

    // Heap-mode graphics pipelines are built with a null layout (see `new_heap`).
    fn layout(&self) -> vk::PipelineLayout {
        vk::PipelineLayout::null()
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe { self.core.device().inner().destroy_pipeline(self.pipeline, None) };
    }
}
