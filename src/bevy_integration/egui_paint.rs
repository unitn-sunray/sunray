//! egui GPU paint backend (heap + Slang).
//!
//! Consumes [`ExtractedEgui`] and paints it onto the swapchain image, *after*
//! `Renderer::render_to_image` has blitted the ray-traced frame there. Uses:
//! - [`GraphicsPipeline::new_heap`] with the `egui.slang` vertex+fragment SPIR-V,
//! - a per-`TextureId` font/user texture (uploaded via [`Image::new_from_data`],
//!   addressed by its `sampled_slot()`),
//! - grow-only host-visible vertex/index [`RawBuffer`]s rebuilt each frame,
//! - a dynamic-rendering (load-op) pass into the swapchain image view.
//!
//! Everything is serialized by `render_to_image`'s internal `device_wait_idle`
//! and the per-image command-buffer fences, so recreating textures/buffers
//! between frames is safe without extra lifetime tracking.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use super::egui_support::ExtractedEgui;
use crate::error::{SrError, SrResult};
use crate::vulkan_abstraction::{self, Buffer, CmdBuffer, Core, GraphicsPipeline, Image, RawBuffer, Sampler};

/// GPU vertex layout (matches `egui.slang`'s `VSIn`): 20 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [u8; 4],
}

/// Push constant — matches `egui.slang`'s `EguiPC` (float2 + 2×DescriptorHandle).
#[repr(C)]
#[derive(Clone, Copy)]
struct EguiPushConstant {
    screen_size_points: [f32; 2],
    tex: [u32; 2],
    samp: [u32; 2],
}

struct EguiTexture {
    /// CPU-side RGBA backing, kept so sub-region (`pos`) deltas can patch then
    /// re-upload the whole image.
    backing: Vec<u8>,
    width: usize,
    height: usize,
    image: Image,
}

pub struct EguiPaint {
    // Declaration order = drop order. cmd_bufs first so their `Drop` waits the
    // in-flight fences before the pipeline / buffers / textures are freed.
    cmd_bufs: Vec<CmdBuffer>,
    textures: HashMap<egui::TextureId, EguiTexture>,
    vtx: Option<RawBuffer>,
    idx: Option<RawBuffer>,
    sampler: Sampler,
    pipeline: GraphicsPipeline,
    core: Arc<Core>,
}

impl EguiPaint {
    pub fn new(core: Arc<Core>, color_format: vk::Format, num_images: usize) -> SrResult<Self> {
        let vert = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/egui_vert.spirv"));
        let frag = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/egui_frag.spirv"));

        let binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<GpuVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(8),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(16),
        ];

        let pipeline = GraphicsPipeline::new_heap(Arc::clone(&core), vert, frag, color_format, binding, &attributes)?;

        let sampler = Sampler::new(
            Arc::clone(&core),
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerMipmapMode::LINEAR,
        )?;

        let cmd_bufs = (0..num_images)
            .map(|_| CmdBuffer::new(Arc::clone(&core)))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            cmd_bufs,
            textures: HashMap::new(),
            vtx: None,
            idx: None,
            sampler,
            pipeline,
            core,
        })
    }

    /// Paint the extracted egui frame onto `image` (the just-rendered swapchain
    /// image, currently in `GENERAL`), then transition it to `PRESENT_SRC` and
    /// submit, signaling `ready_sem`. Replaces the plain present barrier for the
    /// frame. The submit waits `render_done_sem >= render_done_value` — the graph
    /// timeline point at which the renderer's in-graph blit into `image` completed
    /// (frame overlap means there is no longer a `device_wait_idle` to lean on).
    #[allow(clippy::too_many_arguments)]
    pub fn paint_frame(
        &mut self,
        image: vk::Image,
        image_view: vk::ImageView,
        extent: vk::Extent2D,
        img_index: usize,
        extracted: &ExtractedEgui,
        ready_sem: vk::Semaphore,
        render_done_sem: vk::Semaphore,
        render_done_value: u64,
    ) -> SrResult<()> {
        self.apply_texture_deltas(&extracted.textures_delta)?;

        let ppp = if extracted.pixels_per_point > 0.0 {
            extracted.pixels_per_point
        } else {
            1.0
        };

        // Build one big vertex/index buffer + a per-mesh draw list.
        struct DrawCmd {
            clip: egui::Rect,
            tex: egui::TextureId,
            idx_start: u32,
            idx_count: u32,
            vtx_offset: i32,
        }
        let mut vertices: Vec<GpuVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut draws: Vec<DrawCmd> = Vec::new();
        for cp in &extracted.primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &cp.primitive {
                if mesh.indices.is_empty() {
                    continue;
                }
                let vtx_offset = vertices.len() as i32;
                let idx_start = indices.len() as u32;
                for v in &mesh.vertices {
                    vertices.push(GpuVertex {
                        pos: [v.pos.x, v.pos.y],
                        uv: [v.uv.x, v.uv.y],
                        color: v.color.to_array(),
                    });
                }
                indices.extend_from_slice(&mesh.indices);
                draws.push(DrawCmd {
                    clip: cp.clip_rect,
                    tex: mesh.texture_id,
                    idx_start,
                    idx_count: mesh.indices.len() as u32,
                    vtx_offset,
                });
            }
            // egui::epaint::Primitive::Callback is not supported (no use case here).
        }

        // Upload geometry (grow-only host-visible buffers).
        if !vertices.is_empty() {
            let bytes = (vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
            Self::ensure_buffer(
                &mut self.vtx,
                &self.core,
                bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "egui vtx",
            )?;
            let dst = self.vtx.as_mut().unwrap().map_mut::<GpuVertex>()?;
            dst[..vertices.len()].copy_from_slice(&vertices);
        }
        if !indices.is_empty() {
            let bytes = (indices.len() * std::mem::size_of::<u32>()) as u64;
            Self::ensure_buffer(
                &mut self.idx,
                &self.core,
                bytes,
                vk::BufferUsageFlags::INDEX_BUFFER,
                "egui idx",
            )?;
            let dst = self.idx.as_mut().unwrap().map_mut::<u32>()?;
            dst[..indices.len()].copy_from_slice(&indices);
        }

        // Acquire the per-image command buffer (wait its previous submission).
        let cmd = {
            let cmd_buf = &mut self.cmd_bufs[img_index];
            cmd_buf.fence_mut().wait()?;
            cmd_buf.inner()
        };

        // Record (immutable borrows of self only).
        let device = self.core.device().inner();
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd, &begin_info)?;

            // GENERAL (after the renderer's blit) -> COLOR_ATTACHMENT_OPTIMAL.
            vulkan_abstraction::cmd_image_memory_barrier(
                &self.core,
                cmd,
                image,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);
            let color_attachments = [color_attachment];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments);
            device.cmd_begin_rendering(cmd, &rendering_info);

            if !draws.is_empty() {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.inner());
                self.core.descriptor_heap().cmd_bind(cmd);

                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cmd, 0, &[viewport]);
                device.cmd_bind_vertex_buffers(cmd, 0, &[self.vtx.as_ref().unwrap().inner()], &[0]);
                device.cmd_bind_index_buffer(cmd, self.idx.as_ref().unwrap().inner(), 0, vk::IndexType::UINT32);

                let samp_slot = self.sampler.slot();
                let screen = [extent.width as f32 / ppp, extent.height as f32 / ppp];
                for d in &draws {
                    let Some(tex) = self.textures.get(&d.tex) else {
                        continue;
                    };
                    let scissor = clip_to_scissor(d.clip, ppp, extent);
                    if scissor.extent.width == 0 || scissor.extent.height == 0 {
                        continue;
                    }
                    device.cmd_set_scissor(cmd, 0, &[scissor]);

                    let pc = EguiPushConstant {
                        screen_size_points: screen,
                        tex: [tex.image.sampled_slot(), 0],
                        samp: [samp_slot, 0],
                    };
                    let push_info = vk::PushDataInfoEXT::default().offset(0).data(vk::HostAddressRangeConstEXT {
                        address: &pc as *const _ as *const std::ffi::c_void,
                        size: std::mem::size_of::<EguiPushConstant>(),
                        _marker: Default::default(),
                    });
                    self.core.descriptor_heap_device().cmd_push_data(cmd, &push_info);

                    device.cmd_draw_indexed(cmd, d.idx_count, 1, d.idx_start, d.vtx_offset, 0);
                }
            }

            device.cmd_end_rendering(cmd);

            // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC.
            vulkan_abstraction::cmd_image_memory_barrier(
                &self.core,
                cmd,
                image,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::empty(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            device.end_command_buffer(cmd)?;
        }

        let fence = self.cmd_bufs[img_index].fence_mut().submit()?;
        // Wait the graph timeline (renderer's in-graph blit into `image`), signal the
        // binary present sem `queue_present` waits on.
        let waits = [(render_done_sem, render_done_value, vk::PipelineStageFlags2::ALL_COMMANDS)];
        let signals = [(ready_sem, 0u64, vk::PipelineStageFlags2::ALL_COMMANDS)];
        self.core
            .graphics_queue()
            .submit_async_timelines(cmd, &waits, &signals, fence)?;

        Ok(())
    }

    fn ensure_buffer(
        slot: &mut Option<RawBuffer>,
        core: &Arc<Core>,
        needed: u64,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<()> {
        let big_enough = slot.as_ref().map(|b| b.byte_size() >= needed).unwrap_or(false);
        if !big_enough {
            // Grow with headroom to avoid reallocating every few frames.
            let capacity = (needed * 2).max(4096);
            *slot = Some(RawBuffer::new_aligned(
                Arc::clone(core),
                capacity,
                16,
                gpu_allocator::MemoryLocation::CpuToGpu,
                usage,
                name,
            )?);
        }
        Ok(())
    }

    fn apply_texture_deltas(&mut self, delta: &egui::TexturesDelta) -> SrResult<()> {
        for (id, image_delta) in &delta.set {
            let (rgba, w, h) = image_data_to_rgba(&image_delta.image);
            if let Some(pos) = image_delta.pos {
                let tex = self
                    .textures
                    .get_mut(id)
                    .ok_or_else(|| SrError::new_custom("egui sub-region texture update before full set".into()))?;
                let (px, py) = (pos[0], pos[1]);
                for row in 0..h {
                    let dst_start = ((py + row) * tex.width + px) * 4;
                    let src_start = row * w * 4;
                    tex.backing[dst_start..dst_start + w * 4].copy_from_slice(&rgba[src_start..src_start + w * 4]);
                }
                tex.image = Self::create_texture_image(&self.core, &tex.backing, tex.width, tex.height)?;
            } else {
                let image = Self::create_texture_image(&self.core, &rgba, w, h)?;
                self.textures.insert(
                    *id,
                    EguiTexture {
                        backing: rgba,
                        width: w,
                        height: h,
                        image,
                    },
                );
            }
        }
        for id in &delta.free {
            self.textures.remove(id);
        }
        Ok(())
    }

    fn create_texture_image(core: &Arc<Core>, rgba: &[u8], w: usize, h: usize) -> SrResult<Image> {
        Image::new_from_data(
            Arc::clone(core),
            rgba.to_vec(),
            &crate::vulkan_abstraction::image::ImageDesc {
                extent: vk::Extent3D {
                    width: w as u32,
                    height: h as u32,
                    depth: 1,
                },
                format: vk::Format::R8G8B8A8_UNORM,
                tiling: vk::ImageTiling::OPTIMAL,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                usage: vk::ImageUsageFlags::SAMPLED,
                name: "egui texture",
            },
        )
    }
}

/// egui clip rect (points) -> Vulkan scissor (physical pixels), clamped to the
/// framebuffer.
fn clip_to_scissor(clip: egui::Rect, ppp: f32, extent: vk::Extent2D) -> vk::Rect2D {
    let min_x = ((clip.min.x * ppp).floor().max(0.0) as u32).min(extent.width);
    let min_y = ((clip.min.y * ppp).floor().max(0.0) as u32).min(extent.height);
    let max_x = ((clip.max.x * ppp).ceil().max(0.0) as u32).min(extent.width);
    let max_y = ((clip.max.y * ppp).ceil().max(0.0) as u32).min(extent.height);
    vk::Rect2D {
        offset: vk::Offset2D {
            x: min_x as i32,
            y: min_y as i32,
        },
        extent: vk::Extent2D {
            width: max_x.saturating_sub(min_x),
            height: max_y.saturating_sub(min_y),
        },
    }
}

/// Flatten an egui texture delta image to tightly-packed RGBA8 + its size.
fn image_data_to_rgba(image: &egui::ImageData) -> (Vec<u8>, usize, usize) {
    match image {
        egui::ImageData::Color(c) => {
            let [w, h] = c.size;
            let mut out = Vec::with_capacity(w * h * 4);
            for p in &c.pixels {
                out.extend_from_slice(&p.to_array());
            }
            (out, w, h)
        }
    }
}
