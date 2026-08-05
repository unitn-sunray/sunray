//! Swapchain + surface ownership for the Bevy integration.
//!
//! This is a near-verbatim port of `examples/window/{surface,swapchain}.rs`,
//! folded into one module.

use std::sync::Arc;

use crate::{MAX_FRAMES_IN_FLIGHT, error::*, vulkan_abstraction};
use ash::{khr, vk};

/// Everything `build_swapchain` produces: the swapchain plus its images, views,
/// and the extent/format/present-mode actually selected (which may differ from
/// what the caller asked for).
type SwapchainParts = (
    vk::SwapchainKHR,
    Vec<vk::Image>,
    Vec<vk::ImageView>,
    vk::Extent2D,
    vk::Format,
    vk::PresentModeKHR,
);

/// RAII wrapper that destroys the `vk::SurfaceKHR` on drop.
pub struct Surface {
    surface_instance: khr::surface::Instance,
    surface: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance, surface: vk::SurfaceKHR) -> Self {
        let surface_instance = khr::surface::Instance::load(entry, instance);
        Self {
            surface_instance,
            surface,
        }
    }

    pub fn inner(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn instance(&self) -> &khr::surface::Instance {
        &self.surface_instance
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.surface_instance.destroy_surface(self.surface, None) };
    }
}

pub struct Swapchain {
    core: Arc<vulkan_abstraction::Core>,
    swapchain_device: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    image_extent: vk::Extent2D,
    image_format: vk::Format,
    presentation_mode: vk::PresentModeKHR,
}

impl Swapchain {
    pub fn get_extent(
        window_extent: (u32, u32),
        surface_support_details: &vulkan_abstraction::SurfaceSupportDetails,
    ) -> vk::Extent2D {
        if surface_support_details.surface_capabilities.current_extent.width != u32::MAX {
            surface_support_details.surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: window_extent.0.clamp(
                    surface_support_details.surface_capabilities.min_image_extent.width,
                    surface_support_details.surface_capabilities.max_image_extent.width,
                ),
                height: window_extent.1.clamp(
                    surface_support_details.surface_capabilities.min_image_extent.height,
                    surface_support_details.surface_capabilities.max_image_extent.height,
                ),
            }
        }
    }

    /// `requested_format` / `requested_present_mode`: caller preferences. Each is
    /// honored when the surface supports it, otherwise a sensible default is
    /// chosen (BGRA8-SRGB; lowest-latency present mode available). Returns the
    /// format and present mode actually selected so the owner can preserve them
    /// across a rebuild.
    fn build_swapchain(
        core: &Arc<vulkan_abstraction::Core>,
        surface: vk::SurfaceKHR,
        window_extent: (u32, u32),
        old_swapchain: Option<vk::SwapchainKHR>,
        requested_format: Option<vk::Format>,
        requested_present_mode: Option<vk::PresentModeKHR>,
    ) -> SrResult<SwapchainParts> {
        let instance = core.instance();
        let device = core.device();
        let swapchain_device = khr::swapchain::Device::load(instance, device.inner());

        let surface_format = {
            let formats = &device.surface_support_details().surface_formats;
            // Honor a caller-requested format when the surface offers it (preferring
            // the SRGB_NONLINEAR color space); otherwise default to BGRA8-SRGB, then
            // the first supported format.
            let requested = requested_format.and_then(|fmt| {
                formats
                    .iter()
                    .find(|sf| sf.format == fmt && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .or_else(|| formats.iter().find(|sf| sf.format == fmt))
                    .copied()
            });
            if let Some(sf) = requested {
                sf
            } else {
                let bgra8_srgb_nonlinear = formats.iter().find(|surface_format| {
                    surface_format.format == vk::Format::B8G8R8A8_SRGB
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                });
                if let Some(format) = bgra8_srgb_nonlinear {
                    log::warn!("Requested format {requested_format:?} unsupported by this device; falling back to {format:?}");
                    *format
                } else {
                    let format = *formats.first().ok_or(SrError::new_custom(
                        "Physical device does not support any surface formats".to_string(),
                    ))?;
                    log::warn!("BGRA8 SRGB unsupported by this device; falling back to {format:?}");
                    format
                }
            }
        };

        let image_extent = Self::get_extent(window_extent, &device.surface_support_details());

        let present_modes = &device.surface_support_details().surface_present_modes;
        // Honor a caller-requested present mode when supported; otherwise pick the
        // lowest-latency mode available.
        // `SUNRAY_PRESENT_MODE` outranks the caller — it is a diagnostic knob.
        let present_mode = crate::utils::present_mode()
            .or(requested_present_mode)
            .filter(|pm| present_modes.contains(pm))
            .unwrap_or_else(|| {
                let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                    vk::PresentModeKHR::MAILBOX
                } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                    vk::PresentModeKHR::IMMEDIATE
                } else if present_modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
                    vk::PresentModeKHR::FIFO_RELAXED
                } else {
                    vk::PresentModeKHR::FIFO
                };
                log::warn!("Requested presentation mode {requested_present_mode:?} unsupported by this device; falling back to {present_mode:?}");
                present_mode
            });

        let swapchain = {
            let surface_capabilities = &device.surface_support_details().surface_capabilities;
            // Image count is dictated by frames in flight: MAILBOX needs an extra
            // image so there's always one free to render into while another is
            // queued for display (+2); the sequential modes need one more than the
            // frames the CPU keeps in flight (+1). Clamped to the surface's range.
            let desired_image_count = match present_mode {
                vk::PresentModeKHR::MAILBOX => MAX_FRAMES_IN_FLIGHT as u32 + 2,
                _ => MAX_FRAMES_IN_FLIGHT as u32 + 1,
            };
            let mut image_count = desired_image_count.max(surface_capabilities.min_image_count);

            if surface_capabilities.max_image_count > 0 && image_count > surface_capabilities.max_image_count {
                image_count = surface_capabilities.max_image_count;
            }

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(image_extent)
                .image_array_layers(1)
                // TODO TRANSFER_DST: the renderer blits its post-process result into the
                // swapchain image. COLOR_ATTACHMENT: needed for a future egui overlay
                // pass (dynamic rendering, load-op) — see docs/bevy_integration.md §6.
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null()));

            unsafe { swapchain_device.create_swapchain(&swapchain_create_info, None) }?
        };

        let images = unsafe { swapchain_device.get_swapchain_images(swapchain) }?;

        let image_views = images
            .iter()
            .map(|image| {
                let image_view_create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                unsafe { device.inner().create_image_view(&image_view_create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((
            swapchain,
            images,
            image_views,
            image_extent,
            surface_format.format,
            present_mode,
        ))
    }

    pub fn new(
        core: Arc<vulkan_abstraction::Core>,
        surface: vk::SurfaceKHR,
        window_extent: (u32, u32),
        format: Option<vk::Format>,
        present_mode: Option<vk::PresentModeKHR>,
    ) -> SrResult<Self> {
        let swapchain_device = khr::swapchain::Device::load(core.instance(), core.device().inner());
        let (swapchain, images, image_views, image_extent, image_format, presentation_mode) =
            Self::build_swapchain(&core, surface, window_extent, None, format, present_mode)?;

        Ok(Self {
            core,
            swapchain_device,
            swapchain,
            images,
            image_views,
            image_extent,
            image_format,
            presentation_mode,
        })
    }

    pub fn inner(&self) -> vk::SwapchainKHR {
        self.swapchain
    }
    pub fn device(&self) -> &khr::swapchain::Device {
        &self.swapchain_device
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.image_extent
    }
    pub fn format(&self) -> vk::Format {
        self.image_format
    }
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }
    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    pub fn rebuild(&mut self, surface: vk::SurfaceKHR, window_extent: (u32, u32)) -> SrResult<()> {
        for img_view in self.image_views.iter() {
            unsafe { self.core.device().inner().destroy_image_view(*img_view, None) };
        }
        self.image_views = vec![];
        self.images = vec![];

        // Preserve the format and present mode the swapchain was created with.
        let (swapchain, images, image_views, image_extent, image_format, presentation_mode) = Self::build_swapchain(
            &self.core,
            surface,
            window_extent,
            Some(self.swapchain),
            Some(self.image_format),
            Some(self.presentation_mode),
        )?;

        unsafe { self.swapchain_device.destroy_swapchain(self.swapchain, None) };

        self.swapchain = swapchain;
        self.images = images;
        self.image_views = image_views;
        self.image_extent = image_extent;
        self.image_format = image_format;
        self.presentation_mode = presentation_mode;

        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        for img_view in self.image_views.iter() {
            unsafe { self.core.device().inner().destroy_image_view(*img_view, None) };
        }
        if self.swapchain != vk::SwapchainKHR::null() {
            unsafe { self.swapchain_device.destroy_swapchain(self.swapchain, None) };
        }
    }
}

/// Swapchain + present plumbing owned by the renderer when it was constructed
/// with a surface (`new_with_surface`). Kept internal — callers drive frames
/// through [`Renderer::render_to_swapchain`] and never touch the swapchain
/// directly (except read-only via [`Renderer::swapchain`], e.g. for an overlay
/// pass that needs the image format).
pub(crate) struct SwapchainData {
    pub(crate) swapchain: Swapchain,
    pub(crate) surface: Surface,

    /// One per in-flight frame: signaled by acquire, waited by the render blit.
    pub(crate) img_acquired_sems: Vec<vulkan_abstraction::Semaphore>,
    /// One per in-flight frame: the frame-timeline value of the frame that
    /// last used this slot (0 = never used). Waited through the renderer's
    /// frame timeline before the slot's semaphore is reused.
    pub(crate) img_rendered_frames: Vec<u64>,
    /// One per swapchain image: signaled when the image is PRESENT_SRC, waited by present.
    pub(crate) ready_to_present_sems: Vec<vulkan_abstraction::Semaphore>,
    /// One pre-recorded GENERAL -> PRESENT_SRC barrier per swapchain image.
    pub(crate) present_barrier_cmd_bufs: Vec<vulkan_abstraction::CmdBuffer>, //TODO this needs to be move to a method for better rg integration

    pub(crate) frame_count: u64, //TODO need to unify the frame across the board maybe with a last frame of resize and a total frame count
}

/// Everything an external present-finalize pass (e.g. an egui overlay) needs
/// about the swapchain image of the current frame. The renderer's graph has
/// already blitted the frame into `image` and left it in `GENERAL`; the pass must
/// wait `render_done_sem >= render_done_value` (that blit's completion), draw on
/// top, leave the image in `PRESENT_SRC_KHR` and signal `ready_to_present_sem`.
/// The renderer presents right after.
#[derive(Clone, Copy)]
pub struct SwapchainFrame {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub image_index: usize,
    pub ready_to_present_sem: vk::Semaphore,
    /// Graph timeline semaphore + value the frame's in-graph blit signals; an
    /// overlay pass must wait on it before reading/drawing `image`.
    pub render_done_sem: vk::Semaphore,
    pub render_done_value: u64,
}

impl SwapchainData {
    pub(crate) fn new(
        core: &Arc<vulkan_abstraction::Core>,
        surface: Surface,
        window_extent: (u32, u32),
        format: Option<vk::Format>,
        present_mode: Option<vk::PresentModeKHR>,
    ) -> SrResult<Self> {
        let swapchain = Swapchain::new(Arc::clone(core), surface.inner(), window_extent, format, present_mode)?;

        let img_acquired_sems = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| vulkan_abstraction::Semaphore::new(Arc::clone(core)))
            .collect::<Result<Vec<_>, _>>()?;
        let img_rendered_frames = vec![0u64; MAX_FRAMES_IN_FLIGHT];
        let (present_barrier_cmd_bufs, ready_to_present_sems) = Self::build_per_image_objects(core, &swapchain)?;

        Ok(Self {
            swapchain,
            surface,
            img_acquired_sems,
            img_rendered_frames,
            ready_to_present_sems,
            present_barrier_cmd_bufs,
            frame_count: 0,
        })
    }

    /// Per-swapchain-image objects: the pre-recorded GENERAL -> PRESENT_SRC
    /// barrier command buffers and the present-wait semaphores. Rebuilt
    /// whenever the swapchain (and so its image list) is rebuilt.
    pub(crate) fn build_per_image_objects(
        //TODO this is the rg job
        core: &Arc<vulkan_abstraction::Core>,
        swapchain: &Swapchain,
    ) -> SrResult<(Vec<vulkan_abstraction::CmdBuffer>, Vec<vulkan_abstraction::Semaphore>)> {
        let present_barrier_cmd_bufs = swapchain
            .images()
            .iter()
            .map(|image| -> SrResult<vulkan_abstraction::CmdBuffer> {
                let cmd_buf = vulkan_abstraction::CmdBuffer::new(Arc::clone(core))?;
                unsafe {
                    let begin_info = vk::CommandBufferBeginInfo::default();
                    core.device().inner().begin_command_buffer(cmd_buf.inner(), &begin_info)?;
                    vulkan_abstraction::cmd_image_memory_barrier(
                        core,
                        cmd_buf.inner(),
                        *image,
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::AccessFlags2::empty(),
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::PRESENT_SRC_KHR,
                    );
                    core.device().inner().end_command_buffer(cmd_buf.inner())?;
                }
                Ok(cmd_buf)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let ready_to_present_sems = swapchain
            .images()
            .iter()
            .map(|_| vulkan_abstraction::Semaphore::new(Arc::clone(core)))
            .collect::<Result<Vec<_>, _>>()?;

        Ok((present_barrier_cmd_bufs, ready_to_present_sems))
    }
}
