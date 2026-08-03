pub mod camera;
pub mod error;
pub mod finello_pathtracing_pipeline;
pub mod render_graph;
pub mod scene;
pub mod shader_compiler;
pub mod utils;
pub mod vulkan_abstraction;

/// Bevy 0.19 plugin that drives this renderer from inside a Bevy `App`.
#[cfg(feature = "bevy")]
pub mod bevy_integration;

pub use crate::vulkan_abstraction::DiagnosticTool;
pub use camera::*;
use error::*;
pub use scene::*;

use crate::render_graph::graph::{ExportedTemporalResource, RenderGraph};
use crate::render_graph::pass_builder::{
    ComputeRenderPassBuilder, ComputeShaders, PassCommonDataBuilder, RayTracingShaders, RaytracingRenderPassBuilder, ShaderSource,
};
use crate::utils::{
    ENABLE_GPUAV, ENABLE_NSIGHT, ENABLE_NVIDIA_AFTERMATH, ENABLE_VALIDATION_LAYER, IS_DEBUG_BUILD, SERIALIZE_FRAMES,
    env_var_as_bool,
};
use crate::vulkan_abstraction::image::swapchain::{Surface, Swapchain};
use crate::vulkan_abstraction::swapchain::{SwapchainData, SwapchainFrame};
use crate::vulkan_abstraction::{Buffer, HostAccessibleBuffer, PostprocessPushConstant, Reservoir, ReservoirGI};
use ash::vk;
use render_graph::resource::Handle;
use std::hash::Hash;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::{collections::HashMap, rc::Rc, sync::Arc};
use vk_sync_fork as vk_sync;
use vulkan_abstraction::buffer::BufferDesc;
use vulkan_abstraction::image::ImageDesc;

//TODO finello
pub const DENOISE_PASSES: u32 = 4;
//TODO finello
pub const EXPOSURE: f32 = 1.0;

/// Key identifying a GPU asset (BLAS or image) inside the renderer's
/// `ResourceManager`. `group` ties together every asset created by one
/// `load_scene` call so a whole scene can be deallocated in bulk (see
/// [`Renderer::unload_scene`]); `index` is unique within the group.
///
/// Scene loading generates these and converts them into the renderer's actual
/// key type via `K: From<ResourceKey>` — use `ResourceKey` itself when no
/// third party (e.g. Bevy's asset system) supplies its own ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceKey {
    pub group: u64,
    pub index: u64,
}

/// The number of concurrent frames that are processed (both by CPU and GPU).
///
/// Apparently 2 is the most common choice. Empirically it seems like the performance doesn't really
/// get any better with a higher number, but it does get measurably worse with only 1.
///
/// TODO this feature is actually not doing what it is supposed and needs to be reworked, do not go over 2 I think it will crash
/// the render graph is incapable of starting a second frame with a current frame still ongoing
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

//TODO add a list of callbacks to call at the end of frames for cleanup or at start for setup
//TODO deferred deallocation for buffers and acceleration structures
//TODO validate max_frame_in_flight against the swapchain

/// Where [`Renderer::render`] sends the finished (post-processed) frame. The
/// graph itself does the blit into `image` (see [`RenderGraph::run_present`]);
/// the variant only decides the target's final layout and who signals present.
pub enum FrameOutput<'a> {
    /// Swapchain image, renderer finishes it: blit → `PRESENT_SRC`, signal
    /// `present_sem` (waited by `queue_present`). Waits `acquire_sem`.
    Present {
        image: &'a vulkan_abstraction::Image,
        acquire_sem: vk::Semaphore,
        present_sem: vk::Semaphore,
    },
    /// Swapchain image, an overlay finishes it: blit, leave `GENERAL`. The
    /// caller's overlay pass (e.g. egui) draws on top, transitions to
    /// `PRESENT_SRC` and signals present itself. Waits `acquire_sem`.
    PresentOverlay {
        image: &'a vulkan_abstraction::Image,
        acquire_sem: vk::Semaphore,
    },
    /// Host-visible offscreen image (readback): blit, leave `GENERAL`, no present.
    Offscreen { image: &'a vulkan_abstraction::Image },
}

pub type CreateSurfaceFn = dyn Fn(&ash::Entry, &ash::Instance) -> SrResult<vk::SurfaceKHR>;

pub struct Renderer<K: Hash + Eq + Copy + 'static = ResourceKey> {
    /// The one intermediate image the renderer still owns: the post-process
    /// output. The graph writes it (imported), then blits it into the frame's
    /// output target. A single backing suffices — the graph serializes frame N's
    /// body after N-1 (`graph_timeline >= N-1`), so it's never written while a
    /// prior frame's blit still reads it.
    /// ponytail: single backing, per-in-flight-slot only if real overlap lands.
    postprocess_result_image: Arc<vulkan_abstraction::Image>,

    resource_manager: vulkan_abstraction::ResourceManager<K>,

    /// Swapchain + present plumbing, present when constructed with a surface
    /// (`new_with_surface`) — given at startup, owned internally.
    swapchain_data: Option<SwapchainData>,

    /// Next `ResourceKey::group`; one group per `load_scene` call.
    next_group: u64,
    /// Group → every key created for that scene load, for bulk deallocation.
    scene_groups: HashMap<u64, Vec<K>>,

    // Heap-mode ray-tracing SPIR-V (one blob per stage).
    // `RaytracingRenderPassBuilder::generate_render`, which interns the pipeline
    // and SBT in the graph's persistent cache (built once, reused across frame
    // rebuilds). RIS and final share miss/closest-hit/any-hit and differ only in
    // the ray-gen stage, so they intern as two distinct cache entries.
    /// Ray-gen for the RIS pass: finds the best candidates per pixel without tracing many rays.
    ray_gen_ris_spirv: &'static [u8],
    /// Ray-gen for the final pass: traces rays based on the reservoirs the RIS pass produced.
    ray_gen_final_spirv: &'static [u8],

    ray_miss_spirv: &'static [u8],
    closest_hit_spirv: &'static [u8],
    any_hit_spirv: &'static [u8],
    ///The first pass after raytracing merges the previous frame on the next one to reduce bias
    temporal_accumulation_spirv: &'static [u8],
    ///The denoise pass is run after the temporal accumulation to reduce noise even more (a-trous filter)
    denoise_spirv: &'static [u8],
    ///An extra pass to handle post-processing like exposure and color correction. Should be mathematically easy to calculate
    postprocess_spirv: &'static [u8],

    // this is about the frame being worked on by the cpu
    image_extent: vk::Extent3D,
    image_format: vk::Format,

    blue_noise_image: vulkan_abstraction::Image,
    blue_noise_sampler: vulkan_abstraction::Sampler,

    core: Rc<vulkan_abstraction::Core>,

    //TODO finni all of this params are pipeline-specific temporal (cross-frame) stuff. They now
    // live as temporal resources owned by the render graph (created once, re-registered each
    // rebuild, memory preserved across frames). When the path-tracing pipeline is extracted into
    // its own file and the renderer becomes pipelineless, these tokens move out with it.
    /// Ping-pong accumulation images for temporal accumulation. The graph owns
    /// the backing memory; this is just the exported token re-registered each
    /// frame. Ping-pong selection is by [`Self::relative_frame_count`] parity.
    accumulation_temporal: ExportedTemporalResource<vulkan_abstraction::Image>,
    /// Ping-pong a-trous denoise images (same ownership contract as
    /// `accumulation_temporal`).
    denoising_temporal: ExportedTemporalResource<vulkan_abstraction::Image>,
    ///this is used for temporal accumulation, there is an absolute frame counter in the core
    pub relative_frame_count: u32,

    /// Ping-pong reservoir buffers for ReSTIR. The graph owns the backing memory
    /// (a temporal resource): the same buffers are re-registered each frame for
    /// hazard tracking — the RIS pass writes them and the final pass reads them,
    /// so the graph emits the reservoir hand-off barrier between the two RT passes
    /// automatically. The shader still addresses them by device-address (see
    /// `RaytracingHeapPushConstant::reservoirs`, filled from
    /// [`RenderGraph::temporal_buffer_addresses`]); the graph import only governs
    /// synchronization.
    reservoir_temporal: ExportedTemporalResource<vulkan_abstraction::RawBuffer>,
    /// Ping-pong pair of GI reservoir buffers for ReSTIR GI (Ouyang 2021); same
    /// ownership contract as `reservoir_temporal`, storing surface samples (x2)
    /// instead of light samples.
    reservoir_gi_temporal: ExportedTemporalResource<vulkan_abstraction::RawBuffer>,

    prev_view_proj: nalgebra::Matrix4<f32>, //used to calculate motion vectors

    /// Per-frame-in-flight camera-matrices UBOs, indexed by the frame's slot
    /// (`absolute_frame % MAX_FRAMES_IN_FLIGHT`). The RT shaders reach these by
    /// device address baked into the push constant, so the buffer (and thus its
    /// address) must stay registered while the frame's GPU work runs. Recreating
    /// a fresh buffer every frame churned addresses that the driver/GPU-AV would
    /// race against an in-flight overlapping frame's read; a persistent slot pool
    /// keeps the address stable and is only *written* (never destroyed), the write
    /// gated by `wait_for_slot_reuse`. See the reservoir/temporal buffers for the
    /// same pattern.
    matrices_pool: Vec<vulkan_abstraction::UniformBuffer<CameraMatrices>>,

    /// Persistent render graph.
    /// Re-populated each frame (passes / imports change because the ping-pong
    /// indices and per-frame descriptor sets do), but the underlying command
    /// buffer is reused across `compile` calls.
    pub render_graph: RenderGraph,

    /// Last frame value the watcher thread observed on the graph timeline (the
    /// single frame-completion timeline — see [`RenderGraph::wait_graph_timeline`]).
    completed_frame: Arc<AtomicU64>,
    /// Tells the watcher thread to exit (set in `Drop`).
    frame_watcher_shutdown: Arc<AtomicBool>,
    /// Thread (spawned at construction) that waits the frame timeline and
    /// publishes `completed_frame`. The callbacks themselves run on the render
    /// thread (they capture `Rc`-based GPU resources, which are `!Send`) —
    /// `render` drains the ones whose frame the watcher reported complete.
    frame_watcher: Option<std::thread::JoinHandle<()>>,

    //TODO these would love #![feature(unboxed_closures)]
    //these are ordered,the u64 is the absolute frame on which to execute and the actual callback
    start_of_frame_callbacks: Vec<(u64, Box<dyn FnOnce()>)>,
    /// Persistent (FnMut) callbacks invoked on every `resize`.
    resize_callbacks: Vec<Box<dyn FnMut((u32, u32))>>,
    /// Run on the render thread once the tagged frame has *completed on the
    /// GPU* (per `completed_frame`). The per-frame CpuToGpu buffers `render`
    /// creates are deallocated through here.
    end_of_frame_callbacks: Vec<(u64, Box<dyn FnOnce(&mut Renderer<K>)>)>,
}

/// Per-frame GPU inputs of the unified graph that live in frame-local buffers
/// (created on the spot in `render`, deferred-freed via the end-of-frame
/// callbacks): the camera matrices UBO address and the heap slots of the flat
/// transform / emissive indirection buffers.
struct FrameGpuData {
    matrices_address: vk::DeviceAddress,
    entity_transforms_slot: u32,
    emissive_indirection_slot: u32,
}
// `K: 'static` propagated from `ResourceManager` (its deferred frame work is
// stored as boxed callbacks).
impl<K: Hash + Eq + Copy + 'static> Renderer<K> {
    pub fn new(image_extent: (u32, u32), image_format: vk::Format) -> SrResult<Self> {
        Self::new_impl(image_extent, image_format, &[], None)
    }

    // It's necessary to pass a fn to create the surface, because it depends on instance, device depends on it (if present), and both device and
    // instance are created and owned inside Renderer (in Core) so this was deemed a good approach to allow the user to build their own surface.
    // The swapchain for that surface is created here too and kept internal — drive it with `render_to_swapchain`.
    pub fn new_with_surface(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: &CreateSurfaceFn,
    ) -> SrResult<Self> {
        Self::new_impl(image_extent, image_format, instance_exts, Some(create_surface))
    }

    fn new_impl(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: Option<&CreateSurfaceFn>,
    ) -> SrResult<Self> {
        let with_validation_layer = env_var_as_bool(ENABLE_VALIDATION_LAYER).unwrap_or(IS_DEBUG_BUILD);
        let with_gpuav = env_var_as_bool(ENABLE_GPUAV).unwrap_or(false);
        // Select the GPU diagnostic backend from env. `SUNRAY_ENABLE_NSIGHT` forces
        // VK_EXT_debug_utils on (even without validation) and emits per-pass
        // command-buffer labels + object names so an Nsight Graphics capture is
        // readable and can inspect the descriptor heap (which RenderDoc can't).
        // `SUNRAY_ENABLE_NVIDIA_AFTERMATH` (legacy) keeps the crash-dump path.
        let diagnostics = if env_var_as_bool(ENABLE_NSIGHT).unwrap_or(false) {
            DiagnosticTool::NvidiaNsightGraphics
        } else if env_var_as_bool(ENABLE_NVIDIA_AFTERMATH).unwrap_or(false) {
            DiagnosticTool::NvidiaAftermath
        } else {
            DiagnosticTool::None
        };

        let (core, surface) = vulkan_abstraction::Core::new_with_surface(
            with_validation_layer,
            with_gpuav,
            diagnostics,
            image_format,
            instance_exts,
            create_surface,
        )?;

        let core = Rc::new(core);

        let window_extent = image_extent;
        let image_extent = utils::tuple_to_extent3d(image_extent);

        //must be filled by loading a scene
        let resource_manager = vulkan_abstraction::ResourceManager::new_empty(Rc::clone(&core))?;

        let ray_gen_ris_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_gen_ris.spirv"));
        let ray_gen_final_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_gen_final.spirv"));
        let ray_miss_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_miss.spirv"));
        let closest_hit_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/closest_hit.spirv"));
        let any_hit_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/any_hit.spirv"));
        let denoise_spirv = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/denoise.spirv"));
        let postprocess_spirv = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/postprocess.spirv"));
        let temporal_accumulation_spirv = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/temporal_accumulation.spirv"));

        let postprocess_result_image = Self::create_postprocess_result_image(&core, image_extent)?;

        // The cross-frame ping-pong resources (accumulation / denoise images,
        // ReSTIR reservoir buffers) are now temporal resources owned by the
        // render graph — created just below, once the graph exists.

        let blue_noise_bytes = include_bytes!("finello_pathtracing_pipeline/util_files/noise.png");
        let blue_noise_img = image::load_from_memory(blue_noise_bytes).unwrap().to_rgba8();
        let (noise_width, noise_height) = blue_noise_img.dimensions();
        let blue_noise_data = blue_noise_img.into_raw();

        let blue_noise_image = vulkan_abstraction::Image::new_from_data(
            Rc::clone(&core),
            blue_noise_data,
            vk::Extent3D {
                width: noise_width,
                height: noise_height,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::ImageUsageFlags::SAMPLED,
            "blue noise texture",
        )?;

        let blue_noise_sampler = vulkan_abstraction::Sampler::new(
            Rc::clone(&core),
            vk::Filter::NEAREST,
            vk::Filter::NEAREST,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerMipmapMode::NEAREST,
        )?;

        let mut render_graph = RenderGraph::new(Rc::clone(&core))?;

        // Per-slot camera-matrices UBOs (stable device addresses; see field doc).
        let matrices_pool = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| vulkan_abstraction::UniformBuffer::<CameraMatrices>::new(Rc::clone(&core), 1))
            .collect::<SrResult<Vec<_>>>()?;

        // Temporal (cross-frame) resources: the graph owns the backing memory and
        // preserves it across the per-frame rebuild, so each holds its history.
        let num_pixels = (image_extent.width * image_extent.height) as usize;
        let accumulation_temporal =
            render_graph.create_temporal_resource(Self::temporal_image_desc("Accumulation", image_extent))?;
        let denoising_temporal = render_graph.create_temporal_resource(Self::temporal_image_desc("Denoise", image_extent))?;
        let reservoir_temporal = render_graph.create_temporal_resource(Self::reservoir_buffer_desc::<Reservoir>(
            "ReSTIR Reservoir Buffer",
            num_pixels,
        ))?;
        let reservoir_gi_temporal = render_graph.create_temporal_resource(Self::reservoir_buffer_desc::<ReservoirGI>(
            "ReSTIR GI Reservoir Buffer",
            num_pixels,
        ))?;

        // Frame completion is tracked on the render graph's timeline (signaled with
        // the absolute frame count by each frame's submission). It starts at 0 =
        // "frame 0 (nothing) done"; the renderer no longer keeps a second timeline.
        let completed_frame = Arc::new(AtomicU64::new(0));
        let frame_watcher_shutdown = Arc::new(AtomicBool::new(false));

        // Watcher thread: waits the timeline value-by-value and publishes the
        // last completed frame. It only *observes* — the end-of-frame callbacks
        // run on the render thread because they capture `Rc`-based (!Send) GPU
        // resources. `ash::Device` is Send + Sync, so the raw waits are fine here.
        let frame_watcher = {
            let device = core.device().inner().clone();
            let timeline = render_graph.graph_timeline_inner();
            let completed = Arc::clone(&completed_frame);
            let shutdown = Arc::clone(&frame_watcher_shutdown);
            std::thread::Builder::new()
                .name("sunray-frame-watcher".to_string())
                .spawn(move || {
                    let mut next_frame = 1u64;
                    while !shutdown.load(Ordering::Acquire) {
                        let semaphores = [timeline];
                        let values = [next_frame];
                        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
                        // Short timeout so shutdown is honored promptly.
                        match unsafe {
                            device.wait_semaphores(&wait_info, 100_000_000 /* 100ms */)
                        } {
                            Ok(()) => {
                                completed.store(next_frame, Ordering::Release);
                                next_frame += 1;
                            }
                            Err(vk::Result::TIMEOUT) => continue,
                            Err(e) => {
                                log::error!("sunray frame watcher: vkWaitSemaphores failed with {e:?}; exiting");
                                break;
                            }
                        }
                    }
                })
                .map_err(|e| SrError::new_custom(format!("failed to spawn frame watcher thread: {e}")))?
        };

        // The swapchain abstraction is owned internally and built at startup
        // from the surface the caller's closure created.
        let swapchain_data = match surface {
            Some(surface_khr) => {
                let surface = Surface::new(core.entry(), core.instance(), surface_khr);
                // Default format / present mode (None → surface-preferred). The
                // plumbing exists to pin them from here if a caller needs to.
                Some(SwapchainData::new(&core, surface, window_extent, None, None)?)
            }
            None => None,
        };

        let renderer = Self {
            postprocess_result_image,

            swapchain_data,
            next_group: 0,
            scene_groups: HashMap::new(),

            render_graph,

            completed_frame,
            frame_watcher_shutdown,
            frame_watcher: Some(frame_watcher),

            reservoir_temporal,

            ray_gen_ris_spirv,
            ray_gen_final_spirv,
            ray_miss_spirv,
            closest_hit_spirv,
            any_hit_spirv,
            denoise_spirv,
            temporal_accumulation_spirv,
            postprocess_spirv,

            prev_view_proj: nalgebra::zero(),
            matrices_pool,

            image_extent,
            image_format,

            accumulation_temporal,
            denoising_temporal,
            relative_frame_count: 0,

            blue_noise_image,
            blue_noise_sampler,

            resource_manager,
            reservoir_gi_temporal,

            core,

            start_of_frame_callbacks: vec![],
            resize_callbacks: vec![],
            end_of_frame_callbacks: vec![],
        };

        // The temporal images are imported into the graph, which never transitions
        // imported resources; bring their freshly-created (UNDEFINED) backings into
        // GENERAL so the first compute pass that touches them is valid.
        renderer.init_temporal_images_to_general()?;

        Ok(renderer)
    }

    /// The renderer-owned post-process result image (graph writes it, then blits
    /// it into the frame's output target). `R8G8B8A8_UNORM` storage image, sized to
    /// the render extent — recreated on resize.
    fn create_postprocess_result_image(
        core: &Rc<vulkan_abstraction::Core>,
        extent: vk::Extent3D,
    ) -> SrResult<Arc<vulkan_abstraction::Image>> {
        let image = vulkan_abstraction::Image::new(
            Rc::clone(core),
            extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            "sunray (internal, pre-blit) postprocess result image",
        )?;

        // The graph's postprocess pass writes it through a storage descriptor
        // (GENERAL), but it's an *imported* resource so the graph's own
        // created-resource init transition doesn't cover it: discard-init to GENERAL.
        {
            let device = core.device().inner();
            let mut setup = vulkan_abstraction::CmdBuffer::new(Rc::clone(core))?;
            unsafe {
                let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device.begin_command_buffer(setup.inner(), &begin)?;
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.inner())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                let dep = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
                device.cmd_pipeline_barrier2(setup.inner(), &dep);
                device.end_command_buffer(setup.inner())?;
                let fence = setup.fence_mut().submit()?;
                core.graphics_queue().submit_async(setup.inner(), &[], &[], &[], fence)?;
                setup.fence_mut().wait()?;
            }
        }

        Ok(Arc::new(image))
    }

    /// Descriptor for a temporal ping-pong image (accumulation / denoise). The
    /// render graph allocates `MAX_FRAMES_IN_FLIGHT` backings from this.
    //TODO finni: pipeline-specific; moves with the temporal resources when the
    // path-tracing pipeline is extracted and the renderer becomes pipelineless.
    fn temporal_image_desc(name: &'static str, extent: vk::Extent3D) -> ImageDesc {
        ImageDesc {
            extent,
            format: vk::Format::B10G11R11_UFLOAT_PACK32,
            tiling: vk::ImageTiling::OPTIMAL,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            name,
        }
    }

    /// Descriptor for a ReSTIR reservoir buffer holding `num_pixels` elements of
    /// `T`, addressed by device-address in the shader. The render graph allocates
    /// `MAX_FRAMES_IN_FLIGHT` ping-pong backings from this.
    //TODO finni: pipeline-specific; moves with the temporal resources when the
    // path-tracing pipeline is extracted and the renderer becomes pipelineless.
    fn reservoir_buffer_desc<T>(name: &'static str, num_pixels: usize) -> BufferDesc {
        BufferDesc {
            byte_size: (num_pixels * size_of::<T>()) as vk::DeviceSize,
            alignment: 1,
            memory_location: gpu_allocator::MemoryLocation::GpuOnly,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            name,
        }
    }

    /// Transition the freshly-created (UNDEFINED) backings of the temporal
    /// accumulation / denoise images into GENERAL with a one-time submit. The
    /// graph imports these resources and never transitions imported memory
    /// itself, so without this the first compute pass would touch a storage image
    /// in the wrong layout. Run after (re)creating the temporal images.
    //TODO finni: pipeline-specific; moves with the temporal resources when the
    // path-tracing pipeline is extracted and the renderer becomes pipelineless.
    fn init_temporal_images_to_general(&self) -> SrResult<()> {
        let images: Vec<Arc<vulkan_abstraction::Image>> = self
            .render_graph
            .temporal_image_backings(&self.accumulation_temporal)
            .into_iter()
            .chain(self.render_graph.temporal_image_backings(&self.denoising_temporal))
            .collect();

        let device = self.core.device().inner();
        let mut setup_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(setup_cmd_buf.inner(), &begin_info)?;
            let barriers: Vec<vk::ImageMemoryBarrier2> = images
                .iter()
                .map(|image| {
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::empty())
                        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image.inner())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                })
                .collect();
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            device.cmd_pipeline_barrier2(setup_cmd_buf.inner(), &dep_info);
            device.end_command_buffer(setup_cmd_buf.inner())?;
            let fence = setup_cmd_buf.fence_mut().submit()?;
            self.core
                .graphics_queue()
                .submit_async(setup_cmd_buf.inner(), &[], &[], &[], fence)?;
            setup_cmd_buf.fence_mut().wait()?;
        }
        Ok(())
    }
    // ─── Frame / resize callbacks ───────────────────────────────────────────

    /// Schedule `callback` to run on the CPU at the start of the next frame
    /// (before any per-frame upload).
    pub fn add_start_of_frame_callback(&mut self, callback: impl FnOnce() + 'static) {
        let next_frame = *self.core.absolute_frame_count.borrow() as u64 + 1;
        self.start_of_frame_callbacks.push((next_frame, Box::new(callback)));
    }

    /// Schedule `callback` to run once the next rendered frame has *completed
    /// on the GPU* (per the frame timeline). This is the deferred-deallocation
    /// hook: dropping a GPU resource inside the callback is safe because the
    /// frame that used it is provably done.
    pub fn add_end_of_frame_callback(&mut self, callback: impl FnOnce(&mut Renderer<K>) + 'static) {
        let next_frame = *self.core.absolute_frame_count.borrow() as u64 + 1;
        self.end_of_frame_callbacks.push((next_frame, Box::new(callback)));
    }

    /// Register a persistent callback invoked on every [`Self::resize`].
    pub fn add_resize_callback(&mut self, callback: impl FnMut((u32, u32)) + 'static) {
        self.resize_callbacks.push(Box::new(callback));
    }

    /// Run the start-of-frame callbacks due for `upcoming_frame` (kept ordered
    /// by frame).
    fn run_start_of_frame_callbacks(&mut self, upcoming_frame: u64) {
        let mut i = 0;
        while i < self.start_of_frame_callbacks.len() {
            if self.start_of_frame_callbacks[i].0 <= upcoming_frame {
                let (_, callback) = self.start_of_frame_callbacks.remove(i);
                callback();
            } else {
                i += 1;
            }
        }
    }

    /// Run the end-of-frame callbacks whose frame the watcher thread reported
    /// complete on the GPU (deferred deallocation of per-frame resources).
    fn run_due_end_of_frame_callbacks(&mut self) {
        let completed = self.completed_frame.load(Ordering::Acquire);
        let mut i = 0;
        while i < self.end_of_frame_callbacks.len() {
            if self.end_of_frame_callbacks[i].0 <= completed {
                let (_, callback) = self.end_of_frame_callbacks.remove(i);
                callback(self);
            } else {
                i += 1;
            }
        }
    }

    pub fn resize(&mut self, image_extent: (u32, u32)) -> SrResult<()> {
        //TODO currently it is doing a wait idle for simplicity but the tools for deferred deallocation are there
        self.resize_internal_images(image_extent)?;
        self.resize_swapchain(image_extent)?;

        for callback in self.resize_callbacks.iter_mut() {
            callback(image_extent);
        }

        Ok(())
    }

    fn resize_internal_images(&mut self, image_extent: (u32, u32)) -> SrResult<()> {
        let new_extent = utils::tuple_to_extent3d(image_extent);
        if new_extent == self.image_extent {
            return Ok(());
        }
        // Drop the in-flight references to images before we destroy them — without
        // this, a resize that arrives while the previous frame's fence hasn't signaled
        // tears down images/descriptor-heap slots the GPU is still reading.
        unsafe { self.core.device().inner().device_wait_idle() }?;

        let num_pixels = (new_extent.width * new_extent.height) as usize;
        self.image_extent = new_extent;

        // Recreate the post-process result image at the new size (the GPU is idle).
        self.postprocess_result_image = Self::create_postprocess_result_image(&self.core, new_extent)?;

        // Recreate the temporal resources at the new dimensions. The graph owns
        // their backing memory, so (the GPU is idle from the wait above) drop the
        // old backings and re-export fresh tokens; the per-frame rebuild
        // re-registers them. Recreate in the same order as construction.
        self.render_graph.clear_temporal_resources();
        self.accumulation_temporal = self
            .render_graph
            .create_temporal_resource(Self::temporal_image_desc("Accumulation", new_extent))?;
        self.denoising_temporal = self
            .render_graph
            .create_temporal_resource(Self::temporal_image_desc("Denoise", new_extent))?;
        self.reservoir_temporal = self
            .render_graph
            .create_temporal_resource(Self::reservoir_buffer_desc::<Reservoir>(
                "ReSTIR Reservoir Buffer",
                num_pixels,
            ))?;
        self.reservoir_gi_temporal = self
            .render_graph
            .create_temporal_resource(Self::reservoir_buffer_desc::<ReservoirGI>(
                "ReSTIR GI Reservoir Buffer",
                num_pixels,
            ))?;

        // Bring the freshly-created (UNDEFINED) accumulation / denoise backings
        // into GENERAL, exactly as the initial construction does.
        self.init_temporal_images_to_general()?;

        self.relative_frame_count = 0;

        Ok(())
    }

    /// Rebuild the internal swapchain (and everything tied to its images) when
    /// the surface extent changed. No-op without a surface.
    fn resize_swapchain(&mut self, window_extent: (u32, u32)) -> SrResult<()> {
        //TODO this can be reworked after the swapchain changes
        let Some(sc) = self.swapchain_data.as_mut() else {
            return Ok(());
        };

        self.core
            .device()
            .update_surface_support_details(sc.surface.inner(), sc.surface.instance());
        let new_extent = Swapchain::get_extent(window_extent, &self.core.device().surface_support_details());
        if sc.swapchain.extent() == new_extent {
            return Ok(());
        }

        unsafe { self.core.device().inner().device_wait_idle() }?;

        let surface_khr = sc.surface.inner();
        sc.swapchain.rebuild(surface_khr, window_extent)?;
        let (present_barrier_cmd_bufs, ready_to_present_sems) =
            SwapchainData::build_per_image_objects(&self.core, &sc.swapchain)?;
        sc.present_barrier_cmd_bufs = present_barrier_cmd_bufs;
        sc.ready_to_present_sems = ready_to_present_sems;

        Ok(())
    }

    /// Load a glTF file's default scene. See [`Self::load_scene`] for the
    /// return contract.
    pub fn load_gltf(&mut self, path: impl AsRef<Path>) -> SrResult<(u64, Vec<(K, Vec<vk::TransformMatrixKHR>)>)>
    where
        K: From<ResourceKey>,
    {
        let gltf = vulkan_abstraction::gltf::Gltf::new(Rc::clone(&self.core), path)?;
        let (default_scene, scene_data) = gltf.create_default_scene()?;
        self.load_scene(&default_scene, scene_data)
    }

    /// Load a scene's assets into the resource manager. Returns the asset
    /// group index (usable with [`Self::unload_scene`] to free everything this
    /// call created in bulk) and the scene's instances as the
    /// `(blas key, world transforms)` vector. The instance list is *not*
    /// retained anywhere — the caller owns it, mutates it, and passes it to
    /// [`Self::render`] / [`Self::render_to_swapchain`] every frame.
    pub fn load_scene(&mut self, scene: &Scene, scene_data: SceneData) -> SrResult<(u64, Vec<(K, Vec<vk::TransformMatrixKHR>)>)>
    where
        K: From<ResourceKey>,
    {
        // Wait for all in-flight GPU work before invalidating descriptor sets that reference
        // buffers which will be reallocated (e.g. emissive_indirection_gpu).
        unsafe { self.core.device().inner().device_wait_idle() }?;

        let group = self.next_group;
        self.next_group += 1;

        let LoadedScene {
            blases,
            instances,
            textures,
            sampler_descs,
            images,
        } = scene.load_into_gpu(&self.core, scene_data)?;

        let mut group_keys: Vec<K> = Vec::new();
        let mut next_index = 0u64;
        let mut make_key = || {
            let key = K::from(ResourceKey {
                group,
                index: next_index,
            });
            next_index += 1;
            group_keys.push(key);
            key
        };

        let blas_keys = self
            .resource_manager
            .add_scene_assets(blases, textures, sampler_descs, images, &mut make_key)?;
        self.scene_groups.insert(group, group_keys);

        // Group the per-instance transforms by BLAS key, preserving order.
        let mut grouped: Vec<(K, Vec<vk::TransformMatrixKHR>)> = blas_keys.iter().map(|&k| (k, Vec::new())).collect();
        for (blas_index, transform) in instances {
            grouped[blas_index].1.push(transform);
        }

        Ok((group, grouped))
    }

    /// Free every asset created by the `load_scene` call that returned `group`.
    /// Allows loading a scene repeatedly without leaking GPU memory. Instances
    /// referencing the freed keys must no longer be passed to `render`.
    pub fn unload_scene(&mut self, group: u64) -> SrResult<()> {
        unsafe { self.core.device().inner().device_wait_idle() }?;
        if let Some(keys) = self.scene_groups.remove(&group) {
            for key in &keys {
                self.resource_manager.remove(key);
            }
        }
        Ok(())
    }

    /// Build a BLAS from raw triangle-list mesh data at **runtime** and
    /// register it under the caller-supplied `key` — the runtime equivalent of
    /// one scene BLAS, with no glTF involved (the Bevy integration uses this to
    /// turn `Mesh` assets into BLASes on the fly). Texture indices in
    /// `material` cannot be resolved here (no image set accompanies the mesh),
    /// so all texture references are treated as absent; the scalar/color
    /// factors still apply.
    ///
    /// Emissive triangles for NEE are derived from the index list when the
    /// material's emission (`emissive_factor * emissive_strength`) is non-zero.
    /// The mesh is renderable by instances passed to [`Self::render`] from the
    /// upcoming frame on (its mesh-info arena copy is flushed by the
    /// start-of-frame callback the copy queue schedules); no GPU wait is
    /// needed to add it.
    pub fn load_mesh(
        &mut self,
        key: K,
        vertices: &[vulkan_abstraction::gltf::Vertex],
        indices: &[u32],
        material: &vulkan_abstraction::gltf::Material,
    ) -> SrResult<()> {
        if self.resource_manager.contains(&key) {
            return Err(SrError::new_custom(
                "load_mesh: an asset is already registered under this key".to_string(),
            ));
        }
        if vertices.is_empty() || indices.is_empty() || !indices.len().is_multiple_of(3) {
            return Err(SrError::new_custom(format!(
                "load_mesh: invalid mesh ({} vertices, {} indices — need non-empty vertices and a triangle-list index count)",
                vertices.len(),
                indices.len()
            )));
        }
        if let Some(&max_index) = indices.iter().max()
            && max_index as usize >= vertices.len()
        {
            return Err(SrError::new_custom(format!(
                "load_mesh: index {max_index} out of range for {} vertices",
                vertices.len()
            )));
        }

        let emission = [
            material.emissive_factor[0] * material.emissive_strength,
            material.emissive_factor[1] * material.emissive_strength,
            material.emissive_factor[2] * material.emissive_strength,
            0.0,
        ];
        let emissive_triangles: Vec<vulkan_abstraction::gltf::EmissiveTriangle> = if emission[..3].iter().any(|&c| c > 0.0) {
            indices
                .chunks_exact(3)
                .map(|tri| {
                    // `Vertex` is repr(packed): copy the positions out, no refs.
                    let p0 = vertices[tri[0] as usize].position;
                    let p1 = vertices[tri[1] as usize].position;
                    let p2 = vertices[tri[2] as usize].position;
                    vulkan_abstraction::gltf::EmissiveTriangle {
                        v0: [p0[0], p0[1], p0[2], 0.0],
                        v1: [p1[0], p1[1], p1[2], 0.0],
                        v2: [p2[0], p2[1], p2[2], 0.0],
                        emission,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas_from_data(Rc::clone(&self.core), vertices)?;
        let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas_from_data(Rc::clone(&self.core), indices)?;
        // Deferred build: the BLAS resource (and its device address) exists now, so
        // instances can reference it immediately, but the actual
        // `vkCmdBuildAccelerationStructures` is recorded into the next frame's render
        // graph by `ResourceManager::queue_blas_builds`. No GPU wait needed to add it.
        let (blas, build_job) = vulkan_abstraction::Blas::new_deferred(
            Rc::clone(&self.core),
            vertex_buffer,
            index_buffer,
            vulkan_abstraction::BuildType::Static,
        )?;

        // No image set accompanies a runtime mesh: every texture reference
        // resolves to "absent" (NULL slots ignored by the shader).
        let resolve = |_: Option<usize>| {
            (
                vulkan_abstraction::Material::NULL_TEXTURE_INDEX,
                vulkan_abstraction::Material::NULL_TEXTURE_INDEX,
            )
        };
        let gpu_material = vulkan_abstraction::Material::new(material, &resolve);

        self.resource_manager.add_blas(key, blas, gpu_material, &emissive_triangles)?;
        // Stash the build job for the next frame's graph to record.
        self.resource_manager.queue_blas_build_job(key, build_job);
        Ok(())
    }

    /// Free the runtime-loaded mesh registered under `key` by [`Self::load_mesh`].
    /// Instances referencing the key must no longer be passed to `render`.
    ///
    /// The BLAS drop (and arena-slot free) is deferred to an end-of-frame callback
    /// tagged `next_frame + MAX_FRAMES_IN_FLIGHT`, so no device-wide idle wait is
    /// needed — the callback fires once the frame timeline proves every in-flight
    /// frame that could reference it has completed on the GPU (drained by
    /// [`Self::run_due_end_of_frame_callbacks`]). The caller must stop passing `key`
    /// as an instance from now on.
    pub fn unload_mesh(&mut self, key: &K) -> SrResult<()> {
        let key = *key;
        let due = *self.core.absolute_frame_count.borrow() as u64 + 1 + MAX_FRAMES_IN_FLIGHT as u64;
        self.end_of_frame_callbacks.push((
            due,
            Box::new(move |renderer: &mut Renderer<K>| renderer.resource_manager.remove(&key)),
        ));
        Ok(())
    }

    /// Render to dst_image. All per-frame inputs are parameters: the camera and
    /// the instance list (`(blas key, world transforms of its instances)` —
    /// keys come from scene loading). The user may also pass a Semaphore which the user should signal when the image is
    /// ready to be written to (for example after being acquired from a swapchain).
    ///
    /// Returns the frame's **absolute frame number**: the graph timeline is
    /// signaled with it when the frame's GPU work (including the in-graph blit)
    /// completes, so "wait for this render to end" is `wait_frame(value)` — there
    /// is no per-frame fence to track.
    ///
    /// The graph itself blits the post-process result into `output`'s image (see
    /// [`RenderGraph::run_present`]); [`FrameOutput`] decides the target's final
    /// layout (`PRESENT_SRC` vs `GENERAL`) and who signals present.
    pub fn render(
        &mut self,
        camera: &Camera,
        instances: &[(K, Vec<vk::TransformMatrixKHR>)],
        output: FrameOutput<'_>,
    ) -> SrResult<u64> {
        // ── Start of frame: scheduled callbacks + deferred deallocation of the
        // per-frame resources of frames the timeline reported complete.
        let upcoming_frame = *self.core.absolute_frame_count.borrow() as u64 + 1;

        // Gate reuse of this frame's render-graph slot (its command buffer,
        // transient pool and retired passes) on the completion of the frame that
        // used it N frames ago. Done first, before the resource manager reclaims
        // arena slots (which also assumes that frame is done). Non-blocking in
        // steady state — this replaced the former device-wide idle wait between
        // frames, and with it (plus the in-graph arena copy prologue) consecutive
        // frames overlap: the CPU records frame N+1 while the GPU runs frame N.
        self.render_graph.wait_for_slot_reuse()?;

        self.run_start_of_frame_callbacks(upcoming_frame);
        // Drains every end-of-frame callback the GPU has now finished: per-frame
        // buffer drops, the AS-build heuristic fold-ins (`mark_blas_built` /
        // `mark_tlas_built`) that `build_unified_graph` scheduled, and deferred
        // asset removals from `unload_mesh` — the render graph records the builds
        // but can't touch the resource manager's CPU-side wrapper state.
        self.run_due_end_of_frame_callbacks();

        self.resource_manager.start_of_frame(upcoming_frame)?;

        // ── Per-frame GPU data: CpuToGpu buffers created on the spot, local to
        // this frame. They're moved into an end-of-frame callback at the end of
        // this function and freed once the frame timeline passes this frame.
        let mut matrices = camera.as_matrices(self.image_extent);
        // Inject the history matrix saved from the last frame; save the current
        // one to use as history NEXT frame.
        matrices.prev_view_proj = self.prev_view_proj;
        self.prev_view_proj = matrices.view_proj;

        // nalgebra's Matrix4 is column-major in memory. HLSL/Slang's
        // `float4x4(v0, v1, v2, v3)` constructor reads each float4 as a ROW.
        // Transposing here means each on-disk float4 (which the shader reads as
        // a member of `Matrices`) is a ROW of the intended matrix, so the
        // shader's `float4x4(m.vi0, m.vi1, m.vi2, m.vi3)` reconstructs the
        // matrix correctly without any per-shader `transpose()` call.
        // Write into this frame's slot of the persistent matrices pool (stable
        // address, no per-frame create/destroy). `wait_for_slot_reuse` above proved
        // the frame that last used this slot finished its graph, so overwriting the
        // buffer's contents here can't race an in-flight read.
        let matrices_slot = (upcoming_frame as usize) % MAX_FRAMES_IN_FLIGHT;
        // Destructure-copy first: `CameraMatrices` is `repr(C, packed)`, so
        // taking references to its fields (which a method call would) is UB.
        let CameraMatrices {
            view_inverse,
            proj_inverse,
            view_proj,
            prev_view_proj,
        } = matrices;
        self.matrices_pool[matrices_slot].map_mut()?[0] = CameraMatrices {
            view_inverse: view_inverse.transpose(),
            proj_inverse: proj_inverse.transpose(),
            view_proj: view_proj.transpose(),
            prev_view_proj: prev_view_proj.transpose(),
        };
        let matrices_address = self.matrices_pool[matrices_slot].get_device_address();

        let frame_data = self.resource_manager.frame_instance_data(instances)?;
        let instance_count = frame_data.as_instances.len() as u32;

        // Empty slices would produce null buffers (and so invalid heap
        // descriptors / build inputs); pad each with one dummy element. The
        // TLAS build reads 0 instances from the dummy and the shader sees the
        // dummy emissive entry only through the (matching, pre-rework) "no
        // lights" behavior.
        let mut as_instances = frame_data.as_instances;
        if as_instances.is_empty() {
            as_instances.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: 0 },
            });
        }
        let mut transforms = frame_data.transforms;
        if transforms.is_empty() {
            transforms.push(vk::TransformMatrixKHR {
                matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            });
        }
        let mut emissive_entries = frame_data.emissive_entries;
        if emissive_entries.is_empty() {
            emissive_entries.push(vulkan_abstraction::gltf::EmissiveIndirectionEntry {
                blas_tri_index: 0,
                entity_id: 0,
            });
        }

        let instances_buffer = vulkan_abstraction::StagingBuffer::new_from_data(
            Rc::clone(&self.core),
            &as_instances,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "per-frame TLAS instances",
        )?;
        let transforms_buffer = vulkan_abstraction::StagingBuffer::new_from_data(
            Rc::clone(&self.core),
            &transforms,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "per-frame instance transforms",
        )?;
        // Exactly sized: the shader reads num_lights via `GetDimensions`.
        let emissive_indirection_buffer = vulkan_abstraction::StagingBuffer::new_from_data(
            Rc::clone(&self.core),
            &emissive_entries,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "per-frame emissive indirection",
        )?;

        // The TLAS build (and any pending BLAS builds) are no longer submitted
        // synchronously here — they're recorded into the unified graph below by
        // `queue_blas_builds` / `queue_tlas_build`, ordered against the RT trace by
        // graph barriers. `instances_buffer` is read by the deferred TLAS build, so
        // it must stay alive until the graph submission completes; the end-of-frame
        // callback that frees it is already tagged with this frame.

        let frame_gpu_data = FrameGpuData {
            matrices_address,
            entity_transforms_slot: transforms_buffer.raw().storage_slot(),
            emissive_indirection_slot: emissive_indirection_buffer.raw().storage_slot(),
        };

        // The graph's slot (command buffer + transient pool) was already gated for
        // reuse by `wait_for_slot_reuse` at the top of the frame, and nothing has
        // been submitted since, so `build_unified_graph` can safely re-record it.
        let result_extent = self.image_extent;
        let postprocess_out = Arc::clone(&self.postprocess_result_image);

        // Build + compile the unified render graph: RT (RIS + final), temporal
        // accumulation, the 8 a-trous denoise passes, and postprocess. Every pass
        // is heap + Slang; the intermediate G-buffer / RT-output images are
        // graph-internal (transient) resources.
        let source_h = self.build_unified_graph(
            &postprocess_out,
            result_extent,
            &frame_gpu_data,
            instance_count,
            &instances_buffer,
        )?;

        // build_unified_graph advanced the absolute frame count: that's this
        // frame's number on the frame timeline.
        let frame_value = *self.core.absolute_frame_count.borrow() as u64;
        debug_assert_eq!(frame_value, upcoming_frame);

        // The graph's command buffer blits the post-process result into the output
        // image and transitions it to the layout `dst_final` (in-submit).
        // `graph_timeline = N` is the single "frame N complete" signal. The AS
        // builds (TLAS + BLAS) are recorded *inside* the graph and ordered against
        // the RT trace by graph barriers, so there are no external transfer
        // producers left to wait on here.
        match output {
            // Renderer finishes the present: blit → PRESENT_SRC, signal `present_sem`.
            FrameOutput::Present {
                image,
                acquire_sem,
                present_sem,
            } => {
                let extra_waits = [(acquire_sem, 0, vk::PipelineStageFlags2::TRANSFER)];
                let extra_signals = [(present_sem, 0, vk::PipelineStageFlags2::ALL_COMMANDS)];
                self.render_graph
                    .run_present(&source_h, image, vk_sync::AccessType::Present, &extra_waits, &extra_signals)?;
            }
            // Overlay finishes the present: blit, leave GENERAL. The caller's overlay
            // pass (waiting `graph_timeline >= N`) transitions to PRESENT_SRC and
            // signals present itself.
            FrameOutput::PresentOverlay { image, acquire_sem } => {
                let extra_waits = [(acquire_sem, 0, vk::PipelineStageFlags2::TRANSFER)];
                self.render_graph
                    .run_present(&source_h, image, vk_sync::AccessType::General, &extra_waits, &[])?;
            }
            // Offscreen readback: blit, leave GENERAL, no present. `graph_timeline = N`
            // (single submit) now means body+blit done, so `wait_frame` covers it.
            FrameOutput::Offscreen { image } => {
                self.render_graph
                    .run_present(&source_h, image, vk_sync::AccessType::General, &[], &[])?;
            }
        }

        // ── End of frame: the frame-local buffers stay alive until the GPU is
        // done with this frame, then get dropped on the render thread.
        self.end_of_frame_callbacks.push((
            frame_value,
            Box::new(move |_renderer| {
                drop(instances_buffer);
                drop(transforms_buffer);
                drop(emissive_indirection_buffer);
            }),
        ));

        // Whole-frame serialization: block until this frame fully completes on
        // the GPU before returning, so the next frame's CPU recording can't
        // overlap this frame's in-flight GPU work. ON BY DEFAULT.
        //
        // ponytail: serialized by default; opt into overlap with SUNRAY_SERIALIZE_FRAMES=0,
        //           see `utils::SERIALIZE_FRAMES`;
        //           real fix is per-in-flight-slot duplication of every per-frame resource.
        //
        // Why: frame overlap causes an async use-after-free that crashes inside
        // the NVIDIA driver (`nvoglv64` null-deref on a driver worker thread,
        // NOT a DEVICE_LOST) — the driver consumes a per-frame resource on its
        // own thread after the CPU has already recycled it a frame later. It is
        // invisible to the validation layer (the fault happens after the
        // validated submit call returns) and only whole-frame serialization
        // avoids it. The earlier temporal-barrier fix (`RenderGraph::compile`
        // threads each temporal backing's end access into next frame's compile)
        // was necessary but not sufficient. Overlap is also low value here: graph
        // GPU work is already serialized (`RenderGraph::run` waits
        // `graph_timeline >= N-1`) and the workload is heavily GPU-bound, so
        // CPU-record-ahead buys almost nothing. Properly fixing overlap needs
        // every per-frame-referenced resource duplicated per in-flight slot — a
        // rework to do only if profiling shows CPU-ahead actually matters.
        if env_var_as_bool(SERIALIZE_FRAMES).unwrap_or(true) {
            self.render_graph.wait_graph_timeline(frame_value)?;
        }

        Ok(frame_value)
    }

    /// Block until frame `frame_value` (as returned by [`Self::render`]) has
    /// completed on the GPU (graph timeline reached `frame_value`).
    pub fn wait_frame(&self, frame_value: u64) -> SrResult<()> {
        self.render_graph.wait_graph_timeline(frame_value)
    }

    /// Read-only access to the internal swapchain (present only when the
    /// renderer was built with a surface). Useful for overlay passes that need
    /// the swapchain format / image count.
    pub fn swapchain(&self) -> Option<&Swapchain> {
        self.swapchain_data.as_ref().map(|sc| &sc.swapchain)
    }

    /// Render one frame to the internal swapchain and present it: waits the
    /// in-flight fence, acquires an image, calls [`Self::render`], transitions
    /// the image to `PRESENT_SRC` with the pre-recorded barrier, and presents.
    /// All per-frame inputs (camera + instances) come from the caller.
    pub fn render_to_swapchain(&mut self, camera: &Camera, instances: &[(K, Vec<vk::TransformMatrixKHR>)]) -> SrResult<()> {
        self.render_to_swapchain_with(camera, instances, None)
    }

    /// Like [`Self::render_to_swapchain`], but lets the caller replace the
    /// final GENERAL -> PRESENT_SRC transition with its own pass (e.g. an egui
    /// overlay drawn straight onto the swapchain image). The `finalize` closure
    /// must leave the image in `PRESENT_SRC_KHR` and signal
    /// [`SwapchainFrame::ready_to_present_sem`]; the renderer presents after.
    pub fn render_to_swapchain_with(
        &mut self,
        camera: &Camera,
        instances: &[(K, Vec<vk::TransformMatrixKHR>)],
        finalize: Option<&mut dyn FnMut(&SwapchainFrame) -> SrResult<()>>,
    ) -> SrResult<()> {
        let (frame_index, img_acquired_sem, img_rendered_frame) = {
            let sc = self
                .swapchain_data
                .as_mut()
                .ok_or_else(|| SrError::new_custom("render_to_swapchain: renderer was built without a surface".to_string()))?;
            let frame_index = (sc.frame_count as usize) % MAX_FRAMES_IN_FLIGHT;
            sc.frame_count += 1;
            (
                frame_index,
                sc.img_acquired_sems[frame_index].inner(),
                sc.img_rendered_frames[frame_index],
            )
        };

        // Wait (on the frame timeline) for the frame that used this in-flight
        // slot MAX_FRAMES_IN_FLIGHT frames ago before reusing its semaphore.
        self.render_graph.wait_graph_timeline(img_rendered_frame)?;

        let mut frame = {
            let sc = self.swapchain_data.as_ref().unwrap();
            let (image_index, suboptimal) = unsafe {
                sc.swapchain
                    .device()
                    .acquire_next_image(sc.swapchain.inner(), u64::MAX, img_acquired_sem, vk::Fence::null())
            }?;
            if suboptimal {
                log::warn!("VkAcquireNextImageKHR: swapchain is suboptimal for the surface");
            }
            let image_index = image_index as usize;
            SwapchainFrame {
                image: sc.swapchain.images()[image_index],
                image_view: sc.swapchain.image_views()[image_index],
                extent: sc.swapchain.extent(),
                image_index,
                ready_to_present_sem: sc.ready_to_present_sems[image_index].inner(),
                render_done_sem: vk::Semaphore::null(),
                render_done_value: 0,
            }
        };

        // Wrap the acquired swapchain image as a borrowed, non-owning `Image` so the
        // graph's `run_present` can blit into it — the swapchain is known to the
        // graph *only* here, at run.
        let swapchain_image = vulkan_abstraction::Image::from_swapchain_image(
            Rc::clone(&self.core),
            frame.image,
            vk::Extent3D {
                width: frame.extent.width,
                height: frame.extent.height,
                depth: 1,
            },
            self.swapchain_data.as_ref().unwrap().swapchain.format(),
        );

        // With an overlay (`finalize`), the graph blits then leaves the image in
        // GENERAL; the overlay draws on top and finishes the present. Without one,
        // the graph transitions straight to PRESENT_SRC and signals the present sem.
        let output = if finalize.is_some() {
            FrameOutput::PresentOverlay {
                image: &swapchain_image,
                acquire_sem: img_acquired_sem,
            }
        } else {
            FrameOutput::Present {
                image: &swapchain_image,
                acquire_sem: img_acquired_sem,
                present_sem: frame.ready_to_present_sem,
            }
        };
        let rendered_frame = self.render(camera, instances, output)?;
        self.swapchain_data.as_mut().unwrap().img_rendered_frames[frame_index] = rendered_frame;

        // Overlay pass (egui): draws onto the GENERAL image, transitions it to
        // PRESENT_SRC and signals the present sem itself, after waiting the graph
        // timeline (its input is the just-blitted image). See `SwapchainFrame`.
        if let Some(finalize) = finalize {
            frame.render_done_sem = self.render_graph.graph_timeline_inner();
            frame.render_done_value = rendered_frame;
            finalize(&frame)?;
        }

        // Present, waiting on the present semaphore signaled by `run_present` (direct)
        // or the overlay pass.
        let sc = self.swapchain_data.as_ref().unwrap();
        let swapchains = [sc.swapchain.inner()];
        let image_indices = [frame.image_index as u32];
        let wait_semaphores = [frame.ready_to_present_sem];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let queue = self.core.graphics_queue().inner();
        unsafe { sc.swapchain.device().queue_present(queue, &present_info) }?;

        Ok(())
    }

    /// Build + compile the unified render graph for this frame: ray tracing
    /// (RIS + final in one node), temporal accumulation, the 8 a-trous denoise
    /// passes, and postprocess. Every pass is heap-mode + Slang. The G-buffer /
    /// RT-output images are created as graph-internal (transient) resources; the
    /// cross-frame accumulation ping-pong, the denoise ping-pong, and the ReSTIR
    /// reservoir buffers are graph-owned *temporal* resources re-registered each
    /// rebuild; the post-process output is a per-target import.
    //TODO finni
    fn build_unified_graph(
        &mut self,
        postprocess_out: &Arc<vulkan_abstraction::Image>,
        extent: vk::Extent3D,
        frame_gpu_data: &FrameGpuData,
        instance_count: u32,
        instances_buffer: &impl vulkan_abstraction::Buffer,
    ) -> SrResult<Handle<vulkan_abstraction::Image>> {
        let frame_count = self.relative_frame_count;
        let width = extent.width;
        let height = extent.height;
        // Ping-pong: TAA writes accum[accum_idx] (which denoise then reads) and
        // reads accum[history_idx] (last frame's output).
        let accum_idx = (frame_count % 2) as usize;
        let history_idx = ((frame_count + 1) % 2) as usize;

        let pack = |i: u32| -> [u32; 2] { [i, 0] };

        // Non-image fields of the RT push constant: stable slots come from the
        // resource manager, the per-frame ones (matrices address, transforms /
        // emissive indirection slots) from this frame's local buffers. The five
        // RT-output image slots are filled inside the closure from the graph's
        // transient resources (they're created per frame).
        // `tlas` is filled below with the address returned by `queue_tlas_build`
        // (a rebuild yields a fresh structure with a new address); 0 here is a
        // placeholder that is always overwritten before the RT passes are added.
        let mut rt_pc_base = vulkan_abstraction::RaytracingHeapPushConstant {
            tlas: 0,
            matrices: frame_gpu_data.matrices_address,
            meshes_info: pack(self.resource_manager.meshes_info_storage_slot()),
            emissive_triangles: pack(self.resource_manager.emissive_triangles_storage_slot()),
            emissive_indirection: pack(frame_gpu_data.emissive_indirection_slot),
            entity_transforms: pack(frame_gpu_data.entity_transforms_slot),
            blue_noise_tex: pack(self.blue_noise_image.sampled_slot()),
            blue_noise_sampler: pack(self.blue_noise_sampler.slot()),
            // Device addresses of the graph-owned ping-pong reservoir backings;
            // the shader picks current/history internally via `frame_count`.
            reservoirs: self.render_graph.temporal_buffer_addresses(&self.reservoir_temporal),
            reservoirs_gi: self.render_graph.temporal_buffer_addresses(&self.reservoir_gi_temporal),
            frame_count,
            use_srgb: if self.image_format == vk::Format::R8G8B8A8_SRGB {
                1
            } else {
                0
            },
            ..Default::default()
        };

        // RT passes now describe themselves with their per-stage SPIR-V; the
        // graph's pipeline cache builds/reuses the pipeline + SBT (RIS and final
        // share miss/closest-hit/any-hit and differ only in the ray-gen blob, so
        // they intern as two distinct entries). The shader list is ordered
        // [ray_gen, miss, closest_hit, any_hit] and each stage's entry point is
        // selected by its index — see `RayTracingShaders`.
        let make_rt_shaders = |ray_gen: &'static [u8]| -> RayTracingShaders {
            RayTracingShaders::new(
                vec![
                    ShaderSource::Spirv(ray_gen.to_vec()),
                    ShaderSource::Spirv(self.ray_miss_spirv.to_vec()),
                    ShaderSource::Spirv(self.closest_hit_spirv.to_vec()),
                    ShaderSource::Spirv(self.any_hit_spirv.to_vec()),
                ],
                (0, "main"),
                (1, "main"),
                (2, "main"),
                (3, "main"),
            )
        };
        let ris_shaders = make_rt_shaders(self.ray_gen_ris_spirv);
        let final_shaders = make_rt_shaders(self.ray_gen_final_spirv);

        // Compute passes now describe themselves with their SPIR-V; the graph's
        // pipeline cache builds/reuses the pipeline. Snapshot the bytes into
        // locals so the `&mut self.render_graph` borrow below stays disjoint.
        let taa_spirv = self.temporal_accumulation_spirv;
        let denoise_spirv = self.denoise_spirv;
        let postprocess_spirv = self.postprocess_spirv;

        let postprocess_out_arc = Arc::clone(postprocess_out);

        // Snapshot the temporal-resource tokens so they can be re-registered into
        // the graph below while `self.render_graph` is borrowed mutably. Cloning a
        // token is cheap (an index + the resource desc); the backing memory stays
        // owned by the graph and is preserved across this rebuild.
        let accumulation_temporal = self.accumulation_temporal.clone();
        let denoising_temporal = self.denoising_temporal.clone();
        let reservoir_temporal = self.reservoir_temporal.clone();
        let reservoir_gi_temporal = self.reservoir_gi_temporal.clone();

        // Advance the frame counters for the next frame (after snapshotting
        // `frame_count` for this one).
        self.relative_frame_count += 1;
        *self.core.absolute_frame_count.borrow_mut() += 1;

        let rg = &mut self.render_graph;
        rg.reset();

        // Re-import the arena buffers into this fresh build
        let arena_handles = self.resource_manager.import_to_graph(rg);
        let arena_copies = self.resource_manager.take_queued_copies()?;
        // SAFETY: the sources are the resource manager's staging arena buffers.
        // The arena owns them for the renderer's lifetime, they are not graph
        // resources, and `take_queued_copies` hands out only copies staged for
        // this frame.
        unsafe { rg.add_prologue_buffer_copies(arena_copies)? };

        // Record this frame's acceleration-structure builds into the graph before
        // any consumer: pending BLAS builds first (so the TLAS build orders itself
        // after them via graph read edges), then the TLAS build/update. The RT
        // passes below declare a read on `tlas_h`, so `compile` emits the
        // build→trace barrier itself — no separate synchronous AS submit.
        let built_blases = self.resource_manager.queue_blas_builds(rg)?;
        let blas_deps: Vec<_> = built_blases.iter().map(|(_, handle)| handle.clone()).collect();
        let (tlas_h, tlas_address) = self
            .resource_manager
            .queue_tlas_build(rg, instance_count, instances_buffer, &blas_deps)?;
        rt_pc_base.tlas = tlas_address;

        // Fold each recorded build's chosen op back into its CPU-side heuristic
        // state once this frame's GPU work completes (the graph records the build
        // but can't mutate that wrapper state). `frame` is this frame's absolute
        // number (just incremented) — the same value the frame timeline signals on
        // completion, so `run_due_end_of_frame_callbacks` fires these at the right
        // time. Pushed straight onto the field so the `rg` (= `&mut
        // self.render_graph`) borrow held here stays disjoint.
        let frame = *self.core.absolute_frame_count.borrow() as u64;
        for (key, _) in built_blases {
            self.end_of_frame_callbacks.push((
                frame,
                Box::new(move |renderer: &mut Renderer<K>| renderer.resource_manager.mark_blas_built(key)),
            ));
        }
        self.end_of_frame_callbacks.push((
            frame,
            Box::new(|renderer: &mut Renderer<K>| renderer.resource_manager.mark_tlas_built()),
        ));

        let mk_img = |format: vk::Format, usage: vk::ImageUsageFlags, name: &'static str| ImageDesc {
            extent,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            usage,
            name,
        };

        // Internal (transient) RT outputs.
        let raw_color_h = rg.create_resource(mk_img(
            vk::Format::B10G11R11_UFLOAT_PACK32,
            vk::ImageUsageFlags::STORAGE,
            "rg_rt_raw_color",
        ));
        let depth_h = rg.create_resource(mk_img(
            vk::Format::R16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            "rg_depth",
        ));
        let normal_h = rg.create_resource(mk_img(
            vk::Format::R8G8B8A8_SNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            "rg_normal",
        ));
        let diffuse_h = rg.create_resource(mk_img(
            vk::Format::B10G11R11_UFLOAT_PACK32,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            "rg_diffuse",
        ));
        let motion_h = rg.create_resource(mk_img(
            vk::Format::R16G16_SFLOAT,
            vk::ImageUsageFlags::STORAGE,
            "rg_motion_vec",
        ));

        // Temporal (cross-frame) ping-pong images: re-register the graph-owned
        // backings into this rebuild. They are wired in as imports — never aliased,
        // memory preserved across frames — with index `i` the copy for frame `i`.
        let [accum0_h, accum1_h] = rg.register_temporal_resource(&accumulation_temporal);
        let [denoise_a_h, denoise_b_h] = rg.register_temporal_resource(&denoising_temporal);

        // The post-process output is a per-target (per-swapchain-image) import, not
        // a temporal resource — it changes with the destination image.
        let postprocess_out_h = rg.import::<ImageDesc>(postprocess_out_arc);

        // Reservoir ping-pong buffers re-registered for hazard tracking so the
        // graph emits the RIS→final hand-off barrier between the two RT passes
        // itself. The shader still reaches them by device-address (baked into
        // `rt_pc_base` from `temporal_buffer_addresses`).
        let [reservoir0_h, reservoir1_h] = rg.register_temporal_resource(&reservoir_temporal);
        let [reservoir_gi0_h, reservoir_gi1_h] = rg.register_temporal_resource(&reservoir_gi_temporal);
        let reservoir_handles = [reservoir0_h, reservoir1_h, reservoir_gi0_h, reservoir_gi1_h];

        let accum_target_h = if accum_idx == 0 { accum0_h.clone() } else { accum1_h.clone() };
        let accum_history_h = if history_idx == 0 {
            accum0_h.clone()
        } else {
            accum1_h.clone()
        };

        // 1. Ray tracing as two heap-mode passes built through the standard
        // `RaytracingRenderPassBuilder::generate_render` path: RIS audition then
        // final shading, each interning its own pipeline + SBT in the graph cache.
        // They're ordered by the shared G-buffer write-after-write hazard; the
        // reservoir hand-off (RIS writes, final reads) is now a real graph edge on
        // the imported reservoir buffers — no manual barrier.
        Self::add_raytracing_ris_pass(
            rg,
            ris_shaders,
            rt_pc_base,
            raw_color_h.clone(),
            depth_h.clone(),
            normal_h.clone(),
            diffuse_h.clone(),
            motion_h.clone(),
            reservoir_handles.clone(),
            arena_handles.clone(),
            tlas_h.clone(),
            extent,
        )?;
        Self::add_raytracing_final_pass(
            rg,
            final_shaders,
            rt_pc_base,
            raw_color_h.clone(),
            depth_h.clone(),
            normal_h.clone(),
            diffuse_h.clone(),
            motion_h.clone(),
            reservoir_handles,
            arena_handles,
            tlas_h,
            extent,
        )?;

        // 2. Temporal accumulation.
        Self::add_temporal_pass(
            rg,
            taa_spirv,
            raw_color_h.clone(),
            motion_h.clone(),
            accum_history_h,
            accum_target_h.clone(),
            frame_count,
            width,
            height,
        )?;

        // 3. Denoise (8 a-trous passes). Pass 0 reads the TAA output (accum_target).
        Self::add_denoise_passes(
            rg,
            denoise_spirv,
            accum_target_h,
            depth_h,
            normal_h,
            diffuse_h,
            denoise_a_h.clone(),
            denoise_b_h.clone(),
            frame_count,
            width,
            height,
        )?;

        // 4. Postprocess: read the final denoise output, tonemap into the output.
        let final_idx = ((DENOISE_PASSES - 1) % 2) as usize;
        let denoise_input_h = if final_idx == 0 { denoise_a_h } else { denoise_b_h };
        let source_h = postprocess_out_h.clone();
        Self::add_postprocess_pass(
            rg,
            postprocess_spirv,
            denoise_input_h,
            postprocess_out_h,
            width,
            height,
            EXPOSURE,
        )?;

        rg.compile()?;
        // The post-process output is the frame's final image; the caller blits it
        // to the swapchain (or an offscreen target) after `run`/`run_present`.
        Ok(source_h)
    }

    /// Builds the heap-mode raytracing push constant for a frame: the non-image
    /// fields come from `pc_base`, the five G-buffer / RT-output image slots are
    /// resolved from the graph's transient resources, then the whole struct is
    /// serialized to the raw bytes `generate_render` pushes via `cmd_push_data`.
    /// Shared by both RT passes so they push identical bindings.
    fn rt_push_constant_bytes(
        pc_base: &vulkan_abstraction::RaytracingHeapPushConstant,
        tr: &render_graph::transient_resources::TransientResources,
        raw_color_h: &Handle<vulkan_abstraction::Image>,
        depth_h: &Handle<vulkan_abstraction::Image>,
        normal_h: &Handle<vulkan_abstraction::Image>,
        diffuse_h: &Handle<vulkan_abstraction::Image>,
        motion_h: &Handle<vulkan_abstraction::Image>,
    ) -> SrResult<Vec<u8>> {
        let pack = |i: u32| -> [u32; 2] { [i, 0] };
        let mut pc = *pc_base;
        pc.raw_color = pack(tr.image(raw_color_h)?.storage_slot());
        pc.depth_img = pack(tr.image(depth_h)?.storage_slot());
        pc.normal_img = pack(tr.image(normal_h)?.storage_slot());
        pc.diffuse_img = pack(tr.image(diffuse_h)?.storage_slot());
        pc.motion_vec_img = pack(tr.image(motion_h)?.storage_slot());
        // `RaytracingHeapPushConstant` is `#[repr(C)]` plain data, so a verbatim
        // byte copy matches the shader's push-constant layout.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &pc as *const vulkan_abstraction::RaytracingHeapPushConstant as *const u8,
                size_of::<vulkan_abstraction::RaytracingHeapPushConstant>(),
            )
        }
        .to_vec();
        Ok(bytes)
    }

    /// RIS audition ray-tracing pass, built through the standard
    /// `RaytracingRenderPassBuilder::generate_render` path (pipeline + SBT interned
    /// in the graph's cache). Writes the G-buffer / RT-output images and the ReSTIR
    /// reservoirs; the reservoir writes are declared on the imported reservoir
    /// buffers so the graph emits the RIS→final hand-off barrier itself. Ordering
    /// against the final pass also comes from the shared G-buffer write-after-write
    /// hazard.
    #[allow(clippy::too_many_arguments)]
    fn add_raytracing_ris_pass(
        rg: &mut RenderGraph,
        shaders: RayTracingShaders,
        pc_base: vulkan_abstraction::RaytracingHeapPushConstant,
        raw_color_h: Handle<vulkan_abstraction::Image>,
        depth_h: Handle<vulkan_abstraction::Image>,
        normal_h: Handle<vulkan_abstraction::Image>,
        diffuse_h: Handle<vulkan_abstraction::Image>,
        motion_h: Handle<vulkan_abstraction::Image>,
        reservoir_handles: [Handle<vulkan_abstraction::RawBuffer>; 4],
        arena_handles: [Handle<vulkan_abstraction::RawBuffer>; 2],
        tlas_h: Handle<vulkan_abstraction::AccelerationStructure>,
        extent: vk::Extent3D,
    ) -> SrResult<()> {
        let mut common = PassCommonDataBuilder::new(rg, "raytracing_ris");
        // Writes all five outputs via storage descriptors (GENERAL). `General` is
        // used because vk_sync has no ray-tracing-specific write access type.
        common.write(&raw_color_h, vk_sync::AccessType::General)?;
        common.write(&depth_h, vk_sync::AccessType::General)?;
        common.write(&normal_h, vk_sync::AccessType::General)?;
        common.write(&diffuse_h, vk_sync::AccessType::General)?;
        common.write(&motion_h, vk_sync::AccessType::General)?;
        // Read the TLAS: this read against the TLAS build pass's write is the graph
        // edge that becomes the AS-build→trace barrier (so the build is complete
        // before the trace reads it). The shader reaches the TLAS by device address
        // (baked into `pc_base.tlas`); this only governs synchronization.
        common.read(&tlas_h, vk_sync::AccessType::RayTracingShaderReadAccelerationStructure)?;
        // Declare the reservoir SSBO writes so the graph orders the final pass's
        // reservoir reads after this pass (the RIS→final hand-off barrier).
        for h in &reservoir_handles {
            common.write(h, vk_sync::AccessType::AnyShaderWrite)?;
        }
        // Read the mesh-info / emissive-triangle arenas: this read against the
        // prologue copy pass's write is what orders (and barriers) an asset load's
        // staging→GPU copy before the trace that consumes it. Reached by device
        // address / heap slot in the shader; this only governs synchronization.
        for h in &arena_handles {
            common.read(h, vk_sync::AccessType::RayTracingShaderReadOther)?;
        }

        let pass = RaytracingRenderPassBuilder::default()
            .common(common.build())
            .shaders(shaders)
            .trace_extent([extent.width, extent.height, extent.depth])
            .generate_render(rg, move |tr| {
                Self::rt_push_constant_bytes(&pc_base, tr, &raw_color_h, &depth_h, &normal_h, &diffuse_h, &motion_h)
            })?
            .build()
            .map_err(|e| SrError::new_custom(format!("raytracing RIS pass builder failed: {e}")))?;
        rg.add_render_pass(pass);
        Ok(())
    }

    /// Final shading ray-tracing pass, built through the standard
    /// `RaytracingRenderPassBuilder::generate_render` path (pipeline + SBT interned
    /// in the graph's cache). Reads the reservoirs the RIS pass produced
    /// (visibility established by the graph-emitted reservoir barrier) and writes
    /// the final color into the G-buffer outputs. Ordered after the RIS pass by the
    /// shared G-buffer write-after-write hazard the graph tracks.
    #[allow(clippy::too_many_arguments)]
    fn add_raytracing_final_pass(
        rg: &mut RenderGraph,
        shaders: RayTracingShaders,
        pc_base: vulkan_abstraction::RaytracingHeapPushConstant,
        raw_color_h: Handle<vulkan_abstraction::Image>,
        depth_h: Handle<vulkan_abstraction::Image>,
        normal_h: Handle<vulkan_abstraction::Image>,
        diffuse_h: Handle<vulkan_abstraction::Image>,
        motion_h: Handle<vulkan_abstraction::Image>,
        reservoir_handles: [Handle<vulkan_abstraction::RawBuffer>; 4],
        arena_handles: [Handle<vulkan_abstraction::RawBuffer>; 2],
        tlas_h: Handle<vulkan_abstraction::AccelerationStructure>,
        extent: vk::Extent3D,
    ) -> SrResult<()> {
        let mut common = PassCommonDataBuilder::new(rg, "raytracing_final");
        // Re-declares the same writes as the RIS pass: this is what creates the
        // write-after-write hazard edge that orders this pass after RIS.
        common.write(&raw_color_h, vk_sync::AccessType::General)?;
        common.write(&depth_h, vk_sync::AccessType::General)?;
        common.write(&normal_h, vk_sync::AccessType::General)?;
        common.write(&diffuse_h, vk_sync::AccessType::General)?;
        common.write(&motion_h, vk_sync::AccessType::General)?;
        // Read the TLAS (same as the RIS pass) so the AS build is ordered before
        // this trace too; addressed by device address in the shader.
        common.read(&tlas_h, vk_sync::AccessType::RayTracingShaderReadAccelerationStructure)?;
        // Read the reservoirs the RIS pass wrote — this read against the RIS pass's
        // declared writes is the graph edge that becomes the hand-off barrier.
        for h in &reservoir_handles {
            common.read(h, vk_sync::AccessType::RayTracingShaderReadOther)?;
        }
        // Same arena reads as the RIS pass — see there.
        for h in &arena_handles {
            common.read(h, vk_sync::AccessType::RayTracingShaderReadOther)?;
        }

        let pass = RaytracingRenderPassBuilder::default()
            .common(common.build())
            .shaders(shaders)
            .trace_extent([extent.width, extent.height, extent.depth])
            .generate_render(rg, move |tr| {
                Self::rt_push_constant_bytes(&pc_base, tr, &raw_color_h, &depth_h, &normal_h, &diffuse_h, &motion_h)
            })?
            .build()
            .map_err(|e| SrError::new_custom(format!("raytracing final pass builder failed: {e}")))?;
        rg.add_render_pass(pass);
        Ok(())
    }

    /// Temporal accumulation graph node (heap + Slang). Reads the RT raw color +
    /// motion vectors and the history accumulation image, writes the target
    /// accumulation image.
    #[allow(clippy::too_many_arguments)]
    fn add_temporal_pass(
        rg: &mut RenderGraph,
        spirv: &[u8],
        raw_color_h: Handle<vulkan_abstraction::Image>,
        motion_h: Handle<vulkan_abstraction::Image>,
        history_h: Handle<vulkan_abstraction::Image>,
        accum_target_h: Handle<vulkan_abstraction::Image>,
        frame_count: u32,
        width: u32,
        height: u32,
    ) -> SrResult<()> {
        let mut common = PassCommonDataBuilder::new(rg, "temporal_accumulation");
        common.read(&raw_color_h, vk_sync::AccessType::ComputeShaderReadOther)?;
        common.read(&motion_h, vk_sync::AccessType::ComputeShaderReadOther)?;
        common.read(&history_h, vk_sync::AccessType::ComputeShaderReadOther)?;
        common.write(&accum_target_h, vk_sync::AccessType::ComputeShaderWrite)?;

        // Only the shaders + a push-data closure: `generate_render` interns the
        // pipeline in the graph cache and installs the bind/push/dispatch closure.
        let pass = ComputeRenderPassBuilder::default()
            .common(common.build())
            .shaders(ComputeShaders::new(vec![ShaderSource::Spirv(spirv.to_vec())], 0, "main"))
            .generate_render(rg, [width.div_ceil(16), height.div_ceil(16), 1], move |tr| {
                let pack = |i: u32| -> [u32; 2] { [i, 0] };
                Ok(vulkan_abstraction::TemporalAccumulationHeapPushConstant {
                    raw_rt_color: pack(tr.image(&raw_color_h)?.storage_slot()),
                    motion_vector: pack(tr.image(&motion_h)?.storage_slot()),
                    history: pack(tr.image(&history_h)?.storage_slot()),
                    accum_output: pack(tr.image(&accum_target_h)?.storage_slot()),
                    frame_count,
                    width,
                    height,
                })
            })
            .map_err(|e| SrError::new_custom(format!("temporal accumulation pass builder failed: {e}")))?;
        rg.add_render_pass(pass);
        Ok(())
    }

    /// The 8 a-trous denoise passes (heap + Slang). depth/normal/diffuse are read
    /// (sampled) only in pass 0 to register the GENERAL->SHADER_READ transition;
    /// later passes read the same stable slots directly without re-registering.
    #[allow(clippy::too_many_arguments)]
    fn add_denoise_passes(
        rg: &mut RenderGraph,
        spirv: &[u8],
        accum_in_h: Handle<vulkan_abstraction::Image>,
        depth_h: Handle<vulkan_abstraction::Image>,
        normal_h: Handle<vulkan_abstraction::Image>,
        diffuse_h: Handle<vulkan_abstraction::Image>,
        denoise_a_h: Handle<vulkan_abstraction::Image>,
        denoise_b_h: Handle<vulkan_abstraction::Image>,
        frame_count: u32,
        width: u32,
        height: u32,
    ) -> SrResult<()> {
        for pass_index in 0..DENOISE_PASSES {
            let step_width = 1i32 << pass_index;
            let (read_h, write_h) = if pass_index == 0 {
                (accum_in_h.clone(), denoise_a_h.clone())
            } else if pass_index % 2 == 1 {
                (denoise_a_h.clone(), denoise_b_h.clone())
            } else {
                (denoise_b_h.clone(), denoise_a_h.clone())
            };

            let mut common = PassCommonDataBuilder::new(rg, format!("denoise_{pass_index}"));
            common.read(&read_h, vk_sync::AccessType::ComputeShaderReadOther)?;
            common.write(&write_h, vk_sync::AccessType::ComputeShaderWrite)?;
            if pass_index == 0 {
                common.read(
                    &depth_h,
                    vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                )?;
                common.read(
                    &normal_h,
                    vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                )?;
                common.read(
                    &diffuse_h,
                    vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                )?;
            }

            let read_h_c = read_h.clone();
            let write_h_c = write_h.clone();
            let depth_c = depth_h.clone();
            let normal_c = normal_h.clone();
            let diffuse_c = diffuse_h.clone();
            // The same SPIR-V is handed to every a-trous pass; the graph's
            // pipeline cache dedups them to one `vk::Pipeline`.
            let pass = ComputeRenderPassBuilder::default()
                .common(common.build())
                .shaders(ComputeShaders::new(vec![ShaderSource::Spirv(spirv.to_vec())], 0, "main"))
                .generate_render(rg, [width.div_ceil(16), height.div_ceil(16), 1], move |tr| {
                    let pack = |i: u32| -> [u32; 2] { [i, 0] };
                    Ok(vulkan_abstraction::DenoiseHeapPushConstant {
                        temporal_result: pack(tr.image(&read_h_c)?.storage_slot()),
                        depth: pack(tr.image(&depth_c)?.sampled_slot()),
                        normal: pack(tr.image(&normal_c)?.sampled_slot()),
                        diffuse: pack(tr.image(&diffuse_c)?.sampled_slot()),
                        spatial_output: pack(tr.image(&write_h_c)?.storage_slot()),
                        frame_count,
                        step_width,
                        width,
                        height,
                    })
                })
                .map_err(|e| SrError::new_custom(format!("denoise pass builder failed: {e}")))?;
            rg.add_render_pass(pass);
        }
        Ok(())
    }

    /// Postprocess graph node (heap + Slang): tonemap + gamma the final denoise
    /// output into the post-process image.
    #[allow(clippy::too_many_arguments)]
    fn add_postprocess_pass(
        rg: &mut RenderGraph,
        spirv: &[u8],
        denoise_in_h: Handle<vulkan_abstraction::Image>,
        postprocess_out_h: Handle<vulkan_abstraction::Image>,
        width: u32,
        height: u32,
        exposure: f32,
    ) -> SrResult<()> {
        let mut common = PassCommonDataBuilder::new(rg, "postprocess");
        common.read(&denoise_in_h, vk_sync::AccessType::ComputeShaderReadOther)?;
        common.write(&postprocess_out_h, vk_sync::AccessType::ComputeShaderWrite)?;

        let pass = ComputeRenderPassBuilder::default()
            .common(common.build())
            .shaders(ComputeShaders::new(vec![ShaderSource::Spirv(spirv.to_vec())], 0, "main"))
            .generate_render(rg, [width.div_ceil(16), height.div_ceil(16), 1], move |tr| {
                Ok(PostprocessPushConstant {
                    input_idx: tr.image(&denoise_in_h)?.storage_slot(),
                    _input_pad: 0,
                    output_idx: tr.image(&postprocess_out_h)?.storage_slot(),
                    _output_pad: 0,
                    exposure,
                })
            })
            .map_err(|e| SrError::new_custom(format!("postprocess pass builder failed: {e}")))?;
        rg.add_render_pass(pass);
        Ok(())
    }

    pub fn render_to_host_memory(
        &mut self,
        camera: &Camera,
        instances: &[(K, Vec<vk::TransformMatrixKHR>)],
    ) -> SrResult<Vec<u8>> {
        let mut dst_image = vulkan_abstraction::Image::new(
            Rc::clone(&self.core),
            self.image_extent,
            self.image_format,
            vk::ImageTiling::LINEAR,
            gpu_allocator::MemoryLocation::GpuToCpu,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            "mapped sunray output image",
        )?;

        // Warm-up frames: ReSTIR temporal reuse + the a-trous denoise need a few
        // frames of accumulated history before the output is meaningful — a single
        // frame produces near-black because the initial temporal history is empty
        // and the first ReSTIR audition is just one RIS candidate per pixel.
        const WARMUP_FRAMES: u32 = 16;
        for _ in 0..WARMUP_FRAMES {
            // Offscreen: the graph blits into `dst_image` (leaving it GENERAL) as
            // part of the frame submission, so `wait_frame` (graph timeline = N) now
            // covers the blit too — no separate fence.
            let frame = self.render(camera, instances, FrameOutput::Offscreen { image: &dst_image })?;
            self.wait_frame(frame)?;
        }

        dst_image.get_raw_image_data_with_no_padding()
    }

    pub fn core(&self) -> &Rc<vulkan_abstraction::Core> {
        &self.core
    }
}

impl<K: Hash + Eq + Copy + 'static> Drop for Renderer<K> {
    fn drop(&mut self) {
        // Stop the frame watcher before any Vulkan object it touches (the
        // timeline semaphore, the device) can be destroyed by the field drops
        // that follow this body.
        self.frame_watcher_shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.frame_watcher.take() {
            let _ = handle.join();
        }

        match self.core().graphics_queue().wait_idle() {
            Ok(()) => {}
            Err(e) => match e.get_source() {
                ErrorSource::Vulkan(e) => {
                    log::warn!("VkQueueWaitIdle s returned {e:?} in sunray::Renderer::drop")
                }
                _ => log::warn!("VkQueueWaitIdle returned {e} in sunray::Renderer::drop"),
            },
        }
        let callbacks = std::mem::take(&mut self.end_of_frame_callbacks);
        // The queue is idle: every pending frame is complete, so all deferred
        // deallocation callbacks can run now.
        for (_, callback) in callbacks {
            callback(self);
        }
    }
}
