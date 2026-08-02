use std::collections::HashSet;
use std::io;
use std::io::Read;

use ash::vk;
use nalgebra as na;
use rand::random_range;
use std::time::Instant;
use sunray::{
    ResourceKey,
    camera::Camera,
    error::{ErrorSource, SrResult},
    utils::na_mat4_to_vk_transform,
};
use winit::dpi::LogicalSize;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, MouseButton, WindowEvent},
    event_loop::{self, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle_05::{HasRawDisplayHandle, HasRawWindowHandle},
    window::{CursorGrabMode, Window},
};

mod utils;

struct AppResources {
    /// The renderer owns the surface, the swapchain, and all present plumbing.
    pub renderer: sunray::Renderer,
}

struct App {
    window: Option<Window>,
    resources: Option<AppResources>,

    start_time: Option<std::time::SystemTime>,
    frame_count: u64,
    last_fps_check: Option<Instant>,
    frames_since_check: u32,

    // --- CAMERA STATE ---
    camera_pos: na::Point3<f32>,
    camera_yaw: f32,
    camera_pitch: f32,
    keys_down: HashSet<KeyCode>,
    mouse_captured: bool,
    last_frame_time: Option<Instant>,

    // --- PER-FRAME SCENE STATE (owned by the caller, handed to `render_to_swapchain`) ---
    scene_instances: Vec<(ResourceKey, Vec<vk::TransformMatrixKHR>)>,
    /// `(blas entry, transform entry)` of the duplicate spawned by the runtime test.
    spawned_instance: Option<(usize, usize)>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            resources: None,
            start_time: None,
            frame_count: 0,
            last_fps_check: None,
            frames_since_check: 0,

            // Start looking down the -Z axis slightly above the floor
            camera_pos: na::Point3::new(0.0, 2.0, 10.0),
            camera_yaw: -std::f32::consts::FRAC_PI_2,
            camera_pitch: 0.0,
            keys_down: HashSet::new(),
            mouse_captured: false,
            last_frame_time: None,

            scene_instances: Vec::new(),
            spawned_instance: None,
        }
    }
}

impl App {
    fn build_resources(&mut self, size: (u32, u32)) -> SrResult<()> {
        self.resources = None;

        let display_handle = self.window.as_ref().unwrap().raw_display_handle().clone();
        let window_handle = self.window.as_ref().unwrap().raw_window_handle().clone();

        let instance_exts = utils::enumerate_required_extensions(display_handle)?;

        let create_surface = move |entry: &ash::Entry, instance: &ash::Instance| -> SrResult<vk::SurfaceKHR> {
            crate::utils::create_surface(entry, instance, display_handle, window_handle, None)
        };

        // Build the sunray renderer; it creates and owns the surface, the
        // swapchain, and all present plumbing internally.
        let mut renderer = sunray::Renderer::new_with_surface(size, vk::Format::R8G8B8A8_SRGB, instance_exts, &create_surface)?;

        // The scene's instance list belongs to the caller: keep it here and
        // pass it to `render_to_swapchain` every frame.
        let (_scene_group, scene_instances) = renderer.load_gltf("examples/assets/ReflectionRoom.glb")?;
        self.scene_instances = scene_instances;
        log::info!("Loaded {} unique BLASes from scene", self.scene_instances.len());

        self.resources = Some(AppResources { renderer });

        Ok(())
    }

    fn resize(&mut self, size: (u32, u32)) -> SrResult<()> {
        // The renderer resizes its internal images and rebuilds its swapchain itself.
        self.res_mut().renderer.resize(size)
    }

    fn time_elapsed(&self) -> f32 {
        std::time::SystemTime::now()
            .duration_since(self.start_time.unwrap())
            .unwrap()
            .as_millis() as f32
            / 1000.0
    }

    fn draw(&mut self) -> sunray::error::SrResult<()> {
        let now = Instant::now();
        let dt = if let Some(last) = self.last_frame_time {
            now.duration_since(last).as_secs_f32()
        } else {
            0.016
        };
        self.last_frame_time = Some(now);

        // --- UPDATE CAMERA LOGIC ---
        let base_speed = 3.0;
        let speed = if self.keys_down.contains(&KeyCode::ShiftLeft) {
            base_speed * 3.0
        } else {
            base_speed
        };
        let move_dist = speed * dt;

        let forward = na::Vector3::new(
            self.camera_yaw.cos() * self.camera_pitch.cos(),
            self.camera_pitch.sin(),
            self.camera_yaw.sin() * self.camera_pitch.cos(),
        )
        .normalize();

        let right = forward.cross(&na::Vector3::new(0.0, 1.0, 0.0)).normalize();

        if self.keys_down.contains(&KeyCode::KeyW) {
            self.camera_pos += forward * move_dist;
        }
        if self.keys_down.contains(&KeyCode::KeyS) {
            self.camera_pos -= forward * move_dist;
        }
        if self.keys_down.contains(&KeyCode::KeyD) {
            self.camera_pos += right * move_dist;
        }
        if self.keys_down.contains(&KeyCode::KeyA) {
            self.camera_pos -= right * move_dist;
        }
        if self.keys_down.contains(&KeyCode::Space) {
            self.camera_pos += na::Vector3::new(0.0, 1.0, 0.0) * move_dist;
        }
        if self.keys_down.contains(&KeyCode::ControlLeft) {
            self.camera_pos -= na::Vector3::new(0.0, 1.0, 0.0) * move_dist;
        }

        let target = self.camera_pos + forward;

        let camera = Camera::default()
            .set_position(self.camera_pos)
            .set_target(target)
            .set_fov_y(45.0);

        //self.update_runtime_test();

        // The camera and the instance list are per-frame inputs: the renderer
        // retains nothing about them across frames.
        self.resources
            .as_mut()
            .unwrap()
            .renderer
            .render_to_swapchain(&camera, &self.scene_instances)?;

        self.frames_since_check += 1;

        if let Some(last_check) = self.last_fps_check {
            let elapsed = now.duration_since(last_check);

            if elapsed.as_secs() >= 1 {
                let fps = self.frames_since_check as f32 / elapsed.as_secs_f32();
                if let Some(window) = &self.window {
                    window.set_title(&format!("Sunray Vulkan - FPS: {:.1}", fps));
                }
                self.last_fps_check = Some(now);
                self.frames_since_check = 0;
            }
        }

        self.frame_count += 1;
        self.window.as_ref().unwrap().request_redraw();
        Ok(())
    }

    /// Exercises runtime move / add / remove of instances by mutating the
    /// caller-owned instance list — no renderer involvement at all.
    #[allow(unused)]
    fn update_runtime_test(&mut self) {
        let frame = self.frame_count;
        if self.scene_instances.is_empty() {
            return;
        }

        // Animate the first instance: orbit around Y every frame.
        if let Some(first_transform) = self.scene_instances[0].1.first_mut() {
            let angle = frame as f32 * 0.0001;
            let (s, c) = angle.sin_cos();
            let radius = 3.0_f32;
            let translation = na::Translation3::new(c * radius, 0.0, s * radius);
            let rotation = na::UnitQuaternion::from_axis_angle(&na::Vector3::y_axis(), angle);
            *first_transform = na_mat4_to_vk_transform((translation * rotation).to_homogeneous());
        }

        // At frame 120 spawn a duplicate of a random BLAS offset to the side.
        if frame % 120 == 1 && self.spawned_instance.is_none() {
            let blas_entry = random_range(0..self.scene_instances.len());
            let offset = na::Translation3::new(4.0, 0.0, 0.0).to_homogeneous();
            let transforms = &mut self.scene_instances[blas_entry].1;
            transforms.push(na_mat4_to_vk_transform(offset));
            self.spawned_instance = Some((blas_entry, transforms.len() - 1));
            log::info!("[runtime test] spawned duplicate instance of BLAS entry {blas_entry}");
        }

        // At frame 240 remove the spawned duplicate.
        if frame % 240 == 1 {
            if let Some((blas_entry, transform_entry)) = self.spawned_instance.take() {
                self.scene_instances[blas_entry].1.remove(transform_entry);
                log::info!("[runtime test] removed duplicate instance of BLAS entry {blas_entry}");
            }
        }
    }

    fn handle_event(&mut self, event_loop: &event_loop::ActiveEventLoop, event: winit::event::WindowEvent) -> SrResult<()> {
        match event {
            WindowEvent::CloseRequested => {
                let run_time = {
                    let end_time = std::time::SystemTime::now();
                    end_time.duration_since(self.start_time.unwrap()).unwrap().as_millis() as f32 / 1000.0
                };
                let fps = self.frame_count as f32 / run_time;
                log::info!("Frames per second: {fps}");
                unsafe { self.res().renderer.core().device().inner().device_wait_idle() }?;
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.draw()?;
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                if size.width != 0 && size.height != 0 {
                    self.resize(size.into()).unwrap();
                }
            }
            // --- KEYBOARD TRACKING & MOUSE LOCK EXIT ---
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(keycode) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if keycode == KeyCode::Escape {
                                self.mouse_captured = false;
                                if let Some(window) = &self.window {
                                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                                    window.set_cursor_visible(true);
                                }
                            } else {
                                self.keys_down.insert(keycode);
                            }
                        }
                        ElementState::Released => {
                            self.keys_down.remove(&keycode);
                        }
                    }
                }
            }
            // --- MOUSE LOCK ACTIVATION ---
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_captured = true;
                if let Some(window) = &self.window {
                    // Confined works best on Windows, Locked works best on Mac/Linux. We fallback if one fails.
                    let _ = window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));
                    window.set_cursor_visible(false);
                }
            }
            _ => (),
        }
        Ok(())
    }

    fn handle_srresult(&mut self, event_loop: &event_loop::ActiveEventLoop, result: SrResult<()>) {
        if let Err(e) = result {
            if let ErrorSource::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR) = e.get_source() {
                log::warn!("{e}");
            } else {
                log::error!("{e}");
                event_loop.exit();
            }
        }
    }

    fn res_mut(&mut self) -> &mut AppResources {
        self.resources.as_mut().unwrap()
    }

    fn res(&self) -> &AppResources {
        self.resources.as_ref().unwrap()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_inner_size(LogicalSize::new(2650, 1440)))
            .unwrap();
        let window_size = window.inner_size().into();
        self.window = Some(window);

        if self.resources.is_none() {
            let result = self.build_resources(window_size);
            self.handle_srresult(event_loop, result);
        }

        let result = self.resize(window_size);
        self.handle_srresult(event_loop, result);

        self.start_time = Some(std::time::SystemTime::now());
        self.last_fps_check = Some(Instant::now());
        self.frames_since_check = 0;
        self.last_frame_time = Some(Instant::now());

        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let result = self.handle_event(event_loop, event);
        self.handle_srresult(event_loop, result);
    }

    // --- RAW MOUSE MOTION TRACKING ---
    fn device_event(
        &mut self,
        _event_loop: &event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if self.mouse_captured {
            if let DeviceEvent::MouseMotion { delta } = event {
                let sensitivity = 0.002;
                self.camera_yaw += delta.0 as f32 * sensitivity;
                self.camera_pitch -= delta.1 as f32 * sensitivity;

                // Clamp pitch so you don't break your neck looking backward through your legs
                let limit = std::f32::consts::FRAC_PI_2 - 0.01;
                self.camera_pitch = self.camera_pitch.clamp(-limit, limit);
            }
        }
    }
}

fn main() {
    log4rs::config::init_file("examples/log4rs.yaml", log4rs::config::Deserializers::new()).unwrap();

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default_hook(info);
        println!("\nPress Enter to exit...");
        let _ = io::stdin().read(&mut [0u8]);
    }));

    if cfg!(debug_assertions) {
        log::set_max_level(log::LevelFilter::Debug);
    } else {
        log::set_max_level(log::LevelFilter::Warn);
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app).unwrap();
}
