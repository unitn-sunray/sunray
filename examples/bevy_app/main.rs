//! Bevy + sunray example.
//!
//! Builds a Bevy `App` from the **minimal** plugin set (Option B in
//! `docs/bevy_integration.md`) — no `RenderPlugin`, no wgpu — and adds
//! [`SunrayRenderPlugin`] + [`SunrayEguiPlugin`] to render the sunray
//! ray-tracer into the Bevy-managed window.
//!
//! Controls: hold **right mouse** to look, **WASD** to move, **Space/Ctrl** up/down,
//! **Shift** to go faster.
//!
//! Run with:
//! ```text
//! cargo run --release --features bevy --example bevy_app
//! ```
//!
//! egui is fully wired: input + tessellation + extract + a heap+Slang GPU paint
//! pass that overlays the egui window on top of the ray-traced frame (see
//! docs/bevy_integration.md §6).
//!
//! The egui panel lists every `.glb` in `examples/assets` and can load/unload
//! each one in two ways:
//! - **direct** — the renderer's own glTF loader ([`SunrayScene`]); one scene
//!   at a time, full material/texture support;
//! - **bevy** — Bevy's asset pipeline (`bevy_gltf` via [`SunrayGltfScene`]);
//!   async, additive (several scenes can stack), factor-only materials.

// A Bevy system's parameter list is its ECS dependency declaration; see the
// same allow in src/bevy_integration/mod.rs.
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use bevy_a11y::AccessibilityPlugin;
use bevy_app::{App, Startup, TaskPoolPlugin, Update};
use bevy_asset::{AssetPlugin, AssetServer, Assets, Handle};
use bevy_diagnostic::FrameCountPlugin;
use bevy_ecs::prelude::*;
use bevy_gltf::{Gltf, GltfMesh};
use bevy_input::ButtonInput;
use bevy_input::InputPlugin;
use bevy_input::keyboard::KeyCode;
use bevy_input::mouse::{MouseButton, MouseMotion};
use bevy_log::LogPlugin;
use bevy_math::primitives::Cuboid;
use bevy_math::{EulerRot, Quat, Vec2, Vec3};
use bevy_render::mesh::Mesh;
use bevy_time::{Time, TimePlugin};
use bevy_transform::TransformPlugin;
use bevy_transform::components::Transform;
use bevy_window::{Window, WindowPlugin};
use bevy_winit::WinitPlugin;

use sunray::bevy_integration::{
    EguiContext, SunrayCamera, SunrayEguiPlugin, SunrayGltfFailed, SunrayGltfPlugin, SunrayGltfScene, SunrayGltfSpawned,
    SunrayMaterial, SunrayMeshInstance, SunrayRenderPlugin, SunrayScene,
};

/// Directory scanned for `.glb` files; also the Bevy asset root, so the same
/// file name works for both load paths. Absolute, because Bevy resolves a
/// relative asset root against the *executable's* directory while the
/// renderer's direct loader resolves against the CWD.
const ASSET_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets");

/// The `.glb` the BLAS/TLAS stress harness imports (resolved against the asset
/// root, so a bare file name). Its meshes become the churn pool.
const STRESS_GLB: &str = "Room.glb";

/// Simple fly-camera state stored next to the camera's `Transform`.
#[derive(Component)]
struct FlyCam {
    yaw: f32,
    pitch: f32,
}

/// The `.glb` files found in [`ASSET_DIR`] at startup.
#[derive(Resource)]
struct GlbLibrary {
    files: Vec<String>,
}

/// Which library entries are loaded, and through which path.
#[derive(Resource, Default)]
struct SceneManager {
    /// File currently loaded through the renderer's own glTF loader
    /// ([`SunrayScene`]). Single slot — a new direct load replaces it.
    direct: Option<String>,
    /// File → root entity of scenes loaded through Bevy's asset pipeline
    /// ([`SunrayGltfScene`]). These stack additively.
    bevy_loaded: HashMap<String, Entity>,
}

/// BLAS/TLAS allocation stress harness. Imports [`STRESS_GLB`] through Bevy's
/// asset pipeline (loading its meshes on the CPU), then — when enabled —
/// randomly spawns/despawns one `SunrayMeshInstance` per mesh every frame. Each
/// unique mesh asset maps to one BLAS (built on first reference, freed when its
/// last instance is despawned — see `bevy_integration/asset.rs`), and the TLAS
/// is rebuilt every frame from the live instance set, so the churn hammers the
/// BLAS allocator and TLAS-rebuild path.
#[derive(Resource)]
struct StressTest {
    /// Handle kept alive so the glTF (and its meshes) stay CPU-resident.
    gltf: Handle<Gltf>,
    /// One `Handle<Mesh>` per glTF mesh primitive (harvested once, on load).
    pool: Vec<Handle<Mesh>>,
    /// The currently-spawned instance entity for `pool[i]`, or `None` if unloaded.
    slots: Vec<Option<Entity>>,
    /// Toggled from the egui panel; off tears everything down.
    enabled: bool,
    /// Set once `pool` has been filled from the loaded glTF.
    harvested: bool,
    /// Set after the first enabled frame spawned the full set ("load all").
    primed: bool,
}

/// Spawn one churn instance of `mesh` at a random position with a random color.
fn spawn_stress_instance(commands: &mut Commands, mesh: &Handle<Mesh>) -> Entity {
    let pos = Vec3::new(
        rand::random_range(-6.0f32..6.0),
        rand::random_range(0.0f32..5.0),
        rand::random_range(-6.0f32..6.0),
    );
    let material = SunrayMaterial {
        base_color: [
            rand::random_range(0.1f32..1.0),
            rand::random_range(0.1f32..1.0),
            rand::random_range(0.1f32..1.0),
            1.0,
        ],
        roughness: 0.5,
        ..Default::default()
    };
    commands
        .spawn((
            Transform::from_translation(pos),
            SunrayMeshInstance { mesh: mesh.clone() },
            material,
        ))
        .id()
}

/// Drive the stress harness: harvest the glTF's meshes once loaded, then (when
/// enabled) load all on the first frame and randomly load/unload a slice of them
/// every frame after.
fn stress_system(
    mut commands: Commands,
    mut stress: ResMut<StressTest>,
    gltfs: Res<Assets<Gltf>>,
    gltf_meshes: Res<Assets<GltfMesh>>,
) {
    // 1. Harvest the glb's meshes once its asset (and mesh subassets) have loaded.
    if !stress.harvested {
        let Some(gltf) = gltfs.get(&stress.gltf) else {
            return;
        };
        for mesh_handle in &gltf.meshes {
            if let Some(gm) = gltf_meshes.get(mesh_handle) {
                for prim in &gm.primitives {
                    stress.pool.push(prim.mesh.clone());
                }
            }
        }
        stress.slots = vec![None; stress.pool.len()];
        stress.harvested = true;
        log::info!("stress: harvested {} meshes from {STRESS_GLB} (CPU-side)", stress.pool.len());
        return;
    }

    let n = stress.pool.len();

    // 2. Disabled: tear everything down (frees every stressed BLAS) and idle.
    if !stress.enabled {
        if stress.primed {
            for slot in &mut stress.slots {
                if let Some(e) = slot.take() {
                    commands.entity(e).despawn();
                }
            }
            stress.primed = false;
        }
        return;
    }
    if n == 0 {
        return;
    }

    // 3. First enabled frame: load ALL assets (spawn every mesh as an instance).
    if !stress.primed {
        for i in 0..n {
            stress.slots[i] = Some(spawn_stress_instance(&mut commands, &stress.pool[i]));
        }
        stress.primed = true;
        log::info!("stress: primed — spawned all {n} instances");
        return;
    }

    // 4. Every frame after: randomly toggle ~a quarter of the slots — spawn the
    //    unloaded, despawn the loaded — so BLASes churn and the TLAS rebuilds.
    let churn = ((n as f32 * 0.25).ceil() as usize).max(1);
    for _ in 0..churn {
        let i = rand::random_range(0..n);
        if let Some(e) = stress.slots[i].take() {
            commands.entity(e).despawn();
        } else {
            stress.slots[i] = Some(spawn_stress_instance(&mut commands, &stress.pool[i]));
        }
    }
}

fn scan_glb_library() -> Vec<String> {
    let mut files: Vec<String> = std::fs::read_dir(ASSET_DIR)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            name.to_ascii_lowercase().ends_with(".glb").then_some(name)
        })
        .collect();
    files.sort();
    files
}

fn main() {
    App::new()
        // Minimal Bevy: ECS + windowing + input + time + transforms. No render stack.
        .add_plugins((
            LogPlugin::default(),
            TaskPoolPlugin::default(),
            // Assets so `SunrayMeshInstance` can reference `Mesh` assets;
            // `SunrayRenderPlugin` registers `Assets<Mesh>` itself. The asset
            // root points at the glb directory so the Bevy load path can use
            // bare file names.
            AssetPlugin {
                file_path: ASSET_DIR.into(),
                ..Default::default()
            },
            FrameCountPlugin,
            TimePlugin,
            InputPlugin,
            WindowPlugin {
                primary_window: Some(Window {
                    title: "sunray + bevy".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            AccessibilityPlugin,
            WinitPlugin::default(),
            TransformPlugin,
        ))
        // Our custom ash renderer + egui input/extract layer + the Bevy-asset
        // glTF load path (`SunrayGltfScene` entities).
        .add_plugins((SunrayRenderPlugin::default(), SunrayEguiPlugin, SunrayGltfPlugin))
        // Ask the renderer to load this glTF once the device exists (the
        // "direct" path; the egui panel can swap/unload it at runtime).
        .insert_resource(SunrayScene::with_gltf(format!("{ASSET_DIR}/Room.glb")))
        .insert_resource(GlbLibrary {
            files: scan_glb_library(),
        })
        .insert_resource(SceneManager {
            direct: Some("Room.glb".into()),
            ..Default::default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (fly_cam, ui_system, stress_system))
        .run();
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, asset_server: Res<AssetServer>) {
    // Import the stress-test glb through Bevy's asset pipeline (loads on the CPU);
    // `stress_system` harvests its meshes once ready. Off until toggled in egui.
    commands.insert_resource(StressTest {
        gltf: asset_server.load(STRESS_GLB),
        pool: Vec::new(),
        slots: Vec::new(),
        enabled: false,
        harvested: false,
        primed: false,
    });

    // The camera is an ordinary ECS entity: a Transform + SunrayCamera marker.
    commands.spawn((
        Transform::from_xyz(0.0, 2.0, 10.0),
        SunrayCamera { fov_y_degrees: 45.0 },
        FlyCam { yaw: 0.0, pitch: 0.0 },
    ));

    // Runtime mesh assets → BLASes built on the fly, rendered on top of the
    // glTF scene. The material is per mesh *asset* (one BLAS each), so the two
    // entities sharing `cube` share the red material.
    let cube = meshes.add(Mesh::from(Cuboid::new(1.0, 1.0, 1.0)));
    let red = SunrayMaterial {
        base_color: [0.8, 0.15, 0.15, 1.0],
        roughness: 0.4,
        ..Default::default()
    };
    commands.spawn((
        Transform::from_xyz(2.0, 0.5, 0.0),
        SunrayMeshInstance { mesh: cube.clone() },
        red,
    ));
    commands.spawn((Transform::from_xyz(-2.0, 0.5, 0.0), SunrayMeshInstance { mesh: cube }, red));

    // A small emissive cube acting as an extra light (NEE picks up its
    // triangles through the runtime emissive-triangle path).
    let lamp = meshes.add(Mesh::from(Cuboid::new(0.3, 0.3, 0.3)));
    commands.spawn((
        Transform::from_xyz(0.0, 2.5, 0.0),
        SunrayMeshInstance { mesh: lamp },
        SunrayMaterial {
            emissive: [1.0, 0.9, 0.7],
            emissive_strength: 20.0,
            ..Default::default()
        },
    ));
}

fn fly_cam(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
) {
    let Ok((mut transform, mut cam)) = query.single_mut() else {
        return;
    };
    let dt = time.delta_secs();

    // Mouse look while the right button is held.
    if buttons.pressed(MouseButton::Right) {
        let mut delta = Vec2::ZERO;
        for ev in motion.read() {
            delta += ev.delta;
        }
        cam.yaw -= delta.x * 0.002;
        cam.pitch -= delta.y * 0.002;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        cam.pitch = cam.pitch.clamp(-limit, limit);
    } else {
        motion.clear();
    }
    transform.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);

    let forward = *transform.forward();
    let right = *transform.right();
    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        dir += forward;
    }
    if keys.pressed(KeyCode::KeyS) {
        dir -= forward;
    }
    if keys.pressed(KeyCode::KeyD) {
        dir += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        dir -= right;
    }
    if keys.pressed(KeyCode::Space) {
        dir += Vec3::Y;
    }
    if keys.pressed(KeyCode::ControlLeft) {
        dir -= Vec3::Y;
    }

    let speed = if keys.pressed(KeyCode::ShiftLeft) { 9.0 } else { 3.0 };
    if dir != Vec3::ZERO {
        transform.translation += dir.normalize() * speed * dt;
    }
}

fn ui_system(
    egui_ctx: Res<EguiContext>,
    library: Res<GlbLibrary>,
    mut manager: ResMut<SceneManager>,
    mut scene: ResMut<SunrayScene>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut stress: ResMut<StressTest>,
    gltf_roots: Query<(Has<SunrayGltfSpawned>, Has<SunrayGltfFailed>), With<SunrayGltfScene>>,
) {
    egui::Window::new("sunray + bevy").show(egui_ctx.ctx(), |ui| {
        ui.label("Hardware ray tracing, driven by Bevy ECS.");
        ui.label("Right mouse: look · WASD: move · Space/Ctrl: up/down · Shift: faster");
        ui.separator();

        ui.heading("Scenes");
        ui.label("direct: renderer's own glTF loader (one scene at a time)");
        ui.label("bevy: AssetServer + bevy_gltf → mesh-asset BLASes (additive)");
        ui.separator();

        if library.files.is_empty() {
            ui.label(format!("no .glb files found in {ASSET_DIR}"));
        }
        for file in &library.files {
            ui.horizontal(|ui| {
                ui.label(file);

                // Direct path: single slot, renderer-side load.
                if manager.direct.as_deref() == Some(file) {
                    if ui.button("Unload direct").clicked() {
                        scene.clear();
                        manager.direct = None;
                    }
                } else if ui.button("Load direct").clicked() {
                    scene.request(format!("{ASSET_DIR}/{file}"));
                    manager.direct = Some(file.clone());
                }

                // Bevy-asset path: one root entity per file, despawn = unload.
                if let Some(&root) = manager.bevy_loaded.get(file) {
                    let status = match gltf_roots.get(root) {
                        Ok((true, _)) => "loaded",
                        Ok((_, true)) => "failed",
                        Ok(_) => "loading…",
                        Err(_) => "?",
                    };
                    if ui.button("Unload bevy").clicked() {
                        commands.entity(root).despawn();
                        manager.bevy_loaded.remove(file);
                    } else {
                        ui.label(status);
                    }
                } else if ui.button("Load bevy").clicked() {
                    let root = commands
                        .spawn((
                            Transform::IDENTITY,
                            SunrayGltfScene {
                                gltf: asset_server.load(file.clone()),
                            },
                        ))
                        .id();
                    manager.bevy_loaded.insert(file.clone(), root);
                }
            });
        }

        ui.separator();
        ui.heading("BLAS/TLAS stress test");
        ui.label(format!("imports {STRESS_GLB}, churns its meshes as runtime BLASes"));
        if stress.harvested {
            let loaded = stress.slots.iter().filter(|s| s.is_some()).count();
            ui.checkbox(&mut stress.enabled, "Churn: random load/unload every frame");
            ui.label(format!("{loaded}/{} meshes instanced", stress.pool.len()));
        } else {
            ui.label(format!("loading {STRESS_GLB}…"));
        }

        ui.separator();
        ui.label("This panel is painted by a heap+Slang egui pass over the RT frame.");
    });
}
