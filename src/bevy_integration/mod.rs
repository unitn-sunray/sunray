//! Bevy 0.19 integration for the sunray ray-tracer.
//!
//! This module replaces Bevy's stock wgpu `RenderPlugin` with a backend that
//! drives [`crate::Renderer`] directly. It keeps Bevy's ECS, windowing (winit),
//! input, time and transforms, and reuses `bevy_render::extract_plugin::ExtractPlugin`
//! for the `RenderApp` SubApp + extraction bridge. Rendering is **single-threaded**:
//! the render SubApp runs on the main thread and the renderer lives in a NonSend
//! resource. ([`crate::Renderer`] is `Send` but not `Sync`, and the swapchain must
//! be presented from the thread that owns the window.)
//!
//! See `docs/bevy_integration.md` for the full architecture and
//! `examples/bevy_app` for usage.
//!
//! # Quick start
//! ```ignore
//! App::new()
//!     .add_plugins((/* minimal Bevy plugins, NOT RenderPlugin */))
//!     .add_plugins((SunrayRenderPlugin::default(), SunrayEguiPlugin))
//!     .insert_resource(SunrayScene::with_gltf("examples/assets/Room.glb"))
//!     .add_systems(Startup, |mut commands: Commands| {
//!         commands.spawn((Transform::from_xyz(0.0, 2.0, 10.0), SunrayCamera::default()));
//!     })
//!     .run();
//! ```

// A Bevy system's parameter list *is* its ECS dependency declaration, and a
// `Query`'s type *is* its filter — neither can be shortened into a struct
// without losing what the scheduler reads. Both lints misfire on every system.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

mod asset;
mod camera;
pub(crate) mod egui_paint;
mod egui_support;
mod gltf_scene;
mod instance;
mod plugin;
mod state;
mod surface;
mod systems;

pub use asset::{ExtractedMeshAssets, SunrayMaterial, SunrayMeshInstance};
pub use camera::SunrayCamera;
pub use egui_support::{EguiContext, EguiFrameOutput, ExtractedEgui};
pub use gltf_scene::{SunrayGltfFailed, SunrayGltfPlugin, SunrayGltfScene, SunrayGltfSpawned};
pub use instance::SunrayInstance;
pub use plugin::{SunrayEguiPlugin, SunrayRenderPlugin};
pub use state::{ExtractedCamera, ExtractedInstances, ExtractedScene, SunrayRenderState, SunrayScene, SunrayWindows};

/// Marker type whose `TypeId` brands renderer-generated asset ids, keeping
/// them distinct from ids of real Bevy assets.
struct SunrayGeneratedAsset;

/// The Bevy integration keys the renderer by [`bevy_asset::UntypedAssetId`]
/// (untyped because one key space spans BLASes *and* images — a single typed
/// `AssetId<A>` couldn't cover both). Scene loads that aren't driven by Bevy's
/// asset system (e.g. `load_gltf`) generate deterministic UUID ids from the
/// renderer's [`crate::ResourceKey`]; assets extracted from Bevy later can use
/// their real `AssetId<A>.untyped()` in the same key space.
impl From<crate::ResourceKey> for bevy_asset::UntypedAssetId {
    fn from(key: crate::ResourceKey) -> Self {
        bevy_asset::UntypedAssetId::Uuid {
            type_id: std::any::TypeId::of::<SunrayGeneratedAsset>(),
            uuid: bevy_asset::uuid::Uuid::from_u64_pair(key.group, key.index),
        }
    }
}
