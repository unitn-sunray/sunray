//! Resources for the Bevy integration.
//!
//! Three live in the **render world**:
//! - [`SunrayWindows`] / [`ExtractedCamera`] / [`ExtractedScene`] — plain Send
//!   snapshots written by the extract systems.
//! - [`SunrayRenderState`] — the renderer + window-bound GPU objects. It is a
//!   **NonSend** resource because [`crate::Renderer`] is `Arc`-based (`!Send`);
//!   the single-threaded render SubApp guarantees it's only touched on the main
//!   thread (see `docs/bevy_integration.md`).
//!
//! One lives in the **main world**: [`SunrayScene`], the user-facing handle for
//! requesting a glTF load.

use std::collections::HashMap;

use ash::vk;
use bevy_asset::UntypedAssetId;
use bevy_ecs::prelude::*;
use bevy_window::RawHandleWrapper;

use crate::Renderer;

/// Main-world resource: which scene to load. Set [`gltf_path`](Self::request)
/// and the render world will (re)load it when the generation changes.
#[derive(Resource, Default)]
pub struct SunrayScene {
    pub(crate) gltf_path: Option<String>,
    pub(crate) generation: u64,
}

impl SunrayScene {
    /// Request that `path` be loaded (replacing any current scene) on the next
    /// render frame.
    pub fn request<P: Into<String>>(&mut self, path: P) {
        self.gltf_path = Some(path.into());
        self.generation += 1;
    }

    /// Request that the current scene (if any) be unloaded on the next render
    /// frame, leaving only entity-driven instances.
    pub fn clear(&mut self) {
        self.gltf_path = None;
        self.generation += 1;
    }

    /// Path of the currently requested scene, if any.
    pub fn current(&self) -> Option<&str> {
        self.gltf_path.as_deref()
    }

    /// Convenience constructor for `App::insert_resource`.
    pub fn with_gltf<P: Into<String>>(path: P) -> Self {
        Self {
            gltf_path: Some(path.into()),
            generation: 1,
        }
    }
}

/// Extracted (Send) snapshot of the Bevy windows the renderer cares about.
#[derive(Resource, Default)]
pub struct SunrayWindows {
    pub windows: HashMap<Entity, SunrayWindowInfo>,
}

pub struct SunrayWindowInfo {
    /// `RawHandleWrapper` is force-`Send + Sync`; only *using* the handle is
    /// thread-restricted, which we honor by creating the surface in a NonSend
    /// (main-thread) system.
    pub handle: RawHandleWrapper,
    pub physical_size: (u32, u32),
    pub size_changed: bool,
    pub is_primary: bool,
}

/// Extracted (Send) camera for the current frame. Stored as plain components so
/// the resource stays `Send` without depending on `sunray::Camera: Clone`.
#[derive(Resource, Default, Clone, Copy)]
pub struct ExtractedCamera {
    pub present: bool,
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub fov_y_degrees: f32,
}

/// Extracted (Send) copy of the scene-load request.
#[derive(Resource, Default)]
pub struct ExtractedScene {
    pub gltf_path: Option<String>,
    pub generation: u64,
}

/// Extracted (Send) per-frame instance list built from
/// [`super::SunrayInstance`] entities. Rebuilt from scratch every extract, so
/// entity adds / removes / transform edits are picked up with no retained
/// state or change tracking.
#[derive(Resource, Default)]
pub struct ExtractedInstances {
    /// `(BLAS index in the loaded scene, world transform)`, one per
    /// [`super::SunrayInstance`] entity. When non-empty these **replace** the
    /// scene's baked instances (they re-place the same scene BLASes).
    pub instances: Vec<(usize, vk::TransformMatrixKHR)>,
    /// `(mesh asset id, world transform)`, one per
    /// [`super::SunrayMeshInstance`] entity (runtime-built BLAS keyed by the
    /// asset id — see `asset.rs`). Always **additive** on top of the scene.
    pub asset_instances: Vec<(UntypedAssetId, vk::TransformMatrixKHR)>,
}

/// The renderer (which owns its surface + swapchain internally). **NonSend**
/// (see module docs).
#[derive(Default)]
pub struct SunrayRenderState {
    pub renderer: Option<Renderer<UntypedAssetId>>,
    /// Window entity that owns the renderer (single-window).
    pub owner: Option<Entity>,

    /// Per-frame instance list of the currently loaded scene, returned by
    /// `load_gltf` and handed to `render_to_swapchain` each frame. Lives here
    /// (the caller side) — the renderer retains nothing about instances.
    pub scene_instances: Vec<(UntypedAssetId, Vec<vk::TransformMatrixKHR>)>,
    /// BLAS keys of the loaded scene in load order; `SunrayInstance::blas_index`
    /// indexes this list when entity-driven instances are active.
    pub scene_blas_keys: Vec<UntypedAssetId>,
    /// Asset group of the currently loaded scene (for bulk unload on reload).
    pub scene_group: Option<u64>,

    pub loaded_scene_generation: u64,
    pub image_format: vk::Format,

    /// egui GPU paint backend, built lazily on the first frame that has an
    /// `ExtractedEgui` resource (i.e. when `SunrayEguiPlugin` is active).
    pub egui_paint: Option<super::egui_paint::EguiPaint>,
}
