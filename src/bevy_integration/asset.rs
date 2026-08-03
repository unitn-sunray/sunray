//! Runtime mesh-asset loading: Bevy `Mesh` assets → BLASes.
//!
//! Entities carrying [`SunrayMeshInstance`] (+ a `Transform`) reference a Bevy
//! `Mesh` asset directly. Each frame:
//! 1. [`extract_mesh_assets`] (`ExtractSchedule`) collects the referenced asset
//!    ids and converts not-yet-uploaded meshes into the renderer's vertex
//!    format, queueing them in [`ExtractedMeshAssets`]. Meshes still loading
//!    (not yet present in `Assets<Mesh>`) are simply retried next frame.
//! 2. [`upload_mesh_assets`] (`Render`, after `ensure_renderer`, before
//!    `render_frame`) drains the queue into [`crate::Renderer::load_mesh`] —
//!    the BLAS is built at runtime, keyed by the mesh's `UntypedAssetId` (the
//!    same key space scene loads use) — and unloads BLASes whose asset no
//!    longer has any referencing entity.
//! 3. `extract_instances` (systems.rs) emits one instance per entity, keyed by
//!    the asset id; `render_frame` skips instances whose BLAS isn't ready yet.

use std::collections::HashSet;

use bevy_asset::{Assets, Handle, UntypedAssetId};
use bevy_ecs::prelude::*;
use bevy_render::Extract;
use bevy_render::mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues};

use super::state::SunrayRenderState;
use crate::vulkan_abstraction::gltf as sr_gltf;

/// One ray-traced instance of a Bevy `Mesh` asset. Put this on an entity
/// together with a `Transform`; the BLAS is built at runtime (once per asset)
/// when the mesh has finished loading, and the entity renders from then on.
///
/// Unlike [`super::SunrayInstance`] (which re-places the loaded scene's BLASes
/// and therefore *replaces* its baked instances), mesh-asset instances are
/// independent assets and render **additively** on top of whatever scene is
/// active.
#[derive(Component, Clone, Debug)]
pub struct SunrayMeshInstance {
    pub mesh: Handle<Mesh>,
}

/// Optional PBR factors for a runtime-loaded mesh. Textures are not supported
/// on this path (no image set accompanies a raw mesh) — factors only.
///
/// The material is **per mesh asset, not per entity**: the renderer stores one
/// material per BLAS, so the component on the first entity extracted for a
/// given mesh wins.
//TODO per-instance materials need the storage rework flagged in
//     resource_manager.rs (mesh info is keyed per BLAS today).
#[derive(Component, Clone, Copy, Debug)]
pub struct SunrayMaterial {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub emissive_strength: f32,
    pub ior: f32,
    pub transmission: f32,
}

impl Default for SunrayMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0, 0.0, 0.0],
            emissive_strength: 1.0,
            ior: 1.5,
            transmission: 0.0,
        }
    }
}

impl SunrayMaterial {
    fn to_gltf(self) -> sr_gltf::Material {
        sr_gltf::Material {
            pbr_metallic_roughness_properties: sr_gltf::PbrMetallicRoughnessProperties {
                base_color_factor: self.base_color,
                metallic_factor: self.metallic,
                roughness_factor: self.roughness,
                base_color_texture_index: None,
                metallic_roughness_texture_index: None,
            },
            normal_texture_index: None,
            occlusion_texture_index: None,
            emissive_factor: self.emissive,
            emissive_strength: self.emissive_strength,
            emissive_texture_index: None,
            alpha_mode: gltf::material::AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            transmission_factor: self.transmission,
            ior: self.ior,
        }
    }
}

/// A mesh converted to the renderer's vertex format, awaiting BLAS creation.
pub struct PendingMesh {
    pub id: UntypedAssetId,
    pub vertices: Vec<sr_gltf::Vertex>,
    pub indices: Vec<u32>,
    pub material: sr_gltf::Material,
}

/// Render-world bookkeeping for runtime mesh-asset → BLAS uploads.
#[derive(Resource, Default)]
pub struct ExtractedMeshAssets {
    /// Converted meshes waiting for GPU upload (drained by [`upload_mesh_assets`]).
    pub pending: Vec<PendingMesh>,
    /// Asset ids whose BLAS is live in the renderer. Written by the upload
    /// system; read by extraction (skip re-converting) and by `render_frame`
    /// (skip instances whose BLAS isn't ready yet).
    pub loaded: HashSet<UntypedAssetId>,
    /// Asset ids referenced by at least one entity this frame; BLASes of
    /// unreferenced assets are unloaded.
    pub referenced: HashSet<UntypedAssetId>,
    /// Conversion/upload failures, remembered so they don't retry (and re-log)
    /// every frame. Cleared for ids nobody references anymore.
    pub failed: HashSet<UntypedAssetId>,
}

/// Collect the mesh assets referenced by `SunrayMeshInstance` entities and
/// convert the ones whose BLAS doesn't exist yet. Pure CPU-side extraction —
/// the GPU upload happens in [`upload_mesh_assets`].
//TODO react to `AssetEvent::Modified` so editing a Mesh asset rebuilds its BLAS
//     (today a mesh is converted once and kept until unreferenced).
pub(crate) fn extract_mesh_assets(
    mut out: ResMut<ExtractedMeshAssets>,
    // `Option`: apps without `AssetPlugin` have no `Assets<Mesh>` — the system
    // then just tracks references and never converts anything.
    meshes: Extract<Option<Res<Assets<Mesh>>>>,
    query: Extract<Query<(&SunrayMeshInstance, Option<&SunrayMaterial>)>>,
) {
    let ExtractedMeshAssets {
        pending,
        loaded,
        referenced,
        failed,
    } = &mut *out;

    referenced.clear();
    // Ids already queued (possibly from an earlier frame, while the renderer
    // didn't exist yet) must not be converted twice.
    let mut queued: HashSet<UntypedAssetId> = pending.iter().map(|p| p.id).collect();

    let meshes = meshes.as_deref();
    for (mesh_instance, material) in &query {
        let id = mesh_instance.mesh.id().untyped();
        referenced.insert(id);
        if loaded.contains(&id) || failed.contains(&id) || queued.contains(&id) {
            continue;
        }
        let Some(assets) = meshes else {
            continue;
        };
        // Not in `Assets` yet = still loading; retried next frame.
        let Some(mesh) = assets.get(&mesh_instance.mesh) else {
            continue;
        };
        match convert_mesh(mesh) {
            Ok((vertices, indices)) => {
                let material = material.copied().unwrap_or_default().to_gltf();
                pending.push(PendingMesh {
                    id,
                    vertices,
                    indices,
                    material,
                });
                queued.insert(id);
            }
            Err(why) => {
                log::warn!("sunray: mesh asset {id:?} cannot become a BLAS: {why}");
                failed.insert(id);
            }
        }
    }

    // Forget failures of assets nobody references anymore, so a fixed and
    // re-spawned asset gets another attempt.
    failed.retain(|id| referenced.contains(id));
}

/// Drain the pending mesh queue into runtime BLAS creation and unload BLASes
/// whose asset no longer has any referencing entity. Runs on the render world
/// (NonSend, main thread) between `ensure_renderer` and `render_frame`.
pub(crate) fn upload_mesh_assets(mut state: NonSendMut<SunrayRenderState>, mut assets: ResMut<ExtractedMeshAssets>) {
    // No renderer yet: keep the queue, it's drained once the window exists.
    let Some(renderer) = state.renderer.as_mut() else {
        return;
    };
    let ExtractedMeshAssets {
        pending,
        loaded,
        referenced,
        failed,
    } = &mut *assets;

    for mesh in pending.drain(..) {
        match renderer.load_mesh(mesh.id, &mesh.vertices, &mesh.indices, &mesh.material) {
            Ok(()) => {
                log::info!(
                    "sunray: built BLAS for mesh asset {:?} ({} vertices, {} triangles)",
                    mesh.id,
                    mesh.vertices.len(),
                    mesh.indices.len() / 3
                );
                loaded.insert(mesh.id);
            }
            Err(e) => {
                log::error!("sunray: BLAS build for mesh asset {:?} failed: {e}", mesh.id);
                failed.insert(mesh.id);
            }
        }
    }

    // Unload BLASes of assets with no referencing entity left (despawned).
    let orphans: Vec<UntypedAssetId> = loaded.difference(referenced).copied().collect();
    for id in orphans {
        match renderer.unload_mesh(&id) {
            Ok(()) => {
                loaded.remove(&id);
                log::info!("sunray: unloaded BLAS of unreferenced mesh asset {id:?}");
            }
            Err(e) => log::error!("sunray: unloading BLAS of mesh asset {id:?} failed: {e}"),
        }
    }
}

/// Convert a Bevy `Mesh` into the renderer's vertex/index arrays. Requires
/// triangle-list topology and `Float32x3` positions; normals / UVs / tangents
/// are filled with neutral defaults when absent, and a non-indexed mesh gets a
/// sequential index list.
fn convert_mesh(mesh: &Mesh) -> Result<(Vec<sr_gltf::Vertex>, Vec<u32>), String> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(format!(
            "unsupported topology {:?} (need TriangleList)",
            mesh.primitive_topology()
        ));
    }
    let positions = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(VertexAttributeValues::as_float3)
        .ok_or_else(|| "missing or non-Float32x3 POSITION attribute".to_string())?;
    let normals = mesh
        .attribute(Mesh::ATTRIBUTE_NORMAL)
        .and_then(VertexAttributeValues::as_float3);
    let uvs = mesh.attribute(Mesh::ATTRIBUTE_UV_0).and_then(|values| match values {
        VertexAttributeValues::Float32x2(values) => Some(values.as_slice()),
        _ => None,
    });
    let tangents = mesh.attribute(Mesh::ATTRIBUTE_TANGENT).and_then(|values| match values {
        VertexAttributeValues::Float32x4(values) => Some(values.as_slice()),
        _ => None,
    });

    let vertices: Vec<sr_gltf::Vertex> = (0..positions.len())
        .map(|i| {
            // Runtime meshes have a single UV set: use it for every texture
            // coordinate channel (only read if the material ever gets textures).
            let uv = uvs.and_then(|uvs| uvs.get(i)).copied().unwrap_or([0.0, 0.0]);
            sr_gltf::Vertex {
                position: positions[i],
                normal: normals.and_then(|normals| normals.get(i)).copied().unwrap_or([0.0, 0.0, 1.0]),
                tangent: tangents.and_then(|tangents| tangents.get(i)).copied().unwrap_or([0.0; 4]),
                base_color_tex_coord: uv,
                metallic_roughness_tex_coord: uv,
                normal_tex_coord: uv,
                occlusion_tex: uv,
                emissive_tex: uv,
                ..Default::default()
            }
        })
        .collect();

    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(indices)) => indices.clone(),
        Some(Indices::U16(indices)) => indices.iter().map(|&i| i as u32).collect(),
        // Non-indexed: every 3 consecutive vertices form a triangle.
        None => (0..positions.len() as u32).collect(),
    };
    if !indices.len().is_multiple_of(3) {
        return Err(format!("index count {} is not a multiple of 3", indices.len()));
    }

    Ok((vertices, indices))
}
