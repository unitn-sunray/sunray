//! Scene loading through **Bevy's** glTF asset pipeline (`bevy_gltf`).
//!
//! This is the second of the two scene-load paths:
//! - *direct*: [`super::SunrayScene`] hands a path to the renderer's own glTF
//!   loader (`Renderer::load_gltf`) — full material/texture support, but
//!   synchronous and outside Bevy's asset system;
//! - *Bevy assets* (this module): the `.glb`/`.gltf` is loaded by `bevy_gltf`
//!   via the `AssetServer` (async, hot-reloadable, cacheable). Once the
//!   [`Gltf`] asset arrives, [`spawn_gltf_scenes`] expands the node hierarchy
//!   into child entities carrying [`SunrayMeshInstance`] +
//!   [`SunrayMaterial`], which the existing runtime mesh-asset path (asset.rs)
//!   turns into BLASes. Textures are not carried over (the runtime mesh path
//!   is factors-only), so the direct path renders richer materials.
//!
//! Usage: add [`SunrayGltfPlugin`] (after [`super::SunrayRenderPlugin`]), then
//! spawn an entity with a `Transform` and a [`SunrayGltfScene`] pointing at a
//! `Handle<Gltf>`. Despawning that entity unloads the scene (children go with
//! it, and asset.rs unloads BLASes that lose their last referencing entity).

use bevy_app::{App, Plugin, Update};
use bevy_asset::{AssetApp, AssetServer, Assets, Handle, LoadState};
use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::prelude::*;
use bevy_gltf::{Gltf, GltfMaterial, GltfMesh, GltfNode, GltfPlugin};
use bevy_transform::components::Transform;

use super::asset::{SunrayMaterial, SunrayMeshInstance};

/// Spawn this (together with a `Transform`) to instantiate a glTF that was
/// loaded through Bevy's asset pipeline. Once the asset is ready, one child
/// entity per glTF node is spawned under this entity (preserving the node
/// hierarchy and transforms), with one [`SunrayMeshInstance`] grandchild per
/// mesh primitive. Despawn this entity to unload the scene.
#[derive(Component, Clone, Debug)]
pub struct SunrayGltfScene {
    pub gltf: Handle<Gltf>,
}

/// Marker inserted on a [`SunrayGltfScene`] entity once its node hierarchy has
/// been spawned.
#[derive(Component)]
pub struct SunrayGltfSpawned;

/// Marker inserted on a [`SunrayGltfScene`] entity whose asset failed to load
/// (the failure is also logged; the entity is left alone afterwards).
#[derive(Component)]
pub struct SunrayGltfFailed;

/// Registers `bevy_gltf`'s loader (plus the subasset types it emits that the
/// missing `RenderPlugin` would normally register) and the system that expands
/// [`SunrayGltfScene`] entities. Add **after** [`super::SunrayRenderPlugin`]
/// (it relies on `Assets<Mesh>` and the runtime mesh-asset systems).
pub struct SunrayGltfPlugin;

impl Plugin for SunrayGltfPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<GltfPlugin>() {
            app.add_plugins(GltfPlugin::default());
        }
        // The glTF loader emits labeled subassets of these types and expects
        // the render stack to have registered them; without `RenderPlugin` we
        // must do it ourselves or every load fails on the first subasset.
        if !app.world().contains_resource::<Assets<bevy_image::Image>>() {
            app.init_asset::<bevy_image::Image>();
        }
        if !app
            .world()
            .contains_resource::<Assets<bevy_world_serialization::WorldAsset>>()
        {
            app.init_asset::<bevy_world_serialization::WorldAsset>();
        }
        {
            use bevy_render::mesh::skinning::SkinnedMeshInverseBindposes;
            if !app.world().contains_resource::<Assets<SkinnedMeshInverseBindposes>>() {
                app.init_asset::<SkinnedMeshInverseBindposes>();
            }
        }
        app.add_systems(Update, spawn_gltf_scenes);
    }
}

/// Expand every pending [`SunrayGltfScene`] whose [`Gltf`] asset has finished
/// loading into a child-entity hierarchy of [`SunrayMeshInstance`]s.
pub(crate) fn spawn_gltf_scenes(
    mut commands: Commands,
    pending: Query<(Entity, &SunrayGltfScene), (Without<SunrayGltfSpawned>, Without<SunrayGltfFailed>)>,
    asset_server: Res<AssetServer>,
    gltfs: Res<Assets<Gltf>>,
    nodes: Res<Assets<GltfNode>>,
    gltf_meshes: Res<Assets<GltfMesh>>,
    materials: Res<Assets<GltfMaterial>>,
) {
    for (entity, scene) in &pending {
        if let Some(LoadState::Failed(why)) = asset_server.get_load_state(&scene.gltf) {
            log::error!("sunray: bevy glTF load failed: {why}");
            commands.entity(entity).insert(SunrayGltfFailed);
            continue;
        }
        // Not in `Assets` yet = still loading; retried next frame. Labeled
        // subassets (nodes/meshes/materials) land together with the `Gltf`.
        let Some(gltf) = gltfs.get(&scene.gltf) else {
            continue;
        };

        // glTF stores a flat node list; roots are the nodes no other node
        // claims as a child. (This spawns the union of all scenes in the file
        // rather than just the default scene — identical for the common
        // single-scene .glb.)
        let child_ids: std::collections::HashSet<_> = gltf
            .nodes
            .iter()
            .filter_map(|handle| nodes.get(handle))
            .flat_map(|node| node.children.iter().map(Handle::id))
            .collect();
        let mut mesh_count = 0;
        for handle in &gltf.nodes {
            if child_ids.contains(&handle.id()) {
                continue;
            }
            if let Some(node) = nodes.get(handle) {
                spawn_node(&mut commands, entity, node, &nodes, &gltf_meshes, &materials, &mut mesh_count);
            }
        }
        commands.entity(entity).insert(SunrayGltfSpawned);
        log::info!(
            "sunray: spawned bevy glTF scene ({} nodes, {mesh_count} mesh primitives)",
            gltf.nodes.len()
        );
    }
}

/// Spawn `node` as a child entity of `parent` (carrying the node's local
/// transform), one grandchild per mesh primitive, then recurse into children.
fn spawn_node(
    commands: &mut Commands,
    parent: Entity,
    node: &GltfNode,
    nodes: &Assets<GltfNode>,
    gltf_meshes: &Assets<GltfMesh>,
    materials: &Assets<GltfMaterial>,
    mesh_count: &mut usize,
) {
    let node_entity = commands.spawn((node.transform, ChildOf(parent))).id();
    if let Some(mesh_handle) = &node.mesh
        && let Some(gltf_mesh) = gltf_meshes.get(mesh_handle)
    {
        for primitive in &gltf_mesh.primitives {
            let material = primitive
                .material
                .as_ref()
                .and_then(|handle| materials.get(handle))
                .map(sunray_material)
                .unwrap_or_default();
            commands.spawn((
                Transform::IDENTITY,
                ChildOf(node_entity),
                SunrayMeshInstance {
                    mesh: primitive.mesh.clone(),
                },
                material,
            ));
            *mesh_count += 1;
        }
    }
    for child in &node.children {
        if let Some(child_node) = nodes.get(child) {
            spawn_node(commands, node_entity, child_node, nodes, gltf_meshes, materials, mesh_count);
        }
    }
}

/// Factor-only conversion of a `bevy_gltf` material to [`SunrayMaterial`]
/// (textures aren't supported on the runtime mesh path). `GltfMaterial`
/// already folds `KHR_materials_emissive_strength` into `emissive`.
fn sunray_material(material: &GltfMaterial) -> SunrayMaterial {
    let base = material.base_color.to_linear();
    SunrayMaterial {
        base_color: [base.red, base.green, base.blue, base.alpha],
        metallic: material.metallic,
        roughness: material.perceptual_roughness,
        emissive: [material.emissive.red, material.emissive.green, material.emissive.blue],
        emissive_strength: 1.0,
        ior: material.ior,
        transmission: material.specular_transmission,
    }
}
