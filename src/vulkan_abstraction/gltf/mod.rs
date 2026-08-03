use crate::{
    error::{SrError, SrResult},
    vulkan_abstraction,
};
use std::path::Path;
use std::{collections::HashMap, sync::Arc};

use nalgebra as na;

pub mod emissive_triangle;
pub mod image;
pub mod material;
pub mod mesh;
pub mod node;
pub mod primitive;
pub mod texture;
pub mod vertex;

pub use emissive_triangle::*;
pub use image::*;
pub use material::*;
pub use mesh::*;
pub use node::*;
pub use primitive::*;
pub use texture::*;
pub use vertex::*;

macro_rules! get_texture_indices {
    ($material:ident, $texture_name:ident) => {
        match $material.$texture_name() {
            Some(texture_info) => (Some(texture_info.texture().index()), texture_info.tex_coord()),
            None => (None, 0),
        }
    };
}

macro_rules! insert_tex_coords {
    ($reader:ident, $vertices:ident, $tex_coord_index:expr, $texture_name_coord:ident) => {
        $reader
            .read_tex_coords($tex_coord_index)
            .unwrap()
            .into_f32()
            .enumerate()
            .for_each(|(j, coord)| $vertices[j].$texture_name_coord = coord);
    };
}

pub type PrimitiveDataMap = HashMap<vulkan_abstraction::gltf::PrimitiveUniqueKey, vulkan_abstraction::gltf::PrimitiveData>;

pub struct Gltf {
    core: Arc<vulkan_abstraction::Core>,
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
}

impl Gltf {
    pub fn new(core: Arc<vulkan_abstraction::Core>, path: impl AsRef<Path>) -> SrResult<Self> {
        let (document, buffers, images) = gltf::import(path)?;

        Ok(Self {
            core,
            document,
            buffers,
            images,
        })
    }

    pub fn create_default_scene(&self) -> SrResult<(crate::Scene, crate::SceneData)> {
        // find the defualt scene index
        let default_scene_index = match self.document.default_scene() {
            Some(s) => s.index(),
            None => 0,
        };

        self.create_scene(default_scene_index)
    }

    pub fn create_scene(&self, scene_index: usize) -> SrResult<(crate::Scene, crate::SceneData)> {
        let gltf_scene = match self.document.scenes().enumerate().find(|(index, _)| *index == scene_index) {
            Some((_, scene)) => scene,
            None => {
                return Err(SrError::new_custom(format!(
                    "gltf: No scene with index: {} found",
                    scene_index
                )));
            }
        };

        let samplers = self
            .document
            .samplers()
            .map(|sampler| vulkan_abstraction::gltf::Sampler {
                mag_filter: sampler.mag_filter(),
                min_filter: sampler.min_filter(),
                wrap_s_u: sampler.wrap_s(),
                wrap_t_v: sampler.wrap_t(),
            })
            .collect::<Vec<_>>();

        let textures = self
            .document
            .textures()
            .map(|texture| Texture {
                sampler: texture.sampler().index(),
                source: texture.source().index(),
            })
            .collect::<Vec<_>>();

        let images = self
            .images
            .iter()
            .map(|image| Image {
                format: image.format,
                height: image.height as usize,
                width: image.width as usize,
                raw_data: image.pixels.clone(), // TODO: consume gltf
            })
            .collect::<Vec<_>>();

        let mut nodes: Vec<Node> = vec![];
        let mut primitive_data_map: PrimitiveDataMap = PrimitiveDataMap::new();
        for gltf_node in gltf_scene.nodes() {
            // the root nodes do not have a parent transform to apply
            let transform = na::Matrix4::identity();
            let node = self.explore(&gltf_node, transform, &mut primitive_data_map)?;
            nodes.push(node);
        }

        let scene = crate::Scene::new(nodes)?;
        let scene_data = crate::SceneData {
            textures,
            images,
            samplers,
            primitive_data_map,
        };

        Ok((scene, scene_data))
    }

    fn explore(
        &self,
        gltf_node: &gltf::Node,
        parent_transform: na::Matrix4<f32>,
        primitive_data_map: &mut PrimitiveDataMap,
    ) -> SrResult<vulkan_abstraction::gltf::Node> {
        let (transform, mesh) = self.process_node(gltf_node, parent_transform, primitive_data_map)?;

        let children = if gltf_node.children().len() == 0 {
            None
        } else {
            let mut children = vec![];
            for gltf_child in gltf_node.children() {
                let child = self.explore(&gltf_child, transform, primitive_data_map)?;
                children.push(child);
            }

            Some(children)
        };

        vulkan_abstraction::gltf::Node::new(transform, mesh, children)
    }

    fn process_node(
        &self,
        gltf_node: &gltf::Node,
        parent_transform: na::Matrix4<f32>,
        primitive_data_map: &mut PrimitiveDataMap,
    ) -> SrResult<(na::Matrix4<f32>, Option<vulkan_abstraction::gltf::Mesh>)> {
        // the trasnform can also be given decomposed in: translation, rotation and scale
        // but the gltf crate takes care of this:
        // "If the transform is Decomposed, then the matrix is generated with the equation matrix = translation * rotation * scale."
        let transform = parent_transform * na::Matrix4::from(gltf_node.transform().matrix());

        // I dont'use map because `?`` does not work inside a closure
        let mesh = match gltf_node.mesh() {
            Some(gltf_mesh) => Some(self.process_mesh(gltf_mesh, primitive_data_map)?),
            None => None,
        };

        if let Some(_camera) = gltf_node.camera() {
            todo!()
        }

        if let Some(_light) = gltf_node.light() {
            todo!()
        }

        Ok((transform, mesh))
    }

    fn process_mesh(
        &self,
        gltf_mesh: gltf::Mesh,
        primitive_data_map: &mut PrimitiveDataMap,
    ) -> SrResult<vulkan_abstraction::gltf::Mesh> {
        let mut primitives = vec![];

        for (i, primitive) in gltf_mesh.primitives().filter(|p| Self::is_primitive_supported(p)).enumerate() {
            let vertex_position_accessor_index = primitive
                .attributes()
                .find(|(semantic, _)| *semantic == gltf::Semantic::Positions)
                .unwrap()
                .1
                .index();

            let indices_accessor_index = match primitive.indices() {
                Some(accessor) => accessor.index(),
                None => i, // this is a cheap fix in the case that the primitive is a non-indexed geometry
            };

            let primitive_unique_key = (vertex_position_accessor_index, indices_accessor_index);

            let (material, tex_coords, local_emissive_triangles) = {
                let material = primitive.material();
                let material_pbr = primitive.material().pbr_metallic_roughness();

                let base_color_factor = material_pbr.base_color_factor();
                let metallic_factor = material_pbr.metallic_factor();
                let roughness_factor = material_pbr.roughness_factor();
                let emissive_factor = material.emissive_factor();
                // glTF spec: when KHR_materials_emissive_strength is absent the
                // multiplier is 1.0 (emission == emissiveFactor), not 0. Defaulting
                // to 0 zeroed every emitter lacking the extension — e.g. all of
                // Bistro's textured lights rendered as black surfaces.
                let emissive_strength = material.emissive_strength().unwrap_or(1.0);
                let alpha_mode = material.alpha_mode();
                let alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);
                let double_sided = material.double_sided();
                let transmission_factor = material.transmission().map_or(0.0, |t| t.transmission_factor());

                let ior = material.ior().unwrap_or(1.5);

                // The code is repeated because the type of the textures are not the same
                // TODO: crate a macro
                let (base_color_texture_index, base_color_tex_coord_index) =
                    get_texture_indices!(material_pbr, base_color_texture);
                let (metallic_roughness_texture_index, metallic_roughness_tex_coord_index) =
                    get_texture_indices!(material_pbr, metallic_roughness_texture);
                let (normal_texture_index, normal_tex_coord_index) = get_texture_indices!(material, normal_texture);
                let (occlusion_texture_index, occlusion_tex_coord_index) = get_texture_indices!(material, occlusion_texture);
                let (emissive_texture_index, emissive_tex_coord_index) = get_texture_indices!(material, emissive_texture);

                let pbr_metallic_roughness_properties = vulkan_abstraction::gltf::PbrMetallicRoughnessProperties {
                    base_color_factor,
                    metallic_factor,
                    roughness_factor,
                    base_color_texture_index,
                    metallic_roughness_texture_index,
                };

                let material = vulkan_abstraction::gltf::Material {
                    pbr_metallic_roughness_properties,
                    normal_texture_index,
                    occlusion_texture_index,
                    emissive_factor,
                    emissive_strength,
                    emissive_texture_index,
                    alpha_mode,
                    alpha_cutoff,
                    double_sided,
                    transmission_factor,
                    ior,
                };

                let tex_coords = (
                    base_color_tex_coord_index,
                    metallic_roughness_tex_coord_index,
                    normal_tex_coord_index,
                    occlusion_tex_coord_index,
                    emissive_tex_coord_index,
                );

                let mut local_emissive_triangles = Vec::new();

                // Emission is `emissive_factor * emissive_strength`, so a material
                // emits only when BOTH are non-zero. This must be `&&`, not `||`:
                // `emissive_strength` now defaults to 1.0 (glTF spec) when the
                // strength extension is absent, so an `||` here flags every
                // material as emissive and floods the emissive-triangle arena with
                // the entire scene.
                let is_emissive = material.emissive_factor != [0.0, 0.0, 0.0] && material.emissive_strength > 0.0;

                if is_emissive {
                    let reader = primitive.reader(|buffer| Some(&self.buffers[buffer.index()]));

                    let positions: Vec<_> = reader.read_positions().unwrap().collect();

                    let indices: Vec<u32> = if let Some(iter) = reader.read_indices() {
                        iter.into_u32().collect()
                    } else {
                        (0..positions.len() as u32).collect()
                    };

                    for chunk in indices.chunks_exact(3) {
                        let p0 = positions[chunk[0] as usize];
                        let p1 = positions[chunk[1] as usize];
                        let p2 = positions[chunk[2] as usize];

                        local_emissive_triangles.push([
                            na::Vector4::new(p0[0], p0[1], p0[2], 1.0),
                            na::Vector4::new(p1[0], p1[1], p1[2], 1.0),
                            na::Vector4::new(p2[0], p2[1], p2[2], 1.0),
                        ]);
                    }
                }

                (material, tex_coords, local_emissive_triangles)
            };

            if let std::collections::hash_map::Entry::Vacant(e) = primitive_data_map.entry(primitive_unique_key) {
                let reader = primitive.reader(|buffer| Some(&self.buffers[buffer.index()]));

                let mut vertices: Vec<vulkan_abstraction::gltf::Vertex> = vec![];

                let positions: Vec<_> = reader.read_positions().unwrap().collect();
                let normals: Vec<_> = reader.read_normals().unwrap().collect();

                let tangents: Vec<[f32; 4]> = reader
                    .read_tangents()
                    .map_or_else(|| vec![[0.0, 0.0, 0.0, 0.0]; positions.len()], |iter| iter.collect());

                for i in 0..positions.len() {
                    vertices.push(vulkan_abstraction::gltf::Vertex {
                        position: positions[i],
                        normal: normals[i],
                        tangent: tangents[i],
                        ..Default::default()
                    });
                }

                let index_buffer = {
                    let indices = if primitive.indices().is_some() {
                        // get vertices index

                        reader.read_indices().unwrap().into_u32().collect::<Vec<_>>()
                    } else {
                        // if the primitive is a non-indexed geometry we create the indices

                        (0..vertices.len() as u32 / 3).collect::<Vec<_>>()
                    };

                    vulkan_abstraction::IndexBuffer::new_for_blas_from_data(Arc::clone(&self.core), &indices)?
                };

                // This could also be done with zip, but the code would be equally long and with a lot of nested tuples
                // I thought of moving the zip operation to a separate function but the type of reader doesn't allow you to pass it around
                insert_tex_coords!(reader, vertices, tex_coords.0, base_color_tex_coord);
                insert_tex_coords!(reader, vertices, tex_coords.1, metallic_roughness_tex_coord);
                insert_tex_coords!(reader, vertices, tex_coords.2, normal_tex_coord);
                insert_tex_coords!(reader, vertices, tex_coords.3, occlusion_tex);
                insert_tex_coords!(reader, vertices, tex_coords.4, emissive_tex);

                let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas_from_data(Arc::clone(&self.core), &vertices)?;

                let primitive_data = vulkan_abstraction::gltf::PrimitiveData {
                    vertex_buffer,
                    index_buffer,
                };

                e.insert(primitive_data);
            }
            primitives.push(vulkan_abstraction::gltf::Primitive {
                unique_key: primitive_unique_key,
                material,
                local_emissive_triangles,
            });
        }

        vulkan_abstraction::gltf::Mesh::new(primitives)
    }

    fn is_primitive_supported(primitive: &gltf::Primitive) -> bool {
        match primitive.mode() {
            gltf::mesh::Mode::Triangles => true,
            m => {
                log::error!("Found unsupported primitive mode: {:?}", m);

                false
            }
        }
    }
}
