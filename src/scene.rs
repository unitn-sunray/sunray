use std::{collections::HashMap, sync::Arc};

use crate::vulkan_abstraction::image::sampler::SamplerDesc;
use crate::{error::SrResult, vulkan_abstraction};

use crate::utils::na_mat4_to_vk_transform;
use ash::vk;

pub struct SceneData {
    pub textures: Vec<vulkan_abstraction::gltf::Texture>,
    pub samplers: Vec<vulkan_abstraction::gltf::Sampler>,
    pub images: Vec<vulkan_abstraction::gltf::Image>,
    pub primitive_data_map: vulkan_abstraction::gltf::PrimitiveDataMap,
}
//TODO I need to actually look into this and decide how to handle it once and for all
/// One unique BLAS of a loaded scene, together with the data the
/// `ResourceManager` uploads alongside it: the primitive's material and its
/// local-space emissive triangles.
pub struct LoadedBlas {
    pub blas: vulkan_abstraction::Blas,
    pub material: vulkan_abstraction::gltf::Material,
    pub emissive_triangles: Vec<vulkan_abstraction::gltf::EmissiveTriangle>,
}

/// Everything `Scene::load_into_gpu` produced, in a renderer-agnostic form:
/// unique BLASes (with material + emissive triangles), the per-instance
/// `(blas index, transform)` list, and the texture / sampler / image data the
/// materials reference. Samplers are plain `SamplerDesc`s so the
/// `ResourceManager` can dedup them into its finite sampler set.
pub struct LoadedScene {
    pub blases: Vec<LoadedBlas>,
    /// One entry per scene instance: index into `blases` + world transform.
    pub instances: Vec<(usize, vk::TransformMatrixKHR)>,
    pub textures: Vec<vulkan_abstraction::gltf::Texture>,
    pub sampler_descs: Vec<SamplerDesc>,
    pub images: Vec<vulkan_abstraction::Image>,
}

pub struct Scene {
    nodes: Vec<vulkan_abstraction::gltf::Node>,
}

impl Scene {
    pub fn new(nodes: Vec<vulkan_abstraction::gltf::Node>) -> SrResult<Self> {
        Ok(Self { nodes })
    }

    pub fn nodes(&self) -> &[vulkan_abstraction::gltf::Node] {
        &self.nodes
    }

    pub fn load_into_gpu(&self, core: &Arc<vulkan_abstraction::Core>, mut scene_data: crate::SceneData) -> SrResult<LoadedScene> {
        let mut blases = vec![];
        let mut instances = vec![];

        let mut primitives_blas_index: HashMap<vulkan_abstraction::gltf::PrimitiveUniqueKey, usize> = HashMap::new();
        for node in self.nodes() {
            self.explore_node(
                node,
                core,
                &mut blases,
                &mut instances,
                &mut primitives_blas_index,
                &mut scene_data,
            )?;
        }

        let sampler_descs = scene_data
            .samplers
            .iter()
            .map(|sampler| {
                let default = gltf::texture::MinFilter::Linear;

                SamplerDesc {
                    min_filter: vk::Filter::from_gltf(sampler.min_filter.unwrap_or(default)),
                    mag_filter: vk::Filter::from_gltf(sampler.mag_filter.unwrap_or(gltf::texture::MagFilter::Linear)),
                    address_mode_u: vk::SamplerAddressMode::from_gltf(sampler.wrap_s_u),
                    address_mode_v: vk::SamplerAddressMode::from_gltf(sampler.wrap_t_v),
                    address_mode_w: vk::SamplerAddressMode::REPEAT,
                    mipmap_mode: vk::SamplerMipmapMode::from_gltf(sampler.min_filter.unwrap_or(default)),
                }
            })
            .collect();

        let images: Result<Vec<_>, _> = scene_data.images.into_iter().map(|image| to_vk_image(core, image)).collect();

        Ok(LoadedScene {
            blases,
            instances,
            textures: scene_data.textures,
            sampler_descs,
            images: images?,
        })
    }

    fn explore_node(
        &self,
        node: &vulkan_abstraction::gltf::Node,
        core: &Arc<vulkan_abstraction::Core>,
        blases: &mut Vec<LoadedBlas>,
        instances: &mut Vec<(usize, vk::TransformMatrixKHR)>,
        primitives_blas_index: &mut HashMap<vulkan_abstraction::gltf::PrimitiveUniqueKey, usize>,
        scene_data: &mut crate::SceneData,
    ) -> SrResult<()> {
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let primitive_unique_key = primitive.unique_key;

                let blas_index = match primitives_blas_index.get(&primitive_unique_key) {
                    Some(blas_index) => *blas_index,
                    None => {
                        let primitive_data = scene_data.primitive_data_map.remove(&primitive_unique_key).unwrap();

                        // Convert local-space emissive triangles for this primitive
                        let emissive_triangles: Vec<_> = if !primitive.local_emissive_triangles.is_empty() {
                            let material = &primitive.material;
                            let emission = [
                                material.emissive_factor[0] * material.emissive_strength,
                                material.emissive_factor[1] * material.emissive_strength,
                                material.emissive_factor[2] * material.emissive_strength,
                                0.0,
                            ];
                            primitive
                                .local_emissive_triangles
                                .iter()
                                .map(|local_tri| vulkan_abstraction::gltf::EmissiveTriangle {
                                    v0: [local_tri[0].x, local_tri[0].y, local_tri[0].z, 0.0],
                                    v1: [local_tri[1].x, local_tri[1].y, local_tri[1].z, 0.0],
                                    v2: [local_tri[2].x, local_tri[2].y, local_tri[2].z, 0.0],
                                    emission,
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };

                        let blas = vulkan_abstraction::Blas::new(
                            core.clone(),
                            primitive_data.vertex_buffer,
                            primitive_data.index_buffer,
                            !primitive.material.is_alpha_cutout(),
                            vulkan_abstraction::BuildType::Static,
                        )?;

                        blases.push(LoadedBlas {
                            blas,
                            material: primitive.material.clone(),
                            emissive_triangles,
                        });

                        let blas_index = blases.len() - 1;
                        primitives_blas_index.insert(primitive_unique_key, blas_index);

                        blas_index
                    }
                };

                instances.push((blas_index, na_mat4_to_vk_transform(*node.transform())));
            }
        }

        if let Some(children) = node.children() {
            for child in children {
                self.explore_node(child, core, blases, instances, primitives_blas_index, scene_data)?
            }
        }

        Ok(())
    }
}

fn to_vk_image(
    core: &Arc<vulkan_abstraction::Core>,
    image: vulkan_abstraction::gltf::Image,
) -> SrResult<vulkan_abstraction::Image> {
    let format = vk::Format::from_gltf(image.format);

    let image = vulkan_abstraction::Image::new_from_data(
        Arc::clone(core),
        image.raw_data,
        &vulkan_abstraction::image::ImageDesc {
            extent: vk::Extent3D {
                width: image.width as u32,
                height: image.height as u32,
                depth: 1,
            },
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            usage: vk::ImageUsageFlags::SAMPLED,
            name: "gltf image",
        },
    )?;

    Ok(image)
}

// Because of the orphan rule of rust
// it is not possible to implement the trait from
// for the types gltf::image::Format and vk::Format
// so I created a custom trait
pub trait FromGltf<T> {
    fn from_gltf(value: T) -> Self;
}

impl FromGltf<gltf::image::Format> for vk::Format {
    fn from_gltf(value: gltf::image::Format) -> Self {
        match value {
            gltf::image::Format::R8 => vk::Format::R8_UNORM,
            gltf::image::Format::R8G8 => vk::Format::R8G8_UNORM,
            gltf::image::Format::R8G8B8 => vk::Format::R8G8B8_UNORM,
            gltf::image::Format::R8G8B8A8 => vk::Format::R8G8B8A8_UNORM,
            gltf::image::Format::R16 => vk::Format::R16_SFLOAT,
            gltf::image::Format::R16G16 => vk::Format::R16G16_SFLOAT,
            gltf::image::Format::R16G16B16 => vk::Format::R16G16B16_SFLOAT,
            gltf::image::Format::R16G16B16A16 => vk::Format::R16G16B16A16_SFLOAT,
            gltf::image::Format::R32G32B32FLOAT => vk::Format::R32G32B32_SFLOAT,
            gltf::image::Format::R32G32B32A32FLOAT => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}

impl FromGltf<gltf::texture::MinFilter> for vk::SamplerMipmapMode {
    fn from_gltf(value: gltf::texture::MinFilter) -> Self {
        match value {
            gltf::texture::MinFilter::Nearest => vk::SamplerMipmapMode::LINEAR,
            gltf::texture::MinFilter::Linear => vk::SamplerMipmapMode::LINEAR,
            gltf::texture::MinFilter::NearestMipmapNearest => vk::SamplerMipmapMode::NEAREST,
            gltf::texture::MinFilter::LinearMipmapNearest => vk::SamplerMipmapMode::NEAREST,
            gltf::texture::MinFilter::NearestMipmapLinear => vk::SamplerMipmapMode::LINEAR,
            gltf::texture::MinFilter::LinearMipmapLinear => vk::SamplerMipmapMode::LINEAR,
        }
    }
}

impl FromGltf<gltf::texture::MinFilter> for vk::Filter {
    fn from_gltf(value: gltf::texture::MinFilter) -> Self {
        match value {
            gltf::texture::MinFilter::Nearest => vk::Filter::NEAREST,
            gltf::texture::MinFilter::Linear => vk::Filter::LINEAR,
            gltf::texture::MinFilter::NearestMipmapNearest => vk::Filter::NEAREST,
            gltf::texture::MinFilter::LinearMipmapNearest => vk::Filter::LINEAR,
            gltf::texture::MinFilter::NearestMipmapLinear => vk::Filter::NEAREST,
            gltf::texture::MinFilter::LinearMipmapLinear => vk::Filter::LINEAR,
        }
    }
}

impl FromGltf<gltf::texture::MagFilter> for vk::Filter {
    fn from_gltf(value: gltf::texture::MagFilter) -> Self {
        match value {
            gltf::texture::MagFilter::Nearest => vk::Filter::NEAREST,
            gltf::texture::MagFilter::Linear => vk::Filter::LINEAR,
        }
    }
}

impl FromGltf<gltf::texture::WrappingMode> for vk::SamplerAddressMode {
    fn from_gltf(value: gltf::texture::WrappingMode) -> Self {
        match value {
            gltf::texture::WrappingMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            gltf::texture::WrappingMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            gltf::texture::WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
        }
    }
}
