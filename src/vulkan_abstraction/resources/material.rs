use crate::vulkan_abstraction;

/// GPU-ready material. Texture references are stored as *resolved descriptor
/// heap slots* (`(image sampled slot, sampler slot)` pairs), filled in at
/// scene-load time — the shaders dereference the heap directly, there is no
/// texture indirection buffer. A missing texture is `NULL_TEXTURE_INDEX` in
/// the image slot (the sampler slot is then ignored by the shader).
///
/// Layout mirrors the inlined `material_*` fields of
/// `shaders/rt_types.slang::MeshInfo` exactly (112 bytes). `MeshInfo` is read
/// through a `StructuredBuffer` (std430), so the explicit pads keep every
/// `float4` at a 16-aligned offset *within MeshInfo* (this struct starts at
/// offset 16 after the two buffer pointers) and pad the total `MeshInfo` out
/// to a 16-multiple so the array stride matches.
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct Material {
    base_color_value: [f32; 4],

    metallic_factor: f32,
    roughness_factor: f32,
    _pad_mid: [f32; 2],

    //rgb + strength
    emissive_factor: [f32; 4],

    pub alpha_mode: u32,
    pub alpha_cutoff: f32,

    pub transmission_factor: f32,
    pub ior: f32,

    base_color_image: u32,
    base_color_sampler: u32,
    metallic_roughness_image: u32,
    metallic_roughness_sampler: u32,
    normal_image: u32,
    normal_sampler: u32,
    occlusion_image: u32,
    occlusion_sampler: u32,
    emissive_image: u32,
    emissive_sampler: u32,

    _pad_end: [u32; 2],
}

impl Material {
    pub(crate) const NULL_TEXTURE_INDEX: u32 = u32::MAX;

    /// Build the GPU material from the glTF one. `resolve` maps a glTF texture
    /// index (`Option<usize>`) to its `(image heap slot, sampler heap slot)`
    /// pair, returning `NULL_TEXTURE_INDEX` slots for `None`.
    pub(crate) fn new(material: &vulkan_abstraction::gltf::Material, resolve: &impl Fn(Option<usize>) -> (u32, u32)) -> Self {
        let pbr = &material.pbr_metallic_roughness_properties;
        let (base_color_image, base_color_sampler) = resolve(pbr.base_color_texture_index);
        let (metallic_roughness_image, metallic_roughness_sampler) = resolve(pbr.metallic_roughness_texture_index);
        let (normal_image, normal_sampler) = resolve(material.normal_texture_index);
        let (occlusion_image, occlusion_sampler) = resolve(material.occlusion_texture_index);
        let (emissive_image, emissive_sampler) = resolve(material.emissive_texture_index);

        Self {
            base_color_value: pbr.base_color_factor,

            metallic_factor: pbr.metallic_factor,
            roughness_factor: pbr.roughness_factor,

            emissive_factor: [
                material.emissive_factor[0],
                material.emissive_factor[1],
                material.emissive_factor[2],
                material.emissive_strength,
            ],

            // Matches the any-hit shader's test (0 == OPAQUE early-out).
            alpha_mode: match material.alpha_mode {
                gltf::material::AlphaMode::Opaque => 0,
                gltf::material::AlphaMode::Mask => 1,
                gltf::material::AlphaMode::Blend => 2,
            },
            alpha_cutoff: material.alpha_cutoff,
            transmission_factor: material.transmission_factor,
            ior: material.ior,
            _pad_mid: [0.0; 2],
            _pad_end: [0; 2],

            base_color_image,
            base_color_sampler,
            metallic_roughness_image,
            metallic_roughness_sampler,
            normal_image,
            normal_sampler,
            occlusion_image,
            occlusion_sampler,
            emissive_image,
            emissive_sampler,
        }
    }
}
