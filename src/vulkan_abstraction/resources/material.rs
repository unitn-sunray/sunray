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
    /// Which vertex UV set each texture samples, one bit per slot (0 = `uv0`,
    /// 1 = `uv1`); see [`Material::UV_SET_*`]. Occupies what used to be pure
    /// padding, so encoding it costs nothing.
    uv_set_mask: u32,
    _pad_mid_1: f32,

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

    // Bit positions in `uv_set_mask`. Must match `UV_SET_*` in
    // `shaders/rt_types.slang`.
    const UV_SET_BASE_COLOR: u32 = 0;
    const UV_SET_METALLIC_ROUGHNESS: u32 = 1;
    const UV_SET_NORMAL: u32 = 2;
    const UV_SET_OCCLUSION: u32 = 3;
    const UV_SET_EMISSIVE: u32 = 4;

    /// Fold a per-texture `TEXCOORD_n` index (already clamped to 0/1 at load) into
    /// its bit of the mask.
    fn uv_bit(set: u32, bit: u32) -> u32 {
        (set & 1) << bit
    }

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
            uv_set_mask: Self::uv_bit(pbr.base_color_tex_coord_set, Self::UV_SET_BASE_COLOR)
                | Self::uv_bit(pbr.metallic_roughness_tex_coord_set, Self::UV_SET_METALLIC_ROUGHNESS)
                | Self::uv_bit(material.normal_tex_coord_set, Self::UV_SET_NORMAL)
                | Self::uv_bit(material.occlusion_tex_coord_set, Self::UV_SET_OCCLUSION)
                | Self::uv_bit(material.emissive_tex_coord_set, Self::UV_SET_EMISSIVE),
            _pad_mid_1: 0.0,
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
