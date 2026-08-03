#[derive(Clone)]
pub struct PbrMetallicRoughnessProperties {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub base_color_texture_index: Option<usize>,
    pub metallic_roughness_texture_index: Option<usize>,
}

#[derive(Clone)]
pub struct Material {
    pub pbr_metallic_roughness_properties: PbrMetallicRoughnessProperties,
    pub normal_texture_index: Option<usize>,
    pub occlusion_texture_index: Option<usize>,
    pub emissive_factor: [f32; 3],
    pub emissive_strength: f32,
    pub emissive_texture_index: Option<usize>,
    pub alpha_mode: gltf::material::AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub transmission_factor: f32,
    pub ior: f32,
}

impl Material {
    /// Whether this material is rendered with a hard alpha-cutout test in the
    /// any-hit shader (its BLAS geometry must then be built non-opaque so any-hit
    /// runs). Covers MASK and BLEND: true translucency isn't implemented, and
    /// content such as Bistro authors its cutout foliage as BLEND, so BLEND is
    /// approximated as a cutout at `alpha_cutoff` rather than left opaque. Only
    /// OPAQUE stays on the traversal fast path.
    pub fn is_alpha_cutout(&self) -> bool {
        matches!(
            self.alpha_mode,
            gltf::material::AlphaMode::Mask | gltf::material::AlphaMode::Blend
        )
    }
}
