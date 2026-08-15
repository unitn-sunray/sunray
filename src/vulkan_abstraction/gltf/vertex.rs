#[derive(Debug, Clone, Copy, Default)]
#[repr(C, packed)]
pub struct Vertex {
    /*
    NOTE: don't move Self::position or place any attributes before it: the BLAS assumes
    that the vertex_buffer has a vec3 position attribute as its first (not necessarily
    the only) attribute in memory. The stride is derived (`VertexBuffer::stride()` feeds
    `vertex_stride` in `blas.rs::triangle_desc`), so the size may change freely.

    repr(C) is used to avoid reordering of this data, since it will be sent to the gpu,
    and repr(packed) gives us full control of the alignment and we can force it
    (using _padding* attributes) to follow GLSL's rules (specifically std430).

    The main thing to look out for is that vectors cannot straddle over the 4-word
    (1 word = 1 float) boundary: for example a vec4 is always aligned to 4w, and a float
    is aligned to 1w, but a vec3 is either aligned to 4w or to the next word after a 4w
    boundary, and can only be packed together with a single float, whereas a vec2 is
    either aligned to 2w or to the next word after a 4w boundary.

    GLSL (float) arrays do not follow these rules, but it was chosen to use vec(n)
    instead because alignment is a big deal in SIMD-level parallelism (which the GPU
    does massively) and if the GLSL spec specifies a preferred alignment we shouldn't
    ignore it just for slight convenience.

    Mirrors `shaders/rt_types.slang::VertexAttributes`; 64 bytes.

    This used to carry five float2 UV sets, one per texture slot, each filled from
    that texture's glTF `tex_coord` index. Three of the five were never read by any
    shader — base colour, metallic-roughness and emissive were all sampled with the
    base-colour UV regardless — so they cost 24 B/vertex and silently mis-sampled any
    asset whose MR or emissive map used TEXCOORD_1. Now the vertex carries the two
    sets glTF assets actually use and the *material* says which one each texture
    samples (`Material::uv_set_mask`). That also makes the vertex data
    material-independent, which matters because the vertex buffer is cached per
    primitive and can be shared by primitives with different materials.
    */
    pub position: [f32; 3],
    pub _padding0: [f32; 1],
    pub normal: [f32; 3],
    pub _padding1: [f32; 1],
    pub tangent: [f32; 4],
    /// TEXCOORD_0.
    pub uv0: [f32; 2],
    /// TEXCOORD_1, zeroed when the primitive doesn't provide it.
    pub uv1: [f32; 2],
}

// The BLAS build reads `position` at offset 0 with this struct's size as the vertex
// stride, and the RT shaders read it as a std430 `StructuredBuffer<VertexAttributes>`.
// Neither is validated at runtime, so pin the layout.
const _: () = assert!(
    size_of::<Vertex>() == 64,
    "Vertex must stay in lockstep with VertexAttributes in shaders/rt_types.slang"
);
const _: () = assert!(
    std::mem::offset_of!(Vertex, position) == 0,
    "BLAS build requires position first"
);
const _: () = assert!(std::mem::offset_of!(Vertex, normal) == 16);
const _: () = assert!(std::mem::offset_of!(Vertex, tangent) == 32);
const _: () = assert!(std::mem::offset_of!(Vertex, uv0) == 48);
const _: () = assert!(std::mem::offset_of!(Vertex, uv1) == 56);
