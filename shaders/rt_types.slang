// Shared types for the Slang/heap-mode raytracing pipeline. Layouts mirror
// the Rust `#[repr(C)]` structs and the GLSL std430 buffers exactly so the
// same backing buffers can be read from either code path during the
// GLSL → Slang transition.

module rt_types;

// --- Storage-image formats --------------------------------------------------
// `[vk::image_format("...")]` is SILENTLY DROPPED when the resource is reached
// through `DescriptorHandle<>` — the emitted `OpTypeImage` keeps format
// `Unknown` and the module picks up `StorageImageRead/WriteWithoutFormat`, so
// every image access takes the driver's generic path instead of a typed one.
// The format has to ride `RWTexture2D`'s third generic parameter instead, which
// does survive the heap lowering.
//
// That parameter is Slang's own `ImageFormat` ordinal, NOT the SPIR-V
// `ImageFormat` enum — the two orderings diverge after index 2. Verified by
// probing slangc 1.4.357: 4=rg16f, 5=r11f_g11f_b10f, 7=r16f, 10=rgba8,
// 16=rgba8_snorm.
//
// Each alias must match the `VkFormat` its image view was created with (see
// `Renderer::build_frame_graph` / `temporal_image_desc` in src/lib.rs) —
// a mismatch is undefined behaviour, not a validation error.
public typealias RWImage_r11f_g11f_b10f = RWTexture2D<float4, 0, 5>;  // B10G11R11_UFLOAT_PACK32
public typealias RWImage_r16f           = RWTexture2D<float,  0, 7>;  // R16_SFLOAT
public typealias RWImage_rg16f          = RWTexture2D<float2, 0, 4>;  // R16G16_SFLOAT
public typealias RWImage_rgba8_snorm    = RWTexture2D<float4, 0, 16>; // R8G8B8A8_SNORM
public typealias RWImage_rgba8          = RWTexture2D<float4, 0, 10>; // R8G8B8A8_UNORM

// --- Per-ray payload --------------------------------------------------------
public struct RayPayload {
    public float3 emission;
    public float dist;
    public uint albedo_packed;
    public uint normal_packed;
    public uint material_info;
    public uint transmission_ior_packed;
}

// --- Vertex / mesh ----------------------------------------------------------
// Explicit padding fields mirror the `#[repr(C, packed)]` Rust `Vertex` in
// `src/vulkan_abstraction/gltf/vertex.rs`. They are what makes the layout agree
// with the CPU: without them `normal` would land at offset 12 instead of 16.
// Offsets are 0,12,16,28,32,48,56 with a 64-byte stride.
//
// `position` is here because the BLAS build reads it at offset 0 — no shader
// touches it (the hit point comes from `ray_origin + ray_dir * t`).
//
// Only two UV sets: which one each texture samples is a *material* property
// (`material_uv_set_mask`), not a vertex one. See the Rust `Vertex` for why.
public struct VertexAttributes {
    public float3 position;
    public float _pad0;
    public float3 normal;
    public float _pad1;
    public float4 tangent;
    public float2 uv0;
    public float2 uv1;
}

// Bit positions in `MeshInfo.material_uv_set_mask`: 0 selects `uv0`, 1 selects
// `uv1`. Must match `Material::UV_SET_*` in
// `src/vulkan_abstraction/resources/material.rs`.
public static const uint UV_SET_BASE_COLOR = 0;
public static const uint UV_SET_METALLIC_ROUGHNESS = 1;
public static const uint UV_SET_NORMAL = 2;
public static const uint UV_SET_OCCLUSION = 3;
public static const uint UV_SET_EMISSIVE = 4;

// One entry per BLAS, indexed by `InstanceID()`. Read through a heap
// `StructuredBuffer<MeshInfo>` → std430 layout; mirrors `EntityGpuData` in
// `src/vulkan_abstraction/resources/entity.rs`. 128 B stride.
//
// **Material fields are inlined** rather than nested as a `Material` struct.
// Nesting produces *correct-looking* SPIR-V that renders wrong: the
// `OpMemberDecorate` offsets are right (`material` at 16, every sub-member at its
// CPU offset) and the access chain looks well-formed
// (`OpAccessChain %MeshInfo* %uint_3` then `%uint_N`), but the parent member
// offset is not actually applied, so every material read lands 16 bytes low and
// picks up the geometry slots as float data. Re-verified 2026-08-15 on Slang
// 2026.13.1 — still broken.
//
// Do NOT re-test this by diffing `OpMemberDecorate` offsets; they look correct
// either way. The only test that catches it is rendering the scene: materials go
// greyscale and the emissive panel goes dark.
// TODO issue this to the slang repo with the above repro.
//
// Field order/sizes mirror `#[repr(C, packed)]` `Material` in
// `src/vulkan_abstraction/resources/material.rs` exactly (112 bytes after the
// 16-byte geometry header). std430 needs each `float4` on a 16-aligned offset
// (hence `_material_pad_mid_*`) and the struct size a 16-multiple (hence the
// trailing pads) so the array stride stays 128.
//
// `material_*_image` / `material_*_sampler` are *resolved descriptor heap slots*
// (filled in at scene-load time on the CPU) — there is no texture indirection
// buffer. `NULL_TEXTURE` in an image slot means "no texture".
//
// Geometry is reached through the heap rather than by buffer-device-address: the
// BDA form (`VertexAttributes*` / `uint*`) made Slang tag every vertex and index
// load `Aligned 4`, which stops the driver widening a `float3`/`float4` fetch into
// one 128-bit load — a ~2.4x load-count penalty in the two hottest shaders in the
// pipeline. Costs two buffer-section heap slots per BLAS, allocated once at scene
// load.
public struct MeshInfo {
    public uint vertices;                                       // 0   heap slot
    public uint indices;                                        // 4   heap slot
    public uint2 _pad;                                          // 8   float4 needs 16-alignment
    public float4 material_base_color_value;                    // 16
    public float material_metallic_factor;                      // 32
    public float material_roughness_factor;                     // 36
    public uint material_uv_set_mask;                           // 40
    public float _material_pad_mid_1;                           // 44
    public float4 material_emissive_factor;                     // 48
    public uint material_alpha_mode;                            // 64
    public float material_alpha_cutoff;                         // 68
    public float material_transmission_factor;                  // 72
    public float material_ior;                                  // 76
    public uint material_base_color_image;                      // 80
    public uint material_base_color_sampler;                    // 84
    public uint material_metallic_roughness_image;              // 88
    public uint material_metallic_roughness_sampler;            // 92
    public uint material_normal_image;                          // 96
    public uint material_normal_sampler;                        // 100
    public uint material_occlusion_image;                       // 104
    public uint material_occlusion_sampler;                     // 108
    public uint material_emissive_image;                        // 112
    public uint material_emissive_sampler;                      // 116
    public uint _material_pad_end_0;                            // 120
    public uint _material_pad_end_1;                            // 124

    public StructuredBuffer<VertexAttributes> get_vertices() {
        DescriptorHandle<StructuredBuffer<VertexAttributes>> h = { vertices };
        return h;
    }
    public StructuredBuffer<uint> get_indices() {
        DescriptorHandle<StructuredBuffer<uint>> h = { indices };
        return h;
    }
}

// --- Emissive lighting ------------------------------------------------------
public struct EmissiveTriangle {
    public float4 v0;
    public float4 v1;
    public float4 v2;
    public float4 emission;
}

public struct EmissiveIndirectionEntry {
    public uint blas_tri_index;
    public uint entity_id;
}

public struct EntityTransform {
    public float4 rows[3];
}

// --- Camera matrices --------------------------------------------------------
// Stored as 16 flat `float4` columns instead of 4 `float4x4` (or even 4
// `float4[4]`) members because Slang's lowering of struct-member access
// through `OpUntypedPointerKHR` (descriptor-heap path) emits an OpAccessChain
// with a mismatched storage class (Function result over StorageBuffer base)
// for any composite-typed member — matrices, arrays, sub-structs — producing
// invalid SPIR-V → DEVICE_LOST. Scalar-typed members (including `float4`) go
// through the typed-pointer path and are fine. The on-disk byte layout is
// identical (column-major, 16 floats per matrix = 64 bytes) so the Rust upload
// of `MatricesBufferContents` (4 × nalgebra::Matrix4 column-major) is unchanged.
public struct Matrices {
    public float4 vi0; public float4 vi1; public float4 vi2; public float4 vi3;
    public float4 pi0; public float4 pi1; public float4 pi2; public float4 pi3;
    public float4 vp0; public float4 vp1; public float4 vp2; public float4 vp3;
    public float4 pp0; public float4 pp1; public float4 pp2; public float4 pp3;
}

// --- ReSTIR reservoirs ------------------------------------------------------
public struct Reservoir {
    public float3 light_pos;
    public float w_sum;
    public float3 light_normal;
    public float M;
    public uint light_idx;
    public float W;
    public uint hit_normal_packed;
    public float depth;
}

public struct ReservoirGI {
    public float3 sample_pos;
    public float w_sum;
    public float3 sample_radiance;
    public float M;
    public uint sample_normal_packed;
    public float W;
    public uint hit_normal_packed;
    public float depth;
}

// --- Push constant ----------------------------------------------------------
// Every binding from the old `RaytracingDescriptorSetLayout` is a heap slot
// index here. `reservoirs[2]` / `reservoirs_gi[2]` keep the A/B ping-pong
// scheme, indexed by `frame_count & 1u`. Texture access goes through the heap
// slots stored per material in `MeshInfo` — there is no indirection buffer.
//
// Fields are bare `uint`, not `DescriptorHandle<T>`, because the latter lowers
// to a `uint2` whose high word is never read — 4 wasted bytes per resource.
// The type each slot refers to is declared exactly once, in its accessor below;
// `DescriptorHandle<T> h = { slot }` is the documented way to rebuild a handle
// from an index. A generic `heap<T>(slot)` helper is not possible: the
// `DescriptorHandle<T>` → `T` conversion only fires for concrete types
// (`error[E30019]` inside any generic body), which is why these are written out.
//
// Keep in lockstep with `RaytracingHeapPushConstant` in
// `src/vulkan_abstraction/pipelines/ray_tracing_pipeline.rs` — u64 + 18 × u32.
public struct
RaytracingPC {
    // Still a raw device address, not a heap slot — Slang #10671 is only
    // half-fixed. `DescriptorHandle<RaytracingAccelerationStructure>` now does
    // emit the `OpConvertUToAccelerationStructureKHR` the issue was filed for,
    // and the heap read itself is provably correct (verified 2026-08-15 on
    // driver 610.88 / RTX 3060 Ti: the 8-byte-strided heap index resolves to the exact
    // byte the AS descriptor was written to, and that descriptor holds precisely
    // the device address). Feeding that loaded value to `TraceRay` still hangs
    // the GPU, while the identical value taken from this push constant does not.
    // Re-test the `DescriptorHandle` form on a Slang/driver bump; until then the
    // convert stays fed from the push constant.
    public uint64_t tlas;
    public uint raw_color;
    public uint depth_img;
    public uint normal_img;
    public uint diffuse_img;
    public uint motion_vec_img;

    public uint matrices;
    public uint meshes_info;
    public uint emissive_triangles;
    public uint emissive_indirection;
    public uint entity_transforms;

    public uint blue_noise_tex;
    public uint blue_noise_sampler;

    public uint reservoirs[2];
    public uint reservoirs_gi[2];

    public uint frame_count;
    public uint use_srgb;

    public RaytracingAccelerationStructure get_tlas() {
        return spirv_asm {
            result:$$RaytracingAccelerationStructure = OpConvertUToAccelerationStructureKHR $tlas;
        };
    }

    public RWImage_r11f_g11f_b10f get_raw_color()  { DescriptorHandle<RWImage_r11f_g11f_b10f> h = { raw_color };      return h; }
    public RWImage_r16f           get_depth()      { DescriptorHandle<RWImage_r16f>           h = { depth_img };      return h; }
    public RWImage_rgba8_snorm    get_normal()     { DescriptorHandle<RWImage_rgba8_snorm>    h = { normal_img };     return h; }
    public RWImage_r11f_g11f_b10f get_diffuse()    { DescriptorHandle<RWImage_r11f_g11f_b10f> h = { diffuse_img };    return h; }
    public RWImage_rg16f          get_motion_vec() { DescriptorHandle<RWImage_rg16f>          h = { motion_vec_img }; return h; }

    public StructuredBuffer<Matrices>                get_matrices()             { DescriptorHandle<StructuredBuffer<Matrices>>                h = { matrices };             return h; }
    public StructuredBuffer<MeshInfo>                get_meshes_info()          { DescriptorHandle<StructuredBuffer<MeshInfo>>                h = { meshes_info };          return h; }
    public StructuredBuffer<EmissiveTriangle>        get_emissive_triangles()   { DescriptorHandle<StructuredBuffer<EmissiveTriangle>>        h = { emissive_triangles };   return h; }
    public StructuredBuffer<EmissiveIndirectionEntry> get_emissive_indirection() { DescriptorHandle<StructuredBuffer<EmissiveIndirectionEntry>> h = { emissive_indirection }; return h; }
    public StructuredBuffer<EntityTransform>         get_entity_transforms()    { DescriptorHandle<StructuredBuffer<EntityTransform>>         h = { entity_transforms };    return h; }

    public Texture2D<float4> get_blue_noise_tex()     { DescriptorHandle<Texture2D<float4>> h = { blue_noise_tex };     return h; }
    public SamplerState      get_blue_noise_sampler() { DescriptorHandle<SamplerState>      h = { blue_noise_sampler }; return h; }

    public RWStructuredBuffer<Reservoir>   get_reservoir(uint i)    { DescriptorHandle<RWStructuredBuffer<Reservoir>>   h = { reservoirs[i] };    return h; }
    public RWStructuredBuffer<ReservoirGI> get_reservoir_gi(uint i) { DescriptorHandle<RWStructuredBuffer<ReservoirGI>> h = { reservoirs_gi[i] }; return h; }
}

public static const uint NULL_TEXTURE = 0xFFFFFFFFu;
