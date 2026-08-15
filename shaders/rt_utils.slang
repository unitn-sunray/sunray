// Helpers shared across RT entry points: RNG, octahedral normal packing,
// pack/unpack of half/unorm/snorm formats, BRDF terms, ReSTIR merge functions,
// and the heap-indirected texture sampler.

module rt_utils;

import rt_types;

// --- Push constant access ---------------------------------------------------
// Each entry point declares `[vk::push_constant] ConstantBuffer<RaytracingPC> pc;`
// and passes it (or its fields) into these helpers explicitly. Slang doesn't
// share globals across modules, so we keep the helpers pure.

// --- RNG (PCG, one seed per thread) -----------------------------------------
public struct Rng {
    public uint seed;
};

public uint pcg_hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

public Rng init_rng(uint2 pixel, uint frame, uint2 launch_size) {
    Rng rng;
    uint pixel_idx = pixel.y * launch_size.x + pixel.x;
    rng.seed = pcg_hash(pixel_idx ^ pcg_hash(frame));
    return rng;
}

public float rnd(inout Rng rng) {
    rng.seed = rng.seed * 747796405u + 2891336453u;
    uint word = ((rng.seed >> ((rng.seed >> 28u) + 4u)) ^ rng.seed) * 277803737u;
    uint result = (word >> 22u) ^ word;
    return float(result) / 4294967295.0;
}

// --- GLSL-style pack/unpack helpers ----------------------------------------
// The Slang version shipped with this Vulkan SDK reports these GLSL builtins
// (`packUnorm4x8`, `unpackUnorm4x8`, `packHalf2x16`, `unpackHalf2x16`,
// `unpackSnorm2x16`) as "undefined identifier", so we inline them on top of
// the HLSL bit/half intrinsics. Byte layouts match the GLSL `OpExtInst`
// versions exactly, so they read/write the same payload bits as the original
// GLSL pipeline.
public uint pack_snorm_2x16(float2 v) {
    int2 i = int2(round(clamp(v, -1.0, 1.0) * 32767.0));
    return (uint(i.x) & 0xFFFFu) | ((uint(i.y) & 0xFFFFu) << 16);
}
public float2 unpack_snorm_2x16(uint p) {
    int x = int(p << 16) >> 16;
    int y = int(p) >> 16;
    return clamp(float2(x, y) / 32767.0, -1.0, 1.0);
}
public uint pack_unorm_4x8(float4 v) {
    uint4 c = uint4(round(clamp(v, 0.0, 1.0) * 255.0));
    return c.x | (c.y << 8u) | (c.z << 16u) | (c.w << 24u);
}
public float4 unpack_unorm_4x8(uint p) {
    return float4(
        float((p >>  0u) & 0xFFu) / 255.0,
        float((p >>  8u) & 0xFFu) / 255.0,
        float((p >> 16u) & 0xFFu) / 255.0,
        float((p >> 24u) & 0xFFu) / 255.0
    );
}
public uint pack_half_2x16(float2 v) {
    return f32tof16(v.x) | (f32tof16(v.y) << 16u);
}
public float2 unpack_half_2x16(uint p) {
    return float2(f16tof32(p & 0xFFFFu), f16tof32(p >> 16u));
}

// --- Octahedral normal packing ----------------------------------------------
float2 sign_not_zero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

public uint pack_normal(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 p = (n.z >= 0.0) ? n.xy : (1.0 - abs(n.yx)) * sign_not_zero(n.xy);
    return pack_snorm_2x16(p);
}

public float3 unpack_normal(uint p) {
    float2 v = unpack_snorm_2x16(p);
    float3 n = float3(v.x, v.y, 1.0 - abs(v.x) - abs(v.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// --- Heap texture sampler ----------------------------------------------------
// Materials store *resolved* descriptor heap slots (`material_*_image` /
// `material_*_sampler` in `MeshInfo`), filled in on the CPU at scene-load
// time — there is no indirection buffer. `NULL_TEXTURE` in the image slot
// returns `fallback` unchanged.
public float4 sample_texture(
    uint image_slot,
    uint sampler_slot,
    float2 uv,
    float4 fallback
) {
    if (image_slot == NULL_TEXTURE) {
        return fallback;
    }
    Texture2D<float4> tex = DescriptorHandle<Texture2D<float4>>(uint2(image_slot, 0));
    SamplerState smp = DescriptorHandle<SamplerState>(uint2(sampler_slot, 0));
    return tex.SampleLevel(smp, uv, 0.0);
}

// --- Vertex attribute interpolation -----------------------------------------
public VertexAttributes interpolate_vertex_attributes(VertexAttributes v[3], float3 b) {
    VertexAttributes r;
    r.position                    = v[0].position * b.x                    + v[1].position * b.y                    + v[2].position * b.z;
    r.normal                      = v[0].normal * b.x                      + v[1].normal * b.y                      + v[2].normal * b.z;
    r.tangent                     = v[0].tangent * b.x                     + v[1].tangent * b.y                     + v[2].tangent * b.z;
    r.base_color_tex_coord        = v[0].base_color_tex_coord * b.x        + v[1].base_color_tex_coord * b.y        + v[2].base_color_tex_coord * b.z;
    r.metallic_roughness_tex_coord= v[0].metallic_roughness_tex_coord * b.x+ v[1].metallic_roughness_tex_coord * b.y+ v[2].metallic_roughness_tex_coord * b.z;
    r.normal_tex_coord            = v[0].normal_tex_coord * b.x            + v[1].normal_tex_coord * b.y            + v[2].normal_tex_coord * b.z;
    r.occlusion_tex_coord         = v[0].occlusion_tex_coord * b.x         + v[1].occlusion_tex_coord * b.y         + v[2].occlusion_tex_coord * b.z;
    r.emissive_tex_coord          = v[0].emissive_tex_coord * b.x          + v[1].emissive_tex_coord * b.y          + v[2].emissive_tex_coord * b.z;
    return r;
}

// --- BRDF helpers ----------------------------------------------------------
public void build_onb(float3 n, out float3 t, out float3 b) {
    float sign_n = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign_n + n.z);
    float bb = n.x * n.y * a;
    t = float3(1.0 + sign_n * n.x * n.x * a, sign_n * bb, -sign_n * n.x);
    b = float3(bb, sign_n + n.y * n.y * a, -n.y);
}

public float smith_v_ggx(float NdotV, float NdotL, float alpha) {
    float a2 = alpha * alpha;
    float ggxV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float ggxL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(ggxV + ggxL, 0.0001);
}

public float smith_g1_ggx(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    float denom = NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX);
    return 2.0 * NdotX / max(denom, 0.0001);
}

public float3 get_random_bounce(float3 normal, float r1, float r2) {
    float phi = 2.0 * 3.14159 * r1;
    float r = sqrt(r2);
    float3 u, v;
    build_onb(normal, u, v);
    return normalize(u * cos(phi) * r + v * sin(phi) * r + normal * sqrt(1.0 - r2));
}

public float3 sample_ggx_vndf(float3 normal, float3 V_world, float roughness, float r1, float r2) {
    float3 T, B;
    build_onb(normal, T, B);

    float3 Vl = float3(dot(V_world, T), dot(V_world, B), dot(V_world, normal));
    float a = max(roughness * roughness, 0.001);
    float3 Vh = normalize(float3(a * Vl.x, a * Vl.y, Vl.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);

    float rr = sqrt(r1);
    float phi = 2.0 * 3.14159265 * r2;
    float t1 = rr * cos(phi);
    float t2 = rr * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    float3 Hl = normalize(float3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
    return T * Hl.x + B * Hl.y + normal * Hl.z;
}

public float3 eval_unshadowed_light(
    float3 hit_pos, float3 hit_normal, float3 V_view, float3 hit_albedo, float roughness, float metallic,
    EmissiveTriangle light, float3 light_pos, float3 light_normal
) {
    float3 L = light_pos - hit_pos;
    float dist = max(length(L), 0.0001);
    L /= dist;

    float NdotL = max(dot(hit_normal, L), 0.0);
    float cos_light = max(dot(light_normal, -L), 0.0);
    if (NdotL <= 0.0 || cos_light <= 0.0) return float3(0.0);

    float3 H = normalize(V_view + L);
    float NdotH = max(dot(hit_normal, H), 0.0);
    float VdotH = max(dot(V_view, H), 0.0);
    float NdotV = max(dot(hit_normal, V_view), 0.001);

    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    float D = a2 / (3.14159 * denom * denom);

    float3 F0 = lerp(float3(0.04), hit_albedo, metallic);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    float V_term = smith_v_ggx(NdotV, NdotL, a);
    float3 specular_brdf = D * V_term * F;
    float3 diffuse_brdf = hit_albedo * (1.0 - metallic) * (float3(1.0) - F) / 3.14159;
    float geometry = (NdotL * cos_light) / max(dist * dist, 0.0001);

    return light.emission.rgb * (diffuse_brdf + specular_brdf) * geometry;
}

// --- ReSTIR helpers --------------------------------------------------------
public uint get_pixel_index(int2 coord, uint2 launch_size) {
    return uint(coord.y) * launch_size.x + uint(coord.x);
}

public uint current_reservoir_idx(uint frame_count) { return frame_count & 1u; }
public uint history_reservoir_idx(uint frame_count) { return (frame_count & 1u) ^ 1u; }

public void merge_reservoirs(inout Reservoir r, Reservoir new_r, float p_hat_new, float random_val) {
    r.M += new_r.M;
    float weight = p_hat_new * new_r.W * new_r.M;
    r.w_sum += weight;
    if (random_val < (weight / max(r.w_sum, 0.0001))) {
        r.light_idx = new_r.light_idx;
        r.light_pos = new_r.light_pos;
        r.light_normal = new_r.light_normal;
    }
}

public float gi_target_pdf(float3 shade_pos, float3 shade_normal, float3 albedo, float metallic, float3 sample_pos, float3 sample_radiance) {
    float3 w = sample_pos - shade_pos;
    float d = max(length(w), 0.0001);
    w /= d;
    float NdotL = max(dot(shade_normal, w), 0.0);
    float3 f_diffuse = albedo * (1.0 - metallic) / 3.14159;
    float3 contrib = sample_radiance * f_diffuse * NdotL;
    return max(contrib.r, max(contrib.g, contrib.b));
}

public void merge_reservoirs_gi(inout ReservoirGI r, ReservoirGI new_r, float p_hat_new, float jacobian, float random_val) {
    r.M += new_r.M;
    float weight = p_hat_new * new_r.W * new_r.M * jacobian;
    r.w_sum += weight;
    if (random_val < (weight / max(r.w_sum, 0.0001))) {
        r.sample_pos = new_r.sample_pos;
        r.sample_normal_packed = new_r.sample_normal_packed;
        r.sample_radiance = new_r.sample_radiance;
    }
}

// --- Transform helpers -----------------------------------------------------
// `xform.rows` stores a 3x4 row-major transform (same layout as VkTransformMatrixKHR).
public float3 transform_point(EntityTransform xform, float3 p) {
    float4 hp = float4(p, 1.0);
    return float3(dot(xform.rows[0], hp), dot(xform.rows[1], hp), dot(xform.rows[2], hp));
}
