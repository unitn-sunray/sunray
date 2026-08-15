// Closest-hit: surface shading. Reads vertex attributes via BDA, interpolates
// them with the hit barycentrics, applies normal mapping if present, samples
// PBR textures (base color / metallic-roughness / emissive), and packs the
// result into the ray payload. Mirrors shaders/closest_hit.glsl.

import rt_types;
import rt_utils;

[vk::push_constant] ConstantBuffer<RaytracingPC> pc;

[shader("closesthit")]
void closest_hit(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    StructuredBuffer<MeshInfo> meshes = pc.get_meshes_info();

    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                         attribs.barycentrics.x,
                         attribs.barycentrics.y);
    uint blas_instance_id = InstanceID();
    MeshInfo mesh_info = meshes[blas_instance_id];

    uint index_offset = PrimitiveIndex() * 3;
    uint indices[3];
    StructuredBuffer<uint> index_buf = mesh_info.get_indices();
    indices[0] = index_buf[index_offset + 0];
    indices[1] = index_buf[index_offset + 1];
    indices[2] = index_buf[index_offset + 2];

    StructuredBuffer<VertexAttributes> vertex_buf = mesh_info.get_vertices();
    VertexAttributes v0 = vertex_buf[indices[0]];
    VertexAttributes v1 = vertex_buf[indices[1]];
    VertexAttributes v2 = vertex_buf[indices[2]];

    float3 normal      = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
    float3 tangent_dir = v0.tangent.xyz * bary.x + v1.tangent.xyz * bary.y + v2.tangent.xyz * bary.z;
    // Handedness from first vertex only — interpolating .w produces seam corruption.
    float handedness = v0.tangent.w >= 0.0 ? 1.0 : -1.0;

    // Interpolate both UV sets; each texture then picks one via the material mask.
    float2 uv0 = v0.uv0 * bary.x + v1.uv0 * bary.y + v2.uv0 * bary.z;
    float2 uv1 = v0.uv1 * bary.x + v1.uv1 * bary.y + v2.uv1 * bary.z;
    uint uv_mask = mesh_info.material_uv_set_mask;

    float2 base_color_uv = select_uv(uv_mask, UV_SET_BASE_COLOR, uv0, uv1);
    float2 normal_uv     = select_uv(uv_mask, UV_SET_NORMAL, uv0, uv1);

    // Material fields are inlined into MeshInfo (Slang heap-mode nested-struct
    // access-chain bug — see rt_types.slang::MeshInfo) and texture references are
    // pre-resolved heap slots.
    float4 base_color = sample_texture(mesh_info.material_base_color_image, mesh_info.material_base_color_sampler, base_color_uv, mesh_info.material_base_color_value);
    float3 emissive_factor_rgb = mesh_info.material_emissive_factor.rgb;
    float emissive_strength = mesh_info.material_emissive_factor.w;
    float2 emissive_uv = select_uv(uv_mask, UV_SET_EMISSIVE, uv0, uv1);
    float4 emissive_sample = sample_texture(mesh_info.material_emissive_image, mesh_info.material_emissive_sampler, emissive_uv, float4(emissive_factor_rgb, 1.0));
    float3 final_emission = emissive_sample.rgb * emissive_strength;

    // Transform the geometric normal with the inverse-transpose (== Normal * WorldToObject for an affine matrix).
    float3x3 world_to_object_3x3 = (float3x3)WorldToObject3x4();
    float3 world_normal = normalize(mul(normal, world_to_object_3x3));
    float3 final_normal = world_normal;

    float3 out_albedo = base_color.rgb;

    // Normal mapping.
    if (length(tangent_dir) > 0.001) {
        float3x3 object_to_world_3x3 = (float3x3)ObjectToWorld3x4();
        float3 world_tangent = normalize(mul(object_to_world_3x3, tangent_dir));
        world_tangent = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
        float3 world_bitangent = cross(world_normal, world_tangent) * handedness;
        float3x3 TBN = float3x3(world_tangent, world_bitangent, world_normal);

        if (mesh_info.material_normal_image != NULL_TEXTURE) {
            float3 raw_normal_map = sample_texture(mesh_info.material_normal_image, mesh_info.material_normal_sampler, normal_uv, float4(0.5, 0.5, 1.0, 1.0)).rgb;
            float3 sampled_normal = raw_normal_map * 2.0 - 1.0;
            sampled_normal.xy *= 1.0;
            sampled_normal.z = sqrt(clamp(1.0 - dot(sampled_normal.xy, sampled_normal.xy), 0.0, 1.0));
            sampled_normal = normalize(sampled_normal);
            // TBN rows are tangent/bitangent/normal — the same orientation the GLSL `mat3(T,B,N)` used.
            final_normal = normalize(mul(sampled_normal, TBN));
        }
    }

    payload.dist = RayTCurrent();
    payload.emission = final_emission;
    payload.albedo_packed = pack_unorm_4x8(float4(out_albedo, 1.0));
    payload.normal_packed = pack_normal(final_normal);

    float final_roughness = mesh_info.material_roughness_factor;
    float final_metallic = mesh_info.material_metallic_factor;

    if (mesh_info.material_metallic_roughness_image != NULL_TEXTURE) {
        float2 mr_uv = select_uv(uv_mask, UV_SET_METALLIC_ROUGHNESS, uv0, uv1);
        float4 mr_sample = sample_texture(mesh_info.material_metallic_roughness_image, mesh_info.material_metallic_roughness_sampler, mr_uv, float4(1.0));
        // glTF spec: G = roughness, B = metallic.
        final_roughness *= mr_sample.g;
        final_metallic  *= mr_sample.b;
    }

    payload.material_info = pack_half_2x16(float2(final_roughness, final_metallic));
    payload.transmission_ior_packed = pack_half_2x16(float2(mesh_info.material_transmission_factor, mesh_info.material_ior));
}
