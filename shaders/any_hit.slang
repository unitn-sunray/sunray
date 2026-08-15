// Any-hit: alpha-cutout test for glTF MASK materials. Geometry that is fully
// opaque (alpha_mode == 0) returns immediately so the closest-hit can run.
// Mirrors shaders/any_hit.glsl.

import rt_types;
import rt_utils;

[vk::push_constant] ConstantBuffer<RaytracingPC> pc;

[shader("anyhit")]
void any_hit(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    StructuredBuffer<MeshInfo> meshes = pc.get_meshes_info();

    uint blas_instance_id = InstanceID();
    MeshInfo mesh_info = meshes[blas_instance_id];

    // OPAQUE — closest-hit handles it.
    // Material fields are inlined into MeshInfo to work around the Slang
    // heap-mode nested-struct access-chain bug (see rt_types.slang::MeshInfo).
    if (mesh_info.material_alpha_mode == 0) {
        return;
    }

    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                         attribs.barycentrics.x,
                         attribs.barycentrics.y);

    uint index_offset = PrimitiveIndex() * 3;
    StructuredBuffer<uint> index_buf = mesh_info.get_indices();
    uint i0 = index_buf[index_offset + 0];
    uint i1 = index_buf[index_offset + 1];
    uint i2 = index_buf[index_offset + 2];

    // Alpha comes from the base-colour texture, so only that slot's UV set matters.
    StructuredBuffer<VertexAttributes> vertex_buf = mesh_info.get_vertices();
    VertexAttributes a0 = vertex_buf[i0];
    VertexAttributes a1 = vertex_buf[i1];
    VertexAttributes a2 = vertex_buf[i2];
    uint uv_mask = mesh_info.material_uv_set_mask;
    float2 uv = select_uv(uv_mask, UV_SET_BASE_COLOR, a0.uv0, a0.uv1) * bary.x
              + select_uv(uv_mask, UV_SET_BASE_COLOR, a1.uv0, a1.uv1) * bary.y
              + select_uv(uv_mask, UV_SET_BASE_COLOR, a2.uv0, a2.uv1) * bary.z;

    float4 base_color = sample_texture(mesh_info.material_base_color_image, mesh_info.material_base_color_sampler, uv, mesh_info.material_base_color_value);

    if (base_color.a < mesh_info.material_alpha_cutoff) {
        IgnoreHit();
    }
}
