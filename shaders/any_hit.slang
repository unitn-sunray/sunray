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

    StructuredBuffer<VertexAttributes> vertex_buf = mesh_info.get_vertices();
    float2 uv0 = vertex_buf[i0].base_color_tex_coord;
    float2 uv1 = vertex_buf[i1].base_color_tex_coord;
    float2 uv2 = vertex_buf[i2].base_color_tex_coord;
    float2 uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    float4 base_color = sample_texture(mesh_info.material_base_color_image, mesh_info.material_base_color_sampler, uv, mesh_info.material_base_color_value);

    if (base_color.a < mesh_info.material_alpha_cutoff) {
        IgnoreHit();
    }
}
