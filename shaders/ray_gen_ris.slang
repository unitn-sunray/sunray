// First raytracing pass: ReSTIR DI initial-candidate audition + ReSTIR GI
// initial sample. Writes the G-buffer (depth/normal/diffuse/motion vector),
// runs primary trace with mirror/glass/diffuse handling, then performs RIS
// candidate selection with temporal reuse. Mirrors shaders/ray_gen_ris.glsl.

import rt_types;
import rt_utils;

[vk::push_constant] ConstantBuffer<RaytracingPC> pc;

[shader("raygeneration")]
void ray_gen_ris() {
    RaytracingAccelerationStructure tlas = tlas_from_address(pc.tlas);
    // Read matrices via the BDA pointer (see `RaytracingPC.matrices` comment).
    // The CPU side (`set_matrices` in resource_manager.rs) transposes each
    // matrix before upload, so each float4 here IS a row of the intended
    // matrix — `float4x4(rows...)` reconstructs it directly.
    Matrices m = *pc.matrices;
    float4x4 mat_view_inverse   = float4x4(m.vi0, m.vi1, m.vi2, m.vi3);
    float4x4 mat_proj_inverse   = float4x4(m.pi0, m.pi1, m.pi2, m.pi3);
    float4x4 mat_prev_view_proj = float4x4(m.pp0, m.pp1, m.pp2, m.pp3);
    StructuredBuffer<EmissiveTriangle> emissive_triangles = pc.emissive_triangles;
    StructuredBuffer<EmissiveIndirectionEntry> emissive_indirection = pc.emissive_indirection;
    StructuredBuffer<EntityTransform> entity_transforms = pc.entity_transforms;

    RWTexture2D<float>  depth_img      = pc.depth_img;
    RWTexture2D<float4> normal_img     = pc.normal_img;
    RWTexture2D<float4> diffuse_img    = pc.diffuse_img;
    RWTexture2D<float2> motion_vec_img = pc.motion_vec_img;

    uint cur_buf  = current_reservoir_idx(pc.frame_count);
    uint hist_buf = history_reservoir_idx(pc.frame_count);
    Reservoir*   reservoir_cur     = pc.reservoirs[cur_buf];
    Reservoir*   reservoir_hist    = pc.reservoirs[hist_buf];
    ReservoirGI* reservoir_gi_cur  = pc.reservoirs_gi[cur_buf];
    ReservoirGI* reservoir_gi_hist = pc.reservoirs_gi[hist_buf];

    uint2 launch_size = DispatchRaysDimensions().xy;
    uint2 launch_id   = DispatchRaysIndex().xy;
    int2 pixel_coord  = int2(launch_id);

    Rng rng = init_rng(launch_id, pc.frame_count, launch_size);

    float2 pixel_center = float2(launch_id) + float2(0.5);
    float2 inUV = pixel_center / float2(launch_size);
    float2 d = inUV * 2.0 - 1.0;

    float4 origin    = mul(mat_view_inverse, float4(0, 0, 0, 1));
    float4 target    = mul(mat_proj_inverse, float4(d.x, d.y, 1, 1));
    float4 direction = mul(mat_view_inverse, float4(normalize(target.xyz), 0));

    float3 rayOrigin = origin.xyz;
    float3 rayDir    = direction.xyz;

    RayPayload prd = (RayPayload)0;

    // Phase 1: primary ray + glass/mirror passthrough until first diffuse hit.
    float3 hitPos = float3(0);
    float3 hit_normal = float3(0);
    float3 hit_albedo = float3(0);
    float roughness = 0.5;
    float metallic  = 0.0;
    float3 V_view = float3(0);
    float2 prev_uv = float2(-1.0);
    bool prev_valid = false;
    bool found_diffuse_surface = false;
    float virtual_distance = 0.0;

    for (int virtual_bounce = 0; virtual_bounce < 20; virtual_bounce++) {
        RayDesc ray;
        ray.Origin = rayOrigin;
        ray.TMin = 0.001;
        ray.Direction = rayDir;
        ray.TMax = 10000.0;
        TraceRay(tlas, RAY_FLAG_NONE, 0xFFu, 0, 0, 0, ray, prd);

        if (prd.dist < 0.0) {
            break; // sky
        }

        hitPos = rayOrigin + rayDir * prd.dist;
        hit_normal = unpack_normal(prd.normal_packed);
        hit_albedo = unpack_unorm_4x8(prd.albedo_packed).rgb;

        float2 mat_info = unpack_half_2x16(prd.material_info);
        roughness = max(mat_info.x, 0.01);
        metallic = clamp(mat_info.y, 0.0, 1.0);

        float2 trans_ior = unpack_half_2x16(prd.transmission_ior_packed);
        float transmission = trans_ior.x;

        V_view = -rayDir;
        virtual_distance += prd.dist;

        if (transmission > 0.5) {
            float ior = max(trans_ior.y, 1.0);
            bool is_inside = dot(rayDir, hit_normal) > 0.0;
            float3 N = is_inside ? -hit_normal : hit_normal;
            float eta = is_inside ? (ior / 1.0) : (1.0 / ior);

            float cos_theta = min(dot(-rayDir, N), 1.0);
            float R0 = (1.0 - eta) / (1.0 + eta);
            R0 = R0 * R0;
            float fresnel = R0 + (1.0 - R0) * pow(1.0 - cos_theta, 5.0);

            float3 refracted = refract(rayDir, N, eta);
            if (length(refracted) < 0.01) fresnel = 1.0;

            if (rnd(rng) < fresnel) {
                rayDir = reflect(rayDir, N);
            } else {
                rayDir = refracted;
            }
            rayOrigin = hitPos + rayDir * 0.001;
        // Threshold must match ray_gen_final's `roughness > 0.2` rough-surface
        // gate: a metallic surface below it never gets NEE/ReSTIR there (the walk
        // just continues), so it must be followed here or the pixel's reservoirs
        // are built for the wrong surface (or left empty by the same gate below)
        // and the reflection resolves to black.
        } else if (metallic > 0.9 && roughness <= 0.2) {
            rayOrigin = hitPos + hit_normal * 0.001;
            rayDir = reflect(rayDir, hit_normal);
        } else {
            float3 virtual_world_pos = origin.xyz + direction.xyz * virtual_distance;
            float4 prev_clip = mul(mat_prev_view_proj, float4(virtual_world_pos, 1.0));

            const float MIN_PREV_W = 0.01;
            prev_valid = prev_clip.w > MIN_PREV_W;
            if (prev_valid) {
                float2 prev_ndc = prev_clip.xy / prev_clip.w;
                prev_uv = prev_ndc * 0.5 + 0.5;
                prev_valid = all(prev_uv >= float2(0.0)) && all(prev_uv < float2(1.0));
            }

            float2 motion_vector = prev_valid ? (inUV - prev_uv) : (inUV + float2(2.0));
            float3 denoiser_albedo = lerp(hit_albedo, float3(1.0), metallic);

            depth_img[pixel_coord]      = virtual_distance;
            normal_img[pixel_coord]     = float4(hit_normal, roughness);
            diffuse_img[pixel_coord]    = float4(denoiser_albedo, 0.0);
            motion_vec_img[pixel_coord] = motion_vector;

            found_diffuse_surface = true;
            break;
        }
    }

    if (!found_diffuse_surface) {
        // Sky pixel: empty reservoirs, sentinel G-buffer values.
        float2 sky_motion = inUV + float2(2.0);
        float4 sky_prev_clip = mul(mat_prev_view_proj, float4(rayDir, 0.0));
        const float MIN_SKY_W = 0.01;
        if (sky_prev_clip.w > MIN_SKY_W) {
            float2 sky_prev_ndc = sky_prev_clip.xy / sky_prev_clip.w;
            float2 sky_prev_uv = sky_prev_ndc * 0.5 + 0.5;
            if (all(sky_prev_uv >= float2(0.0)) && all(sky_prev_uv < float2(1.0))) {
                sky_motion = inUV - sky_prev_uv;
            }
        }
        depth_img[pixel_coord]      = 100000.0;
        normal_img[pixel_coord]     = float4(0.0);
        diffuse_img[pixel_coord]    = float4(0.0);
        motion_vec_img[pixel_coord] = float2(0.0);

        Reservoir empty;
        empty.light_pos = float3(0);
        empty.w_sum = 0.0;
        empty.light_normal = float3(0);
        empty.M = 0.0;
        empty.light_idx = 0u;
        empty.W = 0.0;
        empty.hit_normal_packed = 0u;
        empty.depth = 0.0;
        uint pi = get_pixel_index(pixel_coord, launch_size);
        reservoir_cur[pi] = empty;
        return;
    }

    // Phase 2: RIS audition.
    Reservoir current_r;
    current_r.light_pos = float3(0);
    current_r.w_sum = 0.0;
    current_r.light_normal = float3(0);
    current_r.M = 0.0;
    current_r.light_idx = 0u;
    current_r.W = 0.0;
    current_r.hit_normal_packed = 0u;
    current_r.depth = 0.0;

    uint num_lights, _stride;
    emissive_indirection.GetDimensions(num_lights, _stride);
    const int RIS_CANDIDATES = 16;

    if (num_lights > 0 && roughness > 0.2) {
        for (int i = 0; i < RIS_CANDIDATES; i++) {
            uint cand_idx = min(uint(rnd(rng) * num_lights), num_lights - 1);
            EmissiveIndirectionEntry entry = emissive_indirection[cand_idx];
            EmissiveTriangle cand_light = emissive_triangles[entry.blas_tri_index];
            EntityTransform xform = entity_transforms[entry.entity_id];

            float3 wv0 = transform_point(xform, cand_light.v0.xyz);
            float3 wv1 = transform_point(xform, cand_light.v1.xyz);
            float3 wv2 = transform_point(xform, cand_light.v2.xyz);

            float3 edge1 = wv1 - wv0;
            float3 edge2 = wv2 - wv0;
            float cand_area = 0.5 * length(cross(edge1, edge2));

            float sqr1 = sqrt(rnd(rng));
            float u = 1.0 - sqr1;
            float v = rnd(rng) * sqr1;
            float w = 1.0 - u - v;

            float3 cand_pos    = wv0 * u + wv1 * v + wv2 * w;
            float3 cand_normal = normalize(cross(wv1 - wv0, wv2 - wv0));

            float3 f_y = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, cand_light, cand_pos, cand_normal);
            float p_hat = max(f_y.r, max(f_y.g, f_y.b));
            float p_y = 1.0 / max(float(num_lights) * cand_area, 0.0001);

            current_r.w_sum += (p_hat / p_y);
            current_r.M += 1.0;
            if (rnd(rng) < ((p_hat / p_y) / max(current_r.w_sum, 0.0001))) {
                current_r.light_idx = cand_idx;
                current_r.light_pos = cand_pos;
                current_r.light_normal = cand_normal;
            }
        }

        if (current_r.w_sum > 0.0) {
            EmissiveIndirectionEntry winner_entry = emissive_indirection[current_r.light_idx];
            EmissiveTriangle winner = emissive_triangles[winner_entry.blas_tri_index];
            float3 f_y_winner = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, winner, current_r.light_pos, current_r.light_normal);
            float p_hat_winner = max(f_y_winner.r, max(f_y_winner.g, f_y_winner.b));
            current_r.W = current_r.w_sum / max(current_r.M * p_hat_winner, 0.0001);
        }

        // Temporal reuse.
        if (pc.frame_count > 0 && prev_valid) {
            float2 prev_pixel_f = prev_uv * float2(launch_size);
            float2 di_jitter = float2(rnd(rng), rnd(rng)) - 0.5;
            int2 prev_coord = int2(prev_pixel_f + di_jitter);
            if (prev_coord.x >= 0 && prev_coord.y >= 0 && prev_coord.x < int(launch_size.x) && prev_coord.y < int(launch_size.y)) {
                uint pi = get_pixel_index(prev_coord, launch_size);
                Reservoir history_r = reservoir_hist[pi];
                history_r.M = min(history_r.M, 10.0);
                history_r.W = min(history_r.W, 20.0);

                float3 hist_normal = unpack_normal(history_r.hit_normal_packed);
                float normal_conf_di = smoothstep(0.9, 0.99, dot(hit_normal, hist_normal));
                float depth_diff_di  = abs(virtual_distance - history_r.depth) / max(virtual_distance, 1e-4);
                float depth_conf_di  = 1.0 - smoothstep(0.05, 0.20, depth_diff_di);
                float conf_di        = normal_conf_di * depth_conf_di;

                history_r.M *= conf_di;

                if (history_r.W > 0.0) {
                    history_r.light_idx = min(history_r.light_idx, num_lights - 1);
                    EmissiveIndirectionEntry hist_entry = emissive_indirection[history_r.light_idx];
                    EmissiveTriangle hist_light = emissive_triangles[hist_entry.blas_tri_index];
                    float3 f_y_hist = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, hist_light, history_r.light_pos, history_r.light_normal);
                    float p_hat_hist = max(f_y_hist.r, max(f_y_hist.g, f_y_hist.b));
                    merge_reservoirs(current_r, history_r, p_hat_hist, rnd(rng));

                    EmissiveIndirectionEntry merged_entry = emissive_indirection[current_r.light_idx];
                    EmissiveTriangle merged_winner = emissive_triangles[merged_entry.blas_tri_index];
                    float3 f_y_merged = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, merged_winner, current_r.light_pos, current_r.light_normal);
                    float p_hat_merged = max(f_y_merged.r, max(f_y_merged.g, f_y_merged.b));
                    current_r.W = current_r.w_sum / max(current_r.M * p_hat_merged, 0.0001);
                }
            }
        }
    }

    // Visibility reuse (Bitterli et al. 2020): the RIS audition and temporal
    // reuse above use an *unshadowed* target function, so an occluded light can
    // win the reservoir and persist across frames with full weight, flickering
    // on/off in penumbrae. Test visibility of the selected sample once and zero
    // W when occluded — keep M so temporal confidence isn't reset. The occluded
    // sample then contributes nothing to next frame's temporal reuse (its merge
    // weight is p_hat * W * M = 0) instead of boiling.
    if (current_r.W > 0.0) {
        float3 vis_dir = current_r.light_pos - hitPos;
        float vis_dist = max(length(vis_dir), 0.0001);
        vis_dir /= vis_dist;

        if (dot(hit_normal, vis_dir) <= 0.0) {
            current_r.W = 0.0;
        } else {
            uint vis_flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
            // Skip degenerate short segments (TMax<TMin); coincident endpoints are visible.
            if (vis_dist > 0.002) {
                prd.dist = 1.0;
                RayDesc vray;
                vray.Origin = hitPos + hit_normal * 0.001;
                vray.TMin = 0.001;
                vray.Direction = vis_dir;
                vray.TMax = vis_dist - 0.001;
                TraceRay(tlas, vis_flags, 0xFFu, 0, 0, 0, vray, prd);
            } else {
                prd.dist = -1.0;
            }
            if (prd.dist >= 0.0) {
                current_r.W = 0.0; // occluded
            }
        }
    }

    current_r.hit_normal_packed = pack_normal(hit_normal);
    current_r.depth = virtual_distance;
    {
        uint pi = get_pixel_index(pixel_coord, launch_size);
        reservoir_cur[pi] = current_r;
    }

    // Phase 3: ReSTIR GI initial sample.
    ReservoirGI current_gi_r;
    current_gi_r.sample_pos = float3(0);
    current_gi_r.w_sum = 0.0;
    current_gi_r.sample_radiance = float3(0);
    current_gi_r.M = 0.0;
    current_gi_r.sample_normal_packed = 0u;
    current_gi_r.W = 0.0;
    current_gi_r.hit_normal_packed = 0u;
    current_gi_r.depth = 0.0;

    float3 gi_dir = get_random_bounce(hit_normal, rnd(rng), rnd(rng));
    float gi_NdotL = max(dot(hit_normal, gi_dir), 0.0);

    if (gi_NdotL > 0.0) {
        float3 gi_origin = hitPos + hit_normal * 0.001;
        RayDesc ray;
        ray.Origin = gi_origin;
        ray.TMin = 0.001;
        ray.Direction = gi_dir;
        ray.TMax = 10000.0;
        TraceRay(tlas, RAY_FLAG_NONE, 0xFFu, 0, 0, 0, ray, prd);

        float3 sample_pos = float3(0.0);
        float3 sample_normal = float3(0.0);
        float3 sample_radiance = float3(0.0);

        if (prd.dist > 0.0) {
            sample_pos = gi_origin + gi_dir * prd.dist;
            sample_normal = unpack_normal(prd.normal_packed);
            float3 x2_albedo = unpack_unorm_4x8(prd.albedo_packed).rgb;
            sample_radiance = prd.emission;

            uint nee_num_lights = num_lights;
            if (nee_num_lights > 0) {
                uint nee_idx = min(uint(rnd(rng) * nee_num_lights), nee_num_lights - 1);
                EmissiveIndirectionEntry nee_entry = emissive_indirection[nee_idx];
                EmissiveTriangle nee_light = emissive_triangles[nee_entry.blas_tri_index];
                EntityTransform nee_xform = entity_transforms[nee_entry.entity_id];

                float3 nwv0 = transform_point(nee_xform, nee_light.v0.xyz);
                float3 nwv1 = transform_point(nee_xform, nee_light.v1.xyz);
                float3 nwv2 = transform_point(nee_xform, nee_light.v2.xyz);

                float sq = sqrt(rnd(rng));
                float nu = 1.0 - sq;
                float nv = rnd(rng) * sq;
                float nw = 1.0 - nu - nv;

                float3 nee_pos    = nwv0 * nu + nwv1 * nv + nwv2 * nw;
                float3 nee_normal = normalize(cross(nwv1 - nwv0, nwv2 - nwv0));
                float nee_area    = 0.5 * length(cross(nwv1 - nwv0, nwv2 - nwv0));

                float3 to_light = nee_pos - sample_pos;
                float nee_dist = max(length(to_light), 0.0001);
                to_light /= nee_dist;

                float nee_cos_surf  = max(dot(sample_normal, to_light), 0.0);
                float nee_cos_light = max(dot(nee_normal, -to_light), 0.0);

                if (nee_cos_surf > 0.0 && nee_cos_light > 0.0) {
                    uint nee_flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
                    // Skip degenerate short segments (TMax<TMin); coincident endpoints are visible.
                    if (nee_dist > 0.002) {
                        prd.dist = 1.0;
                        RayDesc sray;
                        sray.Origin = sample_pos + sample_normal * 0.001;
                        sray.TMin = 0.001;
                        sray.Direction = to_light;
                        sray.TMax = nee_dist - 0.001;
                        TraceRay(tlas, nee_flags, 0xFFu, 0, 0, 0, sray, prd);
                    } else {
                        prd.dist = -1.0;
                    }

                    if (prd.dist < 0.0) {
                        float nee_pdf_sa = (nee_dist * nee_dist) / max(nee_cos_light * nee_area * float(nee_num_lights), 0.0001);
                        sample_radiance += (nee_light.emission.rgb * x2_albedo * nee_cos_surf) / (nee_pdf_sa * 3.14159);
                    }
                }
            }
        }

        const float GI_RADIANCE_CLAMP = 5.0;
        sample_radiance = min(sample_radiance, float3(GI_RADIANCE_CLAMP));

        float p_hat = gi_target_pdf(hitPos, hit_normal, hit_albedo, metallic, sample_pos, sample_radiance);
        float pdf   = gi_NdotL / 3.14159;

        current_gi_r.M                    = 1.0;
        current_gi_r.w_sum                = (pdf > 0.0) ? (p_hat / pdf) : 0.0;
        current_gi_r.W                    = (p_hat > 0.0) ? (current_gi_r.w_sum / (current_gi_r.M * p_hat)) : 0.0;
        current_gi_r.sample_pos           = sample_pos;
        current_gi_r.sample_normal_packed = pack_normal(sample_normal);
        current_gi_r.sample_radiance      = sample_radiance;
    }

    if (pc.frame_count > 0 && prev_valid) {
        float2 prev_pixel_f_gi = prev_uv * float2(launch_size);
        float2 gi_jitter = float2(rnd(rng), rnd(rng)) - 0.5;
        int2 prev_coord_gi = int2(prev_pixel_f_gi + gi_jitter);
        if (prev_coord_gi.x >= 0 && prev_coord_gi.y >= 0 && prev_coord_gi.x < int(launch_size.x) && prev_coord_gi.y < int(launch_size.y)) {
            uint pi = get_pixel_index(prev_coord_gi, launch_size);
            ReservoirGI history_gi = reservoir_gi_hist[pi];

            float3 gi_hist_normal = unpack_normal(history_gi.hit_normal_packed);
            float normal_conf = smoothstep(0.8, 0.95, dot(hit_normal, gi_hist_normal));
            float depth_diff  = abs(virtual_distance - history_gi.depth) / max(virtual_distance, 1e-4);
            float depth_conf  = 1.0 - smoothstep(0.05, 0.20, depth_diff);
            float conf        = normal_conf * depth_conf;

            history_gi.M = min(history_gi.M, 12.0) * conf;
            history_gi.W = min(history_gi.W, 10.0);

            if (history_gi.W > 0.0 && history_gi.M > 0.0) {
                float p_hat_hist = gi_target_pdf(hitPos, hit_normal, hit_albedo, metallic, history_gi.sample_pos, history_gi.sample_radiance);
                merge_reservoirs_gi(current_gi_r, history_gi, p_hat_hist, 1.0, rnd(rng));
                float p_hat_merged = gi_target_pdf(hitPos, hit_normal, hit_albedo, metallic, current_gi_r.sample_pos, current_gi_r.sample_radiance);
                current_gi_r.W = (p_hat_merged > 1e-6) ? (current_gi_r.w_sum / (current_gi_r.M * p_hat_merged)) : 0.0;
            }
        }
    }

    current_gi_r.hit_normal_packed = pack_normal(hit_normal);
    current_gi_r.depth = virtual_distance;
    {
        uint pi = get_pixel_index(pixel_coord, launch_size);
        reservoir_gi_cur[pi] = current_gi_r;
    }
}
