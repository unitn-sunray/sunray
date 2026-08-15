// Second raytracing pass: path tracing with ReSTIR DI/GI spatial reuse at
// the first rough surface, plus NEE on later bounces. Reads the reservoirs
// written by ray_gen_ris. Mirrors shaders/ray_gen_final.glsl.

import rt_types;
import rt_utils;

[vk::push_constant] ConstantBuffer<RaytracingPC> pc;

[shader("raygeneration")]
void ray_gen_final() {
    RaytracingAccelerationStructure tlas = tlas_from_address(pc.tlas);
    // CPU pre-transposes (see `set_matrices` in resource_manager.rs); each
    // float4 here is already a row of the intended matrix.
    Matrices m = *pc.matrices;
    float4x4 mat_view_inverse = float4x4(m.vi0, m.vi1, m.vi2, m.vi3);
    float4x4 mat_proj_inverse = float4x4(m.pi0, m.pi1, m.pi2, m.pi3);
    StructuredBuffer<EmissiveTriangle> emissive_triangles = pc.emissive_triangles;
    StructuredBuffer<EmissiveIndirectionEntry> emissive_indirection = pc.emissive_indirection;
    StructuredBuffer<EntityTransform> entity_transforms = pc.entity_transforms;
    RWImage_r11f_g11f_b10f raw_color   = pc.raw_color;
    RWImage_r16f           depth_img   = pc.depth_img;
    RWImage_rgba8_snorm    normal_img  = pc.normal_img;
    RWImage_rg16f          motion_vec_img = pc.motion_vec_img;

    uint cur_buf = current_reservoir_idx(pc.frame_count);
    Reservoir*   reservoir_cur    = pc.reservoirs[cur_buf];
    ReservoirGI* reservoir_gi_cur = pc.reservoirs_gi[cur_buf];

    Texture2D<float4> blue_noise_tex = pc.blue_noise_tex;
    SamplerState blue_noise_sampler  = pc.blue_noise_sampler;

    uint2 launch_size = DispatchRaysDimensions().xy;
    uint2 launch_id   = DispatchRaysIndex().xy;
    int2 pixel_coord  = int2(launch_id);

    Rng rng = init_rng(launch_id, pc.frame_count, launch_size);

    float3 total_radiance = float3(0.0);
    const int SAMPLES = 1;
    const int BOUNCES = 10;
    const int SHADOW_BOUNCES = BOUNCES / 2;

    uint blue_noise_w, blue_noise_h, blue_noise_levels;
    blue_noise_tex.GetDimensions(0u, blue_noise_w, blue_noise_h, blue_noise_levels);
    int2 bn_size = int2(blue_noise_w, blue_noise_h);
    int2 noise_coord_1 = int2(launch_id) % bn_size;
    int2 noise_coord_2 = (int2(launch_id) + int2(47, 71)) % bn_size;
    float bn_1 = blue_noise_tex.Load(int3(noise_coord_1, 0)).r;
    float bn_2 = blue_noise_tex.Load(int3(noise_coord_2, 0)).r;

    uint num_lights, _stride;
    emissive_indirection.GetDimensions(num_lights, _stride);

    RayPayload prd = (RayPayload)0;

    for (int sample_i = 0; sample_i < SAMPLES; sample_i++) {
        float2 pixel_center = float2(launch_id) + float2(0.5);
        float2 inUV = pixel_center / float2(launch_size);
        float2 d = inUV * 2.0 - 1.0;

        float4 origin    = mul(mat_view_inverse, float4(0, 0, 0, 1));
        float4 target    = mul(mat_proj_inverse, float4(d.x, d.y, 1, 1));
        float4 direction = mul(mat_view_inverse, float4(normalize(target.xyz), 0));

        float3 rayOrigin = origin.xyz;
        float3 rayDir    = direction.xyz;

        float3 throughput = float3(1.0);
        float3 radiance   = float3(0.0);
        bool restir_evaluated = false;
        bool prev_did_nee = false;

        for (int bounce = 0; bounce < BOUNCES; bounce++) {
            RayDesc ray;
            ray.Origin = rayOrigin;
            ray.TMin = 0.001;
            ray.Direction = rayDir;
            ray.TMax = 10000.0;
            TraceRay(tlas, RAY_FLAG_NONE, 0xFFu, 0, 0, 0, ray, prd);

            if (prd.dist < 0.0) {
                break;
            }

            float3 hit_normal = unpack_normal(prd.normal_packed);
            float3 hit_albedo = unpack_unorm_4x8(prd.albedo_packed).rgb;
            float3 hitPos = rayOrigin + rayDir * prd.dist;
            float3 V_view = -rayDir;

            float2 mat_info = unpack_half_2x16(prd.material_info);
            float roughness = max(mat_info.x, 0.01);
            float metallic = clamp(mat_info.y, 0.0, 1.0);

            float2 trans_ior = unpack_half_2x16(prd.transmission_ior_packed);
            float transmission = trans_ior.x;
            float ior = max(trans_ior.y, 1.0);

            if (!prev_did_nee) {
                radiance += prd.emission * throughput;
            }
            prev_did_nee = false;
            float brightness = max(prd.emission.r, max(prd.emission.g, prd.emission.b));
            if (brightness > 1.0) break;

            if (transmission > 0.5) {
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
                    if (is_inside) {
                        float3 absorption = 1.0 - hit_albedo;
                        throughput *= exp(-absorption * prd.dist * 5.0);
                    } else {
                        throughput *= hit_albedo;
                    }
                }

                rayOrigin = hitPos + rayDir * 0.001;
                continue;
            }

            if (num_lights > 0 && bounce < SHADOW_BOUNCES) {
                if (!restir_evaluated && roughness > 0.2) {
                    restir_evaluated = true;

                    uint pi_center = get_pixel_index(pixel_coord, launch_size);
                    Reservoir center_r = reservoir_cur[pi_center];
                    Reservoir spatial_r;
                    spatial_r.light_pos = float3(0);
                    spatial_r.w_sum = 0.0;
                    spatial_r.light_normal = float3(0);
                    spatial_r.M = 0.0;
                    spatial_r.light_idx = 0u;
                    spatial_r.W = 0.0;
                    spatial_r.hit_normal_packed = 0u;
                    spatial_r.depth = 0.0;

                    if (center_r.W > 0.0 && center_r.light_idx < num_lights) {
                        center_r.light_idx = min(center_r.light_idx, num_lights - 1);
                        EmissiveIndirectionEntry center_entry = emissive_indirection[center_r.light_idx];
                        EmissiveTriangle center_light = emissive_triangles[center_entry.blas_tri_index];
                        float3 f_y_center = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, center_light, center_r.light_pos, center_r.light_normal);
                        float p_hat_center = max(f_y_center.r, max(f_y_center.g, f_y_center.b));
                        merge_reservoirs(spatial_r, center_r, p_hat_center, rnd(rng));
                    }

                    const int SPATIAL_SAMPLES = 5;
                    const float SPATIAL_RADIUS = 30.0;
                    float current_depth = length(hitPos - origin.xyz);

                    for (int s = 0; s < SPATIAL_SAMPLES; s++) {
                        float angle = rnd(rng) * 2.0 * 3.14159;
                        float radius = sqrt(rnd(rng)) * SPATIAL_RADIUS;
                        int2 neighbor_coord = pixel_coord + int2(cos(angle) * radius, sin(angle) * radius);
                        if (neighbor_coord.x < 0 || neighbor_coord.y < 0 || neighbor_coord.x >= int(launch_size.x) || neighbor_coord.y >= int(launch_size.y)) continue;

                        float3 neighbor_normal = normal_img[neighbor_coord].xyz;
                        float neighbor_depth   = depth_img[neighbor_coord];

                        if (dot(hit_normal, neighbor_normal) < 0.9) continue;
                        if (abs(current_depth - neighbor_depth) > 0.1 * current_depth) continue;

                        uint pi_n = get_pixel_index(neighbor_coord, launch_size);
                        Reservoir neighbor_r = reservoir_cur[pi_n];
                        neighbor_r.W = min(neighbor_r.W, 20.0);
                        neighbor_r.M = min(neighbor_r.M, 10.0);
                        if (neighbor_r.W > 0.0 && neighbor_r.light_idx < num_lights) {
                            neighbor_r.light_idx = min(neighbor_r.light_idx, num_lights - 1);
                            EmissiveIndirectionEntry neighbor_entry = emissive_indirection[neighbor_r.light_idx];
                            EmissiveTriangle neighbor_light = emissive_triangles[neighbor_entry.blas_tri_index];
                            float3 f_y_neighbor = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, neighbor_light, neighbor_r.light_pos, neighbor_r.light_normal);
                            float p_hat_neighbor = max(f_y_neighbor.r, max(f_y_neighbor.g, f_y_neighbor.b));
                            merge_reservoirs(spatial_r, neighbor_r, p_hat_neighbor, rnd(rng));
                        }
                    }

                    if (spatial_r.w_sum > 0.0) {
                        EmissiveIndirectionEntry winner_entry = emissive_indirection[spatial_r.light_idx];
                        EmissiveTriangle winner = emissive_triangles[winner_entry.blas_tri_index];
                        float3 f_y_winner = eval_unshadowed_light(hitPos, hit_normal, V_view, hit_albedo, roughness, metallic, winner, spatial_r.light_pos, spatial_r.light_normal);
                        float p_hat_winner = max(f_y_winner.r, max(f_y_winner.g, f_y_winner.b));
                        spatial_r.W = spatial_r.w_sum / max(spatial_r.M * p_hat_winner, 1e-3);
                        spatial_r.W = min(spatial_r.W, 50.0);

                        float3 shadow_dir = spatial_r.light_pos - hitPos;
                        float shadow_dist = max(length(shadow_dir), 0.0001);
                        shadow_dir /= shadow_dist;

                        if (dot(hit_normal, shadow_dir) > 0.0) {
                            uint shadow_ray_flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
                            // Skip degenerate short segments (TMax<TMin); coincident endpoints are visible.
                            if (shadow_dist > 0.002) {
                                prd.dist = 1.0;
                                RayDesc sray;
                                sray.Origin = hitPos;
                                sray.TMin = 0.001;
                                sray.Direction = shadow_dir;
                                sray.TMax = shadow_dist - 0.001;
                                TraceRay(tlas, shadow_ray_flags, 0xFFu, 0, 0, 0, sray, prd);
                            } else {
                                prd.dist = -1.0;
                            }

                            if (prd.dist < 0.0) {
                                radiance += f_y_winner * throughput * spatial_r.W;
                            }
                            prev_did_nee = true;
                        }
                    }

                    // ReSTIR GI spatial reuse.
                    uint pi_gi_center = get_pixel_index(pixel_coord, launch_size);
                    ReservoirGI combined = reservoir_gi_cur[pi_gi_center];

                    const int GI_SPATIAL_SAMPLES = 3;
                    const float GI_SPATIAL_RADIUS = 20.0;
                    float gi_current_depth = length(hitPos - origin.xyz);

                    for (int s = 0; s < GI_SPATIAL_SAMPLES; s++) {
                        float gi_angle  = rnd(rng) * 2.0 * 3.14159;
                        float gi_radius = sqrt(rnd(rng)) * GI_SPATIAL_RADIUS;
                        int2 nc = pixel_coord + int2(cos(gi_angle) * gi_radius, sin(gi_angle) * gi_radius);

                        if (all(nc == pixel_coord)) continue;
                        if (nc.x < 0 || nc.y < 0 || nc.x >= int(launch_size.x) || nc.y >= int(launch_size.y)) continue;

                        float3 neighbor_normal = normal_img[nc].xyz;
                        float neighbor_depth   = depth_img[nc];

                        if (dot(hit_normal, neighbor_normal) < 0.9) continue;
                        if (abs(gi_current_depth - neighbor_depth) > 0.1 * gi_current_depth) continue;

                        uint pi_nn = get_pixel_index(nc, launch_size);
                        ReservoirGI neighbor_r = reservoir_gi_cur[pi_nn];
                        if (neighbor_r.W <= 0.0) continue;

                        neighbor_r.W = min(neighbor_r.W, 10.0);
                        neighbor_r.M = min(neighbor_r.M, 10.0);

                        float2 npc = float2(nc) + float2(0.5);
                        float2 nUV = npc / float2(launch_size);
                        float2 nd  = nUV * 2.0 - 1.0;
                        float4 nt  = mul(mat_proj_inverse, float4(nd.x, nd.y, 1, 1));
                        float4 ndir = mul(mat_view_inverse, float4(normalize(nt.xyz), 0));
                        float3 neighbor_x1 = origin.xyz + ndir.xyz * neighbor_depth;

                        float3 w_new = neighbor_r.sample_pos - hitPos;
                        float3 w_old = neighbor_r.sample_pos - neighbor_x1;
                        float d_new = max(length(w_new), 1e-4);
                        float d_old = max(length(w_old), 1e-4);
                        float3 n_x2 = unpack_normal(neighbor_r.sample_normal_packed);
                        float cos_new = max(dot(n_x2, -w_new / d_new), 0.0);
                        float cos_old = max(dot(n_x2, -w_old / d_old), 0.0);
                        if (cos_new <= 0.0 || cos_old <= 0.0) continue;

                        float jacobian = (cos_new * d_old * d_old) / max(cos_old * d_new * d_new, 1e-4);
                        jacobian = clamp(jacobian, 0.0, 10.0);

                        float3 gi_spatial_dir = w_new / d_new;
                        if (dot(hit_normal, gi_spatial_dir) <= 0.0) continue;
                        uint gi_spatial_vis = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
                        // Skip degenerate short segments (TMax<TMin); coincident endpoints are visible.
                        if (d_new > 0.002) {
                            prd.dist = 1.0;
                            RayDesc gray;
                            gray.Origin = hitPos;
                            gray.TMin = 0.001;
                            gray.Direction = gi_spatial_dir;
                            gray.TMax = d_new - 0.001;
                            TraceRay(tlas, gi_spatial_vis, 0xFFu, 0, 0, 0, gray, prd);
                        } else {
                            prd.dist = -1.0;
                        }
                        if (prd.dist >= 0.0) continue;

                        float p_hat_neighbor = gi_target_pdf(hitPos, hit_normal, hit_albedo, metallic, neighbor_r.sample_pos, neighbor_r.sample_radiance);
                        merge_reservoirs_gi(combined, neighbor_r, p_hat_neighbor, jacobian, rnd(rng));
                    }

                    float p_hat_final = gi_target_pdf(hitPos, hit_normal, hit_albedo, metallic, combined.sample_pos, combined.sample_radiance);
                    combined.W = (p_hat_final > 1e-3) ? (combined.w_sum / max(combined.M, 1.0) / p_hat_final) : 0.0;
                    combined.W = min(combined.W, 20.0);

                    if (combined.W > 0.0) {
                        float3 gi_x2_dir = combined.sample_pos - hitPos;
                        float gi_x2_dist = max(length(gi_x2_dir), 0.0001);
                        gi_x2_dir /= gi_x2_dist;

                        float gi_NdotL = max(dot(hit_normal, gi_x2_dir), 0.0);
                        if (gi_NdotL > 0.0) {
                            uint gi_final_vis = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
                            // Only trace when the segment is longer than 2*epsilon;
                            // otherwise TMax (dist-eps) < TMin (eps) builds a degenerate
                            // ray (VUID-RuntimeSpirv-OpTraceRayKHR-06357). Near-coincident
                            // endpoints can't be occluded, so treat as visible.
                            if (gi_x2_dist > 0.002) {
                                prd.dist = 1.0;
                                RayDesc gray2;
                                gray2.Origin = hitPos;
                                gray2.TMin = 0.001;
                                gray2.Direction = gi_x2_dir;
                                gray2.TMax = gi_x2_dist - 0.001;
                                TraceRay(tlas, gi_final_vis, 0xFFu, 0, 0, 0, gray2, prd);
                            } else {
                                prd.dist = -1.0;
                            }

                            if (prd.dist < 0.0) {
                                float3 gi_f_diffuse = hit_albedo * (1.0 - metallic) / 3.14159;
                                radiance += combined.sample_radiance * gi_f_diffuse * gi_NdotL * combined.W * throughput;
                            }
                        }
                    }
                    break; // GI replaces the rest of the random walk at first rough hit.
                } else if (restir_evaluated && roughness > 0.2) {
                    // Standard NEE on subsequent rough bounces.
                    uint light_idx = min(uint(rnd(rng) * num_lights), num_lights - 1);
                    EmissiveIndirectionEntry entry = emissive_indirection[light_idx];
                    EmissiveTriangle light = emissive_triangles[entry.blas_tri_index];
                    EntityTransform xform = entity_transforms[entry.entity_id];

                    float3 wv0 = transform_point(xform, light.v0.xyz);
                    float3 wv1 = transform_point(xform, light.v1.xyz);
                    float3 wv2 = transform_point(xform, light.v2.xyz);

                    float3 edge1 = wv1 - wv0;
                    float3 edge2 = wv2 - wv0;
                    float light_area = 0.5 * length(cross(edge1, edge2));

                    float r1_nee = rnd(rng);
                    float r2_nee = rnd(rng);
                    float sqr1 = sqrt(r1_nee);
                    float u = 1.0 - sqr1;
                    float v = r2_nee * sqr1;
                    float w = 1.0 - u - v;

                    float3 light_pos    = wv0 * u + wv1 * v + wv2 * w;
                    float3 light_normal = normalize(cross(wv1 - wv0, wv2 - wv0));

                    float3 shadow_ray_dir = light_pos - hitPos;
                    float light_dist = length(shadow_ray_dir);
                    shadow_ray_dir /= light_dist;

                    float cos_theta_light = max(dot(light_normal, -shadow_ray_dir), 0.0);
                    float cos_theta_surface = max(dot(hit_normal, shadow_ray_dir), 0.0);

                    if (cos_theta_light > 0.0 && cos_theta_surface > 0.0) {
                        uint shadow_ray_flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
                        // Skip degenerate short segments (TMax<TMin); coincident endpoints are visible.
                        if (light_dist > 0.002) {
                            prd.dist = 1.0;
                            RayDesc sray;
                            sray.Origin = hitPos;
                            sray.TMin = 0.001;
                            sray.Direction = shadow_ray_dir;
                            sray.TMax = light_dist - 0.001;
                            TraceRay(tlas, shadow_ray_flags, 0xFFu, 0, 0, 0, sray, prd);
                        } else {
                            prd.dist = -1.0;
                        }

                        if (prd.dist < 0.0) {
                            float solid_angle_pdf = (light_dist * light_dist) / max(cos_theta_light * light_area * float(num_lights), 1e-4);
                            float3 nee_contrib = (light.emission.rgb * hit_albedo * throughput * cos_theta_surface) / (solid_angle_pdf * 3.14159);
                            radiance += min(nee_contrib, float3(5.0));
                        }
                        prev_did_nee = true;
                    }
                }
            }

            // BRDF bounce.
            float3 N = hit_normal;
            float3 F0 = lerp(float3(0.04), hit_albedo, metallic);
            float cos_theta = max(dot(N, V_view), 0.0);
            float3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
            float p_specular = clamp(max(F.r, max(F.g, F.b)), 0.05, 1.0);

            float r1, r2;
            if (bounce == 0) {
                r1 = frac(bn_1 + float(pc.frame_count % 1024u) * 0.75487766);
                r2 = frac(bn_2 + float(pc.frame_count % 1024u) * 0.56984029);
            } else {
                r1 = rnd(rng);
                r2 = rnd(rng);
            }

            if (rnd(rng) < p_specular) {
                float3 H = sample_ggx_vndf(N, V_view, roughness, r1, r2);
                rayDir = reflect(-V_view, H);

                if (dot(N, rayDir) <= 0.0) {
                    rayDir = get_random_bounce(N, r1, r2);
                    throughput *= hit_albedo * (1.0 - metallic) * (1.0 - F) / (1.0 - p_specular);
                } else {
                    float NdotL_b = max(dot(N, rayDir), 0.001);
                    float alpha_b = roughness * roughness;
                    float G1_L = smith_g1_ggx(NdotL_b, alpha_b);
                    throughput *= (F * G1_L) / p_specular;
                }
            } else {
                rayDir = get_random_bounce(N, r1, r2);
                throughput *= hit_albedo * (1.0 - metallic) * (1.0 - F) / (1.0 - p_specular);
            }

            float p = max(throughput.r, max(throughput.g, throughput.b));
            if (p < 0.001) break;

            if (bounce > 2) {
                if (rnd(rng) > p) break;
                throughput /= p;
            }

            rayOrigin = hitPos + hit_normal * 0.001;
        }

        total_radiance += radiance;
        total_radiance = min(total_radiance, 10.0);
    }

    float3 current_frame_color = total_radiance / float(SAMPLES);
    raw_color[pixel_coord] = float4(current_frame_color, 1.0);
}
