// Heap-mode (VK_EXT_descriptor_heap) port of denoise.glsl.
// Reads temporal_result (storage), depth/normal/diffuse (sampled), writes spatial_output (storage).
// All bindings come through `DescriptorHandle<>` slot indices in the push constant.

import rt_types;  // RWImage_* format aliases (see the comment on them)

struct DenoisePC {
    DescriptorHandle<RWImage_r11f_g11f_b10f> temporal_result;
    DescriptorHandle<Texture2D<float4>> depth;
    DescriptorHandle<Texture2D<float4>> normal;
    DescriptorHandle<Texture2D<float4>> diffuse;
    DescriptorHandle<RWImage_r11f_g11f_b10f> spatial_output;
    uint frame_count;
    int step_width;
    uint width;
    uint height;
}

[vk::push_constant]
ConstantBuffer<DenoisePC> pc;

float get_luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// Fixed B-Spline weights for 5x5 A-Trous.
static const float kernel[5] = { 1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0 };

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    RWImage_r11f_g11f_b10f temporal_result = pc.temporal_result;
    Texture2D<float4> depth = pc.depth;
    Texture2D<float4> normal = pc.normal;
    Texture2D<float4> diffuse = pc.diffuse;
    RWImage_r11f_g11f_b10f spatial_output = pc.spatial_output;

    int2 size = int2(pc.width, pc.height);
    int2 pixel_coords = int2(tid.xy);
    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) {
        return;
    }

    float3 center_color = temporal_result[pixel_coords].rgb;
    float center_depth = depth.Load(int3(pixel_coords, 0)).r;

    // Sky bypass: depth sentinel is huge for misses.
    if (center_depth >= 10000.0) {
        spatial_output[pixel_coords] = float4(center_color, 1.0);
        return;
    }

    float4 normal_data = normal.Load(int3(pixel_coords, 0));
    float3 center_normal = normal_data.rgb;
    float center_roughness = normal_data.a;
    float3 center_diffuse = diffuse.Load(int3(pixel_coords, 0)).rgb;

    if (center_roughness < 0.1) {
        spatial_output[pixel_coords] = float4(center_color, 1.0);
        return;
    }

    // Demodulate: the path tracer bakes albedo into the radiance, so we divide it
    // out and filter the (smooth) illumination, then remodulate with albedo at the
    // end. The `diffuse` G-buffer still drives edge stopping below.
    float3 center_illum = center_color / max(center_diffuse, float3(0.001));

    float center_weight = kernel[2] * kernel[2];
    float3 sum_color = center_illum * center_weight;
    float sum_weight = center_weight;

    const float DEPTH_SENSITIVITY = 8.0;
    const float NORMAL_SENSITIVITY = 80.0;
    const float DIFFUSE_SENSITIVITY = 50.0;

    float center_luma = get_luminance(center_illum);

    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            int2 sample_offset = int2(x, y) * pc.step_width;
            int2 sample_coord = pixel_coords + sample_offset;

            if (sample_coord.x < 0 || sample_coord.y < 0 ||
                sample_coord.x >= size.x || sample_coord.y >= size.y) {
                continue;
            }

            float3 sample_color = temporal_result[sample_coord].rgb;
            float sample_depth = depth.Load(int3(sample_coord, 0)).r;
            float3 sample_normal = normal.Load(int3(sample_coord, 0)).rgb;
            float3 sample_diffuse = diffuse.Load(int3(sample_coord, 0)).rgb;

            // Demodulate the neighbor into illumination space to match the center.
            float3 sample_illum = sample_color / max(sample_diffuse, float3(0.001));

            float sample_luma = get_luminance(sample_illum);

            float diffuse_diff = distance(center_diffuse, sample_diffuse);
            float luma_diff = abs(center_luma - sample_luma);
            float luma_sigma = max(center_luma, sample_luma) * 0.4 + 0.01;
            float luma_ratio = luma_diff / luma_sigma;

            float combined_power =
                -abs(center_depth - sample_depth) * DEPTH_SENSITIVITY
                + (dot(center_normal, sample_normal) - 1.0) * NORMAL_SENSITIVITY
                - diffuse_diff * DIFFUSE_SENSITIVITY
                - luma_ratio * luma_ratio;

            float weight = exp(combined_power) * kernel[x + 2] * kernel[y + 2];

            sum_color += sample_illum * weight;
            sum_weight += weight;
        }
    }

    float3 spatially_denoised_color = (sum_color / max(sum_weight, 0.0001)) * center_diffuse;
    spatial_output[pixel_coords] = float4(spatially_denoised_color, 1.0);
}
