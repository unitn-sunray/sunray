// Heap-mode (VK_EXT_descriptor_heap) port of temporal_accumulation.glsl.
//
// All four images are bound as STORAGE (`RWTexture2D`, GENERAL layout) through
// `DescriptorHandle<>` slots in the push constant — no sampler / sampled view is
// used. The history reprojection (originally a bilinear `texture()` fetch) is
// done with a manual bilinear filter over the storage image. This keeps the
// ping-pong accumulation images in GENERAL the whole time, which sidesteps the
// cross-frame layout tracking the render graph can't yet do for imported images.
//
// The CPU resolves the ping-pong (history vs. accumulation target) and passes
// the already-selected slots, so the shader doesn't index by frame parity.

#define TILE_SIZE 16
#define TILE_BORDER 1
#define TILE_FULL (TILE_SIZE + 2 * TILE_BORDER) // 18

import rt_types;  // RWImage_* format aliases (see the comment on them)

struct TemporalPC {
    DescriptorHandle<RWImage_r11f_g11f_b10f> raw_rt_color; // RT raw color   (storage read)
    DescriptorHandle<RWImage_rg16f> motion_vector;         // rg motion      (storage read)
    DescriptorHandle<RWImage_r11f_g11f_b10f> history;      // accum[history] (storage read, manual bilinear)
    DescriptorHandle<RWImage_r11f_g11f_b10f> accum_output; // accum[target]  (storage write)
    uint frame_count;
    uint width;
    uint height;
}

[vk::push_constant]
ConstantBuffer<TemporalPC> pc;

static const float ACCUMULATION_FACTOR = 0.14;

// Shared tile for the 3x3 neighborhood clamp. 256 threads cooperatively load 324
// texels (~1.27/thread), matching the GLSL version.
groupshared float3 tile_color[TILE_FULL * TILE_FULL];

float get_luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// Bilinear sample of a storage image at a continuous uv (clamp-to-edge), used to
// reproject the history buffer. Equivalent to a LINEAR/CLAMP sampler.
float3 sample_history_bilinear(RWImage_r11f_g11f_b10f history, float2 uv, int2 size) {
    float2 p = uv * float2(size) - 0.5;
    int2 base = int2(floor(p));
    float2 f = p - float2(base);

    int2 c00 = clamp(base + int2(0, 0), int2(0, 0), size - 1);
    int2 c10 = clamp(base + int2(1, 0), int2(0, 0), size - 1);
    int2 c01 = clamp(base + int2(0, 1), int2(0, 0), size - 1);
    int2 c11 = clamp(base + int2(1, 1), int2(0, 0), size - 1);

    float3 h00 = history[c00].rgb;
    float3 h10 = history[c10].rgb;
    float3 h01 = history[c01].rgb;
    float3 h11 = history[c11].rgb;

    return lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
}

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(
    uint3 group_id: SV_GroupID,
    uint3 group_thread_id: SV_GroupThreadID,
    uint group_index: SV_GroupIndex,
    uint3 dispatch_thread_id: SV_DispatchThreadID
) {
    RWImage_r11f_g11f_b10f raw_rt_color = pc.raw_rt_color;
    RWImage_rg16f          motion_vector = pc.motion_vector;
    RWImage_r11f_g11f_b10f history = pc.history;
    RWImage_r11f_g11f_b10f accum_output = pc.accum_output;

    // imageSize() is unreliable on this driver, so the dispatch extent comes via
    // the push constant (matches the GLSL note).
    int2 size = int2(pc.width, pc.height);
    int2 pixel_coords = int2(dispatch_thread_id.xy);
    int2 local_coords = int2(group_thread_id.xy);
    int2 tile_base = int2(group_id.xy) * TILE_SIZE - TILE_BORDER;

    // Cooperative tile load: all 256 threads must participate (do NOT early-exit
    // before the barrier).
    for (uint i = group_index; i < uint(TILE_FULL * TILE_FULL); i += uint(TILE_SIZE * TILE_SIZE)) {
        int2 local_pos = int2(int(i % TILE_FULL), int(i / TILE_FULL));
        int2 sample_pos = clamp(tile_base + local_pos, int2(0, 0), size - 1);
        tile_color[i] = raw_rt_color[sample_pos].rgb;
    }
    GroupMemoryBarrierWithGroupSync();

    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) {
        return;
    }

    int2 tile_center = local_coords + int2(TILE_BORDER, TILE_BORDER);
    float3 current_color = tile_color[tile_center.y * TILE_FULL + tile_center.x];

    float3 min_color = current_color;
    float3 max_color = current_color;
    float center_luma = get_luminance(current_color);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) {
                continue;
            }
            int2 tile_pos = tile_center + int2(x, y);
            float3 neighbor_color = tile_color[tile_pos.y * TILE_FULL + tile_pos.x];

            float neighbor_luma = get_luminance(neighbor_color);
            float luma_threshold = max(center_luma * 5.0, 0.08);
            if (abs(neighbor_luma - center_luma) < luma_threshold) {
                min_color = min(min_color, neighbor_color);
                max_color = max(max_color, neighbor_color);
            }
        }
    }

    float2 uv = (float2(pixel_coords) + 0.5) / float2(size);
    float2 motion = motion_vector[pixel_coords];
    float2 prev_uv = uv - motion;

    float3 accumulated_color = current_color;

    bool is_off_screen = any(prev_uv < float2(0.0)) || any(prev_uv > float2(1.0));

    if (!is_off_screen && pc.frame_count > 2) {
        float3 history_color = sample_history_bilinear(history, prev_uv, size);
        float3 clamped_history = clamp(history_color, min_color, max_color);
        accumulated_color = lerp(clamped_history, current_color, ACCUMULATION_FACTOR);
    }

    accum_output[pixel_coords] = float4(accumulated_color, 1.0);
}
