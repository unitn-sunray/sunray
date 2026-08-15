// egui paint shader (heap mode), used by the Bevy integration's egui overlay.
//
// Vertex: transforms egui vertices (in logical "points") into clip space.
// Fragment: samples the font/user texture through a descriptor-heap handle and
// applies egui's premultiplied-alpha gamma math.
//
// Target is the sRGB swapchain image, so the hardware re-encodes linear->sRGB on
// store; we therefore convert the gamma-space result to linear before output.
// NOTE: gamma/premultiply handling is the classic egui tuning point — if colors
// look washed out or too dark, this `linear_from_srgb` step is the first knob.

// Heap slots are bare `uint`, not `DescriptorHandle<T>` (which lowers to a
// `uint2` whose high word is never read). Accessors below name the type once.
struct EguiPC {
    float2 screen_size_points;   // logical window size (points)
    uint tex;                    // font / user texture (gamma, premultiplied)
    uint samp;

    Texture2D<float4> get_tex()  { DescriptorHandle<Texture2D<float4>> h = { tex };  return h; }
    SamplerState      get_samp() { DescriptorHandle<SamplerState>      h = { samp }; return h; }
};

[vk::push_constant]
ConstantBuffer<EguiPC> pc;

struct VSIn {
    [[vk::location(0)]] float2 pos : POSITION;   // egui points
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float4 color : COLOR0;   // sRGB, unmultiplied, normalized 0..1
};

struct VSOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

[shader("vertex")]
VSOut vertex_main(VSIn input) {
    VSOut o;
    // points -> [0,1] -> NDC. Vulkan NDC y is down and egui y is down, so no flip.
    float2 ndc = input.pos / pc.screen_size_points * 2.0 - 1.0;
    o.position = float4(ndc, 0.0, 1.0);
    o.uv = input.uv;
    o.color = input.color;
    return o;
}

float3 linear_from_srgb(float3 srgb) {
    // gamma 2.2 approximation; swap for the piecewise sRGB curve if needed.
    return pow(max(srgb, 0.0), float3(2.2));
}

[shader("fragment")]
float4 fragment_main(VSOut input) : SV_Target {
    Texture2D<float4> tex = pc.get_tex();
    SamplerState samp = pc.get_samp();
    float4 tex_gamma = tex.Sample(samp, input.uv); // premultiplied, gamma space
    float4 out_gamma = input.color * tex_gamma;    // egui blend math (gamma space)
    return float4(linear_from_srgb(out_gamma.rgb), out_gamma.a);
}
