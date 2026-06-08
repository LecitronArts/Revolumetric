struct EguiFragmentInput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

[[vk::binding(0, 0)]]
Texture2D<float4> egui_texture;

[[vk::binding(1, 0)]]
SamplerState egui_sampler;

[shader("fragment")]
float4 main(EguiFragmentInput input) : SV_Target0 {
    return egui_texture.Sample(egui_sampler, input.uv) * input.color;
}
