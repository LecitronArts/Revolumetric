struct EguiPushConstants {
    float2 screen_size_points;
    float2 _pad0;
};

[[vk::push_constant]]
EguiPushConstants egui_push;

struct EguiVertexInput {
    float2 pos : POSITION;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

struct EguiVertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

[shader("vertex")]
EguiVertexOutput main(EguiVertexInput input) {
    EguiVertexOutput output;
    float2 ndc = float2(
        input.pos.x / egui_push.screen_size_points.x * 2.0 - 1.0,
        input.pos.y / egui_push.screen_size_points.y * 2.0 - 1.0
    );
    output.position = float4(ndc, 0.0, 1.0);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}
