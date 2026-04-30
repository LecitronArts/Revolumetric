// Test pattern compute shader.
// Writes a gradient pattern to a storage image.
// Proves the pipeline: Slang compile → SPIR-V → compute dispatch → image write.

struct PushConstants {
    float time;
    uint width;
    uint height;
    uint _pad;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]]
[[vk::image_format("rgba8")]]
RWTexture2D<float4> output_image;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;

    float2 uv = float2(tid.xy) / float2(pc.width, pc.height);

    float r = 0.5 + 0.5 * sin(uv.x * 6.28 + pc.time);
    float g = 0.5 + 0.5 * sin(uv.y * 6.28 + pc.time * 1.3);
    float b = 0.5 + 0.5 * sin((uv.x + uv.y) * 3.14 + pc.time * 0.7);

    output_image[tid.xy] = float4(r, g, b, 1.0);
}
