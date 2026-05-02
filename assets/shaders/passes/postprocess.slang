// Postprocess compute shader.
// Reads linear HDR lighting, applies exposure, ACES tonemap, and gamma correction.

#include "lighting_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> hdr_image;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba8")]]
RWTexture2D<float4> output_image;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    float3 hdr = max(hdr_image[tid.xy].rgb * scene.exposure, float3(0.0));
    float3 mapped = aces_tonemap(hdr);
    float3 ldr = pow(mapped, float3(1.0 / 2.2));

    output_image[tid.xy] = float4(ldr, 1.0);
}
