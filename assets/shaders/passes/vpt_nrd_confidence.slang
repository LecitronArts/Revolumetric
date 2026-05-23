// VPT NRD confidence pass.
// Emits conservative confidence inputs for later NRD dispatches.

#include "scene_common.slang"
#include "vpt_motion_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> motion_history;

[[vk::binding(2, 0)]]
[[vk::image_format("r32ui")]]
RWTexture2D<uint> motion_flags;

[[vk::binding(3, 0)]]
[[vk::image_format("r32ui")]]
RWTexture2D<uint> surface_brick_generation;

[[vk::binding(4, 0)]]
[[vk::image_format("r32ui")]]
RWTexture2D<uint> previous_surface_brick_generation;

[[vk::binding(5, 0)]]
[[vk::image_format("r16f")]]
RWTexture2D<float> diff_confidence;

[[vk::binding(6, 0)]]
[[vk::image_format("r16f")]]
RWTexture2D<float> spec_confidence;

float vpt_nrd_diffuse_history_confidence(uint2 pixel, SceneUniforms scene) {
    uint flags = motion_flags[pixel];
    if (vpt_motion_flags_reject_history(flags)) {
        return 0.0;
    }

    float2 previous_pixel_f = motion_history[pixel].xy + float2(pixel) + 0.5;
    int2 previous_pixel = int2(floor(previous_pixel_f));
    if (previous_pixel.x < 0 || previous_pixel.y < 0 || previous_pixel.x >= int(scene.resolution.x) || previous_pixel.y >= int(scene.resolution.y)) {
        return 0.0;
    }

    uint previous_generation = previous_surface_brick_generation[uint2(previous_pixel)];
    uint current_generation = surface_brick_generation[pixel];
    if (vpt_surface_generation_rejects_history(previous_generation, current_generation)) {
        return 0.0;
    }

    return 1.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    diff_confidence[tid.xy] = vpt_nrd_diffuse_history_confidence(tid.xy, scene);
    spec_confidence[tid.xy] = 0.0;
}
