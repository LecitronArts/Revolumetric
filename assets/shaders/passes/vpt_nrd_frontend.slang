// VPT NRD frontend packing pass.
// Converts raw VPT NRD noisy resources into the local ReLAX input contract.

#include "scene_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> input_diff_radiance_hitdist;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> input_spec_radiance_hitdist;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> input_residual_radiance;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> input_material_factors;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> packed_diff_radiance_hitdist;

[[vk::binding(6, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> packed_spec_radiance_hitdist;

[[vk::binding(7, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> residual_radiance;

[[vk::binding(8, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> material_factors;

static const float VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE = 65504.0;

float sanitize_relax_scalar(float value) {
    if (value != value) {
        return 0.0;
    }
    return min(max(value, 0.0), VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE);
}

float3 sanitize_relax_radiance(float3 radiance) {
    return float3(
        sanitize_relax_scalar(radiance.x),
        sanitize_relax_scalar(radiance.y),
        sanitize_relax_scalar(radiance.z)
    );
}

float sanitize_relax_hit_distance(float hit_distance) {
    if (hit_distance != hit_distance || hit_distance < 0.0) {
        return VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE;
    }
    return min(hit_distance, VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE);
}

float4 pack_relax_radiance_hitdist(float4 radiance_hitdist) {
    return float4(sanitize_relax_radiance(radiance_hitdist.rgb), sanitize_relax_hit_distance(radiance_hitdist.a));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    packed_diff_radiance_hitdist[tid.xy] = pack_relax_radiance_hitdist(input_diff_radiance_hitdist[tid.xy]);
    packed_spec_radiance_hitdist[tid.xy] = pack_relax_radiance_hitdist(input_spec_radiance_hitdist[tid.xy]);
    residual_radiance[tid.xy] = float4(sanitize_relax_radiance(input_residual_radiance[tid.xy].rgb), 1.0);
    material_factors[tid.xy] = float4(saturate(input_material_factors[tid.xy].rgb), saturate(input_material_factors[tid.xy].a));
}
