#include "scene_common.slang"

struct VptAtrousUniforms {
    uint iteration_index;
    uint atrous_step_width;
    uint pass_count;
    uint _pad0;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> input_radiance_image;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> moments_history_image;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_position_depth;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_normal_roughness;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_albedo_material;

[[vk::binding(6, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> filtered_radiance_image;

[[vk::binding(7, 0)]]
ConstantBuffer<VptAtrousUniforms> vpt_atrous;

static const int2 atrous_offsets[5] = {
    int2(-2, 0),
    int2(-1, 0),
    int2(0, 0),
    int2(1, 0),
    int2(2, 0),
};

static const float atrous_kernel[5] = {
    0.0625,
    0.25,
    0.375,
    0.25,
    0.0625,
};

float luminance_of(float3 radiance) {
    return max(dot(radiance, float3(0.2126, 0.7152, 0.0722)), 0.0);
}

bool pixel_inside(int2 pixel, uint2 resolution) {
    return pixel.x >= 0
        && pixel.y >= 0
        && pixel.x < int(resolution.x)
        && pixel.y < int(resolution.y);
}

bool atrous_passthrough(SceneUniforms scene) {
    return (scene.denoiser_flags & DENOISER_FLAG_ENABLED) == 0u
        || scene.denoiser_atrous_iterations == 0u
        || scene.vpt_debug_view != VPT_DEBUG_VIEW_FINAL
        || vpt_atrous.iteration_index >= scene.denoiser_atrous_iterations;
}

float luminance_variance(float2 moments) {
    float mean_luma = max(moments.x, 0.0);
    return max(moments.y - mean_luma * mean_luma, 0.0);
}

float3 safe_normalize(float3 value) {
    float len2 = dot(value, value);
    return len2 > 1.0e-8 ? value * rsqrt(len2) : float3(0.0, 0.0, 1.0);
}

float normal_weight(float3 center_normal, float3 neighbor_normal) {
    float normal_similarity = saturate(dot(safe_normalize(center_normal), safe_normalize(neighbor_normal)));
    return pow(normal_similarity, 32.0);
}

float depth_weight(float4 center_position_depth, float4 neighbor_position_depth) {
    float depth_scale = max(center_position_depth.w, 1.0);
    float position_delta = distance(center_position_depth.xyz, neighbor_position_depth.xyz);
    return exp(-position_delta / max(0.75, depth_scale * 0.015));
}

float material_weight(float center_material, float neighbor_material) {
    return abs(center_material - neighbor_material) <= 0.5 ? 1.0 : 0.0;
}

float albedo_weight(float3 center_albedo, float3 neighbor_albedo) {
    float3 albedo_delta = abs(center_albedo - neighbor_albedo);
    float max_delta = max(albedo_delta.x, max(albedo_delta.y, albedo_delta.z));
    return exp(-max_delta * 8.0);
}

float variance_weight(float3 center_radiance, float3 neighbor_radiance, float center_variance) {
    float center_luma = luminance_of(center_radiance);
    float neighbor_luma = luminance_of(neighbor_radiance);
    float sigma = sqrt(max(center_variance, 0.0));
    return exp(-abs(neighbor_luma - center_luma) / max(0.05, sigma + center_luma * 0.02));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    uint2 pixel = tid.xy;
    float4 center_radiance = input_radiance_image[pixel];
    float4 center_position_depth = surface_position_depth[pixel];
    float4 center_normal_roughness = surface_normal_roughness[pixel];
    float4 center_albedo_material = surface_albedo_material[pixel];
    float4 center_moments_history = moments_history_image[pixel];

    if (atrous_passthrough(scene) || center_position_depth.w < 0.0 || center_moments_history.w <= 0.0) {
        filtered_radiance_image[pixel] = center_radiance;
        return;
    }

    float center_variance = luminance_variance(center_moments_history.xy);
    int atrous_step_width = int(vpt_atrous.atrous_step_width);
    float3 weighted_radiance = float3(0.0);
    float total_weight = 0.0;

    [unroll]
    for (uint y = 0u; y < 5u; y++) {
        [unroll]
        for (uint x = 0u; x < 5u; x++) {
            int2 tap = int2(atrous_offsets[x].x, atrous_offsets[y].x) * atrous_step_width;
            int2 neighbor_i = int2(pixel) + tap;
            if (!pixel_inside(neighbor_i, scene.resolution)) {
                continue;
            }

            uint2 neighbor_pixel = uint2(neighbor_i);
            float4 neighbor_position_depth = surface_position_depth[neighbor_pixel];
            if (neighbor_position_depth.w < 0.0) {
                continue;
            }

            float4 neighbor_normal_roughness = surface_normal_roughness[neighbor_pixel];
            float4 neighbor_albedo_material = surface_albedo_material[neighbor_pixel];
            float4 neighbor_radiance = input_radiance_image[neighbor_pixel];
            float kernel_weight = atrous_kernel[x] * atrous_kernel[y];
            float guide_weight =
                normal_weight(center_normal_roughness.xyz, neighbor_normal_roughness.xyz)
                * depth_weight(center_position_depth, neighbor_position_depth)
                * albedo_weight(center_albedo_material.rgb, neighbor_albedo_material.rgb)
                * material_weight(center_albedo_material.w, neighbor_albedo_material.w)
                * variance_weight(center_radiance.rgb, neighbor_radiance.rgb, center_variance);
            float tap_weight = kernel_weight * guide_weight;

            weighted_radiance += neighbor_radiance.rgb * tap_weight;
            total_weight += tap_weight;
        }
    }

    float3 filtered = total_weight > 1.0e-5
        ? weighted_radiance / total_weight
        : center_radiance.rgb;
    filtered_radiance_image[pixel] = float4(max(filtered, float3(0.0)), center_radiance.a);
}
