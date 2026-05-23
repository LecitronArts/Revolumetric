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
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_material_roughness;

[[vk::binding(7, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> filtered_radiance_image;

[[vk::binding(8, 0)]]
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

float roughness_weight(float center_roughness, float neighbor_roughness) {
    return exp(-abs(center_roughness - neighbor_roughness) * 6.0);
}

float albedo_weight(float3 center_albedo, float3 neighbor_albedo) {
    float3 albedo_delta = abs(center_albedo - neighbor_albedo);
    float max_delta = max(albedo_delta.x, max(albedo_delta.y, albedo_delta.z));
    return exp(-max_delta * 8.0);
}

float surface_guide_weight(
    float4 center_position_depth,
    float4 neighbor_position_depth,
    float4 center_normal_roughness,
    float4 neighbor_normal_roughness,
    float4 center_albedo_material,
    float4 neighbor_albedo_material,
    float4 center_material_roughness,
    float4 neighbor_material_roughness
) {
    return normal_weight(center_normal_roughness.xyz, neighbor_normal_roughness.xyz)
        * depth_weight(center_position_depth, neighbor_position_depth)
        * albedo_weight(center_albedo_material.rgb, neighbor_albedo_material.rgb)
        * material_weight(center_albedo_material.w, neighbor_albedo_material.w)
        * roughness_weight(center_material_roughness.x, neighbor_material_roughness.x);
}

struct AtrousLuminanceStats {
    float mean_luma;
    float trimmed_mean_luma;
    float variance;
    float trimmed_variance;
    float sample_count;
};

AtrousLuminanceStats gather_atrous_luminance_stats(
    uint2 pixel,
    SceneUniforms scene,
    float4 center_position_depth,
    float4 center_normal_roughness,
    float4 center_albedo_material,
    float4 center_material_roughness
) {
    float weighted_luma = 0.0;
    float weighted_luma_sq = 0.0;
    float total_weight = 0.0;
    float max_luma = -1.0;
    float max_luma_weight = 0.0;
    AtrousLuminanceStats stats = (AtrousLuminanceStats)0;

    [unroll]
    for (uint y = 0u; y < 5u; y++) {
        [unroll]
        for (uint x = 0u; x < 5u; x++) {
            int2 tap = int2(atrous_offsets[x].x, atrous_offsets[y].x);
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
            float4 neighbor_material_roughness = surface_material_roughness[neighbor_pixel];
            float guide_weight = surface_guide_weight(
                center_position_depth,
                neighbor_position_depth,
                center_normal_roughness,
                neighbor_normal_roughness,
                center_albedo_material,
                neighbor_albedo_material,
                center_material_roughness,
                neighbor_material_roughness
            );
            float tap_weight = atrous_kernel[x] * atrous_kernel[y] * guide_weight;
            if (tap_weight <= 1.0e-5) {
                continue;
            }
            float tap_luma = luminance_of(input_radiance_image[neighbor_pixel].rgb);

            weighted_luma += tap_luma * tap_weight;
            weighted_luma_sq += tap_luma * tap_luma * tap_weight;
            total_weight += tap_weight;
            stats.sample_count += 1.0;
            if (tap_luma > max_luma) {
                max_luma = tap_luma;
                max_luma_weight = tap_weight;
            }
        }
    }

    if (total_weight <= 1.0e-5) {
        return stats;
    }
    float mean_luma = weighted_luma / total_weight;
    stats.mean_luma = mean_luma;
    stats.variance = max(weighted_luma_sq / total_weight - mean_luma * mean_luma, 0.0);
    stats.trimmed_mean_luma = mean_luma;
    stats.trimmed_variance = stats.variance;

    if (stats.sample_count >= 4.0 && total_weight > max_luma_weight + 1.0e-5) {
        float trimmed_weight = total_weight - max_luma_weight;
        float trimmed_luma = weighted_luma - max_luma * max_luma_weight;
        float trimmed_luma_sq = weighted_luma_sq - max_luma * max_luma * max_luma_weight;
        stats.trimmed_mean_luma = trimmed_luma / trimmed_weight;
        stats.trimmed_variance = max(
            trimmed_luma_sq / trimmed_weight - stats.trimmed_mean_luma * stats.trimmed_mean_luma,
            0.0
        );
    }

    return stats;
}

float spatial_luminance_variance(AtrousLuminanceStats stats) {
    return stats.trimmed_variance;
}

float atrous_luminance_limit(AtrousLuminanceStats stats) {
    if (stats.sample_count < 4.0) {
        return 1.0e20;
    }

    float sigma = sqrt(max(stats.trimmed_variance, 0.0));
    float variance_limit = stats.trimmed_mean_luma + sigma * 4.0 + 0.25;
    float mean_limit = stats.trimmed_mean_luma * 3.0 + 0.25;
    return max(max(variance_limit, mean_limit), 1.5);
}

float3 clamp_center_radiance_to_spatial_interval(
    float3 radiance,
    float4 center_normal_roughness,
    AtrousLuminanceStats stats
) {
    if (center_normal_roughness.w > 1.0e-4) {
        return radiance;
    }

    float center_luma = luminance_of(radiance);
    float luma_limit = atrous_luminance_limit(stats);
    if (center_luma <= luma_limit || center_luma <= 1.0e-4) {
        return radiance;
    }
    return radiance * (luma_limit / center_luma);
}

float effective_luminance_variance(float4 moments_history, float3 radiance, float spatial_variance) {
    float history_length = max(moments_history.z, 0.0);
    float temporal_variance = luminance_variance(moments_history.xy);
    float luma = luminance_of(radiance);
    float short_history_sigma = max(0.08, luma * 0.12);
    float short_history_variance = max(spatial_variance, short_history_sigma * short_history_sigma);
    float history_confidence = saturate((history_length - 1.0) / 3.0);
    return lerp(short_history_variance, temporal_variance, history_confidence);
}

float3 clamp_radiance_to_luminance_interval(
    float3 radiance,
    float3 center_radiance,
    float luminance_variance
) {
    float center_luma = luminance_of(center_radiance);
    float tap_luma = luminance_of(radiance);
    float sigma = sqrt(max(luminance_variance, 0.0));
    float upper_luma = center_luma + max(0.10, sigma * 2.5 + center_luma * 0.10);
    if (tap_luma <= upper_luma || tap_luma <= 1.0e-4) {
        return radiance;
    }
    return radiance * (upper_luma / tap_luma);
}

float variance_weight(float3 center_radiance, float3 neighbor_radiance, float luminance_variance) {
    float center_luma = luminance_of(center_radiance);
    float neighbor_luma = luminance_of(neighbor_radiance);
    float sigma = sqrt(max(luminance_variance, 0.0));
    return exp(-abs(neighbor_luma - center_luma) / max(0.05, sigma * 4.0 + center_luma * 0.02));
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
    float4 center_material_roughness = surface_material_roughness[pixel];
    float4 center_moments_history = moments_history_image[pixel];

    if (atrous_passthrough(scene) || center_position_depth.w < 0.0 || center_moments_history.w <= 0.0) {
        filtered_radiance_image[pixel] = center_radiance;
        return;
    }

    bool short_history = center_moments_history.z < 4.0;
    bool gather_spatial_stats = short_history && vpt_atrous.iteration_index == 0u;
    AtrousLuminanceStats center_spatial_stats = gather_spatial_stats
        ? gather_atrous_luminance_stats(
            pixel,
            scene,
            center_position_depth,
            center_normal_roughness,
            center_albedo_material,
            center_material_roughness
        )
        : (AtrousLuminanceStats)0;
    float3 center_filtered_radiance = gather_spatial_stats
        ? clamp_center_radiance_to_spatial_interval(
            center_radiance.rgb,
            center_normal_roughness,
            center_spatial_stats
        )
        : center_radiance.rgb;
    center_radiance = float4(center_filtered_radiance, center_radiance.a);
    float center_spatial_variance = gather_spatial_stats
        ? spatial_luminance_variance(center_spatial_stats)
        : 0.0;
    float center_variance = effective_luminance_variance(
        center_moments_history,
        center_radiance.rgb,
        center_spatial_variance
    );
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
            float4 neighbor_material_roughness = surface_material_roughness[neighbor_pixel];
            float4 neighbor_moments_history = moments_history_image[neighbor_pixel];
            if (neighbor_moments_history.w <= 0.0) {
                continue;
            }

            float4 neighbor_radiance = input_radiance_image[neighbor_pixel];
            float neighbor_variance = effective_luminance_variance(
                neighbor_moments_history,
                neighbor_radiance.rgb,
                center_spatial_variance
            );
            float combined_variance = max(center_variance, neighbor_variance);
            float3 clamped_neighbor_radiance = clamp_radiance_to_luminance_interval(
                neighbor_radiance.rgb,
                center_radiance.rgb,
                combined_variance
            );
            float kernel_weight = atrous_kernel[x] * atrous_kernel[y];
            float guide_weight =
                surface_guide_weight(
                    center_position_depth,
                    neighbor_position_depth,
                    center_normal_roughness,
                    neighbor_normal_roughness,
                    center_albedo_material,
                    neighbor_albedo_material,
                    center_material_roughness,
                    neighbor_material_roughness
                )
                * variance_weight(center_radiance.rgb, clamped_neighbor_radiance, combined_variance);
            float tap_weight = kernel_weight * guide_weight;

            weighted_radiance += clamped_neighbor_radiance * tap_weight;
            total_weight += tap_weight;
        }
    }

    float3 filtered = total_weight > 1.0e-5
        ? weighted_radiance / total_weight
        : center_radiance.rgb;
    filtered_radiance_image[pixel] = float4(max(filtered, float3(0.0)), center_radiance.a);
}
