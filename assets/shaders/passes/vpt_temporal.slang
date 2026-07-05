#include "scene_common.slang"
#include "vpt_history_common.slang"
#include "vpt_motion_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> noisy_radiance_image;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> noisy_moments_image;

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
[[vk::image_format("r16f")]]
RWTexture2D<float> surface_material_roughness;

[[vk::binding(7, 0)]]
[[vk::image_format("r32f")]]
RWTexture2D<float> surface_view_z;

[[vk::binding(8, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> previous_surface_position_depth;

[[vk::binding(9, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> previous_surface_normal_roughness;

[[vk::binding(10, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> previous_surface_albedo_material;

[[vk::binding(11, 0)]]
[[vk::image_format("r16f")]]
RWTexture2D<float> previous_surface_material_roughness;

[[vk::binding(12, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> motion_history;

[[vk::binding(13, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> accumulated_radiance_image;

[[vk::binding(14, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> accumulated_moments_history_image;

[[vk::binding(15, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> previous_accumulated_radiance_image;

[[vk::binding(16, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> previous_accumulated_moments_history_image;

[[vk::binding(17, 0)]]
[[vk::image_format("r32ui")]]
RWTexture2D<uint> motion_flags;

[[vk::binding(18, 0)]]
[[vk::image_format("r32ui")]]
RWTexture2D<uint> surface_brick_generation;

[[vk::binding(19, 0)]]
[[vk::image_format("r32ui")]]
RWTexture2D<uint> previous_surface_brick_generation;

static const int2 history_tap_offsets[4] = {
    int2(0, 0),
    int2(1, 0),
    int2(0, 1),
    int2(1, 1),
};

static const int2 firefly_tap_offsets[8] = {
    int2(-1, -1),
    int2(0, -1),
    int2(1, -1),
    int2(-1, 0),
    int2(1, 0),
    int2(-1, 1),
    int2(0, 1),
    int2(1, 1),
};

static const float FIREFLY_CLAMP_MIN_LUMA = 1.5;
static const float FIREFLY_CLAMP_SIGMA_SCALE = 4.0;
static const float FIREFLY_CLAMP_MEDIAN_SCALE = 3.0;
static const float FIREFLY_CLAMP_TRIMMED_MEAN_SCALE = 2.5;

struct ReprojectedHistorySample {
    float3 radiance;
    float2 moments;
    float history_length;
    float history_reliability;
};

struct ClampedRadianceSample {
    float3 radiance;
    float2 moments;
};

struct FireflyClampStats {
    float mean_luma;
    float median_luma;
    float trimmed_mean_luma;
    float robust_sigma_luma;
    float sample_count;
};

float luminance_of(float3 radiance) {
    return max(dot(radiance, float3(0.2126, 0.7152, 0.0722)), 0.0);
}

uint vpt_history_generation_half_bits(uint generation) {
    return f32tof16(float(generation));
}

float vpt_history_generation_alpha(uint history_generation_bits) {
    return f16tof32(history_generation_bits);
}

bool debug_view_bypasses_temporal(uint debug_view) {
    return debug_view == VPT_DEBUG_VIEW_RAW
        || debug_view == VPT_DEBUG_VIEW_RESERVOIR_WEIGHT
        || debug_view == VPT_DEBUG_VIEW_DIRECT
        || debug_view == VPT_DEBUG_VIEW_INDIRECT
        || debug_view == VPT_DEBUG_VIEW_AREA_SUBPIXEL
        || debug_view == VPT_DEBUG_VIEW_AREA_LENS
        || debug_view == VPT_DEBUG_VIEW_AREA_WEIGHT
        || debug_view == VPT_DEBUG_VIEW_AREA_HISTORY_VALID
        || debug_view == VPT_DEBUG_VIEW_AREA_REJECTION
        || debug_view == VPT_DEBUG_VIEW_AREA_JACOBIAN
        || debug_view == VPT_DEBUG_VIEW_VOXEL_BRICK
        || debug_view == VPT_DEBUG_VIEW_VOXEL_LOCAL
        || debug_view == VPT_DEBUG_VIEW_VOXEL_HIT;
}

float3 visualize_luminance_variance(float2 moments) {
    float mean_luma = max(moments.x, 0.0);
    float variance = max(moments.y - mean_luma * mean_luma, 0.0);
    float heat = saturate(log2(variance + 1.0) / 8.0);
    return float3(heat, heat * heat, 1.0 - heat);
}

float3 visualize_linear_depth(float linear_depth) {
    if (linear_depth < 0.0) {
        return float3(0.0);
    }
    float depth = saturate(log2(linear_depth + 1.0) / 12.0);
    return float3(depth);
}

float3 visualize_nrd_normal_roughness(uint2 pixel, bool valid_surface) {
    if (!valid_surface) {
        return float3(0.0);
    }
    float3 normal = normalize(surface_normal_roughness[pixel].xyz) * 0.5 + 0.5;
    float roughness = saturate(surface_material_roughness[pixel]);
    return float3(normal.xy, roughness);
}

float3 visualize_nrd_motion_z(float motion_z, bool valid_motion) {
    if (!valid_motion) {
        return float3(0.0);
    }
    float encoded = clamp(motion_z * 0.05, -0.5, 0.5) + 0.5;
    return float3(encoded, motion_z >= 0.0 ? 1.0 : 0.0, motion_z < 0.0 ? 1.0 : 0.0);
}

bool compatible_current_surface(uint2 pixel, uint2 neighbor_pixel) {
    float4 center_position_depth = surface_position_depth[pixel];
    float4 neighbor_position_depth = surface_position_depth[neighbor_pixel];
    if (center_position_depth.w < 0.0 || neighbor_position_depth.w < 0.0) {
        return false;
    }

    float4 center_normal_roughness = surface_normal_roughness[pixel];
    float4 neighbor_normal_roughness = surface_normal_roughness[neighbor_pixel];
    if (dot(normalize(center_normal_roughness.xyz), normalize(neighbor_normal_roughness.xyz)) < 0.85) {
        return false;
    }

    float4 center_albedo_material = surface_albedo_material[pixel];
    float4 neighbor_albedo_material = surface_albedo_material[neighbor_pixel];
    if (abs(center_albedo_material.w - neighbor_albedo_material.w) > 0.5) {
        return false;
    }

    float roughness_delta = abs(surface_material_roughness[pixel] - surface_material_roughness[neighbor_pixel]);
    if (roughness_delta > 0.35) {
        return false;
    }

    float position_delta = distance(center_position_depth.xyz, neighbor_position_depth.xyz);
    float depth_scale = max(center_position_depth.w, 1.0);
    return position_delta <= max(1.0, depth_scale * 0.02);
}

void sort_firefly_luma(inout float sorted_luma[8], uint count) {
    [unroll]
    for (uint i = 1u; i < 8u; i++) {
        if (i >= count) {
            continue;
        }

        float key = sorted_luma[i];
        uint j = i;
        [unroll]
        for (uint step = 0u; step < 8u; step++) {
            if (j == 0u || sorted_luma[j - 1u] <= key) {
                break;
            }
            sorted_luma[j] = sorted_luma[j - 1u];
            j -= 1u;
        }
        sorted_luma[j] = key;
    }
}

FireflyClampStats gather_firefly_clamp_stats(uint2 pixel, uint2 resolution) {
    float sorted_luma[8];
    float sum_luma = 0.0;
    uint sample_count_u = 0u;

    [unroll]
    for (uint i = 0u; i < 8u; i++) {
        sorted_luma[i] = 0.0;
    }

    [unroll]
    for (uint tap = 0u; tap < 8u; tap++) {
        int2 neighbor_i = int2(pixel) + firefly_tap_offsets[tap];
        if (!history_tap_inside(neighbor_i, resolution)) {
            continue;
        }

        uint2 neighbor_pixel = uint2(neighbor_i);
        if (!compatible_current_surface(pixel, neighbor_pixel)) {
            continue;
        }

        float neighbor_luma = luminance_of(noisy_radiance_image[neighbor_pixel].rgb);
        sorted_luma[sample_count_u] = neighbor_luma;
        sum_luma += neighbor_luma;
        sample_count_u += 1u;
    }

    FireflyClampStats stats;
    stats.mean_luma = 0.0;
    stats.median_luma = 0.0;
    stats.trimmed_mean_luma = 0.0;
    stats.robust_sigma_luma = 0.0;
    stats.sample_count = float(sample_count_u);
    if (sample_count_u > 0u) {
        sort_firefly_luma(sorted_luma, sample_count_u);
        float sample_count = float(sample_count_u);
        stats.mean_luma = sum_luma / sample_count;
        stats.median_luma = sorted_luma[min(sample_count_u / 2u, sample_count_u - 1u)];

        uint trimmed_count = sample_count_u;
        if (sample_count_u >= 6u) {
            trimmed_count = sample_count_u - 2u;
        } else if (sample_count_u >= 4u) {
            trimmed_count = sample_count_u - 1u;
        }

        float trimmed_sum = 0.0;
        float trimmed_sum_sq = 0.0;
        [unroll]
        for (uint i = 0u; i < 8u; i++) {
            if (i >= trimmed_count) {
                continue;
            }
            trimmed_sum += sorted_luma[i];
            trimmed_sum_sq += sorted_luma[i] * sorted_luma[i];
        }

        float trimmed_count_f = float(trimmed_count);
        stats.trimmed_mean_luma = trimmed_sum / trimmed_count_f;
        float trimmed_variance = max(
            trimmed_sum_sq / trimmed_count_f - stats.trimmed_mean_luma * stats.trimmed_mean_luma,
            0.0
        );
        stats.robust_sigma_luma = sqrt(trimmed_variance);
    }
    return stats;
}

float spatial_firefly_luma_limit(FireflyClampStats stats) {
    if (stats.sample_count < 3.0) {
        return 1.0e20;
    }

    float variance_limit = stats.trimmed_mean_luma + stats.robust_sigma_luma * FIREFLY_CLAMP_SIGMA_SCALE + 0.25;
    float median_limit = stats.median_luma * FIREFLY_CLAMP_MEDIAN_SCALE + 0.25;
    float trimmed_mean_limit = stats.trimmed_mean_luma * FIREFLY_CLAMP_TRIMMED_MEAN_SCALE + 0.25;
    return max(max(max(variance_limit, median_limit), trimmed_mean_limit), FIREFLY_CLAMP_MIN_LUMA);
}

ClampedRadianceSample clamp_noisy_radiance_to_spatial_firefly_stats(
    uint2 pixel,
    float3 noisy_radiance,
    float2 noisy_moments,
    FireflyClampStats stats
) {
    ClampedRadianceSample clamped;
    clamped.radiance = noisy_radiance;
    clamped.moments = noisy_moments;

    if (surface_normal_roughness[pixel].w > 1.0e-4) {
        return clamped;
    }

    float noisy_luma = luminance_of(noisy_radiance);
    float luma_limit = spatial_firefly_luma_limit(stats);
    float clamped_luma = min(noisy_luma, luma_limit);
    if (noisy_luma > clamped_luma && noisy_luma > 1.0e-4) {
        clamped.radiance *= clamped_luma / noisy_luma;
        clamped.moments = float2(clamped_luma, clamped_luma * clamped_luma);
    }
    return clamped;
}

bool compatible_history(uint2 pixel, uint2 previous_pixel) {
    if (vpt_surface_generation_rejects_history(
        previous_surface_brick_generation[previous_pixel],
        surface_brick_generation[pixel])) {
        return false;
    }

    float4 current_position_depth = surface_position_depth[pixel];
    float4 previous_position_depth = previous_surface_position_depth[previous_pixel];
    if (current_position_depth.w < 0.0 || previous_position_depth.w < 0.0) {
        return false;
    }

    float4 current_normal_roughness = surface_normal_roughness[pixel];
    float4 previous_normal_roughness = previous_surface_normal_roughness[previous_pixel];
    if (dot(normalize(current_normal_roughness.xyz), normalize(previous_normal_roughness.xyz)) < 0.85) {
        return false;
    }

    float4 current_albedo_material = surface_albedo_material[pixel];
    float4 previous_albedo_material = previous_surface_albedo_material[previous_pixel];
    if (abs(current_albedo_material.w - previous_albedo_material.w) > 0.5) {
        return false;
    }

    float roughness_delta = abs(surface_material_roughness[pixel] - previous_surface_material_roughness[previous_pixel]);
    if (roughness_delta > 0.35) {
        return false;
    }

    float position_delta = distance(current_position_depth.xyz, previous_position_depth.xyz);
    float depth_scale = max(current_position_depth.w, 1.0);
    return position_delta <= max(1.5, depth_scale * 0.03);
}

float bounded_alpha(float history_length) {
    return history_length <= 0.0 ? 1.0 : clamp(1.0 / (history_length + 1.0), 0.04, 0.25);
}

float history_tap_weight(uint tap, float2 history_fraction) {
    float wx = history_tap_offsets[tap].x == 0 ? 1.0 - history_fraction.x : history_fraction.x;
    float wy = history_tap_offsets[tap].y == 0 ? 1.0 - history_fraction.y : history_fraction.y;
    return wx * wy;
}

bool history_tap_inside(int2 previous_pixel, uint2 resolution) {
    return previous_pixel.x >= 0
        && previous_pixel.y >= 0
        && previous_pixel.x < int(resolution.x)
        && previous_pixel.y < int(resolution.y);
}

ReprojectedHistorySample sample_reprojected_history(
    uint2 pixel,
    int2 previous_base_pixel,
    float2 history_fraction,
    uint2 resolution,
    uint current_history_generation_bits
) {
    ReprojectedHistorySample history = (ReprojectedHistorySample)0;

    [unroll]
    for (uint tap = 0u; tap < 4u; tap++) {
        int2 tap_pixel_i = previous_base_pixel + history_tap_offsets[tap];
        float tap_weight = history_tap_weight(tap, history_fraction);
        if (tap_weight <= 0.0 || !history_tap_inside(tap_pixel_i, resolution)) {
            continue;
        }

        uint2 tap_pixel = uint2(tap_pixel_i);
        float4 previous_accumulated_radiance = previous_accumulated_radiance_image[tap_pixel];
        uint previous_history_generation_bits = f32tof16(previous_accumulated_radiance.a);
        if (previous_history_generation_bits != current_history_generation_bits) {
            continue;
        }

        if (!compatible_history(pixel, tap_pixel)) {
            continue;
        }

        float4 previous_moments_history = previous_accumulated_moments_history_image[tap_pixel];
        if (previous_moments_history.z <= 0.0 || previous_moments_history.w <= 0.0) {
            continue;
        }

        history.radiance += previous_accumulated_radiance.rgb * tap_weight;
        history.moments += previous_moments_history.xy * tap_weight;
        history.history_length += previous_moments_history.z * tap_weight;
        history.history_reliability += tap_weight;
    }

    if (history.history_reliability > 0.0) {
        float inv_weight = 1.0 / history.history_reliability;
        history.radiance *= inv_weight;
        history.moments *= inv_weight;
        history.history_length *= inv_weight;
    }

    return history;
}

ClampedRadianceSample clamp_noisy_radiance_to_history(
    float3 noisy_radiance,
    float2 noisy_moments,
    ReprojectedHistorySample history
) {
    ClampedRadianceSample clamped;
    clamped.radiance = noisy_radiance;
    clamped.moments = noisy_moments;

    float noisy_luma = luminance_of(noisy_radiance);
    float history_luma = max(history.moments.x, 0.0);
    float history_variance = max(history.moments.y - history_luma * history_luma, 0.0);
    float history_sigma = sqrt(history_variance);
    float clamp_radius = max(0.05, history_sigma * 3.0 + history_luma * 0.05);
    float clamped_luma = clamp(noisy_luma, max(history_luma - clamp_radius, 0.0), history_luma + clamp_radius);

    if (noisy_luma > 1.0e-4) {
        clamped.radiance *= clamped_luma / noisy_luma;
    }
    clamped.moments = float2(clamped_luma, clamped_luma * clamped_luma);
    return clamped;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    uint2 pixel = tid.xy;
    uint current_history_generation_bits = vpt_history_generation_half_bits(scene.history_reset_generation);
    float history_generation_alpha = vpt_history_generation_alpha(current_history_generation_bits);
    float4 noisy_radiance = noisy_radiance_image[pixel];
    float4 noisy_moments = noisy_moments_image[pixel];
    float4 motion = motion_history[pixel];
    uint flags = motion_flags[pixel];
    bool current_surface_valid = surface_position_depth[pixel].w >= 0.0;
    bool motion_valid = current_surface_valid && !vpt_motion_flags_reject_history(flags);
    FireflyClampStats firefly_stats = (FireflyClampStats)0;

    if (debug_view_bypasses_temporal(scene.vpt_debug_view)) {
        accumulated_radiance_image[pixel] = float4(noisy_radiance.rgb, history_generation_alpha);
        accumulated_moments_history_image[pixel] = float4(noisy_moments.xy, current_surface_valid ? 1.0 : 0.0, current_surface_valid ? 1.0 : 0.0);
        return;
    }

    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_NORMAL) {
        float3 normal_debug = current_surface_valid ? normalize(surface_normal_roughness[pixel].xyz) * 0.5 + 0.5 : float3(0.0);
        accumulated_radiance_image[pixel] = float4(normal_debug, history_generation_alpha);
        accumulated_moments_history_image[pixel] = float4(noisy_moments.xy, current_surface_valid ? 1.0 : 0.0, current_surface_valid ? 1.0 : 0.0);
        return;
    }

    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_DEPTH) {
        accumulated_radiance_image[pixel] = float4(visualize_linear_depth(surface_position_depth[pixel].w), history_generation_alpha);
        accumulated_moments_history_image[pixel] = float4(noisy_moments.xy, current_surface_valid ? 1.0 : 0.0, current_surface_valid ? 1.0 : 0.0);
        return;
    }

    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_NRD_NORMAL_ROUGHNESS
        || scene.vpt_debug_view == VPT_DEBUG_VIEW_NRD_VIEWZ
        || scene.vpt_debug_view == VPT_DEBUG_VIEW_NRD_MOTION
        || scene.vpt_debug_view == VPT_DEBUG_VIEW_NRD_MOTION_Z) {
        float3 guide_debug = float3(0.0);
        if (scene.vpt_debug_view == VPT_DEBUG_VIEW_NRD_NORMAL_ROUGHNESS) {
            guide_debug = visualize_nrd_normal_roughness(pixel, current_surface_valid);
        } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_NRD_VIEWZ) {
            guide_debug = current_surface_valid ? visualize_linear_depth(surface_view_z[pixel]) : float3(0.0);
        } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_NRD_MOTION) {
            float2 delta = motion_valid ? motion.xy : float2(0.0);
            guide_debug = float3(delta * 0.05 + 0.5, motion_valid ? 1.0 : 0.0);
        } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_NRD_MOTION_Z) {
            guide_debug = visualize_nrd_motion_z(motion.z, motion_valid);
        }
        accumulated_radiance_image[pixel] = float4(guide_debug, history_generation_alpha);
        accumulated_moments_history_image[pixel] = float4(noisy_moments.xy, current_surface_valid ? 1.0 : 0.0, current_surface_valid ? 1.0 : 0.0);
        return;
    }

    if ((scene.denoiser_flags & DENOISER_FLAG_ENABLED) == 0u) {
        accumulated_radiance_image[pixel] = float4(noisy_radiance.rgb, history_generation_alpha);
        accumulated_moments_history_image[pixel] = float4(
            noisy_moments.xy,
            current_surface_valid ? 1.0 : 0.0,
            current_surface_valid ? 1.0 : 0.0
        );
        return;
    }

    if (current_surface_valid) {
        firefly_stats = gather_firefly_clamp_stats(pixel, scene.resolution);
        ClampedRadianceSample clamped_noisy = clamp_noisy_radiance_to_spatial_firefly_stats(
            pixel,
            noisy_radiance.rgb,
            noisy_moments.xy,
            firefly_stats
        );
        noisy_radiance = float4(clamped_noisy.radiance, noisy_radiance.a);
        noisy_moments = float4(clamped_noisy.moments, noisy_moments.zw);
    }

    float3 accumulated = noisy_radiance.rgb;
    float2 accumulated_moments = noisy_moments.xy;
    float history_length = current_surface_valid ? 1.0 : 0.0;
    float history_valid = current_surface_valid ? 1.0 : 0.0;
    float history_reused = 0.0;

    if (!vpt_motion_flags_reject_history(flags)) {
        if (current_surface_valid) {
            float2 history_sample = vpt_history_sample_from_motion(pixel, motion);
            int2 previous_base_pixel = int2(floor(history_sample));
            float2 history_fraction = saturate(history_sample - float2(previous_base_pixel));
            ReprojectedHistorySample history = sample_reprojected_history(
                pixel,
                previous_base_pixel,
                history_fraction,
                scene.resolution,
                current_history_generation_bits
            );
            if (history.history_length > 0.0 && history.history_reliability > 0.0) {
                ClampedRadianceSample history_clamped_noisy = clamp_noisy_radiance_to_history(
                    noisy_radiance.rgb,
                    noisy_moments.xy,
                    history
                );
                float alpha = clamp(
                    bounded_alpha(history.history_length) / max(history.history_reliability, 0.25),
                    0.04,
                    0.85
                );
                accumulated = lerp(history.radiance, history_clamped_noisy.radiance, alpha);
                accumulated_moments = lerp(history.moments, history_clamped_noisy.moments, alpha);
                history_length = min(history.history_length + history.history_reliability, 64.0);
                history_valid = 1.0;
                history_reused = history.history_reliability;
            }

            ClampedRadianceSample output_clamped = clamp_noisy_radiance_to_spatial_firefly_stats(
                pixel,
                accumulated,
                accumulated_moments,
                firefly_stats
            );
            accumulated = output_clamped.radiance;
            accumulated_moments = output_clamped.moments;
        }
    }

    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_HISTORY_VALID) {
        accumulated = float3(history_reused);
    } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_MOTION) {
        float2 delta = !vpt_motion_flags_reject_history(flags) ? motion.xy : float2(0.0);
        accumulated = float3(delta * 0.05 + 0.5, !vpt_motion_flags_reject_history(flags) ? 1.0 : 0.0);
    } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_VARIANCE) {
        accumulated = visualize_luminance_variance(accumulated_moments);
    } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_TEMPORAL) {
        accumulated = accumulated;
    }

    accumulated_radiance_image[pixel] = float4(accumulated, history_generation_alpha);
    accumulated_moments_history_image[pixel] = float4(accumulated_moments, history_length, history_valid);
}
