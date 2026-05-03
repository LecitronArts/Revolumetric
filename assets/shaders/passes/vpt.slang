// VPT reference compute shader.
// Traces stochastic voxel paths into current-frame noisy HDR radiance.

#include "voxel_traverse.slang"
#include "material_common.slang"
#include "lighting_common.slang"
#include "restir_di_common.slang"
#include "area_restir_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> noisy_radiance_image;

[[vk::binding(2, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

[[vk::binding(3, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(4, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(5, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

[[vk::binding(6, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(7, 0)]]
StructuredBuffer<RestirDiReservoir> restir_reservoirs;

[[vk::binding(8, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> noisy_moments_image;

[[vk::binding(9, 0)]]
ConstantBuffer<AreaRestirUniforms> area_restir;

[[vk::binding(10, 0)]]
StructuredBuffer<AreaRestirReservoir> area_restir_reservoirs;

uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rand01(inout uint state) {
    state = hash_u32(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

float3 cosine_hemisphere(float3 normal, inout uint state) {
    float r1 = rand01(state);
    float r2 = rand01(state);
    float phi = 6.28318530718 * r1;
    float r = sqrt(r2);
    float3 tangent = normalize(abs(normal.y) < 0.95 ? cross(normal, float3(0, 1, 0)) : cross(normal, float3(1, 0, 0)));
    float3 bitangent = normalize(cross(normal, tangent));
    return normalize(tangent * (cos(phi) * r) + bitangent * (sin(phi) * r) + normal * sqrt(max(0.0, 1.0 - r2)));
}

struct RestirDirectLightSample {
    float3 radiance;
    float selected_weight;
};

struct VptTraceSample {
    float3 radiance;
    float3 direct_radiance;
    float3 indirect_radiance;
    float reservoir_weight;
};

RestirDirectLightSample empty_restir_direct_light_sample() {
    RestirDirectLightSample sample;
    sample.radiance = float3(0.0);
    sample.selected_weight = 0.0;
    return sample;
}

VptTraceSample empty_vpt_trace_sample() {
    VptTraceSample sample;
    sample.radiance = float3(0.0);
    sample.direct_radiance = float3(0.0);
    sample.indirect_radiance = float3(0.0);
    sample.reservoir_weight = 0.0;
    return sample;
}

float3 visualize_restir_reservoir_weight(float selected_weight) {
    float heat = saturate(log2(max(selected_weight, 0.0) + 1.0) / 8.0);
    return float3(heat, heat * heat, 1.0 - heat);
}

float3 visualize_area_restir_heat(float value) {
    float heat = saturate(log2(max(value, 0.0) + 1.0) / 8.0);
    return float3(heat, heat * heat, 1.0 - heat);
}

float3 visualize_area_restir_id(uint value) {
    if (value == 0u) {
        return float3(0.0);
    }
    uint h = hash_u32(value * 747796405u + 2891336453u);
    return float3(
        float((h >> 0) & 0xffu),
        float((h >> 8) & 0xffu),
        float((h >> 16) & 0xffu)
    ) / 255.0;
}

float3 visualize_area_restir_debug(uint2 pixel, SceneUniforms scene) {
    uint index = pixel.y * scene.resolution.x + pixel.x;
    if (area_restir.enabled == 0u || index >= area_restir.reservoir_count) {
        return float3(0.35, 0.0, 0.0);
    }

    AreaRestirReservoir reservoir = area_restir_reservoirs[index];
    bool valid = area_restir_is_valid_reservoir(reservoir);
    if (!valid) {
        if (scene.vpt_debug_view == VPT_DEBUG_VIEW_AREA_REJECTION) {
            return visualize_area_restir_id(max(reservoir.rejection_reason, 1u));
        }
        return float3(0.0);
    }

    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_AREA_SUBPIXEL) {
        return float3(saturate(reservoir.sample_state.subpixel_uv), 1.0);
    }
    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_AREA_LENS) {
        return float3(saturate(reservoir.sample_state.lens_uv), 1.0);
    }
    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_AREA_WEIGHT) {
        return visualize_area_restir_heat(max(reservoir.selected_weight, reservoir.weight_sum));
    }
    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_AREA_HISTORY_VALID) {
        return (reservoir.sample_state.flags & AREA_RESTIR_SAMPLE_HISTORY_VALID) != 0u
            ? float3(0.0, 1.0, 0.0)
            : float3(0.25, 0.0, 0.0);
    }
    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_AREA_REJECTION) {
        return visualize_area_restir_id(reservoir.rejection_reason);
    }
    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_AREA_JACOBIAN) {
        return visualize_area_restir_heat(reservoir.jacobian);
    }
    return float3(0.0);
}

bool is_area_restir_debug_view(uint debug_view) {
    return debug_view >= VPT_DEBUG_VIEW_AREA_SUBPIXEL
        && debug_view <= VPT_DEBUG_VIEW_AREA_JACOBIAN;
}

ScenePrimaryRay resolve_area_restir_primary_ray(uint2 pixel, SceneUniforms scene, inout uint rng_state) {
    uint index = pixel.y * scene.resolution.x + pixel.x;
    if (area_restir.enabled != 0u && index < area_restir.reservoir_count) {
        AreaRestirReservoir reservoir = area_restir_reservoirs[index];
        if (area_restir_is_valid_reservoir(reservoir)) {
            return scene_primary_ray_from_area_sample(
                scene,
                area_restir_pixel_sample(pixel, reservoir.sample_state),
                reservoir.sample_state.lens_uv
            );
        }
    }

    // fallback jitter: preserve the existing stochastic VPT ray when Area ReSTIR is disabled or invalid.
    float2 fallback_jitter = float2(rand01(rng_state), rand01(rng_state)) - 0.5;
    return scene_primary_ray_from_area_sample(scene, float2(pixel) + fallback_jitter, float2(0.5, 0.5));
}

bool restir_di_light_visible_from_hit(HitResult hit, RestirDiReservoir reservoir) {
    float3 shadow_dir;
    float max_light_t;
    if (reservoir.sample_flags == 0u) {
        shadow_dir = normalize(reservoir.sample_position_pdf.xyz);
        max_light_t = 1.0e20;
    } else {
        float3 to_light = reservoir.sample_position_pdf.xyz - hit.position;
        float distance_sq = dot(to_light, to_light);
        if (distance_sq <= 1.0e-4) {
            return true;
        }
        float light_distance = sqrt(distance_sq);
        shadow_dir = to_light / light_distance;
        max_light_t = max(light_distance - 1.0, 0.0);
    }

    Ray shadow_ray = make_ray(hit.position + hit.normal * 0.05 + shadow_dir * 0.05, shadow_dir);
    bool shadow_occluded = trace_any_hit_ray(
        shadow_ray,
        max_light_t,
        ucvh_config,
        hierarchy_l0,
        brick_occupancy
    );
    return !shadow_occluded;
}

RestirDirectLightSample resolve_restir_di_direct_light(uint2 pixel, HitResult hit, SceneUniforms scene) {
    RestirDirectLightSample sample = empty_restir_direct_light_sample();
    uint index = pixel.y * scene.resolution.x + pixel.x;
    if (index >= restir.reservoir_count) {
        return sample;
    }

    RestirDiReservoir reservoir = restir_reservoirs[index];
    if (reservoir.sample_light_id == 0xffffffffu || reservoir.weight_sum <= 0.0 || reservoir.selected_weight <= 0.0) {
        return sample;
    }
    sample.selected_weight = reservoir.selected_weight;

    if (!restir_di_light_visible_from_hit(hit, reservoir)) {
        return sample;
    }

    float3 albedo = material_cell_albedo(hit.cell);
    if (reservoir.sample_flags == 0u) {
        float sun_term = max(dot(hit.normal, scene.sun_direction), 0.0);
        sample.radiance = albedo * reservoir.sample_radiance.rgb * sun_term * reservoir.selected_weight * 0.2;
        return sample;
    }

    float geometry_term = restir_di_emissive_geometry_term(hit.position, hit.normal, reservoir.sample_position_pdf.xyz);
    sample.radiance = albedo * reservoir.sample_radiance.rgb * geometry_term * reservoir.selected_weight;
    return sample;
}

VptTraceSample trace_path(Ray ray, SceneUniforms scene, uint2 pixel, inout uint rng_state) {
    float3 throughput = float3(1.0);
    VptTraceSample sample = empty_vpt_trace_sample();
    uint max_bounces = clamp(scene.vpt_max_bounces, 1u, 8u);

    for (uint bounce = 0u; bounce < max_bounces; bounce++) {
        HitResult hit = trace_primary_ray(ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);
        if (!hit.hit) {
            float3 contribution = throughput * sky_color_for_dir(ray.direction, scene);
            sample.radiance += contribution;
            if (bounce == 0u) {
                sample.direct_radiance += contribution;
            } else {
                sample.indirect_radiance += contribution;
            }
            break;
        }

        float3 emissive = material_emissive(hit.cell) * 3.0;
        if (emissive.x + emissive.y + emissive.z > 0.0) {
            float3 contribution = throughput * emissive;
            sample.radiance += contribution;
            if (bounce == 0u) {
                sample.direct_radiance += contribution;
            } else {
                sample.indirect_radiance += contribution;
            }
            break;
        }

        if (restir.enabled != 0u && bounce == 0u) {
            RestirDirectLightSample direct = resolve_restir_di_direct_light(pixel, hit, scene);
            float3 contribution = throughput * direct.radiance;
            sample.radiance += contribution;
            sample.direct_radiance += contribution;
            sample.reservoir_weight = direct.selected_weight;
        } else {
            float sun_term = max(dot(hit.normal, scene.sun_direction), 0.0);
            float3 contribution = throughput * material_cell_albedo(hit.cell) * scene.sun_intensity * sun_term * 0.2;
            sample.radiance += contribution;
            if (bounce == 0u) {
                sample.direct_radiance += contribution;
            } else {
                sample.indirect_radiance += contribution;
            }
        }
        throughput *= material_cell_albedo(hit.cell);
        ray = make_ray(hit.position + hit.normal * 0.75, cosine_hemisphere(hit.normal, rng_state));
    }

    return sample;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    uint rng_state = hash_u32(
        tid.x * 1973u ^
        tid.y * 9277u ^
        scene.vpt_sample_index * 26699u ^
        uint(scene.time * 1000.0)
    );
    ScenePrimaryRay primary_ray = resolve_area_restir_primary_ray(tid.xy, scene, rng_state);
    VptTraceSample sample = trace_path(
        make_ray(primary_ray.origin, primary_ray.direction),
        scene,
        tid.xy,
        rng_state
    );
    float3 sample_radiance = sample.radiance;

    if (scene.vpt_debug_view == VPT_DEBUG_VIEW_RESERVOIR_WEIGHT) {
        sample_radiance = visualize_restir_reservoir_weight(sample.reservoir_weight);
    } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_DIRECT) {
        sample_radiance = sample.direct_radiance;
    } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_INDIRECT) {
        sample_radiance = sample.indirect_radiance;
    } else if (is_area_restir_debug_view(scene.vpt_debug_view)) {
        sample_radiance = visualize_area_restir_debug(tid.xy, scene);
    }

    float luminance = dot(sample_radiance, float3(0.2126, 0.7152, 0.0722));
    noisy_radiance_image[tid.xy] = float4(sample_radiance, 1.0);
    noisy_moments_image[tid.xy] = float4(luminance, luminance * luminance, 1.0, 1.0);
}
