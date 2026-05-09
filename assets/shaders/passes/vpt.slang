// VPT reference compute shader.
// Traces stochastic voxel paths into current-frame noisy HDR radiance.

#include "voxel_traverse.slang"
#include "material_common.slang"
#include "lighting_common.slang"
#include "restir_di_common.slang"
#include "area_restir_common.slang"
#include "vpt_primary_sample_common.slang"

static const float VPT_RAY_SURFACE_BIAS = 0.01;
static const float VPT_LIGHT_ENDPOINT_BIAS = 0.01;

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
    return vpt_primary_hash_u32(x);
}

float rand01(inout uint state) {
    return vpt_primary_rand01(state);
}

float3 sample_sun_direction(SceneUniforms scene, inout uint rng_state) {
    float sun_radius = max(scene.sun_angular_radius, 0.0);
    if (sun_radius <= 0.0) {
        return scene.sun_direction;
    }

    float cos_min = cos(sun_radius);
    float cos_theta = lerp(cos_min, 1.0, rand01(rng_state));
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    float phi = 6.28318530718 * rand01(rng_state);
    float3 sun_forward = normalize(scene.sun_direction);
    float3 sun_right = normalize(abs(sun_forward.y) < 0.95 ? cross(sun_forward, float3(0, 1, 0)) : cross(sun_forward, float3(1, 0, 0)));
    float3 sun_up = normalize(cross(sun_right, sun_forward));
    return normalize(
        sun_right * (cos(phi) * sin_theta) +
        sun_up * (sin(phi) * sin_theta) +
        sun_forward * cos_theta
    );
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
    HitResult first_hit;
};

RestirDirectLightSample empty_restir_direct_light_sample() {
    RestirDirectLightSample sample;
    sample.radiance = float3(0.0);
    sample.selected_weight = 0.0;
    return sample;
}

bool analytic_sun_visible_from_hit(HitResult hit, SceneUniforms scene, float3 sun_dir) {
    if ((scene.lighting_flags & LIGHTING_FLAG_SHADOWS_ENABLED) == 0u) {
        return true;
    }

    Ray shadow_ray = make_ray(hit.position + hit.normal * VPT_RAY_SURFACE_BIAS, sun_dir);
    bool sun_occluded = trace_any_hit_ray(
        shadow_ray,
        1.0e20,
        ucvh_config,
        hierarchy_l0,
        brick_occupancy
    );
    return !sun_occluded;
}

float3 analytic_sun_direct(HitResult hit, SceneUniforms scene, inout uint rng_state) {
    float3 sun_dir = sample_sun_direction(scene, rng_state);
    float sun_term = max(dot(hit.normal, sun_dir), 0.0);
    if (sun_term <= 0.0 || !analytic_sun_visible_from_hit(hit, scene, sun_dir)) {
        return float3(0.0);
    }
    float3 albedo = material_cell_albedo(hit.cell);
    return albedo * scene.sun_intensity * sun_term * 0.2;
}

VptTraceSample empty_vpt_trace_sample() {
    VptTraceSample sample;
    sample.radiance = float3(0.0);
    sample.direct_radiance = float3(0.0);
    sample.indirect_radiance = float3(0.0);
    sample.reservoir_weight = 0.0;
    sample.first_hit = NO_HIT;
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

float3 visualize_voxel_brick_id(HitResult first_hit) {
    if (!first_hit.hit) {
        return float3(0.0);
    }
    uint h = hash_u32(first_hit.brick_id * 747796405u + 2891336453u);
    return float3(
        float((h >> 0) & 0xffu),
        float((h >> 8) & 0xffu),
        float((h >> 16) & 0xffu)
    ) / 255.0;
}

float3 visualize_voxel_local_coord(HitResult first_hit) {
    if (!first_hit.hit) {
        return float3(0.0);
    }
    return (float3(first_hit.local) + 0.5) / 8.0;
}

float3 visualize_voxel_hit(HitResult first_hit) {
    if (!first_hit.hit) {
        return float3(0.0);
    }
    return normalize(first_hit.normal) * 0.5 + 0.5;
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
    AreaRestirReservoir reservoir = index < area_restir.reservoir_count
        ? area_restir_reservoirs[index]
        : area_restir_invalid_reservoir();
    return vpt_resolve_area_restir_primary_ray(pixel, scene, area_restir, reservoir, index, rng_state);
}

bool restir_di_light_visible_from_hit(HitResult hit, RestirDiReservoir reservoir, SceneUniforms scene) {
    if ((scene.lighting_flags & LIGHTING_FLAG_SHADOWS_ENABLED) == 0u) {
        return true;
    }

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
        max_light_t = max(light_distance - VPT_LIGHT_ENDPOINT_BIAS, 0.0);
    }

    Ray shadow_ray = make_ray(hit.position + hit.normal * VPT_RAY_SURFACE_BIAS, shadow_dir);
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
    if (reservoir.sample_light_id == 0xffffffffu || reservoir.weight_sum <= 0.0) {
        return sample;
    }
    float4 hit_position_depth = float4(hit.position, max(hit.t, 0.0));
    float4 hit_normal_roughness = float4(normalize(hit.normal), 0.0);
    float4 hit_albedo_material = float4(material_cell_albedo(hit.cell), float(voxel_material(hit.cell)));
    float hit_target_pdf = restir_di_target_pdf_for_reservoir(
        hit_position_depth,
        hit_normal_roughness,
        hit_albedo_material,
        reservoir
    );
    if (!restir_di_candidate_finite(hit_target_pdf)) {
        return sample;
    }
    RestirDiReservoir hit_reservoir = reservoir;
    hit_reservoir.target_pdf = hit_target_pdf;
    hit_reservoir.selected_weight = min(
        reservoir.weight_sum / max(hit_target_pdf * float(max(reservoir.sample_count_m, 1u)), 1.0e-4),
        RESTIR_DI_MAX_SELECTED_WEIGHT
    );
    float selected_weight = restir_di_bounded_selected_weight(hit_reservoir);
    if (selected_weight <= 0.0) {
        return sample;
    }
    if (!restir_di_light_visible_from_hit(hit, reservoir, scene)) {
        return sample;
    }
    sample.selected_weight = selected_weight;

    float3 albedo = material_cell_albedo(hit.cell);
    if (reservoir.sample_flags == 0u) {
        float sun_term = max(dot(hit.normal, scene.sun_direction), 0.0);
        sample.radiance = albedo * reservoir.sample_radiance.rgb * sun_term * selected_weight * 0.2;
        return sample;
    }

    float geometry_term = restir_di_emissive_geometry_term(hit.position, hit.normal, reservoir.sample_position_pdf.xyz);
    sample.radiance = albedo * reservoir.sample_radiance.rgb * geometry_term * selected_weight;
    return sample;
}

VptTraceSample trace_path(Ray ray, SceneUniforms scene, uint2 pixel, inout uint rng_state) {
    float3 throughput = float3(1.0);
    VptTraceSample sample = empty_vpt_trace_sample();
    uint max_bounces = clamp(scene.vpt_max_bounces, 1u, 8u);

    for (uint bounce = 0u; bounce < max_bounces; bounce++) {
        HitResult hit = trace_primary_ray(ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);
        if (bounce == 0u) {
            sample.first_hit = hit;
        }
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
            float3 analytic_direct = analytic_sun_direct(hit, scene, rng_state);
            float3 direct_radiance = analytic_direct + direct.radiance;
            float3 contribution = throughput * direct_radiance;
            sample.radiance += contribution;
            sample.direct_radiance += contribution;
            sample.reservoir_weight = direct.selected_weight;
        } else {
            float3 contribution = throughput * analytic_sun_direct(hit, scene, rng_state);
            sample.radiance += contribution;
            if (bounce == 0u) {
                sample.direct_radiance += contribution;
            } else {
                sample.indirect_radiance += contribution;
            }
        }
        throughput *= material_cell_albedo(hit.cell);
        ray = make_ray(hit.position + hit.normal * VPT_RAY_SURFACE_BIAS, cosine_hemisphere(hit.normal, rng_state));
    }

    return sample;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    uint rng_state = vpt_primary_rng_seed(tid.xy, scene);
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
    } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_VOXEL_BRICK) {
        sample_radiance = visualize_voxel_brick_id(sample.first_hit);
    } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_VOXEL_LOCAL) {
        sample_radiance = visualize_voxel_local_coord(sample.first_hit);
    } else if (scene.vpt_debug_view == VPT_DEBUG_VIEW_VOXEL_HIT) {
        sample_radiance = visualize_voxel_hit(sample.first_hit);
    } else if (is_area_restir_debug_view(scene.vpt_debug_view)) {
        sample_radiance = visualize_area_restir_debug(tid.xy, scene);
    }

    float luminance = dot(sample_radiance, float3(0.2126, 0.7152, 0.0722));
    noisy_radiance_image[tid.xy] = float4(sample_radiance, 1.0);
    noisy_moments_image[tid.xy] = float4(luminance, luminance * luminance, 1.0, 1.0);
}
