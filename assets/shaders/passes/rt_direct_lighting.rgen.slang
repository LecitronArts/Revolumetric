#include "restir_di_common.slang"
#include "restir_gi_common.slang"
#include "rt_history_common.slang"
#include "rt_shadow_common.slang"

static const float RT_DIRECT_SURFACE_BIAS = 0.01;
static const float RT_DIRECT_LIGHT_ENDPOINT_BIAS = 0.01;

struct RtDirectLightingUniforms {
    uint restir_di_enabled;
    uint restir_gi_enabled;
    uint debug_view;
    uint width;
    uint height;
    uint shadows_enabled;
    uint sun_sample_index;
    uint _pad0;
    float4 sky_color_sun_angular_radius;
    float4 ground_color_pad;
    float4 sun_direction_pad;
    float4 sun_intensity_pad;
};

[[vk::binding(0, 0)]]
StructuredBuffer<RtSurfacePixel> surface_pixels;

[[vk::binding(1, 0)]]
StructuredBuffer<RestirDiReservoir> direct_reservoirs;

[[vk::binding(2, 0)]]
RWTexture2D<float4> current_radiance;

[[vk::binding(3, 0)]]
ConstantBuffer<RtDirectLightingUniforms> rt_direct;

[[vk::binding(4, 0)]]
RaytracingAccelerationStructure scene_tlas;

[[vk::binding(14, 0)]]
StructuredBuffer<RestirGiReservoir> indirect_reservoirs;

bool rt_direct_surface_valid(RtSurfacePixel surface) {
    return surface.hit_kind == RT_SURFACE_HIT_KIND_VOXEL && surface.position_depth.w >= 0.0;
}

float3 rt_direct_primary_emissive(RtSurfacePixel surface) {
    return surface.emissive_radiance.rgb;
}

float3 rt_direct_visualize_surface(RtSurfacePixel surface) {
    if (!rt_direct_surface_valid(surface)) {
        return float3(0.0);
    }
    return saturate(surface.albedo_material.rgb);
}

float3 rt_direct_visualize_hit_distance(RtSurfacePixel surface) {
    if (!rt_direct_surface_valid(surface)) {
        return float3(0.0);
    }
    float depth = saturate(log2(surface.linear_depth + 1.0) / 12.0);
    return float3(depth);
}

static const float RT_DIRECT_INV_PI = 0.31830988618;

uint rt_direct_sun_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rt_direct_sun_rand01(inout uint rng_state) {
    rng_state = rt_direct_sun_hash_u32(rng_state);
    return float(rng_state & 0x00ffffffu) / 16777216.0;
}

uint rt_direct_sun_rng_seed(uint2 pixel) {
    return rt_direct_sun_hash_u32(
        pixel.x * 1973u ^
        pixel.y * 9277u ^
        rt_direct.sun_sample_index * 26699u
    );
}

float3 rt_direct_sample_sun_direction(inout uint rng_state) {
    float sun_radius = max(rt_direct.sky_color_sun_angular_radius.w, 0.0);
    if (sun_radius <= 0.0) {
        return normalize(rt_direct.sun_direction_pad.xyz);
    }

    float cos_min = cos(sun_radius);
    float cos_theta = lerp(cos_min, 1.0, rt_direct_sun_rand01(rng_state));
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    float phi = 6.28318530718 * rt_direct_sun_rand01(rng_state);
    float3 sun_forward = normalize(rt_direct.sun_direction_pad.xyz);
    float3 sun_right = normalize(abs(sun_forward.y) < 0.95 ? cross(sun_forward, float3(0, 1, 0)) : cross(sun_forward, float3(1, 0, 0)));
    float3 sun_up = normalize(cross(sun_right, sun_forward));
    return normalize(
        sun_right * (cos(phi) * sin_theta) +
        sun_up * (sin(phi) * sin_theta) +
        sun_forward * cos_theta
    );
}

float3 rt_direct_sunlit_ground_color() {
    float ground_ndotl = max(rt_direct.sun_direction_pad.y, 0.0);
    float3 sun_irradiance = rt_direct.sun_intensity_pad.rgb * ground_ndotl;
    return rt_direct.ground_color_pad.rgb * (1.0 + RT_DIRECT_INV_PI * sun_irradiance);
}

float3 rt_direct_background_color_for_dir(float3 direction) {
    float3 miss_dir = normalize(direction);
    float t = saturate(miss_dir.y * 0.5 + 0.5);
    return lerp(rt_direct_sunlit_ground_color(), rt_direct.sky_color_sun_angular_radius.rgb, t);
}

float3 rt_direct_background_color(RtSurfacePixel surface) {
    float3 miss_dir = normalize(surface.view_direction_background.xyz);
    return rt_direct_background_color_for_dir(miss_dir);
}

float3 rt_direct_primary_miss_background_color(RtSurfacePixel surface) {
    float3 miss_dir = normalize(surface.view_direction_background.xyz);
    float t = saturate(miss_dir.y * 0.5 + 0.5);
    float3 horizon_sky = rt_direct.sky_color_sun_angular_radius.rgb * 0.72;
    return lerp(horizon_sky, rt_direct.sky_color_sun_angular_radius.rgb, t);
}

float3 rt_direct_sky_visibility_sample(RtSurfacePixel surface, float3 sky_dir) {
    float3 normal = normalize(surface.normal_roughness.xyz);

    RayDesc ray;
    ray.Origin = surface.position_depth.xyz + normal * RT_DIRECT_SURFACE_BIAS + sky_dir * RT_DIRECT_SURFACE_BIAS;
    ray.Direction = sky_dir;
    ray.TMin = 0.001;
    ray.TMax = 1.0e20;

    RtShadowPayload payload = make_rt_shadow_payload(surface.brick_id, surface.local, ray.TMax);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);
    if (payload.occluded != 0u) {
        return float3(0.0);
    }

    return rt_direct_background_color_for_dir(sky_dir);
}

float3 rt_direct_sample_sky_indirect(RtSurfacePixel surface) {
    float3 normal = normalize(surface.normal_roughness.xyz);
    float3 sky_dir = normal;
    return surface.albedo_material.rgb * rt_direct_sky_visibility_sample(surface, sky_dir);
}

bool rt_direct_sun_visible(RtSurfacePixel surface, float3 sun_dir) {
    if (rt_direct.shadows_enabled == 0u) {
        return true;
    }
    if (!rt_direct_surface_valid(surface)) {
        return false;
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    RayDesc ray;
    ray.Origin = surface.position_depth.xyz + normal * RT_DIRECT_SURFACE_BIAS + sun_dir * RT_DIRECT_SURFACE_BIAS;
    ray.Direction = sun_dir;
    ray.TMin = 0.001;
    ray.TMax = 1.0e20;

    RtShadowPayload payload = make_rt_shadow_payload(surface.brick_id, surface.local, ray.TMax);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);
    return payload.occluded == 0u;
}

float3 rt_direct_analytic_sun_direct(RtSurfacePixel surface, inout uint rng_state) {
    float3 sun_vector = rt_direct.sun_direction_pad.xyz;
    if (dot(sun_vector, sun_vector) <= 1.0e-8) {
        return float3(0.0);
    }

    float3 sun_dir = rt_direct_sample_sun_direction(rng_state);
    float3 normal = normalize(surface.normal_roughness.xyz);
    float sun_term = max(dot(normal, sun_dir), 0.0);
    if (sun_term <= 0.0 || !rt_direct_sun_visible(surface, sun_dir)) {
        return float3(0.0);
    }

    float3 surface_view_dir = normalize(-surface.view_direction_background.xyz);
    float3 direct_brdf = restir_di_direct_brdf(
        normal,
        surface.albedo_material.rgb,
        surface.normal_roughness.w,
        surface_view_dir,
        sun_dir
    );
    return direct_brdf * rt_direct.sun_intensity_pad.rgb * sun_term;
}

float3 rt_direct_visualize_reservoir_weight(RestirDiReservoir reservoir) {
    float selected_weight = restir_di_bounded_selected_weight(reservoir);
    float heat = saturate(log2(max(selected_weight, 0.0) + 1.0) / 8.0);
    return float3(heat, heat * heat, 1.0 - heat);
}

float3 rt_direct_visualize_indirect_reservoir_invalid_reason(RestirGiReservoir reservoir) {
    if (reservoir.sample_count_m == 0u) {
        return float3(1.0, 0.0, 0.0);
    }
    if (!restir_gi_candidate_finite(reservoir.target_pdf)) {
        return float3(1.0, 0.5, 0.0);
    }
    if (!restir_gi_candidate_finite(reservoir.weight_sum)) {
        return float3(1.0, 1.0, 0.0);
    }
    if (!restir_gi_candidate_finite(reservoir.selected_weight)) {
        return float3(1.0, 0.0, 1.0);
    }
    return float3(0.0, 0.0, 0.0);
}

float3 rt_direct_visualize_indirect_reservoir_weight(RestirGiReservoir reservoir) {
    if (!restir_gi_is_valid_reservoir(reservoir)) {
        return rt_direct_visualize_indirect_reservoir_invalid_reason(reservoir);
    }

    float selected_weight = min(reservoir.selected_weight, RESTIR_GI_MAX_SELECTED_WEIGHT);
    float weight_heat = saturate(log2(max(selected_weight, 0.0) + 1.0) / 6.0);
    float m_heat = saturate(log2(float(max(reservoir.sample_count_m, 1u)) + 1.0) / 6.0);
    return float3(weight_heat, m_heat, 1.0);
}

float3 rt_direct_visualize_gi_temporal_reason(RestirGiReservoir reservoir) {
    uint reason = uint(max(reservoir.sample_radiance_pdf.w, 0.0) + 0.5);
    if (reason == RESTIR_GI_TEMPORAL_REUSE_ACCEPTED) {
        return float3(0.0, 1.0, 0.0);
    }
    if (reason == RESTIR_GI_TEMPORAL_REUSE_RESET_OR_DISABLED) {
        return float3(0.0, 0.25, 1.0);
    }
    if (reason == RESTIR_GI_TEMPORAL_REUSE_INVALID_RESOLUTION) {
        return float3(0.0, 1.0, 1.0);
    }
    if (reason == RESTIR_GI_TEMPORAL_REUSE_REPROJECTION_FAILED) {
        return float3(1.0, 0.5, 0.0);
    }
    if (reason == RESTIR_GI_TEMPORAL_REUSE_PREVIOUS_UV_OUTSIDE) {
        return float3(1.0, 1.0, 0.0);
    }
    if (reason == RESTIR_GI_TEMPORAL_REUSE_SURFACE_INCOMPATIBLE) {
        return float3(1.0, 0.0, 1.0);
    }
    if (reason == RESTIR_GI_TEMPORAL_REUSE_HISTORY_RESERVOIR_INVALID) {
        return float3(1.0, 0.0, 0.0);
    }
    if (reason == RESTIR_GI_TEMPORAL_REUSE_CURRENT_WEIGHT_ZERO) {
        return float3(0.5, 0.0, 0.0);
    }
    if (reason == RESTIR_GI_TEMPORAL_REUSE_HISTORY_WEIGHT_ZERO) {
        return float3(0.5, 0.0, 1.0);
    }
    if (reason == RESTIR_GI_TEMPORAL_REUSE_COMBINED_WEIGHT_ZERO) {
        return float3(1.0, 1.0, 1.0);
    }
    return float3(0.0, 0.0, 0.0);
}

float3 rt_direct_visualize_gi_spatial_reason(RestirGiReservoir reservoir) {
    uint reason = uint(max(reservoir.sample_radiance_pdf.w, 0.0) + 0.5);
    if (reason == RESTIR_GI_SPATIAL_REUSE_ACCEPTED_CENTER) {
        return float3(0.0, 0.85, 1.0);
    }
    if (reason == RESTIR_GI_SPATIAL_REUSE_DISABLED_OR_PASSTHROUGH) {
        return float3(0.0, 0.25, 1.0);
    }
    if (reason == RESTIR_GI_SPATIAL_REUSE_INVALID_SURFACE) {
        return float3(1.0, 0.0, 0.0);
    }
    if (reason == RESTIR_GI_SPATIAL_REUSE_CENTER_RESERVOIR_INVALID) {
        return float3(1.0, 0.45, 0.0);
    }
    if (reason == RESTIR_GI_SPATIAL_REUSE_NO_COMPATIBLE_NEIGHBOR) {
        return float3(1.0, 1.0, 0.0);
    }
    if (reason == RESTIR_GI_SPATIAL_REUSE_ACCEPTED_NEIGHBOR) {
        return float3(0.0, 1.0, 0.0);
    }
    if (reason == RESTIR_GI_SPATIAL_REUSE_VISIBILITY_REJECTED) {
        return float3(0.7, 0.0, 1.0);
    }
    return float3(0.0, 0.0, 0.0);
}

float3 rt_direct_visualize_gi_reason(RtSurfacePixel surface, RestirGiReservoir reservoir) {
    if (rt_direct.restir_gi_enabled == 0u) {
        return float3(0.0, 0.25, 1.0);
    }
    if (!restir_gi_is_valid_reservoir(reservoir)) {
        return rt_direct_visualize_indirect_reservoir_invalid_reason(reservoir);
    }

    float selected_weight = min(reservoir.selected_weight, RESTIR_GI_MAX_SELECTED_WEIGHT);
    if (selected_weight <= 0.0) {
        return float3(0.5, 0.0, 1.0);
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    float3 contribution = restir_gi_cosine_sample_contribution(
        surface.albedo_material.rgb,
        surface.position_depth.xyz,
        normal,
        reservoir
    );
    float3 resolved = contribution * selected_weight;
    if (restir_gi_luma(resolved) <= 1.0e-5) {
        return float3(0.0, 0.35, 0.35);
    }

    return restir_gi_is_environment_sample(reservoir)
        ? float3(0.0, 0.85, 1.0)
        : float3(0.0, 1.0, 0.0);
}

bool rt_direct_light_visible(RtSurfacePixel surface, RestirDiReservoir reservoir) {
    if (rt_direct.shadows_enabled == 0u) {
        return true;
    }
    if (!rt_direct_surface_valid(surface)) {
        return false;
    }

    float3 shadow_dir;
    float max_light_t;
    if (reservoir.sample_flags == 0u) {
        shadow_dir = normalize(reservoir.sample_position_pdf.xyz);
        max_light_t = 1.0e20;
    } else {
        float3 to_light = reservoir.sample_position_pdf.xyz - surface.position_depth.xyz;
        float distance_sq = dot(to_light, to_light);
        if (distance_sq <= 1.0e-4) {
            return true;
        }
        float light_distance = sqrt(distance_sq);
        shadow_dir = to_light / light_distance;
        max_light_t = max(light_distance - RT_DIRECT_LIGHT_ENDPOINT_BIAS, 0.0);
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    RayDesc ray;
    ray.Origin = surface.position_depth.xyz + normal * RT_DIRECT_SURFACE_BIAS + shadow_dir * RT_DIRECT_SURFACE_BIAS;
    ray.Direction = shadow_dir;
    ray.TMin = 0.001;
    ray.TMax = max_light_t;

    RtShadowPayload payload = make_rt_shadow_payload(surface.brick_id, surface.local, max_light_t);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);
    return payload.occluded == 0u;
}

float3 rt_direct_resolve_reservoir(RtSurfacePixel surface, RestirDiReservoir reservoir, bool visible) {
    if (!rt_direct_surface_valid(surface) || !restir_di_is_valid_reservoir(reservoir)) {
        return float3(0.0);
    }
    if (!visible) {
        return float3(0.0);
    }

    float selected_weight = restir_di_bounded_selected_weight(reservoir);
    if (selected_weight <= 0.0) {
        return float3(0.0);
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    float3 surface_view_dir = normalize(-surface.view_direction_background.xyz);
    if (reservoir.sample_flags == 0u) {
        float3 light_dir = normalize(reservoir.sample_position_pdf.xyz);
        float sun_term = max(dot(normal, light_dir), 0.0);
        float3 direct_brdf = restir_di_direct_brdf(
            normal,
            surface.albedo_material.rgb,
            surface.normal_roughness.w,
            surface_view_dir,
            light_dir
        );
        float3 resolved_direct =
            direct_brdf * reservoir.sample_radiance.rgb * sun_term * selected_weight;
        return restir_di_clamp_radiance_luma(resolved_direct, RESTIR_DI_MAX_RESOLVED_RADIANCE_LUMA);
    }

    float3 light_dir = normalize(reservoir.sample_position_pdf.xyz - surface.position_depth.xyz);
    float geometry_term = restir_di_emissive_geometry_term(
        surface.position_depth.xyz,
        normal,
        reservoir.sample_position_pdf.xyz
    );
    float3 direct_brdf = restir_di_direct_brdf(
        normal,
        surface.albedo_material.rgb,
        surface.normal_roughness.w,
        surface_view_dir,
        light_dir
    );
    float3 resolved_direct =
        direct_brdf * reservoir.sample_radiance.rgb * geometry_term * selected_weight;
    return restir_di_clamp_radiance_luma(resolved_direct, RESTIR_DI_MAX_RESOLVED_RADIANCE_LUMA);
}

float3 rt_direct_resolve_indirect_reservoir(RtSurfacePixel surface, RestirGiReservoir reservoir) {
    if (!rt_direct_surface_valid(surface) || !restir_gi_is_valid_reservoir(reservoir)) {
        return float3(0.0);
    }

    float selected_weight = min(reservoir.selected_weight, RESTIR_GI_MAX_SELECTED_WEIGHT);
    if (selected_weight <= 0.0) {
        return float3(0.0);
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    float3 contribution = restir_gi_cosine_sample_contribution(
        surface.albedo_material.rgb,
        surface.position_depth.xyz,
        normal,
        reservoir
    );
    if (restir_gi_luma(contribution) <= 1.0e-5) {
        return float3(0.0);
    }
    return restir_gi_clamp_radiance_luma(
        contribution * selected_weight,
        RESTIR_GI_MAX_RESOLVED_RADIANCE_LUMA
    );
}

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    uint index = launch_id.y * rt_direct.width + launch_id.x;
    if (launch_id.x >= rt_direct.width || launch_id.y >= rt_direct.height) {
        return;
    }

    RtSurfacePixel surface = surface_pixels[index];
    if (rt_direct.debug_view == RT_DEBUG_VIEW_SURFACE) {
        current_radiance[launch_id.xy] = float4(rt_direct_visualize_surface(surface), 1.0);
        return;
    }

    if (rt_direct.debug_view == RT_DEBUG_VIEW_HIT_DISTANCE) {
        current_radiance[launch_id.xy] = float4(rt_direct_visualize_hit_distance(surface), 1.0);
        return;
    }

    if (!rt_direct_surface_valid(surface)) {
        current_radiance[launch_id.xy] = float4(rt_direct_primary_miss_background_color(surface), 1.0);
        return;
    }

    RestirDiReservoir reservoir = direct_reservoirs[index];
    if (rt_direct.restir_di_enabled != 0u && rt_direct.debug_view == RT_DEBUG_VIEW_DIRECT_RESERVOIR) {
        current_radiance[launch_id.xy] = float4(rt_direct_visualize_reservoir_weight(reservoir), 1.0);
        return;
    }

    RestirGiReservoir indirect_reservoir = indirect_reservoirs[index];
    if (rt_direct.restir_gi_enabled != 0u && rt_direct.debug_view == RT_DEBUG_VIEW_INDIRECT_RESERVOIR) {
        current_radiance[launch_id.xy] = float4(rt_direct_visualize_indirect_reservoir_weight(indirect_reservoir), 1.0);
        return;
    }

    if (rt_direct.debug_view == RT_DEBUG_VIEW_GI_REASON) {
        current_radiance[launch_id.xy] = float4(rt_direct_visualize_gi_reason(surface, indirect_reservoir), 1.0);
        return;
    }

    if (rt_direct.debug_view == RT_DEBUG_VIEW_GI_INDIRECT) {
        current_radiance[launch_id.xy] = float4(rt_direct_resolve_indirect_reservoir(surface, indirect_reservoir), 1.0);
        return;
    }

    if (rt_direct.restir_gi_enabled != 0u && rt_direct.debug_view == RT_DEBUG_VIEW_GI_TEMPORAL) {
        current_radiance[launch_id.xy] = float4(rt_direct_visualize_gi_temporal_reason(indirect_reservoir), 1.0);
        return;
    }

    if (rt_direct.restir_gi_enabled != 0u && rt_direct.debug_view == RT_DEBUG_VIEW_GI_SPATIAL) {
        current_radiance[launch_id.xy] = float4(rt_direct_visualize_gi_spatial_reason(indirect_reservoir), 1.0);
        return;
    }

    float3 sky_indirect = rt_direct_sample_sky_indirect(surface);
    uint rng_state = rt_direct_sun_rng_seed(launch_id.xy);
    float3 analytic_direct = rt_direct_analytic_sun_direct(surface, rng_state);
    float3 direct = float3(0.0);
    if (rt_direct.restir_di_enabled != 0u) {
        bool visible = rt_direct_light_visible(surface, reservoir);
        direct = rt_direct_resolve_reservoir(surface, reservoir, visible);
    }

    if (rt_direct.debug_view == RT_DEBUG_VIEW_DIRECT) {
        current_radiance[launch_id.xy] = float4(analytic_direct + direct, 1.0);
        return;
    }

    if (rt_direct.debug_view == RT_DEBUG_VIEW_SKY_INDIRECT) {
        current_radiance[launch_id.xy] = float4(sky_indirect, 1.0);
        return;
    }

    float3 primary_emissive = rt_direct_primary_emissive(surface);
    float3 indirect = rt_direct.restir_gi_enabled != 0u
        ? rt_direct_resolve_indirect_reservoir(surface, indirect_reservoir)
        : sky_indirect;

    current_radiance[launch_id.xy] =
        float4(primary_emissive + analytic_direct + direct + indirect, 1.0);
}
