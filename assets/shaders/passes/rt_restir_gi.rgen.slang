#include "restir_gi_common.slang"
#include "restir_di_common.slang"
#include "rt_history_common.slang"
#include "rt_surface_common.slang"

static const float RT_RESTIR_GI_INV_PI = 0.31830988618;
static const float RT_RESTIR_GI_SURFACE_BIAS = 0.01;

[[vk::binding(0, 0)]]
ConstantBuffer<RestirGiUniforms> restir_gi;

[[vk::binding(1, 0)]]
ConstantBuffer<RtHistoryUniforms> rt_history;

[[vk::binding(2, 0)]]
StructuredBuffer<RtSurfacePixel> surface_pixels;

[[vk::binding(3, 0)]]
StructuredBuffer<RestirGiReservoir> history_reservoirs;

[[vk::binding(4, 0)]]
RWStructuredBuffer<RestirGiReservoir> output_reservoirs;

[[vk::binding(5, 0)]]
StructuredBuffer<RtSurfacePixel> previous_surface_history;

[[vk::binding(6, 0)]]
RWStructuredBuffer<RtSurfacePixel> current_surface_history;

[[vk::binding(7, 0)]]
RaytracingAccelerationStructure scene_tlas;

static const int2 rt_restir_gi_temporal_tap_offsets[4] = {
    int2(0, 0),
    int2(1, 0),
    int2(0, 1),
    int2(1, 1),
};

uint gi_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float gi_rand01(inout uint state) {
    state = gi_hash_u32(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

float3 rt_restir_gi_sample_sun_direction(inout uint rng_state) {
    float sun_radius = max(restir_gi.sky_color_sun_angular_radius.w, 0.0);
    if (sun_radius <= 0.0) {
        return normalize(restir_gi.sun_direction_pad.xyz);
    }

    float cos_min = cos(sun_radius);
    float cos_theta = lerp(cos_min, 1.0, gi_rand01(rng_state));
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    float phi = 6.28318530718 * gi_rand01(rng_state);
    float3 sun_forward = normalize(restir_gi.sun_direction_pad.xyz);
    float3 sun_right = normalize(abs(sun_forward.y) < 0.95 ? cross(sun_forward, float3(0, 1, 0)) : cross(sun_forward, float3(1, 0, 0)));
    float3 sun_up = normalize(cross(sun_right, sun_forward));
    return normalize(
        sun_right * (cos(phi) * sin_theta) +
        sun_up * (sin(phi) * sin_theta) +
        sun_forward * cos_theta
    );
}

bool rt_restir_gi_surface_valid(RtSurfacePixel surface) {
    return surface.hit_kind == RT_SURFACE_HIT_KIND_VOXEL
        && surface.position_depth.w >= 0.0
        && surface.history_confidence > 0.0;
}

float3 rt_restir_gi_sunlit_ground_color() {
    float ground_ndotl = max(restir_gi.sun_direction_pad.y, 0.0);
    float3 sun_irradiance = restir_gi.sun_intensity_pad.rgb * ground_ndotl;
    return restir_gi.ground_color_pad.rgb * (1.0 + RT_RESTIR_GI_INV_PI * sun_irradiance);
}

float3 rt_restir_gi_background_color_for_dir(float3 direction) {
    float3 miss_dir = normalize(direction);
    float t = saturate(miss_dir.y * 0.5 + 0.5);
    return lerp(rt_restir_gi_sunlit_ground_color(), restir_gi.sky_color_sun_angular_radius.rgb, t);
}

float3 rt_restir_gi_sky_visibility_sample(RtSurfacePixel surface, float3 sky_dir) {
    float3 normal = normalize(surface.normal_roughness.xyz);

    RayDesc sky_ray;
    sky_ray.Origin = surface.position_depth.xyz
        + normal * RT_RESTIR_GI_SURFACE_BIAS
        + sky_dir * RT_RESTIR_GI_SURFACE_BIAS;
    sky_ray.Direction = sky_dir;
    sky_ray.TMin = 0.001;
    sky_ray.TMax = RT_SURFACE_MISS_T;

    RtSurfacePayload sky_payload = make_rt_surface_payload(sky_ray.Direction);
    // Sky visibility is a binary occlusion query — accept the first hit and end
    // the search instead of finding the closest occluder through the voxel DDA.
    // The closest-hit shader still runs (it sets hit_kind), so the result is
    // bit-identical. (A fuller optimization would switch to RtShadowPayload +
    // SKIP_CLOSEST_HIT_SHADER; deferred to the payload-slimming pass.)
    TraceRay(scene_tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xffu, 0u, 0u, 0u, sky_ray, sky_payload);
    if (sky_payload.hit_kind == RT_SURFACE_HIT_KIND_VOXEL) {
        return float3(0.0);
    }

    return rt_restir_gi_background_color_for_dir(sky_dir);
}

float3 rt_restir_gi_analytic_sun_direct(RtSurfacePixel surface, inout uint rng_state) {
    float3 sun_vector = restir_gi.sun_direction_pad.xyz;
    if (dot(sun_vector, sun_vector) <= 1.0e-8) {
        return float3(0.0);
    }

    float3 sun_dir = rt_restir_gi_sample_sun_direction(rng_state);
    float3 normal = normalize(surface.normal_roughness.xyz);
    float sun_term = max(dot(normal, sun_dir), 0.0);
    if (sun_term <= 0.0) {
        return float3(0.0);
    }

    RayDesc sun_ray;
    sun_ray.Origin = surface.position_depth.xyz
        + normal * RT_RESTIR_GI_SURFACE_BIAS
        + sun_dir * RT_RESTIR_GI_SURFACE_BIAS;
    sun_ray.Direction = sun_dir;
    sun_ray.TMin = 0.001;
    sun_ray.TMax = RT_SURFACE_MISS_T;

    RtSurfacePayload sun_payload = make_rt_surface_payload(sun_ray.Direction);
    // Sun shadow is a binary occlusion query — accept first hit and end search.
    TraceRay(scene_tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xffu, 0u, 0u, 0u, sun_ray, sun_payload);
    if (sun_payload.hit_kind == RT_SURFACE_HIT_KIND_VOXEL) {
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
    return direct_brdf * restir_gi.sun_intensity_pad.rgb * sun_term;
}

float3 rt_restir_gi_estimate_incoming_radiance(RtSurfacePixel surface, inout uint rng_state) {
    float3 emissive = surface.emissive_radiance.rgb;
    if (restir_gi_luma(emissive) > 0.0) {
        return emissive;
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    float3 sky_dir = normal;
    float3 sky = surface.albedo_material.rgb * rt_restir_gi_sky_visibility_sample(surface, sky_dir);
    return sky + rt_restir_gi_analytic_sun_direct(surface, rng_state);
}

bool rt_restir_gi_reproject(float3 world_position, out float2 previous_uv) {
    float4 previous_clip = mul(float4(world_position, 1.0), rt_history.previous_view_proj);
    if (previous_clip.w <= 1.0e-6) {
        previous_uv = float2(0.0);
        return false;
    }

    float2 previous_ndc = previous_clip.xy / previous_clip.w;
    previous_uv = previous_ndc * float2(0.5, -0.5) + float2(0.5, 0.5);
    return true;
}

bool rt_restir_gi_previous_uv_inside(float2 previous_uv) {
    return previous_uv.x >= 0.0
        && previous_uv.x < 1.0
        && previous_uv.y >= 0.0
        && previous_uv.y < 1.0;
}

float rt_restir_gi_temporal_tap_weight(uint tap, float2 history_fraction) {
    int2 offset = rt_restir_gi_temporal_tap_offsets[tap];
    float wx = offset.x == 0 ? 1.0 - history_fraction.x : history_fraction.x;
    float wy = offset.y == 0 ? 1.0 - history_fraction.y : history_fraction.y;
    return wx * wy;
}

bool rt_restir_gi_temporal_tap_inside(int2 previous_pixel, uint2 extent) {
    return previous_pixel.x >= 0
        && previous_pixel.y >= 0
        && previous_pixel.x < int(extent.x)
        && previous_pixel.y < int(extent.y);
}

bool rt_restir_gi_surfaces_compatible(RtSurfacePixel current, RtSurfacePixel previous) {
    if (!rt_restir_gi_surface_valid(current) || !rt_restir_gi_surface_valid(previous)) {
        return false;
    }
    if (current.material_id != previous.material_id) {
        return false;
    }

    float normal_dot = dot(normalize(current.normal_roughness.xyz), normalize(previous.normal_roughness.xyz));
    if (normal_dot < rt_history.normal_threshold) {
        return false;
    }

    float position_delta = distance(current.position_depth.xyz, previous.position_depth.xyz);
    return position_delta <= rt_history_position_threshold(current.position_depth.w, rt_history.depth_threshold);
}

float rt_restir_gi_target_pdf(RtSurfacePixel surface, RestirGiReservoir reservoir) {
    if (!rt_restir_gi_surface_valid(surface)) {
        return 0.0;
    }
    float3 surface_normal = normalize(surface.normal_roughness.xyz);
    float3 contribution = restir_gi_cosine_sample_contribution(
        surface.albedo_material.rgb,
        surface.position_depth.xyz,
        surface_normal,
        reservoir
    );
    return restir_gi_luma(contribution);
}

bool rt_restir_gi_temporal_debug_enabled() {
    return restir_gi.debug_view == RESTIR_GI_DEBUG_VIEW_TEMPORAL_VALID;
}

RestirGiReservoir rt_restir_gi_record_temporal_debug(RestirGiReservoir reservoir, uint reason) {
    if (rt_restir_gi_temporal_debug_enabled()) {
        reservoir.sample_radiance_pdf.w = float(reason);
    }
    return reservoir;
}

bool rt_restir_gi_spatial_debug_passthrough_enabled() {
    return restir_gi.debug_view == RESTIR_GI_DEBUG_VIEW_SPATIAL_VALID;
}

RestirGiReservoir rt_restir_gi_record_spatial_debug_passthrough(RestirGiReservoir reservoir) {
    if (rt_restir_gi_spatial_debug_passthrough_enabled()) {
        reservoir.sample_radiance_pdf.w = float(RESTIR_GI_SPATIAL_REUSE_DISABLED_OR_PASSTHROUGH);
    }
    return reservoir;
}

float3 rt_restir_gi_orthonormal_tangent(float3 normal) {
    float3 up = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    return normalize(cross(up, normal));
}

float3 rt_restir_gi_sample_indirect_direction(float3 normal, inout uint rng_state) {
    float u0 = gi_rand01(rng_state);
    float u1 = gi_rand01(rng_state);
    float phi = 6.28318530718 * u0;
    float cos_theta = sqrt(max(1.0 - u1, 0.0));
    float sin_theta = sqrt(max(u1, 0.0));
    float3 tangent = rt_restir_gi_orthonormal_tangent(normal);
    float3 bitangent = cross(normal, tangent);
    return normalize(tangent * cos(phi) * sin_theta + bitangent * sin(phi) * sin_theta + normal * cos_theta);
}

RtSurfacePixel rt_restir_gi_trace_indirect_surface(RtSurfacePixel surface, float3 indirect_direction) {
    if (!rt_restir_gi_surface_valid(surface)) {
        return surface_pixel_from_payload(make_rt_surface_payload(indirect_direction));
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    RayDesc ray;
    ray.Origin = surface.position_depth.xyz + normal * 0.01 + indirect_direction * 0.01;
    ray.Direction = indirect_direction;
    ray.TMin = 0.001;
    ray.TMax = RT_SURFACE_MISS_T;

    RtSurfacePayload payload = make_rt_surface_payload(indirect_direction);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);
    return surface_pixel_from_payload(payload);
}

RestirGiReservoir rt_restir_gi_generate_initial_reservoir(
    RtSurfacePixel surface,
    uint index,
    uint2 pixel
) {
    RestirGiReservoir reservoir = restir_gi_invalid_reservoir();
    if (!rt_restir_gi_surface_valid(surface)) {
        return reservoir;
    }

    uint rng_state = gi_hash_u32(index * 1973u ^ pixel.x * 9277u ^ pixel.y * 26699u ^ restir_gi.frame_index);
    uint candidate_count = clamp(restir_gi.initial_candidate_count, 1u, 16u);
    float3 normal = normalize(surface.normal_roughness.xyz);
    float weight_sum = 0.0;
    float selected_target_pdf = 0.0;
    uint accepted_count = 0u;

    for (uint candidate_index = 0u; candidate_index < candidate_count; candidate_index++) {
        float3 indirect_direction = rt_restir_gi_sample_indirect_direction(normal, rng_state);
        RtSurfacePixel indirect_surface = rt_restir_gi_trace_indirect_surface(surface, indirect_direction);
        RestirGiReservoir candidate = rt_restir_gi_make_initial_candidate(
            surface,
            indirect_surface,
            indirect_direction,
            rng_state
        );
        if (!restir_gi_is_valid_reservoir(candidate)) {
            continue;
        }

        candidate.sample_radiance_pdf.rgb = restir_gi_clamp_radiance_luma(
            candidate.sample_radiance_pdf.rgb,
            RESTIR_GI_MAX_SAMPLE_RADIANCE_LUMA
        );
        float target_pdf = rt_restir_gi_target_pdf(surface, candidate);
        if (target_pdf <= 0.0) {
            continue;
        }
        float candidate_weight = target_pdf;
        if (!restir_gi_candidate_finite(candidate_weight)) {
            continue;
        }

        float next_weight_sum = weight_sum + candidate_weight;
        if (!restir_gi_candidate_finite(next_weight_sum)) {
            continue;
        }

        if (accepted_count == 0u || gi_rand01(rng_state) * next_weight_sum <= candidate_weight) {
            reservoir = candidate;
            selected_target_pdf = target_pdf;
        }

        weight_sum = next_weight_sum;
        accepted_count += 1u;
    }

    restir_gi_finalize_reservoir(reservoir, selected_target_pdf, weight_sum, accepted_count);
    return reservoir;
}

RestirGiReservoir rt_restir_gi_make_initial_candidate(
    RtSurfacePixel surface,
    RtSurfacePixel indirect_surface,
    float3 indirect_direction,
    inout uint rng_state
) {
    RestirGiReservoir candidate = restir_gi_invalid_reservoir();
    if (!rt_restir_gi_surface_valid(surface)) {
        return candidate;
    }
    if (indirect_surface.hit_kind == RT_SURFACE_HIT_KIND_MISS) {
        return rt_restir_gi_make_environment_candidate(surface, indirect_direction);
    }
    if (!rt_restir_gi_surface_valid(indirect_surface)) {
        return candidate;
    }

    float3 sample_position = indirect_surface.position_depth.xyz;
    float3 sample_radiance = rt_restir_gi_estimate_incoming_radiance(indirect_surface, rng_state);
    sample_radiance = restir_gi_clamp_radiance_luma(
        sample_radiance,
        RESTIR_GI_MAX_SAMPLE_RADIANCE_LUMA
    );
    if (restir_gi_luma(sample_radiance) <= 0.0) {
        return candidate;
    }
    candidate.sample_position_depth = float4(sample_position, indirect_surface.position_depth.w);
    candidate.sample_normal_roughness = indirect_surface.normal_roughness;
    candidate.sample_radiance_pdf = float4(sample_radiance, 1.0);
    candidate.target_pdf = rt_restir_gi_target_pdf(surface, candidate);
    candidate.weight_sum = candidate.target_pdf;
    candidate.selected_weight = 1.0;
    candidate.sample_count_m = 1u;
    return candidate;
}

RestirGiReservoir rt_restir_gi_make_environment_candidate(
    RtSurfacePixel surface,
    float3 indirect_direction
) {
    RestirGiReservoir candidate = restir_gi_invalid_reservoir();
    if (!rt_restir_gi_surface_valid(surface)) {
        return candidate;
    }

    float3 environment_radiance = rt_restir_gi_background_color_for_dir(indirect_direction);
    environment_radiance = restir_gi_clamp_radiance_luma(
        environment_radiance,
        RESTIR_GI_MAX_SAMPLE_RADIANCE_LUMA
    );
    if (restir_gi_luma(environment_radiance) <= 0.0) {
        return candidate;
    }

    candidate.sample_position_depth = float4(indirect_direction, RESTIR_GI_ENVIRONMENT_SAMPLE_DEPTH);
    candidate.sample_normal_roughness = float4(-indirect_direction, 1.0);
    candidate.sample_radiance_pdf = float4(environment_radiance, 1.0);
    candidate.target_pdf = rt_restir_gi_target_pdf(surface, candidate);
    candidate.weight_sum = candidate.target_pdf;
    candidate.selected_weight = 1.0;
    candidate.sample_count_m = 1u;
    return candidate;
}

RestirGiReservoir rt_restir_gi_temporal_reuse(RestirGiReservoir reservoir, RtSurfacePixel surface, uint index) {
    if (rt_history.flags != 0u || restir_gi.temporal_enabled == 0u) {
        return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_RESET_OR_DISABLED);
    }

    float2 previous_uv = float2(0.0);
    bool reprojected = rt_restir_gi_reproject(surface.position_depth.xyz, previous_uv);
    bool valid_resolution = rt_history.previous_resolution.x > 0u && rt_history.previous_resolution.y > 0u;
    if (!valid_resolution) {
        return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_INVALID_RESOLUTION);
    }
    if (!reprojected) {
        return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_REPROJECTION_FAILED);
    }
    if (!rt_restir_gi_previous_uv_inside(previous_uv)) {
        return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_PREVIOUS_UV_OUTSIDE);
    }

    uint history_limit = max(restir_gi.history_length, 1u);
    uint current_m = min(reservoir.sample_count_m, history_limit);
    float current_target_pdf = rt_restir_gi_target_pdf(surface, reservoir);
    float current_weight = restir_gi_reservoir_stream_weight(current_target_pdf, reservoir, current_m);
    uint remaining_m = history_limit - min(current_m, history_limit);
    RestirGiReservoir selected_history = restir_gi_invalid_reservoir();
    float selected_history_target_pdf = 0.0;
    uint history_m = 0u;
    float history_tap_weight_sum = 0.0;
    float history_tap_selection_sum = 0.0;
    bool found_compatible_surface = false;
    bool found_valid_history = false;
    uint rng_state = gi_hash_u32(index * 8191u ^ restir_gi.frame_index);

    if (remaining_m > 0u) {
        uint2 previous_extent = rt_history.previous_resolution;
        float2 previous_sample = previous_uv * float2(previous_extent) - float2(0.5);
        int2 previous_base_pixel = int2(floor(previous_sample));
        float2 history_fraction = saturate(previous_sample - float2(previous_base_pixel));
        for (uint tap = 0u; tap < 4u; tap++) {
            float tap_weight = rt_restir_gi_temporal_tap_weight(tap, history_fraction);
            if (tap_weight <= 0.0) {
                continue;
            }

            int2 tap_pixel_i = previous_base_pixel + rt_restir_gi_temporal_tap_offsets[tap];
            if (!rt_restir_gi_temporal_tap_inside(tap_pixel_i, previous_extent)) {
                continue;
            }

            uint2 previous_pixel = uint2(tap_pixel_i);
            uint previous_index = previous_pixel.y * previous_extent.x + previous_pixel.x;
            RtSurfacePixel previous_surface = previous_surface_history[previous_index];
            if (!rt_restir_gi_surfaces_compatible(surface, previous_surface)) {
                continue;
            }
            found_compatible_surface = true;

            RestirGiReservoir history_candidate = history_reservoirs[previous_index];
            if (!restir_gi_is_valid_reservoir(history_candidate)) {
                continue;
            }
            found_valid_history = true;

            uint candidate_m = min(history_candidate.sample_count_m, remaining_m);
            float candidate_target_pdf = rt_restir_gi_target_pdf(surface, history_candidate);
            float candidate_stream_weight = restir_gi_reservoir_stream_weight(
                candidate_target_pdf,
                history_candidate,
                candidate_m
            );
            float history_candidate_weight = candidate_stream_weight * tap_weight;
            if (history_candidate_weight <= 0.0) {
                continue;
            }

            history_tap_weight_sum += history_candidate_weight;
            float next_selection_sum = history_tap_selection_sum + history_candidate_weight;
            if (gi_rand01(rng_state) * max(next_selection_sum, 1.0e-4) <= history_candidate_weight) {
                selected_history = history_reservoirs[previous_index];
                selected_history_target_pdf = candidate_target_pdf;
                history_m = candidate_m;
            }
            history_tap_selection_sum = next_selection_sum;
        }
    }

    if (remaining_m > 0u && !found_compatible_surface) {
        return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_SURFACE_INCOMPATIBLE);
    }
    if (remaining_m > 0u && !found_valid_history) {
        return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_HISTORY_RESERVOIR_INVALID);
    }

    float history_target_pdf = selected_history_target_pdf;
    float history_weight = history_tap_weight_sum;
    float combined_weight_sum = current_weight + history_weight;
    float selected_target_pdf = current_target_pdf;

    if (rt_restir_gi_temporal_debug_enabled()) {
        if (current_weight <= 0.0) {
            return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_CURRENT_WEIGHT_ZERO);
        }
        if (history_weight <= 0.0) {
            return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_HISTORY_WEIGHT_ZERO);
        }
        if (combined_weight_sum <= 0.0) {
            return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_COMBINED_WEIGHT_ZERO);
        }
    }

    if (history_weight > 0.0 && gi_rand01(rng_state) * max(combined_weight_sum, 1.0e-4) <= history_weight) {
        reservoir = selected_history;
        selected_target_pdf = history_target_pdf;
    }

    uint combined_m = min(current_m + history_m, history_limit);
    restir_gi_finalize_reservoir(reservoir, selected_target_pdf, combined_weight_sum, combined_m);
    return rt_restir_gi_record_temporal_debug(reservoir, RESTIR_GI_TEMPORAL_REUSE_ACCEPTED);
}

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    if (launch_id.x >= restir_gi.width || launch_id.y >= restir_gi.height) return;

    uint index = launch_id.y * restir_gi.width + launch_id.x;
    if (index >= restir_gi.reservoir_count || restir_gi.enabled == 0u) {
        if (index < restir_gi.reservoir_count) {
            output_reservoirs[index] = restir_gi_invalid_reservoir();
        }
        return;
    }

    RtSurfacePixel surface = surface_pixels[index];
    current_surface_history[index] = surface;
    RestirGiReservoir reservoir = rt_restir_gi_generate_initial_reservoir(surface, index, launch_id.xy);
    reservoir = rt_restir_gi_temporal_reuse(reservoir, surface, index);
    reservoir = rt_restir_gi_record_spatial_debug_passthrough(reservoir);
    output_reservoirs[index] = reservoir;
}
