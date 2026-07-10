#include "restir_di_common.slang"
#include "rt_history_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(1, 0)]]
StructuredBuffer<DirectLight> direct_lights;

[[vk::binding(2, 0)]]
StructuredBuffer<RtSurfacePixel> surface_pixels;

[[vk::binding(3, 0)]]
StructuredBuffer<RestirDiReservoir> history_reservoirs;

[[vk::binding(4, 0)]]
RWStructuredBuffer<RestirDiReservoir> output_reservoirs;

[[vk::binding(5, 0)]]
StructuredBuffer<RtSurfacePixel> previous_surface_history;

[[vk::binding(6, 0)]]
RWStructuredBuffer<RtSurfacePixel> current_surface_history;

[[vk::binding(7, 0)]]
ConstantBuffer<RtHistoryUniforms> rt_history;

static const int2 rt_restir_temporal_tap_offsets[4] = {
    int2(0, 0),
    int2(1, 0),
    int2(0, 1),
    int2(1, 1),
};

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

RestirDiReservoir invalid_reservoir() {
    RestirDiReservoir reservoir = (RestirDiReservoir)0;
    reservoir.sample_light_id = 0xffffffffu;
    reservoir.sample_count_m = 0u;
    return reservoir;
}

uint sample_direct_light_id(float random01) {
    uint lo = 0u;
    uint hi = restir.light_count;
    [loop]
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (random01 <= direct_lights[mid].sampling.x) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    return min(lo, restir.light_count - 1u);
}

bool rt_restir_surface_valid(RtSurfacePixel surface) {
    if (surface.hit_kind != RT_SURFACE_HIT_KIND_VOXEL) {
        return false;
    }
    return surface.position_depth.w >= 0.0 && surface.history_confidence > 0.0;
}

bool rt_restir_reproject(float3 world_position, out float2 previous_uv) {
    float4 previous_clip = mul(float4(world_position, 1.0), rt_history.previous_view_proj);
    if (previous_clip.w <= 1.0e-6) {
        previous_uv = float2(0.0);
        return false;
    }

    float2 previous_ndc = previous_clip.xy / previous_clip.w;
    previous_uv = previous_ndc * float2(0.5, -0.5) + float2(0.5, 0.5);
    return true;
}

bool rt_restir_previous_uv_inside(float2 previous_uv) {
    return previous_uv.x >= 0.0
        && previous_uv.x < 1.0
        && previous_uv.y >= 0.0
        && previous_uv.y < 1.0;
}

float rt_restir_temporal_tap_weight(uint tap, float2 history_fraction) {
    int2 offset = rt_restir_temporal_tap_offsets[tap];
    float wx = offset.x == 0 ? 1.0 - history_fraction.x : history_fraction.x;
    float wy = offset.y == 0 ? 1.0 - history_fraction.y : history_fraction.y;
    return wx * wy;
}

bool rt_restir_temporal_tap_inside(int2 previous_pixel, uint2 extent) {
    return previous_pixel.x >= 0
        && previous_pixel.y >= 0
        && previous_pixel.x < int(extent.x)
        && previous_pixel.y < int(extent.y);
}

bool rt_restir_surfaces_compatible(RtSurfacePixel current, RtSurfacePixel previous) {
    if (!rt_restir_surface_valid(current) || !rt_restir_surface_valid(previous)) {
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

RestirDiReservoir rt_restir_generate_initial_reservoir(
    RtSurfacePixel surface,
    uint2 pixel
) {
    RestirDiReservoir reservoir = invalid_reservoir();
    uint rng_state = hash_u32(
        pixel.x * 1973u ^
        pixel.y * 9277u ^
        restir.frame_index * 26699u
    );
    uint candidate_count = clamp(restir.initial_candidate_count, 1u, min(restir.light_count, 16u));
    float weight_sum = 0.0;
    float3 surface_view_dir = normalize(-surface.view_direction_background.xyz);

    for (uint candidate = 0u; candidate < candidate_count; candidate++) {
        uint light_id = sample_direct_light_id(rand01(rng_state));
        DirectLight light = direct_lights[light_id];
        float3 sampled_position = light.position_radius.xyz;
        float light_selection_pdf = max(light.sampling.y, 1.0e-8);
        float sample_pdf = light_selection_pdf;
        if (light.normal_type.w != 0.0) {
            sampled_position = restir_di_sample_emissive_area_point(
                light.position_radius.xyz,
                light.position_radius.w,
                float2(rand01(rng_state), rand01(rng_state))
            );
            sample_pdf *= restir_di_emissive_area_pdf(light.position_radius.w);
        }
        float target_pdf = restir_di_target_pdf_for_light_sample(
            surface.position_depth,
            surface.normal_roughness,
            surface.albedo_material,
            surface_view_dir,
            light,
            sampled_position
        );
        float candidate_weight = target_pdf / max(light_selection_pdf, 1.0e-6);
        if (!restir_di_candidate_finite(candidate_weight)) {
            continue;
        }
        float next_weight_sum = weight_sum + candidate_weight;
        if (!restir_di_candidate_finite(next_weight_sum)) {
            continue;
        }

        if (reservoir.sample_light_id == 0xffffffffu || rand01(rng_state) * next_weight_sum <= candidate_weight) {
            reservoir.sample_light_id = light_id;
            reservoir.sample_flags = uint(round(light.normal_type.w));
            reservoir.target_pdf = target_pdf;
            reservoir.selected_weight = 0.0;
            reservoir.confidence = candidate_weight;
            reservoir.sample_position_pdf = float4(sampled_position, sample_pdf);
            reservoir.sample_radiance = light.color_power;
        }

        weight_sum = next_weight_sum;
        reservoir.sample_count_m += 1u;
    }

    reservoir.weight_sum = weight_sum;
    if (reservoir.sample_light_id != 0xffffffffu && reservoir.target_pdf > 0.0 && reservoir.sample_count_m > 0u) {
        reservoir.weight_sum = min(reservoir.weight_sum, RESTIR_DI_MAX_STREAM_WEIGHT);
        reservoir.selected_weight = min(
            reservoir.weight_sum / max(reservoir.target_pdf * float(reservoir.sample_count_m), 1.0e-4),
            RESTIR_DI_MAX_SELECTED_WEIGHT
        );
        reservoir.confidence = reservoir.weight_sum;
    }

    return reservoir;
}

RestirDiReservoir rt_restir_temporal_reuse(
    RestirDiReservoir reservoir,
    RtSurfacePixel surface,
    uint index
) {
    if (rt_history.flags != 0u) {
        return reservoir;
    }

    float2 previous_uv = float2(0.0);
    bool reprojected = rt_restir_reproject(surface.position_depth.xyz, previous_uv);
    bool valid_resolution = rt_history.previous_resolution.x > 0u && rt_history.previous_resolution.y > 0u;
    if (!valid_resolution || !reprojected || !rt_restir_previous_uv_inside(previous_uv)) {
        return reservoir;
    }

    uint history_m_limit = max(restir.history_length, 1u);
    float3 surface_view_dir = normalize(-surface.view_direction_background.xyz);
    uint current_m = min(reservoir.sample_count_m, history_m_limit);
    float current_target_pdf = restir_di_target_pdf_for_reservoir(
        surface.position_depth,
        surface.normal_roughness,
        surface.albedo_material,
        surface_view_dir,
        reservoir
    );
    float current_weight = restir_di_reservoir_stream_weight(
        current_target_pdf,
        reservoir,
        current_m
    );
    float selected_target_pdf = current_target_pdf;

    uint remaining_m = history_m_limit - min(current_m, history_m_limit);
    RestirDiReservoir selected_history = invalid_reservoir();
    float selected_history_target_pdf = 0.0;
    uint capped_history_m = 0u;
    float history_tap_weight_sum = 0.0;
    float history_tap_selection_sum = 0.0;
    uint rng_state = hash_u32(index * 1973u ^ restir.frame_index);

    if (remaining_m > 0u) {
        uint2 previous_extent = rt_history.previous_resolution;
        float2 previous_sample = previous_uv * float2(previous_extent) - float2(0.5);
        int2 previous_base_pixel = int2(floor(previous_sample));
        float2 history_fraction = saturate(previous_sample - float2(previous_base_pixel));
        for (uint tap = 0u; tap < 4u; tap++) {
            float tap_weight = rt_restir_temporal_tap_weight(tap, history_fraction);
            if (tap_weight <= 0.0) {
                continue;
            }

            int2 tap_pixel_i = previous_base_pixel + rt_restir_temporal_tap_offsets[tap];
            if (!rt_restir_temporal_tap_inside(tap_pixel_i, previous_extent)) {
                continue;
            }

            uint2 previous_pixel = uint2(tap_pixel_i);
            uint previous_index = previous_pixel.y * previous_extent.x + previous_pixel.x;
            RtSurfacePixel previous_surface = previous_surface_history[previous_index];
            if (!rt_restir_surfaces_compatible(surface, previous_surface)) {
                continue;
            }

            RestirDiReservoir history_candidate = history_reservoirs[previous_index];
            if (!restir_di_is_valid_reservoir(history_candidate)) {
                continue;
            }

            uint candidate_m = min(history_candidate.sample_count_m, remaining_m);
            float candidate_target_pdf = restir_di_target_pdf_for_reservoir(
                surface.position_depth,
                surface.normal_roughness,
                surface.albedo_material,
                surface_view_dir,
                history_candidate
            );
            float candidate_stream_weight = restir_di_reservoir_stream_weight(
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
            if (rand01(rng_state) * max(next_selection_sum, 1.0e-4) <= history_candidate_weight) {
                selected_history = history_reservoirs[previous_index];
                selected_history_target_pdf = candidate_target_pdf;
                capped_history_m = candidate_m;
            }
            history_tap_selection_sum = next_selection_sum;
        }
    }

    float history_target_pdf = selected_history_target_pdf;
    float history_weight = history_tap_weight_sum;
    float combined_weight_sum = current_weight + history_weight;

    if (history_weight > 0.0
        && rand01(rng_state) * max(combined_weight_sum, 1.0e-4) <= history_weight) {
        reservoir = selected_history;
        reservoir.target_pdf = history_target_pdf;
        selected_target_pdf = history_target_pdf;
    }

    uint combined_m = min(current_m + capped_history_m, history_m_limit);
    restir_di_finalize_reservoir_on_surface_with_target(
        reservoir,
        selected_target_pdf,
        combined_weight_sum,
        combined_m
    );
    return reservoir;
}

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    uint index = launch_id.y * restir.width + launch_id.x;
    if (launch_id.x >= restir.width || launch_id.y >= restir.height || index >= restir.reservoir_count) return;

    RtSurfacePixel surface = surface_pixels[index];
    current_surface_history[index] = surface;
    if (restir.light_count == 0u || !rt_restir_surface_valid(surface)) {
        output_reservoirs[index] = invalid_reservoir();
        return;
    }

    RestirDiReservoir reservoir = rt_restir_generate_initial_reservoir(surface, launch_id.xy);
    if (restir.temporal_enabled != 0u) {
        reservoir = rt_restir_temporal_reuse(reservoir, surface, index);
    }

    if (reservoir.sample_light_id != 0xffffffffu && reservoir.sample_count_m > 0u && reservoir.target_pdf <= 0.0) {
        float3 surface_view_dir = normalize(-surface.view_direction_background.xyz);
        reservoir.target_pdf = restir_di_target_pdf_for_reservoir(
            surface.position_depth,
            surface.normal_roughness,
            surface.albedo_material,
            surface_view_dir,
            reservoir
        );
    }
    output_reservoirs[index] = reservoir;
}
