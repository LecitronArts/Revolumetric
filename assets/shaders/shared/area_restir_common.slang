#pragma once

static const uint AREA_RESTIR_DEBUG_OFF = 0u;
static const uint AREA_RESTIR_DEBUG_SUBPIXEL = 1u;
static const uint AREA_RESTIR_DEBUG_LENS = 2u;
static const uint AREA_RESTIR_DEBUG_WEIGHT = 3u;
static const uint AREA_RESTIR_DEBUG_HISTORY_VALID = 4u;
static const uint AREA_RESTIR_DEBUG_REJECTION = 5u;
static const uint AREA_RESTIR_DEBUG_JACOBIAN = 6u;

static const uint AREA_RESTIR_SAMPLE_VALID = 1u << 0;
static const uint AREA_RESTIR_SAMPLE_HISTORY_VALID = 1u << 1;

struct AreaRestirUniforms {
    uint enabled;
    uint temporal_enabled;
    uint spatial_enabled;
    uint subpixel_enabled;
    uint lens_enabled;
    uint initial_candidate_count;
    uint spatial_sample_count;
    uint history_length;
    uint frame_index;
    uint reservoir_count;
    uint width;
    uint height;
    float normal_threshold;
    float depth_threshold;
    float spatial_radius;
    uint debug_view;
};

struct AreaRestirSampleState {
    float2 subpixel_uv;
    float2 lens_uv;
    uint path_sample;
    uint flags;
};

struct AreaRestirEvalContext {
    float4 position_depth;
    float4 normal_roughness;
    float4 albedo_material;
    float4 motion_history;
};

struct AreaRestirReservoir {
    AreaRestirSampleState sample_state;
    uint sample_count_m;
    uint pad0;
    float weight_sum;
    float target_pdf;
    float selected_weight;
    float confidence;
    float jacobian;
    float contribution_luma;
    uint rejection_reason;
    uint debug_flags;
};

AreaRestirReservoir area_restir_invalid_reservoir() {
    AreaRestirReservoir reservoir = (AreaRestirReservoir)0;
    reservoir.sample_state.subpixel_uv = float2(-1.0, -1.0);
    reservoir.sample_state.lens_uv = float2(-1.0, -1.0);
    reservoir.sample_state.path_sample = 0xffffffffu;
    reservoir.sample_state.flags = 0u;
    reservoir.sample_count_m = 0u;
    reservoir.target_pdf = 0.0;
    return reservoir;
}

bool area_restir_is_valid_reservoir(AreaRestirReservoir reservoir) {
    return (reservoir.sample_state.flags & AREA_RESTIR_SAMPLE_VALID) != 0u
        && reservoir.sample_count_m > 0u
        && reservoir.target_pdf > 0.0
        && all(reservoir.sample_state.subpixel_uv >= 0.0)
        && all(reservoir.sample_state.subpixel_uv < 1.0)
        && all(reservoir.sample_state.lens_uv >= 0.0)
        && all(reservoir.sample_state.lens_uv < 1.0);
}

bool area_restir_surface_compatible(
    AreaRestirEvalContext center,
    AreaRestirEvalContext candidate,
    float normal_threshold,
    float depth_threshold
) {
    if (center.position_depth.w < 0.0 || candidate.position_depth.w < 0.0) {
        return false;
    }
    if (abs(center.albedo_material.w - candidate.albedo_material.w) > 0.5) {
        return false;
    }
    float normal_dot = dot(normalize(center.normal_roughness.xyz), normalize(candidate.normal_roughness.xyz));
    if (normal_dot < normal_threshold) {
        return false;
    }
    float position_delta = distance(center.position_depth.xyz, candidate.position_depth.xyz);
    float depth_scale = max(center.position_depth.w, 1.0);
    return position_delta <= max(1.0, depth_scale * depth_threshold);
}

bool area_restir_candidate_finite(float value) {
    return value > 0.0 && value < 1.0e20 && value == value;
}

float area_restir_reservoir_stream_weight(AreaRestirReservoir reservoir) {
    if (!area_restir_is_valid_reservoir(reservoir) || !area_restir_candidate_finite(reservoir.weight_sum)) {
        return 0.0;
    }
    float sample_count = float(max(reservoir.sample_count_m, 1u));
    return min(reservoir.weight_sum / sample_count, 1.0e4);
}

float area_restir_reservoir_reuse_weight(AreaRestirReservoir reservoir) {
    if (!area_restir_is_valid_reservoir(reservoir) || !area_restir_candidate_finite(reservoir.selected_weight)) {
        return 0.0;
    }
    return min(max(reservoir.selected_weight, 0.0), 1.0e4);
}

float area_restir_context_target_luma(AreaRestirEvalContext context) {
    if (context.position_depth.w < 0.0) {
        return 0.0;
    }
    float albedo_luma = dot(max(context.albedo_material.rgb, float3(0.0)), float3(0.2126, 0.7152, 0.0722));
    float emissive_hint = max(context.normal_roughness.w, 0.0);
    return max(albedo_luma + emissive_hint, 1.0e-4);
}

float area_restir_sample_domain_pdf(AreaRestirSampleState sample_state) {
    if ((sample_state.flags & AREA_RESTIR_SAMPLE_VALID) == 0u) {
        return 0.0;
    }
    if (any(sample_state.subpixel_uv < 0.0) || any(sample_state.subpixel_uv >= 1.0)) {
        return 0.0;
    }
    if (any(sample_state.lens_uv < 0.0) || any(sample_state.lens_uv >= 1.0)) {
        return 0.0;
    }
    return 1.0;
}

float area_restir_replay_target_pdf(
    AreaRestirEvalContext current_context,
    AreaRestirSampleState sample_state
) {
    float domain_pdf = area_restir_sample_domain_pdf(sample_state);
    if (domain_pdf <= 0.0) {
        return 0.0;
    }
    return area_restir_context_target_luma(current_context) * domain_pdf;
}

float2 area_restir_pixel_sample(uint2 pixel, AreaRestirSampleState sample_state) {
    return float2(pixel) + sample_state.subpixel_uv - 0.5;
}

void area_restir_reservoir_update(
    inout AreaRestirReservoir reservoir,
    AreaRestirReservoir candidate,
    float candidate_target_pdf,
    float candidate_stream_weight,
    uint candidate_m,
    inout uint rng_state
) {
    if (!area_restir_candidate_finite(candidate_target_pdf) || !area_restir_candidate_finite(candidate_stream_weight)) {
        return;
    }
    float candidate_weight_sum = candidate_stream_weight;
    float next_weight_sum = reservoir.weight_sum + candidate_weight_sum;
    if (!area_restir_candidate_finite(next_weight_sum)) {
        return;
    }

    uint previous_m = reservoir.sample_count_m;
    bool choose_candidate = !area_restir_is_valid_reservoir(reservoir);
    if (!choose_candidate) {
        rng_state ^= rng_state >> 16;
        rng_state *= 0x7feb352du;
        rng_state ^= rng_state >> 15;
        float random01 = float(rng_state & 0x00ffffffu) / 16777216.0;
        choose_candidate = random01 * next_weight_sum <= candidate_weight_sum;
    }
    if (choose_candidate) {
        uint merged_m = previous_m + max(candidate_m, 1u);
        reservoir = candidate;
        reservoir.target_pdf = candidate_target_pdf;
        reservoir.sample_count_m = merged_m;
        reservoir.selected_weight = candidate_weight_sum / max(candidate_target_pdf * float(max(candidate_m, 1u)), 1.0e-4);
    } else {
        reservoir.sample_count_m = previous_m + max(candidate_m, 1u);
    }

    reservoir.weight_sum = next_weight_sum;
    reservoir.contribution_luma = candidate_target_pdf;
    reservoir.selected_weight = next_weight_sum / max(reservoir.target_pdf * float(max(reservoir.sample_count_m, 1u)), 1.0e-4);
}

AreaRestirSampleState area_restir_sample_pixel_state(
    uint2 pixel,
    float2 subpixel_uv,
    float2 lens_uv,
    uint path_sample
) {
    AreaRestirSampleState state;
    state.subpixel_uv = subpixel_uv;
    state.lens_uv = lens_uv;
    state.path_sample = path_sample;
    state.flags = AREA_RESTIR_SAMPLE_VALID;
    return state;
}

void area_restir_finalize_reservoir(inout AreaRestirReservoir reservoir, uint history_length) {
    uint original_m_u = max(reservoir.sample_count_m, 1u);
    float original_m = float(original_m_u);
    uint capped_m_u = min(original_m_u, max(history_length, 1u));
    float capped_m = float(capped_m_u);
    if (capped_m < original_m && area_restir_candidate_finite(reservoir.weight_sum)) {
        float keep_ratio = capped_m / original_m;
        reservoir.weight_sum *= keep_ratio;
    }
    reservoir.sample_count_m = capped_m_u;
    if (reservoir.target_pdf > 0.0 && reservoir.sample_count_m > 0u) {
        reservoir.selected_weight = reservoir.weight_sum / max(reservoir.target_pdf * float(reservoir.sample_count_m), 1.0e-4);
        reservoir.confidence = min(reservoir.confidence, float(history_length));
    } else {
        reservoir = area_restir_invalid_reservoir();
    }
}
