static const float RESTIR_GI_MAX_SELECTED_WEIGHT = 32.0;
static const float RESTIR_GI_MAX_STREAM_WEIGHT = 1.0e4;
static const float RESTIR_GI_MAX_SAMPLE_RADIANCE_LUMA = 64.0;
static const float RESTIR_GI_MAX_RESOLVED_RADIANCE_LUMA = 64.0;
static const float RESTIR_GI_ENVIRONMENT_SAMPLE_DEPTH = 1.0e20;

static const uint RESTIR_GI_DEBUG_VIEW_OFF = 0u;
static const uint RESTIR_GI_DEBUG_VIEW_RESERVOIR_WEIGHT = 1u;
static const uint RESTIR_GI_DEBUG_VIEW_TEMPORAL_VALID = 2u;
static const uint RESTIR_GI_DEBUG_VIEW_SPATIAL_VALID = 3u;

static const uint RESTIR_GI_TEMPORAL_REUSE_ACCEPTED = 1u;
static const uint RESTIR_GI_TEMPORAL_REUSE_RESET_OR_DISABLED = 2u;
static const uint RESTIR_GI_TEMPORAL_REUSE_INVALID_RESOLUTION = 3u;
static const uint RESTIR_GI_TEMPORAL_REUSE_REPROJECTION_FAILED = 4u;
static const uint RESTIR_GI_TEMPORAL_REUSE_PREVIOUS_UV_OUTSIDE = 5u;
static const uint RESTIR_GI_TEMPORAL_REUSE_SURFACE_INCOMPATIBLE = 6u;
static const uint RESTIR_GI_TEMPORAL_REUSE_HISTORY_RESERVOIR_INVALID = 7u;
static const uint RESTIR_GI_TEMPORAL_REUSE_CURRENT_WEIGHT_ZERO = 8u;
static const uint RESTIR_GI_TEMPORAL_REUSE_HISTORY_WEIGHT_ZERO = 9u;
static const uint RESTIR_GI_TEMPORAL_REUSE_COMBINED_WEIGHT_ZERO = 10u;

static const uint RESTIR_GI_SPATIAL_REUSE_ACCEPTED_CENTER = 1u;
static const uint RESTIR_GI_SPATIAL_REUSE_DISABLED_OR_PASSTHROUGH = 2u;
static const uint RESTIR_GI_SPATIAL_REUSE_INVALID_SURFACE = 3u;
static const uint RESTIR_GI_SPATIAL_REUSE_CENTER_RESERVOIR_INVALID = 4u;
static const uint RESTIR_GI_SPATIAL_REUSE_NO_COMPATIBLE_NEIGHBOR = 5u;
static const uint RESTIR_GI_SPATIAL_REUSE_ACCEPTED_NEIGHBOR = 6u;
static const uint RESTIR_GI_SPATIAL_REUSE_VISIBILITY_REJECTED = 7u;

struct RestirGiUniforms {
    uint enabled;
    uint temporal_enabled;
    uint max_bounces;
    uint debug_view;
    uint initial_candidate_count;
    uint history_length;
    uint frame_index;
    uint reservoir_count;
    uint width;
    uint height;
    uint spatial_enabled;
    uint spatial_sample_count;
    float4 sky_color_sun_angular_radius;
    float4 ground_color_pad;
    float4 sun_direction_pad;
    float4 sun_intensity_pad;
};

struct RestirGiReservoir {
    float4 sample_position_depth;
    float4 sample_normal_roughness;
    float4 sample_radiance_pdf;
    float target_pdf;
    float weight_sum;
    float selected_weight;
    uint sample_count_m;
};

RestirGiReservoir restir_gi_invalid_reservoir() {
    RestirGiReservoir reservoir = (RestirGiReservoir)0;
    reservoir.sample_count_m = 0u;
    return reservoir;
}

bool restir_gi_candidate_finite(float value) {
    return value == value && value > 0.0 && value < 1.0e20;
}

bool restir_gi_is_valid_reservoir(RestirGiReservoir reservoir) {
    return reservoir.sample_count_m > 0u
        && restir_gi_candidate_finite(reservoir.target_pdf)
        && restir_gi_candidate_finite(reservoir.weight_sum)
        && restir_gi_candidate_finite(reservoir.selected_weight);
}

bool restir_gi_is_environment_sample(RestirGiReservoir reservoir) {
    return reservoir.sample_position_depth.w >= RESTIR_GI_ENVIRONMENT_SAMPLE_DEPTH * 0.5;
}

float restir_gi_luma(float3 radiance) {
    return max(dot(max(radiance, float3(0.0)), float3(0.2126, 0.7152, 0.0722)), 0.0);
}

float3 restir_gi_clamp_radiance_luma(float3 radiance, float max_luma) {
    float luma = restir_gi_luma(radiance);
    if (!restir_gi_candidate_finite(luma)) {
        return float3(0.0);
    }
    if (luma > max_luma && luma > 1.0e-4) {
        return radiance * (max_luma / luma);
    }
    return max(radiance, float3(0.0));
}

float restir_gi_sample_geometry_term(
    float3 surface_position,
    float3 surface_normal,
    RestirGiReservoir reservoir
) {
    float3 sample_vector = reservoir.sample_position_depth.xyz - surface_position;
    float distance2 = max(dot(sample_vector, sample_vector), 1.0e-6);
    float3 sample_direction = sample_vector * rsqrt(distance2);
    float3 sample_normal_raw = reservoir.sample_normal_roughness.xyz;
    float sample_normal_length2 = max(dot(sample_normal_raw, sample_normal_raw), 1.0e-8);
    float3 sample_normal = sample_normal_raw * rsqrt(sample_normal_length2);
    float receiver_term = max(dot(surface_normal, sample_direction), 0.0);
    float sample_term = max(dot(sample_normal, -sample_direction), 0.0);
    float attenuation = min(1.0, 8.0 / max(distance2, 1.0));
    return receiver_term * sample_term * attenuation;
}

float restir_gi_environment_geometry_term(float3 surface_normal, RestirGiReservoir reservoir) {
    return max(dot(surface_normal, normalize(reservoir.sample_position_depth.xyz)), 0.0);
}

float restir_gi_cosine_sample_visibility(
    float3 surface_position,
    float3 surface_normal,
    RestirGiReservoir reservoir
) {
    if (restir_gi_is_environment_sample(reservoir)) {
        float3 sample_direction = normalize(reservoir.sample_position_depth.xyz);
        return dot(surface_normal, sample_direction) > 0.0 ? 1.0 : 0.0;
    }

    float3 sample_vector = reservoir.sample_position_depth.xyz - surface_position;
    float distance2 = max(dot(sample_vector, sample_vector), 1.0e-6);
    float3 sample_direction = sample_vector * rsqrt(distance2);
    float3 sample_normal_raw = reservoir.sample_normal_roughness.xyz;
    float sample_normal_length2 = max(dot(sample_normal_raw, sample_normal_raw), 1.0e-8);
    float3 sample_normal = sample_normal_raw * rsqrt(sample_normal_length2);
    float receiver_term = dot(surface_normal, sample_direction);
    float sample_term = dot(sample_normal, -sample_direction);
    return receiver_term > 0.0 && sample_term > 0.0 ? 1.0 : 0.0;
}

float3 restir_gi_cosine_sample_contribution(
    float3 surface_albedo,
    float3 surface_position,
    float3 surface_normal,
    RestirGiReservoir reservoir
) {
    float visibility = restir_gi_cosine_sample_visibility(
        surface_position,
        surface_normal,
        reservoir
    );
    if (visibility <= 0.0 || restir_gi_luma(reservoir.sample_radiance_pdf.rgb) <= 0.0) {
        return float3(0.0);
    }

    return surface_albedo * reservoir.sample_radiance_pdf.rgb * visibility;
}

float restir_gi_reservoir_stream_weight(
    float target_pdf,
    RestirGiReservoir reservoir,
    uint sample_count_m
) {
    if (!restir_gi_is_valid_reservoir(reservoir) || target_pdf <= 0.0 || sample_count_m == 0u) {
        return 0.0;
    }

    float stream_weight = target_pdf
        * min(reservoir.selected_weight, RESTIR_GI_MAX_SELECTED_WEIGHT)
        * float(sample_count_m);
    if (!restir_gi_candidate_finite(stream_weight)) {
        return 0.0;
    }
    return min(stream_weight, RESTIR_GI_MAX_STREAM_WEIGHT);
}

void restir_gi_finalize_reservoir(
    inout RestirGiReservoir reservoir,
    float selected_target_pdf,
    float combined_weight_sum,
    uint combined_sample_count_m
) {
    if (combined_sample_count_m == 0u || selected_target_pdf <= 0.0 || combined_weight_sum <= 0.0) {
        reservoir = restir_gi_invalid_reservoir();
        return;
    }

    reservoir.target_pdf = selected_target_pdf;
    reservoir.weight_sum = min(combined_weight_sum, RESTIR_GI_MAX_STREAM_WEIGHT);
    reservoir.sample_count_m = combined_sample_count_m;
    reservoir.selected_weight = min(
        reservoir.weight_sum / max(selected_target_pdf * float(combined_sample_count_m), 1.0e-4),
        RESTIR_GI_MAX_SELECTED_WEIGHT
    );
}
