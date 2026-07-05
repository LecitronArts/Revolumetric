static const float RESTIR_GI_MAX_SELECTED_WEIGHT = 32.0;
static const float RESTIR_GI_MAX_STREAM_WEIGHT = 1.0e4;

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
    uint _pad0;
    uint _pad1;
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

float restir_gi_luma(float3 radiance) {
    return max(dot(max(radiance, float3(0.0)), float3(0.2126, 0.7152, 0.0722)), 0.0);
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
