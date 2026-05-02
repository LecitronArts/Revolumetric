struct RestirDiUniforms {
    uint enabled;
    uint temporal_enabled;
    uint spatial_enabled;
    uint debug_view;
    uint initial_candidate_count;
    uint spatial_sample_count;
    uint history_length;
    uint frame_index;
    uint reservoir_count;
    uint light_count;
    uint width;
    uint height;
};

struct DirectLight {
    float4 position_radius;
    float4 normal_type;
    float4 color_power;
};

struct RestirDiReservoir {
    uint sample_light_id;
    uint sample_flags;
    uint sample_count_m;
    uint pad0;
    float target_pdf;
    float weight_sum;
    float selected_weight;
    float confidence;
    float4 sample_position_pdf;
    float4 sample_radiance;
};
