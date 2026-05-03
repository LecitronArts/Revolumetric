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
    float4 sampling;
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

float restir_di_albedo_luma(float3 albedo) {
    return max(dot(albedo, float3(0.2126, 0.7152, 0.0722)), 0.05);
}

float restir_di_emissive_distance_attenuation(float distance_sq) {
    return min(1.0, 8.0 / max(distance_sq, 1.0));
}

float restir_di_emissive_area_pdf(float area_radius) {
    float radius = max(area_radius, 0.5);
    return 1.0 / (4.0 * 3.14159265359 * radius * radius);
}

float3 restir_di_sample_emissive_area_point(float3 center, float area_radius, float2 random_uv) {
    float z = 1.0 - 2.0 * random_uv.x;
    float phi = 6.28318530718 * random_uv.y;
    float r = sqrt(max(1.0 - z * z, 0.0));
    float3 sphere_dir = float3(cos(phi) * r, sin(phi) * r, z);
    return center + sphere_dir * max(area_radius, 0.5);
}

float restir_di_sun_target_pdf(float3 surface_normal, float3 albedo, float3 light_dir, float light_power) {
    float albedo_luma = restir_di_albedo_luma(albedo);
    float light_term = max(dot(surface_normal, light_dir), 0.0);
    return light_power * albedo_luma * light_term;
}

float restir_di_emissive_target_pdf(
    float3 surface_position,
    float3 surface_normal,
    float3 albedo,
    float3 light_position,
    float light_power
) {
    float3 to_light = light_position - surface_position;
    float distance_sq = max(dot(to_light, to_light), 1.0);
    float3 light_dir = to_light * rsqrt(distance_sq);
    float light_term = max(dot(surface_normal, light_dir), 0.0);
    float attenuation = restir_di_emissive_distance_attenuation(distance_sq);
    float albedo_luma = restir_di_albedo_luma(albedo);
    return light_power * albedo_luma * light_term * attenuation;
}

float restir_di_emissive_geometry_term(
    float3 surface_position,
    float3 surface_normal,
    float3 light_position
) {
    float3 to_light = light_position - surface_position;
    float distance_sq = max(dot(to_light, to_light), 1.0);
    float3 light_dir = to_light * rsqrt(distance_sq);
    float light_term = max(dot(surface_normal, light_dir), 0.0);
    return light_term * restir_di_emissive_distance_attenuation(distance_sq);
}

float restir_di_target_pdf_for_light(
    float4 surface_position_depth,
    float4 surface_normal_roughness,
    float4 surface_albedo_material,
    DirectLight light
) {
    float3 surface_normal = normalize(surface_normal_roughness.xyz);
    float light_power = max(light.color_power.w, 0.0001);

    if (light.normal_type.w == 0.0) {
        float3 light_dir = normalize(light.position_radius.xyz);
        return restir_di_sun_target_pdf(
            surface_normal,
            surface_albedo_material.rgb,
            light_dir,
            light_power
        );
    }

    return restir_di_emissive_target_pdf(
        surface_position_depth.xyz,
        surface_normal,
        surface_albedo_material.rgb,
        light.position_radius.xyz,
        light_power
    );
}

float restir_di_target_pdf_for_light_sample(
    float4 surface_position_depth,
    float4 surface_normal_roughness,
    float4 surface_albedo_material,
    DirectLight light,
    float3 light_sample_position
) {
    if (light.normal_type.w == 0.0) {
        return restir_di_target_pdf_for_light(
            surface_position_depth,
            surface_normal_roughness,
            surface_albedo_material,
            light
        );
    }

    float3 surface_normal = normalize(surface_normal_roughness.xyz);
    return restir_di_emissive_target_pdf(
        surface_position_depth.xyz,
        surface_normal,
        surface_albedo_material.rgb,
        light_sample_position,
        max(light.color_power.w, 0.0001)
    );
}

float restir_di_target_pdf_for_reservoir(
    float4 surface_position_depth,
    float4 surface_normal_roughness,
    float4 surface_albedo_material,
    RestirDiReservoir reservoir
) {
    if (reservoir.sample_light_id == 0xffffffffu) {
        return 0.0;
    }

    float3 surface_normal = normalize(surface_normal_roughness.xyz);
    float light_power = max(reservoir.sample_radiance.w, 0.0001);

    if (reservoir.sample_flags == 0u) {
        float3 light_dir = normalize(reservoir.sample_position_pdf.xyz);
        return restir_di_sun_target_pdf(
            surface_normal,
            surface_albedo_material.rgb,
            light_dir,
            light_power
        );
    }

    return restir_di_emissive_target_pdf(
        surface_position_depth.xyz,
        surface_normal,
        surface_albedo_material.rgb,
        reservoir.sample_position_pdf.xyz,
        light_power
    );
}

bool restir_di_is_valid_reservoir(RestirDiReservoir reservoir) {
    return reservoir.sample_light_id != 0xffffffffu
        && reservoir.sample_count_m > 0u
        && reservoir.selected_weight > 0.0;
}

float restir_di_reservoir_stream_weight(
    float target_pdf,
    RestirDiReservoir reservoir,
    uint sample_count_m
) {
    if (!restir_di_is_valid_reservoir(reservoir) || target_pdf <= 0.0 || sample_count_m == 0u) {
        return 0.0;
    }

    return target_pdf * max(reservoir.selected_weight, 0.0) * float(sample_count_m);
}

void restir_di_finalize_reservoir_on_surface_with_target(
    inout RestirDiReservoir reservoir,
    float selected_target_pdf,
    float combined_weight_sum,
    uint combined_sample_count_m
) {
    if (reservoir.sample_light_id == 0xffffffffu || combined_sample_count_m == 0u) {
        reservoir.sample_light_id = 0xffffffffu;
        reservoir.sample_count_m = 0u;
        reservoir.weight_sum = 0.0;
        reservoir.target_pdf = 0.0;
        reservoir.selected_weight = 0.0;
        reservoir.confidence = 0.0;
        return;
    }

    reservoir.target_pdf = selected_target_pdf;
    if (reservoir.target_pdf <= 0.0 || combined_weight_sum <= 0.0) {
        reservoir.sample_light_id = 0xffffffffu;
        reservoir.sample_count_m = 0u;
        reservoir.weight_sum = 0.0;
        reservoir.target_pdf = 0.0;
        reservoir.selected_weight = 0.0;
        reservoir.confidence = 0.0;
        return;
    }
    reservoir.weight_sum = combined_weight_sum;
    reservoir.sample_count_m = combined_sample_count_m;
    reservoir.selected_weight = combined_weight_sum / max(reservoir.target_pdf * float(combined_sample_count_m), 1.0e-4);
    reservoir.confidence = float(combined_sample_count_m);
}

void restir_di_finalize_reservoir_on_surface(
    inout RestirDiReservoir reservoir,
    float4 surface_position_depth,
    float4 surface_normal_roughness,
    float4 surface_albedo_material,
    float combined_weight_sum,
    uint combined_sample_count_m
) {
    float selected_target_pdf = restir_di_target_pdf_for_reservoir(
        surface_position_depth,
        surface_normal_roughness,
        surface_albedo_material,
        reservoir
    );
    restir_di_finalize_reservoir_on_surface_with_target(
        reservoir,
        selected_target_pdf,
        combined_weight_sum,
        combined_sample_count_m
    );
}
