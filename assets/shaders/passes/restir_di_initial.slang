#include "restir_di_common.slang"
#include "vpt_history_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(1, 0)]]
StructuredBuffer<DirectLight> direct_lights;

[[vk::binding(2, 0)]]
RWStructuredBuffer<RestirDiReservoir> output_reservoirs;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> current_surface_position_depth;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> current_surface_normal_roughness;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> current_surface_albedo_material;

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

bool surface_is_valid(uint index) {
    uint2 pixel = uint2(index % restir.width, index / restir.width);
    return current_surface_position_depth[pixel].w >= 0.0;
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

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.y * restir.width + tid.x;
    if (tid.x >= restir.width || tid.y >= restir.height || index >= restir.reservoir_count) return;

    if (restir.light_count == 0u || !surface_is_valid(index)) {
        output_reservoirs[index] = invalid_reservoir();
        return;
    }

    uint2 pixel = tid.xy;
    float4 surface_position_depth = current_surface_position_depth[pixel];
    float4 surface_normal_roughness = current_surface_normal_roughness[pixel];
    float4 surface_albedo_material = current_surface_albedo_material[pixel];

    RestirDiReservoir reservoir = invalid_reservoir();
    uint rng_state = hash_u32(
        tid.x * 1973u ^
        tid.y * 9277u ^
        restir.frame_index * 26699u
    );
    uint candidate_count = clamp(restir.initial_candidate_count, 1u, min(restir.light_count, 16u));
    float weight_sum = 0.0;

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
            surface_position_depth,
            surface_normal_roughness,
            surface_albedo_material,
            light,
            sampled_position
        );
        float candidate_weight = target_pdf / max(light_selection_pdf, 1.0e-6);
        float next_weight_sum = weight_sum + candidate_weight;

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
        reservoir.selected_weight = weight_sum / max(reservoir.target_pdf * float(reservoir.sample_count_m), 1.0e-4);
        reservoir.confidence = weight_sum;
    }
    output_reservoirs[index] = reservoir;
}
