#include "restir_di_common.slang"
#include "vpt_history_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(1, 0)]]
StructuredBuffer<RestirDiReservoir> input_reservoirs;

[[vk::binding(2, 0)]]
StructuredBuffer<RestirDiReservoir> history_reservoirs;

[[vk::binding(3, 0)]]
RWStructuredBuffer<RestirDiReservoir> output_reservoirs;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> current_surface_position_depth;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> current_surface_normal_roughness;

[[vk::binding(6, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> current_surface_albedo_material;

[[vk::binding(7, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> previous_surface_position_depth;

[[vk::binding(8, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> previous_surface_normal_roughness;

[[vk::binding(9, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> previous_surface_albedo_material;

[[vk::binding(10, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> motion_history;

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

bool surface_is_valid(uint index) {
    uint2 pixel = uint2(index % restir.width, index / restir.width);
    return current_surface_position_depth[pixel].w >= 0.0;
}

bool compatible_temporal_surface(
    float4 current_position,
    float4 current_normal_roughness,
    float4 current_albedo_material,
    uint2 previous_pixel
) {
    float4 previous_position = previous_surface_position_depth[previous_pixel];
    float3 current_normal = normalize(current_normal_roughness.xyz);
    float3 previous_normal = normalize(previous_surface_normal_roughness[previous_pixel].xyz);
    float current_material = current_albedo_material.a;
    float previous_material = previous_surface_albedo_material[previous_pixel].a;
    float normal_dot = dot(current_normal, previous_normal);
    float position_delta = length(current_position.xyz - previous_position.xyz);
    return current_position.w >= 0.0
        && previous_position.w >= 0.0
        && normal_dot > 0.85
        && position_delta < 2.0
        && abs(current_material - previous_material) < 0.5;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.y * restir.width + tid.x;
    if (tid.x >= restir.width || tid.y >= restir.height || index >= restir.reservoir_count) return;

    RestirDiReservoir reservoir = input_reservoirs[index];
    if (!surface_is_valid(index)) {
        reservoir.sample_light_id = 0xffffffffu;
        reservoir.sample_count_m = 0u;
        output_reservoirs[index] = reservoir;
        return;
    }

    uint2 pixel = tid.xy;
    float4 surface_position_depth = current_surface_position_depth[pixel];
    float4 surface_normal_roughness = current_surface_normal_roughness[pixel];
    float4 surface_albedo_material = current_surface_albedo_material[pixel];
    float4 motion = motion_history[tid.xy];
    bool previous_pixel_valid = motion.w >= 0.5
        && motion.x >= 0.0
        && motion.y >= 0.0
        && motion.x < float(restir.width)
        && motion.y < float(restir.height);

    if (restir.temporal_enabled != 0u && previous_pixel_valid) {
        uint2 previous_pixel = uint2(motion.xy);
        uint previous_index = previous_pixel.y * restir.width + previous_pixel.x;
        RestirDiReservoir history_reservoir = history_reservoirs[previous_index];
        if (restir_di_is_valid_reservoir(history_reservoir)
            && compatible_temporal_surface(surface_position_depth, surface_normal_roughness, surface_albedo_material, previous_pixel)) {
            uint history_m_limit = max(restir.history_length, 1u);
            uint current_m = min(reservoir.sample_count_m, history_m_limit);
            float current_target_pdf = restir_di_target_pdf_for_reservoir(
                surface_position_depth,
                surface_normal_roughness,
                surface_albedo_material,
                reservoir
            );
            float current_weight = restir_di_reservoir_stream_weight(
                current_target_pdf,
                reservoir,
                current_m
            );

            uint capped_history_m = min(
                history_reservoir.sample_count_m,
                history_m_limit - min(current_m, history_m_limit)
            );
            float history_target_pdf = restir_di_target_pdf_for_reservoir(
                surface_position_depth,
                surface_normal_roughness,
                surface_albedo_material,
                history_reservoir
            );
            float history_weight = restir_di_reservoir_stream_weight(
                history_target_pdf,
                history_reservoir,
                capped_history_m
            );
            float combined_weight_sum = current_weight + history_weight;

            uint rng_state = hash_u32(index * 1973u ^ restir.frame_index * 9277u);
            if (history_weight > 0.0
                && rand01(rng_state) * max(combined_weight_sum, 1.0e-4) <= history_weight) {
                reservoir = history_reservoir;
                reservoir.target_pdf = history_target_pdf;
            }

            uint combined_m = min(current_m + capped_history_m, history_m_limit);
            restir_di_finalize_reservoir_on_surface(
                reservoir,
                surface_position_depth,
                surface_normal_roughness,
                surface_albedo_material,
                combined_weight_sum,
                combined_m
            );
        }
    }
    if (reservoir.sample_light_id != 0xffffffffu && reservoir.sample_count_m > 0u) {
        reservoir.target_pdf = restir_di_target_pdf_for_reservoir(
            surface_position_depth,
            surface_normal_roughness,
            surface_albedo_material,
            reservoir
        );
    }
    output_reservoirs[index] = reservoir;
}
