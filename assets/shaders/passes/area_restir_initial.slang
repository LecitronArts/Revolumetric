#include "area_restir_common.slang"
#include "scene_common.slang"
#include "vpt_history_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<AreaRestirUniforms> area_restir;

[[vk::binding(1, 0)]]
RWStructuredBuffer<AreaRestirReservoir> output_reservoirs;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_position_depth;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_normal_roughness;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_albedo_material;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> area_restir_debug;

[[vk::binding(6, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

uint area_restir_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float area_restir_rand01(inout uint state) {
    state = area_restir_hash_u32(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

float surface_target_luma(float4 position_depth, float4 albedo_material) {
    if (position_depth.w < 0.0) {
        return 0.0;
    }
    return max(dot(albedo_material.rgb, float3(0.2126, 0.7152, 0.0722)), 1.0e-3);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    uint index = pixel.y * area_restir.width + pixel.x;
    if (pixel.x >= area_restir.width || pixel.y >= area_restir.height || index >= area_restir.reservoir_count) {
        return;
    }

    float4 center_position_depth = surface_position_depth[pixel];
    float4 center_albedo_material = surface_albedo_material[pixel];
    if (area_restir.enabled == 0u || center_position_depth.w < 0.0) {
        output_reservoirs[index] = area_restir_invalid_reservoir();
        area_restir_debug[pixel] = float4(0.0);
        return;
    }

    uint rng = area_restir_hash_u32(index ^ (area_restir.frame_index * 747796405u));
    AreaRestirReservoir reservoir = area_restir_invalid_reservoir();
    float weight_sum = 0.0;
    uint candidate_count = max(area_restir.initial_candidate_count, 1u);
    float target_luma = surface_target_luma(center_position_depth, center_albedo_material);

    for (uint candidate = 0u; candidate < candidate_count; candidate++) {
        float2 subpixel_uv = area_restir.subpixel_enabled != 0u
            ? float2(area_restir_rand01(rng), area_restir_rand01(rng))
            : float2(0.5, 0.5);
        float2 lens_uv = area_restir.lens_enabled != 0u
            ? float2(area_restir_rand01(rng), area_restir_rand01(rng))
            : float2(0.5, 0.5);

        AreaRestirSampleState sample_state = area_restir_sample_pixel_state(pixel, subpixel_uv, lens_uv, candidate);
        float target_pdf = target_luma;
        float next_weight_sum = weight_sum + target_pdf;
        if (!area_restir_is_valid_reservoir(reservoir) || area_restir_rand01(rng) * next_weight_sum <= target_pdf) {
            reservoir.sample_state = sample_state;
            reservoir.target_pdf = target_pdf;
            reservoir.selected_radiance = float4(center_albedo_material.rgb, 1.0);
            reservoir.contribution_luma = target_pdf;
        }
        weight_sum = next_weight_sum;
    }

    reservoir.sample_count_m = candidate_count;
    reservoir.weight_sum = weight_sum;
    reservoir.confidence = float(candidate_count);
    area_restir_finalize_reservoir(reservoir, area_restir.history_length);
    output_reservoirs[index] = reservoir;
    area_restir_debug[pixel] = float4(reservoir.sample_state.subpixel_uv, reservoir.selected_weight > 0.0 ? 1.0 : 0.0, 1.0);
}
