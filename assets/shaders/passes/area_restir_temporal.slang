#include "area_restir_common.slang"
#include "scene_common.slang"
#include "vpt_history_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<AreaRestirUniforms> area_restir;

[[vk::binding(1, 0)]]
StructuredBuffer<AreaRestirReservoir> input_reservoirs;

[[vk::binding(2, 0)]]
StructuredBuffer<AreaRestirReservoir> history_reservoirs;

[[vk::binding(3, 0)]]
RWStructuredBuffer<AreaRestirReservoir> output_reservoirs;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_position_depth;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_normal_roughness;

[[vk::binding(6, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_albedo_material;

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

static const float AREA_RESTIR_TEMPORAL_MIN_TAP_WEIGHT = 0.0625;

static const int2 area_restir_history_tap_offsets[4] = {
    int2(0, 0),
    int2(1, 0),
    int2(0, 1),
    int2(1, 1),
};

AreaRestirEvalContext current_context(uint2 pixel) {
    AreaRestirEvalContext context;
    context.position_depth = surface_position_depth[pixel];
    context.normal_roughness = surface_normal_roughness[pixel];
    context.albedo_material = surface_albedo_material[pixel];
    context.motion_history = motion_history[pixel];
    return context;
}

uint area_restir_temporal_hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float area_restir_history_tap_weight(uint tap, float2 history_fraction) {
    float wx = area_restir_history_tap_offsets[tap].x == 0 ? 1.0 - history_fraction.x : history_fraction.x;
    float wy = area_restir_history_tap_offsets[tap].y == 0 ? 1.0 - history_fraction.y : history_fraction.y;
    return wx * wy;
}

bool area_restir_history_tap_inside(int2 previous_pixel, uint width, uint height) {
    return previous_pixel.x >= 0
        && previous_pixel.y >= 0
        && previous_pixel.x < int(width)
        && previous_pixel.y < int(height);
}

bool area_restir_temporal_surface_compatible(
    AreaRestirEvalContext center,
    uint2 previous_pixel
) {
    float4 previous_position = previous_surface_position_depth[previous_pixel];
    if (center.position_depth.w < 0.0 || previous_position.w < 0.0) {
        return false;
    }

    float position_delta = distance(center.position_depth.xyz, previous_position.xyz);
    float depth_scale = max(center.position_depth.w, 1.0);
    if (position_delta > max(1.0, depth_scale * area_restir.depth_threshold)) {
        return false;
    }

    float4 previous_albedo = previous_surface_albedo_material[previous_pixel];
    if (abs(center.albedo_material.w - previous_albedo.w) > 0.5) {
        return false;
    }

    float3 center_normal = normalize(center.normal_roughness.xyz);
    float3 previous_normal = normalize(previous_surface_normal_roughness[previous_pixel].xyz);
    return dot(center_normal, previous_normal) >= area_restir.normal_threshold;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    uint index = pixel.y * area_restir.width + pixel.x;
    if (pixel.x >= area_restir.width || pixel.y >= area_restir.height || index >= area_restir.reservoir_count) {
        return;
    }

    AreaRestirReservoir current = input_reservoirs[index];
    if (area_restir.enabled == 0u || area_restir.temporal_enabled == 0u || !area_restir_is_valid_reservoir(current)) {
        output_reservoirs[index] = current;
        return;
    }

    AreaRestirEvalContext center_context = current_context(pixel);
    uint rng = area_restir_temporal_hash(index ^ (area_restir.frame_index * 2246822519u));
    AreaRestirReservoir reservoir = area_restir_invalid_reservoir();
    float center_target_luma = area_restir_context_target_luma(center_context);
    float current_target_pdf = center_target_luma;
    float current_candidate_weight_sum = current_target_pdf
        * area_restir_reservoir_reuse_weight(current)
        * float(max(current.sample_count_m, 1u));
    current.rejection_reason = 0u;
    current.debug_flags = 0u;
    area_restir_reservoir_update(
        reservoir,
        current,
        current_target_pdf,
        current_candidate_weight_sum,
        max(current.sample_count_m, 1u),
        rng
    );

    float4 motion = center_context.motion_history;
    if (motion.w == 0.0) {
        reservoir.rejection_reason = 1u;
        output_reservoirs[index] = reservoir;
        return;
    }

    float2 history_sample = motion.xy - 0.5;
    int2 previous_base_pixel = int2(floor(history_sample));
    float2 history_fraction = saturate(history_sample - float2(previous_base_pixel));
    bool accepted_history = false;

    [unroll]
    for (uint tap = 0u; tap < 4u; tap++) {
        int2 tap_pixel_i = previous_base_pixel + area_restir_history_tap_offsets[tap];
        float tap_weight = area_restir_history_tap_weight(tap, history_fraction);
        if (tap_weight < AREA_RESTIR_TEMPORAL_MIN_TAP_WEIGHT
            || !area_restir_history_tap_inside(tap_pixel_i, area_restir.width, area_restir.height)) {
            continue;
        }

        uint2 previous_pixel = uint2(tap_pixel_i);
        uint previous_index = previous_pixel.y * area_restir.width + previous_pixel.x;
        AreaRestirReservoir history = history_reservoirs[previous_index];
        if (!area_restir_is_valid_reservoir(history)) {
            continue;
        }

        if (!area_restir_temporal_surface_compatible(center_context, previous_pixel)) {
            continue;
        }

        float history_target_pdf = center_target_luma;
        float bilinear_confidence = max(tap_weight, 0.05);
        float history_candidate_weight_sum = history_target_pdf
            * area_restir_reservoir_reuse_weight(history)
            * float(max(history.sample_count_m, 1u))
            * bilinear_confidence;
        history.sample_state.flags |= AREA_RESTIR_SAMPLE_HISTORY_VALID;
        history.rejection_reason = 0u;
        history.debug_flags |= AREA_RESTIR_SAMPLE_HISTORY_VALID;
        area_restir_reservoir_update(
            reservoir,
            history,
            history_target_pdf,
            history_candidate_weight_sum,
            max(history.sample_count_m, 1u),
            rng
        );
        accepted_history = true;
    }

    if (!accepted_history) {
        reservoir.rejection_reason = 2u;
        output_reservoirs[index] = reservoir;
        return;
    }

    reservoir.sample_count_m = min(reservoir.sample_count_m, area_restir.history_length);
    reservoir.confidence = min(max(current.confidence, reservoir.confidence) + 1.0, float(area_restir.history_length));
    area_restir_finalize_reservoir(reservoir, area_restir.history_length);
    output_reservoirs[index] = reservoir;
}
