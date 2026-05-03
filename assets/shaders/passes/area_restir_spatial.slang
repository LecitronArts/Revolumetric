#include "area_restir_common.slang"
#include "scene_common.slang"
#include "vpt_history_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<AreaRestirUniforms> area_restir;

[[vk::binding(1, 0)]]
StructuredBuffer<AreaRestirReservoir> temporal_reservoirs;

[[vk::binding(2, 0)]]
RWStructuredBuffer<AreaRestirReservoir> output_reservoirs;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_position_depth;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_normal_roughness;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_albedo_material;

[[vk::binding(6, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> area_restir_debug;

static const int2 neighbor_offsets[8] = {
    int2(-1, -1),
    int2(0, -1),
    int2(1, -1),
    int2(-1, 0),
    int2(1, 0),
    int2(-1, 1),
    int2(0, 1),
    int2(1, 1),
};

AreaRestirEvalContext spatial_context(uint2 pixel) {
    AreaRestirEvalContext context;
    context.position_depth = surface_position_depth[pixel];
    context.normal_roughness = surface_normal_roughness[pixel];
    context.albedo_material = surface_albedo_material[pixel];
    context.motion_history = float4(0.0);
    return context;
}

uint area_restir_spatial_hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    uint index = pixel.y * area_restir.width + pixel.x;
    if (pixel.x >= area_restir.width || pixel.y >= area_restir.height || index >= area_restir.reservoir_count) {
        return;
    }

    AreaRestirReservoir center_reservoir = temporal_reservoirs[index];
    if (area_restir.enabled == 0u || area_restir.spatial_enabled == 0u || area_restir.spatial_sample_count == 0u || !area_restir_is_valid_reservoir(center_reservoir)) {
        output_reservoirs[index] = center_reservoir;
        return;
    }

    AreaRestirEvalContext center = spatial_context(pixel);
    uint rng = area_restir_spatial_hash(index ^ (area_restir.frame_index * 3266489917u));
    AreaRestirReservoir reservoir = area_restir_invalid_reservoir();
    float center_target_pdf = area_restir_context_target_luma(center);
    float center_candidate_weight_sum = center_target_pdf
        * area_restir_reservoir_reuse_weight(center_reservoir)
        * float(max(center_reservoir.sample_count_m, 1u));
    area_restir_reservoir_update(
        reservoir,
        center_reservoir,
        center_target_pdf,
        center_candidate_weight_sum,
        max(center_reservoir.sample_count_m, 1u),
        rng
    );

    uint sample_count = min(area_restir.spatial_sample_count, 8u);
    uint rotation = rng % 8u;
    for (uint tap = 0u; tap < sample_count; tap++) {
        uint rotated_tap = (tap + rotation) & 7u;
        int2 offset = neighbor_offsets[rotated_tap] * int(max(area_restir.spatial_radius / 24.0, 1.0));
        int2 neighbor_i = int2(pixel) + offset;
        if (neighbor_i.x < 0 || neighbor_i.y < 0
            || neighbor_i.x >= int(area_restir.width)
            || neighbor_i.y >= int(area_restir.height)) {
            continue;
        }
        uint2 neighbor_pixel = uint2(neighbor_i);
        uint neighbor_index = neighbor_pixel.y * area_restir.width + neighbor_pixel.x;
        AreaRestirReservoir neighbor = temporal_reservoirs[neighbor_index];
        if (!area_restir_is_valid_reservoir(neighbor)) {
            continue;
        }
        if (!area_restir_surface_compatible(center, spatial_context(neighbor_pixel), area_restir.normal_threshold, area_restir.depth_threshold)) {
            continue;
        }

        float neighbor_target_pdf = center_target_pdf;
        float distance_weight = 1.0 / max(length(float2(offset)), 1.0);
        float neighbor_candidate_weight_sum = neighbor_target_pdf
            * area_restir_reservoir_reuse_weight(neighbor)
            * float(max(neighbor.sample_count_m, 1u))
            * distance_weight;
        neighbor.jacobian = distance_weight;
        neighbor.debug_flags |= 4u;
        area_restir_reservoir_update(
            reservoir,
            neighbor,
            neighbor_target_pdf,
            neighbor_candidate_weight_sum,
            max(neighbor.sample_count_m, 1u),
            rng
        );
    }

    reservoir.sample_count_m = min(reservoir.sample_count_m, area_restir.history_length);
    reservoir.jacobian = max(reservoir.jacobian, 1.0);
    reservoir.debug_flags |= 8u;
    area_restir_finalize_reservoir(reservoir, area_restir.history_length);
    output_reservoirs[index] = reservoir;
    if (area_restir.debug_view != 0u) {
        area_restir_debug[pixel] = float4(reservoir.selected_weight, reservoir.confidence, reservoir.jacobian, 1.0);
    }
}
