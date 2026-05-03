#include "restir_di_common.slang"
#include "vpt_history_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(1, 0)]]
StructuredBuffer<RestirDiReservoir> temporal_reservoirs;

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

static const int2 spatial_offsets[8] = {
    int2(-1, -1),
    int2(0, -1),
    int2(1, -1),
    int2(-1, 0),
    int2(1, 0),
    int2(-1, 1),
    int2(0, 1),
    int2(1, 1),
};

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

bool compatible_spatial_surface(uint center_index, uint neighbor_index) {
    uint2 center_pixel = uint2(center_index % restir.width, center_index / restir.width);
    uint2 neighbor_pixel = uint2(neighbor_index % restir.width, neighbor_index / restir.width);
    float4 center_position = current_surface_position_depth[center_pixel];
    float4 neighbor_position = current_surface_position_depth[neighbor_pixel];
    float3 center_normal = normalize(current_surface_normal_roughness[center_pixel].xyz);
    float3 neighbor_normal = normalize(current_surface_normal_roughness[neighbor_pixel].xyz);
    float center_material = current_surface_albedo_material[center_pixel].a;
    float neighbor_material = current_surface_albedo_material[neighbor_pixel].a;
    float normal_dot = dot(center_normal, neighbor_normal);
    float position_delta = length(center_position.xyz - neighbor_position.xyz);
    return center_position.w >= 0.0
        && neighbor_position.w >= 0.0
        && normal_dot > 0.85
        && position_delta < 2.0
        && abs(center_material - neighbor_material) < 0.5;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.y * restir.width + tid.x;
    if (tid.x >= restir.width || tid.y >= restir.height || index >= restir.reservoir_count) return;

    RestirDiReservoir reservoir = temporal_reservoirs[index];
    if (restir.spatial_enabled == 0u) {
        output_reservoirs[index] = reservoir;
        return;
    }

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

    uint spatial_m_limit = min(max(restir.history_length, 1u) + min(restir.spatial_sample_count, 8u), 64u);
    uint combined_m = min(reservoir.sample_count_m, spatial_m_limit);
    float center_target_pdf = restir_di_target_pdf_for_reservoir(
        surface_position_depth,
        surface_normal_roughness,
        surface_albedo_material,
        reservoir
    );
    float combined_weight_sum = restir_di_reservoir_stream_weight(
        center_target_pdf,
        reservoir,
        combined_m
    );

    uint rng_state = hash_u32(index * 1973u ^ restir.frame_index * 9277u);
    uint offset_rotation = hash_u32(rng_state) & 7u;
    for (uint sample = 0u; sample < min(restir.spatial_sample_count, 8u); sample++) {
        if (combined_m >= spatial_m_limit) {
            break;
        }

        int2 neighbor_offset = spatial_offsets[(offset_rotation + sample) & 7u];
        int2 neighbor_pixel_i = int2(tid.xy) + neighbor_offset;
        if (neighbor_pixel_i.x < 0
            || neighbor_pixel_i.y < 0
            || neighbor_pixel_i.x >= int(restir.width)
            || neighbor_pixel_i.y >= int(restir.height)) {
            continue;
        }

        uint2 neighbor_pixel = uint2(neighbor_pixel_i);
        uint neighbor_index = neighbor_pixel.y * restir.width + neighbor_pixel.x;
        RestirDiReservoir neighbor = temporal_reservoirs[neighbor_index];
        if (!compatible_spatial_surface(index, neighbor_index)
            || !restir_di_is_valid_reservoir(neighbor)) {
            continue;
        }

        uint neighbor_m = min(neighbor.sample_count_m, spatial_m_limit - combined_m);
        uint accepted_neighbor_m = neighbor_m;
        float neighbor_target_pdf = restir_di_target_pdf_for_reservoir(
            surface_position_depth,
            surface_normal_roughness,
            surface_albedo_material,
            neighbor
        );
        float neighbor_weight = restir_di_reservoir_stream_weight(
            neighbor_target_pdf,
            neighbor,
            accepted_neighbor_m
        );
        if (neighbor_weight <= 0.0) {
            continue;
        }

        float next_weight_sum = combined_weight_sum + neighbor_weight;
        if (rand01(rng_state) * max(next_weight_sum, 1.0e-4) <= neighbor_weight) {
            reservoir = neighbor;
            reservoir.target_pdf = neighbor_target_pdf;
        }

        combined_weight_sum = next_weight_sum;
        combined_m += accepted_neighbor_m;
    }

    restir_di_finalize_reservoir_on_surface(
        reservoir,
        surface_position_depth,
        surface_normal_roughness,
        surface_albedo_material,
        combined_weight_sum,
        combined_m
    );
    output_reservoirs[index] = reservoir;
}
