#include "restir_di_common.slang"
#include "rt_history_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(1, 0)]]
StructuredBuffer<RestirDiReservoir> temporal_reservoirs;

[[vk::binding(2, 0)]]
RWStructuredBuffer<RestirDiReservoir> output_reservoirs;

[[vk::binding(3, 0)]]
StructuredBuffer<RtSurfacePixel> surface_pixels;

[[vk::binding(4, 0)]]
ConstantBuffer<RtHistoryUniforms> rt_history;

static const int2 rt_restir_spatial_offsets[8] = {
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

RestirDiReservoir rt_restir_spatial_invalid_reservoir() {
    RestirDiReservoir reservoir = (RestirDiReservoir)0;
    reservoir.sample_light_id = 0xffffffffu;
    reservoir.sample_count_m = 0u;
    return reservoir;
}

bool rt_restir_spatial_surface_valid(RtSurfacePixel surface) {
    if (surface.hit_kind != RT_SURFACE_HIT_KIND_VOXEL) {
        return false;
    }
    return surface.position_depth.w >= 0.0 && surface.history_confidence > 0.0;
}

bool rt_restir_spatial_surfaces_compatible(RtSurfacePixel center, RtSurfacePixel neighbor) {
    if (!rt_restir_spatial_surface_valid(center) || !rt_restir_spatial_surface_valid(neighbor)) {
        return false;
    }
    if (center.material_id != neighbor.material_id) {
        return false;
    }

    float3 center_normal = normalize(center.normal_roughness.xyz);
    float3 neighbor_normal = normalize(neighbor.normal_roughness.xyz);
    if (dot(center_normal, neighbor_normal) < rt_history.normal_threshold) {
        return false;
    }

    float position_delta = distance(center.position_depth.xyz, neighbor.position_depth.xyz);
    float depth_scale = max(center.position_depth.w, 1.0);
    return position_delta <= max(1.0, depth_scale * rt_history.depth_threshold);
}

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    uint index = launch_id.y * restir.width + launch_id.x;
    if (launch_id.x >= restir.width || launch_id.y >= restir.height || index >= restir.reservoir_count) return;

    RestirDiReservoir reservoir = temporal_reservoirs[index];
    if (restir.spatial_enabled == 0u || restir.spatial_sample_count == 0u) {
        output_reservoirs[index] = reservoir;
        return;
    }

    RtSurfacePixel surface = surface_pixels[index];
    if (!rt_restir_spatial_surface_valid(surface)) {
        output_reservoirs[index] = rt_restir_spatial_invalid_reservoir();
        return;
    }

    uint spatial_m_limit = min(max(restir.history_length, 1u) + min(restir.spatial_sample_count, 8u), 64u);
    uint combined_m = restir_di_is_valid_reservoir(reservoir)
        ? min(reservoir.sample_count_m, spatial_m_limit)
        : 0u;
    float center_target_pdf = restir_di_target_pdf_for_reservoir(
        surface.position_depth,
        surface.normal_roughness,
        surface.albedo_material,
        reservoir
    );
    float combined_weight_sum = restir_di_reservoir_stream_weight(
        center_target_pdf,
        reservoir,
        combined_m
    );
    float selected_target_pdf = center_target_pdf;

    uint rng_state = hash_u32(index * 1973u);
    uint offset_rotation = hash_u32(index) & 7u;
    for (uint sample = 0u; sample < min(restir.spatial_sample_count, 8u); sample++) {
        if (combined_m >= spatial_m_limit) {
            break;
        }

        int2 neighbor_offset = rt_restir_spatial_offsets[(offset_rotation + sample) & 7u];
        int2 neighbor_pixel_i = int2(launch_id.xy) + neighbor_offset;
        if (neighbor_pixel_i.x < 0
            || neighbor_pixel_i.y < 0
            || neighbor_pixel_i.x >= int(restir.width)
            || neighbor_pixel_i.y >= int(restir.height)) {
            continue;
        }

        uint2 neighbor_pixel = uint2(neighbor_pixel_i);
        uint neighbor_index = neighbor_pixel.y * restir.width + neighbor_pixel.x;
        RestirDiReservoir neighbor = temporal_reservoirs[neighbor_index];
        RtSurfacePixel neighbor_surface = surface_pixels[neighbor_index];
        if (!rt_restir_spatial_surfaces_compatible(surface, neighbor_surface)
            || !restir_di_is_valid_reservoir(neighbor)) {
            continue;
        }

        uint accepted_neighbor_m = min(neighbor.sample_count_m, spatial_m_limit - combined_m);
        float neighbor_target_pdf = restir_di_target_pdf_for_reservoir(
            surface.position_depth,
            surface.normal_roughness,
            surface.albedo_material,
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
            selected_target_pdf = neighbor_target_pdf;
        }

        combined_weight_sum = next_weight_sum;
        combined_m += accepted_neighbor_m;
    }

    restir_di_finalize_reservoir_on_surface_with_target(
        reservoir,
        selected_target_pdf,
        combined_weight_sum,
        combined_m
    );
    output_reservoirs[index] = reservoir;
}
