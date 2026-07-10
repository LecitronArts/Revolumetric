#include "restir_gi_common.slang"
#include "rt_history_common.slang"
#include "rt_surface_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirGiUniforms> restir_gi;

[[vk::binding(1, 0)]]
StructuredBuffer<RestirGiReservoir> temporal_reservoirs;

[[vk::binding(2, 0)]]
RWStructuredBuffer<RestirGiReservoir> output_reservoirs;

[[vk::binding(3, 0)]]
StructuredBuffer<RtSurfacePixel> surface_pixels;

[[vk::binding(4, 0)]]
ConstantBuffer<RtHistoryUniforms> rt_history;

[[vk::binding(7, 0)]]
RaytracingAccelerationStructure scene_tlas;

static const float RT_RESTIR_GI_SPATIAL_VISIBILITY_BIAS = 0.01;

static const int2 rt_restir_gi_spatial_offsets[8] = {
    int2(-1, -1),
    int2(0, -1),
    int2(1, -1),
    int2(-1, 0),
    int2(1, 0),
    int2(-1, 1),
    int2(0, 1),
    int2(1, 1),
};

uint rt_restir_gi_spatial_hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rt_restir_gi_spatial_rand01(inout uint state) {
    state = rt_restir_gi_spatial_hash(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

bool rt_restir_gi_spatial_surface_valid(RtSurfacePixel surface) {
    return surface.hit_kind == RT_SURFACE_HIT_KIND_VOXEL
        && surface.position_depth.w >= 0.0
        && surface.history_confidence > 0.0;
}

bool rt_restir_gi_spatial_surfaces_compatible(RtSurfacePixel center, RtSurfacePixel neighbor) {
    if (!rt_restir_gi_spatial_surface_valid(center) || !rt_restir_gi_spatial_surface_valid(neighbor)) {
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
    return position_delta <= rt_history_position_threshold(center.position_depth.w, rt_history.depth_threshold);
}

float rt_restir_gi_spatial_target_pdf(RtSurfacePixel surface, RestirGiReservoir reservoir) {
    if (!rt_restir_gi_spatial_surface_valid(surface)) {
        return 0.0;
    }

    float3 surface_normal = normalize(surface.normal_roughness.xyz);
    float3 contribution = restir_gi_cosine_sample_contribution(
        surface.albedo_material.rgb,
        surface.position_depth.xyz,
        surface_normal,
        reservoir
    );
    return restir_gi_luma(contribution);
}

bool rt_restir_gi_spatial_sample_visible(RtSurfacePixel surface, RestirGiReservoir reservoir) {
    float3 sample_direction;
    float sample_distance;
    float3 surface_normal = normalize(surface.normal_roughness.xyz);
    if (restir_gi_is_environment_sample(reservoir)) {
        sample_direction = normalize(reservoir.sample_position_depth.xyz);
        sample_distance = RESTIR_GI_ENVIRONMENT_SAMPLE_DEPTH;
    } else {
        float3 sample_vector = reservoir.sample_position_depth.xyz - surface.position_depth.xyz;
        float distance2 = max(dot(sample_vector, sample_vector), 1.0e-6);
        sample_distance = sqrt(distance2);
        sample_direction = sample_vector * rsqrt(distance2);
    }

    RayDesc ray;
    ray.Origin = surface.position_depth.xyz
        + surface_normal * RT_RESTIR_GI_SPATIAL_VISIBILITY_BIAS
        + sample_direction * RT_RESTIR_GI_SPATIAL_VISIBILITY_BIAS;
    ray.Direction = sample_direction;
    ray.TMin = 0.001;
    ray.TMax = restir_gi_is_environment_sample(reservoir)
        ? sample_distance
        : max(sample_distance - 0.03, 0.001);

    RtSurfacePayload payload = make_rt_surface_payload(ray.Direction);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);
    return payload.hit_kind != RT_SURFACE_HIT_KIND_VOXEL;
}

bool rt_restir_gi_spatial_debug_enabled() {
    return restir_gi.debug_view == RESTIR_GI_DEBUG_VIEW_SPATIAL_VALID;
}

RestirGiReservoir rt_restir_gi_record_spatial_debug(RestirGiReservoir reservoir, uint reason) {
    if (rt_restir_gi_spatial_debug_enabled()) {
        reservoir.sample_radiance_pdf.w = float(reason);
    }
    return reservoir;
}

RestirGiReservoir rt_restir_gi_spatial_debug_passthrough(RestirGiReservoir reservoir) {
    return rt_restir_gi_record_spatial_debug(reservoir, RESTIR_GI_SPATIAL_REUSE_DISABLED_OR_PASSTHROUGH);
}

RestirGiReservoir rt_restir_gi_spatial_debug_invalid_surface() {
    return rt_restir_gi_record_spatial_debug(restir_gi_invalid_reservoir(), RESTIR_GI_SPATIAL_REUSE_INVALID_SURFACE);
}

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    if (launch_id.x >= restir_gi.width || launch_id.y >= restir_gi.height) return;

    uint index = launch_id.y * restir_gi.width + launch_id.x;
    if (index >= restir_gi.reservoir_count) return;

    RestirGiReservoir reservoir = temporal_reservoirs[index];
    if (restir_gi.enabled == 0u || restir_gi.spatial_enabled == 0u || restir_gi.spatial_sample_count == 0u) {
        output_reservoirs[index] = rt_restir_gi_spatial_debug_passthrough(reservoir);
        return;
    }

    RtSurfacePixel surface = surface_pixels[index];
    if (!rt_restir_gi_spatial_surface_valid(surface)) {
        output_reservoirs[index] = rt_restir_gi_spatial_debug_invalid_surface();
        return;
    }

    uint spatial_m_limit = min(max(restir_gi.history_length, 1u) + min(restir_gi.spatial_sample_count, 8u), 64u);
    uint combined_m = restir_gi_is_valid_reservoir(reservoir)
        ? min(reservoir.sample_count_m, spatial_m_limit)
        : 0u;
    uint spatial_reuse_reason = RESTIR_GI_SPATIAL_REUSE_CENTER_RESERVOIR_INVALID;
    float center_target_pdf = rt_restir_gi_spatial_target_pdf(surface, reservoir);
    float combined_weight_sum = restir_gi_reservoir_stream_weight(center_target_pdf, reservoir, combined_m);
    float selected_target_pdf = center_target_pdf;
    if (combined_weight_sum > 0.0) {
        spatial_reuse_reason = RESTIR_GI_SPATIAL_REUSE_NO_COMPATIBLE_NEIGHBOR;
    }

    uint rng_state = rt_restir_gi_spatial_hash(index * 1973u ^ restir_gi.frame_index);
    uint offset_rotation = rt_restir_gi_spatial_hash(index ^ restir_gi.frame_index) & 7u;
    for (uint sample = 0u; sample < min(restir_gi.spatial_sample_count, 8u); sample++) {
        if (combined_m >= spatial_m_limit) {
            break;
        }

        int2 neighbor_offset = rt_restir_gi_spatial_offsets[(offset_rotation + sample) & 7u];
        int2 neighbor_pixel_i = int2(launch_id.xy) + neighbor_offset;
        if (neighbor_pixel_i.x < 0
            || neighbor_pixel_i.y < 0
            || neighbor_pixel_i.x >= int(restir_gi.width)
            || neighbor_pixel_i.y >= int(restir_gi.height)) {
            continue;
        }

        uint2 neighbor_pixel = uint2(neighbor_pixel_i);
        uint neighbor_index = neighbor_pixel.y * restir_gi.width + neighbor_pixel.x;
        RestirGiReservoir neighbor = temporal_reservoirs[neighbor_index];
        RtSurfacePixel neighbor_surface = surface_pixels[neighbor_index];
        if (!rt_restir_gi_spatial_surfaces_compatible(surface, neighbor_surface)
            || !restir_gi_is_valid_reservoir(neighbor)) {
            continue;
        }
        if (!rt_restir_gi_spatial_sample_visible(surface, neighbor)) {
            spatial_reuse_reason = RESTIR_GI_SPATIAL_REUSE_VISIBILITY_REJECTED;
            continue;
        }

        uint accepted_neighbor_m = min(neighbor.sample_count_m, spatial_m_limit - combined_m);
        if (accepted_neighbor_m == 0u) {
            continue;
        }

        float neighbor_target_pdf = rt_restir_gi_spatial_target_pdf(surface, neighbor);
        float neighbor_weight = restir_gi_reservoir_stream_weight(
            neighbor_target_pdf,
            neighbor,
            accepted_neighbor_m
        );
        if (neighbor_weight <= 0.0) {
            continue;
        }

        float next_weight_sum = combined_weight_sum + neighbor_weight;
        if (rt_restir_gi_spatial_rand01(rng_state) * max(next_weight_sum, 1.0e-4) <= neighbor_weight) {
            reservoir = neighbor;
            reservoir.target_pdf = neighbor_target_pdf;
            selected_target_pdf = neighbor_target_pdf;
            spatial_reuse_reason = RESTIR_GI_SPATIAL_REUSE_ACCEPTED_NEIGHBOR;
        } else if (spatial_reuse_reason == RESTIR_GI_SPATIAL_REUSE_NO_COMPATIBLE_NEIGHBOR) {
            spatial_reuse_reason = RESTIR_GI_SPATIAL_REUSE_ACCEPTED_CENTER;
        }

        combined_weight_sum = next_weight_sum;
        combined_m += accepted_neighbor_m;
    }

    restir_gi_finalize_reservoir(reservoir, selected_target_pdf, combined_weight_sum, combined_m);
    if (spatial_reuse_reason == RESTIR_GI_SPATIAL_REUSE_NO_COMPATIBLE_NEIGHBOR
        && restir_gi_is_valid_reservoir(reservoir)) {
        spatial_reuse_reason = RESTIR_GI_SPATIAL_REUSE_ACCEPTED_CENTER;
    }
    output_reservoirs[index] = rt_restir_gi_record_spatial_debug(reservoir, spatial_reuse_reason);
}
