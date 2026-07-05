#include "restir_gi_common.slang"
#include "rt_history_common.slang"
#include "rt_surface_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirGiUniforms> restir_gi;

[[vk::binding(1, 0)]]
ConstantBuffer<RtHistoryUniforms> rt_history;

[[vk::binding(2, 0)]]
StructuredBuffer<RtSurfacePixel> surface_pixels;

[[vk::binding(3, 0)]]
StructuredBuffer<RestirGiReservoir> history_reservoirs;

[[vk::binding(4, 0)]]
RWStructuredBuffer<RestirGiReservoir> output_reservoirs;

[[vk::binding(5, 0)]]
StructuredBuffer<RtSurfacePixel> previous_surface_history;

[[vk::binding(6, 0)]]
RWStructuredBuffer<RtSurfacePixel> current_surface_history;

[[vk::binding(7, 0)]]
RaytracingAccelerationStructure scene_tlas;

uint gi_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float gi_rand01(inout uint state) {
    state = gi_hash_u32(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

bool rt_restir_gi_surface_valid(RtSurfacePixel surface) {
    return surface.hit_kind == RT_SURFACE_HIT_KIND_VOXEL
        && surface.position_depth.w >= 0.0
        && surface.history_confidence > 0.0;
}

bool rt_restir_gi_reproject(float3 world_position, out float2 previous_uv) {
    float4 previous_clip = mul(rt_history.previous_view_proj, float4(world_position, 1.0));
    if (previous_clip.w <= 1.0e-6) {
        previous_uv = float2(0.0);
        return false;
    }

    float2 previous_ndc = previous_clip.xy / previous_clip.w;
    previous_uv = previous_ndc * float2(0.5, -0.5) + float2(0.5, 0.5);
    return true;
}

bool rt_restir_gi_previous_uv_inside(float2 previous_uv) {
    return previous_uv.x >= 0.0
        && previous_uv.x < 1.0
        && previous_uv.y >= 0.0
        && previous_uv.y < 1.0;
}

bool rt_restir_gi_surfaces_compatible(RtSurfacePixel current, RtSurfacePixel previous) {
    if (!rt_restir_gi_surface_valid(current) || !rt_restir_gi_surface_valid(previous)) {
        return false;
    }
    if (current.material_id != previous.material_id) {
        return false;
    }

    float normal_dot = dot(normalize(current.normal_roughness.xyz), normalize(previous.normal_roughness.xyz));
    if (normal_dot < rt_history.normal_threshold) {
        return false;
    }

    float position_delta = distance(current.position_depth.xyz, previous.position_depth.xyz);
    float depth_scale = max(current.position_depth.w, 1.0);
    return position_delta <= max(1.0, depth_scale * rt_history.depth_threshold);
}

float rt_restir_gi_target_pdf(RtSurfacePixel surface, RestirGiReservoir reservoir) {
    if (!rt_restir_gi_surface_valid(surface)) {
        return 0.0;
    }
    float3 surface_normal = normalize(surface.normal_roughness.xyz);
    float3 sample_normal = normalize(reservoir.sample_normal_roughness.xyz);
    float normal_term = max(dot(surface_normal, sample_normal), 0.0);
    float albedo_luma = max(dot(surface.albedo_material.rgb, float3(0.2126, 0.7152, 0.0722)), 0.05);
    return restir_gi_luma(reservoir.sample_radiance_pdf.rgb) * albedo_luma * max(normal_term, 0.05);
}

float3 rt_restir_gi_orthonormal_tangent(float3 normal) {
    float3 up = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    return normalize(cross(up, normal));
}

float3 rt_restir_gi_sample_indirect_direction(float3 normal, inout uint rng_state) {
    float u0 = gi_rand01(rng_state);
    float u1 = gi_rand01(rng_state);
    float phi = 6.28318530718 * u0;
    float cos_theta = sqrt(max(1.0 - u1, 0.0));
    float sin_theta = sqrt(max(u1, 0.0));
    float3 tangent = rt_restir_gi_orthonormal_tangent(normal);
    float3 bitangent = cross(normal, tangent);
    return normalize(tangent * cos(phi) * sin_theta + bitangent * sin(phi) * sin_theta + normal * cos_theta);
}

RtSurfacePixel rt_restir_gi_trace_indirect_surface(RtSurfacePixel surface, float3 indirect_direction) {
    if (!rt_restir_gi_surface_valid(surface)) {
        return surface_pixel_from_payload(make_rt_surface_payload(indirect_direction));
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    RayDesc ray;
    ray.Origin = surface.position_depth.xyz + normal * 0.01 + indirect_direction * 0.01;
    ray.Direction = indirect_direction;
    ray.TMin = 0.001;
    ray.TMax = RT_SURFACE_MISS_T;

    RtSurfacePayload payload = make_rt_surface_payload(indirect_direction);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);
    return surface_pixel_from_payload(payload);
}

RestirGiReservoir rt_restir_gi_generate_initial_reservoir(
    RtSurfacePixel surface,
    RtSurfacePixel indirect_surface,
    uint index,
    uint2 pixel
) {
    RestirGiReservoir reservoir = restir_gi_invalid_reservoir();
    if (!rt_restir_gi_surface_valid(surface) || !rt_restir_gi_surface_valid(indirect_surface)) {
        return reservoir;
    }

    uint rng_state = gi_hash_u32(index * 1973u ^ pixel.x * 9277u ^ pixel.y * 26699u ^ restir_gi.frame_index);
    float candidate_jitter = 0.75 + gi_rand01(rng_state) * 0.5;
    float3 sample_position = indirect_surface.position_depth.xyz;
    float3 sample_radiance = max(indirect_surface.albedo_material.rgb, float3(0.0)) * candidate_jitter * 0.05;
    uint candidate_count = clamp(restir_gi.initial_candidate_count, 1u, 16u);
    reservoir.sample_position_depth = indirect_surface.position_depth;
    reservoir.sample_normal_roughness = indirect_surface.normal_roughness;
    reservoir.sample_radiance_pdf = float4(sample_radiance, 1.0);
    float target_pdf = max(rt_restir_gi_target_pdf(surface, reservoir), 1.0e-4);
    float weight_sum = target_pdf * float(candidate_count);

    reservoir.sample_position_depth = float4(sample_position, indirect_surface.position_depth.w);
    restir_gi_finalize_reservoir(reservoir, target_pdf, weight_sum, candidate_count);
    return reservoir;
}

RestirGiReservoir rt_restir_gi_temporal_reuse(RestirGiReservoir reservoir, RtSurfacePixel surface, uint index) {
    if (rt_history.flags != 0u || restir_gi.temporal_enabled == 0u) {
        return reservoir;
    }

    float2 previous_uv = float2(0.0);
    bool reprojected = rt_restir_gi_reproject(surface.position_depth.xyz, previous_uv);
    bool valid_resolution = rt_history.previous_resolution.x > 0u && rt_history.previous_resolution.y > 0u;
    if (!valid_resolution || !reprojected || !rt_restir_gi_previous_uv_inside(previous_uv)) {
        return reservoir;
    }

    uint2 previous_extent = rt_history.previous_resolution;
    uint2 previous_max_pixel = previous_extent - uint2(1u, 1u);
    uint2 previous_pixel = uint2(clamp(
        previous_uv * float2(previous_extent),
        float2(0.0),
        float2(previous_max_pixel)
    ));
    uint previous_index = previous_pixel.y * previous_extent.x + previous_pixel.x;
    RtSurfacePixel previous_surface = previous_surface_history[previous_index];
    if (!rt_restir_gi_surfaces_compatible(surface, previous_surface)) {
        return reservoir;
    }

    RestirGiReservoir history = history_reservoirs[previous_index];
    if (!restir_gi_is_valid_reservoir(history)) {
        return reservoir;
    }

    uint history_limit = max(restir_gi.history_length, 1u);
    uint current_m = min(reservoir.sample_count_m, history_limit);
    uint history_m = min(history.sample_count_m, history_limit - min(current_m, history_limit));
    float current_target_pdf = rt_restir_gi_target_pdf(surface, reservoir);
    float history_target_pdf = rt_restir_gi_target_pdf(surface, history);
    float current_weight = restir_gi_reservoir_stream_weight(current_target_pdf, reservoir, current_m);
    float history_weight = restir_gi_reservoir_stream_weight(history_target_pdf, history, history_m);
    float combined_weight_sum = current_weight + history_weight;
    float selected_target_pdf = current_target_pdf;

    uint rng_state = gi_hash_u32(index * 8191u ^ restir_gi.frame_index);
    if (history_weight > 0.0 && gi_rand01(rng_state) * max(combined_weight_sum, 1.0e-4) <= history_weight) {
        reservoir = history;
        selected_target_pdf = history_target_pdf;
    }

    uint combined_m = min(current_m + history_m, history_limit);
    restir_gi_finalize_reservoir(reservoir, selected_target_pdf, combined_weight_sum, combined_m);
    return reservoir;
}

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    if (launch_id.x >= restir_gi.width || launch_id.y >= restir_gi.height) return;

    uint index = launch_id.y * restir_gi.width + launch_id.x;
    if (index >= restir_gi.reservoir_count || restir_gi.enabled == 0u) {
        if (index < restir_gi.reservoir_count) {
            output_reservoirs[index] = restir_gi_invalid_reservoir();
        }
        return;
    }

    RtSurfacePixel surface = surface_pixels[index];
    current_surface_history[index] = surface;
    uint rng_state = gi_hash_u32(index * 8191u ^ restir_gi.frame_index * 131071u);
    float3 normal = normalize(surface.normal_roughness.xyz);
    float3 indirect_direction = rt_restir_gi_sample_indirect_direction(normal, rng_state);
    RtSurfacePixel indirect_surface = rt_restir_gi_trace_indirect_surface(surface, indirect_direction);
    RestirGiReservoir reservoir = rt_restir_gi_generate_initial_reservoir(surface, indirect_surface, index, launch_id.xy);
    reservoir = rt_restir_gi_temporal_reuse(reservoir, surface, index);
    output_reservoirs[index] = reservoir;
}
