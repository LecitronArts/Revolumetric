#include "rt_history_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<RtSurfacePixel> surface_pixels;

[[vk::binding(1, 0)]]
RWTexture2D<float4> temporal_radiance;

[[vk::binding(2, 0)]]
Texture2D<float4> previous_temporal_radiance;

[[vk::binding(3, 0)]]
ConstantBuffer<RtHistoryUniforms> rt_history;

[[vk::binding(4, 0)]]
RWStructuredBuffer<RtSurfacePixel> current_surface_history;

[[vk::binding(5, 0)]]
StructuredBuffer<RtSurfacePixel> previous_surface_history;

[[vk::binding(6, 0)]]
Texture2D<float4> current_radiance;

bool reproject(float3 world_position, out float2 previous_uv) {
    float4 previous_clip = mul(float4(world_position, 1.0), rt_history.previous_view_proj);
    if (previous_clip.w <= 1.0e-6) {
        previous_uv = float2(0.0);
        return false;
    }

    float2 previous_ndc = previous_clip.xy / previous_clip.w;
    previous_uv = previous_ndc * float2(0.5, -0.5) + float2(0.5, 0.5);
    return true;
}

bool rt_temporal_debug_bypasses_history(uint debug_view) {
    return debug_view == RT_DEBUG_VIEW_SURFACE
        || debug_view == RT_DEBUG_VIEW_HIT_DISTANCE
        || debug_view == RT_DEBUG_VIEW_DIRECT_RESERVOIR
        || debug_view == RT_DEBUG_VIEW_INDIRECT_RESERVOIR;
}

bool previous_uv_inside(float2 previous_uv) {
    return previous_uv.x >= 0.0
        && previous_uv.x < 1.0
        && previous_uv.y >= 0.0
        && previous_uv.y < 1.0;
}

float bounded_history_alpha(uint history_length) {
    float length = float(max(history_length, 1u));
    return clamp(1.0 / (length + 1.0), 0.04, 0.25);
}

bool rt_surface_accepts_temporal_history(RtSurfacePixel surface) {
    return surface.history_confidence > 0.0
        && surface.hit_kind == RT_SURFACE_HIT_KIND_VOXEL
        && surface.position_depth.w >= 0.0;
}

bool rt_surfaces_compatible(RtSurfacePixel current, RtSurfacePixel previous) {
    bool previous_voxel = previous.hit_kind == RT_SURFACE_HIT_KIND_VOXEL;
    if (!rt_surface_accepts_temporal_history(current) || !previous_voxel) {
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

float3 rt_temporal_visualize_reuse(float3 current, float3 blended, float alpha, bool motion_history_valid) {
    float history_weight = 1.0 - alpha;
    float blend_delta = saturate(length(blended - current) * 4.0);
    return saturate(float3(history_weight, blend_delta, motion_history_valid ? 1.0 : 0.0));
}

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    uint3 launch_size = DispatchRaysDimensions();
    uint index = launch_id.y * launch_size.x + launch_id.x;
    RtSurfacePixel surface = surface_pixels[index];
    current_surface_history[index] = surface;

    bool reset = rt_history.flags != 0u;
    bool temporal_enabled = rt_history.temporal_denoise_enabled != 0u;
    bool debug_bypass = rt_temporal_debug_bypasses_history(rt_history.debug_view);
    uint2 previous_extent = rt_history.previous_resolution;
    float2 previous_pixel_f = surface.motion_history.xy + float2(launch_id.xy) + float2(0.5);
    bool motion_history_valid = surface.motion_history.w > 0.0;
    bool valid_resolution = previous_extent.x > 0u && previous_extent.y > 0u;
    bool previous_pixel_valid = valid_resolution && previous_pixel_f.x >= 0.0
        && previous_pixel_f.x < float(previous_extent.x)
        && previous_pixel_f.y >= 0.0
        && previous_pixel_f.y < float(previous_extent.y);
    bool base_history_valid = temporal_enabled
        && !debug_bypass
        && !reset
        && valid_resolution
        && rt_surface_accepts_temporal_history(surface)
        && motion_history_valid
        && previous_pixel_valid;
    float3 current = current_radiance.Load(int3(launch_id.xy, 0)).rgb;
    uint2 previous_pixel = uint2(0u, 0u);
    uint previous_index = 0u;
    bool history_valid = false;

    if (base_history_valid) {
        uint2 previous_max_pixel = previous_extent - uint2(1u, 1u);
        previous_pixel = uint2(clamp(
            previous_pixel_f,
            float2(0.0),
            float2(previous_max_pixel)
        ));
        previous_index = previous_pixel.y * previous_extent.x + previous_pixel.x;
        RtSurfacePixel previous_surface = previous_surface_history[previous_index];
        history_valid = rt_surfaces_compatible(surface, previous_surface);
    }

    if (rt_history.debug_view == RT_DEBUG_VIEW_HISTORY_VALID) {
        temporal_radiance[launch_id.xy] = float4(history_valid ? float3(1.0) : float3(0.0), 1.0);
        return;
    }

    if (rt_history.debug_view == RT_DEBUG_VIEW_TEMPORAL && !history_valid) {
        temporal_radiance[launch_id.xy] = float4(0.0, 0.0, motion_history_valid ? 1.0 : 0.0, 1.0);
        return;
    }

    if (!temporal_enabled || debug_bypass || !history_valid) {
        temporal_radiance[launch_id.xy] = float4(current, 1.0);
        return;
    }

    float3 history = previous_temporal_radiance.Load(int3(previous_pixel, 0)).rgb;
    float3 clamped_history = clamp(history, current - 0.25, current + 0.25);
    float alpha = bounded_history_alpha(rt_history.history_length);
    float3 blended = lerp(clamped_history, current, alpha);
    if (rt_history.debug_view == RT_DEBUG_VIEW_TEMPORAL) {
        float3 temporal_debug = rt_temporal_visualize_reuse(current, blended, alpha, motion_history_valid);
        temporal_radiance[launch_id.xy] = float4(temporal_debug, 1.0);
        return;
    }

    temporal_radiance[launch_id.xy] = float4(blended, 1.0);
}
