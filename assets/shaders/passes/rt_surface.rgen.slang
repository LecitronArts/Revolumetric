#include "rt_history_common.slang"
#include "scene_common.slang"
#include "rt_surface_common.slang"

// Vulkan reflection name expected by the Rust source tests: AccelerationStructureKHR.

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure scene_tlas;

[[vk::binding(1, 0)]]
RWStructuredBuffer<RtSurfacePixel> surface_pixels;

[[vk::binding(2, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(3, 0)]]
StructuredBuffer<RtAabb> rt_aabbs;

[[vk::binding(13, 0)]]
ConstantBuffer<RtHistoryUniforms> rt_history;

bool rt_surface_history_reset_active() {
    return rt_history.flags != 0u;
}

float4 rt_surface_project_previous_pixel(uint2 pixel, float3 world_position) {
    if (rt_surface_history_reset_active()) {
        return rt_surface_invalid_motion_history();
    }

    bool valid_resolution = rt_history.previous_resolution.x > 0u && rt_history.previous_resolution.y > 0u;
    if (!valid_resolution) {
        return rt_surface_invalid_motion_history();
    }

    float4 current_clip = mul(rt_history.current_view_proj, float4(world_position, 1.0));
    float4 previous_clip = mul(rt_history.previous_view_proj, float4(world_position, 1.0));
    if (current_clip.w <= 1.0e-5 || previous_clip.w <= 1.0e-5) {
        return rt_surface_invalid_motion_history();
    }

    float2 previous_ndc = previous_clip.xy / previous_clip.w;
    float2 previous_uv = previous_ndc * float2(0.5, -0.5) + float2(0.5, 0.5);
    float2 current_pixel = float2(pixel) + float2(0.5);
    float2 previous_pixel = previous_uv * float2(rt_history.previous_resolution);

    bool valid = previous_pixel.x >= 0.0
        && previous_pixel.y >= 0.0
        && previous_pixel.x < float(rt_history.previous_resolution.x)
        && previous_pixel.y < float(rt_history.previous_resolution.y);
    if (!valid) {
        return rt_surface_invalid_motion_history();
    }

    float2 motion_delta = previous_pixel - current_pixel;
    return float4(motion_delta, previous_clip.w - current_clip.w, 1.0);
}

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    uint3 launch_size = DispatchRaysDimensions();
    uint index = launch_id.y * launch_size.x + launch_id.x;

    ScenePrimaryRay primary_ray = scene_primary_ray_from_area_sample(
        scene_ubo,
        float2(launch_id.xy) + float2(0.5, 0.5),
        float2(0.5, 0.5)
    );

    RayDesc ray;
    ray.Origin = primary_ray.origin;
    ray.Direction = primary_ray.direction;
    ray.TMin = 0.001;
    ray.TMax = RT_SURFACE_MISS_T;

    RtSurfacePayload payload = make_rt_surface_payload(primary_ray.direction);
    payload.albedo = float3(0.02, 0.025, 0.03);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);

    RtSurfacePixel surface = surface_pixel_from_payload(payload);
    if (surface.hit_kind == RT_SURFACE_HIT_KIND_VOXEL) {
        surface.motion_history = rt_surface_project_previous_pixel(launch_id.xy, surface.position_depth.xyz);
    }
    surface_pixels[index] = surface;
}
