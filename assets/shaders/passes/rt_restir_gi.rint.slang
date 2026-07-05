#include "rt_surface_common.slang"

[[vk::binding(8, 0)]]
StructuredBuffer<RtAabb> rt_aabbs;

[shader("intersection")]
void main() {
    uint primitive_index = PrimitiveIndex();
    RtAabb rt_aabb = rt_aabbs[primitive_index];
    float3 ray_origin = ObjectRayOrigin();
    float3 ray_direction = ObjectRayDirection();
    float3 inv_dir = rt_surface_safe_inv_dir(ray_direction);
    float2 hit_interval = intersect_aabb(ray_origin, inv_dir, rt_aabb_to_aabb(rt_aabb));
    float hit_t = max(hit_interval.x, RayTMin());

    if (hit_interval.x <= hit_interval.y && hit_interval.y >= RayTMin() && hit_t <= RayTCurrent()) {
        RtSurfaceIntersectionAttributes attributes;
        attributes.object_position = ray_origin + ray_direction * hit_t;
        attributes.object_normal = rt_surface_aabb_normal(attributes.object_position, rt_aabb);
        ReportHit(hit_t, RT_SURFACE_HIT_KIND_VOXEL, attributes);
    }
}
