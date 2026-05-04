#pragma once
// Shared math utilities for voxel ray tracing.

// Axis-aligned bounding box.
struct AABB {
    float3 mn; // min corner
    float3 mx; // max corner
};

static const float AABB_PARALLEL_INV_DIR_THRESHOLD = 9.0e7;
static const float AABB_T_MAX = 1.0e30;

float2 aabb_axis_interval(
    float origin_component,
    float inv_dir_component,
    float min_component,
    float max_component
) {
    if (abs(inv_dir_component) >= AABB_PARALLEL_INV_DIR_THRESHOLD) {
        if (origin_component < min_component || origin_component > max_component) {
            return float2(1.0, 0.0);
        }
        return float2(-AABB_T_MAX, AABB_T_MAX);
    }

    float t0 = (min_component - origin_component) * inv_dir_component;
    float t1 = (max_component - origin_component) * inv_dir_component;
    return float2(min(t0, t1), max(t0, t1));
}

// Ray-AABB slab intersection. Returns (tmin, tmax).
// If tmin > tmax or tmax < 0, ray misses the box.
float2 intersect_aabb(float3 origin, float3 inv_dir, AABB box) {
    float2 x = aabb_axis_interval(origin.x, inv_dir.x, box.mn.x, box.mx.x);
    float2 y = aabb_axis_interval(origin.y, inv_dir.y, box.mn.y, box.mx.y);
    float2 z = aabb_axis_interval(origin.z, inv_dir.z, box.mn.z, box.mx.z);
    float enter = max(max(x.x, y.x), z.x);
    float exit  = min(min(x.y, y.y), z.y);
    return float2(enter, exit);
}
