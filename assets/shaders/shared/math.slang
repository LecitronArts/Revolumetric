#pragma once
// Shared math utilities for voxel ray tracing.

// Axis-aligned bounding box.
struct AABB {
    float3 mn; // min corner
    float3 mx; // max corner
};

// Ray-AABB slab intersection. Returns (tmin, tmax).
// If tmin > tmax or tmax < 0, ray misses the box.
float2 intersect_aabb(float3 origin, float3 inv_dir, AABB box) {
    float3 t0 = (box.mn - origin) * inv_dir;
    float3 t1 = (box.mx - origin) * inv_dir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float enter = max(max(tmin.x, tmin.y), tmin.z);
    float exit  = min(min(tmax.x, tmax.y), tmax.z);
    return float2(enter, exit);
}
