// Ray tracing shared utilities.
#include "math.slang"

struct Ray {
    float3 origin;
    float3 direction;
    float3 inv_dir; // 1.0 / direction, precomputed
};

Ray make_ray(float3 origin, float3 direction) {
    Ray r;
    r.origin = origin;
    r.direction = direction;
    // Avoid division by zero: use a large reciprocal for near-zero components
    r.inv_dir = float3(
        abs(direction.x) > 1e-8 ? 1.0 / direction.x : sign(direction.x) * 1e8,
        abs(direction.y) > 1e-8 ? 1.0 / direction.y : sign(direction.y) * 1e8,
        abs(direction.z) > 1e-8 ? 1.0 / direction.z : sign(direction.z) * 1e8
    );
    return r;
}

struct HitResult {
    bool  hit;
    float t;         // distance along ray
    float3 position; // world-space hit point
    float3 normal;   // face normal at hit (axis-aligned, from DDA step)
    uint  brick_id;  // which brick was hit
    uint3 local;     // local voxel coordinate within brick [0,7]³
};

static const HitResult NO_HIT = { false, 1e30, float3(0), float3(0), 0xFFFFFFFFu, uint3(0) };
