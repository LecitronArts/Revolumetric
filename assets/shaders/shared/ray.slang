#pragma once
// Ray tracing shared utilities.
#include "math.slang"
#include "voxel_common.slang"

struct Ray {
    float3 origin;
    float3 direction;
    float3 inv_dir; // 1.0 / direction, precomputed
};

static const float RAY_DIRECTION_EPSILON = 1.0e-8;
static const float RAY_PARALLEL_INV_DIR = 1.0e8;
static const float RAY_T_MAX = 1.0e30;

float ray_safe_rcp(float direction_component) {
    if (abs(direction_component) > RAY_DIRECTION_EPSILON) {
        return 1.0 / direction_component;
    }
    return direction_component < 0.0 ? -RAY_PARALLEL_INV_DIR : RAY_PARALLEL_INV_DIR;
}

float3 ray_safe_inv_dir(float3 direction) {
    return float3(
        ray_safe_rcp(direction.x),
        ray_safe_rcp(direction.y),
        ray_safe_rcp(direction.z)
    );
}

int ray_step_component(float direction_component) {
    if (direction_component > RAY_DIRECTION_EPSILON) {
        return 1;
    }
    if (direction_component < -RAY_DIRECTION_EPSILON) {
        return -1;
    }
    return 0;
}

int3 ray_step_dir(float3 direction) {
    return int3(
        ray_step_component(direction.x),
        ray_step_component(direction.y),
        ray_step_component(direction.z)
    );
}

float ray_axis_t_delta(float direction_component, float inv_dir_component, float cell_size) {
    return ray_step_component(direction_component) == 0
        ? RAY_T_MAX
        : abs(inv_dir_component) * cell_size;
}

float ray_axis_t_max(
    float origin_component,
    float direction_component,
    float inv_dir_component,
    float positive_boundary,
    float negative_boundary
) {
    int step_component = ray_step_component(direction_component);
    if (step_component == 0) {
        return RAY_T_MAX;
    }
    float boundary = step_component > 0 ? positive_boundary : negative_boundary;
    return (boundary - origin_component) * inv_dir_component;
}

Ray make_ray(float3 origin, float3 direction) {
    Ray r;
    r.origin = origin;
    r.direction = direction;
    r.inv_dir = ray_safe_inv_dir(direction);
    return r;
}

static const uint VPT_MOTION_ID_INVALID = 0u;
static const uint VPT_MOTION_ID_STATIC_UCVH = 1u;

struct HitResult {
    bool  hit;
    float t;         // distance along ray
    float3 position; // world-space hit point
    float3 normal;   // face normal at hit (axis-aligned, from DDA step)
    uint  brick_id;  // which brick was hit
    uint3 local;     // local voxel coordinate within brick [0,7]³
    uint  motion_id; // stable motion source identity for future rigid hit reprojection
    VoxelCell cell;  // material + emissive data for the hit voxel
};

static const HitResult NO_HIT = { false, 1e30, float3(0), float3(0), 0xFFFFFFFFu, uint3(0), VPT_MOTION_ID_INVALID, {0, 0} };
