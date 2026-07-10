#pragma once

#include "math.slang"
#include "rt_history_common.slang"

static const float RT_SURFACE_MISS_T = 1.0e20;

struct RtAabb {
    float min_x;
    float min_y;
    float min_z;
    float max_x;
    float max_y;
    float max_z;
};

struct RtSurfaceIntersectionAttributes {
    uint brick_id;
    uint packed_local_normal;
    float3 object_position;
    float3 object_normal;
};

uint rt_surface_axis_normal_code(float3 normal) {
    if (normal.x < -0.5) return 0u;
    if (normal.x > 0.5) return 1u;
    if (normal.y < -0.5) return 2u;
    if (normal.y > 0.5) return 3u;
    if (normal.z < -0.5) return 4u;
    return 5u;
}

uint rt_surface_pack_local_normal(uint3 local, float3 normal) {
    uint packed_local = (local.x & 7u)
                      | ((local.y & 7u) << 3u)
                      | ((local.z & 7u) << 6u);
    return packed_local | (rt_surface_axis_normal_code(normal) << 9u);
}

uint3 rt_surface_unpack_local(uint packed) {
    return uint3(packed & 7u, (packed >> 3u) & 7u, (packed >> 6u) & 7u);
}

float3 rt_surface_unpack_axis_normal(uint packed) {
    uint code = (packed >> 9u) & 7u;
    if (code == 0u) return float3(-1.0, 0.0, 0.0);
    if (code == 1u) return float3(1.0, 0.0, 0.0);
    if (code == 2u) return float3(0.0, -1.0, 0.0);
    if (code == 3u) return float3(0.0, 1.0, 0.0);
    if (code == 4u) return float3(0.0, 0.0, -1.0);
    return float3(0.0, 0.0, 1.0);
}

struct RtSurfacePayload {
    uint hit_kind;
    float hit_t;
    uint primitive_index;
    uint instance_id;
    float3 position;
    float3 normal;
    float3 albedo;
    float roughness;
    uint material_id;
    uint brick_id;
    uint3 local;
    float3 ray_direction;
    float3 emissive_radiance;
};

float3 rt_aabb_min(RtAabb aabb) {
    return float3(aabb.min_x, aabb.min_y, aabb.min_z);
}

float3 rt_aabb_max(RtAabb aabb) {
    return float3(aabb.max_x, aabb.max_y, aabb.max_z);
}

AABB rt_aabb_to_aabb(RtAabb aabb) {
    return AABB(rt_aabb_min(aabb), rt_aabb_max(aabb));
}

float3 rt_surface_safe_inv_dir(float3 direction) {
    float3 safe_dir = select(abs(direction) < 1.0e-8, sign(direction) * 1.0e-8, direction);
    safe_dir = select(safe_dir == float3(0.0), float3(1.0e-8), safe_dir);
    return 1.0 / safe_dir;
}

float3 rt_surface_aabb_normal(float3 position, RtAabb aabb) {
    float3 mn = rt_aabb_min(aabb);
    float3 mx = rt_aabb_max(aabb);
    float3 d_min = abs(position - mn);
    float3 d_max = abs(mx - position);
    float best = d_min.x;
    float3 normal = float3(-1.0, 0.0, 0.0);
    if (d_max.x < best) {
        best = d_max.x;
        normal = float3(1.0, 0.0, 0.0);
    }
    if (d_min.y < best) {
        best = d_min.y;
        normal = float3(0.0, -1.0, 0.0);
    }
    if (d_max.y < best) {
        best = d_max.y;
        normal = float3(0.0, 1.0, 0.0);
    }
    if (d_min.z < best) {
        best = d_min.z;
        normal = float3(0.0, 0.0, -1.0);
    }
    if (d_max.z < best) {
        normal = float3(0.0, 0.0, 1.0);
    }
    return normal;
}

RtSurfacePayload make_rt_surface_payload(float3 ray_direction) {
    RtSurfacePayload payload;
    payload.hit_kind = RT_SURFACE_HIT_KIND_MISS;
    payload.hit_t = RT_SURFACE_MISS_T;
    payload.primitive_index = 0xffffffffu;
    payload.instance_id = 0xffffffffu;
    payload.position = float3(0.0);
    payload.normal = -ray_direction;
    payload.albedo = float3(0.0);
    payload.roughness = 1.0;
    payload.material_id = 0u;
    payload.brick_id = 0xffffffffu;
    payload.local = uint3(0u);
    payload.ray_direction = ray_direction;
    payload.emissive_radiance = float3(0.0);
    return payload;
}

float4 rt_surface_invalid_motion_history() {
    return float4(0.0);
}

RtSurfacePixel surface_pixel_from_payload(RtSurfacePayload payload) {
    RtSurfacePixel pixel;
    pixel.position_depth = float4(payload.position, payload.hit_t);
    pixel.normal_roughness = float4(normalize(payload.normal), payload.roughness);
    pixel.albedo_material = float4(payload.albedo, float(payload.material_id));
    pixel.motion_history = rt_surface_invalid_motion_history();
    pixel.view_direction_background = float4(
        normalize(payload.ray_direction),
        payload.hit_kind == RT_SURFACE_HIT_KIND_MISS ? 1.0 : 0.0
    );
    pixel.linear_depth = payload.hit_t;
    pixel.material_id = payload.material_id;
    pixel.history_confidence = payload.hit_kind == RT_SURFACE_HIT_KIND_VOXEL ? 1.0 : 0.0;
    pixel.hit_kind = payload.hit_kind;
    pixel.brick_id = payload.brick_id;
    pixel.local = payload.local;
    pixel.emissive_radiance = float4(payload.emissive_radiance, 0.0);
    return pixel;
}
