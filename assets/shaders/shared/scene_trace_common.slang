#pragma once
// Backend-agnostic scene tracing contract used by Radiance Cascades.
//
// Scene backends convert their native hit type (voxel, mesh BVH, SDF, hardware RT,
// etc.) into this surface representation. RC trace and full-resolution lighting
// should consume SceneHit, not backend-native material or geometry types.

static const uint SCENE_SURFACE_FLAG_NONE = 0u;
static const uint SCENE_SURFACE_FLAG_VOXEL_BACKED = 1u << 0;

struct SceneHit {
    bool hit;
    float t;
    float3 position;
    float3 normal;
    float3 albedo;
    float3 emissive;
    uint surface_flags;
};

static const SceneHit SCENE_NO_HIT = {
    false,
    1e30,
    float3(0.0),
    float3(0.0),
    float3(0.0),
    float3(0.0),
    SCENE_SURFACE_FLAG_NONE
};

bool scene_hit_has_emissive(SceneHit hit) {
    return hit.emissive.x + hit.emissive.y + hit.emissive.z > 0.001;
}

bool scene_hit_is_voxel_backed(SceneHit hit) {
    return (hit.surface_flags & SCENE_SURFACE_FLAG_VOXEL_BACKED) != 0u;
}
