#pragma once
// Voxel implementation of the backend-agnostic scene tracing contract.

#include "scene_trace_common.slang"
#include "voxel_traverse.slang"
#include "material_common.slang"

bool scene_outside_geometry(
    float3 sp,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    uint3 brick_grid_size,
    uint3 world_size
) {
    int3 p = int3(floor(sp));
    if (any(p < int3(0)) || any(p >= int3(world_size))) return true;
    int3 brick_coord = p / 8;
    if (any(brick_coord < int3(0)) || any(brick_coord >= int3(brick_grid_size))) return true;
    uint l0_idx = uint(brick_coord.x)
                + uint(brick_coord.y) * brick_grid_size.x
                + uint(brick_coord.z) * brick_grid_size.x * brick_grid_size.y;
    NodeL0 node = hierarchy_l0[l0_idx];
    if (node_l0_is_empty(node)) return true;
    BrickOccupancy occ = occupancy_buf[node.brick_id];
    uint3 local = uint3(p) % 8u;
    return !read_occupancy_bit(occ, local);
}

float3 scene_geometry_offset(
    float3 vertex,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    uint3 brick_grid_size,
    uint3 world_size
) {
    for (float x = -0.5; x < 1.0; x += 1.0) {
        for (float y = -0.5; y < 1.0; y += 1.0) {
            for (float z = -0.5; z < 1.0; z += 1.0) {
                if (scene_outside_geometry(vertex + float3(x, y, z), hierarchy_l0, occupancy_buf, brick_grid_size, world_size))
                    return float3(x, y, z) * 0.1;
            }
        }
    }
    return float3(0.001);
}

bool scene_find_nearby_empty_probe_position(
    float3 center,
    float search_radius,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    uint3 brick_grid_size,
    uint3 world_size,
    out float3 probe_pos
) {
    if (scene_outside_geometry(center, hierarchy_l0, occupancy_buf, brick_grid_size, world_size)) {
        probe_pos = center;
        return true;
    }

    float radius = max(search_radius, 0.5);
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 offset_i = int3(x, y, z);
                if (all(offset_i == int3(0))) continue;
                float3 candidate = center + float3(offset_i) * radius;
                if (scene_outside_geometry(candidate, hierarchy_l0, occupancy_buf, brick_grid_size, world_size)) {
                    probe_pos = candidate;
                    return true;
                }
            }
        }
    }

    probe_pos = center;
    return false;
}

float3 scene_voxel_occupancy_gradient_normal(
    float3 position,
    float3 fallback_normal,
    uint3 brick_grid_size,
    uint3 world_size,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> brick_occupancy
) {
    int3 ip = int3(floor(position));
    float3 grad;
    grad.x = float(!scene_outside_geometry(float3(ip) + float3(1,0,0), hierarchy_l0, brick_occupancy, brick_grid_size, world_size))
           - float(!scene_outside_geometry(float3(ip) + float3(-1,0,0), hierarchy_l0, brick_occupancy, brick_grid_size, world_size));
    grad.y = float(!scene_outside_geometry(float3(ip) + float3(0,1,0), hierarchy_l0, brick_occupancy, brick_grid_size, world_size))
           - float(!scene_outside_geometry(float3(ip) + float3(0,-1,0), hierarchy_l0, brick_occupancy, brick_grid_size, world_size));
    grad.z = float(!scene_outside_geometry(float3(ip) + float3(0,0,1), hierarchy_l0, brick_occupancy, brick_grid_size, world_size))
           - float(!scene_outside_geometry(float3(ip) + float3(0,0,-1), hierarchy_l0, brick_occupancy, brick_grid_size, world_size));
    return (length(grad) > 0.001) ? normalize(-grad) : fallback_normal;
}

SceneHit scene_hit_from_voxel_hit(HitResult voxel_hit) {
    if (!voxel_hit.hit) return SCENE_NO_HIT;

    SceneHit scene_hit;
    scene_hit.hit = true;
    scene_hit.t = voxel_hit.t;
    scene_hit.position = voxel_hit.position;
    scene_hit.normal = voxel_hit.normal;
    scene_hit.albedo = material_cell_albedo(voxel_hit.cell);
    scene_hit.emissive = material_emissive(voxel_hit.cell);
    scene_hit.surface_flags = SCENE_SURFACE_FLAG_VOXEL_BACKED;
    return scene_hit;
}

SceneHit trace_scene_ray(
    Ray ray,
    RWStructuredBuffer<uint> traversal_stats,
    uint traversal_stats_slot,
    StructuredBuffer<UcvhConfig> config_buf,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<NodeLN> hierarchy_l1,
    StructuredBuffer<NodeLN> hierarchy_l2,
    StructuredBuffer<NodeLN> hierarchy_l3,
    StructuredBuffer<NodeLN> hierarchy_l4,
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    StructuredBuffer<VoxelCell> material_buf
) {
    return scene_hit_from_voxel_hit(trace_primary_ray(
        ray,
        traversal_stats,
        traversal_stats_slot,
        config_buf,
        hierarchy_l0,
        hierarchy_l1,
        hierarchy_l2,
        hierarchy_l3,
        hierarchy_l4,
        occupancy_buf,
        material_buf
    ));
}
