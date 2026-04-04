// Two-level hierarchical DDA for UCVH voxel traversal.
// Level 1: Brick-grid DDA (L0 hierarchy)
// Level 2: Brick-internal DDA (8^3 occupancy bitmask, Morton-order bit indexing)
//
// Key algorithms from references:
// - Amanatides-Woo DDA stepping (standard)
// - Branching DDA over branchless (from DeadlockCode/voxel_ray_traversal, ~5% faster)
// - occupancy_count skip (from GDVoxelPlayground — if count==0, skip entire brick)
// - pixelToRay single-matrix ray gen (from DeadlockCode/voxel_ray_traversal)

#include "ray.slang"
#include "voxel_common.slang"

// --- Brick-internal DDA (Level 2) ---
// Amanatides-Woo DDA through 8^3 brick occupancy.
// Uses read_occupancy_bit() from voxel_common.slang (Morton-order bit indexing).
// Returns true if a voxel is hit, populating hit_local and face normal.
bool brick_dda(
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    uint brick_id,
    float3 ray_origin,    // in brick-local space [0, 8]
    float3 ray_dir,
    out uint3 hit_local,
    out float3 hit_normal,
    out float hit_t
) {
    hit_local = uint3(0);
    hit_normal = float3(0);
    hit_t = 0.0;

    float3 inv_dir = float3(
        abs(ray_dir.x) > 1e-8 ? 1.0 / ray_dir.x : sign(ray_dir.x) * 1e8,
        abs(ray_dir.y) > 1e-8 ? 1.0 / ray_dir.y : sign(ray_dir.y) * 1e8,
        abs(ray_dir.z) > 1e-8 ? 1.0 / ray_dir.z : sign(ray_dir.z) * 1e8
    );

    // Entry into [0, 8] box
    float2 box_t = intersect_aabb(ray_origin, inv_dir, AABB(float3(0), float3(8)));
    float t_enter = max(box_t.x, 0.0);
    if (t_enter >= box_t.y) return false; // ray misses brick

    float3 pos = ray_origin + ray_dir * (t_enter + 0.001);
    int3 coord = int3(clamp(floor(pos), float3(0), float3(7)));
    int3 step_dir = int3(sign(ray_dir));

    // Entry face normal: use brick AABB slab intersection to find which face was entered.
    // The axis with the largest t_near is the entry face (last axis to "enter").
    float3 t_lo = -ray_origin * inv_dir;          // t to reach face at 0
    float3 t_hi = (float3(8) - ray_origin) * inv_dir; // t to reach face at 8
    float3 t_near = min(t_lo, t_hi);
    if (t_near.x >= t_near.y && t_near.x >= t_near.z)
        hit_normal = float3(-float(step_dir.x), 0, 0);
    else if (t_near.y >= t_near.z)
        hit_normal = float3(0, -float(step_dir.y), 0);
    else
        hit_normal = float3(0, 0, -float(step_dir.z));
    float3 t_delta = abs(inv_dir);   // distance in t to cross one voxel
    // t_max: distance to next voxel boundary along each axis
    float3 t_max = float3(
        ((step_dir.x > 0 ? coord.x + 1 : coord.x) - pos.x) * inv_dir.x,
        ((step_dir.y > 0 ? coord.y + 1 : coord.y) - pos.y) * inv_dir.y,
        ((step_dir.z > 0 ? coord.z + 1 : coord.z) - pos.z) * inv_dir.z
    );

    // Load the full BrickOccupancy once (80 bytes, fits in registers)
    BrickOccupancy occ = occupancy_buf[brick_id];

    // Max steps: 8+8+8 = 24 (diagonal through 8^3)
    for (int i = 0; i < 24; i++) {
        if (any(coord < int3(0)) || any(coord >= int3(8))) break;

        // Morton-order bit read (matches Rust BrickOccupancy storage)
        if (read_occupancy_bit(occ, uint3(coord))) {
            hit_local = uint3(coord);
            hit_t = t_enter + min(min(t_max.x, t_max.y), t_max.z);
            return true;
        }

        // Branching DDA (faster than branchless per DeadlockCode benchmarks)
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                hit_normal = float3(-float(step_dir.x), 0, 0);
                coord.x += step_dir.x;
                t_max.x += t_delta.x;
            } else {
                hit_normal = float3(0, 0, -float(step_dir.z));
                coord.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                hit_normal = float3(0, -float(step_dir.y), 0);
                coord.y += step_dir.y;
                t_max.y += t_delta.y;
            } else {
                hit_normal = float3(0, 0, -float(step_dir.z));
                coord.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        }
    }
    return false;
}

// --- Primary trace through UCVH (Level 1 + Level 2) ---
// DDA through brick grid, descending into occupied bricks.
HitResult trace_primary_ray(
    Ray ray,
    StructuredBuffer<UcvhConfig> config_buf,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    StructuredBuffer<VoxelCell> material_buf
) {
    UcvhConfig config = config_buf[0];
    float3 grid_size = float3(config.brick_grid_size.xyz);
    float3 world_size = grid_size * 8.0; // each brick = 8 voxels

    // Intersect ray with world AABB [0, world_size]
    float2 world_t = intersect_aabb(ray.origin, ray.inv_dir, AABB(float3(0), world_size));
    if (world_t.x > world_t.y || world_t.y < 0.0)
        return NO_HIT;

    float t_enter = max(world_t.x, 0.0);
    float3 pos = ray.origin + ray.direction * (t_enter + 0.001);

    // Brick-grid DDA setup
    float3 inv_dir = ray.inv_dir;
    int3 brick_coord = int3(clamp(floor(pos / 8.0), float3(0), grid_size - 1.0));
    int3 step_dir = int3(sign(ray.direction));
    float3 t_delta = abs(inv_dir) * 8.0; // distance in t to cross one brick (8 voxels)
    float3 t_max = float3(
        ((step_dir.x > 0 ? (brick_coord.x + 1) * 8.0 : brick_coord.x * 8.0) - pos.x) * inv_dir.x,
        ((step_dir.y > 0 ? (brick_coord.y + 1) * 8.0 : brick_coord.y * 8.0) - pos.y) * inv_dir.y,
        ((step_dir.z > 0 ? (brick_coord.z + 1) * 8.0 : brick_coord.z * 8.0) - pos.z) * inv_dir.z
    );

    int3 igrid = int3(grid_size);
    float3 brick_normal = float3(0);

    // Max brick steps: grid diagonal
    int max_steps = igrid.x + igrid.y + igrid.z;
    for (int i = 0; i < max_steps && i < 256; i++) {
        if (any(brick_coord < int3(0)) || any(brick_coord >= igrid))
            break;

        // Lookup L0 node
        uint l0_idx = uint(brick_coord.x)
                    + uint(brick_coord.y) * uint(igrid.x)
                    + uint(brick_coord.z) * uint(igrid.x) * uint(igrid.y);
        NodeL0 node = hierarchy_l0[l0_idx];

        if (!node_l0_is_empty(node)) {
            // Quick check: is brick occupied at all?
            BrickOccupancy occ = occupancy_buf[node.brick_id];
            if (occ.count > 0u) {
                // Compute ray in brick-local space
                float3 brick_origin = float3(brick_coord) * 8.0;
                float3 local_origin = ray.origin - brick_origin;

                uint3 hit_local;
                float3 hit_normal;
                float hit_t;
                if (brick_dda(occupancy_buf, node.brick_id, local_origin, ray.direction,
                              hit_local, hit_normal, hit_t)) {
                    HitResult result;
                    result.hit = true;
                    result.t = hit_t;
                    result.position = brick_origin + float3(hit_local) + 0.5;
                    result.normal = hit_normal;
                    result.brick_id = node.brick_id;
                    result.local = hit_local;
                    return result;
                }
            }
        }

        // Step to next brick (branching DDA)
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                brick_coord.x += step_dir.x;
                t_max.x += t_delta.x;
            } else {
                brick_coord.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                brick_coord.y += step_dir.y;
                t_max.y += t_delta.y;
            } else {
                brick_coord.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        }
    }

    return NO_HIT;
}
