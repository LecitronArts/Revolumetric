// Two-level hierarchical DDA for UCVH voxel traversal.
// Level 1: Brick-grid DDA (L0 hierarchy)
// Level 2: Brick-internal DDA (8^3 occupancy bitmask, Morton-order bit indexing)
//
// Key algorithms from references:
// - Amanatides-Woo DDA stepping (standard)
// - Tie-aware DDA: step every axis whose t_max reaches the next grid boundary
// - occupancy_count skip (from GDVoxelPlayground — if count==0, skip entire brick)
// - pixelToRay single-matrix ray gen (from DeadlockCode/voxel_ray_traversal)

#include "ray.slang"
#include "voxel_common.slang"

static const float DDA_TIE_EPSILON = 1.0e-6;
static const float DDA_GRID_BOUNDARY_EPSILON = 1.0e-5;
static const uint VOXEL_TRAVERSAL_NO_SKIP_BRICK = 0xffffffffu;

float dda_next_t(float3 t_max) {
    return min(t_max.x, min(t_max.y, t_max.z));
}

bool dda_axis_is_tied(float axis_t, float next_t) {
    return abs(axis_t - next_t) <= DDA_TIE_EPSILON * max(abs(next_t), 1.0);
}

float dda_adjust_boundary_position(float position, int step_dir) {
    if (step_dir == 0) {
        return position;
    }

    float nearest_boundary = round(position);
    if (abs(position - nearest_boundary) <= DDA_GRID_BOUNDARY_EPSILON) {
        return position + float(step_dir) * DDA_GRID_BOUNDARY_EPSILON;
    }
    return position;
}

int dda_start_coord(float position, int step_dir, int max_coord) {
    float adjusted = dda_adjust_boundary_position(position, step_dir);
    return int(clamp(floor(adjusted), 0.0, float(max_coord)));
}

int3 dda_start_coord3(float3 position, int3 step_dir, int3 max_coord) {
    return int3(
        dda_start_coord(position.x, step_dir.x, max_coord.x),
        dda_start_coord(position.y, step_dir.y, max_coord.y),
        dda_start_coord(position.z, step_dir.z, max_coord.z)
    );
}

float3 dda_step_normal(bool step_x, bool step_y, bool step_z, int3 step_dir) {
    if (step_x) {
        return float3(-float(step_dir.x), 0.0, 0.0);
    }
    if (step_y) {
        return float3(0.0, -float(step_dir.y), 0.0);
    }
    if (step_z) {
        return float3(0.0, 0.0, -float(step_dir.z));
    }
    return float3(0.0, 0.0, -1.0);
}

void dda_step_voxel(
    inout int3 coord,
    inout float3 t_max,
    float3 t_delta,
    int3 step_dir,
    inout float current_t,
    inout float3 hit_normal
) {
    float next_t = dda_next_t(t_max);
    bool step_x = dda_axis_is_tied(t_max.x, next_t);
    bool step_y = dda_axis_is_tied(t_max.y, next_t);
    bool step_z = dda_axis_is_tied(t_max.z, next_t);
    hit_normal = dda_step_normal(step_x, step_y, step_z, step_dir);
    current_t = next_t;
    if (step_x) {
        coord.x += step_dir.x;
        t_max.x += t_delta.x;
    }
    if (step_y) {
        coord.y += step_dir.y;
        t_max.y += t_delta.y;
    }
    if (step_z) {
        coord.z += step_dir.z;
        t_max.z += t_delta.z;
    }
}

void dda_step_brick(
    inout int3 brick_coord,
    inout float3 t_max,
    float3 t_delta,
    int3 step_dir
) {
    float current_t = 0.0;
    float3 hit_normal = float3(0.0);
    dda_step_voxel(brick_coord, t_max, t_delta, step_dir, current_t, hit_normal);
}

void dda_step_brick_with_t(
    inout int3 brick_coord,
    inout float3 t_max,
    float3 t_delta,
    int3 step_dir,
    inout float current_t
) {
    float3 hit_normal = float3(0.0);
    dda_step_voxel(brick_coord, t_max, t_delta, step_dir, current_t, hit_normal);
}

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

    float3 inv_dir = ray_safe_inv_dir(ray_dir);

    // Entry into [0, 8] box
    float2 box_t = intersect_aabb(ray_origin, inv_dir, AABB(float3(0), float3(8)));
    float t_enter = max(box_t.x, 0.0);
    if (t_enter >= box_t.y) return false; // ray misses brick

    int3 step_dir = ray_step_dir(ray_dir);
    float3 entry_pos = ray_origin + ray_dir * t_enter;
    int3 coord = dda_start_coord3(entry_pos, step_dir, int3(7));
    float current_t = t_enter;

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
    float3 t_delta = float3(
        ray_axis_t_delta(ray_dir.x, inv_dir.x, 1.0),
        ray_axis_t_delta(ray_dir.y, inv_dir.y, 1.0),
        ray_axis_t_delta(ray_dir.z, inv_dir.z, 1.0)
    ); // distance in t to cross one voxel
    // t_max: distance to next voxel boundary along each axis
    float3 t_max = float3(
        ray_axis_t_max(ray_origin.x, ray_dir.x, inv_dir.x, coord.x + 1, coord.x),
        ray_axis_t_max(ray_origin.y, ray_dir.y, inv_dir.y, coord.y + 1, coord.y),
        ray_axis_t_max(ray_origin.z, ray_dir.z, inv_dir.z, coord.z + 1, coord.z)
    );

    // Load the full BrickOccupancy once (80 bytes, fits in registers)
    BrickOccupancy occ = occupancy_buf[brick_id];

    // Max steps: 8+8+8 = 24 (diagonal through 8^3)
    for (int i = 0; i < 24; i++) {
        if (any(coord < int3(0)) || any(coord >= int3(8))) break;

        // Morton-order bit read (matches Rust BrickOccupancy storage)
        if (read_occupancy_bit(occ, uint3(coord))) {
            hit_local = uint3(coord);
            hit_t = current_t;
            return true;
        }

        dda_step_voxel(coord, t_max, t_delta, step_dir, current_t, hit_normal);
    }
    return false;
}

bool voxel_traversal_should_skip_shadow_hit(
    uint brick_id,
    uint3 local,
    uint skip_brick_id,
    uint3 skip_local
) {
    return brick_id == skip_brick_id && all(local == skip_local);
}

bool brick_any_hit(
    BrickOccupancy occ,
    uint brick_id,
    uint skip_brick_id,
    uint3 skip_local,
    float3 ray_origin,
    float3 ray_dir,
    float max_t
) {
    if (max_t <= 0.0) {
        return false;
    }

    float3 inv_dir = ray_safe_inv_dir(ray_dir);

    float2 box_t = intersect_aabb(ray_origin, inv_dir, AABB(float3(0), float3(8)));
    float t_enter = max(box_t.x, 0.0);
    if (t_enter >= box_t.y || t_enter >= max_t) {
        return false;
    }

    int3 step_dir = ray_step_dir(ray_dir);
    float3 entry_pos = ray_origin + ray_dir * t_enter;
    int3 coord = dda_start_coord3(entry_pos, step_dir, int3(7));
    float current_t = t_enter;
    float3 t_delta = float3(
        ray_axis_t_delta(ray_dir.x, inv_dir.x, 1.0),
        ray_axis_t_delta(ray_dir.y, inv_dir.y, 1.0),
        ray_axis_t_delta(ray_dir.z, inv_dir.z, 1.0)
    );
    float3 t_max = float3(
        ray_axis_t_max(ray_origin.x, ray_dir.x, inv_dir.x, coord.x + 1, coord.x),
        ray_axis_t_max(ray_origin.y, ray_dir.y, inv_dir.y, coord.y + 1, coord.y),
        ray_axis_t_max(ray_origin.z, ray_dir.z, inv_dir.z, coord.z + 1, coord.z)
    );

    for (int i = 0; i < 24 && current_t < max_t; i++) {
        if (any(coord < int3(0)) || any(coord >= int3(8))) break;

        if (read_occupancy_bit(occ, uint3(coord))) {
            uint3 local = uint3(coord);
            if (!voxel_traversal_should_skip_shadow_hit(brick_id, local, skip_brick_id, skip_local)) {
                return true;
            }
        }

        float3 unused_normal = float3(0.0);
        dda_step_voxel(coord, t_max, t_delta, step_dir, current_t, unused_normal);
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
    float3 entry_pos = ray.origin + ray.direction * t_enter;

    // Brick-grid DDA setup
    float3 inv_dir = ray.inv_dir;
    int3 step_dir = ray_step_dir(ray.direction);
    int3 igrid = int3(grid_size);
    int3 brick_coord = dda_start_coord3(entry_pos / 8.0, step_dir, igrid - int3(1));
    float3 t_delta = float3(
        ray_axis_t_delta(ray.direction.x, inv_dir.x, 8.0),
        ray_axis_t_delta(ray.direction.y, inv_dir.y, 8.0),
        ray_axis_t_delta(ray.direction.z, inv_dir.z, 8.0)
    ); // distance in t to cross one brick (8 voxels)
    float3 t_max = float3(
        ray_axis_t_max(ray.origin.x, ray.direction.x, inv_dir.x, (brick_coord.x + 1) * 8.0, brick_coord.x * 8.0),
        ray_axis_t_max(ray.origin.y, ray.direction.y, inv_dir.y, (brick_coord.y + 1) * 8.0, brick_coord.y * 8.0),
        ray_axis_t_max(ray.origin.z, ray.direction.z, inv_dir.z, (brick_coord.z + 1) * 8.0, brick_coord.z * 8.0)
    );

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
                    result.position = ray.origin + ray.direction * hit_t;
                    result.normal = hit_normal;
                    result.brick_id = node.brick_id;
                    result.local = hit_local;
                    result.cell = material_buf[node.brick_id * 512u + morton_encode(hit_local)];
                    return result;
                }
            }
        }

        dda_step_brick(brick_coord, t_max, t_delta, step_dir);
    }

    return NO_HIT;
}

bool trace_any_hit_ray_skip_voxel(
    Ray ray,
    float max_t,
    uint skip_brick_id,
    uint3 skip_local,
    StructuredBuffer<UcvhConfig> config_buf,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> occupancy_buf
) {
    if (max_t <= 0.0) {
        return false;
    }

    UcvhConfig config = config_buf[0];
    float3 grid_size = float3(config.brick_grid_size.xyz);
    float3 world_size = grid_size * 8.0;

    float2 world_t = intersect_aabb(ray.origin, ray.inv_dir, AABB(float3(0), world_size));
    if (world_t.x > world_t.y || world_t.y < 0.0) {
        return false;
    }

    float t_enter = max(world_t.x, 0.0);
    if (t_enter >= max_t) {
        return false;
    }

    float3 entry_pos = ray.origin + ray.direction * t_enter;
    float3 inv_dir = ray.inv_dir;
    int3 step_dir = ray_step_dir(ray.direction);
    int3 igrid = int3(grid_size);
    int3 brick_coord = dda_start_coord3(entry_pos / 8.0, step_dir, igrid - int3(1));
    float3 t_delta = float3(
        ray_axis_t_delta(ray.direction.x, inv_dir.x, 8.0),
        ray_axis_t_delta(ray.direction.y, inv_dir.y, 8.0),
        ray_axis_t_delta(ray.direction.z, inv_dir.z, 8.0)
    );
    float3 t_max = float3(
        ray_axis_t_max(ray.origin.x, ray.direction.x, inv_dir.x, (brick_coord.x + 1) * 8.0, brick_coord.x * 8.0),
        ray_axis_t_max(ray.origin.y, ray.direction.y, inv_dir.y, (brick_coord.y + 1) * 8.0, brick_coord.y * 8.0),
        ray_axis_t_max(ray.origin.z, ray.direction.z, inv_dir.z, (brick_coord.z + 1) * 8.0, brick_coord.z * 8.0)
    );

    int max_steps = igrid.x + igrid.y + igrid.z;
    float current_t = t_enter;
    for (int i = 0; i < max_steps && i < 256 && current_t < max_t; i++) {
        if (any(brick_coord < int3(0)) || any(brick_coord >= igrid)) {
            break;
        }

        uint l0_idx = uint(brick_coord.x)
                    + uint(brick_coord.y) * uint(igrid.x)
                    + uint(brick_coord.z) * uint(igrid.x) * uint(igrid.y);
        NodeL0 node = hierarchy_l0[l0_idx];

        if (!node_l0_is_empty(node)) {
            BrickOccupancy occ = occupancy_buf[node.brick_id];
            if (occ.count > 0u) {
                float3 brick_origin = float3(brick_coord) * 8.0;
                float3 local_origin = ray.origin - brick_origin;
                if (brick_any_hit(
                    occ,
                    node.brick_id,
                    skip_brick_id,
                    skip_local,
                    local_origin,
                    ray.direction,
                    max_t
                )) {
                    return true;
                }
            }
        }

        dda_step_brick_with_t(brick_coord, t_max, t_delta, step_dir, current_t);
    }

    return false;
}

bool trace_any_hit_ray(
    Ray ray,
    float max_t,
    StructuredBuffer<UcvhConfig> config_buf,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> occupancy_buf
) {
    return trace_any_hit_ray_skip_voxel(
        ray,
        max_t,
        VOXEL_TRAVERSAL_NO_SKIP_BRICK,
        uint3(0),
        config_buf,
        hierarchy_l0,
        occupancy_buf
    );
}
