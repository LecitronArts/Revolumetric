#pragma once
// GPU struct definitions for UCVH data — must match Rust repr(C) layouts exactly.
// Consumed by all voxel shader passes via #include "voxel_common.slang"

struct UcvhConfig {
    uint4 world_size;       // xyz + pad
    uint4 brick_grid_size;  // xyz + pad
    uint  brick_capacity;
    uint  allocated_bricks;
    uint  _pad0;
    uint  _pad1;
};

// 80 bytes per brick. bits[] indexed by Morton order.
struct BrickOccupancy {
    uint bits[16];  // 512 bits = 16 × 32
    uint count;     // number of solid voxels
    uint _pad[3];   // pad to 80 bytes (16-byte alignment)
};

// 8 bytes per voxel, stored in Morton order within each brick.
struct VoxelCell {
    uint material_flags; // low 16: material id, high 16: flags
    uint emissive_pad;   // low 24: emissive RGB (8 bits each), high 8: padding
};

// 8 bytes per L0 node.
struct NodeL0 {
    uint brick_id;    // index into BrickPool, or 0xFFFFFFFF if empty
    uint flags_pad;   // low 16: flags (bit 0 = has_solid), high 16: padding
};

// 4 bytes per L1-L4 node.
struct NodeLN {
    uint packed; // low 8: child_mask, next 8: flags (bit 0 = any_solid), high 16: padding
};

// --- Accessor helpers ---

uint voxel_material(VoxelCell cell) {
    return cell.material_flags & 0xFFFFu;
}

uint voxel_flags(VoxelCell cell) {
    return cell.material_flags >> 16u;
}

float3 voxel_emissive(VoxelCell cell) {
    float r = float(cell.emissive_pad & 0xFFu) / 255.0;
    float g = float((cell.emissive_pad >> 8u) & 0xFFu) / 255.0;
    float b = float((cell.emissive_pad >> 16u) & 0xFFu) / 255.0;
    return float3(r, g, b);
}

bool node_l0_is_empty(NodeL0 node) {
    return node.brick_id == 0xFFFFFFFFu;
}

uint node_ln_child_mask(NodeLN node) {
    return node.packed & 0xFFu;
}

// Morton encode for 8^3 brick (3-bit per axis, interleaved x-y-z)
uint morton_encode(uint3 p) {
    uint m = 0u;
    for (uint bit = 0; bit < 3; bit++) {
        m |= ((p.x >> bit) & 1u) << (3u * bit);
        m |= ((p.y >> bit) & 1u) << (3u * bit + 1u);
        m |= ((p.z >> bit) & 1u) << (3u * bit + 2u);
    }
    return m;
}

// Read a single voxel's occupancy bit from a BrickOccupancy.
// Uses Morton-order bit indexing.
bool read_occupancy_bit(BrickOccupancy brick, uint3 local) {
    uint m = morton_encode(local);
    uint word = m / 32u;
    uint bit = m % 32u;
    return (brick.bits[word] & (1u << bit)) != 0u;
}

// Encodes an axis-aligned normal to a 0-5 ID.
// Assumes input is a unit axis-aligned vector from DDA face normals.
// Fallback to -Z (5) for degenerate zero normals (should not occur in practice).
//   0 = +X    1 = -X
//   2 = +Y    3 = -Y
//   4 = +Z    5 = -Z
uint encode_normal_id(float3 n) {
    if (n.x >  0.5) return 0;
    if (n.x < -0.5) return 1;
    if (n.y >  0.5) return 2;
    if (n.y < -0.5) return 3;
    if (n.z >  0.5) return 4;
    return 5; // -Z
}
