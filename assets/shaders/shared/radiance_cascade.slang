#pragma once
// Radiance Cascade shared helpers.
// Ported from Shadertoy M3ycWt (common.glsl).

#include "voxel_common.slang"

static const float RC_PI = 3.141592653;

// --- Probe direction mapping ---

float3 ComputeDirEven(float2 uv, float probeSize) {
    float2 probeRel = uv - probeSize * 0.5;
    float probeThetai = max(abs(probeRel.x), abs(probeRel.y));
    float probeTheta = probeThetai / probeSize * RC_PI;
    float probePhi = 0.0;
    if (probeRel.x + 0.5 > probeThetai && probeRel.y - 0.5 > -probeThetai) {
        probePhi = probeRel.x - probeRel.y;
    } else if (probeRel.y - 0.5 < -probeThetai && probeRel.x - 0.5 > -probeThetai) {
        probePhi = probeThetai * 2.0 - probeRel.y - probeRel.x;
    } else if (probeRel.x - 0.5 < -probeThetai && probeRel.y + 0.5 < probeThetai) {
        probePhi = probeThetai * 4.0 - probeRel.x + probeRel.y;
    } else if (probeRel.y + 0.5 > probeThetai && probeRel.x + 0.5 < probeThetai) {
        probePhi = probeThetai * 8.0 - (probeRel.y - probeRel.x);
    }
    probePhi = probePhi * RC_PI * 2.0 / (4.0 + 8.0 * floor(probeThetai));
    return float3(float2(sin(probePhi), cos(probePhi)) * sin(probeTheta), cos(probeTheta));
}

float3 ComputeDir(float2 uv, float probeSize) {
    if (probeSize > 4.5) return ComputeDirEven(uv, probeSize);
    float2 probeRel = uv - 1.5;
    if (length(probeRel) < 0.1) return float3(0.0, 0.0, 1.0);
    float probePhi = atan2(probeRel.x, probeRel.y) + RC_PI * 1.75;
    float probeTheta = RC_PI * 0.25;
    return float3(float2(sin(probePhi), cos(probePhi)) * sin(probeTheta), cos(probeTheta));
}

float2 ProjectDir(float3 dir, float probeSize) {
    if (dir.z <= 0.0) return float2(-1.0);
    float thetai = min(floor((1.0 - acos(length(dir.xy) / length(dir)) / (RC_PI * 0.5)) * (probeSize * 0.5)), probeSize * 0.5 - 1.0);
    float phiF = atan2(-dir.x, -dir.y);
    float phiI = floor((phiF / RC_PI * 0.5 + 0.5) * (4.0 + 8.0 * thetai) + 0.5) + 0.5;
    float2 phiUV;
    float phiLen = 2.0 * thetai + 1.0;
    float sideLen = phiLen + 1.0;
    if (phiI < phiLen) phiUV = float2(sideLen - 0.5, sideLen - phiI);
    else if (phiI < phiLen * 2.0) phiUV = float2(sideLen - (phiI - phiLen), 0.5);
    else if (phiI < phiLen * 3.0) phiUV = float2(0.5, phiI - phiLen * 2.0);
    else phiUV = float2(phiI - phiLen * 3.0, sideLen - 0.5);
    return float2((probeSize - sideLen) * 0.5) + phiUV;
}

// --- Probe face TBN ---

void get_face_tbn(uint face, out float3 normal, out float3 tangent, out float3 bitangent) {
    if (face == 0)      { normal = float3( 1, 0, 0); tangent = float3(0, 0, 1); bitangent = float3(0, 1, 0); }
    else if (face == 1) { normal = float3(-1, 0, 0); tangent = float3(0, 0, 1); bitangent = float3(0, 1, 0); }
    else if (face == 2) { normal = float3(0,  1, 0); tangent = float3(1, 0, 0); bitangent = float3(0, 0, 1); }
    else if (face == 3) { normal = float3(0, -1, 0); tangent = float3(1, 0, 0); bitangent = float3(0, 0, 1); }
    else if (face == 4) { normal = float3(0, 0,  1); tangent = float3(1, 0, 0); bitangent = float3(0, 1, 0); }
    else                { normal = float3(0, 0, -1); tangent = float3(1, 0, 0); bitangent = float3(0, 1, 0); }
}

// --- Probe buffer indexing ---

uint rc_probe_index(uint3 probe_coord, uint grid_dim, uint dirs_per_face, uint face, uint dir_index, uint buffer_offset) {
    uint probe_linear = (probe_coord.z * grid_dim + probe_coord.y) * grid_dim + probe_coord.x;
    return buffer_offset + probe_linear * 6u * dirs_per_face + face * dirs_per_face + dir_index;
}

// --- Geometry offset ---

bool rc_outside_geo(float3 sp,
                    StructuredBuffer<NodeL0> hierarchy_l0,
                    StructuredBuffer<BrickOccupancy> occupancy_buf,
                    uint grid_dim_l0) {
    int3 p = int3(floor(sp));
    if (any(p < int3(0)) || any(p >= int3(128))) return true;
    int3 brick_coord = p / 8;
    uint l0_idx = uint(brick_coord.x) + uint(brick_coord.y) * grid_dim_l0 + uint(brick_coord.z) * grid_dim_l0 * grid_dim_l0;
    NodeL0 node = hierarchy_l0[l0_idx];
    if (node_l0_is_empty(node)) return true;
    BrickOccupancy occ = occupancy_buf[node.brick_id];
    uint3 local = uint3(p) % 8u;
    return !read_occupancy_bit(occ, local);
}

float3 rc_geo_offset(float3 vertex,
                     StructuredBuffer<NodeL0> hierarchy_l0,
                     StructuredBuffer<BrickOccupancy> occupancy_buf,
                     uint grid_dim_l0) {
    for (float x = -0.5; x < 1.0; x += 1.0) {
        for (float y = -0.5; y < 1.0; y += 1.0) {
            for (float z = -0.5; z < 1.0; z += 1.0) {
                if (rc_outside_geo(vertex + float3(x, y, z), hierarchy_l0, occupancy_buf, grid_dim_l0))
                    return float3(x, y, z) * 0.1;
            }
        }
    }
    return float3(0.001);
}

// --- integrate_probe (visibility-weighted trilinear) ---

uint normal_to_face(float3 n) {
    return encode_normal_id(n);
}

float3 read_probe_face(uint3 probe_coord, uint face, uint3 grid_size,
                       uint c0_offset, StructuredBuffer<float4> rc_probes) {
    uint base = c0_offset +
        ((probe_coord.z * grid_size.y + probe_coord.y) * grid_size.x + probe_coord.x)
        * 6u * 9u + face * 9u;
    float3 total = float3(0.0);
    for (uint i = 0; i < 9; i++) {
        total += rc_probes[base + i].xyz;
    }
    return total;
}

float probe_visibility(uint3 probe_coord, float3 world_pos, uint face,
                       uint3 grid_size, uint c0_offset, float probe_spacing,
                       StructuredBuffer<float4> rc_probes) {
    float3 probe_world = (float3(probe_coord) + 0.5) * probe_spacing;
    float dist = length(world_pos - probe_world);
    uint center_idx = c0_offset +
        ((probe_coord.z * grid_size.y + probe_coord.y) * grid_size.x + probe_coord.x)
        * 6u * 9u + face * 9u + 4u;
    float ray_dist = rc_probes[center_idx].w;
    return (ray_dist < dist - 0.5) ? 0.00001 : 1.0;
}

float3 integrate_probe(float3 world_pos, float3 normal,
                       StructuredBuffer<float4> rc_probes,
                       uint c0_offset, uint3 c0_grid_size) {
    uint face = normal_to_face(normal);
    float spacing = 8.0;

    float3 grid_pos = world_pos / spacing - 0.5;
    int3 base_coord = int3(floor(grid_pos));
    float3 frac_pos = grid_pos - float3(base_coord);

    float3 total_radiance = float3(0.0);
    float total_weight = 0.0;

    for (uint i = 0; i < 8; i++) {
        int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 coord = base_coord + offset;
        uint3 clamped = uint3(clamp(coord, int3(0), int3(c0_grid_size) - 1));

        float3 w = lerp(1.0 - frac_pos, frac_pos, float3(offset));
        float trilinear_w = w.x * w.y * w.z;

        float vis_w = probe_visibility(clamped, world_pos, face,
                                       c0_grid_size, c0_offset, spacing, rc_probes);

        float weight = trilinear_w * vis_w;
        total_radiance += read_probe_face(clamped, face, c0_grid_size, c0_offset, rc_probes) * weight;
        total_weight += weight;
    }

    return (total_weight > 0.0001) ? total_radiance / total_weight : float3(0.0);
}
