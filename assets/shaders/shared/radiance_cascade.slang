#pragma once
// Radiance Cascade shared helpers.
// Ported from Shadertoy M3ycWt (common.glsl).

#include "scene_common.slang"
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

float rc_probe_spacing(uint cascade_level) {
    return 8.0 * float(1u << cascade_level);
}

float3 rc_probe_world_position(uint3 probe_coord, uint cascade_level) {
    return (float3(probe_coord) + 0.5) * rc_probe_spacing(cascade_level);
}

float3 rc_world_to_c0_grid_position(float3 world_pos) {
    return world_pos / rc_probe_spacing(0u) - 0.5;
}

uint rc_probe_index(uint3 probe_coord, uint grid_dim, uint dirs_per_face, uint face, uint dir_index, uint buffer_offset) {
    uint probe_linear = (probe_coord.z * grid_dim + probe_coord.y) * grid_dim + probe_coord.x;
    return buffer_offset + probe_linear * 6u * dirs_per_face + face * dirs_per_face + dir_index;
}

// --- Geometry offset ---

bool rc_outside_geo(float3 sp,
                    StructuredBuffer<NodeL0> hierarchy_l0,
                    StructuredBuffer<BrickOccupancy> occupancy_buf,
                    uint3 brick_grid_size,
                    uint3 world_size) {
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

// --- integrate_probe (trilinear, multi-face blend, skip inside-geo probes) ---
// Reads primary face (matching surface normal) + 4 adjacent faces to capture
// light sources on the same wall plane (e.g., sconces next to a wall surface).

uint normal_to_face(float3 n) {
    float3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z) {
        return (n.x >= 0.0) ? 0u : 1u;
    }
    if (a.y >= a.z) {
        return (n.y >= 0.0) ? 2u : 3u;
    }
    return (n.z >= 0.0) ? 4u : 5u;
}

// Face normals lookup for cosine weighting.
static const float3 FACE_NORMALS[6] = {
    float3(1,0,0), float3(-1,0,0),
    float3(0,1,0), float3(0,-1,0),
    float3(0,0,1), float3(0,0,-1),
};

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

// Read probe with cosine-weighted face blending based on continuous normal.
// For axis-aligned normals, this reduces to single-face reading.
// For smooth normals (e.g. gradient-based), this smoothly blends between faces.
float3 read_probe_cosine(uint3 probe_coord, float3 normal, uint3 grid_size,
                         uint c0_offset, StructuredBuffer<float4> rc_probes) {
    float3 radiance = float3(0.0);
    float total_w = 0.0;
    for (uint f = 0; f < 6; f++) {
        float w = max(dot(FACE_NORMALS[f], normal), 0.0);
        if (w > 0.001) {
            radiance += read_probe_face(probe_coord, f, grid_size, c0_offset, rc_probes) * w;
            total_w += w;
        }
    }
    return (total_w > 0.001) ? radiance / total_w : float3(0.0);
}

float3 integrate_probe(float3 world_pos, float3 normal,
                       StructuredBuffer<float4> rc_probes,
                       uint c0_offset, uint3 c0_grid_size) {
    float3 grid_pos = rc_world_to_c0_grid_position(world_pos);
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

        // Detect inside-geo probes (w == -1) and reduce their weight.
        // These contribute zero radiance; full weight would darken areas near geometry.
        uint probe_base = c0_offset +
            ((clamped.z * c0_grid_size.y + clamped.y) * c0_grid_size.x + clamped.x)
            * 6u * 9u;
        float inside_geo_w = (rc_probes[probe_base].w < -0.5) ? 0.001 : 1.0;

        total_radiance += read_probe_cosine(clamped, normal, c0_grid_size, c0_offset, rc_probes) * trilinear_w * inside_geo_w;
        total_weight += trilinear_w * inside_geo_w;
    }

    return (total_weight > 0.0001) ? total_radiance / total_weight : float3(0.0);
}

float3 integrate_probe_fast(float3 world_pos, float3 normal,
                            StructuredBuffer<float4> rc_probes,
                            uint c0_offset, uint3 c0_grid_size) {
    float3 grid_pos = rc_world_to_c0_grid_position(world_pos);
    uint3 coord = uint3(clamp(int3(round(grid_pos)), int3(0), int3(c0_grid_size) - 1));
    uint face = normal_to_face(normal);

    uint probe_base = c0_offset +
        ((coord.z * c0_grid_size.y + coord.y) * c0_grid_size.x + coord.x)
        * 6u * 9u;
    if (rc_probes[probe_base].w < -0.5) {
        return float3(0.0);
    }

    return read_probe_face(coord, face, c0_grid_size, c0_offset, rc_probes);
}

float3 integrate_probe_quality(float3 world_pos, float3 normal,
                               StructuredBuffer<float4> rc_probes,
                               uint c0_offset, uint3 c0_grid_size,
                               uint quality) {
    if (quality == RC_PROBE_QUALITY_FAST) {
        return integrate_probe_fast(world_pos, normal, rc_probes, c0_offset, c0_grid_size);
    }
    return integrate_probe(world_pos, normal, rc_probes, c0_offset, c0_grid_size);
}
