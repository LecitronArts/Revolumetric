#pragma once
// Shared material helpers for voxel shader passes.

#include "voxel_common.slang"

static const float3 MATERIAL_ALBEDO[8] = {
    float3(0.7, 0.7, 0.7),       // 0: default
    float3(0.65, 0.63, 0.60),    // 1: stone
    float3(0.70, 0.20, 0.20),    // 2: red cloth
    float3(0.20, 0.70, 0.20),    // 3: green cloth
    float3(0.35, 0.42, 0.65),    // 4: blue
    float3(0.55, 0.40, 0.30),    // 5: brick
    float3(0.40, 0.37, 0.34),    // 6: dark stone
    float3(0.45, 0.30, 0.18),    // 7: wood
};

float3 material_albedo(uint material_id) {
    return MATERIAL_ALBEDO[min(material_id, 7u)];
}

float3 material_cell_albedo(VoxelCell cell) {
    return material_albedo(voxel_material(cell));
}

float3 material_emissive(VoxelCell cell) {
    return voxel_emissive(cell);
}

bool material_has_emissive(float3 emissive) {
    return emissive.x + emissive.y + emissive.z > 0.001;
}
