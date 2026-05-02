#pragma once
// VCT MVP helpers.
//
// This is a compile-safe diffuse cone approximation over the existing UCVH
// ray traversal. It intentionally does not pretend to be a radiance mip cone
// hierarchy; the current UCVH shader path only exposes L0 brick traversal.

#include "material_common.slang"
#include "lighting_common.slang"

float3 vct_probe_direction(float3 normal, float3 tangent_hint, float scale) {
    return normalize(normal + tangent_hint * scale);
}

float3 vct_trace_lobe(
    float3 origin,
    float3 direction,
    SceneUniforms scene,
    StructuredBuffer<UcvhConfig> config_buf,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    StructuredBuffer<VoxelCell> material_buf
) {
    HitResult hit = trace_primary_ray(
        make_ray(origin, direction),
        config_buf,
        hierarchy_l0,
        occupancy_buf,
        material_buf
    );

    if (!hit.hit) {
        return sky_color_for_dir(direction, scene);
    }

    float3 hit_albedo = material_cell_albedo(hit.cell);
    float3 emissive = material_emissive(hit.cell) * 3.0;
    float sun_term = max(dot(hit.normal, scene.sun_direction), 0.0);
    float distance_falloff = 1.0 / (1.0 + hit.t * 0.05);
    return emissive + hit_albedo * (scene.sky_color * 0.35 + scene.sun_intensity * sun_term * 0.15) * distance_falloff;
}

float3 vct_trace_diffuse(
    float3 position,
    float3 normal,
    SceneUniforms scene,
    StructuredBuffer<UcvhConfig> config_buf,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    StructuredBuffer<VoxelCell> material_buf
) {
    float3 origin = position + normal * 0.75;
    float3 tangent = normalize(abs(normal.y) < 0.95 ? cross(normal, float3(0, 1, 0)) : cross(normal, float3(1, 0, 0)));
    float3 bitangent = normalize(cross(normal, tangent));

    float3 indirect = float3(0.0);
    indirect += vct_trace_lobe(origin, normal, scene, config_buf, hierarchy_l0, occupancy_buf, material_buf) * 0.40;
    indirect += vct_trace_lobe(origin, vct_probe_direction(normal, tangent, 0.70), scene, config_buf, hierarchy_l0, occupancy_buf, material_buf) * 0.20;
    indirect += vct_trace_lobe(origin, vct_probe_direction(normal, -tangent, 0.70), scene, config_buf, hierarchy_l0, occupancy_buf, material_buf) * 0.20;
    indirect += vct_trace_lobe(origin, vct_probe_direction(normal, bitangent, 0.70), scene, config_buf, hierarchy_l0, occupancy_buf, material_buf) * 0.10;
    indirect += vct_trace_lobe(origin, vct_probe_direction(normal, -bitangent, 0.70), scene, config_buf, hierarchy_l0, occupancy_buf, material_buf) * 0.10;
    return max(indirect, float3(0.0));
}
