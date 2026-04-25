#pragma once
// Scene-wide uniforms shared by all passes.
// Must match Rust GpuSceneUniforms in src/render/scene_ubo.rs exactly (176 bytes).

static const uint LIGHTING_FLAG_SHADOWS_ENABLED = 1u << 0;
static const uint LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS = 1u << 1;

static const uint RC_NORMAL_STRATEGY_AXIS_NORMAL = 0u;
static const uint RC_NORMAL_STRATEGY_OCCUPANCY_GRADIENT = 1u;

static const uint RC_PROBE_QUALITY_FULL = 0u;

struct SceneUniforms {
    float4x4 pixel_to_ray;   // 64B
    uint2    resolution;      // 8B
    uint2    _pad0;           // 8B
    float3   sun_direction;   // 12B
    float    _pad1;           // 4B
    float3   sun_intensity;   // 12B
    float    _pad2;           // 4B
    float3   sky_color;       // 12B
    float    _pad3;           // 4B
    float3   ground_color;    // 12B
    float    time;            // 4B
    // --- RC fields (individual uints to avoid uint3 16-byte alignment in std140) ---
    uint     rc_c0_grid_x;    // 4B
    uint     rc_c0_grid_y;    // 4B
    uint     rc_c0_grid_z;    // 4B
    uint     rc_c0_offset;    // 4B
    uint     rc_enabled;      // 4B
    uint     lighting_flags;  // 4B
    uint     rc_normal_strategy; // 4B
    uint     rc_probe_quality; // 4B
};                            // total: 176B, matches Rust repr(C) layout exactly
