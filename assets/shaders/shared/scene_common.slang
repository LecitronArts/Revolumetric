#pragma once
// Scene-wide uniforms shared by all passes.
// Must match Rust GpuSceneUniforms in src/render/scene_ubo.rs exactly (176 bytes).

static const uint LIGHTING_FLAG_SHADOWS_ENABLED = 1u << 0;
static const uint LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS = 1u << 1;
static const uint LIGHTING_FLAG_VCT_ENABLED = 1u << 2;
static const uint LIGHTING_DEBUG_VIEW_SHIFT = 28u;
static const uint LIGHTING_DEBUG_VIEW_MASK = 0xFu << LIGHTING_DEBUG_VIEW_SHIFT;
static const uint LIGHTING_DEBUG_VIEW_FINAL = 0u;
static const uint LIGHTING_DEBUG_VIEW_DIRECT_DIFFUSE = 1u;
static const uint LIGHTING_DEBUG_VIEW_NORMAL = 2u;
static const uint RENDER_MODE_VCT = 0u;
static const uint RENDER_MODE_VPT = 1u;

uint lighting_debug_view(uint flags) {
    return (flags & LIGHTING_DEBUG_VIEW_MASK) >> LIGHTING_DEBUG_VIEW_SHIFT;
}

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
    uint     lighting_flags;  // 4B
    float    exposure;        // 4B
    uint     render_mode;     // 4B
    uint     vpt_sample_index;// 4B
    uint     vpt_max_bounces; // 4B
    uint3    _pad4;           // 12B
};                            // total: 176B, matches Rust repr(C) layout exactly
