// Scene-wide uniforms shared by all passes.
// Must match Rust GpuSceneUniforms in src/render/scene_ubo.rs exactly (144 bytes).

struct SceneUniforms {
    float4x4 pixel_to_ray;   // 64B — col 0-2: direction matrix, col 3: camera origin
    uint2    resolution;      // 8B
    uint2    _pad0;           // 8B
    float3   sun_direction;   // 12B — normalized, world space, points TOWARD sun
    float    _pad1;           // 4B
    float3   sun_intensity;   // 12B — HDR color * intensity
    float    _pad2;           // 4B
    float3   sky_color;       // 12B — hemisphere ambient upper
    float    _pad3;           // 4B
    float3   ground_color;    // 12B — hemisphere ambient lower
    float    time;            // 4B
};                            // total: 144B, 16-byte aligned
