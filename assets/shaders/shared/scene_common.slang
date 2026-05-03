#pragma once
// Scene-wide uniforms shared by all passes.
// Must match Rust GpuSceneUniforms in src/render/scene_ubo.rs exactly (224 bytes).

static const uint LIGHTING_FLAG_SHADOWS_ENABLED = 1u << 0;
static const uint LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS = 1u << 1;
static const uint LIGHTING_DEBUG_VIEW_SHIFT = 28u;
static const uint LIGHTING_DEBUG_VIEW_MASK = 0xFu << LIGHTING_DEBUG_VIEW_SHIFT;
static const uint LIGHTING_DEBUG_VIEW_FINAL = 0u;
static const uint LIGHTING_DEBUG_VIEW_DIRECT_DIFFUSE = 1u;
static const uint LIGHTING_DEBUG_VIEW_NORMAL = 2u;
static const uint RENDER_MODE_VPT = 1u;
static const uint DENOISER_FLAG_ENABLED = 1u << 0;
static const uint VPT_DEBUG_VIEW_FINAL = 0u;
static const uint VPT_DEBUG_VIEW_RAW = 1u;
static const uint VPT_DEBUG_VIEW_TEMPORAL = 2u;
static const uint VPT_DEBUG_VIEW_VARIANCE = 3u;
static const uint VPT_DEBUG_VIEW_HISTORY_VALID = 4u;
static const uint VPT_DEBUG_VIEW_MOTION = 5u;
static const uint VPT_DEBUG_VIEW_NORMAL = 6u;
static const uint VPT_DEBUG_VIEW_DEPTH = 7u;
static const uint VPT_DEBUG_VIEW_RESERVOIR_WEIGHT = 8u;
static const uint VPT_DEBUG_VIEW_DIRECT = 9u;
static const uint VPT_DEBUG_VIEW_INDIRECT = 10u;
static const uint VPT_DEBUG_VIEW_AREA_SUBPIXEL = 11u;
static const uint VPT_DEBUG_VIEW_AREA_LENS = 12u;
static const uint VPT_DEBUG_VIEW_AREA_WEIGHT = 13u;
static const uint VPT_DEBUG_VIEW_AREA_HISTORY_VALID = 14u;
static const uint VPT_DEBUG_VIEW_AREA_REJECTION = 15u;
static const uint VPT_DEBUG_VIEW_AREA_JACOBIAN = 16u;

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
    uint     denoiser_flags;  // 4B
    uint     denoiser_atrous_iterations; // 4B
    uint     vpt_debug_view;  // 4B
    float3   camera_right;    // 12B
    float    aperture_radius; // 4B
    float3   camera_up;       // 12B
    float    focal_distance;  // 4B
    float3   camera_forward;  // 12B
    float    _pad4;           // 4B
};                            // total: 224B, matches Rust repr(C) layout exactly

struct ScenePrimaryRay {
    float3 origin;
    float3 direction;
};

float2 sample_disk_for_lens(float2 uv) {
    float radius = sqrt(saturate(uv.x));
    float phi = 6.28318530718 * uv.y;
    return float2(cos(phi), sin(phi)) * radius;
}

float3 scene_primary_ray_dir_from_sample(SceneUniforms scene, float2 pixel_sample) {
    float3x3 dir_mat = float3x3(
        scene.pixel_to_ray[0].xyz,
        scene.pixel_to_ray[1].xyz,
        scene.pixel_to_ray[2].xyz
    );
    return normalize(mul(float3(pixel_sample, 1.0), dir_mat));
}

ScenePrimaryRay scene_primary_ray_from_area_sample(SceneUniforms scene, float2 pixel_sample, float2 lens_uv) {
    ScenePrimaryRay ray;
    float3 camera_origin = scene.pixel_to_ray[3].xyz;
    float3 pinhole_dir = scene_primary_ray_dir_from_sample(scene, pixel_sample);
    ray.origin = camera_origin;
    ray.direction = pinhole_dir;

    if (scene.aperture_radius > 0.0) {
        float2 lens_disk = sample_disk_for_lens(lens_uv);
        float3 lens_offset = scene.aperture_radius
            * (lens_disk.x * normalize(scene.camera_right) + lens_disk.y * normalize(scene.camera_up));
        float focus_denom = max(dot(pinhole_dir, normalize(scene.camera_forward)), 1.0e-4);
        float focus_t = scene.focal_distance / focus_denom;
        float3 focus_point = camera_origin + pinhole_dir * focus_t;
        ray.origin = camera_origin + lens_offset;
        ray.direction = normalize(focus_point - ray.origin);
    }
    return ray;
}
