#pragma once

static const uint VPT_HISTORY_FLAG_CAMERA_CUT = 1u << 0;
static const uint VPT_HISTORY_FLAG_RESIZE = 1u << 1;
static const uint VPT_HISTORY_FLAG_SCENE_INVALIDATED = 1u << 2;
static const uint VPT_HISTORY_FLAG_LIGHTS_INVALIDATED = 1u << 3;

struct VptHistoryUniforms {
    float4x4 current_view_proj;
    float4x4 previous_view_proj;
    uint2 current_resolution;
    uint2 previous_resolution;
    float2 current_jitter;
    float2 previous_jitter;
    uint frame_index;
    uint history_reset_generation;
    uint flags;
    uint _pad0;
};

struct VptSurfacePixel {
    float4 position_depth;
    float4 normal_roughness;
    float4 albedo_material;
    float4 motion_history;
    float linear_depth;
    uint material_id;
    float history_confidence;
    uint _pad0;
};

float2 vpt_previous_pixel_center_from_motion(uint2 pixel, float4 motion) {
    return float2(pixel) + 0.5 + motion.xy;
}

float2 vpt_history_sample_from_motion(uint2 pixel, float4 motion) {
    return vpt_previous_pixel_center_from_motion(pixel, motion) - 0.5;
}
