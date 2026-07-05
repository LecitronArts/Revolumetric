#pragma once
// RT temporal history ABI shared by RT raygen, temporal, and resolve passes.

static const uint RT_HISTORY_FLAG_CAMERA_CUT = 1u << 0;
static const uint RT_HISTORY_FLAG_RESIZE = 1u << 1;
static const uint RT_HISTORY_FLAG_SCENE_INVALIDATED = 1u << 2;
static const uint RT_HISTORY_FLAG_AS_REBUILT = 1u << 3;
static const uint RT_HISTORY_FLAG_LIGHTS_INVALIDATED = 1u << 4;

static const uint RT_SURFACE_HIT_KIND_MISS = 0u;
static const uint RT_SURFACE_HIT_KIND_VOXEL = 1u;

static const uint RT_DEBUG_VIEW_OFF = 0u;
static const uint RT_DEBUG_VIEW_SURFACE = 1u;
static const uint RT_DEBUG_VIEW_HIT_DISTANCE = 2u;
static const uint RT_DEBUG_VIEW_HISTORY_VALID = 3u;
static const uint RT_DEBUG_VIEW_DIRECT_RESERVOIR = 4u;
static const uint RT_DEBUG_VIEW_INDIRECT_RESERVOIR = 5u;
static const uint RT_DEBUG_VIEW_TEMPORAL = 6u;

struct RtHistoryUniforms {
    float4x4 current_view_proj;
    float4x4 previous_view_proj;
    uint2 current_resolution;
    uint2 previous_resolution;
    float2 current_jitter;
    float2 previous_jitter;
    uint frame_index;
    uint history_reset_generation;
    uint as_rebuild_generation;
    uint scene_generation;
    uint lights_generation;
    uint temporal_denoise_enabled;
    uint flags;
    uint debug_view;
    uint history_length;
    float normal_threshold;
    float depth_threshold;
    uint _pad0;
};

struct RtSurfacePixel {
    float4 position_depth;
    float4 normal_roughness;
    float4 albedo_material;
    float4 motion_history;
    float linear_depth;
    uint material_id;
    float history_confidence;
    uint hit_kind;
    uint brick_id;
    uint3 local;
};
