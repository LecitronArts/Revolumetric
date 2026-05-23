#pragma once

static const uint VPT_MOTION_FLAG_HISTORY_VALID = 1u << 0;
static const uint VPT_MOTION_FLAG_CAMERA_STATIC = 1u << 1;
static const uint VPT_MOTION_FLAG_UCVH_REGION_MOVE = 1u << 2;
static const uint VPT_MOTION_FLAG_DISOCCLUDED = 1u << 4;
static const uint VPT_MOTION_FLAG_HISTORY_RESET = 1u << 5;
static const uint VPT_MOTION_FLAG_BEHIND_CAMERA = 1u << 6;

static const uint VPT_NO_BRICK_GENERATION = 0xffffffffu;
static const uint VPT_UCVH_MOTION_EVENT_CAPACITY = 64u;

struct UcvhMotionEvent {
    uint4 region_min;
    uint4 region_max_exclusive;
    int4 world_delta_current_from_previous;
    uint generation;
    uint3 _pad;
};

struct VptMotionGuidePushConstants {
    uint motion_event_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

bool vpt_motion_flags_reject_history(uint flags) {
    return (flags & VPT_MOTION_FLAG_HISTORY_VALID) == 0u
        || (flags & (VPT_MOTION_FLAG_HISTORY_RESET | VPT_MOTION_FLAG_DISOCCLUDED | VPT_MOTION_FLAG_BEHIND_CAMERA)) != 0u;
}

bool vpt_surface_generation_rejects_history(uint previous_generation, uint current_generation) {
    return previous_generation != current_generation;
}

bool vpt_ucvh_motion_event_contains(UcvhMotionEvent event, uint3 voxel_pos) {
    return all(voxel_pos >= event.region_min.xyz) && all(voxel_pos < event.region_max_exclusive.xyz);
}
