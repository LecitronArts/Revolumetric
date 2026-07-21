#pragma once

#include "rt_surface_common.slang"

struct RtShadowPayload {
    uint occluded;
    float max_t;
    uint skip_brick_id;
    uint3 skip_local;
};

RtShadowPayload make_rt_shadow_payload(uint skip_brick_id, uint3 skip_local, float max_t) {
    RtShadowPayload payload;
    // Default to occluded. Shadow/visibility rays are traced with
    // RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
    // so the closest-hit shader (which would set occluded = 1) never runs. Any
    // committed hit therefore leaves this default in place, while the miss shader
    // clears it to 0. This lets the driver skip closest-hit invocation entirely.
    payload.occluded = 1u;
    payload.max_t = max_t;
    payload.skip_brick_id = skip_brick_id;
    payload.skip_local = skip_local;
    return payload;
}
