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
    payload.occluded = 0u;
    payload.max_t = max_t;
    payload.skip_brick_id = skip_brick_id;
    payload.skip_local = skip_local;
    return payload;
}
