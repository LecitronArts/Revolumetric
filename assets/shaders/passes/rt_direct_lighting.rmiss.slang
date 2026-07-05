#include "rt_shadow_common.slang"

[shader("miss")]
void main(inout RtShadowPayload payload) {
    payload.occluded = 0u;
}
