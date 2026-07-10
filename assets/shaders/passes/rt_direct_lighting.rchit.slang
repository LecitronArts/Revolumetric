#include "rt_shadow_common.slang"

[shader("closesthit")]
void main(inout RtShadowPayload payload, in RtSurfaceIntersectionAttributes attributes) {
    payload.occluded = 1u;
}
