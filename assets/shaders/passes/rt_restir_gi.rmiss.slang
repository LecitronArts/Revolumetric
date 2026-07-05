#include "rt_surface_common.slang"

[shader("miss")]
void main(inout RtSurfacePayload payload) {
    payload.hit_kind = RT_SURFACE_HIT_KIND_MISS;
    payload.hit_t = RT_SURFACE_MISS_T;
    payload.primitive_index = 0xffffffffu;
    payload.instance_id = 0xffffffffu;
    payload.position = float3(0.0);
    payload.normal = -payload.ray_direction;
}
