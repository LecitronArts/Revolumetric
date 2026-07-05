#include "rt_shadow_common.slang"
#include "voxel_traverse.slang"

[[vk::binding(6, 0)]]
ConstantBuffer<UcvhConfig> ucvh_config;

[[vk::binding(7, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(8, 0)]]
StructuredBuffer<NodeLN> hierarchy_l1;

[[vk::binding(9, 0)]]
StructuredBuffer<NodeLN> hierarchy_l2;

[[vk::binding(10, 0)]]
StructuredBuffer<NodeLN> hierarchy_l3;

[[vk::binding(11, 0)]]
StructuredBuffer<NodeLN> hierarchy_l4;

[[vk::binding(12, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(13, 0)]]
RWStructuredBuffer<uint> traversal_stats;

[shader("closesthit")]
void main(inout RtShadowPayload payload, in RtSurfaceIntersectionAttributes attributes) {
    Ray shadow_ray = make_ray(WorldRayOrigin(), WorldRayDirection());
    bool occluded = trace_any_hit_ray_skip_voxel(
        shadow_ray,
        payload.max_t,
        payload.skip_brick_id,
        payload.skip_local,
        ucvh_config,
        hierarchy_l0,
        hierarchy_l1,
        hierarchy_l2,
        hierarchy_l3,
        hierarchy_l4,
        brick_occupancy,
        traversal_stats,
        false
    );
    payload.occluded = occluded ? 1u : 0u;
}
