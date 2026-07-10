#include "rt_surface_common.slang"
#include "voxel_traverse.slang"

[[vk::binding(3, 0)]]
StructuredBuffer<RtAabb> rt_aabbs;

[[vk::binding(4, 0)]]
ConstantBuffer<UcvhConfig> ucvh_config;

[[vk::binding(5, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(10, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(12, 0)]]
RWStructuredBuffer<uint> traversal_stats;

[shader("intersection")]
void main() {
    uint primitive_index = PrimitiveIndex();
    RtAabb rt_aabb = rt_aabbs[primitive_index];
    float3 ray_origin = ObjectRayOrigin();
    float3 ray_direction = ObjectRayDirection();
    float3 inv_dir = rt_surface_safe_inv_dir(ray_direction);
    float2 hit_interval = intersect_aabb(ray_origin, inv_dir, rt_aabb_to_aabb(rt_aabb));
    float hit_t = max(hit_interval.x, RayTMin());

    if (hit_interval.x <= hit_interval.y && hit_interval.y >= RayTMin() && hit_t <= RayTCurrent()) {
        UcvhConfig config = ucvh_config;
        uint3 brick_coord = uint3(rt_aabb_min(rt_aabb) / 8.0 + float3(0.5));
        if (any(brick_coord >= config.brick_grid_size.xyz)) {
            return;
        }

        uint l0_idx = brick_coord.x
                    + brick_coord.y * config.brick_grid_size.x
                    + brick_coord.z * config.brick_grid_size.x * config.brick_grid_size.y;
        NodeL0 node = hierarchy_l0[l0_idx];
        if (node_l0_is_empty(node) || brick_occupancy[node.brick_id].count == 0u) {
            return;
        }

        float3 local_origin = ray_origin - rt_aabb_min(rt_aabb);
        uint3 hit_local;
        float3 hit_normal;
        float voxel_hit_t;
        if (!brick_dda(
            brick_occupancy,
            node.brick_id,
            local_origin,
            ray_direction,
            hit_local,
            hit_normal,
            voxel_hit_t,
            traversal_stats,
            false
        )) {
            return;
        }

        hit_t = voxel_hit_t;
        if (hit_t < RayTMin() || hit_t > RayTCurrent()) {
            return;
        }

        RtSurfaceIntersectionAttributes attributes;
        attributes.brick_id = node.brick_id;
        attributes.packed_local_normal = rt_surface_pack_local_normal(hit_local, hit_normal);
        attributes.object_position = ray_origin + ray_direction * hit_t;
        attributes.object_normal = hit_normal;
        ReportHit(hit_t, RT_SURFACE_HIT_KIND_VOXEL, attributes);
    }
}
