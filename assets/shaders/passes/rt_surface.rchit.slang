#include "rt_surface_common.slang"
#include "voxel_traverse.slang"
#include "material_common.slang"

[[vk::binding(4, 0)]]
ConstantBuffer<UcvhConfig> ucvh_config;

[[vk::binding(5, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(6, 0)]]
StructuredBuffer<NodeLN> hierarchy_l1;

[[vk::binding(7, 0)]]
StructuredBuffer<NodeLN> hierarchy_l2;

[[vk::binding(8, 0)]]
StructuredBuffer<NodeLN> hierarchy_l3;

[[vk::binding(9, 0)]]
StructuredBuffer<NodeLN> hierarchy_l4;

[[vk::binding(10, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(11, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

[[vk::binding(12, 0)]]
RWStructuredBuffer<uint> traversal_stats;

[shader("closesthit")]
void main(inout RtSurfacePayload payload, in RtSurfaceIntersectionAttributes attributes) {
    float hit_t = RayTCurrent();
    float3 world_origin = WorldRayOrigin();
    float3 world_direction = WorldRayDirection();
    uint primitive_index = PrimitiveIndex();

    HitResult hit = trace_primary_ray(
        make_ray(world_origin, world_direction),
        ucvh_config,
        hierarchy_l0,
        hierarchy_l1,
        hierarchy_l2,
        hierarchy_l3,
        hierarchy_l4,
        brick_occupancy,
        brick_materials,
        traversal_stats,
        false
    );

    payload.primitive_index = primitive_index;
    payload.instance_id = InstanceID();
    if (!hit.hit) {
        payload.hit_kind = RT_SURFACE_HIT_KIND_MISS;
        payload.hit_t = hit_t;
        payload.position = float3(0.0);
        payload.normal = normalize(attributes.object_normal);
        payload.albedo = float3(0.0);
        payload.roughness = 1.0;
        payload.material_id = 0u;
        payload.brick_id = 0xffffffffu;
        payload.local = uint3(0u);
        return;
    }

    payload.hit_kind = RT_SURFACE_HIT_KIND_VOXEL;
    payload.hit_t = hit.t;
    payload.position = hit.position;
    payload.normal = normalize(hit.normal);
    payload.albedo = material_cell_albedo(hit.cell);
    payload.roughness = material_cell_roughness(hit.cell);
    payload.material_id = voxel_material(hit.cell);
    payload.brick_id = hit.brick_id;
    payload.local = hit.local;
}
