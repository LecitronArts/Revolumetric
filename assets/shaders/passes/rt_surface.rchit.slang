#include "rt_surface_common.slang"
#include "material_common.slang"

[[vk::binding(11, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

[shader("closesthit")]
void main(inout RtSurfacePayload payload, in RtSurfaceIntersectionAttributes attributes) {
    float hit_t = RayTCurrent();
    float3 world_origin = WorldRayOrigin();
    float3 world_direction = WorldRayDirection();
    uint primitive_index = PrimitiveIndex();
    uint brick_id = attributes.brick_id;
    uint3 local = rt_surface_unpack_local(attributes.packed_local_normal);
    float3 hit_normal = rt_surface_unpack_axis_normal(attributes.packed_local_normal);
    VoxelCell cell = brick_materials[brick_id * 512u + morton_encode(local)];

    payload.hit_kind = RT_SURFACE_HIT_KIND_VOXEL;
    payload.hit_t = hit_t;
    payload.ray_direction = world_direction;
    payload.position = world_origin + world_direction * hit_t;
    payload.normal = hit_normal;
    payload.albedo = material_cell_albedo(cell);
    payload.roughness = material_cell_roughness(cell);
    payload.material_id = voxel_material(cell);
    payload.brick_id = brick_id;
    payload.local = local;
    payload.emissive_radiance = material_emissive(cell) * 3.0;
}
