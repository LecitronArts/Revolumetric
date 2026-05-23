#include "scene_common.slang"
#include "voxel_traverse.slang"
#include "material_common.slang"
#include "vpt_history_common.slang"
#include "vpt_motion_common.slang"
#include "area_restir_common.slang"
#include "vpt_primary_sample_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_position_depth;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_normal_roughness;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_albedo_material;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> motion_history;

[[vk::binding(5, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

[[vk::binding(6, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(7, 0)]]
StructuredBuffer<NodeLN> hierarchy_l1;

[[vk::binding(8, 0)]]
StructuredBuffer<NodeLN> hierarchy_l2;

[[vk::binding(9, 0)]]
StructuredBuffer<NodeLN> hierarchy_l3;

[[vk::binding(10, 0)]]
StructuredBuffer<NodeLN> hierarchy_l4;

[[vk::binding(11, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(12, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

[[vk::binding(13, 0)]]
ConstantBuffer<VptHistoryUniforms> vpt_history;

[[vk::binding(14, 0)]]
ConstantBuffer<AreaRestirUniforms> area_restir;

[[vk::binding(15, 0)]]
StructuredBuffer<AreaRestirReservoir> area_restir_reservoirs;

[[vk::binding(16, 0)]]
[[vk::image_format("r32ui")]]
RWTexture2D<uint> motion_flags;

[[vk::binding(17, 0)]]
[[vk::image_format("r32ui")]]
RWTexture2D<uint> surface_brick_generation;

[[vk::binding(18, 0)]]
StructuredBuffer<uint> brick_generations;

[[vk::binding(19, 0)]]
StructuredBuffer<UcvhMotionEvent> ucvh_motion_events;

[[vk::push_constant]]
VptMotionGuidePushConstants vpt_motion_guide;

bool vpt_history_reset_active() {
    return (vpt_history.flags & VPT_HISTORY_FLAG_CAMERA_CUT) != 0u
        || (vpt_history.flags & VPT_HISTORY_FLAG_RESIZE) != 0u
        || (vpt_history.flags & VPT_HISTORY_FLAG_SCENE_INVALIDATED) != 0u
        || (vpt_history.flags & VPT_HISTORY_FLAG_LIGHTS_INVALIDATED) != 0u;
}

void write_surface_motion_outputs(uint2 pixel, float4 motion, uint flags, uint brick_generation) {
    motion_history[pixel] = motion;
    motion_flags[pixel] = flags;
    surface_brick_generation[pixel] = brick_generation;
}

void write_miss(uint2 pixel) {
    float4 position_depth = float4(0.0, 0.0, 0.0, -1.0);
    surface_position_depth[pixel] = position_depth;
    surface_normal_roughness[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    surface_albedo_material[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    uint flags = vpt_history_reset_active()
        ? (VPT_MOTION_FLAG_DISOCCLUDED | VPT_MOTION_FLAG_HISTORY_RESET)
        : VPT_MOTION_FLAG_DISOCCLUDED;
    write_surface_motion_outputs(pixel, float4(0.0, 0.0, 0.0, 0.0), flags, VPT_NO_BRICK_GENERATION);
}

float4 project_previous_pixel(uint2 pixel, float3 world_position, inout uint flags) {
    if (vpt_history_reset_active()) {
        flags = VPT_MOTION_FLAG_DISOCCLUDED | VPT_MOTION_FLAG_HISTORY_RESET;
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float4 current_clip = mul(float4(world_position, 1.0), vpt_history.current_view_proj);
    float4 previous_clip = mul(float4(world_position, 1.0), vpt_history.previous_view_proj);
    if (current_clip.w <= 1.0e-5 || previous_clip.w <= 1.0e-5) {
        flags &= ~VPT_MOTION_FLAG_UCVH_REGION_MOVE;
        flags |= VPT_MOTION_FLAG_BEHIND_CAMERA | VPT_MOTION_FLAG_DISOCCLUDED;
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float2 previous_ndc = previous_clip.xy / previous_clip.w;
    float2 current_pixel = float2(pixel) + 0.5;
    float2 previous_pixel = float2(
        (previous_ndc.x * 0.5 + 0.5) * float(vpt_history.previous_resolution.x),
        (0.5 - previous_ndc.y * 0.5) * float(vpt_history.previous_resolution.y)
    );
    float2 motion = previous_pixel - current_pixel;

    bool valid = previous_pixel.x >= 0.0
        && previous_pixel.y >= 0.0
        && previous_pixel.x < float(vpt_history.previous_resolution.x)
        && previous_pixel.y < float(vpt_history.previous_resolution.y);
    if (!valid) {
        flags &= ~VPT_MOTION_FLAG_UCVH_REGION_MOVE;
        flags |= VPT_MOTION_FLAG_DISOCCLUDED;
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    flags |= VPT_MOTION_FLAG_HISTORY_VALID;
    if ((flags & VPT_MOTION_FLAG_UCVH_REGION_MOVE) == 0u) {
        flags |= VPT_MOTION_FLAG_CAMERA_STATIC;
    }
    return float4(motion, 0.0, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    uint2 pixel = tid.xy;
    uint index = pixel.y * scene.resolution.x + pixel.x;
    AreaRestirReservoir reservoir = index < area_restir.reservoir_count
        ? area_restir_reservoirs[index]
        : area_restir_invalid_reservoir();
    ScenePrimaryRay primary_ray = vpt_resolve_surface_primary_ray(pixel, scene, area_restir, reservoir, index);
    HitResult hit = trace_primary_ray(
        make_ray(primary_ray.origin, primary_ray.direction),
        ucvh_config,
        hierarchy_l0,
        hierarchy_l1,
        hierarchy_l2,
        hierarchy_l3,
        hierarchy_l4,
        brick_occupancy,
        brick_materials
    );

    if (!hit.hit) {
        write_miss(pixel);
        return;
    }

    float material_id = float(voxel_material(hit.cell));
    float3 albedo = material_cell_albedo(hit.cell);
    float emissive_luma = dot(material_emissive(hit.cell), float3(0.2126, 0.7152, 0.0722));

    surface_position_depth[pixel] = float4(hit.position, hit.t);
    surface_normal_roughness[pixel] = float4(normalize(hit.normal), emissive_luma);
    surface_albedo_material[pixel] = float4(albedo, material_id);
    uint flags = 0u;
    float3 previous_world_position = hit.position;
    uint3 hit_voxel = hit.brick_coord * 8u + hit.local;
    uint motion_event_count = min(vpt_motion_guide.motion_event_count, VPT_UCVH_MOTION_EVENT_CAPACITY);
    for (uint event_index = 0u; event_index < motion_event_count; event_index++) {
        UcvhMotionEvent event = ucvh_motion_events[event_index];
        if (vpt_ucvh_motion_event_contains(event, hit_voxel)) {
            previous_world_position = hit.position - float3(event.world_delta_current_from_previous.xyz);
            flags |= VPT_MOTION_FLAG_UCVH_REGION_MOVE;
            break;
        }
    }
    float4 motion = project_previous_pixel(pixel, previous_world_position, flags);
    motion.z = float(hit.motion_id);
    uint brick_generation = hit.brick_id == 0xffffffffu
        ? VPT_NO_BRICK_GENERATION
        : brick_generations[hit.brick_id];
    if ((flags & VPT_MOTION_FLAG_HISTORY_VALID) == 0u) {
        brick_generation = VPT_NO_BRICK_GENERATION;
    }
    write_surface_motion_outputs(pixel, motion, flags, brick_generation);
}
