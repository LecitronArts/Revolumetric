#include "area_restir_common.slang"
#include "scene_common.slang"
#include "voxel_traverse.slang"
#include "material_common.slang"
#include "vpt_history_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<AreaRestirUniforms> area_restir;

[[vk::binding(1, 0)]]
RWStructuredBuffer<AreaRestirReservoir> output_reservoirs;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_position_depth;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_normal_roughness;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_albedo_material;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> area_restir_debug;

[[vk::binding(6, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(7, 0)]]
ConstantBuffer<UcvhConfig> ucvh_config;

[[vk::binding(8, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(9, 0)]]
StructuredBuffer<NodeLN> hierarchy_l1;

[[vk::binding(10, 0)]]
StructuredBuffer<NodeLN> hierarchy_l2;

[[vk::binding(11, 0)]]
StructuredBuffer<NodeLN> hierarchy_l3;

[[vk::binding(12, 0)]]
StructuredBuffer<NodeLN> hierarchy_l4;

[[vk::binding(13, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(14, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

struct AreaRestirCandidateSurface {
    float4 position_depth;
    float4 normal_roughness;
    float4 albedo_material;
};

uint area_restir_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float area_restir_rand01(inout uint state) {
    state = area_restir_hash_u32(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

float surface_target_luma(float4 position_depth, float4 albedo_material) {
    if (position_depth.w < 0.0) {
        return 0.0;
    }
    return max(dot(albedo_material.rgb, float3(0.2126, 0.7152, 0.0722)), 1.0e-3);
}

AreaRestirCandidateSurface invalid_candidate_surface() {
    AreaRestirCandidateSurface surface;
    surface.position_depth = float4(0.0, 0.0, 0.0, -1.0);
    surface.normal_roughness = float4(0.0);
    surface.albedo_material = float4(0.0);
    return surface;
}

AreaRestirCandidateSurface read_center_surface(uint2 pixel) {
    AreaRestirCandidateSurface surface;
    surface.position_depth = surface_position_depth[pixel];
    surface.normal_roughness = surface_normal_roughness[pixel];
    surface.albedo_material = surface_albedo_material[pixel];
    return surface;
}

AreaRestirEvalContext candidate_surface_context(AreaRestirCandidateSurface surface) {
    AreaRestirEvalContext context;
    context.position_depth = surface.position_depth;
    context.normal_roughness = surface.normal_roughness;
    context.albedo_material = surface.albedo_material;
    context.motion_history = float4(0.0);
    return context;
}

AreaRestirCandidateSurface evaluate_area_restir_candidate_surface(
    SceneUniforms scene,
    uint2 pixel,
    AreaRestirSampleState sample_state
) {
    float2 pixel_sample = area_restir_pixel_sample(pixel, sample_state);
    ScenePrimaryRay primary_ray = scene_primary_ray_from_area_sample(
        scene,
        pixel_sample,
        sample_state.lens_uv
    );
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
        return invalid_candidate_surface();
    }

    AreaRestirCandidateSurface surface;
    surface.position_depth = float4(hit.position, hit.t);
    float emissive_luma = dot(material_emissive(hit.cell), float3(0.2126, 0.7152, 0.0722));
    surface.normal_roughness = float4(normalize(hit.normal), emissive_luma);
    surface.albedo_material = float4(material_cell_albedo(hit.cell), float(voxel_material(hit.cell)));
    return surface;
}

float area_restir_candidate_target_pdf(
    AreaRestirCandidateSurface center_surface,
    AreaRestirCandidateSurface candidate_surface
) {
    if (!area_restir_surface_compatible(
        candidate_surface_context(center_surface),
        candidate_surface_context(candidate_surface),
        area_restir.normal_threshold,
        area_restir.depth_threshold
    )) {
        return 0.0;
    }
    return surface_target_luma(candidate_surface.position_depth, candidate_surface.albedo_material);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = tid.xy;
    uint index = pixel.y * area_restir.width + pixel.x;
    if (pixel.x >= area_restir.width || pixel.y >= area_restir.height || index >= area_restir.reservoir_count) {
        return;
    }

    float4 center_position_depth = surface_position_depth[pixel];
    AreaRestirCandidateSurface center_surface = read_center_surface(pixel);
    if (area_restir.enabled == 0u || center_position_depth.w < 0.0) {
        output_reservoirs[index] = area_restir_invalid_reservoir();
        if (area_restir.debug_view != 0u) {
            area_restir_debug[pixel] = float4(0.0);
        }
        return;
    }

    uint rng = area_restir_hash_u32(index ^ (area_restir.frame_index * 747796405u));
    AreaRestirReservoir reservoir = area_restir_invalid_reservoir();
    float weight_sum = 0.0;
    uint candidate_count = max(area_restir.initial_candidate_count, 1u);

    for (uint candidate = 0u; candidate < candidate_count; candidate++) {
        float2 subpixel_uv = area_restir.subpixel_enabled != 0u
            ? float2(area_restir_rand01(rng), area_restir_rand01(rng))
            : float2(0.5, 0.5);
        float2 lens_uv = area_restir.lens_enabled != 0u
            ? float2(area_restir_rand01(rng), area_restir_rand01(rng))
            : float2(0.5, 0.5);

        AreaRestirSampleState sample_state = area_restir_sample_pixel_state(pixel, subpixel_uv, lens_uv, candidate);
        AreaRestirCandidateSurface candidate_surface = evaluate_area_restir_candidate_surface(scene_ubo, pixel, sample_state);
        float target_pdf = area_restir_candidate_target_pdf(center_surface, candidate_surface);
        if (target_pdf <= 0.0) {
            continue;
        }
        float next_weight_sum = weight_sum + target_pdf;
        if (!area_restir_is_valid_reservoir(reservoir) || area_restir_rand01(rng) * next_weight_sum <= target_pdf) {
            reservoir.sample_state = sample_state;
            reservoir.target_pdf = target_pdf;
            reservoir.contribution_luma = target_pdf;
        }
        weight_sum = next_weight_sum;
    }

    reservoir.sample_count_m = candidate_count;
    reservoir.weight_sum = weight_sum;
    reservoir.confidence = float(candidate_count);
    area_restir_finalize_reservoir(reservoir, area_restir.history_length);
    output_reservoirs[index] = reservoir;
    if (area_restir.debug_view != 0u) {
        area_restir_debug[pixel] = float4(reservoir.sample_state.subpixel_uv, reservoir.selected_weight > 0.0 ? 1.0 : 0.0, 1.0);
    }
}
