// Primary ray compute shader — screen-space voxel ray tracing.
// Traces one ray per pixel through the UCVH brick grid.
// Output: G-buffer (position, albedo, normal+flags) for deferred lighting.

#include "scene_common.slang"
#include "voxel_traverse.slang"

// Binding 0: Scene UBO (ConstantBuffer — maps to VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

// Binding 1-3: G-buffer outputs
[[vk::binding(1, 0)]]
RWTexture2D<float4> gbuffer_pos;    // RGBA32F: xyz = voxel center, w = hit_t (-1 = miss)

[[vk::binding(2, 0)]]
[[vk::image_format("rgba8")]]
RWTexture2D<float4> gbuffer0;       // RGBA8_UNORM: rgb = base_color, a = ao

[[vk::binding(3, 0)]]
[[vk::image_format("rgba8ui")]]
RWTexture2D<uint4> gbuffer1;        // RGBA8_UINT: r = normal_id, g = roughness, b = metallic, a = flags

static const float3 MATERIAL_ALBEDO[8] = {
    float3(1.0, 1.0, 1.0),       // 0: default white
    float3(0.95, 0.925, 0.9),    // 1: stone
    float3(0.99, 0.4, 0.4),      // 2: red cloth
    float3(0.4, 0.99, 0.4),      // 3: green cloth
    float3(0.5, 0.6, 0.9),       // 4: blue
    float3(0.75, 0.65, 0.5),     // 5: brick
    float3(0.55, 0.52, 0.48),    // 6: dark stone
    float3(0.6, 0.4, 0.25),      // 7: wood
};

// Binding 4-7: UCVH buffers (order matches trace_primary_ray parameter order)
[[vk::binding(4, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

[[vk::binding(5, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(6, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(7, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Read scene uniforms (ConstantBuffer — no array index)
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    // Generate ray from pixel_to_ray matrix (same math as before).
    float2 pixel = float2(tid.xy);
    float3 origin = scene.pixel_to_ray[3].xyz;
    float3x3 dir_mat = float3x3(
        scene.pixel_to_ray[0].xyz,
        scene.pixel_to_ray[1].xyz,
        scene.pixel_to_ray[2].xyz
    );
    float3 dir = normalize(mul(float3(pixel, 1.0), dir_mat));

    Ray ray = make_ray(origin, dir);
    HitResult hit = trace_primary_ray(ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);

    if (hit.hit) {
        gbuffer_pos[tid.xy] = float4(hit.position, hit.t);
        uint mat_id = voxel_material(hit.cell);
        float3 albedo = MATERIAL_ALBEDO[min(mat_id, 7u)];
        gbuffer0[tid.xy] = float4(albedo, 1.0);
        float3 emissive = voxel_emissive(hit.cell);
        uint3 emissive_raw = uint3(emissive * 255.0);
        gbuffer1[tid.xy] = uint4(encode_normal_id(hit.normal), emissive_raw.r, emissive_raw.g, emissive_raw.b);
    } else {
        gbuffer_pos[tid.xy] = float4(0, 0, 0, -1.0);  // w < 0 = miss
        gbuffer0[tid.xy] = float4(0, 0, 0, 0);
        gbuffer1[tid.xy] = uint4(0, 0, 0, 0);
    }
}
