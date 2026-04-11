// Deferred lighting compute shader.
// Reads G-buffer, traces shadow rays, computes Lambert + hemisphere ambient + ACES tonemap.

#include "voxel_traverse.slang"
#include "lighting_common.slang"
#include "radiance_cascade.slang"

// Binding 0: Scene UBO
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

// Binding 1-3: G-buffer inputs (read-only via RWTexture2D — no sampler needed)
[[vk::binding(1, 0)]]
RWTexture2D<float4> gbuffer_pos;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba8")]]
RWTexture2D<float4> gbuffer0;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba8ui")]]
RWTexture2D<uint4> gbuffer1;

// Binding 4: output image
[[vk::binding(4, 0)]]
RWTexture2D<float4> output_image;

// Binding 5-8: UCVH buffers (for shadow ray tracing)
[[vk::binding(5, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

[[vk::binding(6, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(7, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(8, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

// Binding 9: RC probe buffer (read-only for indirect lighting)
[[vk::binding(9, 0)]]
StructuredBuffer<float4> rc_probes;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    // Read G-buffer
    float4 pos_data = gbuffer_pos[tid.xy];

    // Miss check: w < 0 means no geometry hit
    if (pos_data.w < 0.0) {
        // Reconstruct ray direction for sky gradient
        float2 pixel = float2(tid.xy);
        float3x3 dir_mat = float3x3(
            scene.pixel_to_ray[0].xyz,
            scene.pixel_to_ray[1].xyz,
            scene.pixel_to_ray[2].xyz
        );
        float3 ray_dir = normalize(mul(float3(pixel, 1.0), dir_mat));
        float3 sky = sky_color_for_dir(ray_dir, scene);
        // Apply tonemap + gamma to sky for consistency
        sky = aces_tonemap(sky);
        sky = pow(sky, float3(1.0 / 2.2));
        output_image[tid.xy] = float4(sky, 1.0);
        return;
    }

    // Read surface data
    float3 position = pos_data.xyz;
    float3 base_color = gbuffer0[tid.xy].rgb;
    uint4 gb1 = gbuffer1[tid.xy];
    uint normal_id = gb1.r;
    float3 normal = NORMAL_TABLE[min(normal_id, 5u)];

    // Emissive early-out
    uint3 emissive_raw = uint3(gb1.g, gb1.b, gb1.a);
    if (emissive_raw.x + emissive_raw.y + emissive_raw.z > 0u) {
        float3 emissive = float3(emissive_raw) / 255.0;
        output_image[tid.xy] = float4(pow(emissive, float3(1.0 / 2.2)), 1.0);
        return;
    }

    // Shadow ray: offset from voxel center along normal to exit the solid voxel
    float3 shadow_origin = position + normal * 0.51;
    Ray shadow_ray = make_ray(shadow_origin, scene.sun_direction);
    HitResult shadow_hit = trace_primary_ray(
        shadow_ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials
    );

    // Lighting
    float shadow = shadow_hit.hit ? 0.0 : 1.0;
    float ndotl = max(dot(normal, scene.sun_direction), 0.0);
    float3 diffuse = base_color * scene.sun_intensity * ndotl * shadow;
    float3 ambient;
    if (scene.rc_enabled > 0u) {
        float3 indirect = integrate_probe(position, normal, rc_probes, scene.rc_c0_offset, scene.rc_c0_grid);
        ambient = base_color * indirect;
    } else {
        ambient = base_color * hemisphere_ambient(normal, scene.sky_color, scene.ground_color);
    }
    float3 hdr_color = diffuse + ambient;

    // Tonemap (ACES Filmic) + gamma correction
    float3 mapped = aces_tonemap(hdr_color);
    float3 ldr = pow(mapped, float3(1.0 / 2.2));

    output_image[tid.xy] = float4(ldr, 1.0);
}
