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

float3 compute_occupancy_gradient_normal(
    float3 position,
    float3 fallback_normal,
    uint grid_dim_l0,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> brick_occupancy
) {
    int3 ip = int3(floor(position));
    float3 grad;
    grad.x = float(!rc_outside_geo(float3(ip) + float3(1,0,0), hierarchy_l0, brick_occupancy, grid_dim_l0))
           - float(!rc_outside_geo(float3(ip) + float3(-1,0,0), hierarchy_l0, brick_occupancy, grid_dim_l0));
    grad.y = float(!rc_outside_geo(float3(ip) + float3(0,1,0), hierarchy_l0, brick_occupancy, grid_dim_l0))
           - float(!rc_outside_geo(float3(ip) + float3(0,-1,0), hierarchy_l0, brick_occupancy, grid_dim_l0));
    grad.z = float(!rc_outside_geo(float3(ip) + float3(0,0,1), hierarchy_l0, brick_occupancy, grid_dim_l0))
           - float(!rc_outside_geo(float3(ip) + float3(0,0,-1), hierarchy_l0, brick_occupancy, grid_dim_l0));
    return (length(grad) > 0.001) ? normalize(-grad) : fallback_normal;
}

float3 select_rc_normal(
    SceneUniforms scene,
    float3 position,
    float3 axis_normal,
    uint grid_dim_l0,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> brick_occupancy
) {
    if (scene.rc_normal_strategy == RC_NORMAL_STRATEGY_OCCUPANCY_GRADIENT) {
        return compute_occupancy_gradient_normal(position, axis_normal, grid_dim_l0, hierarchy_l0, brick_occupancy);
    }
    return axis_normal;
}

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

    // Emissive early-out: Reinhard tonemap preserves warm color better than ACES at high values
    uint3 emissive_raw = uint3(gb1.g, gb1.b, gb1.a);
    if (emissive_raw.x + emissive_raw.y + emissive_raw.z > 0u) {
        float3 emissive = float3(emissive_raw) / 255.0 * 3.0;
        float3 mapped = emissive / (1.0 + emissive);
        output_image[tid.xy] = float4(pow(mapped, float3(1.0 / 2.2)), 1.0);
        return;
    }

    float ndotl = max(dot(normal, scene.sun_direction), 0.0);
    float shadow = 1.0;
    bool shadows_enabled = (scene.lighting_flags & LIGHTING_FLAG_SHADOWS_ENABLED) != 0u;
    bool skip_backface_shadows = (scene.lighting_flags & LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS) != 0u;
    if (shadows_enabled && (ndotl > 0.0 || !skip_backface_shadows)) {
        float3 shadow_origin = position + normal * 0.51;
        Ray shadow_ray = make_ray(shadow_origin, scene.sun_direction);
        HitResult shadow_hit = trace_primary_ray(
            shadow_ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials
        );
        shadow = shadow_hit.hit ? 0.0 : 1.0;
    }
    float3 diffuse = base_color * scene.sun_intensity * ndotl * shadow;

    // Indirect: RC probes or hemisphere fallback
    float3 ambient;
    if (scene.rc_enabled > 0u) {
        uint grid_dim_l0 = ucvh_config[0].brick_grid_size.x;
        float3 rc_normal = select_rc_normal(scene, position, normal, grid_dim_l0, hierarchy_l0, brick_occupancy);
        uint3 c0_grid = uint3(scene.rc_c0_grid_x, scene.rc_c0_grid_y, scene.rc_c0_grid_z);
        float3 indirect = integrate_probe(position, rc_normal, rc_probes, scene.rc_c0_offset, c0_grid);
        ambient = base_color * indirect;
    } else {
        ambient = base_color * hemisphere_ambient(normal, scene.sky_color, scene.ground_color);
    }
    float3 hdr_color = diffuse + ambient;

    // DEBUG: uncomment one of these to visualize RC components
    // hdr_color = ambient;                    // RC indirect only
    // hdr_color = float3(normal * 0.5 + 0.5); // normal visualization
    // hdr_color = diffuse;                    // direct sun only

    // Tonemap (ACES Filmic) + gamma correction
    float3 mapped = aces_tonemap(hdr_color);
    float3 ldr = pow(mapped, float3(1.0 / 2.2));

    output_image[tid.xy] = float4(ldr, 1.0);
}
