// VPT reference compute shader.
// Traces stochastic voxel paths into a progressive HDR accumulation image.

#include "voxel_traverse.slang"
#include "material_common.slang"
#include "lighting_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> accumulation_image;

[[vk::binding(2, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

[[vk::binding(3, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(4, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(5, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rand01(inout uint state) {
    state = hash_u32(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

float3 cosine_hemisphere(float3 normal, inout uint state) {
    float r1 = rand01(state);
    float r2 = rand01(state);
    float phi = 6.28318530718 * r1;
    float r = sqrt(r2);
    float3 tangent = normalize(abs(normal.y) < 0.95 ? cross(normal, float3(0, 1, 0)) : cross(normal, float3(1, 0, 0)));
    float3 bitangent = normalize(cross(normal, tangent));
    return normalize(tangent * (cos(phi) * r) + bitangent * (sin(phi) * r) + normal * sqrt(max(0.0, 1.0 - r2)));
}

float3 trace_path(Ray ray, SceneUniforms scene, inout uint rng_state) {
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    uint max_bounces = clamp(scene.vpt_max_bounces, 1u, 8u);

    for (uint bounce = 0u; bounce < max_bounces; bounce++) {
        HitResult hit = trace_primary_ray(ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);
        if (!hit.hit) {
            radiance += throughput * sky_color_for_dir(ray.direction, scene);
            break;
        }

        float3 emissive = material_emissive(hit.cell) * 3.0;
        if (emissive.x + emissive.y + emissive.z > 0.0) {
            radiance += throughput * emissive;
            break;
        }

        float sun_term = max(dot(hit.normal, scene.sun_direction), 0.0);
        radiance += throughput * material_cell_albedo(hit.cell) * scene.sun_intensity * sun_term * 0.2;
        throughput *= material_cell_albedo(hit.cell);
        ray = make_ray(hit.position + hit.normal * 0.75, cosine_hemisphere(hit.normal, rng_state));
    }

    return radiance;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    float2 pixel = float2(tid.xy);
    float3x3 dir_mat = float3x3(
        scene.pixel_to_ray[0].xyz,
        scene.pixel_to_ray[1].xyz,
        scene.pixel_to_ray[2].xyz
    );
    float3 origin = scene.pixel_to_ray[3].xyz;

    uint rng_state = hash_u32(
        tid.x * 1973u ^
        tid.y * 9277u ^
        scene.vpt_sample_index * 26699u ^
        uint(scene.time * 1000.0)
    );
    float2 jitter = float2(rand01(rng_state), rand01(rng_state)) - 0.5;
    float3 ray_dir = normalize(mul(float3(pixel + jitter, 1.0), dir_mat));
    float3 sample_radiance = trace_path(make_ray(origin, ray_dir), scene, rng_state);

    float sample_count = float(scene.vpt_sample_index + 1u);
    float3 accumulated = sample_radiance;
    if (scene.vpt_sample_index != 0u) {
        float3 previous = accumulation_image[tid.xy].rgb;
        accumulated = lerp(previous, sample_radiance, 1.0 / sample_count);
    }
    accumulation_image[tid.xy] = float4(accumulated, 1.0);
}
