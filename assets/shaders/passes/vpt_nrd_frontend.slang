// VPT NRD frontend packing pass.
// Converts raw VPT NRD noisy resources into the local NRD input contract.

#include "scene_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> input_diff_radiance_hitdist;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> input_spec_radiance_hitdist;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> input_residual_radiance;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> input_material_factors;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_normal_roughness;

[[vk::binding(6, 0)]]
[[vk::image_format("r16f")]]
RWTexture2D<float> surface_material_roughness;

[[vk::binding(7, 0)]]
[[vk::image_format("r32f")]]
RWTexture2D<float> surface_view_z;

[[vk::binding(8, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> packed_diff_radiance_hitdist;

[[vk::binding(9, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> packed_spec_radiance_hitdist;

[[vk::binding(10, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> residual_radiance;

[[vk::binding(11, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> material_factors;

[[vk::binding(12, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> packed_normal_roughness;

static const float VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE = 65504.0;
static const float3 VPT_NRD_REBLUR_HIT_DISTANCE_PARAMETERS = float3(3.0, 0.1, 20.0);

float sanitize_relax_scalar(float value) {
    if (value != value) {
        return 0.0;
    }
    return min(max(value, 0.0), VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE);
}

float3 sanitize_relax_radiance(float3 radiance) {
    return float3(
        sanitize_relax_scalar(radiance.x),
        sanitize_relax_scalar(radiance.y),
        sanitize_relax_scalar(radiance.z)
    );
}

float sanitize_relax_hit_distance(float hit_distance) {
    if (hit_distance != hit_distance || hit_distance < 0.0) {
        return VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE;
    }
    return min(hit_distance, VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE);
}

float4 pack_relax_radiance_hitdist(float4 radiance_hitdist) {
    return float4(sanitize_relax_radiance(radiance_hitdist.rgb), sanitize_relax_hit_distance(radiance_hitdist.a));
}

float3 nrd_linear_to_ycocg(float3 color) {
    float y = dot(color, float3(0.25, 0.5, 0.25));
    float co = dot(color, float3(0.5, 0.0, -0.5));
    float cg = dot(color, float3(-0.25, 0.5, -0.25));
    return float3(y, co, cg);
}

float _NRD_GetSpecMagicCurve(float roughness, float power) {
    float f = 1.0 - exp2(-200.0 * roughness * roughness);
    f *= pow(saturate(roughness), power);
    return f;
}

float _REBLUR_GetHitDistanceNormalization(float view_z, float roughness, float3 hit_distance_parameters) {
    float smc = _NRD_GetSpecMagicCurve(roughness, 0.5);
    return (hit_distance_parameters.x + abs(view_z) * hit_distance_parameters.y) * lerp(hit_distance_parameters.z, 1.0, smc);
}

float sanitize_reblur_view_z(float view_z) {
    if (view_z != view_z) {
        return 0.0;
    }
    return view_z;
}

bool is_valid_reblur_hit_distance(float hit_distance) {
    return hit_distance == hit_distance
        && hit_distance >= 0.0
        && hit_distance < VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE;
}

float REBLUR_FrontEnd_GetNormHitDist(float hit_distance, float view_z, float roughness, float3 hit_distance_parameters) {
    float f = _REBLUR_GetHitDistanceNormalization(sanitize_reblur_view_z(view_z), roughness, hit_distance_parameters);
    return saturate(hit_distance / max(f, 1.0e-6));
}

float4 REBLUR_FrontEnd_PackRadianceAndNormHitDist(float4 radiance_hitdist, float view_z, float roughness) {
    float norm_hit_distance = is_valid_reblur_hit_distance(radiance_hitdist.a)
        ? REBLUR_FrontEnd_GetNormHitDist(radiance_hitdist.a, view_z, roughness, VPT_NRD_REBLUR_HIT_DISTANCE_PARAMETERS)
        : 0.0;
    return float4(nrd_linear_to_ycocg(sanitize_relax_radiance(radiance_hitdist.rgb)), saturate(norm_hit_distance));
}

float3 sanitize_nrd_normal(float3 normal) {
    float length_sq = dot(normal, normal);
    if (length_sq != length_sq || length_sq <= 1.0e-8) {
        return float3(0.0, 0.0, 1.0);
    }
    return normal * rsqrt(length_sq);
}

float2 encode_nrd_unit_vector_oct(float3 normal) {
    normal /= max(abs(normal.x) + abs(normal.y) + abs(normal.z), 1.0e-8);

    float2 encoded = normal.xy;
    if (normal.z < 0.0) {
        encoded = (1.0 - abs(encoded.yx)) * (step(0.0, encoded) * 2.0 - 1.0);
    }
    return encoded * 0.5 + 0.5;
}

float4 pack_nrd_normal_roughness(float3 normal, float roughness, float material_id) {
    float2 encoded_normal = encode_nrd_unit_vector_oct(sanitize_nrd_normal(normal));
    return float4(encoded_normal, saturate(roughness), saturate(material_id / 3.0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    bool use_reblur = (scene.denoiser_flags & DENOISER_MODE_MASK) == DENOISER_MODE_REBLUR;
    float view_z = surface_view_z[tid.xy];
    float roughness = surface_material_roughness[tid.xy];
    packed_diff_radiance_hitdist[tid.xy] = use_reblur ? REBLUR_FrontEnd_PackRadianceAndNormHitDist(input_diff_radiance_hitdist[tid.xy], view_z, roughness) : pack_relax_radiance_hitdist(input_diff_radiance_hitdist[tid.xy]);
    packed_spec_radiance_hitdist[tid.xy] = pack_relax_radiance_hitdist(input_spec_radiance_hitdist[tid.xy]);
    residual_radiance[tid.xy] = float4(sanitize_relax_radiance(input_residual_radiance[tid.xy].rgb), 1.0);
    material_factors[tid.xy] = float4(saturate(input_material_factors[tid.xy].rgb), input_material_factors[tid.xy].a);
    packed_normal_roughness[tid.xy] = pack_nrd_normal_roughness(surface_normal_roughness[tid.xy].xyz, surface_material_roughness[tid.xy], input_material_factors[tid.xy].a);
}
