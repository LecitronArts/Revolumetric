// VPT NRD resolve pass.
// Re-modulates NRD diffuse output and composites it with the non-denoised residual signal.

#include "scene_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> denoised_diff_radiance_hitdist;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> residual_radiance;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> material_factors;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> resolved_radiance;

static const float VPT_NRD_RESOLVE_MAX_RADIANCE = 65504.0;

float sanitize_nrd_resolve_scalar(float value) {
    if (value != value) {
        return 0.0;
    }
    return min(max(value, 0.0), VPT_NRD_RESOLVE_MAX_RADIANCE);
}

float3 sanitize_nrd_resolve_radiance(float3 radiance) {
    return float3(
        sanitize_nrd_resolve_scalar(radiance.x),
        sanitize_nrd_resolve_scalar(radiance.y),
        sanitize_nrd_resolve_scalar(radiance.z)
    );
}

float3 nrd_ycocg_to_linear(float3 color) {
    float y = color.x;
    float co = color.y;
    float cg = color.z;
    return float3(y + co - cg, y + cg, y - co - cg);
}

float3 unpack_nrd_diffuse_radiance(float3 radiance, bool use_reblur) {
    return use_reblur ? nrd_ycocg_to_linear(radiance) : radiance;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    bool use_reblur = (scene.denoiser_flags & DENOISER_MODE_MASK) == DENOISER_MODE_REBLUR;
    float3 denoised_diffuse = sanitize_nrd_resolve_radiance(unpack_nrd_diffuse_radiance(denoised_diff_radiance_hitdist[tid.xy].rgb, use_reblur));
    float3 residual = sanitize_nrd_resolve_radiance(residual_radiance[tid.xy].rgb);
    float3 albedo = saturate(material_factors[tid.xy].rgb);
    resolved_radiance[tid.xy] = float4(sanitize_nrd_resolve_radiance(residual + denoised_diffuse * albedo), 1.0);
}
