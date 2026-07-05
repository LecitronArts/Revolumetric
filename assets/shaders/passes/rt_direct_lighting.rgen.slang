#include "restir_di_common.slang"
#include "restir_gi_common.slang"
#include "rt_history_common.slang"
#include "rt_shadow_common.slang"

static const float RT_LIGHTING_INV_PI = 0.31830988618;
static const float RT_DIRECT_SURFACE_BIAS = 0.01;
static const float RT_DIRECT_LIGHT_ENDPOINT_BIAS = 0.01;

struct RtDirectLightingUniforms {
    uint restir_di_enabled;
    uint restir_gi_enabled;
    uint debug_view;
    uint width;
    uint height;
    uint shadows_enabled;
};

[[vk::binding(0, 0)]]
StructuredBuffer<RtSurfacePixel> surface_pixels;

[[vk::binding(1, 0)]]
StructuredBuffer<RestirDiReservoir> direct_reservoirs;

[[vk::binding(2, 0)]]
RWTexture2D<float4> current_radiance;

[[vk::binding(3, 0)]]
ConstantBuffer<RtDirectLightingUniforms> rt_direct;

[[vk::binding(4, 0)]]
RaytracingAccelerationStructure scene_tlas;

[[vk::binding(14, 0)]]
StructuredBuffer<RestirGiReservoir> indirect_reservoirs;

bool rt_direct_surface_valid(RtSurfacePixel surface) {
    return surface.hit_kind == RT_SURFACE_HIT_KIND_VOXEL && surface.position_depth.w >= 0.0;
}

float3 rt_direct_visualize_reservoir_weight(RestirDiReservoir reservoir) {
    float selected_weight = restir_di_bounded_selected_weight(reservoir);
    float heat = saturate(log2(max(selected_weight, 0.0) + 1.0) / 8.0);
    return float3(heat, heat * heat, 1.0 - heat);
}

bool rt_direct_light_visible(RtSurfacePixel surface, RestirDiReservoir reservoir) {
    if (rt_direct.shadows_enabled == 0u) {
        return true;
    }
    if (!rt_direct_surface_valid(surface)) {
        return false;
    }

    float3 shadow_dir;
    float max_light_t;
    if (reservoir.sample_flags == 0u) {
        shadow_dir = normalize(reservoir.sample_position_pdf.xyz);
        max_light_t = 1.0e20;
    } else {
        float3 to_light = reservoir.sample_position_pdf.xyz - surface.position_depth.xyz;
        float distance_sq = dot(to_light, to_light);
        if (distance_sq <= 1.0e-4) {
            return true;
        }
        float light_distance = sqrt(distance_sq);
        shadow_dir = to_light / light_distance;
        max_light_t = max(light_distance - RT_DIRECT_LIGHT_ENDPOINT_BIAS, 0.0);
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    RayDesc ray;
    ray.Origin = surface.position_depth.xyz + normal * RT_DIRECT_SURFACE_BIAS + shadow_dir * RT_DIRECT_SURFACE_BIAS;
    ray.Direction = shadow_dir;
    ray.TMin = 0.001;
    ray.TMax = max_light_t;

    RtShadowPayload payload = make_rt_shadow_payload(surface.brick_id, surface.local, max_light_t);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);
    return payload.occluded == 0u;
}

float3 rt_direct_resolve_reservoir(RtSurfacePixel surface, RestirDiReservoir reservoir, bool visible) {
    if (!rt_direct_surface_valid(surface) || !restir_di_is_valid_reservoir(reservoir)) {
        return float3(0.0);
    }
    if (!visible) {
        return float3(0.0);
    }

    float selected_weight = restir_di_bounded_selected_weight(reservoir);
    if (selected_weight <= 0.0) {
        return float3(0.0);
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    float3 albedo = surface.albedo_material.rgb;
    if (reservoir.sample_flags == 0u) {
        float3 light_dir = normalize(reservoir.sample_position_pdf.xyz);
        float sun_term = max(dot(normal, light_dir), 0.0);
        return albedo * RT_LIGHTING_INV_PI * reservoir.sample_radiance.rgb * sun_term * selected_weight;
    }

    float geometry_term = restir_di_emissive_geometry_term(
        surface.position_depth.xyz,
        normal,
        reservoir.sample_position_pdf.xyz
    );
    return albedo * RT_LIGHTING_INV_PI * reservoir.sample_radiance.rgb * geometry_term * selected_weight;
}

float3 rt_direct_resolve_indirect_reservoir(RtSurfacePixel surface, RestirGiReservoir reservoir) {
    if (!rt_direct_surface_valid(surface) || !restir_gi_is_valid_reservoir(reservoir)) {
        return float3(0.0);
    }

    float selected_weight = min(reservoir.selected_weight, RESTIR_GI_MAX_SELECTED_WEIGHT);
    if (selected_weight <= 0.0) {
        return float3(0.0);
    }

    float3 albedo = surface.albedo_material.rgb;
    float3 normal = normalize(surface.normal_roughness.xyz);
    float3 sample_normal = normalize(reservoir.sample_normal_roughness.xyz);
    float geometry = max(dot(normal, sample_normal), 0.0);
    return albedo
        * RT_LIGHTING_INV_PI
        * reservoir.sample_radiance_pdf.rgb
        * max(geometry, 0.05)
        * selected_weight;
}

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    uint index = launch_id.y * rt_direct.width + launch_id.x;
    if (launch_id.x >= rt_direct.width || launch_id.y >= rt_direct.height) {
        return;
    }

    RtSurfacePixel surface = surface_pixels[index];
    if (!rt_direct_surface_valid(surface)) {
        current_radiance[launch_id.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float3 albedo = surface.albedo_material.rgb;
    RestirDiReservoir reservoir = direct_reservoirs[index];
    if (rt_direct.restir_di_enabled != 0u && rt_direct.debug_view == RT_DEBUG_VIEW_DIRECT_RESERVOIR) {
        current_radiance[launch_id.xy] = float4(rt_direct_visualize_reservoir_weight(reservoir), 1.0);
        return;
    }

    float3 direct = float3(0.0);
    if (rt_direct.restir_di_enabled != 0u) {
        bool visible = rt_direct_light_visible(surface, reservoir);
        direct = rt_direct_resolve_reservoir(surface, reservoir, visible);
    }

    RestirGiReservoir indirect_reservoir = indirect_reservoirs[index];
    float3 indirect = rt_direct.restir_gi_enabled != 0u
        ? rt_direct_resolve_indirect_reservoir(surface, indirect_reservoir)
        : float3(0.0);

    current_radiance[launch_id.xy] = float4(albedo + direct + indirect, 1.0);
}
