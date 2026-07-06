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
    uint2 _pad0;
    float4 sky_color_sun_angular_radius;
    float4 ground_color_pad;
    float4 sun_direction_pad;
    float4 sun_intensity_pad;
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

static const float RT_DIRECT_TWO_PI = 6.28318530718;
static const float RT_DIRECT_INV_PI = 0.31830988618;

float rt_direct_sun_disk_solid_angle() {
    float sun_radius = rt_direct.sky_color_sun_angular_radius.w;
    if (sun_radius <= 0.0) {
        return 0.0;
    }

    return max(RT_DIRECT_TWO_PI * (1.0 - cos(sun_radius)), 1.0e-8);
}

float3 rt_direct_sunlit_ground_color() {
    float ground_ndotl = max(rt_direct.sun_direction_pad.y, 0.0);
    float3 finite_sun_irradiance =
        rt_direct.sun_intensity_pad.rgb * ground_ndotl * rt_direct_sun_disk_solid_angle();
    return rt_direct.ground_color_pad.rgb * (1.0 + RT_DIRECT_INV_PI * finite_sun_irradiance);
}

float3 rt_direct_background_color_for_dir(float3 direction) {
    float3 miss_dir = normalize(direction);
    float t = saturate(miss_dir.y * 0.5 + 0.5);
    return lerp(rt_direct_sunlit_ground_color(), rt_direct.sky_color_sun_angular_radius.rgb, t);
}

float3 rt_direct_background_color(RtSurfacePixel surface) {
    float3 miss_dir = normalize(surface.view_direction_background.xyz);
    return rt_direct_background_color_for_dir(miss_dir);
}

float3 rt_direct_sky_visibility_sample(RtSurfacePixel surface, float3 sky_dir) {
    float3 normal = normalize(surface.normal_roughness.xyz);

    RayDesc ray;
    ray.Origin = surface.position_depth.xyz + normal * RT_DIRECT_SURFACE_BIAS + sky_dir * RT_DIRECT_SURFACE_BIAS;
    ray.Direction = sky_dir;
    ray.TMin = 0.001;
    ray.TMax = 1.0e20;

    RtShadowPayload payload = make_rt_shadow_payload(surface.brick_id, surface.local, ray.TMax);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);
    if (payload.occluded != 0u) {
        return float3(0.0);
    }

    return rt_direct_background_color_for_dir(sky_dir);
}

float3 rt_direct_sample_sky_indirect(RtSurfacePixel surface) {
    float3 normal = normalize(surface.normal_roughness.xyz);
    float3 sky_dir = normal;
    return surface.albedo_material.rgb * rt_direct_sky_visibility_sample(surface, sky_dir);
}

bool rt_direct_sun_visible(RtSurfacePixel surface, float3 sun_dir) {
    if (rt_direct.shadows_enabled == 0u) {
        return true;
    }
    if (!rt_direct_surface_valid(surface)) {
        return false;
    }

    float3 normal = normalize(surface.normal_roughness.xyz);
    RayDesc ray;
    ray.Origin = surface.position_depth.xyz + normal * RT_DIRECT_SURFACE_BIAS + sun_dir * RT_DIRECT_SURFACE_BIAS;
    ray.Direction = sun_dir;
    ray.TMin = 0.001;
    ray.TMax = 1.0e20;

    RtShadowPayload payload = make_rt_shadow_payload(surface.brick_id, surface.local, ray.TMax);
    TraceRay(scene_tlas, 0u, 0xffu, 0u, 0u, 0u, ray, payload);
    return payload.occluded == 0u;
}

float3 rt_direct_analytic_sun_direct(RtSurfacePixel surface) {
    float solid_angle = rt_direct_sun_disk_solid_angle();
    float3 sun_vector = rt_direct.sun_direction_pad.xyz;
    if (solid_angle <= 0.0 || dot(sun_vector, sun_vector) <= 1.0e-8) {
        return float3(0.0);
    }

    float3 sun_dir = normalize(sun_vector);
    float3 normal = normalize(surface.normal_roughness.xyz);
    float sun_term = max(dot(normal, sun_dir), 0.0);
    if (sun_term <= 0.0 || !rt_direct_sun_visible(surface, sun_dir)) {
        return float3(0.0);
    }

    return surface.albedo_material.rgb
        * RT_LIGHTING_INV_PI
        * rt_direct.sun_intensity_pad.rgb
        * sun_term
        * solid_angle;
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
        current_radiance[launch_id.xy] = float4(rt_direct_background_color(surface), 1.0);
        return;
    }

    RestirDiReservoir reservoir = direct_reservoirs[index];
    if (rt_direct.restir_di_enabled != 0u && rt_direct.debug_view == RT_DEBUG_VIEW_DIRECT_RESERVOIR) {
        current_radiance[launch_id.xy] = float4(rt_direct_visualize_reservoir_weight(reservoir), 1.0);
        return;
    }

    float3 sky_indirect = rt_direct_sample_sky_indirect(surface);
    float3 analytic_direct = rt_direct_analytic_sun_direct(surface);
    float3 direct = float3(0.0);
    if (rt_direct.restir_di_enabled != 0u) {
        bool visible = rt_direct_light_visible(surface, reservoir);
        direct = rt_direct_resolve_reservoir(surface, reservoir, visible);
    }

    RestirGiReservoir indirect_reservoir = indirect_reservoirs[index];
    float3 indirect = rt_direct.restir_gi_enabled != 0u
        ? rt_direct_resolve_indirect_reservoir(surface, indirect_reservoir)
        : float3(0.0);

    current_radiance[launch_id.xy] = float4(sky_indirect + analytic_direct + direct + indirect, 1.0);
}
