#include "lighting_common.slang"
#include "rt_history_common.slang"

struct RtResolveUniforms {
    float exposure;
    float3 _pad0;
};

[[vk::binding(0, 0)]]
Texture2D<float4> temporal_radiance;

[[vk::binding(1, 0)]]
RWTexture2D<float4> resolved_output;

[[vk::binding(2, 0)]]
ConstantBuffer<RtResolveUniforms> rt_resolve;

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    float3 hdr = max(temporal_radiance.Load(int3(launch_id.xy, 0)).rgb * rt_resolve.exposure, float3(0.0));
    float3 mapped = aces_tonemap(hdr);
    float3 ldr = pow(mapped, float3(1.0 / 2.2));
    resolved_output[launch_id.xy] = float4(ldr, 1.0);
}
