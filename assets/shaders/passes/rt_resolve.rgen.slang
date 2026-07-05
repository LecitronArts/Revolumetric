#include "rt_history_common.slang"

[[vk::binding(0, 0)]]
Texture2D<float4> temporal_radiance;

[[vk::binding(1, 0)]]
RWTexture2D<float4> resolved_output;

[shader("raygeneration")]
void main() {
    uint3 launch_id = DispatchRaysIndex();
    float3 color = temporal_radiance.Load(int3(launch_id.xy, 0)).rgb;
    resolved_output[launch_id.xy] = float4(color, 1.0);
}
