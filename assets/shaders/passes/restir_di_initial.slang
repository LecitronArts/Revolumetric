#include "restir_di_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(1, 0)]]
StructuredBuffer<DirectLight> direct_lights;

[[vk::binding(2, 0)]]
RWStructuredBuffer<RestirDiReservoir> output_reservoirs;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.y * restir.width + tid.x;
    if (tid.x >= restir.width || tid.y >= restir.height || index >= restir.reservoir_count) return;

    RestirDiReservoir reservoir = (RestirDiReservoir)0;
    reservoir.sample_light_id = 0xffffffffu;
    reservoir.sample_count_m = 0u;
    output_reservoirs[index] = reservoir;
}
