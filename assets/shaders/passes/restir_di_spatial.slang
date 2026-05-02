#include "restir_di_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(1, 0)]]
StructuredBuffer<RestirDiReservoir> temporal_reservoirs;

[[vk::binding(2, 0)]]
RWStructuredBuffer<RestirDiReservoir> output_reservoirs;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.y * restir.width + tid.x;
    if (tid.x >= restir.width || tid.y >= restir.height || index >= restir.reservoir_count) return;

    output_reservoirs[index] = temporal_reservoirs[index];
}
