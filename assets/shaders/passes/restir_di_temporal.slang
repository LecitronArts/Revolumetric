#include "restir_di_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(1, 0)]]
StructuredBuffer<RestirDiReservoir> input_reservoirs;

[[vk::binding(2, 0)]]
StructuredBuffer<RestirDiReservoir> history_reservoirs;

[[vk::binding(3, 0)]]
RWStructuredBuffer<RestirDiReservoir> output_reservoirs;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.y * restir.width + tid.x;
    if (tid.x >= restir.width || tid.y >= restir.height || index >= restir.reservoir_count) return;

    RestirDiReservoir reservoir = input_reservoirs[index];
    if (restir.temporal_enabled != 0u && history_reservoirs[index].sample_light_id != 0xffffffffu) {
        reservoir.sample_count_m += history_reservoirs[index].sample_count_m;
    }
    output_reservoirs[index] = reservoir;
}
