use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::rc_probe_buffer::{self, RcProbeBuffer};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::voxel::gpu_upload::UcvhGpuResources;

#[repr(C)]
#[derive(Clone, Copy)]
struct RcTracePushConstants {
    cascade_level: u32,
    probe_grid_dim: u32,
    probe_size: u32,
    buffer_offset: u32,
}

struct CascadeParams {
    level: u32,
    grid_dim: u32,
    probe_size: u32,
    offset: u32,
    total_invocations: u32,
}

const CASCADE_PARAMS: [CascadeParams; 3] = [
    CascadeParams {
        level: 0,
        grid_dim: 16,
        probe_size: 3,
        offset: rc_probe_buffer::RC_C0_OFFSET,
        total_invocations: 16 * 16 * 16 * 6 * 9,
    },
    CascadeParams {
        level: 1,
        grid_dim: 8,
        probe_size: 6,
        offset: rc_probe_buffer::RC_C1_OFFSET,
        total_invocations: 8 * 8 * 8 * 6 * 36,
    },
    CascadeParams {
        level: 2,
        grid_dim: 4,
        probe_size: 12,
        offset: rc_probe_buffer::RC_C2_OFFSET,
        total_invocations: 4 * 4 * 4 * 6 * 144,
    },
];

pub struct RcTracePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RcTracePass {
    pub fn new(
        device: &ash::Device,
        _allocator: &GpuAllocator,
        spirv_bytes: &[u8],
        ucvh_gpu: &UcvhGpuResources,
        scene_ubo: &SceneUniformBuffer,
        rc_probes: &RcProbeBuffer,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                1,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                2,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                3,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                4,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                5,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                6,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 6 * frame_count as u32,
            },
        ];
        let descriptor_pool = match DescriptorPool::new(device, frame_count as u32, &pool_sizes) {
            Ok(pool) => pool,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let layouts: Vec<_> = (0..frame_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = match descriptor_pool.allocate(device, &layouts) {
            Ok(sets) => sets,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        let ucvh_buffers = [
            &ucvh_gpu.config_buffer,
            &ucvh_gpu.hierarchy_l0_buffer,
            &ucvh_gpu.occupancy_buffer,
            &ucvh_gpu.material_buffer,
        ];

        for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(scene_ubo.buffer_handle(set_idx))
                .offset(0)
                .range(std::mem::size_of::<GpuSceneUniforms>() as u64);

            let ubo_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info));

            let ucvh_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers
                .iter()
                .map(|buf| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.handle)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();

            let ucvh_writes: Vec<vk::WriteDescriptorSet> = ucvh_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding((i + 1) as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            let read_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.read_buffer(set_idx))
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let read_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&read_info));

            let write_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer(set_idx))
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&write_info));

            let mut all_writes = vec![ubo_write];
            all_writes.extend(ucvh_writes);
            all_writes.push(read_write);
            all_writes.push(write_write);
            unsafe { device.update_descriptor_sets(&all_writes, &[]) };
        }

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<RcTracePushConstants>() as u32);

        let shader_module = match create_shader_module(device, spirv_bytes) {
            Ok(module) => module,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let pipeline = match ComputePipeline::new(
            device,
            shader_module,
            c"main",
            &[descriptor_set_layout],
            &[push_range],
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe { device.destroy_shader_module(shader_module, None) };
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
        })
    }

    pub fn update_probe_descriptors(
        &self,
        device: &ash::Device,
        rc_probes: &RcProbeBuffer,
        frame_slot: usize,
    ) {
        let ds = self.descriptor_sets[frame_slot];
        let read_info = vk::DescriptorBufferInfo::default()
            .buffer(rc_probes.read_buffer(frame_slot))
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let read_write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&read_info));

        let write_info = vk::DescriptorBufferInfo::default()
            .buffer(rc_probes.write_buffer(frame_slot))
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let write_write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&write_info));

        unsafe { device.update_descriptor_sets(&[read_write, write_write], &[]) };
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        self.bind(device, cmd, frame_slot);

        for cascade_index in 0..CASCADE_PARAMS.len() {
            self.record_cascade(device, cmd, cascade_index);
        }
    }

    pub fn bind(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.descriptor_sets[frame_slot]],
                &[],
            );
        }
    }

    pub fn record_cascade(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        cascade_index: usize,
    ) {
        let params = CASCADE_PARAMS
            .get(cascade_index)
            .expect("radiance cascade trace index out of range");
        let push = RcTracePushConstants {
            cascade_level: params.level,
            probe_grid_dim: params.grid_dim,
            probe_size: params.probe_size,
            buffer_offset: params.offset,
        };
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                &push as *const RcTracePushConstants as *const u8,
                std::mem::size_of::<RcTracePushConstants>(),
            )
        };
        unsafe {
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(cmd, params.total_invocations.div_ceil(64), 1, 1);
        }
    }

    pub fn destroy(self, device: &ash::Device, _allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rc_trace_temporally_blends_radiance_but_keeps_current_distance() {
        let source =
            include_str!("../../../assets/shaders/passes/rc_trace.slang").replace("\r\n", "\n");

        assert!(source.contains("float4 rc_temporal_blend(float4 current, float4 previous)"));
        assert!(source.contains(
            "return float4(lerp(previous.xyz, current.xyz, RC_TEMPORAL_BLEND), current.w);"
        ));
        assert!(source.contains("probe_write[idx] = rc_temporal_blend(result, probe_read[idx]);"));
    }

    #[test]
    fn rc_trace_uses_shared_material_helpers() {
        let source =
            include_str!("../../../assets/shaders/passes/rc_trace.slang").replace("\r\n", "\n");

        assert!(source.contains("#include \"material_common.slang\""));
        assert!(source.contains("material_emissive(hit.cell)"));
        assert!(source.contains("material_cell_albedo(hit.cell)"));
        assert!(!source.contains("if (mat_id == 1u)"));
    }

    #[test]
    fn rc_trace_uses_the_same_grid_center_positions_as_sampling_paths() {
        let source =
            include_str!("../../../assets/shaders/passes/rc_trace.slang").replace("\r\n", "\n");
        let shared = include_str!("../../../assets/shaders/shared/radiance_cascade.slang")
            .replace("\r\n", "\n");

        assert!(
            shared
                .contains("float3 rc_probe_world_position(uint3 probe_coord, uint cascade_level)"),
            "trace and merge should share the same probe_coord -> world position mapping"
        );
        assert!(
            shared.contains("float3 rc_world_to_c0_grid_position(float3 world_pos)"),
            "lighting should share the inverse C0 world -> grid mapping"
        );
        assert!(
            source.contains(
                "float3 probe_pos = rc_probe_world_position(probe_coord, push.cascade_level);"
            ),
            "trace should use the shared probe position helper"
        );
        assert!(
            !source.contains("rc_find_nearest_empty_probe_position"),
            "C0 trace must not relocate a probe slot away from the world position used by integrate_probe and rc_merge"
        );

        let inside_geo_rejection = source
            .find("if (!rc_outside_geo(probe_pos")
            .expect("rc_trace should still reject probes trapped inside geometry");

        assert!(
            source.find("float3 probe_pos = rc_probe_world_position") < Some(inside_geo_rejection),
            "trace should reject inside-geometry probes at the same grid-center position used by merge"
        );
        assert!(
            !source.contains("rc_geo_offset"),
            "trace must not offset C1/C2 probe origins unless merge uses the same offset positions"
        );
    }

    #[test]
    fn rc_geometry_queries_use_uploaded_world_dimensions() {
        let shared = include_str!("../../../assets/shaders/shared/radiance_cascade.slang")
            .replace("\r\n", "\n");
        let trace =
            include_str!("../../../assets/shaders/passes/rc_trace.slang").replace("\r\n", "\n");

        assert!(
            shared.contains("uint3 brick_grid_size,\n                    uint3 world_size"),
            "RC geometry queries should receive uploaded UCVH dimensions"
        );
        assert!(
            !shared.contains("int3(128)"),
            "RC geometry queries must not hard-code a 128^3 world"
        );
        assert!(
            trace.contains("uint3 brick_grid_size = ucvh_config[0].brick_grid_size.xyz;"),
            "rc_trace should pass the uploaded brick grid dimensions"
        );
        assert!(
            trace.contains("uint3 world_size = ucvh_config[0].world_size.xyz;"),
            "rc_trace should pass the uploaded world dimensions"
        );
    }
}
