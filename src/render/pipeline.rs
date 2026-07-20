use anyhow::{Context, Result, anyhow};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

pub struct ComputePipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl ComputePipeline {
    pub fn new(
        device: &ash::Device,
        shader_module: vk::ShaderModule,
        entry_point: &std::ffi::CStr,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .context("failed to create pipeline layout")?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let handle = match unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        } {
            Ok(mut pipelines) => match pipelines.pop() {
                Some(pipeline) => pipeline,
                None => {
                    unsafe { device.destroy_pipeline_layout(layout, None) };
                    return Err(anyhow!("Vulkan returned no compute pipelines"));
                }
            },
            Err((pipelines, error)) => {
                unsafe {
                    for pipeline in pipelines {
                        device.destroy_pipeline(pipeline, None);
                    }
                    device.destroy_pipeline_layout(layout, None);
                }
                return Err(error).context("failed to create compute pipeline");
            }
        };

        Ok(Self { handle, layout })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.handle, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub fn create_shader_module(device: &ash::Device, spirv: &[u8]) -> Result<vk::ShaderModule> {
    assert!(
        spirv.len().is_multiple_of(4),
        "SPIR-V byte length must be a multiple of 4"
    );
    // Copy into an aligned Vec<u32> to avoid UB from misaligned &[u8] → &[u32] cast.
    // include_bytes!() only guarantees 1-byte alignment.
    let mut code = vec![0u32; spirv.len() / 4];
    unsafe {
        std::ptr::copy_nonoverlapping(spirv.as_ptr(), code.as_mut_ptr() as *mut u8, spirv.len());
    }
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&create_info, None) }
        .context("failed to create shader module")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RayTracingShaderGroupCounts {
    pub raygen: u32,
    pub miss: u32,
    pub hit: u32,
    pub callable: u32,
}

impl RayTracingShaderGroupCounts {
    fn total(self) -> u32 {
        self.raygen + self.miss + self.hit + self.callable
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderBindingTableRegionLayout {
    pub offset: vk::DeviceSize,
    pub stride: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderBindingTableLayout {
    pub handle_size: vk::DeviceSize,
    pub handle_stride: vk::DeviceSize,
    pub total_size: vk::DeviceSize,
    pub raygen: ShaderBindingTableRegionLayout,
    pub miss: ShaderBindingTableRegionLayout,
    pub hit: ShaderBindingTableRegionLayout,
    pub callable: ShaderBindingTableRegionLayout,
}

impl ShaderBindingTableLayout {
    pub fn new(
        handle_size: u32,
        handle_alignment: u32,
        base_alignment: u32,
        group_counts: RayTracingShaderGroupCounts,
    ) -> Result<Self> {
        if group_counts.raygen != 1 {
            return Err(anyhow!(
                "ray tracing shader binding table requires exactly one raygen group"
            ));
        }
        let handle_size = u64::from(handle_size);
        let handle_alignment = u64::from(handle_alignment);
        let base_alignment = u64::from(base_alignment);
        let handle_stride = align_up(handle_size, handle_alignment)
            .ok_or_else(|| anyhow!("invalid SBT handle alignment {handle_alignment}"))?;

        let mut offset = 0;
        let raygen = make_sbt_region(&mut offset, group_counts.raygen, handle_stride, 1)?;
        let miss = make_sbt_region(
            &mut offset,
            group_counts.miss,
            handle_stride,
            base_alignment,
        )?;
        let hit = make_sbt_region(&mut offset, group_counts.hit, handle_stride, base_alignment)?;
        let callable = make_sbt_region(
            &mut offset,
            group_counts.callable,
            handle_stride,
            base_alignment,
        )?;

        Ok(Self {
            handle_size,
            handle_stride,
            total_size: offset,
            raygen,
            miss,
            hit,
            callable,
        })
    }
}

#[derive(Clone, Copy)]
pub struct ShaderBindingTableRegions {
    pub raygen: vk::StridedDeviceAddressRegionKHR,
    pub miss: vk::StridedDeviceAddressRegionKHR,
    pub hit: vk::StridedDeviceAddressRegionKHR,
    pub callable: vk::StridedDeviceAddressRegionKHR,
}

pub struct ShaderBindingTable {
    buffer: GpuBuffer,
    regions: ShaderBindingTableRegions,
}

impl ShaderBindingTable {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        ray_tracing_pipeline_loader: &ash::khr::ray_tracing_pipeline::Device,
        pipeline: vk::Pipeline,
        properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'_>,
        group_counts: RayTracingShaderGroupCounts,
    ) -> Result<Self> {
        let layout = ShaderBindingTableLayout::new(
            properties.shader_group_handle_size,
            properties.shader_group_handle_alignment,
            properties.shader_group_base_alignment,
            group_counts,
        )?;
        let group_count = group_counts.total();
        let handle_data_size = layout
            .handle_size
            .checked_mul(u64::from(group_count))
            .ok_or_else(|| anyhow!("SBT shader group handle data size overflow"))?
            as usize;
        let handles = unsafe {
            ray_tracing_pipeline_loader.get_ray_tracing_shader_group_handles(
                pipeline,
                0,
                group_count,
                handle_data_size,
            )
        }
        .context("failed to query ray tracing shader group handles")?;
        let buffer = GpuBuffer::new(
            device,
            allocator,
            layout.total_size,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
            "rt_shader_binding_table",
        )?;
        write_sbt_handles(&buffer, &handles, &layout, group_counts)?;

        let base_address = buffer_device_address(device, buffer.handle);
        let regions = ShaderBindingTableRegions {
            raygen: make_sbt_region_address(base_address, layout.raygen),
            miss: make_sbt_region_address(base_address, layout.miss),
            hit: make_sbt_region_address(base_address, layout.hit),
            callable: make_sbt_region_address(base_address, layout.callable),
        };
        Ok(Self { buffer, regions })
    }

    pub fn regions(&self) -> ShaderBindingTableRegions {
        self.regions
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.buffer.destroy(device, allocator);
    }
}

pub struct RayTracingPipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub group_counts: RayTracingShaderGroupCounts,
}

#[derive(Debug, Clone, Copy)]
pub struct RtShaderStageSpec<'a> {
    pub stage: vk::ShaderStageFlags,
    pub module: vk::ShaderModule,
    pub entry_point: &'a std::ffi::CStr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtHitGroupKind {
    Procedural {
        closest_hit_stage: u32,
        intersection_stage: u32,
    },
    Triangles {
        closest_hit_stage: u32,
    },
}

#[derive(Debug)]
pub struct RtMixedSurfaceGroupPlan {
    pub groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR<'static>>,
    pub group_counts: RayTracingShaderGroupCounts,
}

impl RtMixedSurfaceGroupPlan {
    pub fn new(stages: &[RtShaderStageSpec<'_>], hit_groups: &[RtHitGroupKind]) -> Result<Self> {
        if stages
            .iter()
            .any(|stage| stage.stage == vk::ShaderStageFlags::ANY_HIT_KHR)
        {
            return Err(anyhow!(
                "mixed surface pipeline does not allow any-hit stages"
            ));
        }
        if !matches!(
            hit_groups,
            [
                RtHitGroupKind::Procedural { .. },
                RtHitGroupKind::Triangles { .. }
            ]
        ) {
            return Err(anyhow!(
                "mixed page surface hit groups must be Reference procedural then CompactExact triangles"
            ));
        }
        let raygen_stage = unique_stage_index(stages, vk::ShaderStageFlags::RAYGEN_KHR)?;
        let miss_stage = unique_stage_index(stages, vk::ShaderStageFlags::MISS_KHR)?;
        let mut groups = Vec::with_capacity(2 + hit_groups.len());
        groups.push(general_shader_group(raygen_stage));
        groups.push(general_shader_group(miss_stage));
        for hit_group in hit_groups {
            let group = match *hit_group {
                RtHitGroupKind::Procedural {
                    closest_hit_stage,
                    intersection_stage,
                } => {
                    validate_stage_index(
                        stages,
                        closest_hit_stage,
                        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    )?;
                    validate_stage_index(
                        stages,
                        intersection_stage,
                        vk::ShaderStageFlags::INTERSECTION_KHR,
                    )?;
                    vk::RayTracingShaderGroupCreateInfoKHR::default()
                        .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                        .general_shader(vk::SHADER_UNUSED_KHR)
                        .closest_hit_shader(closest_hit_stage)
                        .any_hit_shader(vk::SHADER_UNUSED_KHR)
                        .intersection_shader(intersection_stage)
                }
                RtHitGroupKind::Triangles { closest_hit_stage } => {
                    validate_stage_index(
                        stages,
                        closest_hit_stage,
                        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    )?;
                    vk::RayTracingShaderGroupCreateInfoKHR::default()
                        .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                        .general_shader(vk::SHADER_UNUSED_KHR)
                        .closest_hit_shader(closest_hit_stage)
                        .any_hit_shader(vk::SHADER_UNUSED_KHR)
                        .intersection_shader(vk::SHADER_UNUSED_KHR)
                }
            };
            groups.push(group);
        }
        let hit_count = u32::try_from(hit_groups.len())
            .map_err(|_| anyhow!("mixed surface hit-group count exceeds u32"))?;
        Ok(Self {
            groups,
            group_counts: RayTracingShaderGroupCounts {
                raygen: 1,
                miss: 1,
                hit: hit_count,
                callable: 0,
            },
        })
    }
}

fn unique_stage_index(
    stages: &[RtShaderStageSpec<'_>],
    expected: vk::ShaderStageFlags,
) -> Result<u32> {
    let matches = stages
        .iter()
        .enumerate()
        .filter(|(_, stage)| stage.stage == expected)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(anyhow!(
            "mixed surface pipeline requires exactly one {expected:?} stage, found {}",
            matches.len()
        ));
    }
    u32::try_from(matches[0]).map_err(|_| anyhow!("shader stage index exceeds u32"))
}

fn validate_stage_index(
    stages: &[RtShaderStageSpec<'_>],
    index: u32,
    expected: vk::ShaderStageFlags,
) -> Result<()> {
    let stage = stages
        .get(index as usize)
        .ok_or_else(|| anyhow!("shader stage index {index} is out of range"))?;
    if stage.stage != expected {
        return Err(anyhow!(
            "shader stage {index} must be {expected:?}, found {:?}",
            stage.stage
        ));
    }
    Ok(())
}

fn general_shader_group(stage: u32) -> vk::RayTracingShaderGroupCreateInfoKHR<'static> {
    vk::RayTracingShaderGroupCreateInfoKHR::default()
        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
        .general_shader(stage)
        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR)
}

impl RayTracingPipeline {
    pub fn new_mixed_surface_pipeline(
        device: &ash::Device,
        ray_tracing_pipeline_loader: &ash::khr::ray_tracing_pipeline::Device,
        stages: &[RtShaderStageSpec<'_>],
        hit_groups: &[RtHitGroupKind],
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        let group_plan = RtMixedSurfaceGroupPlan::new(stages, hit_groups)?;
        let layout = create_ray_tracing_pipeline_layout(
            device,
            descriptor_set_layouts,
            push_constant_ranges,
        )?;
        let vk_stages = stages
            .iter()
            .map(|stage| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(stage.stage)
                    .module(stage.module)
                    .name(stage.entry_point)
            })
            .collect::<Vec<_>>();
        let handle = create_ray_tracing_pipeline(
            device,
            ray_tracing_pipeline_loader,
            layout,
            &vk_stages,
            &group_plan.groups,
        )?;
        Ok(Self {
            handle,
            layout,
            group_counts: group_plan.group_counts,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_surface_pipeline(
        device: &ash::Device,
        ray_tracing_pipeline_loader: &ash::khr::ray_tracing_pipeline::Device,
        raygen_module: vk::ShaderModule,
        miss_module: vk::ShaderModule,
        closest_hit_module: vk::ShaderModule,
        intersection_module: vk::ShaderModule,
        entry_point: &std::ffi::CStr,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        let stages = [
            RtShaderStageSpec {
                stage: vk::ShaderStageFlags::RAYGEN_KHR,
                module: raygen_module,
                entry_point,
            },
            RtShaderStageSpec {
                stage: vk::ShaderStageFlags::MISS_KHR,
                module: miss_module,
                entry_point,
            },
            RtShaderStageSpec {
                stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                module: closest_hit_module,
                entry_point,
            },
            RtShaderStageSpec {
                stage: vk::ShaderStageFlags::INTERSECTION_KHR,
                module: intersection_module,
                entry_point,
            },
        ];
        Self::new_mixed_surface_pipeline(
            device,
            ray_tracing_pipeline_loader,
            &stages,
            &[
                RtHitGroupKind::Procedural {
                    closest_hit_stage: 2,
                    intersection_stage: 3,
                },
                RtHitGroupKind::Triangles {
                    closest_hit_stage: 2,
                },
            ],
            descriptor_set_layouts,
            push_constant_ranges,
        )
    }

    pub fn new_raygen_only(
        device: &ash::Device,
        ray_tracing_pipeline_loader: &ash::khr::ray_tracing_pipeline::Device,
        shader_module: vk::ShaderModule,
        entry_point: &std::ffi::CStr,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        let layout = create_ray_tracing_pipeline_layout(
            device,
            descriptor_set_layouts,
            push_constant_ranges,
        )?;

        let stages = [vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(shader_module)
            .name(entry_point)];
        let groups = [vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)];
        let handle = create_ray_tracing_pipeline(
            device,
            ray_tracing_pipeline_loader,
            layout,
            &stages,
            &groups,
        )?;

        Ok(Self {
            handle,
            layout,
            group_counts: RayTracingShaderGroupCounts {
                raygen: 1,
                miss: 0,
                hit: 0,
                callable: 0,
            },
        })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.handle, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

fn create_ray_tracing_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    push_constant_ranges: &[vk::PushConstantRange],
) -> Result<vk::PipelineLayout> {
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(descriptor_set_layouts)
        .push_constant_ranges(push_constant_ranges);
    unsafe { device.create_pipeline_layout(&layout_info, None) }
        .context("failed to create ray tracing pipeline layout")
}

fn create_ray_tracing_pipeline(
    device: &ash::Device,
    ray_tracing_pipeline_loader: &ash::khr::ray_tracing_pipeline::Device,
    layout: vk::PipelineLayout,
    stages: &[vk::PipelineShaderStageCreateInfo<'_>],
    groups: &[vk::RayTracingShaderGroupCreateInfoKHR<'_>],
) -> Result<vk::Pipeline> {
    let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(stages)
        .groups(groups)
        .max_pipeline_ray_recursion_depth(1)
        .layout(layout)
        .base_pipeline_index(-1);

    match unsafe {
        ray_tracing_pipeline_loader.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &[pipeline_info],
            None,
        )
    } {
        Ok(mut pipelines) => match pipelines.pop() {
            Some(pipeline) => Ok(pipeline),
            None => {
                unsafe { device.destroy_pipeline_layout(layout, None) };
                Err(anyhow!("Vulkan returned no ray tracing pipelines"))
            }
        },
        Err((pipelines, error)) => {
            unsafe {
                for pipeline in pipelines {
                    device.destroy_pipeline(pipeline, None);
                }
                device.destroy_pipeline_layout(layout, None);
            }
            Err(error).context("failed to create ray tracing pipeline")
        }
    }
}

fn align_up(value: vk::DeviceSize, alignment: vk::DeviceSize) -> Option<vk::DeviceSize> {
    if alignment == 0 {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn make_sbt_region(
    offset: &mut vk::DeviceSize,
    group_count: u32,
    stride: vk::DeviceSize,
    base_alignment: vk::DeviceSize,
) -> Result<ShaderBindingTableRegionLayout> {
    if group_count == 0 {
        return Ok(ShaderBindingTableRegionLayout {
            offset: *offset,
            stride: 0,
            size: 0,
        });
    }
    let aligned_offset = align_up(*offset, base_alignment)
        .ok_or_else(|| anyhow!("invalid SBT region base alignment {base_alignment}"))?;
    let size = stride
        .checked_mul(u64::from(group_count))
        .ok_or_else(|| anyhow!("SBT region size overflow"))?;
    *offset = aligned_offset
        .checked_add(size)
        .ok_or_else(|| anyhow!("SBT total size overflow"))?;
    Ok(ShaderBindingTableRegionLayout {
        offset: aligned_offset,
        stride,
        size,
    })
}

fn write_sbt_handles(
    buffer: &GpuBuffer,
    handles: &[u8],
    layout: &ShaderBindingTableLayout,
    group_counts: RayTracingShaderGroupCounts,
) -> Result<()> {
    let mapped = buffer
        .mapped_ptr()
        .ok_or_else(|| anyhow!("SBT buffer must be host visible"))?;
    let handle_size = layout.handle_size as usize;
    let mut group_index = 0usize;
    unsafe {
        write_sbt_region_handles(
            mapped,
            handles,
            handle_size,
            &layout.raygen,
            group_counts.raygen,
            &mut group_index,
        );
        write_sbt_region_handles(
            mapped,
            handles,
            handle_size,
            &layout.miss,
            group_counts.miss,
            &mut group_index,
        );
        write_sbt_region_handles(
            mapped,
            handles,
            handle_size,
            &layout.hit,
            group_counts.hit,
            &mut group_index,
        );
        write_sbt_region_handles(
            mapped,
            handles,
            handle_size,
            &layout.callable,
            group_counts.callable,
            &mut group_index,
        );
    }
    Ok(())
}

unsafe fn write_sbt_region_handles(
    mapped: *mut u8,
    handles: &[u8],
    handle_size: usize,
    region: &ShaderBindingTableRegionLayout,
    group_count: u32,
    group_index: &mut usize,
) {
    for local_group in 0..group_count as usize {
        let src_offset = (*group_index + local_group) * handle_size;
        let dst_offset = region.offset as usize + local_group * region.stride as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                handles.as_ptr().add(src_offset),
                mapped.add(dst_offset),
                handle_size,
            );
        }
    }
    *group_index += group_count as usize;
}

fn buffer_device_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    unsafe { device.get_buffer_device_address(&address_info) }
}

fn make_sbt_region_address(
    base_address: vk::DeviceAddress,
    layout: ShaderBindingTableRegionLayout,
) -> vk::StridedDeviceAddressRegionKHR {
    if layout.size == 0 {
        vk::StridedDeviceAddressRegionKHR::default()
    } else {
        vk::StridedDeviceAddressRegionKHR::default()
            .device_address(base_address + layout.offset)
            .stride(layout.stride)
            .size(layout.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn stage(stage: vk::ShaderStageFlags, handle: u64) -> RtShaderStageSpec<'static> {
        RtShaderStageSpec {
            stage,
            module: vk::ShaderModule::from_raw(handle),
            entry_point: c"main",
        }
    }

    #[test]
    fn shader_binding_table_layout_aligns_group_regions() {
        let layout = ShaderBindingTableLayout::new(
            32,
            32,
            64,
            RayTracingShaderGroupCounts {
                raygen: 1,
                miss: 2,
                hit: 3,
                callable: 0,
            },
        )
        .expect("valid SBT alignment should produce a layout");

        assert_eq!(layout.handle_size, 32);
        assert_eq!(layout.handle_stride, 32);
        assert_eq!(layout.raygen.offset, 0);
        assert_eq!(layout.raygen.size, 32);
        assert_eq!(layout.raygen.stride, 32);
        assert_eq!(layout.miss.offset, 64);
        assert_eq!(layout.miss.size, 64);
        assert_eq!(layout.hit.offset, 128);
        assert_eq!(layout.hit.size, 96);
        assert_eq!(layout.callable.size, 0);
        assert_eq!(layout.total_size, 224);
    }

    #[test]
    fn shader_binding_table_layout_rejects_missing_raygen_group() {
        let error = ShaderBindingTableLayout::new(
            32,
            32,
            64,
            RayTracingShaderGroupCounts {
                raygen: 0,
                miss: 1,
                hit: 0,
                callable: 0,
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("raygen"));
    }

    #[test]
    fn ray_tracing_pipeline_source_builds_miss_and_procedural_hit_groups() {
        let source = crate::render::source_checks::read_source("src/render/pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("pipeline implementation should precede tests");

        for token in [
            "new_surface_pipeline",
            "ShaderStageFlags::MISS_KHR",
            "ShaderStageFlags::CLOSEST_HIT_KHR",
            "ShaderStageFlags::INTERSECTION_KHR",
            "RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP",
            "closest_hit_stage: 2",
            "intersection_stage: 3",
            "new_mixed_surface_pipeline",
        ] {
            assert!(
                implementation.contains(token),
                "RT surface pipeline group setup missing {token}"
            );
        }
    }

    #[test]
    fn mixed_surface_pipeline_keeps_reference_then_compact_exact_sbt_order() {
        let stages = [
            stage(vk::ShaderStageFlags::RAYGEN_KHR, 1),
            stage(vk::ShaderStageFlags::MISS_KHR, 2),
            stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, 3),
            stage(vk::ShaderStageFlags::INTERSECTION_KHR, 4),
            stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, 5),
        ];
        let plan = RtMixedSurfaceGroupPlan::new(
            &stages,
            &[
                RtHitGroupKind::Procedural {
                    closest_hit_stage: 2,
                    intersection_stage: 3,
                },
                RtHitGroupKind::Triangles {
                    closest_hit_stage: 4,
                },
            ],
        )
        .unwrap();

        assert_eq!(plan.group_counts.hit, 2);
        assert_eq!(
            plan.groups[2].ty,
            vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
        );
        assert_eq!(plan.groups[2].closest_hit_shader, 2);
        assert_eq!(plan.groups[2].intersection_shader, 3);
        assert_eq!(
            plan.groups[3].ty,
            vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
        );
        assert_eq!(plan.groups[3].closest_hit_shader, 4);
        assert_eq!(plan.groups[3].intersection_shader, vk::SHADER_UNUSED_KHR);
        assert!(
            plan.groups
                .iter()
                .all(|group| group.any_hit_shader == vk::SHADER_UNUSED_KHR)
        );
    }

    #[test]
    fn mixed_surface_pipeline_rejects_out_of_range_or_wrong_kind_stage_indices() {
        let stages = [
            stage(vk::ShaderStageFlags::RAYGEN_KHR, 1),
            stage(vk::ShaderStageFlags::MISS_KHR, 2),
            stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, 3),
            stage(vk::ShaderStageFlags::INTERSECTION_KHR, 4),
        ];

        assert!(
            RtMixedSurfaceGroupPlan::new(
                &stages,
                &[
                    RtHitGroupKind::Procedural {
                        closest_hit_stage: 2,
                        intersection_stage: 3,
                    },
                    RtHitGroupKind::Triangles {
                        closest_hit_stage: 9,
                    },
                ],
            )
            .unwrap_err()
            .to_string()
            .contains("out of range")
        );
        assert!(
            RtMixedSurfaceGroupPlan::new(
                &stages,
                &[
                    RtHitGroupKind::Procedural {
                        closest_hit_stage: 2,
                        intersection_stage: 1,
                    },
                    RtHitGroupKind::Triangles {
                        closest_hit_stage: 2,
                    },
                ],
            )
            .unwrap_err()
            .to_string()
            .contains("INTERSECTION")
        );
    }

    #[test]
    fn mixed_surface_pipeline_rejects_any_hit_stages() {
        let stages = [
            stage(vk::ShaderStageFlags::RAYGEN_KHR, 1),
            stage(vk::ShaderStageFlags::MISS_KHR, 2),
            stage(vk::ShaderStageFlags::ANY_HIT_KHR, 3),
        ];

        assert!(
            RtMixedSurfaceGroupPlan::new(&stages, &[])
                .unwrap_err()
                .to_string()
                .contains("any-hit")
        );
    }

    #[test]
    fn mixed_surface_pipeline_rejects_reversed_page_hit_group_semantics() {
        let stages = [
            stage(vk::ShaderStageFlags::RAYGEN_KHR, 1),
            stage(vk::ShaderStageFlags::MISS_KHR, 2),
            stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, 3),
            stage(vk::ShaderStageFlags::INTERSECTION_KHR, 4),
        ];

        assert!(
            RtMixedSurfaceGroupPlan::new(
                &stages,
                &[
                    RtHitGroupKind::Triangles {
                        closest_hit_stage: 2,
                    },
                    RtHitGroupKind::Procedural {
                        closest_hit_stage: 2,
                        intersection_stage: 3,
                    },
                ],
            )
            .unwrap_err()
            .to_string()
            .contains("Reference procedural then CompactExact triangles")
        );
    }
}
