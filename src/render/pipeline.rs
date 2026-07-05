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

impl RayTracingPipeline {
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
        let layout = create_ray_tracing_pipeline_layout(
            device,
            descriptor_set_layouts,
            push_constant_ranges,
        )?;

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(raygen_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(miss_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(closest_hit_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
                .module(intersection_module)
                .name(entry_point),
        ];
        let groups = [
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(3),
        ];

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
                miss: 1,
                hit: 1,
                callable: 0,
            },
        })
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
            "closest_hit_shader(2)",
            "intersection_shader(3)",
            "miss: 1",
            "hit: 1",
        ] {
            assert!(
                implementation.contains(token),
                "RT surface pipeline group setup missing {token}"
            );
        }
    }
}
