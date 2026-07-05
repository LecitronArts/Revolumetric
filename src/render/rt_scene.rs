use anyhow::{Context, Result, anyhow};
use ash::vk;
use glam::{UVec3, Vec3};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::voxel::ucvh::{Ucvh, UcvhInvalidationRegion};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtBrickBounds {
    pub brick_coord: UVec3,
    pub brick_id: u32,
    pub min: Vec3,
    pub max: Vec3,
    pub generation: u32,
}

impl RtBrickBounds {
    fn acceleration_geometry_eq(&self, other: &Self) -> bool {
        self.brick_coord == other.brick_coord
            && self.brick_id == other.brick_id
            && self.min == other.min
            && self.max == other.max
    }
}

#[derive(Default)]
pub struct RtSceneBackend {
    pub build_generation: u32,
    pub dirty_generation: u32,
    pub brick_bounds: Vec<RtBrickBounds>,
    pub last_rebuild_sampled_bricks: u32,
    bounds_initialized: bool,
    gpu_resources: Option<RtSceneGpuBuildResources>,
}

#[derive(Debug, Clone)]
pub struct RtSceneAsBuildInputs {
    pub aabbs: Vec<vk::AabbPositionsKHR>,
}

impl RtSceneAsBuildInputs {
    pub fn from_brick_bounds(bounds: &[RtBrickBounds]) -> Self {
        let aabbs = bounds
            .iter()
            .map(|bounds| {
                vk::AabbPositionsKHR::default()
                    .min_x(bounds.min.x)
                    .min_y(bounds.min.y)
                    .min_z(bounds.min.z)
                    .max_x(bounds.max.x)
                    .max_y(bounds.max.y)
                    .max_z(bounds.max.z)
            })
            .collect();
        Self { aabbs }
    }

    pub fn blas_primitive_count(&self) -> u32 {
        self.aabbs.len().min(u32::MAX as usize) as u32
    }

    pub fn tlas_instance_count(&self) -> u32 {
        u32::from(!self.aabbs.is_empty())
    }

    pub fn can_update_in_place(&self, next: &Self) -> bool {
        !self.aabbs.is_empty()
            && !next.aabbs.is_empty()
            && self.blas_primitive_count() == next.blas_primitive_count()
            && self.tlas_instance_count() == next.tlas_instance_count()
    }

    pub fn blas_geometry(
        &self,
        aabb_buffer_address: vk::DeviceAddress,
    ) -> vk::AccelerationStructureGeometryKHR<'static> {
        let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: aabb_buffer_address,
            })
            .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as vk::DeviceSize);
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs })
            .flags(vk::GeometryFlagsKHR::OPAQUE)
    }

    pub fn blas_build_range(&self) -> vk::AccelerationStructureBuildRangeInfoKHR {
        vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(self.blas_primitive_count())
    }

    pub fn tlas_geometry(
        &self,
        instance_buffer_address: vk::DeviceAddress,
    ) -> vk::AccelerationStructureGeometryKHR<'static> {
        let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer_address,
            });
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances })
    }

    pub fn tlas_build_range(&self) -> vk::AccelerationStructureBuildRangeInfoKHR {
        vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(self.tlas_instance_count())
    }

    pub fn aabb_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
    }

    pub fn instance_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    }

    pub fn as_storage_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    }

    pub fn scratch_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    }
}

pub struct RtAccelerationStructure {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: GpuBuffer,
    pub ty: vk::AccelerationStructureTypeKHR,
    pub size: vk::DeviceSize,
}

impl RtAccelerationStructure {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        ty: vk::AccelerationStructureTypeKHR,
        size: vk::DeviceSize,
        name: &str,
    ) -> Result<Self> {
        if size == 0 {
            return Err(anyhow!(
                "acceleration structure storage size must be non-zero"
            ));
        }
        let buffer = GpuBuffer::new(
            device,
            allocator,
            size,
            RtSceneAsBuildInputs::as_storage_buffer_usage(),
            MemoryLocation::GpuOnly,
            name,
        )?;
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(ty)
            .buffer(buffer.handle)
            .size(size);
        let handle = match unsafe {
            acceleration_structure_loader.create_acceleration_structure(&create_info, None)
        } {
            Ok(handle) => handle,
            Err(error) => {
                buffer.destroy(device, allocator);
                return Err(error).context("failed to create acceleration structure");
            }
        };

        Ok(Self {
            handle,
            buffer,
            ty,
            size,
        })
    }

    pub fn device_address(
        &self,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    ) -> vk::DeviceAddress {
        let address_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
            .acceleration_structure(self.handle);
        unsafe {
            acceleration_structure_loader.get_acceleration_structure_device_address(&address_info)
        }
    }

    pub fn destroy(
        self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    ) {
        unsafe { acceleration_structure_loader.destroy_acceleration_structure(self.handle, None) };
        self.buffer.destroy(device, allocator);
    }
}

pub struct RtSceneAsBuildPlan {
    blas_geometries: [vk::AccelerationStructureGeometryKHR<'static>; 1],
    tlas_geometries: [vk::AccelerationStructureGeometryKHR<'static>; 1],
    blas_range: [vk::AccelerationStructureBuildRangeInfoKHR; 1],
    tlas_range: [vk::AccelerationStructureBuildRangeInfoKHR; 1],
    blas: vk::AccelerationStructureKHR,
    tlas: vk::AccelerationStructureKHR,
    scratch_address: vk::DeviceAddress,
}

impl RtSceneAsBuildPlan {
    pub fn new(
        inputs: &RtSceneAsBuildInputs,
        aabb_buffer_address: vk::DeviceAddress,
        instance_buffer_address: vk::DeviceAddress,
        blas: vk::AccelerationStructureKHR,
        tlas: vk::AccelerationStructureKHR,
        scratch_address: vk::DeviceAddress,
    ) -> Self {
        Self {
            blas_geometries: [inputs.blas_geometry(aabb_buffer_address)],
            tlas_geometries: [inputs.tlas_geometry(instance_buffer_address)],
            blas_range: [inputs.blas_build_range()],
            tlas_range: [inputs.tlas_build_range()],
            blas,
            tlas,
            scratch_address,
        }
    }

    pub fn blas_build_info(&self) -> vk::AccelerationStructureBuildGeometryInfoKHR<'_> {
        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(acceleration_structure_build_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.blas)
            .geometries(&self.blas_geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.scratch_address,
            })
    }

    pub fn tlas_build_info(&self) -> vk::AccelerationStructureBuildGeometryInfoKHR<'_> {
        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(acceleration_structure_build_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.tlas)
            .geometries(&self.tlas_geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.scratch_address,
            })
    }

    pub fn blas_range(&self) -> &[vk::AccelerationStructureBuildRangeInfoKHR] {
        &self.blas_range
    }

    pub fn tlas_range(&self) -> &[vk::AccelerationStructureBuildRangeInfoKHR] {
        &self.tlas_range
    }
}

pub struct RtSceneGpuBuildResources {
    pub aabb_buffer: GpuBuffer,
    pub instance_buffer: GpuBuffer,
    pub scratch_buffer: GpuBuffer,
    pub blas: RtAccelerationStructure,
    pub tlas: RtAccelerationStructure,
    pub inputs: RtSceneAsBuildInputs,
    pub blas_build_sizes: vk::AccelerationStructureBuildSizesInfoKHR<'static>,
    pub tlas_build_sizes: vk::AccelerationStructureBuildSizesInfoKHR<'static>,
    scratch_alignment: vk::DeviceSize,
}

impl RtSceneGpuBuildResources {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        inputs: RtSceneAsBuildInputs,
        scratch_alignment: vk::DeviceSize,
    ) -> Result<Self> {
        if inputs.aabbs.is_empty() {
            return Err(anyhow!(
                "RT scene acceleration structure build requires at least one occupied brick"
            ));
        }

        let aabb_buffer = GpuBuffer::new(
            device,
            allocator,
            slice_size(&inputs.aabbs),
            RtSceneAsBuildInputs::aabb_buffer_usage(),
            MemoryLocation::CpuToGpu,
            "rt_scene_aabbs",
        )?;
        if let Err(error) = write_mapped_slice(&aabb_buffer, &inputs.aabbs, "RT scene AABB") {
            aabb_buffer.destroy(device, allocator);
            return Err(error);
        }
        let aabb_buffer_address = buffer_device_address(device, aabb_buffer.handle);

        let blas_build_sizes =
            query_blas_build_sizes(acceleration_structure_loader, &inputs, aabb_buffer_address);
        let blas = match RtAccelerationStructure::new(
            device,
            allocator,
            acceleration_structure_loader,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            blas_build_sizes.acceleration_structure_size,
            "rt_scene_blas",
        ) {
            Ok(blas) => blas,
            Err(error) => {
                aabb_buffer.destroy(device, allocator);
                return Err(error);
            }
        };

        let blas_device_address = blas.device_address(acceleration_structure_loader);
        let tlas_instance = make_tlas_instance(blas_device_address);
        let instance_buffer = match GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as vk::DeviceSize,
            RtSceneAsBuildInputs::instance_buffer_usage(),
            MemoryLocation::CpuToGpu,
            "rt_scene_tlas_instance",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                blas.destroy(device, allocator, acceleration_structure_loader);
                aabb_buffer.destroy(device, allocator);
                return Err(error);
            }
        };
        if let Err(error) =
            write_mapped_value(&instance_buffer, &tlas_instance, "RT scene TLAS instance")
        {
            instance_buffer.destroy(device, allocator);
            blas.destroy(device, allocator, acceleration_structure_loader);
            aabb_buffer.destroy(device, allocator);
            return Err(error);
        }
        let instance_buffer_address = buffer_device_address(device, instance_buffer.handle);

        let tlas_build_sizes = query_tlas_build_sizes(
            acceleration_structure_loader,
            &inputs,
            instance_buffer_address,
        );
        let tlas = match RtAccelerationStructure::new(
            device,
            allocator,
            acceleration_structure_loader,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            tlas_build_sizes.acceleration_structure_size,
            "rt_scene_tlas",
        ) {
            Ok(tlas) => tlas,
            Err(error) => {
                instance_buffer.destroy(device, allocator);
                blas.destroy(device, allocator, acceleration_structure_loader);
                aabb_buffer.destroy(device, allocator);
                return Err(error);
            }
        };

        let scratch_size = blas_build_sizes
            .build_scratch_size
            .max(blas_build_sizes.update_scratch_size)
            .max(tlas_build_sizes.build_scratch_size)
            .max(tlas_build_sizes.update_scratch_size);
        let scratch_buffer_size = scratch_buffer_allocation_size(scratch_size, scratch_alignment);
        let scratch_buffer = match GpuBuffer::new(
            device,
            allocator,
            scratch_buffer_size,
            RtSceneAsBuildInputs::scratch_buffer_usage(),
            MemoryLocation::GpuOnly,
            "rt_scene_as_scratch",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                tlas.destroy(device, allocator, acceleration_structure_loader);
                instance_buffer.destroy(device, allocator);
                blas.destroy(device, allocator, acceleration_structure_loader);
                aabb_buffer.destroy(device, allocator);
                return Err(error);
            }
        };

        Ok(Self {
            aabb_buffer,
            instance_buffer,
            scratch_buffer,
            blas,
            tlas,
            inputs,
            blas_build_sizes,
            tlas_build_sizes,
            scratch_alignment,
        })
    }

    pub fn can_update_in_place(&self, inputs: &RtSceneAsBuildInputs) -> bool {
        self.inputs.can_update_in_place(inputs)
    }

    pub fn update_inputs(&mut self, inputs: RtSceneAsBuildInputs) -> Result<()> {
        if !self.can_update_in_place(&inputs) {
            return Err(anyhow!(
                "RT scene AS update requires matching primitive counts: previous_blas={} next_blas={} previous_tlas={} next_tlas={}",
                self.inputs.blas_primitive_count(),
                inputs.blas_primitive_count(),
                self.inputs.tlas_instance_count(),
                inputs.tlas_instance_count()
            ));
        }
        write_mapped_slice(&self.aabb_buffer, &inputs.aabbs, "RT scene AABB update")?;
        self.inputs = inputs;
        Ok(())
    }

    pub fn record_build(
        &self,
        device: &ash::Device,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        command_buffer: vk::CommandBuffer,
    ) {
        let scratch_address = self.scratch_device_address(device);
        let plan = RtSceneAsBuildPlan::new(
            &self.inputs,
            buffer_device_address(device, self.aabb_buffer.handle),
            buffer_device_address(device, self.instance_buffer.handle),
            self.blas.handle,
            self.tlas.handle,
            scratch_address,
        );
        let blas_build_info = plan.blas_build_info();
        let blas_build_range = plan.blas_range();
        unsafe {
            acceleration_structure_loader.cmd_build_acceleration_structures(
                command_buffer,
                std::slice::from_ref(&blas_build_info),
                &[blas_build_range],
            );
        }

        let blas_to_tlas_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            );
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&blas_to_tlas_barrier),
                &[],
                &[],
            );
        }

        let tlas_build_info = plan.tlas_build_info();
        let tlas_build_range = plan.tlas_range();
        unsafe {
            acceleration_structure_loader.cmd_build_acceleration_structures(
                command_buffer,
                std::slice::from_ref(&tlas_build_info),
                &[tlas_build_range],
            );
        }

        let tlas_to_trace_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&tlas_to_trace_barrier),
                &[],
                &[],
            );
        }
    }

    pub fn record_update(
        &self,
        device: &ash::Device,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        command_buffer: vk::CommandBuffer,
    ) {
        let scratch_address = self.scratch_device_address(device);
        let plan = RtSceneAsBuildPlan::new(
            &self.inputs,
            buffer_device_address(device, self.aabb_buffer.handle),
            buffer_device_address(device, self.instance_buffer.handle),
            self.blas.handle,
            self.tlas.handle,
            scratch_address,
        );
        let blas_update_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(acceleration_structure_build_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .src_acceleration_structure(self.blas.handle)
            .dst_acceleration_structure(self.blas.handle)
            .geometries(&plan.blas_geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            });
        let blas_build_range = plan.blas_range();
        unsafe {
            acceleration_structure_loader.cmd_build_acceleration_structures(
                command_buffer,
                std::slice::from_ref(&blas_update_info),
                &[blas_build_range],
            );
        }

        let blas_to_tlas_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            );
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&blas_to_tlas_barrier),
                &[],
                &[],
            );
        }

        let tlas_update_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(acceleration_structure_build_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .src_acceleration_structure(self.tlas.handle)
            .dst_acceleration_structure(self.tlas.handle)
            .geometries(&plan.tlas_geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            });
        let tlas_build_range = plan.tlas_range();
        unsafe {
            acceleration_structure_loader.cmd_build_acceleration_structures(
                command_buffer,
                std::slice::from_ref(&tlas_update_info),
                &[tlas_build_range],
            );
        }

        let tlas_to_trace_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&tlas_to_trace_barrier),
                &[],
                &[],
            );
        }
    }

    pub fn destroy(
        self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    ) {
        self.tlas
            .destroy(device, allocator, acceleration_structure_loader);
        self.blas
            .destroy(device, allocator, acceleration_structure_loader);
        self.scratch_buffer.destroy(device, allocator);
        self.instance_buffer.destroy(device, allocator);
        self.aabb_buffer.destroy(device, allocator);
    }

    fn scratch_device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        align_device_address(
            buffer_device_address(device, self.scratch_buffer.handle),
            self.scratch_alignment,
        )
    }
}

pub fn collect_occupied_brick_bounds(ucvh: &Ucvh) -> Vec<RtBrickBounds> {
    collect_occupied_brick_bounds_with_sample_count(ucvh).0
}

fn collect_occupied_brick_bounds_with_sample_count(ucvh: &Ucvh) -> (Vec<RtBrickBounds>, u32) {
    let mut bounds = Vec::new();
    let mut sampled_bricks = 0u32;
    let grid = ucvh.config.brick_grid_size;

    for z in 0..grid.z {
        for y in 0..grid.y {
            for x in 0..grid.x {
                sampled_bricks = sampled_bricks.saturating_add(1);
                let brick_coord = UVec3::new(x, y, z);
                if let Some(bound) = collect_brick_bound(ucvh, brick_coord) {
                    bounds.push(bound);
                };
            }
        }
    }

    (bounds, sampled_bricks)
}

fn collect_occupied_brick_bounds_in_regions(
    ucvh: &Ucvh,
    regions: &[UcvhInvalidationRegion],
) -> (Vec<RtBrickBounds>, Vec<UVec3>, u32) {
    let mut bounds = Vec::new();
    let mut sampled_coords = Vec::new();
    let grid = ucvh.config.brick_grid_size;

    for region in regions {
        let min = region.brick_min.min(grid);
        let max = region.brick_max_exclusive.min(grid);
        for z in min.z..max.z {
            for y in min.y..max.y {
                for x in min.x..max.x {
                    let brick_coord = UVec3::new(x, y, z);
                    if sampled_coords.contains(&brick_coord) {
                        continue;
                    }
                    sampled_coords.push(brick_coord);
                    if let Some(bound) = collect_brick_bound(ucvh, brick_coord) {
                        bounds.push(bound);
                    }
                }
            }
        }
    }

    let sampled_bricks = sampled_coords.len().min(u32::MAX as usize) as u32;
    (bounds, sampled_coords, sampled_bricks)
}

fn collect_brick_bound(ucvh: &Ucvh, brick_coord: UVec3) -> Option<RtBrickBounds> {
    let brick_id = ucvh.brick_id_at(brick_coord)?;
    if !brick_contains_solid_voxel(ucvh, brick_coord) {
        return None;
    }

    Some(RtBrickBounds {
        brick_coord,
        brick_id,
        min: brick_coord.as_vec3() * 8.0,
        max: (brick_coord + UVec3::ONE).as_vec3() * 8.0,
        generation: ucvh.brick_generation(brick_id).unwrap_or_default(),
    })
}

fn brick_contains_solid_voxel(ucvh: &Ucvh, brick_coord: UVec3) -> bool {
    let base = brick_coord * 8u32;
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                if !ucvh.get_voxel(base + UVec3::new(x, y, z)).is_air() {
                    return true;
                }
            }
        }
    }
    false
}

impl RtSceneBackend {
    pub fn rebuild(&mut self, ucvh: &Ucvh) -> bool {
        let invalidation_regions = ucvh.invalidation_regions();
        let pending_invalidation_regions = invalidation_regions
            .iter()
            .copied()
            .filter(|region| !self.bounds_initialized || region.generation > self.dirty_generation)
            .collect::<Vec<_>>();
        let dirty_generation = invalidation_regions
            .last()
            .map(|region| region.generation)
            .unwrap_or(self.dirty_generation);
        let (brick_bounds, sampled_bricks) = if !self.bounds_initialized {
            collect_occupied_brick_bounds_with_sample_count(ucvh)
        } else if pending_invalidation_regions.is_empty() {
            (self.brick_bounds.clone(), 0)
        } else {
            let (dirty_bounds, sampled_coords, sampled_bricks) =
                collect_occupied_brick_bounds_in_regions(ucvh, &pending_invalidation_regions);
            (
                apply_dirty_brick_bounds(&self.brick_bounds, &sampled_coords, dirty_bounds),
                sampled_bricks,
            )
        };
        self.last_rebuild_sampled_bricks = sampled_bricks;
        if self.dirty_generation == dirty_generation && self.brick_bounds == brick_bounds {
            self.bounds_initialized = true;
            return false;
        }
        if brick_bounds_acceleration_geometry_eq(&self.brick_bounds, &brick_bounds) {
            self.dirty_generation = dirty_generation;
            self.brick_bounds = brick_bounds;
            self.bounds_initialized = true;
            return false;
        }

        self.build_generation = self.build_generation.wrapping_add(1);
        self.dirty_generation = dirty_generation;
        self.brick_bounds = brick_bounds;
        self.bounds_initialized = true;
        true
    }

    pub fn rebuild_gpu(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        command_buffer: vk::CommandBuffer,
        scratch_alignment: vk::DeviceSize,
        ucvh: &Ucvh,
    ) -> Result<()> {
        let scene_changed = self.rebuild(ucvh);
        if !scene_changed && self.gpu_resources.is_some() {
            return Ok(());
        }

        let inputs = RtSceneAsBuildInputs::from_brick_bounds(&self.brick_bounds);
        if inputs.aabbs.is_empty() {
            self.clear_gpu_resources(device, allocator, acceleration_structure_loader);
            return Ok(());
        }

        if let Some(resources) = self.gpu_resources.as_mut() {
            if resources.can_update_in_place(&inputs) {
                resources
                    .update_inputs(inputs)
                    .context("failed to update RT scene AABB buffer")?;
                resources.record_update(device, acceleration_structure_loader, command_buffer);
                return Ok(());
            }
        }

        let new_resources = RtSceneGpuBuildResources::new(
            device,
            allocator,
            acceleration_structure_loader,
            inputs,
            scratch_alignment,
        )
        .context("failed to create RT scene GPU acceleration structure resources")?;
        new_resources.record_build(device, acceleration_structure_loader, command_buffer);
        self.clear_gpu_resources(device, allocator, acceleration_structure_loader);
        self.gpu_resources = Some(new_resources);
        Ok(())
    }

    pub fn tlas_handle(&self) -> Option<vk::AccelerationStructureKHR> {
        self.gpu_resources
            .as_ref()
            .map(|resources| resources.tlas.handle)
    }

    pub fn aabb_buffer(&self) -> Option<&GpuBuffer> {
        self.gpu_resources
            .as_ref()
            .map(|resources| &resources.aabb_buffer)
    }

    pub fn tlas_device_address(
        &self,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    ) -> Option<vk::DeviceAddress> {
        self.gpu_resources
            .as_ref()
            .map(|resources| resources.tlas.device_address(acceleration_structure_loader))
    }

    pub fn destroy(
        self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: Option<&ash::khr::acceleration_structure::Device>,
    ) {
        if let (Some(resources), Some(acceleration_structure_loader)) =
            (self.gpu_resources, acceleration_structure_loader)
        {
            resources.destroy(device, allocator, acceleration_structure_loader);
        }
    }

    fn clear_gpu_resources(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    ) {
        if let Some(resources) = self.gpu_resources.take() {
            resources.destroy(device, allocator, acceleration_structure_loader);
        }
    }
}

fn acceleration_structure_build_flags() -> vk::BuildAccelerationStructureFlagsKHR {
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
}

fn apply_dirty_brick_bounds(
    current: &[RtBrickBounds],
    dirty_coords: &[UVec3],
    dirty_bounds: Vec<RtBrickBounds>,
) -> Vec<RtBrickBounds> {
    let mut merged = current
        .iter()
        .copied()
        .filter(|bounds| !dirty_coords.contains(&bounds.brick_coord))
        .collect::<Vec<_>>();
    merged.extend(dirty_bounds);
    merged.sort_by_key(|bounds| {
        (
            bounds.brick_coord.z,
            bounds.brick_coord.y,
            bounds.brick_coord.x,
        )
    });
    merged
}

fn brick_bounds_acceleration_geometry_eq(left: &[RtBrickBounds], right: &[RtBrickBounds]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.acceleration_geometry_eq(right))
}

fn query_blas_build_sizes(
    acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    inputs: &RtSceneAsBuildInputs,
    aabb_buffer_address: vk::DeviceAddress,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let blas_geometries = [inputs.blas_geometry(aabb_buffer_address)];
    let blas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(acceleration_structure_build_flags())
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&blas_geometries);
    let max_primitive_counts = [inputs.blas_primitive_count()];
    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        acceleration_structure_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &blas_build_info,
            &max_primitive_counts,
            &mut size_info,
        );
    }
    size_info
}

fn query_tlas_build_sizes(
    acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    inputs: &RtSceneAsBuildInputs,
    instance_buffer_address: vk::DeviceAddress,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let tlas_geometries = [inputs.tlas_geometry(instance_buffer_address)];
    let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(acceleration_structure_build_flags())
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&tlas_geometries);
    let max_primitive_counts = [inputs.tlas_instance_count()];
    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        acceleration_structure_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &tlas_build_info,
            &max_primitive_counts,
            &mut size_info,
        );
    }
    size_info
}

fn make_tlas_instance(
    blas_device_address: vk::DeviceAddress,
) -> vk::AccelerationStructureInstanceKHR {
    vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            0,
            vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas_device_address,
        },
    }
}

fn slice_size<T>(values: &[T]) -> vk::DeviceSize {
    std::mem::size_of_val(values) as vk::DeviceSize
}

fn scratch_buffer_allocation_size(
    scratch_size: vk::DeviceSize,
    scratch_alignment: vk::DeviceSize,
) -> vk::DeviceSize {
    scratch_size.saturating_add(scratch_alignment.saturating_sub(1))
}

fn align_device_address(
    address: vk::DeviceAddress,
    alignment: vk::DeviceSize,
) -> vk::DeviceAddress {
    if alignment <= 1 {
        return address;
    }
    let remainder = address % alignment;
    if remainder == 0 {
        address
    } else {
        address + (alignment - remainder)
    }
}

fn write_mapped_value<T: Copy>(buffer: &GpuBuffer, value: &T, label: &str) -> Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
    };
    write_mapped_bytes(buffer, bytes, label)
}

fn write_mapped_slice<T: Copy>(buffer: &GpuBuffer, values: &[T], label: &str) -> Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    write_mapped_bytes(buffer, bytes, label)
}

fn write_mapped_bytes(buffer: &GpuBuffer, bytes: &[u8], label: &str) -> Result<()> {
    if bytes.len() as vk::DeviceSize > buffer.size {
        return Err(anyhow!(
            "{label} upload exceeds mapped buffer: bytes={} buffer_size={}",
            bytes.len(),
            buffer.size
        ));
    }
    let mapped = buffer
        .mapped_ptr()
        .ok_or_else(|| anyhow!("{label} buffer must be host visible"))?;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped, bytes.len()) };
    Ok(())
}

fn buffer_device_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    unsafe { device.get_buffer_device_address(&address_info) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::brick::VoxelCell;
    use crate::voxel::ucvh::UcvhConfig;

    #[test]
    fn collect_occupied_brick_bounds_deduplicates_bricks() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        assert!(ucvh.set_voxel(UVec3::new(1, 2, 3), VoxelCell::new(1, 0, [0; 3])));
        assert!(ucvh.set_voxel(UVec3::new(9, 2, 3), VoxelCell::new(1, 0, [0; 3])));
        ucvh.rebuild_hierarchy();

        let bounds = collect_occupied_brick_bounds(&ucvh);

        assert_eq!(bounds.len(), 2);
        assert_eq!(bounds[0].brick_coord, UVec3::new(0, 0, 0));
        assert_eq!(bounds[1].brick_coord, UVec3::new(1, 0, 0));
        assert_eq!(bounds[0].min, Vec3::ZERO);
        assert_eq!(bounds[0].max, Vec3::splat(8.0));
    }

    #[test]
    fn scene_backend_rebuild_tracks_generation_and_bounds() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));
        assert!(ucvh.set_voxel(UVec3::new(0, 0, 0), VoxelCell::new(2, 0, [0; 3])));
        ucvh.rebuild_hierarchy();

        let mut backend = RtSceneBackend::default();
        backend.rebuild(&ucvh);

        assert_eq!(backend.build_generation, 1);
        assert_eq!(backend.brick_bounds.len(), 1);
        assert_eq!(backend.brick_bounds[0].brick_id, 0);
    }

    #[test]
    fn scene_backend_rebuild_skips_unchanged_brick_bounds() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));
        assert!(ucvh.set_voxel(UVec3::new(0, 0, 0), VoxelCell::new(2, 0, [0; 3])));
        ucvh.rebuild_hierarchy();

        let mut backend = RtSceneBackend::default();
        backend.rebuild(&ucvh);
        backend.rebuild(&ucvh);

        assert_eq!(
            backend.build_generation, 1,
            "unchanged RT scene bounds must not force per-frame AS rebuilds"
        );
    }

    #[test]
    fn scene_backend_rebuild_ignores_generation_only_dirty_region_when_aabb_is_stable() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));
        assert!(ucvh.set_voxel(UVec3::new(0, 0, 0), VoxelCell::new(2, 0, [0; 3])));
        ucvh.rebuild_hierarchy();

        let mut backend = RtSceneBackend::default();
        assert!(backend.rebuild(&ucvh));
        assert_eq!(backend.build_generation, 1);

        assert!(ucvh.set_voxel(UVec3::new(1, 0, 0), VoxelCell::new(3, 0, [0; 3])));
        ucvh.rebuild_hierarchy();

        assert!(
            !backend.rebuild(&ucvh),
            "dirty edits inside an already occupied brick must not rebuild AS when the brick AABB is unchanged"
        );
        assert_eq!(backend.build_generation, 1);
        assert_eq!(backend.brick_bounds.len(), 1);
        assert_eq!(backend.brick_bounds[0].brick_coord, UVec3::ZERO);
    }

    #[test]
    fn scene_backend_rebuild_resamples_only_dirty_brick_region_after_initial_build() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        assert!(ucvh.set_voxel(UVec3::new(0, 0, 0), VoxelCell::new(2, 0, [0; 3])));
        assert!(ucvh.set_voxel(UVec3::new(24, 24, 24), VoxelCell::new(4, 0, [0; 3])));
        ucvh.rebuild_hierarchy();

        let mut backend = RtSceneBackend::default();
        assert!(backend.rebuild(&ucvh));
        assert_eq!(backend.last_rebuild_sampled_bricks, 64);

        assert!(ucvh.set_voxel(UVec3::new(9, 0, 0), VoxelCell::new(3, 0, [0; 3])));
        ucvh.rebuild_hierarchy();

        assert!(backend.rebuild(&ucvh));
        assert_eq!(
            backend.last_rebuild_sampled_bricks, 1,
            "incremental RT scene rebuild must resample only dirty brick-space regions after the initial full build"
        );
        assert_eq!(backend.brick_bounds.len(), 3);
        assert!(
            backend
                .brick_bounds
                .iter()
                .any(|bounds| bounds.brick_coord == UVec3::new(1, 0, 0))
        );
    }

    #[test]
    fn scene_backend_rebuild_removes_emptied_dirty_brick_from_cached_aabbs() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));
        assert!(ucvh.set_voxel(UVec3::ZERO, VoxelCell::new(2, 0, [0; 3])));
        ucvh.rebuild_hierarchy();

        let mut backend = RtSceneBackend::default();
        assert!(backend.rebuild(&ucvh));
        assert_eq!(backend.brick_bounds.len(), 1);

        assert!(ucvh.set_voxel(UVec3::ZERO, VoxelCell::AIR));
        ucvh.rebuild_hierarchy();

        assert!(backend.rebuild(&ucvh));
        assert_eq!(backend.last_rebuild_sampled_bricks, 1);
        assert!(
            backend.brick_bounds.is_empty(),
            "dirty brick that becomes empty must be removed from cached RT AABBs"
        );
    }

    #[test]
    fn rt_scene_build_inputs_pack_brick_bounds_as_procedural_aabbs() {
        let bounds = [RtBrickBounds {
            brick_coord: UVec3::new(2, 3, 4),
            brick_id: 9,
            min: Vec3::new(16.0, 24.0, 32.0),
            max: Vec3::new(24.0, 32.0, 40.0),
            generation: 7,
        }];

        let inputs = RtSceneAsBuildInputs::from_brick_bounds(&bounds);

        assert_eq!(inputs.aabbs.len(), 1);
        assert_eq!(inputs.aabbs[0].min_x, 16.0);
        assert_eq!(inputs.aabbs[0].min_y, 24.0);
        assert_eq!(inputs.aabbs[0].min_z, 32.0);
        assert_eq!(inputs.aabbs[0].max_x, 24.0);
        assert_eq!(inputs.aabbs[0].max_y, 32.0);
        assert_eq!(inputs.aabbs[0].max_z, 40.0);
        assert_eq!(inputs.blas_primitive_count(), 1);
        assert_eq!(inputs.tlas_instance_count(), 1);
        assert!(RtSceneAsBuildInputs::aabb_buffer_usage().contains(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
        ));
        assert!(RtSceneAsBuildInputs::as_storage_buffer_usage().contains(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        ));
        assert!(RtSceneAsBuildInputs::scratch_buffer_usage().contains(
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        ));
    }

    #[test]
    fn rt_scene_build_inputs_create_blas_and_tlas_geometry_descriptors() {
        let inputs = RtSceneAsBuildInputs::from_brick_bounds(&[RtBrickBounds {
            brick_coord: UVec3::ZERO,
            brick_id: 0,
            min: Vec3::ZERO,
            max: Vec3::splat(8.0),
            generation: 1,
        }]);

        let blas_geometry = inputs.blas_geometry(0x1000);
        let blas_range = inputs.blas_build_range();
        assert_eq!(blas_geometry.geometry_type, vk::GeometryTypeKHR::AABBS);
        assert!(blas_geometry.flags.contains(vk::GeometryFlagsKHR::OPAQUE));
        assert_eq!(blas_range.primitive_count, 1);
        unsafe {
            assert_eq!(blas_geometry.geometry.aabbs.data.device_address, 0x1000);
            assert_eq!(
                blas_geometry.geometry.aabbs.stride,
                std::mem::size_of::<vk::AabbPositionsKHR>() as u64
            );
        }

        let tlas_geometry = inputs.tlas_geometry(0x2000);
        let tlas_range = inputs.tlas_build_range();
        assert_eq!(tlas_geometry.geometry_type, vk::GeometryTypeKHR::INSTANCES);
        assert_eq!(tlas_range.primitive_count, 1);
        unsafe {
            assert_eq!(
                tlas_geometry.geometry.instances.array_of_pointers,
                vk::FALSE
            );
            assert_eq!(tlas_geometry.geometry.instances.data.device_address, 0x2000);
        }
    }

    #[test]
    fn rt_acceleration_structure_resource_owns_storage_buffer_and_handle_lifecycle() {
        let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT scene implementation should precede tests");

        for token in [
            "pub struct RtAccelerationStructure",
            "create_acceleration_structure",
            "destroy_acceleration_structure",
            "get_acceleration_structure_device_address",
            "ACCELERATION_STRUCTURE_STORAGE_KHR",
            "SHADER_DEVICE_ADDRESS",
        ] {
            assert!(
                implementation.contains(token),
                "RT acceleration structure resource missing {token}"
            );
        }
    }

    #[test]
    fn rt_scene_gpu_build_resources_query_sizes_and_own_build_buffers() {
        let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT scene implementation should precede tests");

        for token in [
            "pub struct RtSceneGpuBuildResources",
            "aabb_buffer: GpuBuffer",
            "instance_buffer: GpuBuffer",
            "scratch_buffer: GpuBuffer",
            "blas: RtAccelerationStructure",
            "tlas: RtAccelerationStructure",
            "get_acceleration_structure_build_sizes",
            "AccelerationStructureBuildTypeKHR::DEVICE",
            "MemoryLocation::CpuToGpu",
            "MemoryLocation::GpuOnly",
        ] {
            assert!(
                implementation.contains(token),
                "RT scene GPU build resources missing {token}"
            );
        }
    }

    #[test]
    fn rt_scene_records_blas_then_tlas_build_with_dependency_barrier() {
        let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT scene implementation should precede tests");

        for token in [
            "AccelerationStructureBuildGeometryInfoKHR",
            "AccelerationStructureTypeKHR::BOTTOM_LEVEL",
            "AccelerationStructureTypeKHR::TOP_LEVEL",
            "BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE",
            "BuildAccelerationStructureModeKHR::BUILD",
            "cmd_pipeline_barrier",
            "ACCELERATION_STRUCTURE_WRITE_KHR",
            "ACCELERATION_STRUCTURE_READ_KHR",
        ] {
            assert!(
                implementation.contains(token),
                "RT scene GPU build recording missing {token}"
            );
        }

        let blas_info = implementation
            .find("let blas_build_info")
            .expect("BLAS build info should be prepared before recording builds");
        let first_build = implementation[blas_info..]
            .find("cmd_build_acceleration_structures")
            .map(|offset| blas_info + offset)
            .expect("BLAS build command should be recorded");
        let barrier = implementation[first_build..]
            .find("cmd_pipeline_barrier")
            .map(|offset| first_build + offset)
            .expect("BLAS/TLAS build dependency barrier should be recorded");
        let second_build = implementation[barrier..]
            .find("cmd_build_acceleration_structures")
            .map(|offset| barrier + offset)
            .expect("TLAS build command should be recorded after the barrier");

        assert!(blas_info < first_build);
        assert!(first_build < barrier);
        assert!(barrier < second_build);
    }

    #[test]
    fn rt_scene_records_tlas_build_to_trace_read_barrier() {
        let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT scene implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        assert!(compact.contains("lettlas_to_trace_barrier"));
        assert!(compact.contains("ACCELERATION_STRUCTURE_WRITE_KHR"));
        assert!(compact.contains("ACCELERATION_STRUCTURE_READ_KHR"));
        assert!(compact.contains("PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR"));
        assert!(compact.contains("PipelineStageFlags::RAY_TRACING_SHADER_KHR"));

        let tlas_build = compact
            .find("lettlas_build_info")
            .expect("TLAS build info should be prepared");
        let second_build = compact[tlas_build..]
            .find("cmd_build_acceleration_structures")
            .map(|offset| tlas_build + offset)
            .expect("TLAS build command should be recorded");
        let tlas_to_trace = compact[second_build..]
            .find("lettlas_to_trace_barrier")
            .map(|offset| second_build + offset)
            .expect("TLAS trace-read barrier should follow TLAS build");

        assert!(second_build < tlas_to_trace);
    }

    #[test]
    fn rt_scene_update_compatibility_requires_matching_blas_and_tlas_counts() {
        let previous = RtSceneAsBuildInputs {
            aabbs: vec![test_aabb(0.0), test_aabb(8.0)],
        };
        let same_count = RtSceneAsBuildInputs {
            aabbs: vec![test_aabb(16.0), test_aabb(24.0)],
        };
        let added_primitive = RtSceneAsBuildInputs {
            aabbs: vec![test_aabb(0.0), test_aabb(8.0), test_aabb(16.0)],
        };
        let emptied_scene = RtSceneAsBuildInputs { aabbs: Vec::new() };

        assert!(
            previous.can_update_in_place(&same_count),
            "same BLAS primitive and TLAS instance counts should be update-compatible"
        );
        assert!(
            !previous.can_update_in_place(&added_primitive),
            "changing BLAS primitive count must force a full AS rebuild"
        );
        assert!(
            !previous.can_update_in_place(&emptied_scene),
            "empty scenes must clear AS resources instead of updating in place"
        );
    }

    #[test]
    fn rt_scene_gpu_update_path_uses_vulkan_update_mode_and_reuses_resources() {
        let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT scene implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE",
            "BuildAccelerationStructureModeKHR::UPDATE",
            "update_scratch_size",
            "pub fn can_update_in_place",
            "pub fn update_inputs",
            "pub fn record_update",
            "resources.record_update",
        ] {
            assert!(
                implementation.contains(token),
                "RT scene GPU update path missing {token}"
            );
        }

        for token in [
            ".src_acceleration_structure(self.blas.handle)",
            ".dst_acceleration_structure(self.blas.handle)",
            ".src_acceleration_structure(self.tlas.handle)",
            ".dst_acceleration_structure(self.tlas.handle)",
            "resources.update_inputs(inputs)",
        ] {
            assert!(
                compact.contains(token),
                "RT scene GPU update path missing compact token {token}"
            );
        }
    }

    #[test]
    fn rt_scene_scratch_buffer_addresses_are_aligned_with_padding_capacity() {
        assert_eq!(align_device_address(0x1000, 256), 0x1000);
        assert_eq!(align_device_address(0x1001, 256), 0x1100);
        assert_eq!(align_device_address(0x1001, 0), 0x1001);
        assert_eq!(scratch_buffer_allocation_size(4096, 256), 4096 + 255);
        assert_eq!(scratch_buffer_allocation_size(4096, 1), 4096);

        let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT scene implementation should precede tests");
        assert!(implementation.contains("scratch_alignment"));
        assert!(implementation.contains("scratch_device_address"));
    }

    fn test_aabb(offset: f32) -> vk::AabbPositionsKHR {
        vk::AabbPositionsKHR::default()
            .min_x(offset)
            .min_y(offset)
            .min_z(offset)
            .max_x(offset + 8.0)
            .max_y(offset + 8.0)
            .max_z(offset + 8.0)
    }
}
