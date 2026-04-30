use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::render::allocator::GpuAllocator;

pub struct GpuBuffer {
    pub handle: vk::Buffer,
    pub size: vk::DeviceSize,
    pub allocation: Option<Allocation>,
    pub usage: vk::BufferUsageFlags,
}

impl GpuBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { device.create_buffer(&buffer_info, None) }
            .context("failed to create buffer")?;

        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe { device.bind_buffer_memory(handle, allocation.memory(), allocation.offset()) }
            .context("failed to bind buffer memory")?;

        Ok(Self {
            handle,
            size,
            allocation: Some(allocation),
            usage,
        })
    }

    pub fn destroy(mut self, device: &ash::Device, allocator: &GpuAllocator) {
        unsafe { device.destroy_buffer(self.handle, None) };
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
    }

    /// Returns a mapped pointer if the buffer is host-visible.
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.allocation
            .as_ref()
            .and_then(|a| a.mapped_ptr())
            .map(|p| p.as_ptr() as *mut u8)
    }
}
