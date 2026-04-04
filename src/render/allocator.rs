use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;

pub struct GpuAllocator {
    inner: Mutex<Allocator>,
}

impl GpuAllocator {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;
        Ok(Self {
            inner: Mutex::new(allocator),
        })
    }

    pub fn allocate(&self, desc: &AllocationCreateDesc) -> Result<Allocation> {
        Ok(self.inner.lock().allocate(desc)?)
    }

    pub fn free(&self, allocation: Allocation) -> Result<()> {
        Ok(self.inner.lock().free(allocation)?)
    }
}
