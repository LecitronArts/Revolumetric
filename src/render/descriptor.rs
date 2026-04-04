use anyhow::{Context, Result};
use ash::vk;

pub struct DescriptorLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
}

impl DescriptorLayoutBuilder {
    pub fn new() -> Self {
        Self { bindings: Vec::new() }
    }

    pub fn add_binding(
        mut self,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
        count: u32,
    ) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(count)
                .stage_flags(stage_flags),
        );
        self
    }

    pub fn build(&self, device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.bindings);
        unsafe { device.create_descriptor_set_layout(&create_info, None) }
            .context("failed to create descriptor set layout")
    }
}

pub struct DescriptorPool {
    pub handle: vk::DescriptorPool,
}

impl DescriptorPool {
    pub fn new(
        device: &ash::Device,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self> {
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let handle = unsafe { device.create_descriptor_pool(&create_info, None) }
            .context("failed to create descriptor pool")?;
        Ok(Self { handle })
    }

    pub fn allocate(
        &self,
        device: &ash::Device,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Vec<vk::DescriptorSet>> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.handle)
            .set_layouts(layouts);
        unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .context("failed to allocate descriptor sets")
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_pool(self.handle, None) };
    }
}
