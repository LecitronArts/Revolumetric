use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle {
    pub id: u32,
    pub version: u32,
}

#[derive(Debug, Clone)]
pub enum ResourceDesc {
    Image {
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    },
    Buffer {
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    Graphics,
    Compute,
    Transfer,
}

#[derive(Debug, Clone)]
pub struct PassDecl {
    pub name: &'static str,
    pub queue_type: QueueType,
    pub reads: Vec<ResourceHandle>,
    pub writes: Vec<ResourceHandle>,
}
