use crate::render::resource::{QueueType, ResourceDesc, ResourceHandle};
use ash::vk;

pub struct PassBuilder {
    pub name: &'static str,
    pub queue_type: QueueType,
    pub reads: Vec<ResourceHandle>,
    pub writes: Vec<ResourceHandle>,
    pub(crate) resource_descs: Vec<(ResourceHandle, ResourceDesc)>,
    pub(crate) next_resource_id: u32,
}

impl PassBuilder {
    pub fn new(name: &'static str, queue_type: QueueType, next_id: u32) -> Self {
        Self {
            name,
            queue_type,
            reads: Vec::new(),
            writes: Vec::new(),
            resource_descs: Vec::new(),
            next_resource_id: next_id,
        }
    }

    pub fn read(&mut self, handle: ResourceHandle) {
        self.reads.push(handle);
    }

    pub fn create_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> ResourceHandle {
        let handle = ResourceHandle {
            id: self.next_resource_id,
            version: 0,
        };
        self.next_resource_id += 1;
        self.writes.push(handle);
        self.resource_descs.push((
            handle,
            ResourceDesc::Image {
                width,
                height,
                format,
                usage,
            },
        ));
        handle
    }

    pub fn create_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> ResourceHandle {
        let handle = ResourceHandle {
            id: self.next_resource_id,
            version: 0,
        };
        self.next_resource_id += 1;
        self.writes.push(handle);
        self.resource_descs
            .push((handle, ResourceDesc::Buffer { size, usage }));
        handle
    }

    pub fn write(&mut self, handle: ResourceHandle) -> ResourceHandle {
        let new = ResourceHandle {
            id: handle.id,
            version: handle.version + 1,
        };
        self.writes.push(new);
        new
    }
}

pub struct PassContext<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub frame_index: u64,
}
