use ash::vk;
use crate::render::resource::{ResourceHandle, QueueType};

pub struct PassBuilder {
    pub name: &'static str,
    pub queue_type: QueueType,
    pub reads: Vec<ResourceHandle>,
    pub writes: Vec<ResourceHandle>,
    pub(crate) next_resource_id: u32,
}

impl PassBuilder {
    pub fn new(name: &'static str, queue_type: QueueType, next_id: u32) -> Self {
        Self {
            name,
            queue_type,
            reads: Vec::new(),
            writes: Vec::new(),
            next_resource_id: next_id,
        }
    }

    pub fn read(&mut self, handle: ResourceHandle) {
        self.reads.push(handle);
    }

    pub fn create_image(
        &mut self,
        _width: u32,
        _height: u32,
        _format: vk::Format,
        _usage: vk::ImageUsageFlags,
    ) -> ResourceHandle {
        let handle = ResourceHandle {
            id: self.next_resource_id,
            version: 0,
        };
        self.next_resource_id += 1;
        self.writes.push(handle);
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
