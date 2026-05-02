use crate::render::resource::{
    AccessKind, QueueType, ResourceAccess, ResourceDesc, ResourceHandle,
};
use ash::vk;

pub struct PassBuilder {
    pub name: &'static str,
    pub queue_type: QueueType,
    pub reads: Vec<ResourceHandle>,
    pub writes: Vec<ResourceHandle>,
    pub accesses: Vec<ResourceAccess>,
    pub final_accesses: Vec<ResourceAccess>,
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
            accesses: Vec::new(),
            final_accesses: Vec::new(),
            resource_descs: Vec::new(),
            next_resource_id: next_id,
        }
    }

    pub fn read(&mut self, handle: ResourceHandle) {
        self.read_as(handle, AccessKind::ComputeShaderRead);
    }

    pub fn read_as(&mut self, handle: ResourceHandle, kind: AccessKind) {
        self.reads.push(handle);
        self.accesses.push(ResourceAccess { handle, kind });
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
        self.accesses.push(ResourceAccess {
            handle,
            kind: AccessKind::ComputeShaderWrite,
        });
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
        self.accesses.push(ResourceAccess {
            handle,
            kind: AccessKind::ComputeShaderWrite,
        });
        self.resource_descs
            .push((handle, ResourceDesc::Buffer { size, usage }));
        handle
    }

    pub fn write(&mut self, handle: ResourceHandle) -> ResourceHandle {
        self.write_as(handle, AccessKind::ComputeShaderWrite)
    }

    pub fn write_as(&mut self, handle: ResourceHandle, kind: AccessKind) -> ResourceHandle {
        let new = ResourceHandle {
            id: handle.id,
            version: handle.version + 1,
        };
        if kind == AccessKind::ComputeShaderReadWrite {
            self.reads.push(handle);
        }
        self.writes.push(new);
        self.accesses.push(ResourceAccess { handle: new, kind });
        new
    }

    pub fn finish_as(&mut self, handle: ResourceHandle, kind: AccessKind) {
        self.final_accesses.push(ResourceAccess { handle, kind });
    }
}

pub struct PassContext<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub frame_index: u64,
}
