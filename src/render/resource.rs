use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle {
    pub id: u32,
    pub version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    Undefined,
    ComputeShaderRead,
    ComputeShaderReadWrite,
    ComputeShaderWrite,
    TransferRead,
    TransferWrite,
    ColorAttachmentWrite,
    Present,
}

impl AccessKind {
    pub fn from_swapchain_layout(layout: vk::ImageLayout) -> Option<Self> {
        match layout {
            vk::ImageLayout::UNDEFINED => Some(Self::Undefined),
            vk::ImageLayout::PRESENT_SRC_KHR => Some(Self::Present),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => Some(Self::TransferWrite),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => Some(Self::ColorAttachmentWrite),
            _ => None,
        }
    }

    pub fn stage_flags(self) -> vk::PipelineStageFlags {
        match self {
            Self::Undefined => vk::PipelineStageFlags::TOP_OF_PIPE,
            Self::ComputeShaderRead | Self::ComputeShaderReadWrite | Self::ComputeShaderWrite => {
                vk::PipelineStageFlags::COMPUTE_SHADER
            }
            Self::TransferRead | Self::TransferWrite => vk::PipelineStageFlags::TRANSFER,
            Self::ColorAttachmentWrite => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            Self::Present => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        }
    }

    pub fn access_flags(self) -> vk::AccessFlags {
        match self {
            Self::Undefined => vk::AccessFlags::empty(),
            Self::ComputeShaderRead => vk::AccessFlags::SHADER_READ,
            Self::ComputeShaderReadWrite => {
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
            }
            Self::ComputeShaderWrite => vk::AccessFlags::SHADER_WRITE,
            Self::TransferRead => vk::AccessFlags::TRANSFER_READ,
            Self::TransferWrite => vk::AccessFlags::TRANSFER_WRITE,
            Self::ColorAttachmentWrite => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            Self::Present => vk::AccessFlags::empty(),
        }
    }

    pub fn image_layout(self) -> vk::ImageLayout {
        match self {
            Self::Undefined => vk::ImageLayout::UNDEFINED,
            Self::ComputeShaderRead | Self::ComputeShaderReadWrite | Self::ComputeShaderWrite => {
                vk::ImageLayout::GENERAL
            }
            Self::TransferRead => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            Self::TransferWrite => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            Self::ColorAttachmentWrite => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Self::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceAccess {
    pub handle: ResourceHandle,
    pub kind: AccessKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceOrigin {
    Imported,
    Transient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceLifetime {
    pub resource_id: u32,
    pub origin: ResourceOrigin,
    pub first_step: usize,
    pub last_step: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransientAliasCandidate {
    pub first_resource_id: u32,
    pub second_resource_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransientResourceSlot {
    pub slot_index: usize,
    pub desc: ResourceDesc,
    pub resource_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct PassDecl {
    pub name: &'static str,
    pub queue_type: QueueType,
    pub reads: Vec<ResourceHandle>,
    pub writes: Vec<ResourceHandle>,
    pub accesses: Vec<ResourceAccess>,
    pub final_accesses: Vec<ResourceAccess>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendergraph_resource_lifecycle_origin_values_are_copyable() {
        let imported = ResourceOrigin::Imported;
        let transient = ResourceOrigin::Transient;

        assert_eq!(imported, ResourceOrigin::Imported);
        assert_eq!(transient, ResourceOrigin::Transient);
    }

    #[test]
    fn rendergraph_resource_lifecycle_structs_expose_ids_and_steps() {
        let lifetime = ResourceLifetime {
            resource_id: 7,
            origin: ResourceOrigin::Transient,
            first_step: 1,
            last_step: 3,
        };
        let candidate = TransientAliasCandidate {
            first_resource_id: 7,
            second_resource_id: 9,
        };

        assert_eq!(lifetime.resource_id, 7);
        assert_eq!(lifetime.origin, ResourceOrigin::Transient);
        assert_eq!(lifetime.first_step, 1);
        assert_eq!(lifetime.last_step, 3);
        assert_eq!(candidate.first_resource_id, 7);
        assert_eq!(candidate.second_resource_id, 9);
    }

    #[test]
    fn rendergraph_transient_slot_plan_struct_exposes_slot_desc_and_resources() {
        let slot = TransientResourceSlot {
            slot_index: 2,
            desc: ResourceDesc::Buffer {
                size: 256,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
            resource_ids: vec![4, 9],
        };

        assert_eq!(slot.slot_index, 2);
        assert_eq!(slot.resource_ids, vec![4, 9]);
        assert_eq!(
            slot.desc,
            ResourceDesc::Buffer {
                size: 256,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            }
        );
    }
}
