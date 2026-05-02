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
    Present,
}

impl AccessKind {
    pub fn from_swapchain_layout(layout: vk::ImageLayout) -> Option<Self> {
        match layout {
            vk::ImageLayout::UNDEFINED => Some(Self::Undefined),
            vk::ImageLayout::PRESENT_SRC_KHR => Some(Self::Present),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => Some(Self::TransferWrite),
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
            Self::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceAccess {
    pub handle: ResourceHandle,
    pub kind: AccessKind,
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
