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
    RayTracing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    Undefined,
    ComputeShaderRead,
    ComputeShaderReadWrite,
    ComputeShaderWrite,
    RayTracingShaderRead,
    RayTracingShaderReadWrite,
    RayTracingShaderWrite,
    AccelerationStructureBuildRead,
    AccelerationStructureBuildReadWrite,
    AccelerationStructureBuildWrite,
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
            Self::RayTracingShaderRead
            | Self::RayTracingShaderReadWrite
            | Self::RayTracingShaderWrite => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            Self::AccelerationStructureBuildRead
            | Self::AccelerationStructureBuildReadWrite
            | Self::AccelerationStructureBuildWrite => {
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
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
            Self::RayTracingShaderRead => vk::AccessFlags::SHADER_READ,
            Self::RayTracingShaderReadWrite => {
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
            }
            Self::RayTracingShaderWrite => vk::AccessFlags::SHADER_WRITE,
            Self::AccelerationStructureBuildRead => {
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
            }
            Self::AccelerationStructureBuildReadWrite => {
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
            }
            Self::AccelerationStructureBuildWrite => {
                vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
            }
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
            Self::RayTracingShaderRead
            | Self::RayTracingShaderReadWrite
            | Self::RayTracingShaderWrite
            | Self::AccelerationStructureBuildRead
            | Self::AccelerationStructureBuildReadWrite
            | Self::AccelerationStructureBuildWrite => vk::ImageLayout::GENERAL,
            Self::TransferRead => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            Self::TransferWrite => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            Self::ColorAttachmentWrite => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Self::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    pub fn is_read_write(self) -> bool {
        matches!(
            self,
            Self::ComputeShaderReadWrite
                | Self::RayTracingShaderReadWrite
                | Self::AccelerationStructureBuildReadWrite
        )
    }

    pub fn is_ray_tracing_shader(self) -> bool {
        matches!(
            self,
            Self::RayTracingShaderRead
                | Self::RayTracingShaderReadWrite
                | Self::RayTracingShaderWrite
        )
    }

    pub fn is_acceleration_structure_build(self) -> bool {
        matches!(
            self,
            Self::AccelerationStructureBuildRead
                | Self::AccelerationStructureBuildReadWrite
                | Self::AccelerationStructureBuildWrite
        )
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransientResourceBinding {
    pub resource_id: u32,
    pub slot_index: usize,
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

    #[test]
    fn rendergraph_transient_resource_binding_exposes_resource_and_slot_ids() {
        let binding = TransientResourceBinding {
            resource_id: 12,
            slot_index: 3,
        };

        assert_eq!(binding.resource_id, 12);
        assert_eq!(binding.slot_index, 3);
    }

    #[test]
    fn access_kind_maps_rt_and_acceleration_structure_accesses_to_vulkan_barrier_fields() {
        let cases = [
            (
                AccessKind::RayTracingShaderRead,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags::SHADER_READ,
                vk::ImageLayout::GENERAL,
            ),
            (
                AccessKind::RayTracingShaderReadWrite,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                vk::ImageLayout::GENERAL,
            ),
            (
                AccessKind::RayTracingShaderWrite,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags::SHADER_WRITE,
                vk::ImageLayout::GENERAL,
            ),
            (
                AccessKind::AccelerationStructureBuildRead,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                vk::ImageLayout::GENERAL,
            ),
            (
                AccessKind::AccelerationStructureBuildReadWrite,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                vk::ImageLayout::GENERAL,
            ),
            (
                AccessKind::AccelerationStructureBuildWrite,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                vk::ImageLayout::GENERAL,
            ),
        ];

        for (kind, stage, access, layout) in cases {
            assert_eq!(kind.stage_flags(), stage);
            assert_eq!(kind.access_flags(), access);
            assert_eq!(kind.image_layout(), layout);
        }
        assert!(AccessKind::RayTracingShaderRead.is_ray_tracing_shader());
        assert!(AccessKind::AccelerationStructureBuildWrite.is_acceleration_structure_build());
    }
}
