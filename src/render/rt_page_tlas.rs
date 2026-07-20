use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::rt_page_registry::{
    RtPageRegistry, RtPageRepresentation, RtPageSlot, RtPageState,
};
use crate::render::rt_scene::{RtAccelerationStructure, RtSceneAsBuildInputs};
use crate::voxel::brick::BRICK_EDGE;
use anyhow::{Result as AnyResult, anyhow, bail};
use ash::vk;
use glam::UVec3;
use gpu_allocator::MemoryLocation;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const RT_PAGE_REFERENCE_HIT_GROUP_OFFSET: u32 = 0;
pub const RT_PAGE_COMPACT_EXACT_HIT_GROUP_OFFSET: u32 = 1;
pub const RT_PAGE_TLAS_MAX_INSTANCE_SLOTS: u32 = 0x00ff_ffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPageTlasInstanceBinding {
    Dummy {
        blas_address: vk::DeviceAddress,
    },
    Reference {
        blas_address: vk::DeviceAddress,
    },
    CompactExact {
        blas_address: vk::DeviceAddress,
        resource_version: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RtPageTlasError {
    #[error("RT page TLAS requires at least the dummy instance slot")]
    EmptyCapacity,
    #[error("RT page TLAS requires at least one frame slot")]
    NoFrameSlots,
    #[error("RT page TLAS instance slot exceeds 24-bit capacity: {slot}")]
    InstanceSlotOutOfRange { slot: u32 },
    #[error("RT page TLAS BLAS device address is null")]
    NullBlasAddress,
    #[error(
        "RT page TLAS {label} input address is misaligned: address={address:#x} alignment={alignment}"
    )]
    MisalignedInputAddress {
        label: &'static str,
        address: u64,
        alignment: u64,
    },
    #[error("RT page TLAS page translation overflows for {page:?}")]
    PageTranslationOverflow { page: UVec3 },
    #[error(
        "RT page TLAS capacity exceeds the supported limit: requested={requested} limit={limit}"
    )]
    CapacityExceedsLimit { requested: u32, limit: u32 },
    #[error(
        "RT page TLAS capacity growth requires quiescence: current={current} required={required}"
    )]
    CapacityGrowthRequiresQuiescence { current: u32, required: u32 },
    #[error("RT page TLAS frame slot is out of range: slot={frame_slot} count={frame_slot_count}")]
    FrameSlotOutOfRange {
        frame_slot: usize,
        frame_slot_count: usize,
    },
    #[error("RT page TLAS frame slot {frame_slot} is still in flight at generation {generation}")]
    FrameSlotInFlight { frame_slot: usize, generation: u64 },
    #[error(
        "RT page TLAS frame generation mismatch: slot={frame_slot} expected={expected} actual={actual}"
    )]
    FrameGenerationMismatch {
        frame_slot: usize,
        expected: u64,
        actual: u64,
    },
    #[error("RT page TLAS frame slot {frame_slot} has no in-flight submission")]
    FrameSlotIdle { frame_slot: usize },
    #[error("RT page TLAS registry slot is unknown: {slot}")]
    UnknownRegistrySlot { slot: RtPageSlot },
    #[error(
        "RT page TLAS slot exceeds the fixed instance capacity: slot={slot} capacity={capacity}"
    )]
    SlotExceedsCapacity { slot: RtPageSlot, capacity: u32 },
    #[error("RT page TLAS CompactExact binding is missing for resource version {resource_version}")]
    MissingCompactBinding { resource_version: u64 },
    #[error(
        "RT page TLAS CompactExact binding version mismatch: expected={expected} actual={actual}"
    )]
    CompactBindingVersionMismatch { expected: u64, actual: u64 },
    #[error("RT page TLAS representation is not implemented in the mixed pipeline: {0:?}")]
    UnsupportedRepresentation(RtPageRepresentation),
}

pub fn make_page_tlas_instance(
    slot: u32,
    page: UVec3,
    binding: RtPageTlasInstanceBinding,
) -> Result<vk::AccelerationStructureInstanceKHR, RtPageTlasError> {
    if slot >= RT_PAGE_TLAS_MAX_INSTANCE_SLOTS {
        return Err(RtPageTlasError::InstanceSlotOutOfRange { slot });
    }
    let (blas_address, mask, hit_group_offset) = match binding {
        RtPageTlasInstanceBinding::Dummy { blas_address } => {
            (blas_address, 0, RT_PAGE_REFERENCE_HIT_GROUP_OFFSET)
        }
        RtPageTlasInstanceBinding::Reference { blas_address } => {
            (blas_address, 0xff, RT_PAGE_REFERENCE_HIT_GROUP_OFFSET)
        }
        RtPageTlasInstanceBinding::CompactExact {
            blas_address,
            resource_version: _,
        } => (blas_address, 0xff, RT_PAGE_COMPACT_EXACT_HIT_GROUP_OFFSET),
    };
    if blas_address == 0 {
        return Err(RtPageTlasError::NullBlasAddress);
    }
    let translation = page
        .to_array()
        .map(|coordinate| coordinate.checked_mul(BRICK_EDGE))
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(RtPageTlasError::PageTranslationOverflow { page })?;
    let force_opaque = vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8;

    Ok(vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: [
                1.0,
                0.0,
                0.0,
                translation[0] as f32,
                0.0,
                1.0,
                0.0,
                translation[1] as f32,
                0.0,
                0.0,
                1.0,
                translation[2] as f32,
            ],
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(slot, mask),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            hit_group_offset,
            force_opaque,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas_address,
        },
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageTlasCapacity {
    current: u32,
    limit: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPageTlasCapacityPlan {
    Reuse,
    Grow { new_capacity: u32 },
}

impl RtPageTlasCapacity {
    pub fn new(current: u32, limit: u32) -> Result<Self, RtPageTlasError> {
        if current == 0 || limit == 0 {
            return Err(RtPageTlasError::EmptyCapacity);
        }
        let limit = limit.min(RT_PAGE_TLAS_MAX_INSTANCE_SLOTS);
        if current > limit {
            return Err(RtPageTlasError::CapacityExceedsLimit {
                requested: current,
                limit,
            });
        }
        Ok(Self { current, limit })
    }

    pub fn device_limit(max_instance_count: u64) -> u32 {
        max_instance_count.min(u64::from(RT_PAGE_TLAS_MAX_INSTANCE_SLOTS)) as u32
    }

    pub fn plan_for_required_slots(
        self,
        required: u32,
        all_frame_slots_quiescent: bool,
    ) -> Result<RtPageTlasCapacityPlan, RtPageTlasError> {
        if required <= self.current {
            return Ok(RtPageTlasCapacityPlan::Reuse);
        }
        if required > self.limit {
            return Err(RtPageTlasError::CapacityExceedsLimit {
                requested: required,
                limit: self.limit,
            });
        }
        if !all_frame_slots_quiescent {
            return Err(RtPageTlasError::CapacityGrowthRequiresQuiescence {
                current: self.current,
                required,
            });
        }
        Ok(RtPageTlasCapacityPlan::Grow {
            new_capacity: required.next_power_of_two().min(self.limit),
        })
    }
}

#[derive(Debug, Default)]
struct RtPageTlasFrameSlotReferences {
    generation: Option<u64>,
    resource_versions: BTreeSet<u64>,
}

#[derive(Debug)]
pub struct RtPageTlasFrameReferences {
    frame_slots: Vec<RtPageTlasFrameSlotReferences>,
}

impl RtPageTlasFrameReferences {
    pub fn new(frame_slot_count: usize) -> Result<Self, RtPageTlasError> {
        if frame_slot_count == 0 {
            return Err(RtPageTlasError::NoFrameSlots);
        }
        Ok(Self {
            frame_slots: (0..frame_slot_count)
                .map(|_| RtPageTlasFrameSlotReferences::default())
                .collect(),
        })
    }

    pub fn record_submission(
        &mut self,
        frame_slot: usize,
        generation: u64,
        resource_versions: impl IntoIterator<Item = u64>,
    ) -> Result<(), RtPageTlasError> {
        let frame_slot_count = self.frame_slots.len();
        let slot =
            self.frame_slots
                .get_mut(frame_slot)
                .ok_or(RtPageTlasError::FrameSlotOutOfRange {
                    frame_slot,
                    frame_slot_count,
                })?;
        if let Some(generation) = slot.generation {
            return Err(RtPageTlasError::FrameSlotInFlight {
                frame_slot,
                generation,
            });
        }
        slot.generation = Some(generation);
        slot.resource_versions.extend(resource_versions);
        Ok(())
    }

    pub fn complete_submission(
        &mut self,
        frame_slot: usize,
        generation: u64,
    ) -> Result<(), RtPageTlasError> {
        let frame_slot_count = self.frame_slots.len();
        let slot =
            self.frame_slots
                .get_mut(frame_slot)
                .ok_or(RtPageTlasError::FrameSlotOutOfRange {
                    frame_slot,
                    frame_slot_count,
                })?;
        let expected = slot
            .generation
            .ok_or(RtPageTlasError::FrameSlotIdle { frame_slot })?;
        if expected != generation {
            return Err(RtPageTlasError::FrameGenerationMismatch {
                frame_slot,
                expected,
                actual: generation,
            });
        }
        slot.generation = None;
        slot.resource_versions.clear();
        Ok(())
    }

    pub fn complete_through(
        &mut self,
        frame_slot: usize,
        completed_epoch: u64,
    ) -> Result<(), RtPageTlasError> {
        let frame_slot_count = self.frame_slots.len();
        let slot =
            self.frame_slots
                .get_mut(frame_slot)
                .ok_or(RtPageTlasError::FrameSlotOutOfRange {
                    frame_slot,
                    frame_slot_count,
                })?;
        let expected = slot
            .generation
            .ok_or(RtPageTlasError::FrameSlotIdle { frame_slot })?;
        if completed_epoch < expected {
            return Err(RtPageTlasError::FrameGenerationMismatch {
                frame_slot,
                expected,
                actual: completed_epoch,
            });
        }
        slot.generation = None;
        slot.resource_versions.clear();
        Ok(())
    }

    pub fn can_retire(&self, resource_version: u64) -> bool {
        self.frame_slots
            .iter()
            .all(|slot| !slot.resource_versions.contains(&resource_version))
    }

    pub fn in_flight_generation(&self, frame_slot: usize) -> Option<u64> {
        self.frame_slots
            .get(frame_slot)
            .and_then(|slot| slot.generation)
    }

    pub fn all_quiescent(&self) -> bool {
        self.frame_slots
            .iter()
            .all(|slot| slot.generation.is_none())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageCompactTlasBinding {
    pub resource_version: u64,
    pub blas_address: vk::DeviceAddress,
}

pub struct RtPageTlasInstanceStore {
    dummy_blas_address: vk::DeviceAddress,
    reference_blas_address: vk::DeviceAddress,
    instances: Vec<vk::AccelerationStructureInstanceKHR>,
    resource_versions_by_slot: Vec<Option<u64>>,
    resource_version_ref_counts: BTreeMap<u64, u32>,
}

impl RtPageTlasInstanceStore {
    pub fn new(
        capacity: u32,
        dummy_blas_address: vk::DeviceAddress,
        reference_blas_address: vk::DeviceAddress,
    ) -> Result<Self, RtPageTlasError> {
        RtPageTlasCapacity::new(capacity, RT_PAGE_TLAS_MAX_INSTANCE_SLOTS)?;
        if dummy_blas_address == 0 || reference_blas_address == 0 {
            return Err(RtPageTlasError::NullBlasAddress);
        }
        let instances = (0..capacity)
            .map(|slot| {
                make_page_tlas_instance(
                    slot,
                    UVec3::ZERO,
                    RtPageTlasInstanceBinding::Dummy {
                        blas_address: dummy_blas_address,
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            dummy_blas_address,
            reference_blas_address,
            instances,
            resource_versions_by_slot: vec![None; capacity as usize],
            resource_version_ref_counts: BTreeMap::new(),
        })
    }

    pub fn sync_registry_slot(
        &mut self,
        registry: &RtPageRegistry,
        slot: RtPageSlot,
        compact_binding: Option<RtPageCompactTlasBinding>,
    ) -> Result<(), RtPageTlasError> {
        if slot as usize >= self.instances.len() {
            return Err(RtPageTlasError::SlotExceedsCapacity {
                slot,
                capacity: self.instances.len() as u32,
            });
        }
        if slot == 0 {
            return self.set_slot(
                slot,
                UVec3::ZERO,
                RtPageTlasInstanceBinding::Dummy {
                    blas_address: self.dummy_blas_address,
                },
            );
        }
        let page = registry
            .page_for_slot(slot)
            .ok_or(RtPageTlasError::UnknownRegistrySlot { slot })?;
        let binding = match registry.trace_representation(page) {
            RtPageRepresentation::Missing => RtPageTlasInstanceBinding::Dummy {
                blas_address: self.dummy_blas_address,
            },
            RtPageRepresentation::Reference => RtPageTlasInstanceBinding::Reference {
                blas_address: self.reference_blas_address,
            },
            RtPageRepresentation::CompactExact => {
                let resource_version = match registry.state_for_slot(slot) {
                    Some(RtPageState::Resident {
                        representation: RtPageRepresentation::CompactExact,
                        resource_version,
                        ..
                    }) => resource_version,
                    _ => return Err(RtPageTlasError::UnknownRegistrySlot { slot }),
                };
                let compact_binding = compact_binding
                    .ok_or(RtPageTlasError::MissingCompactBinding { resource_version })?;
                if compact_binding.resource_version != resource_version {
                    return Err(RtPageTlasError::CompactBindingVersionMismatch {
                        expected: resource_version,
                        actual: compact_binding.resource_version,
                    });
                }
                RtPageTlasInstanceBinding::CompactExact {
                    blas_address: compact_binding.blas_address,
                    resource_version,
                }
            }
            representation => {
                return Err(RtPageTlasError::UnsupportedRepresentation(representation));
            }
        };
        self.set_slot(slot, page, binding)
    }

    pub fn instances(&self) -> &[vk::AccelerationStructureInstanceKHR] {
        &self.instances
    }

    pub fn instance(&self, slot: RtPageSlot) -> Option<&vk::AccelerationStructureInstanceKHR> {
        self.instances.get(slot as usize)
    }

    pub fn resource_versions(&self) -> impl Iterator<Item = u64> + '_ {
        self.resource_version_ref_counts.keys().copied()
    }

    fn set_slot(
        &mut self,
        slot: RtPageSlot,
        page: UVec3,
        binding: RtPageTlasInstanceBinding,
    ) -> Result<(), RtPageTlasError> {
        let index = slot as usize;
        if index >= self.instances.len() {
            return Err(RtPageTlasError::SlotExceedsCapacity {
                slot,
                capacity: self.instances.len() as u32,
            });
        }
        let instance = make_page_tlas_instance(slot, page, binding)?;
        if let Some(old_version) = self.resource_versions_by_slot[index].take() {
            let count = self
                .resource_version_ref_counts
                .get_mut(&old_version)
                .expect("tracked TLAS slot version must retain a reference count");
            *count -= 1;
            if *count == 0 {
                self.resource_version_ref_counts.remove(&old_version);
            }
        }
        let new_version = match binding {
            RtPageTlasInstanceBinding::CompactExact {
                resource_version, ..
            } => Some(resource_version),
            _ => None,
        };
        self.instances[index] = instance;
        self.resource_versions_by_slot[index] = new_version;
        if let Some(new_version) = new_version {
            *self
                .resource_version_ref_counts
                .entry(new_version)
                .or_default() += 1;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageTlasGpuConfig {
    pub frame_slot_count: usize,
    pub instance_capacity: u32,
    pub max_instance_count: u64,
    pub min_acceleration_structure_scratch_offset_alignment: vk::DeviceSize,
}

pub struct RtPageTlasFrameSlotGpuResources {
    instance_buffer: GpuBuffer,
    tlas: RtAccelerationStructure,
    scratch_buffer: GpuBuffer,
    scratch_address: vk::DeviceAddress,
    capacity: u32,
    initialized: bool,
}

pub struct RtPageTlasGpuResources {
    dummy_vertex_buffer: GpuBuffer,
    reference_aabb_buffer: GpuBuffer,
    dummy_blas: RtAccelerationStructure,
    reference_blas: RtAccelerationStructure,
    dummy_scratch_buffer: GpuBuffer,
    dummy_scratch_address: vk::DeviceAddress,
    reference_scratch_buffer: GpuBuffer,
    reference_scratch_address: vk::DeviceAddress,
    frame_slots: Vec<RtPageTlasFrameSlotGpuResources>,
}

impl RtPageTlasGpuResources {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        config: RtPageTlasGpuConfig,
    ) -> AnyResult<Self> {
        if config.frame_slot_count == 0 {
            bail!(RtPageTlasError::NoFrameSlots);
        }
        let device_limit = RtPageTlasCapacity::device_limit(config.max_instance_count);
        RtPageTlasCapacity::new(config.instance_capacity, device_limit)
            .map_err(anyhow::Error::new)?;

        let mut guard = RtPageTlasBuildGuard::new(device, allocator, acceleration_structure_loader);
        let input_usage = fallback_input_buffer_usage();
        let dummy_vertex_buffer = guard.create_buffer(
            std::mem::size_of::<[[f32; 3]; 3]>() as u64,
            input_usage,
            MemoryLocation::CpuToGpu,
            "rt_page_dummy_vertices",
        )?;
        let reference_aabb_buffer = guard.create_buffer(
            std::mem::size_of::<vk::AabbPositionsKHR>() as u64,
            input_usage,
            MemoryLocation::CpuToGpu,
            "rt_page_reference_aabb",
        )?;
        write_mapped_slice(
            guard.buffer(dummy_vertex_buffer),
            &[[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "RT page dummy vertices",
        )?;
        write_mapped_slice(
            guard.buffer(reference_aabb_buffer),
            &[vk::AabbPositionsKHR {
                min_x: 0.0,
                min_y: 0.0,
                min_z: 0.0,
                max_x: BRICK_EDGE as f32,
                max_y: BRICK_EDGE as f32,
                max_z: BRICK_EDGE as f32,
            }],
            "RT page reference AABB",
        )?;

        let dummy_address = guard.buffer(dummy_vertex_buffer).device_address(device)?;
        let reference_address = guard.buffer(reference_aabb_buffer).device_address(device)?;
        validate_as_input_address(dummy_address, 4, "dummy triangle")?;
        validate_as_input_address(reference_address, 8, "reference AABB")?;
        let dummy_geometry = dummy_blas_geometry(dummy_address);
        let reference_geometry = reference_blas_geometry(reference_address);
        let dummy_sizes = query_build_sizes(
            acceleration_structure_loader,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            page_tlas_build_flags(),
            std::slice::from_ref(&dummy_geometry),
            &[1],
        );
        let reference_sizes = query_build_sizes(
            acceleration_structure_loader,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            page_tlas_build_flags(),
            std::slice::from_ref(&reference_geometry),
            &[1],
        );
        let dummy_blas = guard.create_acceleration_structure(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            dummy_sizes.acceleration_structure_size,
            "rt_page_dummy_blas",
        )?;
        let reference_blas = guard.create_acceleration_structure(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            reference_sizes.acceleration_structure_size,
            "rt_page_reference_blas",
        )?;
        let (dummy_scratch_buffer, dummy_scratch_address) = guard.create_scratch(
            dummy_sizes.build_scratch_size,
            config.min_acceleration_structure_scratch_offset_alignment,
            "rt_page_dummy_blas_scratch",
        )?;
        let (reference_scratch_buffer, reference_scratch_address) = guard.create_scratch(
            reference_sizes.build_scratch_size,
            config.min_acceleration_structure_scratch_offset_alignment,
            "rt_page_reference_blas_scratch",
        )?;

        let dummy_blas_address = guard
            .acceleration_structure(dummy_blas)
            .device_address(acceleration_structure_loader);
        if dummy_blas_address == 0 {
            bail!("Vulkan returned a null dummy BLAS device address");
        }
        let dummy_instances = (0..config.instance_capacity)
            .map(|slot| {
                make_page_tlas_instance(
                    slot,
                    UVec3::ZERO,
                    RtPageTlasInstanceBinding::Dummy {
                        blas_address: dummy_blas_address,
                    },
                )
                .map_err(anyhow::Error::new)
            })
            .collect::<AnyResult<Vec<_>>>()?;

        let mut frame_slot_indices = Vec::with_capacity(config.frame_slot_count);
        for frame_slot in 0..config.frame_slot_count {
            let instance_buffer = guard.create_buffer(
                instance_buffer_bytes(config.instance_capacity)?,
                RtSceneAsBuildInputs::instance_buffer_usage(),
                MemoryLocation::CpuToGpu,
                &format!("rt_page_instances_{frame_slot}"),
            )?;
            write_mapped_slice(
                guard.buffer(instance_buffer),
                &dummy_instances,
                "RT page TLAS instances",
            )?;
            let instance_address = guard.buffer(instance_buffer).device_address(device)?;
            validate_as_input_address(instance_address, 16, "instance")?;
            let geometry = tlas_geometry(instance_address);
            let tlas_sizes = query_build_sizes(
                acceleration_structure_loader,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                page_tlas_build_flags(),
                std::slice::from_ref(&geometry),
                &[config.instance_capacity],
            );
            let tlas = guard.create_acceleration_structure(
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                tlas_sizes.acceleration_structure_size,
                &format!("rt_page_tlas_{frame_slot}"),
            )?;
            let (scratch_buffer, scratch_address) = guard.create_scratch(
                tlas_sizes
                    .build_scratch_size
                    .max(tlas_sizes.update_scratch_size),
                config.min_acceleration_structure_scratch_offset_alignment,
                &format!("rt_page_tlas_scratch_{frame_slot}"),
            )?;
            frame_slot_indices.push((instance_buffer, tlas, scratch_buffer, scratch_address));
        }

        let frame_slots = frame_slot_indices
            .into_iter()
            .map(|(instance_buffer, tlas, scratch_buffer, scratch_address)| {
                RtPageTlasFrameSlotGpuResources {
                    instance_buffer: guard.take_buffer(instance_buffer),
                    tlas: guard.take_acceleration_structure(tlas),
                    scratch_buffer: guard.take_buffer(scratch_buffer),
                    scratch_address,
                    capacity: config.instance_capacity,
                    initialized: false,
                }
            })
            .collect();
        Ok(Self {
            dummy_vertex_buffer: guard.take_buffer(dummy_vertex_buffer),
            reference_aabb_buffer: guard.take_buffer(reference_aabb_buffer),
            dummy_blas: guard.take_acceleration_structure(dummy_blas),
            reference_blas: guard.take_acceleration_structure(reference_blas),
            dummy_scratch_buffer: guard.take_buffer(dummy_scratch_buffer),
            dummy_scratch_address,
            reference_scratch_buffer: guard.take_buffer(reference_scratch_buffer),
            reference_scratch_address,
            frame_slots,
        })
    }

    pub fn record_initial_builds(
        &mut self,
        device: &ash::Device,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        command_buffer: vk::CommandBuffer,
        current_frame_slot: usize,
        current_instances: &[vk::AccelerationStructureInstanceKHR],
    ) -> AnyResult<()> {
        let current_slot = self
            .frame_slots
            .get(current_frame_slot)
            .ok_or_else(|| anyhow!("RT page TLAS initial frame slot is out of range"))?;
        if current_instances.len() != current_slot.capacity as usize {
            bail!(
                "RT page TLAS initial BUILD requires fixed capacity instances: instances={} capacity={}",
                current_instances.len(),
                current_slot.capacity
            );
        }
        write_mapped_slice(
            &current_slot.instance_buffer,
            current_instances,
            "RT page initial TLAS instances",
        )?;
        let host_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&host_barrier),
                &[],
                &[],
            );
        }

        let dummy_geometry = dummy_blas_geometry(self.dummy_vertex_buffer.device_address(device)?);
        let reference_geometry =
            reference_blas_geometry(self.reference_aabb_buffer.device_address(device)?);
        record_single_geometry_build(
            acceleration_structure_loader,
            command_buffer,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            page_tlas_build_flags(),
            vk::BuildAccelerationStructureModeKHR::BUILD,
            vk::AccelerationStructureKHR::null(),
            self.dummy_blas.handle,
            self.dummy_scratch_address,
            std::slice::from_ref(&dummy_geometry),
            1,
        );
        record_single_geometry_build(
            acceleration_structure_loader,
            command_buffer,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            page_tlas_build_flags(),
            vk::BuildAccelerationStructureModeKHR::BUILD,
            vk::AccelerationStructureKHR::null(),
            self.reference_blas.handle,
            self.reference_scratch_address,
            std::slice::from_ref(&reference_geometry),
            1,
        );
        record_as_build_barrier(
            device,
            command_buffer,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        );

        for slot in &mut self.frame_slots {
            let geometry = tlas_geometry(slot.instance_buffer.device_address(device)?);
            record_single_geometry_build(
                acceleration_structure_loader,
                command_buffer,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                page_tlas_build_flags(),
                vk::BuildAccelerationStructureModeKHR::BUILD,
                vk::AccelerationStructureKHR::null(),
                slot.tlas.handle,
                slot.scratch_address,
                std::slice::from_ref(&geometry),
                slot.capacity,
            );
            slot.initialized = true;
        }
        record_as_build_barrier(
            device,
            command_buffer,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        );
        Ok(())
    }

    pub fn record_frame_slot_update(
        &self,
        device: &ash::Device,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        command_buffer: vk::CommandBuffer,
        profiler: Option<&GpuProfiler>,
        frame_slot: usize,
        instances: &[vk::AccelerationStructureInstanceKHR],
    ) -> AnyResult<()> {
        let slot = self
            .frame_slots
            .get(frame_slot)
            .ok_or_else(|| anyhow!("RT page TLAS frame slot out of range: {frame_slot}"))?;
        if !slot.initialized {
            bail!("RT page TLAS frame slot must be built before UPDATE");
        }
        if instances.len() != slot.capacity as usize {
            bail!(
                "RT page TLAS UPDATE requires fixed capacity instances: instances={} capacity={}",
                instances.len(),
                slot.capacity
            );
        }
        write_mapped_slice(&slot.instance_buffer, instances, "RT page TLAS instances")?;
        let host_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&host_barrier),
                &[],
                &[],
            );
        }
        let geometry = tlas_geometry(slot.instance_buffer.device_address(device)?);
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(page_tlas_build_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .src_acceleration_structure(slot.tlas.handle)
            .dst_acceleration_structure(slot.tlas.handle)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: slot.scratch_address,
            });
        let range =
            vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(slot.capacity);
        if let Some(profiler) = profiler {
            profiler.begin_scope(
                device,
                command_buffer,
                frame_slot,
                GpuProfileScope::RtTlasWork,
            );
        }
        unsafe {
            acceleration_structure_loader.cmd_build_acceleration_structures(
                command_buffer,
                std::slice::from_ref(&build_info),
                &[std::slice::from_ref(&range)],
            );
        }
        if let Some(profiler) = profiler {
            profiler.end_scope(
                device,
                command_buffer,
                frame_slot,
                GpuProfileScope::RtTlasWork,
            );
        }
        record_as_build_barrier(
            device,
            command_buffer,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        );
        Ok(())
    }

    pub fn tlas_handle(&self, frame_slot: usize) -> Option<vk::AccelerationStructureKHR> {
        self.frame_slots
            .get(frame_slot)
            .map(|slot| slot.tlas.handle)
    }

    pub fn dummy_blas_address(
        &self,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    ) -> vk::DeviceAddress {
        self.dummy_blas
            .device_address(acceleration_structure_loader)
    }

    pub fn reference_blas_address(
        &self,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    ) -> vk::DeviceAddress {
        self.reference_blas
            .device_address(acceleration_structure_loader)
    }

    pub fn destroy(
        self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    ) {
        for slot in self.frame_slots {
            slot.scratch_buffer.destroy(device, allocator);
            slot.tlas
                .destroy(device, allocator, acceleration_structure_loader);
            slot.instance_buffer.destroy(device, allocator);
        }
        self.reference_scratch_buffer.destroy(device, allocator);
        self.dummy_scratch_buffer.destroy(device, allocator);
        self.reference_blas
            .destroy(device, allocator, acceleration_structure_loader);
        self.dummy_blas
            .destroy(device, allocator, acceleration_structure_loader);
        self.reference_aabb_buffer.destroy(device, allocator);
        self.dummy_vertex_buffer.destroy(device, allocator);
    }
}

fn page_tlas_build_flags() -> vk::BuildAccelerationStructureFlagsKHR {
    // This TLAS is traced every frame (primary, shadow, GI) and updated in-place via UPDATE
    // mode. PREFER_FAST_TRACE optimizes for traversal throughput rather than build time.
    // Benchmark against PREFER_FAST_BUILD if build latency becomes a measured bottleneck.
    vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
        | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
}

fn fallback_input_buffer_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
}

fn instance_buffer_bytes(capacity: u32) -> AnyResult<u64> {
    u64::from(capacity)
        .checked_mul(std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64)
        .ok_or_else(|| anyhow!("RT page TLAS instance buffer size overflow"))
}

fn validate_as_input_address(
    address: vk::DeviceAddress,
    alignment: u64,
    label: &'static str,
) -> Result<(), RtPageTlasError> {
    if address == 0 {
        return Err(RtPageTlasError::NullBlasAddress);
    }
    if !address.is_multiple_of(alignment) {
        return Err(RtPageTlasError::MisalignedInputAddress {
            label,
            address,
            alignment,
        });
    }
    Ok(())
}

fn dummy_blas_geometry(
    address: vk::DeviceAddress,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: address,
        })
        .vertex_stride(std::mem::size_of::<[f32; 3]>() as u64)
        .max_vertex(2)
        .index_type(vk::IndexType::NONE_KHR);
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

fn reference_blas_geometry(
    address: vk::DeviceAddress,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: address,
        })
        .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as u64);
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::AABBS)
        .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

fn tlas_geometry(address: vk::DeviceAddress) -> vk::AccelerationStructureGeometryKHR<'static> {
    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: address,
        });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances })
}

fn query_build_sizes(
    loader: &ash::khr::acceleration_structure::Device,
    ty: vk::AccelerationStructureTypeKHR,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    geometries: &[vk::AccelerationStructureGeometryKHR<'_>],
    primitive_counts: &[u32],
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(flags)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geometries);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &info,
            primitive_counts,
            &mut sizes,
        );
    }
    sizes
}

#[allow(clippy::too_many_arguments)]
fn record_single_geometry_build(
    loader: &ash::khr::acceleration_structure::Device,
    command_buffer: vk::CommandBuffer,
    ty: vk::AccelerationStructureTypeKHR,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    mode: vk::BuildAccelerationStructureModeKHR,
    source: vk::AccelerationStructureKHR,
    destination: vk::AccelerationStructureKHR,
    scratch_address: vk::DeviceAddress,
    geometries: &[vk::AccelerationStructureGeometryKHR<'_>],
    primitive_count: u32,
) {
    let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(flags)
        .mode(mode)
        .src_acceleration_structure(source)
        .dst_acceleration_structure(destination)
        .geometries(geometries)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_address,
        });
    let range =
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);
    unsafe {
        loader.cmd_build_acceleration_structures(
            command_buffer,
            std::slice::from_ref(&info),
            &[std::slice::from_ref(&range)],
        );
    }
}

fn record_as_build_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    destination_stage: vk::PipelineStageFlags,
    destination_access: vk::AccessFlags,
) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(destination_access);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            destination_stage,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
    }
}

fn write_mapped_slice<T: Copy>(buffer: &GpuBuffer, values: &[T], label: &str) -> AnyResult<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    if bytes.len() as u64 > buffer.size {
        bail!("{label} exceeds mapped buffer size");
    }
    let mapped = buffer
        .mapped_ptr()
        .ok_or_else(|| anyhow!("{label} buffer must be host visible"))?;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped, bytes.len()) };
    Ok(())
}

struct RtPageTlasBuildGuard<'a> {
    device: &'a ash::Device,
    allocator: &'a GpuAllocator,
    loader: &'a ash::khr::acceleration_structure::Device,
    buffers: Vec<Option<GpuBuffer>>,
    acceleration_structures: Vec<Option<RtAccelerationStructure>>,
}

impl<'a> RtPageTlasBuildGuard<'a> {
    fn new(
        device: &'a ash::Device,
        allocator: &'a GpuAllocator,
        loader: &'a ash::khr::acceleration_structure::Device,
    ) -> Self {
        Self {
            device,
            allocator,
            loader,
            buffers: Vec::new(),
            acceleration_structures: Vec::new(),
        }
    }

    fn create_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> AnyResult<usize> {
        let buffer = GpuBuffer::new(self.device, self.allocator, size, usage, location, name)?;
        self.buffers.push(Some(buffer));
        Ok(self.buffers.len() - 1)
    }

    fn create_acceleration_structure(
        &mut self,
        ty: vk::AccelerationStructureTypeKHR,
        size: u64,
        name: &str,
    ) -> AnyResult<usize> {
        let acceleration_structure =
            RtAccelerationStructure::new(self.device, self.allocator, self.loader, ty, size, name)?;
        self.acceleration_structures
            .push(Some(acceleration_structure));
        Ok(self.acceleration_structures.len() - 1)
    }

    fn create_scratch(
        &mut self,
        required_size: u64,
        alignment: u64,
        name: &str,
    ) -> AnyResult<(usize, vk::DeviceAddress)> {
        if required_size == 0 || !alignment.is_power_of_two() {
            bail!("invalid RT page acceleration-structure scratch requirements");
        }
        let allocation_size = required_size
            .checked_add(alignment - 1)
            .ok_or_else(|| anyhow!("RT page scratch allocation size overflow"))?;
        let index = self.create_buffer(
            allocation_size,
            RtSceneAsBuildInputs::scratch_buffer_usage(),
            MemoryLocation::GpuOnly,
            name,
        )?;
        let base = self.buffer(index).device_address(self.device)?;
        let address = base
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .ok_or_else(|| anyhow!("RT page scratch address overflow"))?;
        Ok((index, address))
    }

    fn buffer(&self, index: usize) -> &GpuBuffer {
        self.buffers[index]
            .as_ref()
            .expect("buffer must still be owned")
    }

    fn acceleration_structure(&self, index: usize) -> &RtAccelerationStructure {
        self.acceleration_structures[index]
            .as_ref()
            .expect("acceleration structure must still be owned")
    }

    fn take_buffer(&mut self, index: usize) -> GpuBuffer {
        self.buffers[index]
            .take()
            .expect("buffer must still be owned")
    }

    fn take_acceleration_structure(&mut self, index: usize) -> RtAccelerationStructure {
        self.acceleration_structures[index]
            .take()
            .expect("acceleration structure must still be owned")
    }
}

impl Drop for RtPageTlasBuildGuard<'_> {
    fn drop(&mut self) {
        for acceleration_structure in self.acceleration_structures.drain(..).flatten().rev() {
            acceleration_structure.destroy(self.device, self.allocator, self.loader);
        }
        for buffer in self.buffers.drain(..).flatten().rev() {
            buffer.destroy(self.device, self.allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;
    use glam::UVec3;

    fn source(
        page: UVec3,
        topology_revision: u64,
    ) -> crate::render::rt_surface_mask::SurfaceMaskSourceStamp {
        use crate::render::rt_surface_mask::{
            SURFACE_MASK_DIRECTION_COUNT, SurfaceMaskDependencyStamp, SurfaceMaskSourceStamp,
        };
        SurfaceMaskSourceStamp {
            page,
            dependencies: [SurfaceMaskDependencyStamp {
                brick_id: 7,
                generation: 3,
                topology_revision,
            }; SURFACE_MASK_DIRECTION_COUNT + 1],
        }
    }

    #[test]
    fn dummy_slot_has_zero_mask_and_reference_uses_page_translation() {
        let dummy = make_page_tlas_instance(
            0,
            UVec3::ZERO,
            RtPageTlasInstanceBinding::Dummy {
                blas_address: 0x1000,
            },
        )
        .unwrap();
        assert_eq!(dummy.instance_custom_index_and_mask.low_24(), 0);
        assert_eq!(dummy.instance_custom_index_and_mask.high_8(), 0);

        let reference = make_page_tlas_instance(
            7,
            UVec3::new(2, 3, 4),
            RtPageTlasInstanceBinding::Reference {
                blas_address: 0x2000,
            },
        )
        .unwrap();
        assert_eq!(reference.transform.matrix[3], 16.0);
        assert_eq!(reference.transform.matrix[7], 24.0);
        assert_eq!(reference.transform.matrix[11], 32.0);
        assert_eq!(reference.instance_custom_index_and_mask.low_24(), 7);
        assert_eq!(reference.instance_custom_index_and_mask.high_8(), 0xff);
        assert_eq!(
            reference
                .instance_shader_binding_table_record_offset_and_flags
                .low_24(),
            RT_PAGE_REFERENCE_HIT_GROUP_OFFSET
        );
    }

    #[test]
    fn compact_exact_instance_selects_triangle_hit_group_and_blas_address() {
        let instance = make_page_tlas_instance(
            9,
            UVec3::ONE,
            RtPageTlasInstanceBinding::CompactExact {
                blas_address: 0x3000,
                resource_version: 44,
            },
        )
        .unwrap();

        assert_eq!(
            instance
                .instance_shader_binding_table_record_offset_and_flags
                .low_24(),
            RT_PAGE_COMPACT_EXACT_HIT_GROUP_OFFSET
        );
        assert_eq!(
            unsafe { instance.acceleration_structure_reference.device_handle },
            0x3000
        );
    }

    #[test]
    fn instance_slot_and_capacity_respect_both_24_bit_and_device_limits() {
        assert_eq!(
            make_page_tlas_instance(
                RT_PAGE_TLAS_MAX_INSTANCE_SLOTS,
                UVec3::ZERO,
                RtPageTlasInstanceBinding::Dummy {
                    blas_address: 0x1000,
                },
            )
            .err(),
            Some(RtPageTlasError::InstanceSlotOutOfRange {
                slot: RT_PAGE_TLAS_MAX_INSTANCE_SLOTS,
            })
        );
        assert_eq!(
            RtPageTlasCapacity::new(8, 4),
            Err(RtPageTlasError::CapacityExceedsLimit {
                requested: 8,
                limit: 4,
            })
        );
        assert_eq!(RtPageTlasCapacity::device_limit(u64::MAX), 0x00ff_ffff);
        assert_eq!(RtPageTlasCapacity::device_limit(1024), 1024);
    }

    #[test]
    fn blas_version_waits_for_every_referencing_frame_slot() {
        let mut references = RtPageTlasFrameReferences::new(3).unwrap();
        references.record_submission(0, 10, [41, 42]).unwrap();
        references.record_submission(1, 11, [42]).unwrap();
        references.record_submission(2, 12, []).unwrap();

        assert!(!references.can_retire(42));
        references.complete_submission(0, 10).unwrap();
        assert!(!references.can_retire(42));
        references.complete_submission(1, 11).unwrap();
        assert!(references.can_retire(42));
    }

    #[test]
    fn stale_frame_completion_cannot_release_newer_tlas_references() {
        let mut references = RtPageTlasFrameReferences::new(1).unwrap();
        references.record_submission(0, 20, [7]).unwrap();

        assert_eq!(
            references.complete_submission(0, 19),
            Err(RtPageTlasError::FrameGenerationMismatch {
                frame_slot: 0,
                expected: 20,
                actual: 19,
            })
        );
        assert!(!references.can_retire(7));
    }

    #[test]
    fn later_non_rt_epoch_completes_an_older_rt_slot_reference() {
        let mut references = RtPageTlasFrameReferences::new(1).unwrap();
        references.record_submission(0, 20, [7]).unwrap();

        references.complete_through(0, 23).unwrap();

        assert!(references.can_retire(7));
        assert_eq!(references.in_flight_generation(0), None);
    }

    #[test]
    fn capacity_growth_requires_quiescence_but_in_capacity_edits_do_not() {
        let capacity = RtPageTlasCapacity::new(8, 64).unwrap();
        assert_eq!(
            capacity.plan_for_required_slots(7, false).unwrap(),
            RtPageTlasCapacityPlan::Reuse
        );
        assert_eq!(
            capacity.plan_for_required_slots(9, false),
            Err(RtPageTlasError::CapacityGrowthRequiresQuiescence {
                current: 8,
                required: 9,
            })
        );
        assert_eq!(
            capacity.plan_for_required_slots(9, true).unwrap(),
            RtPageTlasCapacityPlan::Grow { new_capacity: 16 }
        );
    }

    #[test]
    fn instance_flags_force_opaque_without_changing_hit_group_offset() {
        let instance = make_page_tlas_instance(
            1,
            UVec3::ZERO,
            RtPageTlasInstanceBinding::Reference {
                blas_address: 0x1000,
            },
        )
        .unwrap();
        let packed = instance.instance_shader_binding_table_record_offset_and_flags;

        assert_eq!(packed.low_24(), RT_PAGE_REFERENCE_HIT_GROUP_OFFSET);
        assert_eq!(
            packed.high_8(),
            vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8
        );
    }

    #[test]
    fn page_tlas_gpu_resources_own_shared_fallbacks_and_one_tlas_per_frame_slot() {
        let source = crate::render::source_checks::read_source("src/render/rt_page_tlas.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("production TLAS source must precede tests");

        for token in [
            "pub struct RtPageTlasGpuResources",
            "dummy_blas: RtAccelerationStructure",
            "reference_blas: RtAccelerationStructure",
            "frame_slots: Vec<RtPageTlasFrameSlotGpuResources>",
            "AccelerationStructureGeometryAabbsDataKHR",
            "vk::AabbPositionsKHR",
            "RtAccelerationStructure::new",
            "MemoryLocation::GpuOnly",
            "ALLOW_UPDATE",
            "PREFER_FAST_BUILD",
        ] {
            assert!(
                production.contains(token),
                "frame-slot TLAS GPU path must contain {token}"
            );
        }
        assert!(!production.contains("wait_idle"));
    }

    #[test]
    fn frame_slot_tlas_update_is_profiled_and_published_before_trace() {
        let source = crate::render::source_checks::read_source("src/render/rt_page_tlas.rs");
        let record = source
            .split("pub fn record_frame_slot_update")
            .nth(1)
            .expect("frame-slot TLAS resources must record updates")
            .split("pub fn")
            .next()
            .expect("frame-slot TLAS update method must have a bounded body");
        let compact = crate::render::source_checks::compact(record);

        for token in [
            "BuildAccelerationStructureModeKHR::UPDATE",
            "GpuProfileScope::RtTlasWork",
            "cmd_build_acceleration_structures",
            "cmd_pipeline_barrier",
            "PipelineStageFlags::RAY_TRACING_SHADER_KHR",
            "AccessFlags::ACCELERATION_STRUCTURE_READ_KHR",
        ] {
            assert!(compact.contains(token), "TLAS update must contain {token}");
        }
        let build = compact.find("cmd_build_acceleration_structures").unwrap();
        let barrier = compact[build..]
            .find("record_as_build_barrier")
            .map(|offset| build + offset)
            .expect("TLAS build must be followed by a trace-read barrier");
        assert!(build < barrier);
    }

    #[test]
    fn initial_tlas_builds_publish_to_future_slot_updates_and_trace() {
        let source = crate::render::source_checks::read_source("src/render/rt_page_tlas.rs");
        let initial = source
            .split("pub fn record_initial_builds")
            .nth(1)
            .expect("initial page TLAS build method must exist")
            .split("pub fn record_frame_slot_update")
            .next()
            .expect("initial builds must end before update recording");
        let compact = crate::render::source_checks::compact(initial);

        assert!(compact.contains(
            "PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR|vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR"
        ));
        // dst_access_mask must be READ-only: the barrier makes the completed TLAS builds
        // visible for subsequent UPDATE operations and ray-tracing reads.  Including
        // ACCELERATION_STRUCTURE_WRITE in dst_access was overly broad and has been removed.
        assert!(compact.contains(
            "AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,"
        ));
        assert!(!compact.contains(
            "AccessFlags::ACCELERATION_STRUCTURE_READ_KHR|vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR"
        ), "dst_access_mask must not include WRITE — it over-broadens the barrier and masks hazards");
    }

    #[test]
    fn registry_invalidation_replaces_resident_instance_with_reference_before_trace() {
        use crate::render::rt_page_registry::{RtPageRegistry, RtPageRepresentation};

        let page = UVec3::new(2, 1, 3);
        let mut registry = RtPageRegistry::new(8);
        let slot = registry.ensure_reference_page(page).expect("test slot allocation");
        registry.invalidate_topology(page, 1);
        let source = source(page, 10);
        let ticket = registry
            .begin_build(page, RtPageRepresentation::CompactExact, source)
            .unwrap();
        registry.install_build(ticket, source, 41, 2).unwrap();
        let mut store = RtPageTlasInstanceStore::new(8, 0x1000, 0x2000).unwrap();
        store
            .sync_registry_slot(
                &registry,
                slot,
                Some(RtPageCompactTlasBinding {
                    resource_version: 41,
                    blas_address: 0x3000,
                }),
            )
            .unwrap();
        assert_eq!(store.resource_versions().collect::<Vec<_>>(), vec![41]);

        registry.invalidate_topology(page, 3);
        store.sync_registry_slot(&registry, slot, None).unwrap();

        let instance = store.instance(slot).unwrap();
        assert_eq!(
            unsafe { instance.acceleration_structure_reference.device_handle },
            0x2000
        );
        assert_eq!(store.resource_versions().count(), 0);
    }

    #[test]
    fn instance_store_keeps_unallocated_capacity_mask_zero() {
        let store = RtPageTlasInstanceStore::new(4, 0x1000, 0x2000).unwrap();

        assert_eq!(store.instances().len(), 4);
        assert!(
            store
                .instances()
                .iter()
                .all(|instance| instance.instance_custom_index_and_mask.high_8() == 0)
        );
    }

    #[test]
    fn acceleration_structure_inputs_reject_misaligned_device_addresses() {
        assert_eq!(
            validate_as_input_address(0x1004, 16, "instance"),
            Err(RtPageTlasError::MisalignedInputAddress {
                label: "instance",
                address: 0x1004,
                alignment: 16,
            })
        );
        assert!(validate_as_input_address(0x1010, 16, "instance").is_ok());
    }
}
