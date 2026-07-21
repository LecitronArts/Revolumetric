use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::rt_page_geometry::{RT_PAGE_LATTICE_VERTEX_COUNT, RtCompactPageGeometry};
use crate::render::rt_page_gpu::{RtPageGeometryAllocation, RtPageGpuResources};
use crate::render::rt_page_registry::{
    RtPageBuildGeneration, RtPageBuildTicket, RtPageInstallError, RtPageRegistry,
    RtPageRepresentation, RtPageResidentVersion,
};
use crate::render::rt_representation_metrics::{RtRepresentationKind, RtRepresentationMetrics};
use crate::render::rt_scene::{RtAccelerationStructure, RtSceneAsBuildInputs};
use crate::render::rt_surface_mask::SurfaceMaskSourceStamp;
use anyhow::{Context, Result, bail};
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::collections::VecDeque;
use thiserror::Error;

pub const RT_COMPACT_BLAS_VERTEX_STRIDE: u64 = 12;
pub const RT_COMPACT_BLAS_MAX_VERTEX: u32 = (RT_PAGE_LATTICE_VERTEX_COUNT - 1) as u32;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RtCompactBlasPlanError {
    #[error("CompactExact BLAS geometry has indices without faces or faces without indices")]
    PartialEmptyGeometry,
    #[error("CompactExact BLAS index count is not divisible by three: {index_count}")]
    IndexCountNotTriangulated { index_count: usize },
    #[error(
        "CompactExact BLAS primitive count does not match two triangles per face: primitives={primitive_count} faces={face_count}"
    )]
    PrimitiveFaceMismatch {
        primitive_count: usize,
        face_count: usize,
    },
    #[error("CompactExact BLAS primitive count exceeds Vulkan u32 range: {primitive_count}")]
    PrimitiveCountOverflow { primitive_count: usize },
    #[error("CompactExact BLAS contains an out-of-range lattice index: {index}")]
    LatticeIndexOutOfRange { index: u16 },
    #[error("CompactExact BLAS {label} device address is null")]
    NullDeviceAddress { label: &'static str },
    #[error(
        "CompactExact BLAS {label} device address is misaligned: address={address:#x} alignment={alignment}"
    )]
    MisalignedDeviceAddress {
        label: &'static str,
        address: u64,
        alignment: u64,
    },
}

#[derive(Debug, Clone)]
pub struct RtCompactBlasBuildPlan {
    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR<'static>,
    geometries: [vk::AccelerationStructureGeometryKHR<'static>; 1],
    ranges: [vk::AccelerationStructureBuildRangeInfoKHR; 1],
    primitive_count: u32,
}

impl RtCompactBlasBuildPlan {
    pub fn from_geometry(
        geometry: &RtCompactPageGeometry,
        lattice_device_address: vk::DeviceAddress,
        index_device_address: vk::DeviceAddress,
    ) -> Result<Option<Self>, RtCompactBlasPlanError> {
        if geometry.indices.is_empty() && geometry.faces.is_empty() {
            return Ok(None);
        }
        if geometry.indices.is_empty() || geometry.faces.is_empty() {
            return Err(RtCompactBlasPlanError::PartialEmptyGeometry);
        }
        if !geometry.indices.len().is_multiple_of(3) {
            return Err(RtCompactBlasPlanError::IndexCountNotTriangulated {
                index_count: geometry.indices.len(),
            });
        }
        let primitive_count = geometry.indices.len() / 3;
        if primitive_count != geometry.faces.len().saturating_mul(2) {
            return Err(RtCompactBlasPlanError::PrimitiveFaceMismatch {
                primitive_count,
                face_count: geometry.faces.len(),
            });
        }
        let primitive_count = u32::try_from(primitive_count).map_err(|_| {
            RtCompactBlasPlanError::PrimitiveCountOverflow {
                primitive_count: geometry.indices.len() / 3,
            }
        })?;
        if let Some(&index) = geometry
            .indices
            .iter()
            .find(|&&index| usize::from(index) >= RT_PAGE_LATTICE_VERTEX_COUNT)
        {
            return Err(RtCompactBlasPlanError::LatticeIndexOutOfRange { index });
        }
        validate_device_address(lattice_device_address, 4, "lattice")?;
        validate_device_address(index_device_address, 2, "index")?;

        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: lattice_device_address,
            })
            .vertex_stride(RT_COMPACT_BLAS_VERTEX_STRIDE)
            .max_vertex(RT_COMPACT_BLAS_MAX_VERTEX)
            .index_type(vk::IndexType::UINT16)
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: index_device_address,
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
            .flags(vk::GeometryFlagsKHR::OPAQUE);
        let range =
            vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);
        Ok(Some(Self {
            triangles,
            geometries: [geometry],
            ranges: [range],
            primitive_count,
        }))
    }

    pub fn triangles(&self) -> vk::AccelerationStructureGeometryTrianglesDataKHR<'static> {
        self.triangles
    }

    pub fn geometry(&self) -> &vk::AccelerationStructureGeometryKHR<'static> {
        &self.geometries[0]
    }

    pub fn primitive_count(&self) -> u32 {
        self.primitive_count
    }

    pub fn max_primitive_counts(&self) -> [u32; 1] {
        [self.primitive_count]
    }

    pub fn build_range(&self) -> vk::AccelerationStructureBuildRangeInfoKHR {
        self.ranges[0]
    }

    pub fn size_query_info(&self) -> vk::AccelerationStructureBuildGeometryInfoKHR<'_> {
        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(compact_blas_build_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&self.geometries)
    }

    pub fn build_info(
        &self,
        destination: vk::AccelerationStructureKHR,
        scratch_address: vk::DeviceAddress,
    ) -> vk::AccelerationStructureBuildGeometryInfoKHR<'_> {
        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(compact_blas_build_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(destination)
            .geometries(&self.geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            })
    }

    pub fn build_ranges(&self) -> &[vk::AccelerationStructureBuildRangeInfoKHR] {
        &self.ranges
    }
}

pub const fn compact_blas_build_flags() -> vk::BuildAccelerationStructureFlagsKHR {
    // CompactExact page BLASes are build-once/trace-many: a page is rebuilt only on
    // a topology edit (always an out-of-place BUILD, never an in-place UPDATE — see
    // `compact_blas_build_is_out_of_place_fast_trace_only`), then traced for many
    // frames by primary, shadow, and GI rays. PREFER_FAST_TRACE optimizes traversal
    // throughput (deeper, higher-SAH-quality BVH) at the cost of slower builds, which
    // is the correct trade for near-static voxel surfaces. ALLOW_UPDATE is
    // deliberately omitted because we never refit these BLASes.
    //
    // NVIDIA "Best Practices for Using NVIDIA RTX Ray Tracing": reserve
    // PREFER_FAST_BUILD for geometry rebuilt every frame.
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtCompactBlasBarrier {
    pub src_stage: vk::PipelineStageFlags,
    pub dst_stage: vk::PipelineStageFlags,
    pub src_access: vk::AccessFlags,
    pub dst_access: vk::AccessFlags,
}

pub fn compact_blas_to_tlas_barrier() -> RtCompactBlasBarrier {
    RtCompactBlasBarrier {
        src_stage: vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        dst_stage: vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        src_access: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
        dst_access: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RtCompactBlasScratchError {
    #[error("CompactExact BLAS scratch size must be nonzero")]
    EmptyScratch,
    #[error("CompactExact BLAS scratch alignment must be a nonzero power of two: {alignment}")]
    InvalidAlignment { alignment: u64 },
    #[error("CompactExact BLAS scratch allocation arithmetic overflow")]
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtCompactBlasScratchLayout {
    pub required_bytes: u64,
    pub alignment: u64,
    pub allocation_bytes: u64,
}

impl RtCompactBlasScratchLayout {
    pub fn new(required_bytes: u64, alignment: u64) -> Result<Self, RtCompactBlasScratchError> {
        if required_bytes == 0 {
            return Err(RtCompactBlasScratchError::EmptyScratch);
        }
        if !alignment.is_power_of_two() {
            return Err(RtCompactBlasScratchError::InvalidAlignment { alignment });
        }
        let allocation_bytes = required_bytes
            .checked_add(alignment - 1)
            .ok_or(RtCompactBlasScratchError::ArithmeticOverflow)?;
        Ok(Self {
            required_bytes,
            alignment,
            allocation_bytes,
        })
    }

    pub fn align_address(
        self,
        address: vk::DeviceAddress,
    ) -> Result<vk::DeviceAddress, RtCompactBlasScratchError> {
        let mask = self.alignment - 1;
        address
            .checked_add(mask)
            .map(|value| value & !mask)
            .ok_or(RtCompactBlasScratchError::ArithmeticOverflow)
    }
}

pub fn compact_blas_metrics(
    face_count: u32,
    sizes: &vk::AccelerationStructureBuildSizesInfoKHR<'_>,
) -> RtRepresentationMetrics {
    let mut metrics = RtRepresentationMetrics::empty(RtRepresentationKind::CompactExact);
    metrics.page_count = 1;
    metrics.exposed_face_count = u64::from(face_count);
    metrics.candidate_primitive_count = u64::from(face_count) * 2;
    metrics.memory.persistent_bytes = sizes.acceleration_structure_size;
    metrics.memory.scratch_bytes = sizes.build_scratch_size;
    metrics
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtCompactBlasBuildState {
    Allocated,
    Recorded,
    Submitted,
    Installed,
    RetirePending,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid CompactExact BLAS lifecycle transition: {from:?} -> {to:?}")]
pub struct RtCompactBlasLifecycleError {
    from: RtCompactBlasBuildState,
    to: RtCompactBlasBuildState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtCompactBlasBuildLifecycle {
    state: RtCompactBlasBuildState,
}

impl RtCompactBlasBuildLifecycle {
    pub fn new() -> Self {
        Self {
            state: RtCompactBlasBuildState::Allocated,
        }
    }

    pub fn state(self) -> RtCompactBlasBuildState {
        self.state
    }

    pub fn installable(self) -> bool {
        self.state == RtCompactBlasBuildState::Submitted
    }

    pub fn mark_recorded(&mut self) -> Result<(), RtCompactBlasLifecycleError> {
        self.transition(
            RtCompactBlasBuildState::Allocated,
            RtCompactBlasBuildState::Recorded,
        )
    }

    pub fn mark_submitted(&mut self) -> Result<(), RtCompactBlasLifecycleError> {
        self.transition(
            RtCompactBlasBuildState::Recorded,
            RtCompactBlasBuildState::Submitted,
        )
    }

    pub fn mark_installed(&mut self) -> Result<(), RtCompactBlasLifecycleError> {
        self.transition(
            RtCompactBlasBuildState::Submitted,
            RtCompactBlasBuildState::Installed,
        )
    }

    pub fn mark_retire_pending(&mut self) -> Result<(), RtCompactBlasLifecycleError> {
        // RetirePending is reachable from any pre-retire state.  Calling it twice is a
        // caller bug (double-enqueue), so we validate the transition explicitly rather than
        // silently no-op.
        if self.state == RtCompactBlasBuildState::RetirePending {
            return Err(RtCompactBlasLifecycleError {
                from: self.state,
                to: RtCompactBlasBuildState::RetirePending,
            });
        }
        self.state = RtCompactBlasBuildState::RetirePending;
        Ok(())
    }

    fn transition(
        &mut self,
        expected: RtCompactBlasBuildState,
        next: RtCompactBlasBuildState,
    ) -> Result<(), RtCompactBlasLifecycleError> {
        if self.state != expected {
            return Err(RtCompactBlasLifecycleError {
                from: self.state,
                to: next,
            });
        }
        self.state = next;
        Ok(())
    }
}

impl Default for RtCompactBlasBuildLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RtCompactBlasCreateInfo<'a> {
    pub page_gpu: &'a RtPageGpuResources,
    pub geometry_allocation: RtPageGeometryAllocation,
    pub geometry: &'a RtCompactPageGeometry,
    pub source: SurfaceMaskSourceStamp,
    pub build_generation: RtPageBuildGeneration,
    pub resource_version: u64,
    pub min_acceleration_structure_scratch_offset_alignment: vk::DeviceSize,
}

pub struct RtCompactBlasBuildResources {
    pub geometry_allocation: RtPageGeometryAllocation,
    pub source: SurfaceMaskSourceStamp,
    pub build_generation: RtPageBuildGeneration,
    pub resource_version: u64,
    /// Frame epoch of the last command-buffer submission that references this BLAS.
    /// Set by `RtCompactBlasRetirementQueue::enqueue`; `drain_completed` uses it to
    /// ensure the GPU has finished before `destroy` is called.
    pub retire_epoch: Option<u64>,
    blas: RtAccelerationStructure,
    blas_device_address: vk::DeviceAddress,
    /// Build scratch. Consumed only by the one BLAS build command recorded in
    /// `record_build`; dead once that submission's fence signals. `take_scratch`
    /// removes it after submission so an installed (resident) BLAS no longer pins
    /// scratch VRAM for its whole lifetime (previously ~doubled per-page AS memory).
    /// Retire/error paths that still hold scratch free it in `destroy`.
    scratch_buffer: Option<GpuBuffer>,
    scratch_address: vk::DeviceAddress,
    plan: RtCompactBlasBuildPlan,
    build_sizes: vk::AccelerationStructureBuildSizesInfoKHR<'static>,
    metrics: RtRepresentationMetrics,
    lifecycle: RtCompactBlasBuildLifecycle,
}

pub struct RtInstalledCompactBlas {
    pub resident: RtPageResidentVersion,
    pub resources: RtCompactBlasBuildResources,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RtCompactBlasInstallError {
    #[error("CompactExact BLAS cannot install from lifecycle state {state:?}")]
    NotSubmitted { state: RtCompactBlasBuildState },
    #[error("CompactExact BLAS resource identity does not match its registry build ticket")]
    ResourceTicketMismatch,
    #[error("CompactExact BLAS registry install was rejected: {0:?}")]
    Registry(RtPageInstallError),
}

impl RtCompactBlasBuildResources {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        create_info: RtCompactBlasCreateInfo<'_>,
    ) -> Result<Option<Self>> {
        let lattice_address = create_info
            .page_gpu
            .lattice_device_address(device)
            .context("failed to query CompactExact lattice device address")?;
        let index_address = create_info
            .page_gpu
            .index_device_address(device, create_info.geometry_allocation)
            .context("failed to query CompactExact index device address")?;
        let Some(plan) = RtCompactBlasBuildPlan::from_geometry(
            create_info.geometry,
            lattice_address,
            index_address,
        )
        .map_err(anyhow::Error::new)?
        else {
            return Ok(None);
        };

        let build_sizes = query_compact_blas_build_sizes(acceleration_structure_loader, &plan);
        let blas = RtAccelerationStructure::new(
            device,
            allocator,
            acceleration_structure_loader,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            build_sizes.acceleration_structure_size,
            "rt_compact_exact_blas",
        )?;
        let scratch_layout = match RtCompactBlasScratchLayout::new(
            build_sizes.build_scratch_size,
            create_info.min_acceleration_structure_scratch_offset_alignment,
        ) {
            Ok(layout) => layout,
            Err(error) => {
                blas.destroy(device, allocator, acceleration_structure_loader);
                return Err(anyhow::Error::new(error));
            }
        };
        let scratch_buffer = match GpuBuffer::new(
            device,
            allocator,
            scratch_layout.allocation_bytes,
            RtSceneAsBuildInputs::scratch_buffer_usage(),
            MemoryLocation::GpuOnly,
            "rt_compact_exact_blas_scratch",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                blas.destroy(device, allocator, acceleration_structure_loader);
                return Err(error);
            }
        };
        let scratch_address = match scratch_buffer.device_address(device).and_then(|address| {
            scratch_layout
                .align_address(address)
                .map_err(anyhow::Error::new)
        }) {
            Ok(address) => address,
            Err(error) => {
                scratch_buffer.destroy(device, allocator);
                blas.destroy(device, allocator, acceleration_structure_loader);
                return Err(error).context("failed to align CompactExact BLAS scratch address");
            }
        };
        let blas_device_address = blas.device_address(acceleration_structure_loader);
        if blas_device_address == 0 {
            scratch_buffer.destroy(device, allocator);
            blas.destroy(device, allocator, acceleration_structure_loader);
            bail!("Vulkan returned a null CompactExact BLAS device address");
        }
        let metrics = compact_blas_metrics(create_info.geometry.faces.len() as u32, &build_sizes);

        Ok(Some(Self {
            geometry_allocation: create_info.geometry_allocation,
            source: create_info.source,
            build_generation: create_info.build_generation,
            resource_version: create_info.resource_version,
            retire_epoch: None,
            blas,
            blas_device_address,
            scratch_buffer: Some(scratch_buffer),
            scratch_address,
            plan,
            build_sizes,
            metrics,
            lifecycle: RtCompactBlasBuildLifecycle::new(),
        }))
    }

    pub fn record_build(
        &mut self,
        device: &ash::Device,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        command_buffer: vk::CommandBuffer,
        profiler: Option<&GpuProfiler>,
        frame_slot: usize,
    ) -> Result<()> {
        if self.lifecycle.state() != RtCompactBlasBuildState::Allocated {
            return Err(anyhow::Error::new(RtCompactBlasLifecycleError {
                from: self.lifecycle.state(),
                to: RtCompactBlasBuildState::Recorded,
            }));
        }

        let build_info = self.plan.build_info(self.blas.handle, self.scratch_address);
        let build_range = self.plan.build_range();
        if let Some(profiler) = profiler {
            profiler.begin_scope(
                device,
                command_buffer,
                frame_slot,
                GpuProfileScope::RtBlasWork,
            );
        }
        unsafe {
            acceleration_structure_loader.cmd_build_acceleration_structures(
                command_buffer,
                std::slice::from_ref(&build_info),
                &[std::slice::from_ref(&build_range)],
            );
        }
        if let Some(profiler) = profiler {
            profiler.end_scope(
                device,
                command_buffer,
                frame_slot,
                GpuProfileScope::RtBlasWork,
            );
        }

        let barrier = compact_blas_to_tlas_barrier();
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(barrier.src_access)
            .dst_access_mask(barrier.dst_access);
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                barrier.src_stage,
                barrier.dst_stage,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&memory_barrier),
                &[],
                &[],
            );
        }
        self.lifecycle.mark_recorded().map_err(anyhow::Error::new)
    }

    pub fn mark_submitted(&mut self) -> Result<()> {
        self.lifecycle.mark_submitted().map_err(anyhow::Error::new)
    }

    pub fn mark_installed(&mut self) -> Result<()> {
        self.lifecycle.mark_installed().map_err(anyhow::Error::new)
    }

    pub fn installable(&self) -> bool {
        self.lifecycle.installable()
    }

    pub fn state(&self) -> RtCompactBlasBuildState {
        self.lifecycle.state()
    }

    pub fn blas_handle(&self) -> vk::AccelerationStructureKHR {
        self.blas.handle
    }

    pub fn blas_device_address(&self) -> vk::DeviceAddress {
        self.blas_device_address
    }

    pub fn build_sizes(&self) -> &vk::AccelerationStructureBuildSizesInfoKHR<'static> {
        &self.build_sizes
    }

    pub fn metrics(&self) -> RtRepresentationMetrics {
        self.metrics
    }

    pub fn install_or_retire(
        mut self,
        registry: &mut RtPageRegistry,
        ticket: RtPageBuildTicket,
        current_source: SurfaceMaskSourceStamp,
        frame_index: u64,
        retirement_queue: &mut RtCompactBlasRetirementQueue,
    ) -> std::result::Result<RtInstalledCompactBlas, RtCompactBlasInstallError> {
        if let Err(error) =
            validate_compact_blas_install_ticket(self.build_generation, self.source, ticket)
        {
            // Enqueue with current frame_index as a conservative retire epoch: we may not
            // know exactly when this BLAS was last referenced by a submitted command buffer,
            // so waiting until frame_index completes is always safe.
            retirement_queue
                .enqueue(self, frame_index)
                .expect("ticket-mismatch BLAS retirement enqueue must not double-enqueue");
            return Err(error);
        }
        if self.lifecycle.state() != RtCompactBlasBuildState::Submitted {
            let error = RtCompactBlasInstallError::NotSubmitted {
                state: self.lifecycle.state(),
            };
            registry.fail_build(ticket, frame_index);
            retirement_queue
                .enqueue(self, frame_index)
                .expect("not-submitted BLAS retirement enqueue must not double-enqueue");
            return Err(error);
        }
        match registry.install_build(ticket, current_source, self.resource_version, frame_index) {
            Ok(resident) => {
                self.lifecycle
                    .mark_installed()
                    .expect("submitted CompactExact BLAS must transition to installed");
                Ok(RtInstalledCompactBlas {
                    resident,
                    resources: self,
                })
            }
            Err(error) => {
                registry.fail_build(ticket, frame_index);
                retirement_queue
                    .enqueue(self, frame_index)
                    .expect("registry-error BLAS retirement enqueue must not double-enqueue");
                Err(RtCompactBlasInstallError::Registry(error))
            }
        }
    }

    /// Take ownership of the build scratch buffer, leaving `None` behind. Call after
    /// the BLAS build has been recorded and submitted — the scratch is only read by
    /// that one build command, so once its submission fence signals the buffer can be
    /// freed. Returns `None` if scratch was already taken (idempotent).
    pub fn take_scratch(&mut self) -> Option<GpuBuffer> {
        self.scratch_buffer.take()
    }

    pub fn destroy(
        self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        page_gpu: &mut RtPageGpuResources,
    ) -> Result<()> {
        if let Some(scratch_buffer) = self.scratch_buffer {
            scratch_buffer.destroy(device, allocator);
        }
        self.blas
            .destroy(device, allocator, acceleration_structure_loader);
        page_gpu
            .free_geometry(self.geometry_allocation)
            .map_err(anyhow::Error::new)
    }
}

fn validate_compact_blas_install_ticket(
    build_generation: RtPageBuildGeneration,
    source: SurfaceMaskSourceStamp,
    ticket: RtPageBuildTicket,
) -> std::result::Result<(), RtCompactBlasInstallError> {
    if ticket.target != RtPageRepresentation::CompactExact
        || build_generation != ticket.generation
        || source != ticket.source
    {
        return Err(RtCompactBlasInstallError::ResourceTicketMismatch);
    }
    Ok(())
}

#[derive(Default)]
pub struct RtCompactBlasRetirementQueue {
    pending: VecDeque<RtCompactBlasBuildResources>,
}

impl RtCompactBlasRetirementQueue {
    /// Enqueue a BLAS for retirement after the GPU completes `last_ref_epoch`.
    ///
    /// `last_ref_epoch` should be the submission epoch of the last command buffer that
    /// references this BLAS (e.g., a TLAS build/update that reads its device address).
    /// The BLAS will not be destroyed until `drain_completed(epoch)` is called with
    /// `epoch >= last_ref_epoch`, ensuring the GPU has finished using it.
    pub fn enqueue(
        &mut self,
        mut resources: RtCompactBlasBuildResources,
        last_ref_epoch: u64,
    ) -> Result<()> {
        resources
            .lifecycle
            .mark_retire_pending()
            .map_err(anyhow::Error::new)?;
        resources.retire_epoch = Some(last_ref_epoch);
        self.pending.push_back(resources);
        Ok(())
    }

    /// Drain and destroy all retired BLAS resources whose `retire_epoch <= completed_epoch`.
    ///
    /// This ensures that only BLAS resources no longer referenced by any in-flight GPU
    /// work are destroyed.  Call this after advancing the completed epoch (e.g., after
    /// a frame fence signals).
    pub fn drain_completed(
        &mut self,
        completed_epoch: u64,
        device: &ash::Device,
        allocator: &GpuAllocator,
        acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
        page_gpu: &mut RtPageGpuResources,
    ) -> Result<usize> {
        let mut destroyed_count = 0;
        while let Some(resources) = self.pending.front() {
            let retire_epoch = resources
                .retire_epoch
                .expect("enqueued BLAS must have a retire epoch");
            if retire_epoch > completed_epoch {
                break; // Queue is ordered by epoch, so we're done
            }
            let resources = self.pending.pop_front().unwrap();
            resources.destroy(device, allocator, acceleration_structure_loader, page_gpu)?;
            destroyed_count += 1;
        }
        Ok(destroyed_count)
    }

    /// Pop the front of the queue without epoch gating.
    ///
    /// **UNSAFE for production use** — the caller must ensure the GPU has finished with
    /// the BLAS before calling `destroy()`.  Prefer `drain_completed()` for correct
    /// frame-safety.  This is kept for backward compatibility with tests.
    #[deprecated(note = "use drain_completed(epoch, ...) for frame-safe retirement")]
    pub fn pop_front(&mut self) -> Option<RtCompactBlasBuildResources> {
        self.pending.pop_front()
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

/// Fence-gated free-queue for BLAS build scratch buffers (T1-E).
///
/// A CompactExact BLAS's build scratch is consumed only by the single
/// `cmd_build_acceleration_structures` recorded in `record_build`. Once that
/// submission's fence signals, the scratch is dead — but the BLAS itself remains
/// resident (traced) for potentially hundreds of frames. Previously the scratch was
/// owned for the BLAS's whole life, roughly doubling per-page AS VRAM on populated
/// scenes. This queue holds taken scratch buffers with the frame epoch that last
/// referenced them and frees each once `drain_completed(epoch)` confirms the GPU is
/// done — the same fence-safety model as [`RtCompactBlasRetirementQueue`].
#[derive(Default)]
pub struct RtCompactBlasScratchFreeQueue {
    pending: VecDeque<(GpuBuffer, u64)>,
}

impl RtCompactBlasScratchFreeQueue {
    /// Enqueue a scratch buffer to free after the GPU completes `last_ref_epoch`
    /// (the frame index whose command buffer recorded the build that read it).
    pub fn enqueue(&mut self, scratch: GpuBuffer, last_ref_epoch: u64) {
        self.pending.push_back((scratch, last_ref_epoch));
    }

    /// Free all scratch buffers whose recorded epoch has completed on the GPU.
    /// Call after advancing the completed epoch (e.g., after a frame fence signals).
    pub fn drain_completed(
        &mut self,
        completed_epoch: u64,
        device: &ash::Device,
        allocator: &GpuAllocator,
    ) -> usize {
        let mut freed = 0;
        while let Some((_, epoch)) = self.pending.front() {
            if *epoch > completed_epoch {
                break; // ordered by enqueue epoch (monotonic frame_index)
            }
            let (scratch, _) = self.pending.pop_front().unwrap();
            scratch.destroy(device, allocator);
            freed += 1;
        }
        freed
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

fn query_compact_blas_build_sizes(
    acceleration_structure_loader: &ash::khr::acceleration_structure::Device,
    plan: &RtCompactBlasBuildPlan,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let build_info = plan.size_query_info();
    let max_primitive_counts = plan.max_primitive_counts();
    let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        acceleration_structure_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &max_primitive_counts,
            &mut build_sizes,
        );
    }
    build_sizes
}

fn validate_device_address(
    address: vk::DeviceAddress,
    alignment: u64,
    label: &'static str,
) -> Result<(), RtCompactBlasPlanError> {
    if address == 0 {
        return Err(RtCompactBlasPlanError::NullDeviceAddress { label });
    }
    if !address.is_multiple_of(alignment) {
        return Err(RtCompactBlasPlanError::MisalignedDeviceAddress {
            label,
            address,
            alignment,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::rt_page_geometry::{RtCompactFaceRecord, RtCompactPageGeometry};
    use crate::render::rt_surface_mask::{
        FaceDirection, SURFACE_MASK_DIRECTION_COUNT, SurfaceMaskDependencyStamp,
    };
    use ash::vk;
    use ash::vk::Handle;
    use glam::UVec3;

    fn one_face_geometry() -> RtCompactPageGeometry {
        RtCompactPageGeometry {
            indices: vec![0, 1, 10, 0, 10, 9],
            faces: vec![RtCompactFaceRecord::new(
                UVec3::ZERO,
                FaceDirection::NegativeZ,
            )],
        }
    }

    fn source(page: UVec3, topology_revision: u64) -> SurfaceMaskSourceStamp {
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
    fn compact_blas_install_identity_rejects_a_crossed_ticket() {
        let page = UVec3::new(1, 2, 3);
        let resource_source = source(page, 10);
        let crossed_ticket = RtPageBuildTicket {
            slot: 4,
            page,
            target: RtPageRepresentation::CompactExact,
            generation: 12,
            source: source(page, 11),
        };

        assert_eq!(
            validate_compact_blas_install_ticket(11, resource_source, crossed_ticket),
            Err(RtCompactBlasInstallError::ResourceTicketMismatch)
        );
    }

    #[test]
    fn empty_geometry_skips_blas_creation() {
        let geometry = RtCompactPageGeometry::default();

        assert!(
            RtCompactBlasBuildPlan::from_geometry(&geometry, 0x1000, 0x2000)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn compact_blas_plan_uses_shared_lattice_and_uint16_indices() {
        let geometry = one_face_geometry();
        let plan = RtCompactBlasBuildPlan::from_geometry(&geometry, 0x1000, 0x2000)
            .unwrap()
            .unwrap();
        let triangles = plan.triangles();

        assert_eq!(triangles.vertex_format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(unsafe { triangles.vertex_data.device_address }, 0x1000);
        assert_eq!(triangles.vertex_stride, 12);
        assert_eq!(triangles.max_vertex, 728);
        assert_eq!(triangles.index_type, vk::IndexType::UINT16);
        assert_eq!(unsafe { triangles.index_data.device_address }, 0x2000);
        assert_eq!(plan.primitive_count(), 2);
        assert_eq!(plan.max_primitive_counts(), [2]);
        assert_eq!(plan.build_range().primitive_count, 2);
        assert_eq!(
            plan.geometry().geometry_type,
            vk::GeometryTypeKHR::TRIANGLES
        );
        assert_eq!(plan.geometry().flags, vk::GeometryFlagsKHR::OPAQUE);
    }

    #[test]
    fn compact_blas_plan_requires_two_triangles_per_face() {
        let mut geometry = one_face_geometry();
        geometry.indices.pop();

        assert!(matches!(
            RtCompactBlasBuildPlan::from_geometry(&geometry, 0x1000, 0x2000),
            Err(RtCompactBlasPlanError::IndexCountNotTriangulated { .. })
                | Err(RtCompactBlasPlanError::PrimitiveFaceMismatch { .. })
        ));
    }

    #[test]
    fn compact_blas_build_is_out_of_place_fast_trace_only() {
        let geometry = one_face_geometry();
        let plan = RtCompactBlasBuildPlan::from_geometry(&geometry, 0x1000, 0x2000)
            .unwrap()
            .unwrap();
        let destination = vk::AccelerationStructureKHR::from_raw(7);
        let info = plan.build_info(destination, 0x4000);

        assert_eq!(info.ty, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        assert_eq!(info.mode, vk::BuildAccelerationStructureModeKHR::BUILD);
        assert_eq!(
            info.src_acceleration_structure,
            vk::AccelerationStructureKHR::null()
        );
        assert_eq!(info.dst_acceleration_structure, destination);
        // Build-once/trace-many voxel pages must prefer traversal throughput.
        assert_eq!(
            info.flags,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        );
        // We never refit these BLASes — every topology edit is an out-of-place BUILD,
        // so ALLOW_UPDATE must stay off (it would degrade BVH quality for no benefit).
        assert!(
            !info
                .flags
                .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE)
        );
        assert_eq!(unsafe { info.scratch_data.device_address }, 0x4000);
    }

    #[test]
    fn blas_to_tlas_barrier_exposes_build_write_to_build_read() {
        let barrier = compact_blas_to_tlas_barrier();

        assert_eq!(
            barrier.src_stage,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
        );
        assert_eq!(
            barrier.dst_stage,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
        );
        assert_eq!(
            barrier.src_access,
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
        );
        assert_eq!(
            barrier.dst_access,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
        );
    }

    #[test]
    fn scratch_allocation_covers_an_aligned_slice() {
        let layout = RtCompactBlasScratchLayout::new(1_000, 256).unwrap();

        assert_eq!(layout.required_bytes, 1_000);
        assert_eq!(layout.alignment, 256);
        assert_eq!(layout.allocation_bytes, 1_255);
        assert_eq!(layout.align_address(0x1010).unwrap(), 0x1100);
        assert_eq!(layout.align_address(0x1100).unwrap(), 0x1100);
    }

    #[test]
    fn queried_sizes_populate_compact_exact_metrics() {
        let sizes = vk::AccelerationStructureBuildSizesInfoKHR::default()
            .acceleration_structure_size(8_192)
            .build_scratch_size(4_096);

        let metrics = compact_blas_metrics(12, &sizes);

        assert_eq!(
            metrics.representation,
            crate::render::rt_representation_metrics::RtRepresentationKind::CompactExact
        );
        assert_eq!(metrics.page_count, 1);
        assert_eq!(metrics.exposed_face_count, 12);
        assert_eq!(metrics.candidate_primitive_count, 24);
        assert_eq!(metrics.memory.persistent_bytes, 8_192);
        assert_eq!(metrics.memory.scratch_bytes, 4_096);
        assert_eq!(metrics.timings.blas_update_ms, None);
    }

    #[test]
    fn build_lifecycle_never_installs_unsubmitted_or_stale_resources() {
        let mut lifecycle = RtCompactBlasBuildLifecycle::new();
        assert_eq!(lifecycle.state(), RtCompactBlasBuildState::Allocated);
        assert!(!lifecycle.installable());

        lifecycle.mark_recorded().unwrap();
        assert_eq!(lifecycle.state(), RtCompactBlasBuildState::Recorded);
        assert!(!lifecycle.installable());

        lifecycle.mark_submitted().unwrap();
        assert!(lifecycle.installable());

        lifecycle.mark_retire_pending().expect("retire_pending from Submitted must succeed");
        assert_eq!(lifecycle.state(), RtCompactBlasBuildState::RetirePending);
        assert!(!lifecycle.installable());

        // Double-retire must be rejected.
        assert!(lifecycle.mark_retire_pending().is_err());
    }

    #[test]
    fn compact_blas_resources_query_and_own_real_device_build_state() {
        let source = crate::render::source_checks::read_source("src/render/rt_page_blas.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("production BLAS source must precede tests");

        for token in [
            "pub struct RtCompactBlasBuildResources",
            "get_acceleration_structure_build_sizes",
            "RtAccelerationStructure::new",
            "MemoryLocation::GpuOnly",
            "scratch_buffer: Option<GpuBuffer>",
            "min_acceleration_structure_scratch_offset_alignment",
            "cmd_build_acceleration_structures",
            "GpuProfileScope::RtBlasWork",
            "pub struct RtCompactBlasRetirementQueue",
        ] {
            assert!(
                production.contains(token),
                "CompactExact BLAS device path must contain {token}"
            );
        }
        assert!(
            !production.contains("BuildAccelerationStructureModeKHR::UPDATE"),
            "CompactExact topology changes must never use BLAS UPDATE"
        );
    }

    #[test]
    fn profiler_wraps_only_the_compact_blas_build_command() {
        let source = crate::render::source_checks::read_source("src/render/rt_page_blas.rs");
        let record = source
            .split("pub fn record_build")
            .nth(1)
            .expect("CompactExact BLAS resources must expose record_build")
            .split("pub fn mark_submitted")
            .next()
            .expect("record_build must end before submission state transition");
        let compact = crate::render::source_checks::compact(record);
        let begin = compact
            .find("GpuProfileScope::RtBlasWork")
            .expect("BLAS build must begin the dedicated profiler scope");
        let build = compact
            .find("cmd_build_acceleration_structures")
            .expect("BLAS build command must be recorded");
        let end = compact[build..]
            .find("GpuProfileScope::RtBlasWork")
            .map(|offset| build + offset)
            .expect("BLAS build must end the dedicated profiler scope");
        let barrier = compact
            .find("cmd_pipeline_barrier")
            .expect("BLAS build must publish writes for TLAS reads");

        assert!(begin < build && build < end && end < barrier);
    }

    #[test]
    fn compact_blas_installation_atomically_installs_or_retires_owned_resources() {
        let source = crate::render::source_checks::read_source("src/render/rt_page_blas.rs");
        let install = source
            .split("pub fn install_or_retire")
            .nth(1)
            .expect("CompactExact BLAS resources must own the install-or-retire transition")
            .split("pub fn destroy")
            .next()
            .expect("install-or-retire must end before explicit destruction");
        let compact = crate::render::source_checks::compact(install);

        for token in [
            "registry.install_build",
            "retirement_queue.enqueue",
            "validate_compact_blas_install_ticket",
            "mark_installed",
            "fail_build",
        ] {
            assert!(
                compact.contains(token),
                "atomic CompactExact install path must contain {token}"
            );
        }
    }
}
