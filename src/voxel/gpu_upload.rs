// src/voxel/gpu_upload.rs
use anyhow::{Result, bail};
use ash::vk;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::voxel::brick::{BRICK_VOLUME, BrickOccupancy, VoxelCell};
use crate::voxel::occupancy::{CascadedOccupancy, CascadedOccupancyChanges, NodeL0, NodeLN};
use crate::voxel::ucvh::{Ucvh, UcvhMotionEvent, UcvhRenderChangeBatch};
use std::sync::atomic::{AtomicBool, Ordering};

pub const UCVH_MOTION_EVENT_CAPACITY: usize = 64;
pub const INITIAL_UCVH_UPLOAD_FRAME_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// Baseline per-frame-slot material staging size (T1-B). Both the incremental and
/// the initial upload paths now write material into staging with COMPACT addressing
/// (chunk/brick index, not device offset — see `upload_segment_chunk` and
/// `upload_incremental_changes`), so staging need only hold one frame's material
/// traffic instead of the entire material pool (previously ≈400 MiB at 100k bricks ×
/// 2-3 slots ≈ 1 GiB of host-visible memory serving a few-brick delta). It must be at
/// least one initial-upload chunk (`INITIAL_UCVH_UPLOAD_FRAME_BUDGET_BYTES`) so the
/// initial material segment fits; that also comfortably covers ~16k bricks/frame of
/// incremental edits. Rare oversized incremental batches grow the slot's staging in
/// place via `ensure_material_staging_capacity` (fence-safe by frame-slot rotation).
pub const MATERIAL_STAGING_BUDGET_BYTES: usize = INITIAL_UCVH_UPLOAD_FRAME_BUDGET_BYTES;

/// Bytes of one brick's material block: BRICK_VOLUME voxels × sizeof(VoxelCell).
const BRICK_MATERIAL_BYTES: usize = BRICK_VOLUME * std::mem::size_of::<VoxelCell>();
const INITIAL_UCVH_UPLOAD_SEGMENT_COUNT: usize = 9;
static MOTION_EVENT_OVERFLOW_WARNED: AtomicBool = AtomicBool::new(false);

/// Exact device-buffer ranges touched by one deduplicated set of changed bricks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UcvhIncrementalUploadPlan {
    brick_ids: Vec<u32>,
}

impl UcvhIncrementalUploadPlan {
    pub fn for_bricks(brick_ids: &[u32]) -> Self {
        let mut brick_ids = brick_ids.to_vec();
        brick_ids.sort_unstable();
        brick_ids.dedup();
        Self { brick_ids }
    }

    pub fn brick_ids(&self) -> &[u32] {
        &self.brick_ids
    }

    pub fn occupancy_ranges(&self) -> Vec<(u64, u64)> {
        self.ranges_for_stride(std::mem::size_of::<BrickOccupancy>() as u64)
    }

    pub fn material_ranges(&self) -> Vec<(u64, u64)> {
        self.ranges_for_stride((BRICK_VOLUME * std::mem::size_of::<VoxelCell>()) as u64)
    }

    pub fn generation_ranges(&self) -> Vec<(u64, u64)> {
        self.ranges_for_stride(std::mem::size_of::<u32>() as u64)
    }

    fn ranges_for_stride(&self, stride: u64) -> Vec<(u64, u64)> {
        self.brick_ids
            .iter()
            .map(|&brick_id| (u64::from(brick_id) * stride, stride))
            .collect()
    }
}

fn capped_motion_event_upload_count(event_len: usize) -> usize {
    event_len.min(UCVH_MOTION_EVENT_CAPACITY)
}

fn should_warn_motion_event_overflow(event_len: usize) -> bool {
    event_len > UCVH_MOTION_EVENT_CAPACITY
        && !MOTION_EVENT_OVERFLOW_WARNED.swap(true, Ordering::Relaxed)
}

fn initial_upload_brick_count(ucvh: &Ucvh) -> usize {
    ucvh.pool.occupancy_pool().len()
}

fn initial_occupancy_upload_slice(ucvh: &Ucvh) -> &[BrickOccupancy] {
    &ucvh.pool.occupancy_pool()[..initial_upload_brick_count(ucvh)]
}

fn initial_material_upload_slice(ucvh: &Ucvh) -> &[VoxelCell] {
    let voxel_count = initial_upload_brick_count(ucvh) * BRICK_VOLUME;
    &ucvh.pool.material_pool()[..voxel_count]
}

fn initial_generation_upload_slice(ucvh: &Ucvh) -> &[u32] {
    &ucvh.brick_generations()[..initial_upload_brick_count(ucvh)]
}

fn initial_device_pool_sizes(ucvh: &Ucvh) -> (usize, usize, usize) {
    (
        std::mem::size_of_val(initial_occupancy_upload_slice(ucvh)),
        std::mem::size_of_val(initial_material_upload_slice(ucvh)),
        std::mem::size_of_val(initial_generation_upload_slice(ucvh)),
    )
}

fn initial_gpu_config(ucvh: &Ucvh) -> UcvhGpuConfig {
    UcvhGpuConfig {
        world_size: [
            ucvh.config.world_size.x,
            ucvh.config.world_size.y,
            ucvh.config.world_size.z,
            0,
        ],
        brick_grid_size: [
            ucvh.config.brick_grid_size.x,
            ucvh.config.brick_grid_size.y,
            ucvh.config.brick_grid_size.z,
            0,
        ],
        brick_capacity: ucvh.pool.capacity(),
        allocated_bricks: ucvh.pool.allocated_count(),
        _pad: [0; 2],
    }
}

fn hierarchy_level_staging_offset(
    hierarchy: &crate::voxel::occupancy::CascadedOccupancy,
    level_index: usize,
) -> u64 {
    let l0_size = hierarchy.level0.len() * std::mem::size_of::<NodeL0>();
    let preceding_ln_size = hierarchy.levels[..level_index]
        .iter()
        .map(|level| level.len() * std::mem::size_of::<NodeLN>())
        .sum::<usize>();
    (l0_size + preceding_ln_size) as u64
}

fn bounded_upload_chunk_len(remaining_bytes: usize, budget_bytes: usize) -> usize {
    let chunk_len = remaining_bytes.min(budget_bytes);
    if chunk_len == remaining_bytes {
        return chunk_len;
    }
    chunk_len - (chunk_len % 4)
}

/// GPU-side config matching the shader UBO.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UcvhGpuConfig {
    pub world_size: [u32; 4],      // xyz + pad
    pub brick_grid_size: [u32; 4], // xyz + pad
    pub brick_capacity: u32,
    pub allocated_bricks: u32,
    pub _pad: [u32; 2],
}

/// All GPU buffers for UCVH data.
pub struct UcvhGpuResources {
    pub config_buffer: GpuBuffer,
    pub occupancy_buffer: GpuBuffer,
    pub material_buffer: GpuBuffer,
    pub hierarchy_l0_buffer: GpuBuffer,
    pub hierarchy_ln_buffers: [GpuBuffer; 4], // L1-L4
    pub brick_generation_buffer: GpuBuffer,
    pub motion_event_buffer: GpuBuffer,
    brick_capacity: usize,
    // One host-visible staging set per frame slot. RenderDevice waits the
    // slot's fence before reusing its command buffer, so that also protects
    // this set from CPU overwrite while an earlier transfer still reads it.
    staging: Vec<UcvhStagingResources>,
}

struct UcvhStagingResources {
    occupancy: GpuBuffer,
    material: GpuBuffer,
    hierarchy: GpuBuffer,
    config: GpuBuffer,
    brick_generations: GpuBuffer,
    motion_events: GpuBuffer,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UcvhIncrementalUploadResult {
    pub changed_bricks: u32,
    pub bytes_uploaded: usize,
    /// True when the RT backend skipped uploading changed L1-L4 hierarchy levels
    /// (T1-C). The GPU L1-L4 buffers are now stale versus the CPU hierarchy; the
    /// runtime must record a full L1-L4 re-upload before the next VPT trace.
    pub skipped_ln_hierarchy: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct UcvhInitialUploadProgress {
    segment_index: usize,
    segment_offset: usize,
    completed: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct UcvhInitialUploadFrameResult {
    pub completed: bool,
    pub bytes_uploaded: usize,
}

struct InitialUploadSegment<'a> {
    data: &'a [u8],
    staging: &'a GpuBuffer,
    destination: &'a GpuBuffer,
    staging_base_offset: u64,
    destination_base_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuUcvhMotionEvent {
    pub region_min: [u32; 4],
    pub region_max_exclusive: [u32; 4],
    pub world_delta_current_from_previous: [i32; 4],
    pub generation: u32,
    pub _pad: [u32; 3],
}

impl From<UcvhMotionEvent> for GpuUcvhMotionEvent {
    fn from(event: UcvhMotionEvent) -> Self {
        Self {
            region_min: [
                event.region_min.x,
                event.region_min.y,
                event.region_min.z,
                0,
            ],
            region_max_exclusive: [
                event.region_max_exclusive.x,
                event.region_max_exclusive.y,
                event.region_max_exclusive.z,
                0,
            ],
            world_delta_current_from_previous: [
                event.world_delta_current_from_previous.x,
                event.world_delta_current_from_previous.y,
                event.world_delta_current_from_previous.z,
                0,
            ],
            generation: event.generation,
            _pad: [0; 3],
        }
    }
}

impl UcvhGpuResources {
    pub(crate) fn config_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
    }

    pub(crate) fn device_storage_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        ucvh: &Ucvh,
        frame_slot_count: usize,
    ) -> Result<Self> {
        if frame_slot_count == 0 {
            bail!("UCVH GPU resources require at least one frame slot");
        }
        let (initial_occ_size, initial_mat_size, initial_generation_size) =
            initial_device_pool_sizes(ucvh);
        let motion_event_size =
            UCVH_MOTION_EVENT_CAPACITY * std::mem::size_of::<GpuUcvhMotionEvent>();

        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;

        // Device-local SSBOs
        let config_buffer = GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<UcvhGpuConfig>() as u64,
            Self::config_buffer_usage(),
            MemoryLocation::GpuOnly,
            "ucvh_config",
        )?;
        let ssbo_usage = Self::device_storage_buffer_usage();
        let occupancy_buffer = GpuBuffer::new(
            device,
            allocator,
            initial_occ_size.max(16) as u64,
            ssbo_usage,
            MemoryLocation::GpuOnly,
            "ucvh_occupancy",
        )?;
        let material_buffer = GpuBuffer::new(
            device,
            allocator,
            initial_mat_size.max(16) as u64,
            ssbo_usage,
            MemoryLocation::GpuOnly,
            "ucvh_materials",
        )?;
        let brick_generation_buffer = GpuBuffer::new(
            device,
            allocator,
            initial_generation_size.max(16) as u64,
            ssbo_usage,
            MemoryLocation::GpuOnly,
            "ucvh_brick_generations",
        )?;
        let motion_event_buffer = GpuBuffer::new(
            device,
            allocator,
            motion_event_size.max(16) as u64,
            ssbo_usage,
            MemoryLocation::GpuOnly,
            "ucvh_motion_events",
        )?;

        // Hierarchy buffers
        let h = &ucvh.hierarchy;
        let l0_size = h.level0.len() * std::mem::size_of::<NodeL0>();
        let ln_sizes: [usize; 4] =
            std::array::from_fn(|i| h.levels[i].len() * std::mem::size_of::<NodeLN>());

        let hierarchy_l0_buffer = GpuBuffer::new(
            device,
            allocator,
            l0_size.max(16) as u64,
            ssbo_usage,
            MemoryLocation::GpuOnly,
            "ucvh_hierarchy_l0",
        )?;
        let hierarchy_ln_buffers = [
            GpuBuffer::new(
                device,
                allocator,
                ln_sizes[0].max(16) as u64,
                ssbo_usage,
                MemoryLocation::GpuOnly,
                "ucvh_hierarchy_l1",
            )?,
            GpuBuffer::new(
                device,
                allocator,
                ln_sizes[1].max(16) as u64,
                ssbo_usage,
                MemoryLocation::GpuOnly,
                "ucvh_hierarchy_l2",
            )?,
            GpuBuffer::new(
                device,
                allocator,
                ln_sizes[2].max(16) as u64,
                ssbo_usage,
                MemoryLocation::GpuOnly,
                "ucvh_hierarchy_l3",
            )?,
            GpuBuffer::new(
                device,
                allocator,
                ln_sizes[3].max(16) as u64,
                ssbo_usage,
                MemoryLocation::GpuOnly,
                "ucvh_hierarchy_l4",
            )?,
        ];

        // Staging buffers (host-visible)
        let total_hierarchy = l0_size + ln_sizes.iter().sum::<usize>();

        let mut staging = Vec::with_capacity(frame_slot_count);
        for frame_slot in 0..frame_slot_count {
            staging.push(UcvhStagingResources {
                occupancy: GpuBuffer::new(
                    device,
                    allocator,
                    initial_occ_size.max(16) as u64,
                    staging_usage,
                    MemoryLocation::CpuToGpu,
                    &format!("ucvh_staging_occupancy_{frame_slot}"),
                )?,
                // T1-B: material staging uses compact slots, so it is sized to a
                // per-frame delta budget rather than the whole material pool. Capped at
                // the initial pool size (never need more than the whole pool) and grown
                // in place for rare oversized batches by ensure_material_staging_capacity.
                material: GpuBuffer::new(
                    device,
                    allocator,
                    MATERIAL_STAGING_BUDGET_BYTES.min(initial_mat_size.max(16)).max(16) as u64,
                    staging_usage,
                    MemoryLocation::CpuToGpu,
                    &format!("ucvh_staging_material_{frame_slot}"),
                )?,
                hierarchy: GpuBuffer::new(
                    device,
                    allocator,
                    total_hierarchy.max(16) as u64,
                    staging_usage,
                    MemoryLocation::CpuToGpu,
                    &format!("ucvh_staging_hierarchy_{frame_slot}"),
                )?,
                config: GpuBuffer::new(
                    device,
                    allocator,
                    std::mem::size_of::<UcvhGpuConfig>() as u64,
                    staging_usage,
                    MemoryLocation::CpuToGpu,
                    &format!("ucvh_staging_config_{frame_slot}"),
                )?,
                brick_generations: GpuBuffer::new(
                    device,
                    allocator,
                    initial_generation_size.max(16) as u64,
                    staging_usage,
                    MemoryLocation::CpuToGpu,
                    &format!("ucvh_staging_generations_{frame_slot}"),
                )?,
                motion_events: GpuBuffer::new(
                    device,
                    allocator,
                    motion_event_size.max(16) as u64,
                    staging_usage,
                    MemoryLocation::CpuToGpu,
                    &format!("ucvh_staging_motion_events_{frame_slot}"),
                )?,
            });
        }

        Ok(Self {
            config_buffer,
            occupancy_buffer,
            material_buffer,
            hierarchy_l0_buffer,
            hierarchy_ln_buffers,
            brick_generation_buffer,
            motion_event_buffer,
            brick_capacity: initial_upload_brick_count(ucvh),
            staging,
        })
    }

    pub fn brick_capacity(&self) -> usize {
        self.brick_capacity
    }

    fn staging_for_frame_slot(&self, frame_slot: usize) -> Result<&UcvhStagingResources> {
        self.staging.get(frame_slot).ok_or_else(|| {
            anyhow::anyhow!(
                "UCVH staging frame slot is out of range: frame_slot={frame_slot} slot_count={}",
                self.staging.len()
            )
        })
    }

    /// Ensure the given frame slot's material staging buffer can hold `brick_count`
    /// bricks' material (T1-B). Material staging is budget-sized (compact slots), so a
    /// rare oversized edit batch grows it in place. Fence-safe: the caller (runtime)
    /// waits on this frame slot's fence before reusing the slot's command buffer, which
    /// also guarantees no in-flight transfer still reads this staging buffer, so
    /// destroying and reallocating it here cannot race the GPU. Grow-only: the buffer
    /// keeps the larger size after a big batch (huge incremental batches are rare, and
    /// the persistent baseline stays at the budget for typical few-brick edits).
    pub fn ensure_material_staging_capacity(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        frame_slot: usize,
        brick_count: usize,
    ) -> Result<()> {
        let required = brick_count.saturating_mul(BRICK_MATERIAL_BYTES).max(16);
        let staging = self.staging.get_mut(frame_slot).ok_or_else(|| {
            anyhow::anyhow!(
                "UCVH staging frame slot is out of range: frame_slot={frame_slot}"
            )
        })?;
        if (staging.material.size as usize) >= required {
            return Ok(());
        }
        let new_buffer = GpuBuffer::new(
            device,
            allocator,
            required as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            &format!("ucvh_staging_material_{frame_slot}"),
        )?;
        let old = std::mem::replace(&mut staging.material, new_buffer);
        old.destroy(device, allocator);
        Ok(())
    }

    pub fn upload_incremental_changes(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        ucvh: &mut Ucvh,
        batch: &UcvhRenderChangeBatch,
        skip_ln_hierarchy_upload: bool,
    ) -> Result<UcvhIncrementalUploadResult> {
        if batch.is_empty() {
            return Ok(UcvhIncrementalUploadResult::default());
        }
        if self.brick_capacity < ucvh.pool.occupancy_pool().len() {
            bail!(
                "incremental UCVH upload requires GPU buffer growth: gpu_capacity={} cpu_storage={}",
                self.brick_capacity,
                ucvh.pool.occupancy_pool().len()
            );
        }
        let staging = self.staging_for_frame_slot(frame_slot)?;

        let hierarchy_changes = ucvh.update_hierarchy_for_render_change_batch(batch);
        let plan = UcvhIncrementalUploadPlan::for_bricks(
            &batch
                .bricks
                .iter()
                .map(|brick| brick.brick_id)
                .collect::<Vec<_>>(),
        );
        // T1-B: material staging is addressed by a compact per-frame slot index, not
        // the brick's device offset, so material staging need only hold this frame's
        // changed bricks (not the whole pool). Occupancy (80 B) and generation (4 B)
        // stay device-offset-addressed — their full-pool staging is small. The caller
        // must have grown this slot's material staging to fit the batch via
        // `ensure_material_staging_capacity`; bail defensively if it did not.
        let required_material_staging = batch
            .bricks
            .len()
            .saturating_mul(BRICK_MATERIAL_BYTES);
        if (staging.material.size as usize) < required_material_staging {
            bail!(
                "material staging too small for incremental batch: have={} need={} (call ensure_material_staging_capacity first)",
                staging.material.size,
                required_material_staging
            );
        }
        let mut bytes_uploaded = 0usize;
        for (compact_slot, &brick_id) in plan.brick_ids().iter().enumerate() {
            let brick_index = brick_id as usize;
            let occupancy_bytes = bytes_of(ucvh.pool.occupancy(brick_id));
            let material_start = brick_index * BRICK_VOLUME;
            let material_bytes = cast_slice::<VoxelCell, u8>(
                &ucvh.pool.material_pool()[material_start..material_start + BRICK_VOLUME],
            );
            let generation = ucvh
                .brick_generation(brick_id)
                .ok_or_else(|| anyhow::anyhow!("missing UCVH generation for brick {brick_id}"))?;
            let generation_bytes = bytes_of(&generation);

            let occupancy_offset = brick_index * std::mem::size_of::<BrickOccupancy>();
            let material_device_offset = material_start * std::mem::size_of::<VoxelCell>();
            let material_staging_offset = compact_slot * BRICK_MATERIAL_BYTES;
            let generation_offset = brick_index * std::mem::size_of::<u32>();
            Self::copy_to_staging_offset(&staging.occupancy, occupancy_bytes, occupancy_offset)?;
            Self::record_copy_region(
                device,
                cmd,
                &staging.occupancy,
                &self.occupancy_buffer,
                occupancy_offset as u64,
                occupancy_offset as u64,
                occupancy_bytes.len() as u64,
            );
            // Compact staging slot -> device offset (T1-B).
            Self::copy_to_staging_offset(&staging.material, material_bytes, material_staging_offset)?;
            Self::record_copy_region(
                device,
                cmd,
                &staging.material,
                &self.material_buffer,
                material_staging_offset as u64,
                material_device_offset as u64,
                material_bytes.len() as u64,
            );
            Self::copy_to_staging_offset(
                &staging.brick_generations,
                generation_bytes,
                generation_offset,
            )?;
            Self::record_copy_region(
                device,
                cmd,
                &staging.brick_generations,
                &self.brick_generation_buffer,
                generation_offset as u64,
                generation_offset as u64,
                generation_bytes.len() as u64,
            );
            bytes_uploaded += occupancy_bytes.len() + material_bytes.len() + generation_bytes.len();
        }
        bytes_uploaded += self.upload_incremental_hierarchy(
            device,
            cmd,
            staging,
            ucvh,
            &hierarchy_changes,
            skip_ln_hierarchy_upload,
        )?;
        Self::record_upload_barrier(device, cmd);

        Ok(UcvhIncrementalUploadResult {
            changed_bricks: plan.brick_ids().len() as u32,
            bytes_uploaded,
            skipped_ln_hierarchy: skip_ln_hierarchy_upload && !hierarchy_changes.levels.is_empty(),
        })
    }

    /// Upload all UCVH data to GPU. Call once after scene generation.
    /// Records copy commands into `cmd` — must be called between begin/end command buffer.
    pub fn upload_all(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        ucvh: &Ucvh,
    ) -> Result<()> {
        let staging = self.staging_for_frame_slot(frame_slot)?;
        // Upload config
        let gpu_config = initial_gpu_config(ucvh);
        Self::copy_to_staging(&staging.config, bytes_of(&gpu_config))?;

        // Upload occupancy pool
        let occ_bytes = cast_slice::<BrickOccupancy, u8>(initial_occupancy_upload_slice(ucvh));
        Self::copy_to_staging(&staging.occupancy, occ_bytes)?;

        // Upload material pool
        let mat_bytes = cast_slice::<VoxelCell, u8>(initial_material_upload_slice(ucvh));
        Self::copy_to_staging(&staging.material, mat_bytes)?;

        let generation_bytes = cast_slice::<u32, u8>(initial_generation_upload_slice(ucvh));
        Self::copy_to_staging(&staging.brick_generations, generation_bytes)?;

        // Upload hierarchy
        let mut offset = 0u64;
        let l0_bytes = cast_slice::<NodeL0, u8>(&ucvh.hierarchy.level0);
        Self::copy_to_staging_offset(&staging.hierarchy, l0_bytes, offset as usize)?;
        offset += l0_bytes.len() as u64;

        let mut ln_offsets = [0u64; 4];
        for (i, ln_offset) in ln_offsets.iter_mut().enumerate() {
            *ln_offset = offset;
            let ln_bytes = cast_slice::<NodeLN, u8>(&ucvh.hierarchy.levels[i]);
            Self::copy_to_staging_offset(&staging.hierarchy, ln_bytes, offset as usize)?;
            offset += ln_bytes.len() as u64;
        }

        Self::record_copy(
            device,
            cmd,
            &staging.config,
            &self.config_buffer,
            std::mem::size_of::<UcvhGpuConfig>() as u64,
        );
        Self::record_copy(
            device,
            cmd,
            &staging.occupancy,
            &self.occupancy_buffer,
            occ_bytes.len() as u64,
        );
        Self::record_copy(
            device,
            cmd,
            &staging.material,
            &self.material_buffer,
            mat_bytes.len() as u64,
        );
        Self::record_copy(
            device,
            cmd,
            &staging.brick_generations,
            &self.brick_generation_buffer,
            generation_bytes.len() as u64,
        );

        // Record copies: staging_hierarchy -> individual device-local buffers
        let l0_size = l0_bytes.len() as u64;
        Self::record_copy_region(
            device,
            cmd,
            &staging.hierarchy,
            &self.hierarchy_l0_buffer,
            0,
            0,
            l0_size,
        );
        for (i, ln_offset) in ln_offsets.iter().enumerate() {
            let ln_size = (ucvh.hierarchy.levels[i].len() * std::mem::size_of::<NodeLN>()) as u64;
            Self::record_copy_region(
                device,
                cmd,
                &staging.hierarchy,
                &self.hierarchy_ln_buffers[i],
                *ln_offset,
                0,
                ln_size,
            );
        }

        Self::record_upload_barrier(device, cmd);

        Ok(())
    }

    pub(crate) fn upload_initial_incremental(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        ucvh: &Ucvh,
        progress: &mut UcvhInitialUploadProgress,
        budget_bytes: usize,
    ) -> Result<UcvhInitialUploadFrameResult> {
        if progress.completed {
            return Ok(UcvhInitialUploadFrameResult {
                completed: true,
                bytes_uploaded: 0,
            });
        }
        let staging = self.staging_for_frame_slot(frame_slot)?;

        let mut remaining_budget = budget_bytes;
        let mut bytes_uploaded = 0usize;

        while remaining_budget > 0 && progress.segment_index < INITIAL_UCVH_UPLOAD_SEGMENT_COUNT {
            let chunk_uploaded = if progress.segment_index == 0 {
                let gpu_config = initial_gpu_config(ucvh);
                let config_bytes = bytes_of(&gpu_config);
                let segment = InitialUploadSegment {
                    data: config_bytes,
                    staging: &staging.config,
                    destination: &self.config_buffer,
                    staging_base_offset: 0,
                    destination_base_offset: 0,
                };
                Self::upload_segment_chunk(
                    device,
                    cmd,
                    &segment,
                    progress.segment_offset,
                    remaining_budget,
                )?
            } else {
                let Some(segment) =
                    self.initial_upload_segment(staging, ucvh, progress.segment_index)
                else {
                    progress.segment_index += 1;
                    progress.segment_offset = 0;
                    continue;
                };
                Self::upload_segment_chunk(
                    device,
                    cmd,
                    &segment,
                    progress.segment_offset,
                    remaining_budget,
                )?
            };

            if chunk_uploaded == 0 {
                break;
            }
            progress.segment_offset += chunk_uploaded;
            remaining_budget -= chunk_uploaded;
            bytes_uploaded += chunk_uploaded;

            let segment_len = if progress.segment_index == 0 {
                std::mem::size_of::<UcvhGpuConfig>()
            } else {
                self.initial_upload_segment(staging, ucvh, progress.segment_index)
                    .map(|segment| segment.data.len())
                    .unwrap_or(0)
            };
            if progress.segment_offset >= segment_len {
                progress.segment_index += 1;
                progress.segment_offset = 0;
            }
        }

        if bytes_uploaded > 0 {
            Self::record_upload_barrier(device, cmd);
        }

        if progress.segment_index >= INITIAL_UCVH_UPLOAD_SEGMENT_COUNT {
            progress.completed = true;
        }

        Ok(UcvhInitialUploadFrameResult {
            completed: progress.completed,
            bytes_uploaded,
        })
    }

    fn upload_incremental_hierarchy(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        staging: &UcvhStagingResources,
        ucvh: &Ucvh,
        changes: &CascadedOccupancyChanges,
        skip_ln_levels: bool,
    ) -> Result<usize> {
        // L0 is always uploaded — the hardware RT procedural intersection shader
        // (rt_surface.rint.slang) reads hierarchy_l0 to recover brick_id/occupancy.
        // L1-L4 are consumed only by the VPT software traversal's empty-space skip
        // (voxel_traverse.slang), so on the RT backend `skip_ln_levels` elides their
        // GPU upload (T1-C). The CPU-side L1-L4 recompute in
        // `update_hierarchy_for_render_change_batch` still runs, so CPU levels stay
        // current and an RT→VPT switch only needs a full L1-L4 re-upload (see
        // `record_full_ln_hierarchy_upload`), never a rebuild.
        let mut bytes_uploaded = 0usize;
        for position in &changes.l0 {
            let offset = CascadedOccupancy::flat_index(*position, ucvh.hierarchy.dims[0])
                * std::mem::size_of::<NodeL0>();
            let bytes = bytes_of(&ucvh.hierarchy.level0[offset / std::mem::size_of::<NodeL0>()]);
            Self::copy_to_staging_offset(&staging.hierarchy, bytes, offset)?;
            Self::record_copy_region(
                device,
                cmd,
                &staging.hierarchy,
                &self.hierarchy_l0_buffer,
                offset as u64,
                offset as u64,
                bytes.len() as u64,
            );
            bytes_uploaded += bytes.len();
        }
        if skip_ln_levels {
            return Ok(bytes_uploaded);
        }
        for (level_index, positions) in changes.levels.iter().enumerate() {
            let staging_base =
                hierarchy_level_staging_offset(&ucvh.hierarchy, level_index) as usize;
            for position in positions {
                let offset =
                    CascadedOccupancy::flat_index(*position, ucvh.hierarchy.dims[level_index + 1])
                        * std::mem::size_of::<NodeLN>();
                let bytes = bytes_of(
                    &ucvh.hierarchy.levels[level_index][offset / std::mem::size_of::<NodeLN>()],
                );
                Self::copy_to_staging_offset(&staging.hierarchy, bytes, staging_base + offset)?;
                Self::record_copy_region(
                    device,
                    cmd,
                    &staging.hierarchy,
                    &self.hierarchy_ln_buffers[level_index],
                    (staging_base + offset) as u64,
                    offset as u64,
                    bytes.len() as u64,
                );
                bytes_uploaded += bytes.len();
            }
        }
        Ok(bytes_uploaded)
    }

    /// Re-upload the entire L1-L4 hierarchy from the current CPU state (T1-C).
    ///
    /// Called on an RT→VPT backend switch after the RT backend skipped incremental
    /// L1-L4 uploads (`skipped_ln_hierarchy`). The CPU L1-L4 levels are always kept
    /// current by `update_hierarchy_for_render_change_batch`, so a wholesale copy of
    /// each level buffer resynchronizes the GPU without any rebuild. The staging
    /// hierarchy buffer is sized to the full hierarchy at creation, so all four levels
    /// fit in one staging pass. L0 is not touched (it is always kept current on RT).
    pub fn record_full_ln_hierarchy_upload(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        ucvh: &Ucvh,
    ) -> Result<usize> {
        let staging = self.staging_for_frame_slot(frame_slot)?;
        let mut bytes_uploaded = 0usize;
        for level_index in 0..ucvh.hierarchy.levels.len() {
            let level = &ucvh.hierarchy.levels[level_index];
            if level.is_empty() {
                continue;
            }
            let staging_base =
                hierarchy_level_staging_offset(&ucvh.hierarchy, level_index) as usize;
            let bytes = cast_slice::<NodeLN, u8>(level);
            Self::copy_to_staging_offset(&staging.hierarchy, bytes, staging_base)?;
            Self::record_copy_region(
                device,
                cmd,
                &staging.hierarchy,
                &self.hierarchy_ln_buffers[level_index],
                staging_base as u64,
                0,
                bytes.len() as u64,
            );
            bytes_uploaded += bytes.len();
        }
        Self::record_upload_barrier(device, cmd);
        Ok(bytes_uploaded)
    }

    fn initial_upload_segment<'a>(
        &'a self,
        staging: &'a UcvhStagingResources,
        ucvh: &'a Ucvh,
        index: usize,
    ) -> Option<InitialUploadSegment<'a>> {
        let h = &ucvh.hierarchy;
        match index {
            1 => Some(InitialUploadSegment {
                data: cast_slice::<BrickOccupancy, u8>(initial_occupancy_upload_slice(ucvh)),
                staging: &staging.occupancy,
                destination: &self.occupancy_buffer,
                staging_base_offset: 0,
                destination_base_offset: 0,
            }),
            2 => Some(InitialUploadSegment {
                data: cast_slice::<VoxelCell, u8>(initial_material_upload_slice(ucvh)),
                staging: &staging.material,
                destination: &self.material_buffer,
                staging_base_offset: 0,
                destination_base_offset: 0,
            }),
            3 => Some(InitialUploadSegment {
                data: cast_slice::<u32, u8>(initial_generation_upload_slice(ucvh)),
                staging: &staging.brick_generations,
                destination: &self.brick_generation_buffer,
                staging_base_offset: 0,
                destination_base_offset: 0,
            }),
            4 => Some(InitialUploadSegment {
                data: cast_slice::<NodeL0, u8>(&h.level0),
                staging: &staging.hierarchy,
                destination: &self.hierarchy_l0_buffer,
                staging_base_offset: 0,
                destination_base_offset: 0,
            }),
            5..=8 => {
                let level_index = index - 5;
                Some(InitialUploadSegment {
                    data: cast_slice::<NodeLN, u8>(&h.levels[level_index]),
                    staging: &staging.hierarchy,
                    destination: &self.hierarchy_ln_buffers[level_index],
                    staging_base_offset: hierarchy_level_staging_offset(h, level_index),
                    destination_base_offset: 0,
                })
            }
            _ => None,
        }
    }

    fn upload_segment_chunk(
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        segment: &InitialUploadSegment<'_>,
        segment_offset: usize,
        budget_bytes: usize,
    ) -> Result<usize> {
        if segment_offset >= segment.data.len() {
            return Ok(0);
        }
        let chunk_len = bounded_upload_chunk_len(segment.data.len() - segment_offset, budget_bytes);
        if chunk_len == 0 {
            return Ok(0);
        }
        let chunk = &segment.data[segment_offset..segment_offset + chunk_len];
        // T1-B: write each chunk at the segment's staging BASE (compact), not at
        // base + segment_offset. Each segment uploads at most one chunk per frame (the
        // shared budget is consumed per chunk), and frames rotate frame slots with a
        // fence wait before reuse, so successive chunks of the same segment across
        // frames never race on the same staging region. This decouples staging size
        // from the device buffer size, so staging.material need only hold one budget
        // chunk instead of the whole material pool. The device copy still targets the
        // growing destination offset.
        let staging_offset = segment.staging_base_offset;
        let destination_offset = segment.destination_base_offset + segment_offset as u64;
        Self::copy_to_staging_offset(segment.staging, chunk, staging_offset as usize)?;
        Self::record_copy_region(
            device,
            cmd,
            segment.staging,
            segment.destination,
            staging_offset,
            destination_offset,
            chunk_len as u64,
        );
        Ok(chunk_len)
    }

    pub fn upload_motion_guide(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        motion_events: &[UcvhMotionEvent],
    ) -> Result<u32> {
        let staging = self.staging_for_frame_slot(frame_slot)?;

        let event_count = capped_motion_event_upload_count(motion_events.len());
        if should_warn_motion_event_overflow(motion_events.len()) {
            tracing::warn!(
                count = motion_events.len(),
                cap = UCVH_MOTION_EVENT_CAPACITY,
                "dropping excess UCVH motion events for this frame"
            );
        }
        let mut gpu_events = [GpuUcvhMotionEvent::zeroed(); UCVH_MOTION_EVENT_CAPACITY];
        for (dst, src) in gpu_events.iter_mut().zip(motion_events.iter().copied()) {
            *dst = src.into();
        }
        let event_bytes = cast_slice::<GpuUcvhMotionEvent, u8>(&gpu_events);
        Self::copy_to_staging(&staging.motion_events, event_bytes)?;
        Self::record_copy(
            device,
            cmd,
            &staging.motion_events,
            &self.motion_event_buffer,
            event_bytes.len() as u64,
        );

        Self::record_upload_barrier(device, cmd);

        Ok(event_count as u32)
    }

    fn record_upload_barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
                    | vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    fn copy_to_staging(buffer: &GpuBuffer, data: &[u8]) -> Result<()> {
        Self::copy_to_staging_offset(buffer, data, 0)
    }

    fn copy_to_staging_offset(buffer: &GpuBuffer, data: &[u8], offset: usize) -> Result<()> {
        Self::copy_to_mapped_staging(buffer.mapped_ptr(), buffer.size, data, offset)
    }

    fn copy_to_mapped_staging(
        mapped_ptr: Option<*mut u8>,
        buffer_size: vk::DeviceSize,
        data: &[u8],
        offset: usize,
    ) -> Result<()> {
        let size = buffer_size as usize;
        let end = match offset.checked_add(data.len()) {
            Some(end) => end,
            None => {
                bail!(
                    "staging copy exceeds staging buffer: offset={} bytes={} buffer_size={}",
                    offset,
                    data.len(),
                    buffer_size
                );
            }
        };
        if offset > size || end > size {
            bail!(
                "staging copy exceeds staging buffer: offset={} bytes={} buffer_size={}",
                offset,
                data.len(),
                buffer_size
            );
        }
        let Some(ptr) = mapped_ptr else {
            bail!(
                "staging buffer not mapped: offset={} bytes={} buffer_size={}",
                offset,
                data.len(),
                buffer_size
            );
        };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len()) };
        Ok(())
    }

    fn record_copy(
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        size: u64,
    ) {
        Self::record_copy_region(device, cmd, src, dst, 0, 0, size);
    }

    fn record_copy_region(
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        if size == 0 {
            return;
        }
        let region = vk::BufferCopy {
            src_offset,
            dst_offset,
            size,
        };
        unsafe { device.cmd_copy_buffer(cmd, src.handle, dst.handle, &[region]) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.config_buffer.destroy(device, allocator);
        self.occupancy_buffer.destroy(device, allocator);
        self.material_buffer.destroy(device, allocator);
        self.hierarchy_l0_buffer.destroy(device, allocator);
        for buf in self.hierarchy_ln_buffers {
            buf.destroy(device, allocator);
        }
        self.brick_generation_buffer.destroy(device, allocator);
        self.motion_event_buffer.destroy(device, allocator);
        for staging in self.staging {
            staging.occupancy.destroy(device, allocator);
            staging.material.destroy(device, allocator);
            staging.hierarchy.destroy(device, allocator);
            staging.config.destroy(device, allocator);
            staging.brick_generations.destroy(device, allocator);
            staging.motion_events.destroy(device, allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unmapped_buffer(size: vk::DeviceSize) -> GpuBuffer {
        GpuBuffer {
            handle: vk::Buffer::null(),
            size,
            allocation: None,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
        }
    }

    #[test]
    fn copy_to_staging_rejects_unmapped_buffer() {
        let buffer = unmapped_buffer(4);

        let err = UcvhGpuResources::copy_to_staging(&buffer, &[1, 2])
            .expect_err("unmapped staging buffer should fail");

        assert!(err.to_string().contains("not mapped"));
    }

    #[test]
    fn copy_to_staging_rejects_data_larger_than_buffer() {
        let buffer = unmapped_buffer(2);

        let err = UcvhGpuResources::copy_to_staging(&buffer, &[1, 2, 3])
            .expect_err("oversized staging copy should fail");

        assert!(err.to_string().contains("exceeds staging buffer"));
    }

    #[test]
    fn copy_to_staging_offset_rejects_write_past_buffer_end() {
        let buffer = unmapped_buffer(4);

        let err = UcvhGpuResources::copy_to_staging_offset(&buffer, &[1, 2], 3)
            .expect_err("offset staging copy past the end should fail");

        assert!(err.to_string().contains("exceeds staging buffer"));
    }

    #[test]
    fn copy_to_staging_offset_rejects_offset_past_buffer_end() {
        let buffer = unmapped_buffer(4);

        let err = UcvhGpuResources::copy_to_staging_offset(&buffer, &[], 5)
            .expect_err("offset past the end should fail");

        assert!(err.to_string().contains("exceeds staging buffer"));
    }

    #[test]
    fn copy_to_mapped_staging_accepts_exact_fit() {
        let mut storage = [0_u8; 4];

        UcvhGpuResources::copy_to_mapped_staging(
            Some(storage.as_mut_ptr()),
            storage.len() as vk::DeviceSize,
            &[1, 2, 3, 4],
            0,
        )
        .expect("exact-fit staging copy should succeed");

        assert_eq!(storage, [1, 2, 3, 4]);
    }

    #[test]
    fn copy_to_mapped_staging_writes_at_offset() {
        let mut storage = [9_u8; 5];

        UcvhGpuResources::copy_to_mapped_staging(
            Some(storage.as_mut_ptr()),
            storage.len() as vk::DeviceSize,
            &[1, 2, 3],
            1,
        )
        .expect("offset staging copy should succeed");

        assert_eq!(storage, [9, 1, 2, 3, 9]);
    }

    #[test]
    fn incremental_upload_plan_targets_only_changed_brick_ranges() {
        let plan = UcvhIncrementalUploadPlan::for_bricks(&[9, 3, 9]);

        assert_eq!(plan.occupancy_ranges(), vec![(240, 80), (720, 80)]);
        assert_eq!(
            plan.material_ranges(),
            vec![(12_288, 4_096), (36_864, 4_096)]
        );
        assert_eq!(plan.generation_ranges(), vec![(12, 4), (36, 4)]);
    }

    #[test]
    fn ucvh_config_buffer_usage_matches_constant_buffer_descriptors() {
        let vpt_shader =
            crate::render::source_checks::read_source("assets/shaders/passes/vpt.slang");
        let surface_shader =
            crate::render::source_checks::read_source("assets/shaders/passes/vpt_surface.slang");
        let rt_surface_shader = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_surface.rint.slang",
        );

        assert!(vpt_shader.contains("ConstantBuffer<UcvhConfig> ucvh_config"));
        assert!(surface_shader.contains("ConstantBuffer<UcvhConfig> ucvh_config"));
        assert!(rt_surface_shader.contains("ConstantBuffer<UcvhConfig> ucvh_config"));
        let config_usage = UcvhGpuResources::config_buffer_usage();
        let storage_usage = UcvhGpuResources::device_storage_buffer_usage();

        assert!(config_usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER));
        assert!(config_usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
        assert!(!config_usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER));
        assert!(storage_usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER));
        assert!(storage_usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
        assert!(!storage_usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER));
    }

    #[test]
    fn initial_ucvh_upload_slices_are_limited_to_allocated_brick_prefix() {
        let mut ucvh = Ucvh::new(crate::voxel::ucvh::UcvhConfig::with_brick_capacity(
            glam::UVec3::new(2048, 768, 2048),
            64,
        ));
        assert!(ucvh.set_voxel(glam::UVec3::new(0, 0, 0), VoxelCell::new(2, 0, [0; 3])));
        assert!(ucvh.set_voxel(
            glam::UVec3::new(2040, 760, 2040),
            VoxelCell::new(3, 0, [0; 3])
        ));

        assert_eq!(initial_occupancy_upload_slice(&ucvh).len(), 2);
        assert_eq!(initial_material_upload_slice(&ucvh).len(), 2 * BRICK_VOLUME);
        assert_eq!(initial_generation_upload_slice(&ucvh).len(), 2);
        assert_eq!(
            initial_device_pool_sizes(&ucvh),
            (
                2 * std::mem::size_of::<BrickOccupancy>(),
                2 * BRICK_VOLUME * std::mem::size_of::<VoxelCell>(),
                2 * std::mem::size_of::<u32>(),
            )
        );
    }

    #[test]
    fn gpu_resource_creation_sizes_pool_buffers_from_initial_allocated_bricks() {
        let source = crate::render::source_checks::read_source("src/voxel/gpu_upload.rs");
        let new_body = source
            .split("pub fn new")
            .nth(1)
            .expect("UcvhGpuResources::new should exist")
            .split("/// Upload all UCVH data")
            .next()
            .expect("new should end before upload_all docs");

        assert!(
            new_body.contains("initial_device_pool_sizes(ucvh)"),
            "initial GPU pool buffers must be sized by allocated brick prefix, not brick capacity"
        );
    }

    #[test]
    fn initial_ucvh_upload_exposes_incremental_frame_budget() {
        let source = crate::render::source_checks::read_source("src/voxel/gpu_upload.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("implementation should precede tests");

        assert!(
            implementation.contains("INITIAL_UCVH_UPLOAD_FRAME_BUDGET_BYTES"),
            "initial UCVH upload must have a bounded per-frame byte budget"
        );
        assert!(
            implementation.contains("pub(crate) fn upload_initial_incremental"),
            "initial UCVH upload must be resumable across frames instead of one blocking upload"
        );
        assert!(
            implementation.contains("UcvhInitialUploadProgress"),
            "initial UCVH upload must retain progress between frames"
        );
    }

    #[test]
    fn initial_ucvh_upload_frame_budget_stays_below_unresponsive_frame_size() {
        let budget = std::hint::black_box(INITIAL_UCVH_UPLOAD_FRAME_BUDGET_BYTES);
        assert!(
            budget <= 64 * 1024 * 1024,
            "initial UCVH upload must not push hundreds of MB in one frame"
        );
    }

    #[test]
    fn ucvh_upload_barriers_make_buffers_visible_to_compute_and_ray_tracing_shaders() {
        let source = crate::render::source_checks::read_source("src/voxel/gpu_upload.rs");
        let upload_all = source
            .split("pub fn upload_all")
            .nth(1)
            .expect("UcvhGpuResources::upload_all should exist")
            .split("pub fn upload_motion_guide")
            .next()
            .expect("upload_all should end before upload_motion_guide");
        let upload_motion = source
            .split("pub fn upload_motion_guide")
            .nth(1)
            .expect("UcvhGpuResources::upload_motion_guide should exist")
            .split("fn copy_to_staging")
            .next()
            .expect("upload_motion_guide should end before copy helpers");

        let incremental_upload = source
            .split("pub(crate) fn upload_initial_incremental")
            .nth(1)
            .expect("UcvhGpuResources::upload_initial_incremental should exist")
            .split("pub fn upload_motion_guide")
            .next()
            .expect("upload_initial_incremental should end before upload_motion_guide");
        let barrier_helper = source
            .split("fn record_upload_barrier")
            .nth(1)
            .expect("record_upload_barrier should exist")
            .split("fn copy_to_staging")
            .next()
            .expect("record_upload_barrier should end before copy helpers");

        for body in [upload_all, incremental_upload, upload_motion] {
            assert!(
                body.contains("record_upload_barrier"),
                "each UCVH upload path must emit a transfer-to-shader barrier"
            );
        }
        assert!(barrier_helper.contains("PipelineStageFlags::COMPUTE_SHADER"));
        assert!(
            barrier_helper.contains("PipelineStageFlags::RAY_TRACING_SHADER_KHR"),
            "UCVH upload barriers must make copied buffers visible to RT shaders"
        );
        let compact_barrier = crate::render::source_checks::compact(barrier_helper);
        assert!(compact_barrier.contains(
            "dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,)"
        ));
    }

    #[test]
    fn incremental_upload_records_only_changed_bricks_and_hierarchy_ancestors() {
        let source = crate::render::source_checks::read_source("src/voxel/gpu_upload.rs");
        let upload = source
            .split("pub fn upload_incremental_changes")
            .nth(1)
            .expect("incremental UCVH upload should exist")
            .split("/// Upload all UCVH data")
            .next()
            .expect("incremental upload should end before full upload");
        let compact = crate::render::source_checks::compact(upload);

        for token in [
            "UcvhIncrementalUploadPlan::for_bricks(",
            "ucvh.update_hierarchy_for_render_change_batch(batch)",
            "upload_incremental_hierarchy(",
            "Self::record_upload_barrier(device,cmd)",
        ] {
            assert!(
                compact.contains(token),
                "incremental upload must include {token}"
            );
        }
        assert!(
            !upload.contains("initial_generation_upload_slice(ucvh)"),
            "ordinary edits must not copy the whole generation prefix"
        );
    }

    #[test]
    fn incremental_upload_uses_frame_slot_staging_and_never_rejects_nonzero_slots() {
        let source = crate::render::source_checks::read_source("src/voxel/gpu_upload.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("implementation should precede tests");
        let upload = implementation
            .split("pub fn upload_incremental_changes")
            .nth(1)
            .expect("incremental UCVH upload should exist")
            .split("/// Upload all UCVH data")
            .next()
            .expect("incremental upload should end before full upload");

        assert!(
            implementation.contains("staging: Vec<UcvhStagingResources>"),
            "UCVH uploader must own one complete staging set per frame slot"
        );
        assert!(
            upload.contains("self.staging_for_frame_slot(frame_slot)?"),
            "incremental uploads must select a frame-slot-owned staging set"
        );
        assert!(
            !upload.contains("if frame_slot != 0"),
            "incremental uploads must not reject nonzero frame slots"
        );
    }

    #[test]
    fn ucvh_motion_event_shader_layout_matches_gpu_upload_abi() {
        assert_eq!(std::mem::size_of::<GpuUcvhMotionEvent>(), 64);
        assert_eq!(std::mem::offset_of!(GpuUcvhMotionEvent, region_min), 0);
        assert_eq!(
            std::mem::offset_of!(GpuUcvhMotionEvent, region_max_exclusive),
            16
        );
        assert_eq!(
            std::mem::offset_of!(GpuUcvhMotionEvent, world_delta_current_from_previous),
            32
        );
        assert_eq!(std::mem::offset_of!(GpuUcvhMotionEvent, generation), 48);

        let shader = crate::render::source_checks::read_source(
            "assets/shaders/shared/vpt_motion_common.slang",
        );
        for token in [
            "uint4 region_min",
            "uint4 region_max_exclusive",
            "int4 world_delta_current_from_previous",
            "uint generation",
            "uint3 _pad",
            "event.region_min.xyz",
            "event.region_max_exclusive.xyz",
        ] {
            assert!(
                shader.contains(token),
                "vpt_motion_common.slang must match GpuUcvhMotionEvent ABI token {token}"
            );
        }
    }

    #[test]
    fn motion_event_buffer_overflow_drops_tail() {
        assert_eq!(capped_motion_event_upload_count(0), 0);
        assert_eq!(
            capped_motion_event_upload_count(UCVH_MOTION_EVENT_CAPACITY),
            UCVH_MOTION_EVENT_CAPACITY
        );
        assert_eq!(
            capped_motion_event_upload_count(UCVH_MOTION_EVENT_CAPACITY + 1),
            UCVH_MOTION_EVENT_CAPACITY
        );
    }
}
