use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::rt_page_geometry::{
    RT_PAGE_LATTICE_VERTEX_COUNT, RtCompactFaceRecord, RtCompactPageGeometry, lattice_vertex,
};
use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

pub const RT_PAGE_LATTICE_VERTEX_STRIDE: u64 = std::mem::size_of::<[f32; 3]>() as u64;
pub const RT_PAGE_INDICES_PER_FACE: u64 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageGpuConfig {
    pub index_capacity_bytes: u64,
    pub face_capacity_records: u32,
    pub page_record_capacity: u32,
    pub staging_bytes_per_frame: u64,
    pub frame_slot_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageGpuBufferSizes {
    pub lattice_bytes: u64,
    pub index_bytes: u64,
    pub face_bytes: u64,
    pub page_record_bytes: u64,
    pub staging_bytes_per_frame: u64,
    pub frame_slot_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RtPageGpuConfigError {
    #[error("RT page GPU resources require at least one frame slot")]
    NoFrameSlots,
    #[error("RT page GPU resources require a nonzero index arena")]
    EmptyIndexArena,
    #[error("RT page GPU resources require a nonzero face arena")]
    EmptyFaceArena,
    #[error("RT page GPU resources require at least the dummy page-record slot")]
    EmptyPageRecordArena,
    #[error("RT page index capacity must be aligned to 16-bit indices: {capacity_bytes}")]
    MisalignedIndexCapacity { capacity_bytes: u64 },
    #[error("RT page GPU buffer size arithmetic overflow")]
    SizeOverflow,
    #[error(
        "RT page staging cannot hold the immutable lattice: staging={staging_bytes} lattice={lattice_bytes}"
    )]
    StagingTooSmall {
        staging_bytes: u64,
        lattice_bytes: u64,
    },
}

impl RtPageGpuConfig {
    pub fn validate(self) -> std::result::Result<RtPageGpuBufferSizes, RtPageGpuConfigError> {
        if self.frame_slot_count == 0 {
            return Err(RtPageGpuConfigError::NoFrameSlots);
        }
        if self.index_capacity_bytes == 0 {
            return Err(RtPageGpuConfigError::EmptyIndexArena);
        }
        if self.face_capacity_records == 0 {
            return Err(RtPageGpuConfigError::EmptyFaceArena);
        }
        if self.page_record_capacity == 0 {
            return Err(RtPageGpuConfigError::EmptyPageRecordArena);
        }
        if !self
            .index_capacity_bytes
            .is_multiple_of(std::mem::size_of::<u16>() as u64)
        {
            return Err(RtPageGpuConfigError::MisalignedIndexCapacity {
                capacity_bytes: self.index_capacity_bytes,
            });
        }
        let lattice_bytes = (RT_PAGE_LATTICE_VERTEX_COUNT as u64)
            .checked_mul(RT_PAGE_LATTICE_VERTEX_STRIDE)
            .ok_or(RtPageGpuConfigError::SizeOverflow)?;
        let face_bytes = u64::from(self.face_capacity_records)
            .checked_mul(std::mem::size_of::<RtCompactFaceRecord>() as u64)
            .ok_or(RtPageGpuConfigError::SizeOverflow)?;
        let page_record_bytes = u64::from(self.page_record_capacity)
            .checked_mul(std::mem::size_of::<GpuRtPageRecord>() as u64)
            .ok_or(RtPageGpuConfigError::SizeOverflow)?;
        if self.staging_bytes_per_frame < lattice_bytes {
            return Err(RtPageGpuConfigError::StagingTooSmall {
                staging_bytes: self.staging_bytes_per_frame,
                lattice_bytes,
            });
        }
        Ok(RtPageGpuBufferSizes {
            lattice_bytes,
            index_bytes: self.index_capacity_bytes,
            face_bytes,
            page_record_bytes,
            staging_bytes_per_frame: self.staging_bytes_per_frame,
            frame_slot_count: self.frame_slot_count,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuRtPageRecord {
    pub brick_id: u32,
    pub face_record_offset: u32,
    pub face_count: u32,
    pub representation: u32,
    pub page_coord: [u32; 4],
    pub topology_revision: u64,
    pub resource_version: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageGeometryAllocation {
    pub index_offset_bytes: u64,
    pub index_size_bytes: u64,
    pub face_offset_records: u32,
    pub face_count: u32,
    pub allocation_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RtPageArenaError {
    #[error("RT page geometry requires at least one face")]
    ZeroFaceCount,
    #[error("RT page index capacity must be aligned to 16-bit indices: {capacity_bytes}")]
    MisalignedIndexCapacity { capacity_bytes: u64 },
    #[error("RT page index byte count overflow for {face_count} faces")]
    IndexSizeOverflow { face_count: u32 },
    #[error("RT page index arena is exhausted: requested={requested_bytes}")]
    IndexArenaExhausted { requested_bytes: u64 },
    #[error("RT page face arena is exhausted: requested={requested_records}")]
    FaceArenaExhausted { requested_records: u32 },
    #[error("RT page allocation ID space is exhausted")]
    AllocationIdExhausted,
    #[error("unknown RT page geometry allocation {allocation_id}")]
    UnknownAllocation { allocation_id: u64 },
    #[error("RT page geometry allocation {allocation_id} does not match its active generation")]
    AllocationMismatch { allocation_id: u64 },
}

pub fn rt_page_index_bytes(face_count: u32) -> Result<u64, RtPageArenaError> {
    if face_count == 0 {
        return Err(RtPageArenaError::ZeroFaceCount);
    }
    u64::from(face_count)
        .checked_mul(RT_PAGE_INDICES_PER_FACE)
        .and_then(|indices| indices.checked_mul(std::mem::size_of::<u16>() as u64))
        .ok_or(RtPageArenaError::IndexSizeOverflow { face_count })
}

pub fn shared_rt_page_lattice_vertices() -> [[f32; 3]; RT_PAGE_LATTICE_VERTEX_COUNT] {
    std::array::from_fn(|index| {
        let vertex = lattice_vertex(index as u16);
        [vertex.x as f32, vertex.y as f32, vertex.z as f32]
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRange {
    offset: u64,
    size: u64,
}

#[derive(Debug)]
struct FreeRangeAllocator {
    capacity: u64,
    free_by_offset: BTreeMap<u64, u64>,
}

impl FreeRangeAllocator {
    fn new(capacity: u64) -> Self {
        let mut free_by_offset = BTreeMap::new();
        if capacity != 0 {
            free_by_offset.insert(0, capacity);
        }
        Self {
            capacity,
            free_by_offset,
        }
    }

    fn allocate(&mut self, size: u64, alignment: u64) -> Option<FreeRange> {
        if size == 0 || alignment == 0 {
            return None;
        }
        let candidate = self
            .free_by_offset
            .iter()
            .find_map(|(&range_offset, &range_size)| {
                let aligned_offset = checked_align_up(range_offset, alignment)?;
                let range_end = range_offset.checked_add(range_size)?;
                let allocation_end = aligned_offset.checked_add(size)?;
                (allocation_end <= range_end).then_some((
                    range_offset,
                    range_size,
                    aligned_offset,
                    allocation_end,
                    range_end,
                ))
            })?;
        let (range_offset, _, aligned_offset, allocation_end, range_end) = candidate;
        self.free_by_offset.remove(&range_offset);
        if range_offset < aligned_offset {
            self.free_by_offset
                .insert(range_offset, aligned_offset - range_offset);
        }
        if allocation_end < range_end {
            self.free_by_offset
                .insert(allocation_end, range_end - allocation_end);
        }
        Some(FreeRange {
            offset: aligned_offset,
            size,
        })
    }

    fn free(&mut self, range: FreeRange) -> Result<(), ()> {
        let end = range.offset.checked_add(range.size).ok_or(())?;
        if range.size == 0 || end > self.capacity {
            return Err(());
        }

        let previous = self
            .free_by_offset
            .range(..=range.offset)
            .next_back()
            .map(|(&offset, &size)| (offset, size));
        if previous.is_some_and(|(offset, size)| offset + size > range.offset) {
            return Err(());
        }
        let next = self
            .free_by_offset
            .range(range.offset..)
            .next()
            .map(|(&offset, &size)| (offset, size));
        if next.is_some_and(|(offset, _)| end > offset) {
            return Err(());
        }

        let mut merged_offset = range.offset;
        let mut merged_end = end;
        if let Some((offset, size)) = previous
            && offset + size == range.offset
        {
            self.free_by_offset.remove(&offset);
            merged_offset = offset;
        }
        if let Some((offset, size)) = next
            && end == offset
        {
            self.free_by_offset.remove(&offset);
            merged_end = offset + size;
        }
        self.free_by_offset
            .insert(merged_offset, merged_end - merged_offset);
        Ok(())
    }
}

#[derive(Debug)]
pub struct RtPageGeometryArena {
    index_ranges: FreeRangeAllocator,
    face_ranges: FreeRangeAllocator,
    active: HashMap<u64, RtPageGeometryAllocation>,
    next_allocation_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RtPageGeometryUploadError {
    #[error("RT page geometry allocation is not active: {allocation_id}")]
    InactiveAllocation { allocation_id: u64 },
    #[error("RT page face count mismatch: expected={expected} actual={actual}")]
    FaceCountMismatch { expected: u32, actual: usize },
    #[error("RT page index count mismatch: expected={expected} actual={actual}")]
    IndexCountMismatch { expected: usize, actual: usize },
    #[error("RT page geometry contains lattice index {index} outside {vertex_count} vertices")]
    LatticeIndexOutOfRange { index: u16, vertex_count: usize },
}

impl RtPageGeometryArena {
    pub fn new(
        index_capacity_bytes: u64,
        face_capacity_records: u32,
    ) -> Result<Self, RtPageArenaError> {
        if !index_capacity_bytes.is_multiple_of(std::mem::size_of::<u16>() as u64) {
            return Err(RtPageArenaError::MisalignedIndexCapacity {
                capacity_bytes: index_capacity_bytes,
            });
        }
        Ok(Self {
            index_ranges: FreeRangeAllocator::new(index_capacity_bytes),
            face_ranges: FreeRangeAllocator::new(u64::from(face_capacity_records)),
            active: HashMap::new(),
            next_allocation_id: 1,
        })
    }

    pub fn allocate(
        &mut self,
        face_count: u32,
    ) -> Result<RtPageGeometryAllocation, RtPageArenaError> {
        let index_size_bytes = rt_page_index_bytes(face_count)?;
        let next_allocation_id = self
            .next_allocation_id
            .checked_add(1)
            .ok_or(RtPageArenaError::AllocationIdExhausted)?;
        let index_range = self
            .index_ranges
            .allocate(index_size_bytes, std::mem::size_of::<u16>() as u64)
            .ok_or(RtPageArenaError::IndexArenaExhausted {
                requested_bytes: index_size_bytes,
            })?;
        let Some(face_range) = self.face_ranges.allocate(u64::from(face_count), 1) else {
            self.index_ranges
                .free(index_range)
                .expect("newly allocated index range must roll back cleanly");
            return Err(RtPageArenaError::FaceArenaExhausted {
                requested_records: face_count,
            });
        };
        let allocation = RtPageGeometryAllocation {
            index_offset_bytes: index_range.offset,
            index_size_bytes: index_range.size,
            face_offset_records: u32::try_from(face_range.offset)
                .expect("face arena offset must fit its u32 capacity"),
            face_count,
            allocation_id: self.next_allocation_id,
        };
        self.next_allocation_id = next_allocation_id;
        self.active.insert(allocation.allocation_id, allocation);
        Ok(allocation)
    }

    pub fn free(&mut self, allocation: RtPageGeometryAllocation) -> Result<(), RtPageArenaError> {
        let Some(active) = self.active.get(&allocation.allocation_id).copied() else {
            return Err(RtPageArenaError::UnknownAllocation {
                allocation_id: allocation.allocation_id,
            });
        };
        if active != allocation {
            return Err(RtPageArenaError::AllocationMismatch {
                allocation_id: allocation.allocation_id,
            });
        }
        self.active.remove(&allocation.allocation_id);
        self.index_ranges
            .free(FreeRange {
                offset: allocation.index_offset_bytes,
                size: allocation.index_size_bytes,
            })
            .expect("active index allocation must return to its arena");
        self.face_ranges
            .free(FreeRange {
                offset: u64::from(allocation.face_offset_records),
                size: u64::from(allocation.face_count),
            })
            .expect("active face allocation must return to its arena");
        Ok(())
    }

    pub fn contains(&self, allocation: RtPageGeometryAllocation) -> bool {
        self.active.get(&allocation.allocation_id) == Some(&allocation)
    }

    pub fn active_allocation_count(&self) -> usize {
        self.active.len()
    }

    pub fn validate_upload(
        &self,
        allocation: RtPageGeometryAllocation,
        geometry: &RtCompactPageGeometry,
    ) -> std::result::Result<(), RtPageGeometryUploadError> {
        if !self.contains(allocation) {
            return Err(RtPageGeometryUploadError::InactiveAllocation {
                allocation_id: allocation.allocation_id,
            });
        }
        if geometry.faces.len() != allocation.face_count as usize {
            return Err(RtPageGeometryUploadError::FaceCountMismatch {
                expected: allocation.face_count,
                actual: geometry.faces.len(),
            });
        }
        let expected_indices = allocation.face_count as usize * RT_PAGE_INDICES_PER_FACE as usize;
        if geometry.indices.len() != expected_indices {
            return Err(RtPageGeometryUploadError::IndexCountMismatch {
                expected: expected_indices,
                actual: geometry.indices.len(),
            });
        }
        if let Some(&index) = geometry
            .indices
            .iter()
            .find(|&&index| usize::from(index) >= RT_PAGE_LATTICE_VERTEX_COUNT)
        {
            return Err(RtPageGeometryUploadError::LatticeIndexOutOfRange {
                index,
                vertex_count: RT_PAGE_LATTICE_VERTEX_COUNT,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RtPageRecordError {
    #[error("RT page record arena requires at least one slot")]
    EmptyArena,
    #[error("RT page record slot is out of range: slot={slot} capacity={capacity}")]
    SlotOutOfRange { slot: u32, capacity: u32 },
}

#[derive(Debug)]
pub struct RtPageRecordArena {
    records: Vec<GpuRtPageRecord>,
}

impl RtPageRecordArena {
    pub fn new(capacity: u32) -> Result<Self, RtPageRecordError> {
        if capacity == 0 {
            return Err(RtPageRecordError::EmptyArena);
        }
        Ok(Self {
            records: vec![GpuRtPageRecord::zeroed(); capacity as usize],
        })
    }

    pub fn capacity(&self) -> u32 {
        self.records.len() as u32
    }

    pub fn write(&mut self, slot: u32, record: GpuRtPageRecord) -> Result<(), RtPageRecordError> {
        let capacity = self.capacity();
        let target = self
            .records
            .get_mut(slot as usize)
            .ok_or(RtPageRecordError::SlotOutOfRange { slot, capacity })?;
        *target = record;
        Ok(())
    }

    pub fn get(&self, slot: u32) -> Option<GpuRtPageRecord> {
        self.records.get(slot as usize).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RtPageRecordJournalError {
    #[error("RT page-record journal requires at least one frame slot")]
    NoFrameSlots,
    #[error("RT page-record journal requires at least one page-record slot")]
    EmptyRecordCapacity,
    #[error("RT page-record journal frame slot is out of range: {frame_slot}")]
    FrameSlotOutOfRange { frame_slot: usize },
    #[error("RT page-record journal slot is out of range: slot={slot} capacity={capacity}")]
    RecordSlotOutOfRange { slot: u32, capacity: u32 },
    #[error("RT page-record journal frame mismatch: expected={expected} actual={actual}")]
    FrameMismatch { expected: u64, actual: u64 },
    #[error("RT page-record journal capacity mismatch: expected={expected} actual={actual}")]
    RecordCapacityMismatch { expected: u32, actual: u32 },
}

#[derive(Debug, Default)]
struct RtPageRecordJournalSlot {
    frame_index: Option<u64>,
    writes: BTreeMap<u32, GpuRtPageRecord>,
}

#[derive(Debug)]
pub struct RtPageRecordJournal {
    record_capacity: u32,
    slots: Vec<RtPageRecordJournalSlot>,
}

impl RtPageRecordJournal {
    pub fn new(
        frame_slot_count: usize,
        record_capacity: u32,
    ) -> std::result::Result<Self, RtPageRecordJournalError> {
        if frame_slot_count == 0 {
            return Err(RtPageRecordJournalError::NoFrameSlots);
        }
        if record_capacity == 0 {
            return Err(RtPageRecordJournalError::EmptyRecordCapacity);
        }
        Ok(Self {
            record_capacity,
            slots: (0..frame_slot_count)
                .map(|_| RtPageRecordJournalSlot::default())
                .collect(),
        })
    }

    pub fn stage(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
        page_slot: u32,
        record: GpuRtPageRecord,
    ) -> std::result::Result<(), RtPageRecordJournalError> {
        if page_slot >= self.record_capacity {
            return Err(RtPageRecordJournalError::RecordSlotOutOfRange {
                slot: page_slot,
                capacity: self.record_capacity,
            });
        }
        let slot = self
            .slots
            .get_mut(frame_slot)
            .ok_or(RtPageRecordJournalError::FrameSlotOutOfRange { frame_slot })?;
        if let Some(expected) = slot.frame_index
            && expected != frame_index
        {
            return Err(RtPageRecordJournalError::FrameMismatch {
                expected,
                actual: frame_index,
            });
        }
        slot.frame_index = Some(frame_index);
        slot.writes.insert(page_slot, record);
        Ok(())
    }

    pub fn commit(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
        records: &mut RtPageRecordArena,
    ) -> std::result::Result<(), RtPageRecordJournalError> {
        if records.capacity() != self.record_capacity {
            return Err(RtPageRecordJournalError::RecordCapacityMismatch {
                expected: self.record_capacity,
                actual: records.capacity(),
            });
        }
        let slot = self
            .slots
            .get_mut(frame_slot)
            .ok_or(RtPageRecordJournalError::FrameSlotOutOfRange { frame_slot })?;
        if let Some(expected) = slot.frame_index
            && expected != frame_index
        {
            return Err(RtPageRecordJournalError::FrameMismatch {
                expected,
                actual: frame_index,
            });
        }
        for (page_slot, record) in std::mem::take(&mut slot.writes) {
            records
                .write(page_slot, record)
                .expect("journal validated every page-record slot before commit");
        }
        slot.frame_index = None;
        Ok(())
    }

    pub fn cancel(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
    ) -> std::result::Result<(), RtPageRecordJournalError> {
        let slot = self
            .slots
            .get_mut(frame_slot)
            .ok_or(RtPageRecordJournalError::FrameSlotOutOfRange { frame_slot })?;
        if let Some(expected) = slot.frame_index
            && expected != frame_index
        {
            return Err(RtPageRecordJournalError::FrameMismatch {
                expected,
                actual: frame_index,
            });
        }
        slot.frame_index = None;
        slot.writes.clear();
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageStagingRange {
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RtPageStagingSlotState {
    Idle,
    Recording { frame_index: u64, cursor: u64 },
    InFlight { frame_index: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RtPageStagingError {
    #[error("RT page staging requires at least one frame slot")]
    NoFrameSlots,
    #[error("RT page staging capacity must be nonzero")]
    EmptyCapacity,
    #[error("RT page staging frame slot is out of range: {frame_slot}")]
    FrameSlotOutOfRange { frame_slot: usize },
    #[error("RT page staging frame slot {frame_slot} is already recording frame {frame_index}")]
    FrameSlotRecording { frame_slot: usize, frame_index: u64 },
    #[error("RT page staging frame slot {frame_slot} is still in flight for frame {frame_index}")]
    FrameSlotInFlight { frame_slot: usize, frame_index: u64 },
    #[error("RT page staging frame mismatch: expected={expected} actual={actual}")]
    FrameMismatch { expected: u64, actual: u64 },
    #[error("RT page staging frame slot {frame_slot} is idle")]
    FrameSlotIdle { frame_slot: usize },
    #[error("invalid RT page staging alignment: {alignment}")]
    InvalidAlignment { alignment: u64 },
    #[error("RT page staging range overflow")]
    RangeOverflow,
    #[error("RT page staging capacity exceeded: required={required} capacity={capacity}")]
    CapacityExceeded { required: u64, capacity: u64 },
}

#[derive(Debug)]
pub struct RtPageStagingTracker {
    capacity_bytes: u64,
    slots: Vec<RtPageStagingSlotState>,
}

impl RtPageStagingTracker {
    pub fn new(frame_slot_count: usize, capacity_bytes: u64) -> Result<Self, RtPageStagingError> {
        if frame_slot_count == 0 {
            return Err(RtPageStagingError::NoFrameSlots);
        }
        if capacity_bytes == 0 {
            return Err(RtPageStagingError::EmptyCapacity);
        }
        Ok(Self {
            capacity_bytes,
            slots: vec![RtPageStagingSlotState::Idle; frame_slot_count],
        })
    }

    pub fn begin_frame_slot(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
    ) -> Result<(), RtPageStagingError> {
        let state = self
            .slots
            .get_mut(frame_slot)
            .ok_or(RtPageStagingError::FrameSlotOutOfRange { frame_slot })?;
        match *state {
            RtPageStagingSlotState::Idle => {
                *state = RtPageStagingSlotState::Recording {
                    frame_index,
                    cursor: 0,
                };
                Ok(())
            }
            RtPageStagingSlotState::Recording { frame_index, .. } => {
                Err(RtPageStagingError::FrameSlotRecording {
                    frame_slot,
                    frame_index,
                })
            }
            RtPageStagingSlotState::InFlight { frame_index } => {
                Err(RtPageStagingError::FrameSlotInFlight {
                    frame_slot,
                    frame_index,
                })
            }
        }
    }

    pub fn reserve_ranges(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
        ranges: &[(u64, u64)],
    ) -> Result<Vec<RtPageStagingRange>, RtPageStagingError> {
        let state = self
            .slots
            .get_mut(frame_slot)
            .ok_or(RtPageStagingError::FrameSlotOutOfRange { frame_slot })?;
        let RtPageStagingSlotState::Recording {
            frame_index: active_frame,
            cursor,
        } = state
        else {
            return match *state {
                RtPageStagingSlotState::Idle => {
                    Err(RtPageStagingError::FrameSlotIdle { frame_slot })
                }
                RtPageStagingSlotState::InFlight { frame_index } => {
                    Err(RtPageStagingError::FrameSlotInFlight {
                        frame_slot,
                        frame_index,
                    })
                }
                RtPageStagingSlotState::Recording { .. } => unreachable!(),
            };
        };
        if *active_frame != frame_index {
            return Err(RtPageStagingError::FrameMismatch {
                expected: *active_frame,
                actual: frame_index,
            });
        }

        let mut planned_cursor = *cursor;
        let mut planned = Vec::with_capacity(ranges.len());
        for &(size, alignment) in ranges {
            if alignment == 0 {
                return Err(RtPageStagingError::InvalidAlignment { alignment });
            }
            let offset = checked_align_up(planned_cursor, alignment)
                .ok_or(RtPageStagingError::RangeOverflow)?;
            let end = offset
                .checked_add(size)
                .ok_or(RtPageStagingError::RangeOverflow)?;
            if end > self.capacity_bytes {
                return Err(RtPageStagingError::CapacityExceeded {
                    required: end,
                    capacity: self.capacity_bytes,
                });
            }
            planned.push(RtPageStagingRange { offset, size });
            planned_cursor = end;
        }
        *cursor = planned_cursor;
        Ok(planned)
    }

    pub fn finish_frame_slot(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
    ) -> Result<(), RtPageStagingError> {
        let state = self
            .slots
            .get_mut(frame_slot)
            .ok_or(RtPageStagingError::FrameSlotOutOfRange { frame_slot })?;
        match *state {
            RtPageStagingSlotState::Recording {
                frame_index: active_frame,
                ..
            } if active_frame == frame_index => {
                *state = RtPageStagingSlotState::InFlight { frame_index };
                Ok(())
            }
            RtPageStagingSlotState::Recording {
                frame_index: active_frame,
                ..
            } => Err(RtPageStagingError::FrameMismatch {
                expected: active_frame,
                actual: frame_index,
            }),
            RtPageStagingSlotState::Idle => Err(RtPageStagingError::FrameSlotIdle { frame_slot }),
            RtPageStagingSlotState::InFlight { frame_index } => {
                Err(RtPageStagingError::FrameSlotInFlight {
                    frame_slot,
                    frame_index,
                })
            }
        }
    }

    pub fn complete_frame_slot(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
    ) -> Result<(), RtPageStagingError> {
        let state = self
            .slots
            .get_mut(frame_slot)
            .ok_or(RtPageStagingError::FrameSlotOutOfRange { frame_slot })?;
        match *state {
            RtPageStagingSlotState::InFlight {
                frame_index: active_frame,
            } if active_frame == frame_index => {
                *state = RtPageStagingSlotState::Idle;
                Ok(())
            }
            RtPageStagingSlotState::InFlight {
                frame_index: active_frame,
            } => Err(RtPageStagingError::FrameMismatch {
                expected: active_frame,
                actual: frame_index,
            }),
            RtPageStagingSlotState::Recording { frame_index, .. } => {
                Err(RtPageStagingError::FrameSlotRecording {
                    frame_slot,
                    frame_index,
                })
            }
            RtPageStagingSlotState::Idle => Err(RtPageStagingError::FrameSlotIdle { frame_slot }),
        }
    }

    pub fn cancel_frame_slot(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
    ) -> Result<(), RtPageStagingError> {
        let state = self
            .slots
            .get_mut(frame_slot)
            .ok_or(RtPageStagingError::FrameSlotOutOfRange { frame_slot })?;
        match *state {
            RtPageStagingSlotState::Recording {
                frame_index: active_frame,
                ..
            } if active_frame == frame_index => {
                *state = RtPageStagingSlotState::Idle;
                Ok(())
            }
            RtPageStagingSlotState::Recording {
                frame_index: active_frame,
                ..
            } => Err(RtPageStagingError::FrameMismatch {
                expected: active_frame,
                actual: frame_index,
            }),
            RtPageStagingSlotState::InFlight { frame_index } => {
                Err(RtPageStagingError::FrameSlotInFlight {
                    frame_slot,
                    frame_index,
                })
            }
            RtPageStagingSlotState::Idle => Err(RtPageStagingError::FrameSlotIdle { frame_slot }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPageUploadTarget {
    Lattice,
    Indices,
    Faces,
    PageRecords,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageUploadBarrier {
    pub target: RtPageUploadTarget,
    pub offset: u64,
    pub size: u64,
    pub src_stage: vk::PipelineStageFlags,
    pub src_access: vk::AccessFlags,
    pub dst_stage: vk::PipelineStageFlags,
    pub dst_access: vk::AccessFlags,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RtPageUploadPlanError {
    #[error("RT page upload range arithmetic overflow")]
    ArithmeticOverflow,
}

pub fn rt_page_geometry_upload_barriers(
    allocation: RtPageGeometryAllocation,
    page_slot: u32,
) -> Result<[RtPageUploadBarrier; 3], RtPageUploadPlanError> {
    let face_offset = u64::from(allocation.face_offset_records)
        .checked_mul(std::mem::size_of::<RtCompactFaceRecord>() as u64)
        .ok_or(RtPageUploadPlanError::ArithmeticOverflow)?;
    let face_size = u64::from(allocation.face_count)
        .checked_mul(std::mem::size_of::<RtCompactFaceRecord>() as u64)
        .ok_or(RtPageUploadPlanError::ArithmeticOverflow)?;
    let page_record_offset = u64::from(page_slot)
        .checked_mul(std::mem::size_of::<GpuRtPageRecord>() as u64)
        .ok_or(RtPageUploadPlanError::ArithmeticOverflow)?;
    let transfer_stage = vk::PipelineStageFlags::TRANSFER;
    let transfer_write = vk::AccessFlags::TRANSFER_WRITE;
    Ok([
        RtPageUploadBarrier {
            target: RtPageUploadTarget::Indices,
            offset: allocation.index_offset_bytes,
            size: allocation.index_size_bytes,
            src_stage: transfer_stage,
            src_access: transfer_write,
            dst_stage: vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            dst_access: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        },
        RtPageUploadBarrier {
            target: RtPageUploadTarget::Faces,
            offset: face_offset,
            size: face_size,
            src_stage: transfer_stage,
            src_access: transfer_write,
            dst_stage: vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            dst_access: vk::AccessFlags::SHADER_READ,
        },
        RtPageUploadBarrier {
            target: RtPageUploadTarget::PageRecords,
            offset: page_record_offset,
            size: std::mem::size_of::<GpuRtPageRecord>() as u64,
            src_stage: transfer_stage,
            src_access: transfer_write,
            dst_stage: vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            dst_access: vk::AccessFlags::SHADER_READ,
        },
    ])
}

pub fn rt_page_record_upload_barrier(
    page_slot: u32,
) -> std::result::Result<RtPageUploadBarrier, RtPageUploadPlanError> {
    let offset = u64::from(page_slot)
        .checked_mul(std::mem::size_of::<GpuRtPageRecord>() as u64)
        .ok_or(RtPageUploadPlanError::ArithmeticOverflow)?;
    Ok(RtPageUploadBarrier {
        target: RtPageUploadTarget::PageRecords,
        offset,
        size: std::mem::size_of::<GpuRtPageRecord>() as u64,
        src_stage: vk::PipelineStageFlags::TRANSFER,
        src_access: vk::AccessFlags::TRANSFER_WRITE,
        dst_stage: vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        dst_access: vk::AccessFlags::SHADER_READ,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RtPageLatticeUploadState {
    Pending,
    Recording { frame_slot: usize, frame_index: u64 },
    Submitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageGeometryUploadReceipt {
    pub allocation_id: u64,
    pub frame_slot: usize,
    pub frame_index: u64,
    pub staging_index_range: RtPageStagingRange,
    pub staging_face_range: RtPageStagingRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageRecordUploadReceipt {
    pub page_slot: u32,
    pub frame_slot: usize,
    pub frame_index: u64,
    pub staging_range: RtPageStagingRange,
}

pub struct RtPageGpuResources {
    lattice_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    face_buffer: GpuBuffer,
    page_record_buffer: GpuBuffer,
    staging_buffers: Vec<GpuBuffer>,
    geometry_arena: RtPageGeometryArena,
    page_records: RtPageRecordArena,
    page_record_journal: RtPageRecordJournal,
    staging_tracker: RtPageStagingTracker,
    lattice_upload_state: RtPageLatticeUploadState,
}

impl RtPageGpuResources {
    pub fn lattice_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::TRANSFER_DST
    }

    pub fn index_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::TRANSFER_DST
    }

    pub fn record_buffer_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        config: RtPageGpuConfig,
    ) -> Result<Self> {
        let sizes = config.validate().map_err(anyhow::Error::new)?;
        let geometry_arena =
            RtPageGeometryArena::new(config.index_capacity_bytes, config.face_capacity_records)
                .map_err(anyhow::Error::new)?;
        let page_records =
            RtPageRecordArena::new(config.page_record_capacity).map_err(anyhow::Error::new)?;
        let page_record_journal =
            RtPageRecordJournal::new(config.frame_slot_count, config.page_record_capacity)
                .map_err(anyhow::Error::new)?;
        let staging_tracker =
            RtPageStagingTracker::new(config.frame_slot_count, config.staging_bytes_per_frame)
                .map_err(anyhow::Error::new)?;

        let mut buffers = GpuBufferBuildGuard::new(device, allocator);
        buffers.create(
            sizes.lattice_bytes,
            Self::lattice_buffer_usage(),
            MemoryLocation::GpuOnly,
            "rt_page_lattice",
        )?;
        buffers.create(
            sizes.index_bytes,
            Self::index_buffer_usage(),
            MemoryLocation::GpuOnly,
            "rt_page_indices",
        )?;
        buffers.create(
            sizes.face_bytes,
            Self::record_buffer_usage(),
            MemoryLocation::GpuOnly,
            "rt_page_faces",
        )?;
        buffers.create(
            sizes.page_record_bytes,
            Self::record_buffer_usage(),
            MemoryLocation::GpuOnly,
            "rt_page_records",
        )?;
        for frame_slot in 0..sizes.frame_slot_count {
            buffers.create(
                sizes.staging_bytes_per_frame,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                &format!("rt_page_staging_{frame_slot}"),
            )?;
        }

        let lattice_address = buffers.buffers[0]
            .device_address(device)
            .context("failed to query RT page lattice device address")?;
        ensure_device_address_alignment(
            lattice_address,
            std::mem::align_of::<f32>() as u64,
            "RT page lattice",
        )?;
        let index_address = buffers.buffers[1]
            .device_address(device)
            .context("failed to query RT page index arena device address")?;
        ensure_device_address_alignment(
            index_address,
            std::mem::align_of::<u16>() as u64,
            "RT page index arena",
        )?;

        let mut created = buffers.finish().into_iter();
        let lattice_buffer = created.next().expect("lattice buffer must exist");
        let index_buffer = created.next().expect("index buffer must exist");
        let face_buffer = created.next().expect("face buffer must exist");
        let page_record_buffer = created.next().expect("page-record buffer must exist");
        let staging_buffers = created.collect::<Vec<_>>();
        debug_assert_eq!(staging_buffers.len(), sizes.frame_slot_count);

        Ok(Self {
            lattice_buffer,
            index_buffer,
            face_buffer,
            page_record_buffer,
            staging_buffers,
            geometry_arena,
            page_records,
            page_record_journal,
            staging_tracker,
            lattice_upload_state: RtPageLatticeUploadState::Pending,
        })
    }

    pub fn lattice_buffer(&self) -> &GpuBuffer {
        &self.lattice_buffer
    }

    pub fn index_buffer(&self) -> &GpuBuffer {
        &self.index_buffer
    }

    pub fn face_buffer(&self) -> &GpuBuffer {
        &self.face_buffer
    }

    pub fn page_record_buffer(&self) -> &GpuBuffer {
        &self.page_record_buffer
    }

    pub fn allocate_geometry(
        &mut self,
        face_count: u32,
    ) -> std::result::Result<RtPageGeometryAllocation, RtPageArenaError> {
        self.geometry_arena.allocate(face_count)
    }

    pub fn free_geometry(
        &mut self,
        allocation: RtPageGeometryAllocation,
    ) -> std::result::Result<(), RtPageArenaError> {
        self.geometry_arena.free(allocation)
    }

    pub fn lattice_device_address(&self, device: &ash::Device) -> Result<vk::DeviceAddress> {
        let address = self.lattice_buffer.device_address(device)?;
        ensure_device_address_alignment(
            address,
            std::mem::align_of::<f32>() as u64,
            "RT page lattice",
        )?;
        Ok(address)
    }

    pub fn index_device_address(
        &self,
        device: &ash::Device,
        allocation: RtPageGeometryAllocation,
    ) -> Result<vk::DeviceAddress> {
        if !self.geometry_arena.contains(allocation) {
            bail!(
                "RT page index address requested for inactive allocation {}",
                allocation.allocation_id
            );
        }
        let address = self
            .index_buffer
            .device_address(device)?
            .checked_add(allocation.index_offset_bytes)
            .ok_or_else(|| anyhow!("RT page index device address overflow"))?;
        ensure_device_address_alignment(
            address,
            std::mem::align_of::<u16>() as u64,
            "RT page index allocation",
        )?;
        Ok(address)
    }

    pub fn begin_frame_slot(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
    ) -> std::result::Result<(), RtPageStagingError> {
        self.staging_tracker
            .begin_frame_slot(frame_slot, frame_index)
    }

    pub fn finish_frame_slot(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
    ) -> std::result::Result<(), RtPageStagingError> {
        self.staging_tracker
            .finish_frame_slot(frame_slot, frame_index)?;
        self.page_record_journal
            .commit(frame_slot, frame_index, &mut self.page_records)
            .expect("page-record journal and staging tracker must share frame ownership");
        if self.lattice_upload_state
            == (RtPageLatticeUploadState::Recording {
                frame_slot,
                frame_index,
            })
        {
            self.lattice_upload_state = RtPageLatticeUploadState::Submitted;
        }
        Ok(())
    }

    pub fn complete_frame_slot(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
    ) -> std::result::Result<(), RtPageStagingError> {
        self.staging_tracker
            .complete_frame_slot(frame_slot, frame_index)
    }

    pub fn cancel_frame_slot(
        &mut self,
        frame_slot: usize,
        frame_index: u64,
    ) -> std::result::Result<(), RtPageStagingError> {
        self.staging_tracker
            .cancel_frame_slot(frame_slot, frame_index)?;
        self.page_record_journal
            .cancel(frame_slot, frame_index)
            .expect("page-record journal and staging tracker must share frame ownership");
        if self.lattice_upload_state
            == (RtPageLatticeUploadState::Recording {
                frame_slot,
                frame_index,
            })
        {
            self.lattice_upload_state = RtPageLatticeUploadState::Pending;
        }
        Ok(())
    }

    pub fn record_lattice_upload(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
        frame_index: u64,
    ) -> Result<RtPageStagingRange> {
        if self.lattice_upload_state != RtPageLatticeUploadState::Pending {
            bail!("RT page lattice upload was already recorded or submitted");
        }
        let vertices = shared_rt_page_lattice_vertices();
        let bytes = bytemuck::cast_slice::<[f32; 3], u8>(&vertices);
        let range = self
            .staging_tracker
            .reserve_ranges(
                frame_slot,
                frame_index,
                &[(bytes.len() as u64, std::mem::align_of::<f32>() as u64)],
            )
            .map_err(anyhow::Error::new)?
            .into_iter()
            .next()
            .expect("one lattice staging range must be reserved");
        let staging = self.staging_buffer(frame_slot)?;
        write_staging_bytes(staging, range, bytes)?;
        record_buffer_copy(
            device,
            command_buffer,
            staging,
            &self.lattice_buffer,
            range.offset,
            0,
            range.size,
        );
        let barrier = RtPageUploadBarrier {
            target: RtPageUploadTarget::Lattice,
            offset: 0,
            size: bytes.len() as u64,
            src_stage: vk::PipelineStageFlags::TRANSFER,
            src_access: vk::AccessFlags::TRANSFER_WRITE,
            dst_stage: vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            dst_access: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        };
        self.record_upload_barriers(device, command_buffer, std::slice::from_ref(&barrier));
        self.lattice_upload_state = RtPageLatticeUploadState::Recording {
            frame_slot,
            frame_index,
        };
        Ok(range)
    }

    pub fn record_geometry_upload(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
        frame_index: u64,
        allocation: RtPageGeometryAllocation,
        geometry: &RtCompactPageGeometry,
    ) -> Result<RtPageGeometryUploadReceipt> {
        self.geometry_arena
            .validate_upload(allocation, geometry)
            .map_err(anyhow::Error::new)?;
        let index_bytes = bytemuck::cast_slice::<u16, u8>(&geometry.indices);
        let face_bytes = bytemuck::cast_slice::<RtCompactFaceRecord, u8>(&geometry.faces);
        let ranges = self
            .staging_tracker
            .reserve_ranges(
                frame_slot,
                frame_index,
                &[
                    (index_bytes.len() as u64, std::mem::align_of::<u16>() as u64),
                    (
                        face_bytes.len() as u64,
                        std::mem::align_of::<RtCompactFaceRecord>() as u64,
                    ),
                ],
            )
            .map_err(anyhow::Error::new)?;
        let staging_index_range = ranges[0];
        let staging_face_range = ranges[1];
        let staging = self.staging_buffer(frame_slot)?;
        write_staging_bytes(staging, staging_index_range, index_bytes)?;
        write_staging_bytes(staging, staging_face_range, face_bytes)?;
        record_buffer_copy(
            device,
            command_buffer,
            staging,
            &self.index_buffer,
            staging_index_range.offset,
            allocation.index_offset_bytes,
            staging_index_range.size,
        );
        let face_offset_bytes = u64::from(allocation.face_offset_records)
            .checked_mul(std::mem::size_of::<RtCompactFaceRecord>() as u64)
            .ok_or_else(|| anyhow!("RT page face upload offset overflow"))?;
        record_buffer_copy(
            device,
            command_buffer,
            staging,
            &self.face_buffer,
            staging_face_range.offset,
            face_offset_bytes,
            staging_face_range.size,
        );
        let barriers = rt_page_geometry_upload_barriers(allocation, 0)?;
        self.record_upload_barriers(device, command_buffer, &barriers[..2]);
        Ok(RtPageGeometryUploadReceipt {
            allocation_id: allocation.allocation_id,
            frame_slot,
            frame_index,
            staging_index_range,
            staging_face_range,
        })
    }

    pub fn record_page_record_upload(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
        frame_index: u64,
        page_slot: u32,
        record: GpuRtPageRecord,
    ) -> Result<RtPageRecordUploadReceipt> {
        if page_slot >= self.page_records.capacity() {
            return Err(anyhow::Error::new(RtPageRecordError::SlotOutOfRange {
                slot: page_slot,
                capacity: self.page_records.capacity(),
            }));
        }
        let bytes = bytemuck::bytes_of(&record);
        let staging_range = self
            .staging_tracker
            .reserve_ranges(
                frame_slot,
                frame_index,
                &[(
                    bytes.len() as u64,
                    std::mem::align_of::<GpuRtPageRecord>() as u64,
                )],
            )
            .map_err(anyhow::Error::new)?
            .into_iter()
            .next()
            .expect("one page-record staging range must be reserved");
        let staging = self.staging_buffer(frame_slot)?;
        write_staging_bytes(staging, staging_range, bytes)?;
        let barrier = rt_page_record_upload_barrier(page_slot)?;
        record_buffer_copy(
            device,
            command_buffer,
            staging,
            &self.page_record_buffer,
            staging_range.offset,
            barrier.offset,
            staging_range.size,
        );
        self.record_upload_barriers(device, command_buffer, std::slice::from_ref(&barrier));
        self.page_record_journal
            .stage(frame_slot, frame_index, page_slot, record)
            .map_err(anyhow::Error::new)?;
        Ok(RtPageRecordUploadReceipt {
            page_slot,
            frame_slot,
            frame_index,
            staging_range,
        })
    }

    pub fn page_record(&self, page_slot: u32) -> Option<GpuRtPageRecord> {
        self.page_records.get(page_slot)
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        for staging in self.staging_buffers {
            staging.destroy(device, allocator);
        }
        self.page_record_buffer.destroy(device, allocator);
        self.face_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
        self.lattice_buffer.destroy(device, allocator);
    }

    fn staging_buffer(&self, frame_slot: usize) -> Result<&GpuBuffer> {
        self.staging_buffers.get(frame_slot).ok_or_else(|| {
            anyhow!(
                "RT page staging frame slot is out of range: frame_slot={frame_slot} slot_count={}",
                self.staging_buffers.len()
            )
        })
    }

    fn record_upload_barriers(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        barriers: &[RtPageUploadBarrier],
    ) {
        if barriers.is_empty() {
            return;
        }
        let mut dst_stages = vk::PipelineStageFlags::empty();
        let vk_barriers = barriers
            .iter()
            .map(|barrier| {
                dst_stages |= barrier.dst_stage;
                let buffer = match barrier.target {
                    RtPageUploadTarget::Lattice => self.lattice_buffer.handle,
                    RtPageUploadTarget::Indices => self.index_buffer.handle,
                    RtPageUploadTarget::Faces => self.face_buffer.handle,
                    RtPageUploadTarget::PageRecords => self.page_record_buffer.handle,
                };
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(barrier.src_access)
                    .dst_access_mask(barrier.dst_access)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer)
                    .offset(barrier.offset)
                    .size(barrier.size)
            })
            .collect::<Vec<_>>();
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                dst_stages,
                vk::DependencyFlags::empty(),
                &[],
                &vk_barriers,
                &[],
            );
        }
    }
}

struct GpuBufferBuildGuard<'a> {
    device: &'a ash::Device,
    allocator: &'a GpuAllocator,
    buffers: Vec<GpuBuffer>,
}

impl<'a> GpuBufferBuildGuard<'a> {
    fn new(device: &'a ash::Device, allocator: &'a GpuAllocator) -> Self {
        Self {
            device,
            allocator,
            buffers: Vec::new(),
        }
    }

    fn create(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<()> {
        self.buffers.push(GpuBuffer::new(
            self.device,
            self.allocator,
            size,
            usage,
            location,
            name,
        )?);
        Ok(())
    }

    fn finish(mut self) -> Vec<GpuBuffer> {
        std::mem::take(&mut self.buffers)
    }
}

impl Drop for GpuBufferBuildGuard<'_> {
    fn drop(&mut self) {
        while let Some(buffer) = self.buffers.pop() {
            buffer.destroy(self.device, self.allocator);
        }
    }
}

fn write_staging_bytes(staging: &GpuBuffer, range: RtPageStagingRange, bytes: &[u8]) -> Result<()> {
    if range.size != bytes.len() as u64 {
        bail!(
            "RT page staging range size mismatch: range={} bytes={}",
            range.size,
            bytes.len()
        );
    }
    let end = range
        .offset
        .checked_add(range.size)
        .ok_or_else(|| anyhow!("RT page staging write range overflow"))?;
    if end > staging.size {
        bail!(
            "RT page staging write exceeds buffer: end={end} size={}",
            staging.size
        );
    }
    let mapped = staging
        .mapped_ptr()
        .ok_or_else(|| anyhow!("RT page staging buffer is not host visible"))?;
    let offset = usize::try_from(range.offset)
        .context("RT page staging offset does not fit host address space")?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.add(offset), bytes.len());
    }
    Ok(())
}

fn record_buffer_copy(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    source: &GpuBuffer,
    destination: &GpuBuffer,
    source_offset: u64,
    destination_offset: u64,
    size: u64,
) {
    let region = vk::BufferCopy::default()
        .src_offset(source_offset)
        .dst_offset(destination_offset)
        .size(size);
    unsafe {
        device.cmd_copy_buffer(
            command_buffer,
            source.handle,
            destination.handle,
            std::slice::from_ref(&region),
        );
    }
}

fn ensure_device_address_alignment(address: u64, alignment: u64, label: &str) -> Result<()> {
    if address == 0 {
        bail!("{label} device address is null");
    }
    if alignment == 0 || !address.is_multiple_of(alignment) {
        bail!("{label} device address is misaligned: address={address:#x} alignment={alignment}");
    }
    Ok(())
}

fn checked_align_up(value: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::rt_page_geometry::{RtCompactFaceRecord, RtCompactPageGeometry};
    use crate::render::rt_surface_mask::FaceDirection;
    use ash::vk;
    use bytemuck::Zeroable;

    #[test]
    fn gpu_rt_page_record_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuRtPageRecord>(), 48);
        assert_eq!(std::mem::align_of::<GpuRtPageRecord>(), 8);
        assert_eq!(std::mem::offset_of!(GpuRtPageRecord, brick_id), 0);
        assert_eq!(std::mem::offset_of!(GpuRtPageRecord, face_record_offset), 4);
        assert_eq!(std::mem::offset_of!(GpuRtPageRecord, face_count), 8);
        assert_eq!(std::mem::offset_of!(GpuRtPageRecord, representation), 12);
        assert_eq!(std::mem::offset_of!(GpuRtPageRecord, page_coord), 16);
        assert_eq!(std::mem::offset_of!(GpuRtPageRecord, topology_revision), 32);
        assert_eq!(std::mem::offset_of!(GpuRtPageRecord, resource_version), 40);
        assert_eq!(GpuRtPageRecord::zeroed(), GpuRtPageRecord::default());
    }

    #[test]
    fn shared_lattice_is_deterministic_and_covers_zero_through_eight() {
        let vertices = shared_rt_page_lattice_vertices();

        assert_eq!(vertices.len(), 729);
        assert_eq!(std::mem::size_of_val(&vertices[0]), 12);
        assert_eq!(vertices[0], [0.0, 0.0, 0.0]);
        assert_eq!(vertices[8], [8.0, 0.0, 0.0]);
        assert_eq!(vertices[80], [8.0, 8.0, 0.0]);
        assert_eq!(vertices[728], [8.0, 8.0, 8.0]);
    }

    #[test]
    fn gpu_page_buffer_usages_cover_as_build_and_shader_reads() {
        let lattice = RtPageGpuResources::lattice_buffer_usage();
        assert!(lattice.contains(vk::BufferUsageFlags::VERTEX_BUFFER));
        assert!(lattice.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS));
        assert!(
            lattice
                .contains(vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR)
        );
        assert!(lattice.contains(vk::BufferUsageFlags::TRANSFER_DST));

        let indices = RtPageGpuResources::index_buffer_usage();
        assert!(indices.contains(vk::BufferUsageFlags::INDEX_BUFFER));
        assert!(indices.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS));
        assert!(
            indices
                .contains(vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR)
        );
        assert!(indices.contains(vk::BufferUsageFlags::TRANSFER_DST));

        let records = RtPageGpuResources::record_buffer_usage();
        assert!(records.contains(vk::BufferUsageFlags::STORAGE_BUFFER));
        assert!(records.contains(vk::BufferUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn geometry_arena_honors_alignment_and_exact_fit() {
        let face_count = 7;
        let index_bytes = rt_page_index_bytes(face_count).unwrap();
        let mut arena = RtPageGeometryArena::new(index_bytes, face_count).unwrap();

        let allocation = arena.allocate(face_count).unwrap();

        assert_eq!(allocation.index_offset_bytes, 0);
        assert_eq!(
            allocation.index_offset_bytes % std::mem::size_of::<u16>() as u64,
            0
        );
        assert_eq!(allocation.index_size_bytes, index_bytes);
        assert_eq!(allocation.face_offset_records, 0);
        assert_eq!(
            u64::from(allocation.face_offset_records)
                * std::mem::size_of::<crate::render::rt_page_geometry::RtCompactFaceRecord>()
                    as u64
                % 4,
            0
        );
        assert_eq!(allocation.face_count, face_count);
        assert!(matches!(
            arena.allocate(1),
            Err(RtPageArenaError::IndexArenaExhausted { .. })
                | Err(RtPageArenaError::FaceArenaExhausted { .. })
        ));
    }

    #[test]
    fn geometry_arena_reuses_fragmentation_without_aliasing_live_allocations() {
        let per_page = rt_page_index_bytes(4).unwrap();
        let mut arena = RtPageGeometryArena::new(per_page * 3, 12).unwrap();
        let first = arena.allocate(4).unwrap();
        let middle = arena.allocate(4).unwrap();
        let last = arena.allocate(4).unwrap();

        arena.free(middle).unwrap();
        let reused = arena.allocate(4).unwrap();

        assert_eq!(reused.index_offset_bytes, middle.index_offset_bytes);
        assert_eq!(reused.face_offset_records, middle.face_offset_records);
        assert_ne!(reused.allocation_id, middle.allocation_id);
        assert!(arena.contains(first));
        assert!(arena.contains(last));
        assert!(arena.contains(reused));
        assert_eq!(arena.active_allocation_count(), 3);
    }

    #[test]
    fn geometry_arena_rejects_double_free() {
        let mut arena = RtPageGeometryArena::new(rt_page_index_bytes(2).unwrap(), 2).unwrap();
        let allocation = arena.allocate(2).unwrap();

        arena.free(allocation).unwrap();

        assert_eq!(
            arena.free(allocation),
            Err(RtPageArenaError::UnknownAllocation {
                allocation_id: allocation.allocation_id,
            })
        );
    }

    #[test]
    fn geometry_arena_rejects_stale_generation_after_range_reuse() {
        let mut arena = RtPageGeometryArena::new(rt_page_index_bytes(3).unwrap(), 3).unwrap();
        let old = arena.allocate(3).unwrap();
        arena.free(old).unwrap();
        let current = arena.allocate(3).unwrap();
        assert_eq!(old.index_offset_bytes, current.index_offset_bytes);
        assert_ne!(old.allocation_id, current.allocation_id);

        assert_eq!(
            arena.free(old),
            Err(RtPageArenaError::UnknownAllocation {
                allocation_id: old.allocation_id,
            })
        );
        assert!(arena.contains(current));
        arena.free(current).unwrap();
    }

    #[test]
    fn failed_face_allocation_rolls_back_index_range() {
        let four_faces = rt_page_index_bytes(4).unwrap();
        let mut arena = RtPageGeometryArena::new(four_faces * 2, 4).unwrap();

        assert!(matches!(
            arena.allocate(5),
            Err(RtPageArenaError::FaceArenaExhausted { .. })
        ));
        let exact = arena.allocate(4).unwrap();

        assert_eq!(exact.index_offset_bytes, 0);
        assert_eq!(exact.face_offset_records, 0);
    }

    #[test]
    fn replacement_geometry_stays_distinct_until_install() {
        let per_page = rt_page_index_bytes(6).unwrap();
        let mut arena = RtPageGeometryArena::new(per_page * 2, 12).unwrap();

        let resident = arena.allocate(6).unwrap();
        let replacement = arena.allocate(6).unwrap();

        assert_ne!(resident.allocation_id, replacement.allocation_id);
        assert_ne!(resident.index_offset_bytes, replacement.index_offset_bytes);
        assert_ne!(
            resident.face_offset_records,
            replacement.face_offset_records
        );
        assert!(arena.contains(resident));
        assert!(arena.contains(replacement));
    }

    #[test]
    fn page_record_arena_is_bounded_by_stable_slot() {
        let mut records = RtPageRecordArena::new(3).unwrap();
        let record = GpuRtPageRecord {
            brick_id: 9,
            face_record_offset: 12,
            face_count: 6,
            representation: 2,
            page_coord: [4, 5, 6, 0],
            topology_revision: 17,
            resource_version: 23,
        };

        records.write(2, record).unwrap();

        assert_eq!(records.get(2), Some(record));
        assert!(matches!(
            records.write(3, record),
            Err(RtPageRecordError::SlotOutOfRange { .. })
        ));
    }

    #[test]
    fn staging_slot_cannot_be_reused_before_submission_completes() {
        let mut staging = RtPageStagingTracker::new(2, 256).unwrap();
        staging.begin_frame_slot(0, 10).unwrap();
        let ranges = staging
            .reserve_ranges(0, 10, &[(12, 4), (6, 2), (48, 8)])
            .unwrap();
        assert_eq!(ranges[0].offset, 0);
        assert_eq!(ranges[1].offset, 12);
        assert_eq!(ranges[2].offset, 24);
        staging.finish_frame_slot(0, 10).unwrap();

        assert!(matches!(
            staging.begin_frame_slot(0, 11),
            Err(RtPageStagingError::FrameSlotInFlight { .. })
        ));
        staging.complete_frame_slot(0, 10).unwrap();
        staging.begin_frame_slot(0, 11).unwrap();
        let range = staging.reserve_ranges(0, 11, &[(16, 8)]).unwrap();
        assert_eq!(range[0].offset, 0);
    }

    #[test]
    fn upload_barrier_plan_separates_as_and_shader_consumers() {
        let allocation = RtPageGeometryAllocation {
            index_offset_bytes: 24,
            index_size_bytes: 72,
            face_offset_records: 11,
            face_count: 6,
            allocation_id: 5,
        };

        let barriers = rt_page_geometry_upload_barriers(allocation, 3).unwrap();

        assert_eq!(barriers.len(), 3);
        for barrier in &barriers {
            assert_eq!(barrier.src_stage, vk::PipelineStageFlags::TRANSFER);
            assert_eq!(barrier.src_access, vk::AccessFlags::TRANSFER_WRITE);
        }
        let indices = barriers
            .iter()
            .find(|barrier| barrier.target == RtPageUploadTarget::Indices)
            .unwrap();
        assert_eq!(
            indices.dst_stage,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
        );
        assert_eq!(
            indices.dst_access,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
        );

        for target in [RtPageUploadTarget::Faces, RtPageUploadTarget::PageRecords] {
            let barrier = barriers
                .iter()
                .find(|barrier| barrier.target == target)
                .unwrap();
            assert_eq!(
                barrier.dst_stage,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
            );
            assert_eq!(barrier.dst_access, vk::AccessFlags::SHADER_READ);
        }
    }

    #[test]
    fn gpu_config_computes_checked_buffer_sizes() {
        let config = RtPageGpuConfig {
            index_capacity_bytes: 96,
            face_capacity_records: 12,
            page_record_capacity: 5,
            staging_bytes_per_frame: 16 * 1024,
            frame_slot_count: 3,
        };

        let sizes = config.validate().unwrap();

        assert_eq!(sizes.lattice_bytes, 729 * 12);
        assert_eq!(sizes.index_bytes, 96);
        assert_eq!(sizes.face_bytes, 12 * 4);
        assert_eq!(sizes.page_record_bytes, 5 * 48);
        assert_eq!(sizes.staging_bytes_per_frame, 16 * 1024);
        assert_eq!(sizes.frame_slot_count, 3);
    }

    #[test]
    fn gpu_config_rejects_missing_slots_and_undersized_lattice_staging() {
        let valid = RtPageGpuConfig {
            index_capacity_bytes: 96,
            face_capacity_records: 12,
            page_record_capacity: 5,
            staging_bytes_per_frame: 16 * 1024,
            frame_slot_count: 3,
        };

        assert!(matches!(
            RtPageGpuConfig {
                frame_slot_count: 0,
                ..valid
            }
            .validate(),
            Err(RtPageGpuConfigError::NoFrameSlots)
        ));
        assert!(matches!(
            RtPageGpuConfig {
                staging_bytes_per_frame: 128,
                ..valid
            }
            .validate(),
            Err(RtPageGpuConfigError::StagingTooSmall { .. })
        ));
    }

    #[test]
    fn geometry_upload_validation_requires_the_active_exact_allocation() {
        let mut arena = RtPageGeometryArena::new(rt_page_index_bytes(2).unwrap() * 2, 4).unwrap();
        let allocation = arena.allocate(2).unwrap();
        let geometry = RtCompactPageGeometry {
            indices: vec![0; 12],
            faces: vec![
                RtCompactFaceRecord::new(glam::UVec3::ZERO, FaceDirection::NegativeX),
                RtCompactFaceRecord::new(glam::UVec3::ZERO, FaceDirection::PositiveX),
            ],
        };

        arena.validate_upload(allocation, &geometry).unwrap();
        arena.free(allocation).unwrap();

        assert!(matches!(
            arena.validate_upload(allocation, &geometry),
            Err(RtPageGeometryUploadError::InactiveAllocation { .. })
        ));
    }

    #[test]
    fn geometry_upload_validation_rejects_face_and_index_mismatch() {
        let mut arena = RtPageGeometryArena::new(rt_page_index_bytes(2).unwrap(), 2).unwrap();
        let allocation = arena.allocate(2).unwrap();
        let one_face = RtCompactPageGeometry {
            indices: vec![0; 6],
            faces: vec![RtCompactFaceRecord::new(
                glam::UVec3::ZERO,
                FaceDirection::NegativeX,
            )],
        };
        assert!(matches!(
            arena.validate_upload(allocation, &one_face),
            Err(RtPageGeometryUploadError::FaceCountMismatch { .. })
        ));

        let wrong_indices = RtCompactPageGeometry {
            indices: vec![0; 11],
            faces: vec![
                RtCompactFaceRecord::new(glam::UVec3::ZERO, FaceDirection::NegativeX),
                RtCompactFaceRecord::new(glam::UVec3::ZERO, FaceDirection::PositiveX),
            ],
        };
        assert!(matches!(
            arena.validate_upload(allocation, &wrong_indices),
            Err(RtPageGeometryUploadError::IndexCountMismatch { .. })
        ));
    }

    #[test]
    fn cancelled_recording_releases_staging_without_claiming_submission() {
        let mut staging = RtPageStagingTracker::new(1, 256).unwrap();
        staging.begin_frame_slot(0, 4).unwrap();
        staging.reserve_ranges(0, 4, &[(64, 8)]).unwrap();

        staging.cancel_frame_slot(0, 4).unwrap();

        staging.begin_frame_slot(0, 5).unwrap();
        assert_eq!(
            staging.reserve_ranges(0, 5, &[(8, 8)]).unwrap()[0].offset,
            0
        );
    }

    #[test]
    fn page_record_journal_commits_only_submitted_frames() {
        let mut records = RtPageRecordArena::new(4).unwrap();
        let mut journal = RtPageRecordJournal::new(2, records.capacity()).unwrap();
        let first = GpuRtPageRecord {
            brick_id: 1,
            resource_version: 10,
            ..GpuRtPageRecord::zeroed()
        };
        journal.stage(0, 7, 2, first).unwrap();
        assert_eq!(records.get(2), Some(GpuRtPageRecord::zeroed()));

        journal.cancel(0, 7).unwrap();
        assert_eq!(records.get(2), Some(GpuRtPageRecord::zeroed()));

        let submitted = GpuRtPageRecord {
            brick_id: 2,
            resource_version: 11,
            ..GpuRtPageRecord::zeroed()
        };
        journal.stage(0, 8, 2, submitted).unwrap();
        journal.commit(0, 8, &mut records).unwrap();
        assert_eq!(records.get(2), Some(submitted));
    }

    #[test]
    fn rt_page_gpu_resources_own_real_buffers_and_record_upload_commands() {
        let source = crate::render::source_checks::read_source("src/render/rt_page_gpu.rs");

        for token in [
            "lattice_buffer: GpuBuffer",
            "index_buffer: GpuBuffer",
            "face_buffer: GpuBuffer",
            "page_record_buffer: GpuBuffer",
            "staging_buffers: Vec<GpuBuffer>",
            "MemoryLocation::GpuOnly",
            "MemoryLocation::CpuToGpu",
            "cmd_copy_buffer",
            "cmd_pipeline_barrier",
            "device_address(",
            "record_lattice_upload",
            "record_geometry_upload",
            "record_page_record_upload",
        ] {
            assert!(
                source.contains(token),
                "RT page GPU resources must implement {token}"
            );
        }
    }
}
