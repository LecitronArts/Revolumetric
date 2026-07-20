use crate::render::rt_surface_mask::SurfaceMaskSourceStamp;
use crate::voxel::ucvh::UcvhRenderChangeBatch;
use glam::UVec3;
use std::collections::{BTreeSet, HashMap, VecDeque};
use thiserror::Error;

/// Maximum page slot value that fits in the TLAS 24-bit InstanceCustomIndex.
/// Matches `RT_PAGE_TLAS_MAX_INSTANCE_SLOTS` in `rt_page_tlas.rs`.
pub const RT_PAGE_MAX_SLOTS: RtPageSlot = 0x00FF_FFFE;

/// Slot space exhausted and no cold pages are available for eviction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("RT page slot capacity exhausted ({RT_PAGE_MAX_SLOTS} slots); evict cold pages first")]
pub struct RtPageSlotExhausted;

pub type RtPageSlot = u32;
pub type RtPageBuildGeneration = u64;

pub const RT_PAGE_DUMMY_SLOT: RtPageSlot = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPageRepresentation {
    Missing,
    Reference,
    CompactExact,
    HotOmm,
    HotInterface,
    CompactGreedy,
}

impl RtPageRepresentation {
    const fn is_build_target(self) -> bool {
        matches!(
            self,
            Self::CompactExact | Self::HotOmm | Self::HotInterface | Self::CompactGreedy
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPageState {
    Missing,
    Reference,
    Building {
        target: RtPageRepresentation,
        generation: RtPageBuildGeneration,
        source: SurfaceMaskSourceStamp,
    },
    Resident {
        representation: RtPageRepresentation,
        generation: RtPageBuildGeneration,
        source: SurfaceMaskSourceStamp,
        resource_version: u64,
    },
    Failed {
        generation: RtPageBuildGeneration,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageBuildTicket {
    pub slot: RtPageSlot,
    pub page: UVec3,
    pub target: RtPageRepresentation,
    pub generation: RtPageBuildGeneration,
    pub source: SurfaceMaskSourceStamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageResidentVersion {
    pub slot: RtPageSlot,
    pub page: UVec3,
    pub representation: RtPageRepresentation,
    pub generation: RtPageBuildGeneration,
    pub source: SurfaceMaskSourceStamp,
    pub resource_version: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageDirtyPage {
    pub slot: RtPageSlot,
    pub page: UVec3,
    pub enqueued_frame: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPageQueuePush {
    Enqueued,
    AlreadyPending,
    Overflow,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RtPageQueueReport {
    pub invalidated_pages: usize,
    pub enqueued_pages: usize,
    pub already_pending_pages: usize,
    pub overflowed_pages: usize,
    pub displaced_residents: Vec<RtPageResidentVersion>,
}

impl RtPageQueueReport {
    fn record(&mut self, invalidation: RtPageInvalidation) {
        self.invalidated_pages += 1;
        match invalidation.queue {
            RtPageQueuePush::Enqueued => self.enqueued_pages += 1,
            RtPageQueuePush::AlreadyPending => self.already_pending_pages += 1,
            RtPageQueuePush::Overflow => self.overflowed_pages += 1,
        }
        if let Some(resident) = invalidation.displaced_resident {
            self.displaced_residents.push(resident);
        }
    }

    pub fn overflowed(&self) -> bool {
        self.overflowed_pages != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtPageInvalidation {
    pub slot: RtPageSlot,
    pub queue: RtPageQueuePush,
    pub displaced_resident: Option<RtPageResidentVersion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPageBuildStartError {
    UnknownPage,
    InvalidTarget,
    SourcePageMismatch,
    NotPending,
    NotQueued,
    GenerationExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPageInstallError {
    UnknownPage,
    NotBuilding,
    StaleGeneration {
        expected: RtPageBuildGeneration,
        actual: RtPageBuildGeneration,
    },
    TicketMismatch,
    StaleSource,
}

#[derive(Debug, Clone)]
struct RtPageRecord {
    page: Option<UVec3>,
    state: RtPageState,
    dirty_since_frame: Option<u64>,
    queued: bool,
}

impl RtPageRecord {
    fn dummy() -> Self {
        Self {
            page: None,
            state: RtPageState::Missing,
            dirty_since_frame: None,
            queued: false,
        }
    }

    fn reference(page: UVec3) -> Self {
        Self {
            page: Some(page),
            state: RtPageState::Reference,
            dirty_since_frame: None,
            queued: false,
        }
    }
}

#[derive(Debug)]
pub struct RtPageRegistry {
    slots_by_page: HashMap<UVec3, RtPageSlot>,
    records: Vec<RtPageRecord>,
    /// Reusable slot indices returned by evict_cold_page.
    slot_free_list: Vec<RtPageSlot>,
    dirty_queue: VecDeque<RtPageSlot>,
    overflow_dirty: BTreeSet<(u64, RtPageSlot)>,
    dirty_queue_capacity: usize,
    next_build_generation: RtPageBuildGeneration,
}

impl RtPageRegistry {
    pub fn new(dirty_queue_capacity: usize) -> Self {
        Self {
            slots_by_page: HashMap::new(),
            records: vec![RtPageRecord::dummy()],
            slot_free_list: Vec::new(),
            dirty_queue: VecDeque::with_capacity(dirty_queue_capacity),
            overflow_dirty: BTreeSet::new(),
            dirty_queue_capacity,
            next_build_generation: 1,
        }
    }

    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    pub fn dirty_queue_capacity(&self) -> usize {
        self.dirty_queue_capacity
    }

    pub fn slot_for_page(&self, page: UVec3) -> Option<RtPageSlot> {
        self.slots_by_page.get(&page).copied()
    }

    pub fn page_for_slot(&self, slot: RtPageSlot) -> Option<UVec3> {
        self.records
            .get(slot as usize)
            .and_then(|record| record.page)
    }

    pub fn state_for_slot(&self, slot: RtPageSlot) -> Option<RtPageState> {
        self.records.get(slot as usize).map(|record| record.state)
    }

    pub fn state_for_page(&self, page: UVec3) -> Option<RtPageState> {
        self.slot_for_page(page)
            .and_then(|slot| self.state_for_slot(slot))
    }

    pub fn trace_representation(&self, page: UVec3) -> RtPageRepresentation {
        match self.state_for_page(page) {
            None | Some(RtPageState::Missing) => RtPageRepresentation::Missing,
            Some(RtPageState::Resident { representation, .. }) => representation,
            Some(
                RtPageState::Reference | RtPageState::Building { .. } | RtPageState::Failed { .. },
            ) => RtPageRepresentation::Reference,
        }
    }

    pub fn ensure_reference_page(
        &mut self,
        page: UVec3,
    ) -> Result<RtPageSlot, RtPageSlotExhausted> {
        if let Some(slot) = self.slot_for_page(page) {
            return Ok(slot);
        }

        // Reuse a freed slot if available.
        if let Some(slot) = self.slot_free_list.pop() {
            self.records[slot as usize] = RtPageRecord::reference(page);
            self.slots_by_page.insert(page, slot);
            return Ok(slot);
        }

        // Allocate a new slot, enforcing the 24-bit TLAS InstanceCustomIndex limit.
        let next = self.records.len() as u64;
        if next > u64::from(RT_PAGE_MAX_SLOTS) {
            return Err(RtPageSlotExhausted);
        }
        let slot = next as RtPageSlot;
        self.records.push(RtPageRecord::reference(page));
        self.slots_by_page.insert(page, slot);
        Ok(slot)
    }

    /// Evict a cold page (Missing or Reference state, not dirty, not building) and return
    /// its slot to the free-list so it can be reused for a new page.  Returns `true` on
    /// success, `false` if the page cannot be evicted (e.g. it is building or resident).
    pub fn evict_cold_page(&mut self, page: UVec3) -> bool {
        let Some(slot) = self.slots_by_page.get(&page).copied() else {
            return false;
        };
        let record = &self.records[slot as usize];
        // Only evict pages that are not actively in use.
        let evictable = matches!(
            record.state,
            RtPageState::Missing | RtPageState::Reference
        ) && record.dirty_since_frame.is_none()
            && !record.queued;
        if !evictable {
            return false;
        }
        self.slots_by_page.remove(&page);
        self.records[slot as usize] = RtPageRecord::dummy();
        self.slot_free_list.push(slot);
        true
    }

    pub fn bootstrap_pages(
        &mut self,
        pages: impl IntoIterator<Item = UVec3>,
        frame_index: u64,
    ) -> RtPageQueueReport {
        let mut report = RtPageQueueReport::default();
        for page in pages {
            report.record(self.invalidate_topology(page, frame_index));
        }
        report
    }

    pub fn ingest_render_change_batch(
        &mut self,
        batch: &UcvhRenderChangeBatch,
        frame_index: u64,
    ) -> RtPageQueueReport {
        let mut report = RtPageQueueReport::default();
        for &page in &batch.invalidated_pages {
            report.record(self.invalidate_topology(page, frame_index));
        }
        report
    }

    pub fn invalidate_topology(&mut self, page: UVec3, frame_index: u64) -> RtPageInvalidation {
        // If slot allocation fails (capacity exhausted), route to overflow with no slot.
        // The page will not be tracked until a slot becomes available via evict_cold_page.
        let slot = match self.ensure_reference_page(page) {
            Ok(s) => s,
            Err(RtPageSlotExhausted) => {
                return RtPageInvalidation {
                    slot: RT_PAGE_DUMMY_SLOT,
                    queue: RtPageQueuePush::Overflow,
                    displaced_resident: None,
                };
            }
        };
        let displaced_resident = match self.records[slot as usize].state {
            RtPageState::Resident {
                representation,
                generation,
                source,
                resource_version,
            } => Some(RtPageResidentVersion {
                slot,
                page,
                representation,
                generation,
                source,
                resource_version,
            }),
            _ => None,
        };
        self.records[slot as usize].state = RtPageState::Reference;
        let queue = self.mark_dirty(slot, frame_index);
        RtPageInvalidation {
            slot,
            queue,
            displaced_resident,
        }
    }

    pub fn pending_dirty_count(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.dirty_since_frame.is_some())
            .count()
    }

    pub fn queued_dirty_count(&self) -> usize {
        self.dirty_queue.len()
    }

    pub fn dirty_since_frame(&self, page: UVec3) -> Option<u64> {
        let slot = self.slot_for_page(page)?;
        self.records
            .get(slot as usize)
            .and_then(|record| record.dirty_since_frame)
    }

    pub fn peek_dirty_page(&self) -> Option<RtPageDirtyPage> {
        let &slot = self.dirty_queue.front()?;
        let record = self.records.get(slot as usize)?;
        Some(RtPageDirtyPage {
            slot,
            page: record.page?,
            enqueued_frame: record.dirty_since_frame?,
        })
    }

    pub fn begin_build(
        &mut self,
        page: UVec3,
        target: RtPageRepresentation,
        source: SurfaceMaskSourceStamp,
    ) -> Result<RtPageBuildTicket, RtPageBuildStartError> {
        if !target.is_build_target() {
            return Err(RtPageBuildStartError::InvalidTarget);
        }
        if source.page != page {
            return Err(RtPageBuildStartError::SourcePageMismatch);
        }
        let slot = self
            .slot_for_page(page)
            .ok_or(RtPageBuildStartError::UnknownPage)?;
        self.refill_dirty_queue();
        let record = &self.records[slot as usize];
        if record.dirty_since_frame.is_none() {
            return Err(RtPageBuildStartError::NotPending);
        }
        if !record.queued {
            return Err(RtPageBuildStartError::NotQueued);
        }

        let generation = self.next_build_generation;
        self.next_build_generation = generation
            .checked_add(1)
            .ok_or(RtPageBuildStartError::GenerationExhausted)?;
        let queue_index = self
            .dirty_queue
            .iter()
            .position(|&queued_slot| queued_slot == slot)
            .expect("queued RT page record must have a dirty queue entry");
        self.dirty_queue.remove(queue_index);

        let record = &mut self.records[slot as usize];
        record.dirty_since_frame = None;
        record.queued = false;
        record.state = RtPageState::Building {
            target,
            generation,
            source,
        };
        self.refill_dirty_queue();

        Ok(RtPageBuildTicket {
            slot,
            page,
            target,
            generation,
            source,
        })
    }

    pub fn install_build(
        &mut self,
        ticket: RtPageBuildTicket,
        current_source: SurfaceMaskSourceStamp,
        resource_version: u64,
        frame_index: u64,
    ) -> Result<RtPageResidentVersion, RtPageInstallError> {
        let Some(record) = self.records.get(ticket.slot as usize) else {
            return Err(RtPageInstallError::UnknownPage);
        };
        if record.page != Some(ticket.page) || self.slot_for_page(ticket.page) != Some(ticket.slot)
        {
            return Err(RtPageInstallError::UnknownPage);
        }
        let (target, generation, expected_source) = match record.state {
            RtPageState::Building {
                target,
                generation,
                source,
            } => (target, generation, source),
            _ => return Err(RtPageInstallError::NotBuilding),
        };
        if generation != ticket.generation {
            return Err(RtPageInstallError::StaleGeneration {
                expected: generation,
                actual: ticket.generation,
            });
        }
        if target != ticket.target || expected_source != ticket.source {
            return Err(RtPageInstallError::TicketMismatch);
        }
        if current_source != expected_source {
            self.records[ticket.slot as usize].state = RtPageState::Reference;
            self.mark_dirty(ticket.slot, frame_index);
            return Err(RtPageInstallError::StaleSource);
        }

        let resident = RtPageResidentVersion {
            slot: ticket.slot,
            page: ticket.page,
            representation: target,
            generation,
            source: expected_source,
            resource_version,
        };
        self.records[ticket.slot as usize].state = RtPageState::Resident {
            representation: target,
            generation,
            source: expected_source,
            resource_version,
        };
        Ok(resident)
    }

    pub fn fail_build(&mut self, ticket: RtPageBuildTicket, frame_index: u64) -> bool {
        let Some(record) = self.records.get(ticket.slot as usize) else {
            return false;
        };
        if record.page != Some(ticket.page)
            || record.state
                != (RtPageState::Building {
                    target: ticket.target,
                    generation: ticket.generation,
                    source: ticket.source,
                })
        {
            return false;
        }

        self.records[ticket.slot as usize].state = RtPageState::Failed {
            generation: ticket.generation,
        };
        self.mark_dirty(ticket.slot, frame_index);
        true
    }

    fn mark_dirty(&mut self, slot: RtPageSlot, frame_index: u64) -> RtPageQueuePush {
        let record = &mut self.records[slot as usize];
        if let Some(oldest_frame) = record.dirty_since_frame.as_mut() {
            let previous_frame = *oldest_frame;
            *oldest_frame = previous_frame.min(frame_index);
            if !record.queued && *oldest_frame != previous_frame {
                let removed = self.overflow_dirty.remove(&(previous_frame, slot));
                debug_assert!(removed, "overflow RT page must retain its age index");
                self.overflow_dirty.insert((*oldest_frame, slot));
            }
            return RtPageQueuePush::AlreadyPending;
        }

        record.dirty_since_frame = Some(frame_index);
        if self.dirty_queue.len() >= self.dirty_queue_capacity {
            self.overflow_dirty.insert((frame_index, slot));
            return RtPageQueuePush::Overflow;
        }
        record.queued = true;
        self.dirty_queue.push_back(slot);
        RtPageQueuePush::Enqueued
    }

    fn refill_dirty_queue(&mut self) {
        while self.dirty_queue.len() < self.dirty_queue_capacity {
            let Some((frame, slot)) = self.overflow_dirty.pop_first() else {
                break;
            };
            let record = &mut self.records[slot as usize];
            debug_assert_eq!(record.dirty_since_frame, Some(frame));
            debug_assert!(!record.queued);
            record.queued = true;
            self.dirty_queue.push_back(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::rt_surface_mask::{
        SURFACE_MASK_DIRECTION_COUNT, SurfaceMaskDependencyStamp, SurfaceMaskSourceStamp,
    };
    use crate::voxel::ucvh::{UcvhChangedBrick, UcvhPageBoundaryMask, UcvhRenderChangeBatch};
    use glam::UVec3;

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

    fn material_only_batch(page: UVec3) -> UcvhRenderChangeBatch {
        UcvhRenderChangeBatch {
            id: 11,
            bricks: vec![UcvhChangedBrick {
                brick_id: 7,
                brick_coord: page,
                generation: 3,
                revision: 9,
                occupancy_changed: false,
                material_changed: true,
                touched_boundaries: UcvhPageBoundaryMask::NONE,
            }],
            invalidated_pages: Vec::new(),
            invalidated_render_cells: vec![page],
        }
    }

    fn install_compact_exact(
        registry: &mut RtPageRegistry,
        page: UVec3,
        source: SurfaceMaskSourceStamp,
        resource_version: u64,
    ) -> RtPageBuildTicket {
        registry.invalidate_topology(page, 1);
        let ticket = registry
            .begin_build(page, RtPageRepresentation::CompactExact, source)
            .expect("dirty page should start a CompactExact build");
        registry
            .install_build(ticket, source, resource_version, 2)
            .expect("current build should install");
        ticket
    }

    #[test]
    fn dummy_slot_is_reserved_and_page_slot_stays_stable() {
        let mut registry = RtPageRegistry::new(8);
        let page = UVec3::new(17, 3, 91);

        assert_eq!(registry.record_count(), 1);
        assert_eq!(registry.page_for_slot(RT_PAGE_DUMMY_SLOT), None);
        assert_eq!(
            registry.state_for_slot(RT_PAGE_DUMMY_SLOT),
            Some(RtPageState::Missing)
        );

        let first = registry.ensure_reference_page(page).expect("first slot allocation");
        let second = registry.ensure_reference_page(page).expect("idempotent allocation");

        assert_eq!(first, 1);
        assert_eq!(second, first);
        assert_eq!(registry.slot_for_page(page), Some(first));
        assert_eq!(registry.page_for_slot(first), Some(page));
        assert_eq!(registry.record_count(), 2);
        assert_eq!(registry.state_for_page(page), Some(RtPageState::Reference));
    }

    #[test]
    fn occupancy_invalidation_selects_reference_before_rescheduling() {
        let mut registry = RtPageRegistry::new(8);
        let page = UVec3::new(4, 5, 6);
        let original = source(page, 12);
        install_compact_exact(&mut registry, page, original, 44);

        let invalidation = registry.invalidate_topology(page, 30);

        assert_eq!(registry.state_for_page(page), Some(RtPageState::Reference));
        assert_eq!(
            registry.trace_representation(page),
            RtPageRepresentation::Reference
        );
        assert_eq!(
            invalidation.displaced_resident.unwrap().resource_version,
            44
        );
        assert_eq!(registry.dirty_since_frame(page), Some(30));
        assert_eq!(registry.peek_dirty_page().unwrap().page, page);
    }

    #[test]
    fn material_only_batch_does_not_schedule_topology_work() {
        let mut registry = RtPageRegistry::new(8);
        let page = UVec3::new(2, 1, 9);
        let current = source(page, 15);
        install_compact_exact(&mut registry, page, current, 50);

        let report = registry.ingest_render_change_batch(&material_only_batch(page), 40);

        assert_eq!(report, RtPageQueueReport::default());
        assert_eq!(registry.pending_dirty_count(), 0);
        assert_eq!(
            registry.state_for_page(page),
            Some(RtPageState::Resident {
                representation: RtPageRepresentation::CompactExact,
                generation: 1,
                source: current,
                resource_version: 50,
            })
        );
    }

    #[test]
    fn old_generation_result_cannot_replace_a_newer_build() {
        let mut registry = RtPageRegistry::new(8);
        let page = UVec3::new(1, 1, 1);
        let old_source = source(page, 20);
        registry.invalidate_topology(page, 1);
        let old = registry
            .begin_build(page, RtPageRepresentation::CompactExact, old_source)
            .unwrap();

        registry.invalidate_topology(page, 2);
        let current_source = source(page, 21);
        let current = registry
            .begin_build(page, RtPageRepresentation::CompactExact, current_source)
            .unwrap();

        assert_eq!(
            registry.install_build(old, current_source, 70, 3),
            Err(RtPageInstallError::StaleGeneration {
                expected: current.generation,
                actual: old.generation,
            })
        );
        assert_eq!(
            registry.state_for_page(page),
            Some(RtPageState::Building {
                target: RtPageRepresentation::CompactExact,
                generation: current.generation,
                source: current_source,
            })
        );
    }

    #[test]
    fn stale_source_result_is_rejected_and_page_is_requeued() {
        let mut registry = RtPageRegistry::new(8);
        let page = UVec3::new(8, 2, 3);
        let built_source = source(page, 30);
        registry.invalidate_topology(page, 1);
        let ticket = registry
            .begin_build(page, RtPageRepresentation::CompactExact, built_source)
            .unwrap();
        let current_source = source(page, 31);

        assert_eq!(
            registry.install_build(ticket, current_source, 80, 9),
            Err(RtPageInstallError::StaleSource)
        );
        assert_eq!(registry.state_for_page(page), Some(RtPageState::Reference));
        assert_eq!(registry.dirty_since_frame(page), Some(9));
        assert_eq!(registry.peek_dirty_page().unwrap().page, page);
    }

    #[test]
    fn successful_install_selects_exactly_the_requested_representation() {
        let mut registry = RtPageRegistry::new(8);
        let page = UVec3::new(3, 4, 5);
        let build_source = source(page, 41);
        registry.invalidate_topology(page, 1);
        let ticket = registry
            .begin_build(page, RtPageRepresentation::HotInterface, build_source)
            .unwrap();

        let installed = registry.install_build(ticket, build_source, 91, 2).unwrap();

        assert_eq!(installed.representation, RtPageRepresentation::HotInterface);
        assert_eq!(installed.resource_version, 91);
        assert_eq!(
            registry.trace_representation(page),
            RtPageRepresentation::HotInterface
        );
        assert_eq!(
            registry.state_for_page(page),
            Some(RtPageState::Resident {
                representation: RtPageRepresentation::HotInterface,
                generation: ticket.generation,
                source: build_source,
                resource_version: 91,
            })
        );
    }

    #[test]
    fn failed_build_remains_reference_traceable_and_retryable() {
        let mut registry = RtPageRegistry::new(8);
        let page = UVec3::new(9, 9, 9);
        let build_source = source(page, 51);
        registry.invalidate_topology(page, 4);
        let ticket = registry
            .begin_build(page, RtPageRepresentation::CompactExact, build_source)
            .unwrap();

        assert!(registry.fail_build(ticket, 12));

        assert_eq!(
            registry.state_for_page(page),
            Some(RtPageState::Failed {
                generation: ticket.generation,
            })
        );
        assert_eq!(
            registry.trace_representation(page),
            RtPageRepresentation::Reference
        );
        assert_eq!(registry.dirty_since_frame(page), Some(12));
        assert_eq!(registry.peek_dirty_page().unwrap().page, page);
    }

    #[test]
    fn queue_overflow_preserves_page_identity_and_oldest_frame() {
        let mut registry = RtPageRegistry::new(1);
        let first = UVec3::new(1, 0, 0);
        let overflowed = UVec3::new(2, 0, 0);

        assert_eq!(
            registry.invalidate_topology(first, 7).queue,
            RtPageQueuePush::Enqueued
        );
        assert_eq!(
            registry.invalidate_topology(overflowed, 8).queue,
            RtPageQueuePush::Overflow
        );
        assert!(registry.slot_for_page(overflowed).is_some());
        assert_eq!(registry.dirty_since_frame(overflowed), Some(8));
        assert_eq!(registry.pending_dirty_count(), 2);
        assert_eq!(registry.queued_dirty_count(), 1);

        assert_eq!(
            registry.invalidate_topology(overflowed, 20).queue,
            RtPageQueuePush::AlreadyPending
        );
        assert_eq!(registry.dirty_since_frame(overflowed), Some(8));

        let first_work = registry.peek_dirty_page().unwrap();
        assert_eq!(first_work.page, first);
        registry
            .begin_build(first, RtPageRepresentation::CompactExact, source(first, 1))
            .unwrap();

        assert_eq!(registry.pending_dirty_count(), 1);
        assert_eq!(registry.queued_dirty_count(), 1);
        assert_eq!(registry.peek_dirty_page().unwrap().page, overflowed);
    }
}
