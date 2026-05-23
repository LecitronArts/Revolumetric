use crate::voxel::ucvh::{UCVH_NO_BRICK_GENERATION, Ucvh};
use glam::UVec3;

pub struct BrickGenerationAtlas {
    cpu_shadow: Vec<u32>,
    upload_staging: Vec<(u32, u32)>,
}

impl BrickGenerationAtlas {
    pub fn new(slot_count: u32) -> Self {
        Self {
            cpu_shadow: vec![UCVH_NO_BRICK_GENERATION; slot_count as usize],
            upload_staging: Vec::new(),
        }
    }

    pub fn initialize_from_ucvh(&mut self, ucvh: &Ucvh) {
        self.cpu_shadow.clear();
        self.cpu_shadow.extend_from_slice(ucvh.brick_generations());
        self.upload_staging.clear();
        self.upload_staging.extend(
            self.cpu_shadow
                .iter()
                .copied()
                .enumerate()
                .map(|(slot, generation)| (slot as u32, generation)),
        );
    }

    pub fn drain_ucvh_dirty_slots(&mut self, ucvh: &mut Ucvh) {
        for region in ucvh.take_invalidation_regions() {
            for brick_pos in brick_positions_in_region(region.brick_min, region.brick_max_exclusive)
            {
                let Some(brick_id) = ucvh.brick_id_at(brick_pos) else {
                    continue;
                };
                let generation = ucvh.brick_generation(brick_id).unwrap_or(region.generation);
                self.queue_upload(brick_id, generation);
            }
        }
    }

    pub fn cpu_shadow(&self) -> &[u32] {
        &self.cpu_shadow
    }

    pub fn pending_uploads(&self) -> &[(u32, u32)] {
        &self.upload_staging
    }

    pub fn clear_pending_uploads(&mut self) {
        self.upload_staging.clear();
    }

    fn queue_upload(&mut self, slot: u32, generation: u32) {
        let slot_index = slot as usize;
        if slot_index >= self.cpu_shadow.len() {
            return;
        }
        self.cpu_shadow[slot_index] = generation;
        if let Some(existing) = self
            .upload_staging
            .iter_mut()
            .find(|(existing_slot, _)| *existing_slot == slot)
        {
            existing.1 = generation;
        } else {
            self.upload_staging.push((slot, generation));
        }
    }
}

fn brick_positions_in_region(min: UVec3, max_exclusive: UVec3) -> impl Iterator<Item = UVec3> {
    (min.z..max_exclusive.z).flat_map(move |z| {
        (min.y..max_exclusive.y)
            .flat_map(move |y| (min.x..max_exclusive.x).map(move |x| UVec3::new(x, y, z)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::brick::{BrickData, VoxelCell};
    use crate::voxel::ucvh::{UCVH_NO_BRICK_GENERATION, Ucvh, UcvhConfig};
    use glam::UVec3;

    #[test]
    fn brick_generation_atlas_first_frame_initializes_all_slots() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));
        assert!(ucvh.set_voxel(UVec3::new(1, 2, 3), VoxelCell::new(4, 0, [0; 3])));
        let brick_id = ucvh
            .brick_id_at(UVec3::ZERO)
            .expect("first brick should be allocated");
        let generation = ucvh
            .brick_generation(brick_id)
            .expect("allocated brick generation should exist");

        let mut atlas = BrickGenerationAtlas::new(ucvh.pool.capacity());
        atlas.initialize_from_ucvh(&ucvh);

        assert_eq!(atlas.cpu_shadow()[brick_id as usize], generation);
        assert_eq!(atlas.cpu_shadow().len(), ucvh.pool.capacity() as usize);
        assert!(
            atlas
                .cpu_shadow()
                .iter()
                .enumerate()
                .filter(|(slot, _)| *slot != brick_id as usize)
                .all(|(_, generation)| *generation == UCVH_NO_BRICK_GENERATION)
        );
        assert_eq!(atlas.pending_uploads().len(), ucvh.pool.capacity() as usize);
    }

    #[test]
    fn brick_generation_atlas_sparse_update() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        assert!(ucvh.set_voxel(UVec3::new(1, 2, 3), VoxelCell::new(4, 0, [0; 3])));
        let _ = ucvh.take_invalidation_regions();

        let mut atlas = BrickGenerationAtlas::new(ucvh.pool.capacity());
        atlas.initialize_from_ucvh(&ucvh);
        atlas.clear_pending_uploads();

        let mut data = BrickData::new();
        data.set_voxel(0, 0, 0, VoxelCell::new(9, 0, [0; 3]));
        assert!(ucvh.write_brick(UVec3::new(1, 0, 0), &data));
        let brick_id = ucvh
            .brick_id_at(UVec3::new(1, 0, 0))
            .expect("edited brick should be allocated");
        let expected_generation = ucvh
            .invalidation_regions()
            .first()
            .expect("edit should record invalidation")
            .generation;

        atlas.drain_ucvh_dirty_slots(&mut ucvh);

        assert_eq!(atlas.pending_uploads(), &[(brick_id, expected_generation)]);
        assert_eq!(atlas.cpu_shadow()[brick_id as usize], expected_generation);
        assert!(ucvh.invalidation_regions().is_empty());
    }
}
