use glam::{Mat4, Vec3};
use std::collections::HashMap;

pub const VPT_MOTION_SOURCE_FLAG_HISTORY_VALID: u32 = 1 << 0;
pub const VPT_MOTION_SOURCE_FLAG_GENERATION_MISMATCH: u32 = 1 << 1;

const VPT_MOTION_SOURCE_STATUS_MASK: u32 =
    VPT_MOTION_SOURCE_FLAG_HISTORY_VALID | VPT_MOTION_SOURCE_FLAG_GENERATION_MISMATCH;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionSource {
    pub motion_id: u32,
    pub current_world_from_local: Mat4,
    pub previous_world_from_local: Mat4,
    pub generation: u32,
    pub flags: u32,
}

impl MotionSource {
    pub fn previous_world_position_from_current_hit(
        &self,
        current_world_position: Vec3,
    ) -> Option<Vec3> {
        self.previous_world_position_for_generation(self.generation, current_world_position)
    }

    pub fn previous_world_position_for_generation(
        &self,
        generation: u32,
        current_world_position: Vec3,
    ) -> Option<Vec3> {
        if (self.flags & VPT_MOTION_SOURCE_FLAG_GENERATION_MISMATCH) != 0 {
            return None;
        }
        if generation != self.generation {
            return None;
        }

        let determinant = self.current_world_from_local.determinant();
        if !determinant.is_finite() || determinant.abs() <= 1.0e-8 {
            return None;
        }

        let current_local = self
            .current_world_from_local
            .inverse()
            .transform_point3(current_world_position);
        let previous_world = self
            .previous_world_from_local
            .transform_point3(current_local);
        previous_world.is_finite().then_some(previous_world)
    }
}

#[derive(Debug, Clone, Copy)]
struct MotionSourceSnapshot {
    current_world_from_local: Mat4,
    generation: u32,
}

#[derive(Debug, Default)]
pub struct MotionSourceHistory {
    snapshots: HashMap<u32, MotionSourceSnapshot>,
}

impl MotionSourceHistory {
    pub fn record(&mut self, source: MotionSource) -> MotionSource {
        let mut resolved = source;
        resolved.flags &= !VPT_MOTION_SOURCE_STATUS_MASK;

        match self.snapshots.get(&source.motion_id).copied() {
            Some(snapshot) if snapshot.generation == source.generation => {
                resolved.previous_world_from_local = snapshot.current_world_from_local;
                resolved.flags |= VPT_MOTION_SOURCE_FLAG_HISTORY_VALID;
            }
            Some(_) => {
                resolved.previous_world_from_local = source.current_world_from_local;
                resolved.flags |= VPT_MOTION_SOURCE_FLAG_GENERATION_MISMATCH;
            }
            None => {
                resolved.previous_world_from_local = source.current_world_from_local;
            }
        }

        self.snapshots.insert(
            source.motion_id,
            MotionSourceSnapshot {
                current_world_from_local: source.current_world_from_local,
                generation: source.generation,
            },
        );
        resolved
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_source_reconstructs_previous_world_position_from_current_local_hit() {
        let source = MotionSource {
            motion_id: 7,
            current_world_from_local: glam::Mat4::from_translation(glam::vec3(4.0, 0.0, 0.0)),
            previous_world_from_local: glam::Mat4::from_translation(glam::vec3(1.0, 0.0, 0.0)),
            generation: 12,
            flags: 0,
        };

        let previous = source
            .previous_world_position_from_current_hit(glam::vec3(6.0, 0.0, 0.0))
            .expect("history should be valid");

        assert_eq!(previous, glam::vec3(3.0, 0.0, 0.0));
    }

    #[test]
    fn motion_source_rejects_generation_mismatch() {
        let source = MotionSource {
            motion_id: 9,
            current_world_from_local: glam::Mat4::IDENTITY,
            previous_world_from_local: glam::Mat4::from_translation(glam::vec3(-2.0, 0.0, 0.0)),
            generation: 3,
            flags: 0,
        };

        assert!(
            source
                .previous_world_position_for_generation(4, glam::vec3(0.0, 0.0, 0.0))
                .is_none()
        );
    }

    #[test]
    fn motion_source_history_retains_previous_transform_for_matching_motion_id() {
        let mut history = MotionSourceHistory::default();

        let first = history.record(MotionSource {
            motion_id: 5,
            current_world_from_local: glam::Mat4::from_translation(glam::vec3(1.0, 0.0, 0.0)),
            previous_world_from_local: glam::Mat4::IDENTITY,
            generation: 1,
            flags: 0,
        });
        assert_eq!(
            first.previous_world_from_local,
            first.current_world_from_local
        );

        let second = history.record(MotionSource {
            motion_id: 5,
            current_world_from_local: glam::Mat4::from_translation(glam::vec3(3.0, 0.0, 0.0)),
            previous_world_from_local: glam::Mat4::IDENTITY,
            generation: 1,
            flags: 0,
        });

        assert_eq!(
            second.previous_world_from_local,
            glam::Mat4::from_translation(glam::vec3(1.0, 0.0, 0.0))
        );
    }

    #[test]
    fn motion_source_history_generation_mismatch_invalidates_reprojection() {
        let mut history = MotionSourceHistory::default();

        history.record(MotionSource {
            motion_id: 5,
            current_world_from_local: glam::Mat4::from_translation(glam::vec3(1.0, 0.0, 0.0)),
            previous_world_from_local: glam::Mat4::IDENTITY,
            generation: 1,
            flags: 0,
        });
        let second = history.record(MotionSource {
            motion_id: 5,
            current_world_from_local: glam::Mat4::from_translation(glam::vec3(3.0, 0.0, 0.0)),
            previous_world_from_local: glam::Mat4::IDENTITY,
            generation: 2,
            flags: 0,
        });

        assert_ne!(second.flags & VPT_MOTION_SOURCE_FLAG_GENERATION_MISMATCH, 0);
        assert!(
            second
                .previous_world_position_from_current_hit(glam::vec3(3.0, 0.0, 0.0))
                .is_none()
        );
    }
}
