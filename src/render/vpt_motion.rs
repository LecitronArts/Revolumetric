use glam::{Mat4, Vec3};
use std::collections::HashMap;

pub const VPT_MOTION_SOURCE_FLAG_HISTORY_VALID: u32 = 1 << 0;
pub const VPT_MOTION_SOURCE_FLAG_GENERATION_MISMATCH: u32 = 1 << 1;

pub const VPT_MOTION_FLAG_HISTORY_VALID: u32 = 1 << 0;
pub const VPT_MOTION_FLAG_CAMERA_STATIC: u32 = 1 << 1;
pub const VPT_MOTION_FLAG_UCVH_REGION_MOVE: u32 = 1 << 2;
pub const VPT_MOTION_FLAG_DISOCCLUDED: u32 = 1 << 4;
pub const VPT_MOTION_FLAG_HISTORY_RESET: u32 = 1 << 5;
pub const VPT_MOTION_FLAG_BEHIND_CAMERA: u32 = 1 << 6;
pub const VPT_NO_BRICK_GENERATION: u32 = u32::MAX;

const VPT_MOTION_SOURCE_STATUS_MASK: u32 =
    VPT_MOTION_SOURCE_FLAG_HISTORY_VALID | VPT_MOTION_SOURCE_FLAG_GENERATION_MISMATCH;
const VPT_MOTION_FLAG_KNOWN_MASK: u32 = VPT_MOTION_FLAG_HISTORY_VALID
    | VPT_MOTION_FLAG_CAMERA_STATIC
    | VPT_MOTION_FLAG_UCVH_REGION_MOVE
    | VPT_MOTION_FLAG_DISOCCLUDED
    | VPT_MOTION_FLAG_HISTORY_RESET
    | VPT_MOTION_FLAG_BEHIND_CAMERA;

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

pub fn vpt_motion_flag_is_history_valid(flags: u32) -> bool {
    (flags & VPT_MOTION_FLAG_HISTORY_VALID) != 0
        && (flags
            & (VPT_MOTION_FLAG_DISOCCLUDED
                | VPT_MOTION_FLAG_HISTORY_RESET
                | VPT_MOTION_FLAG_BEHIND_CAMERA))
            == 0
}

pub fn vpt_motion_flag_is_camera_static(flags: u32) -> bool {
    (flags & VPT_MOTION_FLAG_CAMERA_STATIC) != 0
        && vpt_motion_flag_is_history_valid(flags)
        && (flags & VPT_MOTION_FLAG_UCVH_REGION_MOVE) == 0
}

pub fn vpt_motion_flag_requires_disocclusion(flags: u32) -> bool {
    (flags & VPT_MOTION_FLAG_BEHIND_CAMERA) != 0
        || (flags & VPT_MOTION_FLAG_HISTORY_RESET) != 0
        || (flags & VPT_MOTION_FLAG_DISOCCLUDED) != 0
}

pub fn vpt_motion_flag_requires_history_reset(flags: u32) -> bool {
    (flags & VPT_MOTION_FLAG_HISTORY_RESET) != 0
        && (flags & VPT_MOTION_FLAG_DISOCCLUDED) != 0
        && (flags
            & (VPT_MOTION_FLAG_HISTORY_VALID
                | VPT_MOTION_FLAG_UCVH_REGION_MOVE
                | VPT_MOTION_FLAG_CAMERA_STATIC))
            == 0
}

pub fn vpt_motion_flags_are_legal(flags: u32) -> bool {
    if flags == 0 || (flags & !VPT_MOTION_FLAG_KNOWN_MASK) != 0 {
        return false;
    }
    matches!(
        flags,
        x if x == (VPT_MOTION_FLAG_HISTORY_VALID | VPT_MOTION_FLAG_CAMERA_STATIC)
            || x == (VPT_MOTION_FLAG_HISTORY_VALID | VPT_MOTION_FLAG_UCVH_REGION_MOVE)
            || x == VPT_MOTION_FLAG_DISOCCLUDED
            || x == (VPT_MOTION_FLAG_DISOCCLUDED | VPT_MOTION_FLAG_BEHIND_CAMERA)
            || x == (VPT_MOTION_FLAG_DISOCCLUDED | VPT_MOTION_FLAG_HISTORY_RESET)
    )
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

    #[test]
    fn motion_flag_bits_are_pinned() {
        assert_eq!(VPT_MOTION_FLAG_HISTORY_VALID, 1 << 0);
        assert_eq!(VPT_MOTION_FLAG_CAMERA_STATIC, 1 << 1);
        assert_eq!(VPT_MOTION_FLAG_UCVH_REGION_MOVE, 1 << 2);
        assert_eq!(VPT_MOTION_FLAG_DISOCCLUDED, 1 << 4);
        assert_eq!(VPT_MOTION_FLAG_HISTORY_RESET, 1 << 5);
        assert_eq!(VPT_MOTION_FLAG_BEHIND_CAMERA, 1 << 6);
        assert_eq!(
            VPT_MOTION_FLAG_HISTORY_RESET
                & (VPT_MOTION_FLAG_HISTORY_VALID
                    | VPT_MOTION_FLAG_CAMERA_STATIC
                    | VPT_MOTION_FLAG_UCVH_REGION_MOVE
                    | VPT_MOTION_FLAG_DISOCCLUDED
                    | VPT_MOTION_FLAG_BEHIND_CAMERA),
            0
        );
    }

    #[test]
    fn motion_flag_combinations_are_legal() {
        for flags in [
            VPT_MOTION_FLAG_HISTORY_VALID | VPT_MOTION_FLAG_CAMERA_STATIC,
            VPT_MOTION_FLAG_HISTORY_VALID | VPT_MOTION_FLAG_UCVH_REGION_MOVE,
            VPT_MOTION_FLAG_DISOCCLUDED,
            VPT_MOTION_FLAG_DISOCCLUDED | VPT_MOTION_FLAG_BEHIND_CAMERA,
            VPT_MOTION_FLAG_DISOCCLUDED | VPT_MOTION_FLAG_HISTORY_RESET,
        ] {
            assert!(
                vpt_motion_flags_are_legal(flags),
                "flags {flags:#010x} should be legal"
            );
        }

        for flags in [
            0,
            VPT_MOTION_FLAG_HISTORY_VALID,
            VPT_MOTION_FLAG_CAMERA_STATIC,
            VPT_MOTION_FLAG_UCVH_REGION_MOVE,
            VPT_MOTION_FLAG_HISTORY_VALID | VPT_MOTION_FLAG_BEHIND_CAMERA,
            VPT_MOTION_FLAG_HISTORY_VALID | VPT_MOTION_FLAG_HISTORY_RESET,
            VPT_MOTION_FLAG_CAMERA_STATIC | VPT_MOTION_FLAG_DISOCCLUDED,
        ] {
            assert!(
                !vpt_motion_flags_are_legal(flags),
                "flags {flags:#010x} should be rejected"
            );
        }
    }

    #[test]
    fn motion_flag_predicates_match_temporal_rejection_rules() {
        assert!(vpt_motion_flag_requires_disocclusion(
            VPT_MOTION_FLAG_BEHIND_CAMERA
        ));
        assert!(vpt_motion_flag_requires_disocclusion(
            VPT_MOTION_FLAG_DISOCCLUDED | VPT_MOTION_FLAG_HISTORY_RESET
        ));
        assert!(vpt_motion_flag_requires_history_reset(
            VPT_MOTION_FLAG_DISOCCLUDED | VPT_MOTION_FLAG_HISTORY_RESET
        ));
        assert!(vpt_motion_flag_is_camera_static(
            VPT_MOTION_FLAG_HISTORY_VALID | VPT_MOTION_FLAG_CAMERA_STATIC
        ));
        assert!(!vpt_motion_flag_is_camera_static(
            VPT_MOTION_FLAG_HISTORY_VALID | VPT_MOTION_FLAG_UCVH_REGION_MOVE
        ));
        assert!(vpt_motion_flag_is_history_valid(
            VPT_MOTION_FLAG_HISTORY_VALID | VPT_MOTION_FLAG_CAMERA_STATIC
        ));
        assert!(!vpt_motion_flag_is_history_valid(
            VPT_MOTION_FLAG_DISOCCLUDED
        ));
    }
}
