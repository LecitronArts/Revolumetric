# VPT Motion Guide Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the CPU-side motion source and UCVH invalidation primitives that the VPT motion guide spec needs, without pretending rigid-object or semantic voxel motion is already wired into shaders.

**Architecture:** Keep the existing shader-side motion-vector contract unchanged for now. Add a small CPU motion source module with stable motion identity, previous-transform retention, and generation checks, then extend UCVH with explicit region invalidation and semantic move events so later render integration can consume real motion metadata instead of inferring it from dirty bricks.

**Tech Stack:** Rust 2024, `glam`, existing unit tests, existing voxel pool/UCVH code, existing render module structure.

---

### Task 1: Add Failing Tests For CPU Motion Source And UCVH Motion Metadata

**Files:**
- Create: `src/render/vpt_motion.rs`
- Modify: `src/render/mod.rs`
- Modify: `src/voxel/ucvh.rs`

- [ ] **Step 1: Write the failing MotionSource tests**

Add tests that define the expected CPU contract:

```rust
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

    assert!(source.previous_world_position_for_generation(4, glam::vec3(0.0, 0.0, 0.0)).is_none());
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
    assert!(second.previous_world_position_from_current_hit(glam::vec3(3.0, 0.0, 0.0)).is_none());
}
```

- [ ] **Step 2: Write the failing UCVH invalidation tests**

Add tests that define the new edit and semantic-motion contract:

```rust
#[test]
fn set_voxel_records_a_single_invalidated_brick_region() {
    let mut ucvh = test_ucvh();
    let cell = VoxelCell::new(1, 0, [0; 3]);

    assert!(ucvh.set_voxel(UVec3::new(2, 3, 4), cell));

    let invalidations = ucvh.take_invalidation_regions();
    assert_eq!(invalidations.len(), 1);
    assert_eq!(invalidations[0].brick_min, UVec3::ZERO);
    assert_eq!(invalidations[0].brick_max_exclusive, UVec3::new(1, 1, 1));
}

#[test]
fn write_brick_records_the_full_brick_region_and_generation() {
    let mut ucvh = test_ucvh();
    let data = BrickData::new();

    assert!(ucvh.write_brick(UVec3::new(1, 2, 3), &data));

    let invalidations = ucvh.take_invalidation_regions();
    assert_eq!(invalidations.len(), 1);
    assert_eq!(invalidations[0].brick_min, UVec3::new(1, 2, 3));
    assert_eq!(invalidations[0].generation, 0);
}

#[test]
fn semantic_move_rejects_overlapping_regions() {
    let mut ucvh = test_ucvh();
    let event = UcvhMotionEvent {
        region_min: UVec3::new(4, 4, 4),
        region_max_exclusive: UVec3::new(8, 8, 8),
        world_delta_current_from_previous: glam::IVec3::new(1, 0, 0),
        generation: 42,
    };

    assert!(ucvh.push_motion_event(event));
    assert!(!ucvh.push_motion_event(UcvhMotionEvent {
        region_min: UVec3::new(6, 6, 6),
        region_max_exclusive: UVec3::new(9, 9, 9),
        world_delta_current_from_previous: glam::IVec3::new(0, 1, 0),
        generation: 43,
    }));
}
```

- [ ] **Step 3: Verify red**

Run: `cargo test render::vpt_motion --lib && cargo test voxel::ucvh --lib`

Expected: FAIL because the module and new UCVH APIs do not exist yet.

### Task 2: Implement The CPU MotionSource Module

**Files:**
- Create: `src/render/vpt_motion.rs`
- Modify: `src/render/mod.rs`

- [ ] **Step 1: Add the minimal MotionSource types and helpers**

Implement the module with a stable identity contract:

```rust
use glam::{Mat4, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionSource {
    pub motion_id: u32,
    pub current_world_from_local: Mat4,
    pub previous_world_from_local: Mat4,
    pub generation: u32,
    pub flags: u32,
}

impl MotionSource {
    pub fn previous_world_position_from_current_hit(&self, current_world_position: Vec3) -> Option<Vec3> {
        self.previous_world_position_for_generation(self.generation, current_world_position)
    }

    pub fn previous_world_position_for_generation(
        &self,
        generation: u32,
        current_world_position: Vec3,
    ) -> Option<Vec3> {
        if generation != self.generation {
            return None;
        }
        let current_local = self.current_world_from_local.inverse().transform_point3(current_world_position);
        Some(self.previous_world_from_local.transform_point3(current_local))
    }
}
```

Add `MotionSourceHistory::record` so later render integration can keep previous transforms per `motion_id`. On matching `motion_id` and `generation`, it should copy the prior frame's current matrix into `previous_world_from_local` and set `VPT_MOTION_SOURCE_FLAG_HISTORY_VALID`; on generation mismatch, it should set `VPT_MOTION_SOURCE_FLAG_GENERATION_MISMATCH`, reset `previous_world_from_local` to the current matrix, and make reprojection helpers return `None`.

- [ ] **Step 2: Re-run the targeted MotionSource tests**

Run: `cargo test render::vpt_motion --lib`

Expected: PASS.

### Task 3: Extend UCVH With Explicit Invalidations And Semantic Move Events

**Files:**
- Modify: `src/voxel/ucvh.rs`

- [ ] **Step 1: Add the new region/event data types**

Implement:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UcvhInvalidationRegion {
    pub brick_min: UVec3,
    pub brick_max_exclusive: UVec3,
    pub generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UcvhMotionEvent {
    pub region_min: UVec3,
    pub region_max_exclusive: UVec3,
    pub world_delta_current_from_previous: glam::IVec3,
    pub generation: u32,
}
```

Track a dedicated invalidation list and a semantic move list on `Ucvh`. `set_voxel` and `write_brick` should record the affected brick region when the content actually changes, and `push_motion_event` should reject overlaps instead of inventing composition rules.

- [ ] **Step 2: Make UCVH edits record invalidation metadata**

Update `set_voxel` and `write_brick` so they append explicit invalidation regions before marking hierarchy dirty. Preserve the existing dirty-brick behavior.

- [ ] **Step 3: Re-run the targeted UCVH tests**

Run: `cargo test voxel::ucvh --lib`

Expected: PASS.

### Task 4: Full Verification And Clean Diff

**Files:**
- No intentional extra files.

- [ ] **Step 1: Run library tests**

Run: `$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE`

Expected: PASS.

- [ ] **Step 2: Check the diff**

Run: `git diff --check`

Expected: no whitespace or patch-format errors.

- [ ] **Step 3: Review touched files**

Run: `git status --short`

Expected: only the intended plan, motion-source, render mod, and UCVH files changed, plus any existing local IDE artifacts left untouched.
