# App Orchestration Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move pure per-frame orchestration logic out of `src/app.rs` while preserving runtime behavior.

**Architecture:** Extract logic that does not own Vulkan handles first: fly-camera input integration and scene uniform construction. Keep GPU resource creation, pass recording, and swapchain lifecycle in `app.rs` for now because those paths are tightly coupled to Vulkan object lifetimes.

**Tech Stack:** Rust 2024, Cargo tests, `glam`, existing `InputState`, `CameraRig`, `GpuSceneUniforms`, and `LightingSettings`.

---

## Chunk 1: Pure Runtime Helpers

### Task 1: Fly Camera Input Integration

**Files:**
- Modify: `src/scene/camera.rs`
- Modify: `src/app.rs`

- [x] **Step 1: Write failing tests**

Add tests in `src/scene/camera.rs` for:
- scroll input scales and clamps controller speed
- mouse delta updates yaw/pitch and recomputes forward
- movement axes move the camera by normalized velocity

- [x] **Step 2: Verify red**

Run: `cargo test scene::camera::tests::fly_camera`

Expected: tests fail because the helper does not exist yet.

- [x] **Step 3: Implement helper**

Add `pub fn update_fly_camera(rig: &mut CameraRig, input: InputState, dt: f32)` to `src/scene/camera.rs`.

- [x] **Step 4: Replace app inline logic**

Change `RevolumetricApp::update_camera()` to fetch `InputState` and `CameraRig`, then call `update_fly_camera()`.

- [x] **Step 5: Verify green**

Run: `cargo test scene::camera::tests::fly_camera`

Expected: tests pass.

### Task 2: Scene Uniform Construction

**Files:**
- Modify: `src/render/scene_ubo.rs`
- Modify: `src/app.rs`

- [x] **Step 1: Write failing tests**

Add tests in `src/render/scene_ubo.rs` for:
- settings are applied to flags and RC strategy fields
- resolution, sun fields, colors, time, and RC enable flag are copied into `GpuSceneUniforms`

- [x] **Step 2: Verify red**

Run: `cargo test render::scene_ubo::tests::build_scene_uniforms`

Expected: tests fail because the helper does not exist yet.

- [x] **Step 3: Implement helper**

Add a small `SceneUniformInputs` struct and `pub fn build_scene_uniforms(inputs: SceneUniformInputs) -> GpuSceneUniforms`.

- [x] **Step 4: Replace app inline construction**

Change `src/app.rs` to call `build_scene_uniforms()` instead of constructing `GpuSceneUniforms` inline.

- [x] **Step 5: Verify green**

Run: `cargo test render::scene_ubo::tests::build_scene_uniforms`

Expected: tests pass.

## Chunk 2: Verification

### Task 3: Final Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-04-26-app-orchestration-cleanup.md`

- [x] **Step 1: Format**

Run: `cargo fmt`

Expected: no errors.

- [x] **Step 2: Full tests**

Run: `cargo test`

Expected: all tests pass.

- [x] **Step 3: Lint**

Run: `cargo clippy --all-targets -- -D warnings`

Expected: no clippy errors.

- [x] **Step 4: Build**

Run: `cargo build`

Expected: build completes.
