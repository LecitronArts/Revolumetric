# Camera Flythrough Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic camera flythrough test mode and a PowerShell script that launches it for RT/VPT observation or optional single/multi-frame capture.

**Architecture:** Add focused camera-path code next to existing camera logic, keep app orchestration limited to selecting manual vs automatic camera update, and expose the feature through environment variables used by a small script.

**Tech Stack:** Rust 2024, `glam`, existing `CameraRig`, Winit app loop, PowerShell tooling, Cargo unit tests.

---

### Task 1: Camera Path Model

**Files:**
- Modify: `src/scene/camera.rs`

- [ ] Add failing tests for parsing `REVOLUMETRIC_CAMERA_PATH=orbit`, invalid path fallback, and deterministic orbit application.
- [ ] Implement `CameraPathConfig`, `CameraPathKind`, `from_values`, `from_env`, and `apply_camera_path`.
- [ ] Run the targeted camera tests.

### Task 2: App Integration

**Files:**
- Modify: `src/app.rs`

- [ ] Add a failing test proving automatic camera path updates the rig and ignores manual movement.
- [ ] Store parsed camera path config on `RevolumetricApp`.
- [ ] Update `update_camera` to apply the path for `rendered_frames` when enabled and otherwise call `update_fly_camera`.
- [ ] Run the targeted app test.

### Task 3: Script Wrapper

**Files:**
- Create: `tools/rt_flythrough_capture.ps1`
- Modify: `src/render/source_checks.rs`

- [ ] Add a source test proving the script exists and sets the camera path, render mode, exit frame, and capture env vars.
- [ ] Implement the PowerShell script with safe env restoration.
- [ ] Run the targeted source check.

### Task 4: Final Verification

- [ ] Run `cargo fmt`.
- [ ] Run only the targeted Cargo tests for this feature.
- [ ] Inspect `git diff` to verify the change scope.
