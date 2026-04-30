# Phase 9: Camera & Input — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded camera with an interactive FPS camera controller (WASD + right-click mouse look + scroll-wheel speed).

**Architecture:** Expand existing stubs (`InputState`, `FlyCameraController`, `CameraRig`) with full fields. Handle winit events in `app.rs::window_event` to feed `InputState`. Apply input to camera in `tick_frame` before rendering. No new files or dependencies.

**Tech Stack:** Rust, winit 0.30 (events), glam 0.30 (math), custom ECS (type-erased resources)

**Spec:** `docs/superpowers/specs/2026-04-05-camera-input-design.md`

---

## File Map

| File | Role | Change Type |
|---|---|---|
| `src/platform/input.rs` | Input state accumulator | Modify |
| `src/scene/camera.rs` | Camera struct + defaults | Modify |
| `src/scene/components.rs` | FlyCameraController + CameraRig | Modify |
| `src/scene/systems.rs` | Bootstrap scene + tick_time | Modify |
| `src/app.rs` | Event handling + camera update + render integration | Modify |

---

### Task 1: Expand InputState

**Files:**
- Modify: `src/platform/input.rs`

- [ ] **Step 1: Write test for InputState default**

Add to the bottom of `src/platform/input.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_state_default_is_zeroed() {
        let state = InputState::default();
        assert_eq!(state.move_forward, 0.0);
        assert_eq!(state.move_right, 0.0);
        assert_eq!(state.move_up, 0.0);
        assert_eq!(state.mouse_dx, 0.0);
        assert_eq!(state.mouse_dy, 0.0);
        assert!(!state.right_mouse_held);
        assert_eq!(state.scroll_delta, 0.0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib platform::input::tests::input_state_default_is_zeroed`

Expected: FAIL — fields `mouse_dx`, `mouse_dy`, `right_mouse_held`, `scroll_delta` don't exist yet.

- [ ] **Step 3: Expand InputState struct**

Replace the entire `InputState` in `src/platform/input.rs` with:

```rust
#[derive(Debug, Default, Clone, Copy)]
pub struct InputState {
    /// Movement axes (-1.0 to 1.0, accumulated from key press/release)
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    /// Mouse delta in pixels this frame (accumulated, cleared each frame)
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    /// Right mouse button currently held
    pub right_mouse_held: bool,
    /// Scroll wheel delta this frame (accumulated, cleared each frame)
    pub scroll_delta: f32,
}

impl InputState {
    /// Clear per-frame accumulators. Called at the end of each frame.
    pub fn clear_per_frame(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.scroll_delta = 0.0;
    }

    /// Reset all axes to zero. Called on focus loss.
    pub fn reset_axes(&mut self) {
        self.move_forward = 0.0;
        self.move_right = 0.0;
        self.move_up = 0.0;
    }
}
```

- [ ] **Step 4: Add tests for helper methods**

Append to the `tests` module:

```rust
#[test]
fn clear_per_frame_preserves_axes() {
    let mut state = InputState {
        move_forward: 1.0,
        move_right: -1.0,
        move_up: 1.0,
        mouse_dx: 10.0,
        mouse_dy: -5.0,
        right_mouse_held: true,
        scroll_delta: 2.0,
    };
    state.clear_per_frame();
    assert_eq!(state.move_forward, 1.0);
    assert_eq!(state.move_right, -1.0);
    assert_eq!(state.mouse_dx, 0.0);
    assert_eq!(state.mouse_dy, 0.0);
    assert_eq!(state.scroll_delta, 0.0);
    assert!(state.right_mouse_held); // not cleared
}

#[test]
fn reset_axes_clears_movement() {
    let mut state = InputState {
        move_forward: 1.0,
        move_right: -1.0,
        move_up: 1.0,
        mouse_dx: 10.0,
        ..InputState::default()
    };
    state.reset_axes();
    assert_eq!(state.move_forward, 0.0);
    assert_eq!(state.move_right, 0.0);
    assert_eq!(state.move_up, 0.0);
    assert_eq!(state.mouse_dx, 10.0); // not cleared
}
```

- [ ] **Step 5: Run all tests**

Run: `cargo test --lib platform::input`

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/platform/input.rs
git commit -m "feat(input): expand InputState with mouse, scroll, and helper methods"
```

---

### Task 2: Expand FlyCameraController and Update Camera Defaults

**Files:**
- Modify: `src/scene/camera.rs`
- Modify: `src/scene/components.rs`

- [ ] **Step 1: Write test for FlyCameraController defaults**

Add to the bottom of `src/scene/components.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_camera_controller_defaults() {
        let ctrl = FlyCameraController::default();
        assert_eq!(ctrl.move_speed, 50.0);
        assert_eq!(ctrl.min_speed, 5.0);
        assert_eq!(ctrl.max_speed, 500.0);
        assert!((ctrl.scroll_multiplier - 1.2).abs() < 1e-5);
        assert!((ctrl.mouse_sensitivity - 0.3).abs() < 1e-5);
        assert!((ctrl.pitch - (-0.153)).abs() < 0.01);
        assert_eq!(ctrl.yaw, 0.0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib scene::components::tests::fly_camera_controller_defaults`

Expected: FAIL — new fields don't exist.

- [ ] **Step 3: Expand FlyCameraController**

Replace `FlyCameraController` and its `Default` impl in `src/scene/components.rs`:

```rust
#[derive(Debug, Clone)]
pub struct FlyCameraController {
    pub move_speed: f32,
    pub min_speed: f32,
    pub max_speed: f32,
    pub scroll_multiplier: f32,
    pub mouse_sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl Default for FlyCameraController {
    fn default() -> Self {
        Self {
            move_speed: 50.0,
            min_speed: 5.0,
            max_speed: 500.0,
            scroll_multiplier: 1.2,
            mouse_sensitivity: 0.3,
            pitch: -0.153, // ≈ -8.7°, derived from hardcoded camera looking at sphere
            yaw: 0.0,
        }
    }
}
```

- [ ] **Step 4: Update Camera defaults in `src/scene/camera.rs`**

Change `Camera::default()` to match the spec initial values:

```rust
impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(64.0, 80.0, -40.0),
            forward: Vec3::new(0.0, -0.152, 0.988).normalize(),
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_4, // 45°
        }
    }
}
```

- [ ] **Step 5: Add Camera default test**

Add to `src/scene/camera.rs` (or append to existing tests if any):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_default_matches_spec() {
        let cam = Camera::default();
        assert!((cam.position - Vec3::new(64.0, 80.0, -40.0)).length() < 1e-3);
        assert!((cam.fov_y_radians - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
        assert!(cam.forward.z > 0.9, "should look along +Z");
        assert!(cam.forward.y < 0.0, "should look slightly down");
    }
}
```

- [ ] **Step 6: Run all tests**

Run: `cargo test --lib scene::components::tests && cargo test --lib scene::camera::tests`

Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/scene/components.rs src/scene/camera.rs
git commit -m "feat(camera): expand FlyCameraController params; update Camera defaults to match scene"
```

---

### Task 3: Register InputState and Remove Hardcoded tick_time

**Files:**
- Modify: `src/scene/systems.rs`

- [ ] **Step 1: Update `bootstrap_scene` to register InputState**

In `src/scene/systems.rs`, add the import and registration:

```rust
use crate::platform::input::InputState;
```

Add inside `bootstrap_scene`, after existing registrations:

```rust
world.insert_resource(InputState::default());
```

- [ ] **Step 2: Remove `tick_time` system**

Delete the `tick_time` function entirely from `src/scene/systems.rs`. Also remove the `use crate::platform::time::Time;` import if it becomes unused.

- [ ] **Step 3: Remove tick_time from schedule in `src/app.rs`**

In `src/app.rs`, delete the line:

```rust
schedule.add_system(Stage::PreUpdate, systems::tick_time);
```

- [ ] **Step 4: Verify build**

Run: `cargo build`

Expected: Compiles successfully (no references to `tick_time` remain).

- [ ] **Step 5: Run all existing tests**

Run: `cargo test`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/scene/systems.rs src/app.rs
git commit -m "feat(systems): register InputState resource; remove hardcoded tick_time"
```

---

### Task 4: Add Event Handling in app.rs

**Files:**
- Modify: `src/app.rs`

- [ ] **Step 1: Add new fields to `RevolumetricApp`**

Add these fields to the struct definition:

```rust
last_cursor_pos: Option<(f64, f64)>,
```

Remove `start_time` field (no longer needed — `last_frame_time` replaces it for delta timing). Or keep it if used elsewhere; check first.

Initialize in `RevolumetricApp::new()`:

```rust
last_cursor_pos: None,
```

- [ ] **Step 2: Add `last_frame_time` initialization at end of `resumed()`**

Add field to struct:

```rust
last_frame_time: Option<std::time::Instant>,
```

Initialize as `None` in `new()`. At the very end of `resumed()` (after all init, before the final `tracing::info!`), set:

```rust
self.last_frame_time = Some(std::time::Instant::now());
```

Using `Option` avoids the first-frame spike: if `None`, the first `tick_frame` sets it to `now()` and uses `dt = 0`.

- [ ] **Step 3: Add keyboard input handling**

Add new imports at the top of `app.rs` (alongside existing `use` statements):

```rust
use winit::keyboard::{KeyCode, PhysicalKey};
use crate::platform::input::InputState;
use crate::scene::components::CameraRig;
```

Then use the short names `InputState` and `CameraRig` in all event handlers and `update_camera` below (not fully-qualified paths).

In `window_event`, add a new match arm before `_ => {}`:

```rust
WindowEvent::KeyboardInput { event, .. } => {
    if event.repeat {
        return; // ignore key repeat
    }
    let pressed = event.state == winit::event::ElementState::Pressed;
    let value = if pressed { 1.0_f32 } else { -1.0 };

    if let PhysicalKey::Code(key) = event.physical_key {
        if let Some(input) = self.world.resource_mut::<InputState>() {
            match key {
                KeyCode::KeyW => input.move_forward += value,
                KeyCode::KeyS => input.move_forward -= value,
                KeyCode::KeyD => input.move_right += value,
                KeyCode::KeyA => input.move_right -= value,
                KeyCode::Space => input.move_up += value,
                KeyCode::ShiftLeft => input.move_up -= value,
                _ => {}
            }
        }
    }
}
```

- [ ] **Step 4: Add mouse button handling**

Add match arm:

```rust
WindowEvent::MouseInput { state, button, .. } => {
    if button == winit::event::MouseButton::Right {
        let pressed = state == winit::event::ElementState::Pressed;
        if let Some(input) = self.world.resource_mut::<InputState>() {
            input.right_mouse_held = pressed;
        }
        if !pressed {
            self.last_cursor_pos = None; // prevent jump on re-press
        }
    }
}
```

- [ ] **Step 5: Add cursor moved handling**

Add match arm:

```rust
WindowEvent::CursorMoved { position, .. } => {
    if let Some(input) = self.world.resource_mut::<crate::platform::input::InputState>() {
        if input.right_mouse_held {
            if let Some((last_x, last_y)) = self.last_cursor_pos {
                input.mouse_dx += (position.x - last_x) as f32;
                input.mouse_dy += (position.y - last_y) as f32;
            }
            self.last_cursor_pos = Some((position.x, position.y));
        }
    }
}
```

- [ ] **Step 6: Add mouse wheel handling**

Add match arm:

```rust
WindowEvent::MouseWheel { delta, .. } => {
    let scroll = match delta {
        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 120.0,
    };
    if let Some(input) = self.world.resource_mut::<crate::platform::input::InputState>() {
        input.scroll_delta += scroll;
    }
}
```

- [ ] **Step 7: Add focus loss handling**

Add match arm:

```rust
WindowEvent::Focused(false) => {
    if let Some(input) = self.world.resource_mut::<crate::platform::input::InputState>() {
        input.reset_axes();
    }
    self.last_cursor_pos = None;
}
```

- [ ] **Step 8: Verify build**

Run: `cargo build`

Expected: Compiles. No runtime test yet — camera update in next task.

- [ ] **Step 9: Commit**

```bash
git add src/app.rs
git commit -m "feat(input): handle keyboard, mouse, scroll, and focus events in app.rs"
```

---

### Task 5: Camera Update Logic and Render Integration

**Files:**
- Modify: `src/app.rs`

- [ ] **Step 1: Add camera update method**

Add a private method to `RevolumetricApp`:

```rust
fn update_camera(&mut self, dt: f32) {
    // Clone InputState (it's Copy) to avoid borrow conflicts
    let input = match self.world.resource::<InputState>() {
        Some(input) => *input,
        None => return,
    };

    let rig = match self.world.resource_mut::<CameraRig>() {
        Some(rig) => rig,
        None => return,
    };
    let ctrl = &mut rig.controller;
    let cam = &mut rig.camera;

    // Scroll → adjust speed
    if input.scroll_delta != 0.0 {
        ctrl.move_speed *= ctrl.scroll_multiplier.powf(input.scroll_delta);
        ctrl.move_speed = ctrl.move_speed.clamp(ctrl.min_speed, ctrl.max_speed);
    }

    // Mouse → yaw/pitch
    let sens_rad = ctrl.mouse_sensitivity * std::f32::consts::PI / 180.0;
    ctrl.yaw += input.mouse_dx * sens_rad;
    ctrl.pitch -= input.mouse_dy * sens_rad;
    ctrl.pitch = ctrl.pitch.clamp(-1.553, 1.553); // ±89°

    // Recompute forward from yaw/pitch
    cam.forward = glam::Vec3::new(
        ctrl.pitch.cos() * ctrl.yaw.sin(),
        ctrl.pitch.sin(),
        ctrl.pitch.cos() * ctrl.yaw.cos(),
    );

    // Horizontal movement
    let hz_forward = glam::Vec3::new(ctrl.yaw.sin(), 0.0, ctrl.yaw.cos());
    let hz_right = glam::Vec3::Y.cross(hz_forward);

    let mut velocity = hz_forward * input.move_forward
        + hz_right * input.move_right
        + glam::Vec3::Y * input.move_up;
    if velocity.length_squared() > 0.0 {
        velocity = velocity.normalize();
    }

    cam.position += velocity * ctrl.move_speed * dt;
}
```

- [ ] **Step 2: Integrate into tick_frame**

At the start of `tick_frame`, before `self.schedule.run_stage(Stage::PreUpdate, ...)`:

```rust
// Real delta time
let now = std::time::Instant::now();
let dt = match self.last_frame_time {
    Some(last) => now.duration_since(last).as_secs_f32().min(0.1),
    None => 0.0,
};
self.last_frame_time = Some(now);

if let Some(time) = self.world.resource_mut::<Time>() {
    time.advance(dt);
}
```

Insert on a new line immediately after `self.schedule.run_stage(Stage::PostUpdate, &mut self.world)?;`, before the `ExtractRender` stage call:

```rust
self.update_camera(dt);
```

- [ ] **Step 3: Replace hardcoded camera with CameraRig**

In `tick_frame`, inside the `if let Some(pass) = &self.primary_ray_pass` block, replace the hardcoded camera section (lines that define `camera_pos`, `camera_target`, `camera_forward`, `camera_up`, `fov_y`) with:

```rust
let (cam_pos, cam_forward, cam_up, fov_y) = {
    let rig = self.world.resource::<CameraRig>();
    match rig {
        Some(rig) => (
            rig.camera.position,
            rig.camera.forward,
            rig.camera.up,
            rig.camera.fov_y_radians,
        ),
        None => (
            glam::Vec3::new(64.0, 80.0, -40.0),
            glam::Vec3::Z,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
        ),
    }
};

let pixel_to_ray = compute_pixel_to_ray(
    cam_pos, cam_forward, cam_up, fov_y,
    frame.swapchain_extent.width, frame.swapchain_extent.height,
);
```

Delete the old hardcoded `camera_pos`, `camera_target`, `camera_forward`, `camera_up`, `fov_y` lines.

- [ ] **Step 4: Clear per-frame input at end of tick_frame**

After the render block, before `self.schedule.run_stage(Stage::ExecuteRender, ...)`:

```rust
if let Some(input) = self.world.resource_mut::<InputState>() {
    input.clear_per_frame();
}
```

- [ ] **Step 5: Clean up unused imports in app.rs**

Remove these imports that are no longer needed (the hardcoded camera values are gone):

Check if `crate::scene::systems` import for `tick_time` path is still needed (it should be, for `bootstrap_scene`). Remove any dead imports the compiler warns about.

- [ ] **Step 6: Build and verify**

Run: `cargo build`

Expected: Compiles with no errors or warnings.

- [ ] **Step 7: Run all tests**

Run: `cargo test`

Expected: All tests pass (including existing `camera.rs` tests).

- [ ] **Step 8: Manual test**

Run: `cargo run`

Verify:
1. Window opens, sphere is visible (same initial view as before)
2. Right-click + drag rotates the camera
3. WASD moves on horizontal plane
4. Space goes up, Left Shift goes down
5. Scroll wheel changes movement speed
6. Releasing right-click and re-pressing doesn't cause a view jump
7. Alt-tabbing away and back doesn't cause camera drift

- [ ] **Step 9: Commit**

```bash
git add src/app.rs
git commit -m "feat(camera): interactive FPS camera with right-click look, WASD move, scroll speed"
```

---

### Task 6: Final Cleanup

**Files:**
- Possibly: `src/app.rs` (remove `start_time` if unused)

- [ ] **Step 1: Check for dead code**

Run: `cargo build 2>&1 | grep warning`

If `start_time` or any other field produces a dead code warning, remove it.

- [ ] **Step 2: Run full test suite**

Run: `cargo test`

Expected: All tests pass.

- [ ] **Step 3: Commit if any cleanup was needed**

```bash
git add -u
git commit -m "chore: remove dead code from camera integration"
```
