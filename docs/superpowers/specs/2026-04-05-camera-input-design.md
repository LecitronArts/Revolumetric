# Phase 9: Camera & Input — Design Spec

## Goal

Replace the hardcoded camera in `app.rs` with an interactive FPS camera controller: WASD movement on the horizontal plane, mouse look (right-click drag), scroll-wheel speed adjustment, and real delta time.

## Constraints

- **No new files** — extend existing stubs (`InputState`, `CameraRig`, `FlyCameraController`)
- **No new dependencies** — winit 0.30 already provides all needed events
- **5 files modified**: `platform/input.rs`, `scene/camera.rs`, `scene/components.rs`, `scene/systems.rs`, `app.rs`
- **Camera style**: Classic FPS — movement projected to XZ horizontal plane, pitch limited ±89°
- **Mouse mode**: Right-click drag to rotate (cursor stays free otherwise)

## Data Structures

### `InputState` (`src/platform/input.rs`)

Expanded from current 3-field stub:

```rust
#[derive(Debug, Default, Clone, Copy)]
pub struct InputState {
    // Movement axes (-1.0 to 1.0, accumulated from key presses)
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    // Mouse delta (pixels this frame, cleared each frame)
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    // Right mouse button held
    pub right_mouse_held: bool,
    // Scroll wheel delta (this frame, cleared each frame)
    pub scroll_delta: f32,
}
```

Registered as a `World` resource in `bootstrap_scene`.

### `FlyCameraController` (`src/scene/components.rs`)

Expanded from current single-field stub:

```rust
#[derive(Debug, Clone)]
pub struct FlyCameraController {
    pub move_speed: f32,        // Current speed (units/s), default 50.0
    pub min_speed: f32,         // 5.0
    pub max_speed: f32,         // 500.0
    pub scroll_multiplier: f32, // 1.2 per scroll tick
    pub mouse_sensitivity: f32, // 0.3 degrees/pixel
    pub pitch: f32,             // Current pitch (radians), clamped ±89°
    pub yaw: f32,               // Current yaw (radians)
}
```

### `CameraRig` (`src/scene/components.rs`)

No structural change — still holds `Camera` + `FlyCameraController` + `Transform`. The `Transform` field becomes unused for now (camera position lives in `Camera::position`).

### Initial Values

Derived from current hardcoded camera (pos=(64,80,-40), target=(64,64,64)):

```
forward = normalize((0, -16, 104)) ≈ (0, -0.152, 0.988)
pitch = asin(-0.152) ≈ -0.153 rad ≈ -8.7°
yaw = atan2(0, 0.988) ≈ 0 rad
```

Default values:
- `Camera::position`: `(64.0, 80.0, -40.0)`
- `Camera::fov_y_radians`: `FRAC_PI_4` (45°, matches current hardcoded render value)
- `Camera::forward`: computed from yaw/pitch below (replaces old default `(0, 0, -1)`)
- `FlyCameraController::pitch`: `-0.153`
- `FlyCameraController::yaw`: `0.0`
- `FlyCameraController::move_speed`: `50.0`

Note: `Camera::default()` in `scene/camera.rs` must be updated to match these values.

## Event Handling (`app.rs`)

### New State on `RevolumetricApp`

```rust
last_cursor_pos: Option<(f64, f64)>,
last_frame_time: Option<std::time::Instant>,
```

`last_frame_time` starts as `None`. On first `tick_frame`, it is set to `now()` with `dt = 0`. This avoids a multi-second first-frame delta spike from Vulkan init. Additionally, delta_time is capped at 0.1s as a safety measure.

### `window_event` Additions

| winit Event | Action |
|---|---|
| `KeyboardInput` (W/S pressed/released) | `move_forward += 1.0` / `-= 1.0` (W=+1, S=-1) |
| `KeyboardInput` (A/D pressed/released) | `move_right += 1.0` / `-= 1.0` (D=+1, A=-1) |
| `KeyboardInput` (Space/LShift pressed/released) | `move_up += 1.0` / `-= 1.0` (Space=+1, LShift=-1) |
| `Focused(false)` | Reset all movement axes to `0.0` (prevents stuck keys on focus loss) |
| `MouseInput` (Right button) | `right_mouse_held = true/false`; clear `last_cursor_pos` on release |
| `CursorMoved` | If `right_mouse_held`: compute delta from `last_cursor_pos`, **accumulate** `mouse_dx/dy` (`+=`); update `last_cursor_pos` |
| `MouseWheel` | Accumulate `scroll_delta` (positive = scroll up = speed increase) |

Key state uses add/subtract pattern so opposing keys cancel (W+S = 0).

Clearing `last_cursor_pos` on right-button release prevents jump when re-pressing.

### `tick_frame` Camera Update

Executed before render, after `PreUpdate` stage:

```
1. Compute real delta_time = now - last_frame_time (capped at 0.1s); update Time resource
2. Read InputState, CameraRig, Time from World
3. Scroll → move_speed *= scroll_multiplier^scroll_delta, clamp to [min, max]
4. Mouse delta → yaw += dx * sensitivity * (pi/180), pitch -= dy * sensitivity * (pi/180)
   pitch clamped to ±89° (±1.553 rad)
5. Recompute forward from yaw/pitch:
     forward.x = cos(pitch) * sin(yaw)
     forward.y = sin(pitch)
     forward.z = cos(pitch) * cos(yaw)
6. Horizontal movement:
     hz_forward = (sin(yaw), 0, cos(yaw))
     hz_right = cross(Y, hz_forward)
     velocity = (hz_forward * move_forward + hz_right * move_right + Y * move_up)
     if velocity.length_squared() > 0.0: velocity = velocity.normalize()
     position += velocity * move_speed * delta_time
7. Write back to CameraRig.camera (position, forward)
8. Clear per-frame InputState fields (mouse_dx, mouse_dy, scroll_delta)
9. Use CameraRig.camera for compute_pixel_to_ray (replaces hardcoded values)
```

## Real Delta Time (`src/scene/systems.rs`)

Change `tick_time` from hardcoded `1.0/60.0` to use `Instant`-based measurement. The `last_frame_time` field on `RevolumetricApp` provides the anchor; delta is computed at the start of `tick_frame` and written to the `Time` resource. The `tick_time` system in the schedule becomes a no-op or is removed — delta is set directly in `tick_frame`.

Simpler approach: remove `tick_time` from the schedule, compute delta in `tick_frame`, call `time.advance(real_delta)` directly.

## Camera Math

### Yaw/Pitch to Forward

```
forward = (cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw))
```

- Yaw=0 → forward=(0, 0, 1) → looks along +Z (into the scene)
- Pitch=0 → horizontal
- Pitch>0 → looks up, Pitch<0 → looks down

### Pitch Clamp

```
pitch = clamp(pitch, -89°, +89°) = clamp(pitch, -1.553 rad, +1.553 rad)
```

Prevents gimbal lock / camera flip at poles.

### Horizontal Plane Projection

Movement always on XZ plane regardless of pitch:

```
hz_forward = (sin(yaw), 0, cos(yaw))   // already unit length (sin²+cos²=1)
hz_right = Vec3::Y.cross(hz_forward)    // also unit length (perpendicular unit vectors)
```

## Files Changed Summary

| File | Change |
|---|---|
| `src/platform/input.rs` | Expand `InputState` with mouse/scroll/right-click fields |
| `src/scene/camera.rs` | Update `Camera::default()` to match initial values (position, FOV) |
| `src/scene/components.rs` | Expand `FlyCameraController` with speed/sensitivity/yaw/pitch; update defaults |
| `src/scene/systems.rs` | Remove `tick_time` system; register `InputState` resource in `bootstrap_scene` |
| `src/app.rs` | Add keyboard/mouse/focus event handling; add camera update logic in `tick_frame`; use CameraRig for rendering; add `last_cursor_pos`/`last_frame_time` fields; real delta time; init `last_frame_time` at end of `resumed()` |

## Out of Scope

- Configurable keybindings (future)
- Cursor lock/hide (chose free cursor + right-click drag)
- Collision detection
- Camera smoothing / interpolation
- Gamepad support
