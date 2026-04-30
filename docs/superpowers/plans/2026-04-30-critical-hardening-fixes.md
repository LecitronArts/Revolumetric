# Critical Hardening Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the high-risk review findings around voxel bounds, occupancy propagation, Vulkan frame synchronization, RC probe buffer reuse, shader storage image ABI, and build verification.

**Architecture:** Keep fixes local to existing modules. CPU voxel fixes stay inside `src/voxel`; Vulkan frame/resource fixes stay inside `src/render`; RC synchronization is handled by making probe buffers frame-slot-owned; shader/build fixes make the compiled shader ABI explicit and make missing shader compilation visible.

**Tech Stack:** Rust 2024, ash/Vulkan, Slang shaders, Cargo build script, existing unit tests.

---

## File Ownership

- CPU voxel correctness: `src/voxel/ucvh.rs`, `src/voxel/occupancy.rs`, `src/voxel/brick_pool.rs`.
- Vulkan frame/resource lifecycle: `src/render/device.rs`, `src/render/pipeline.rs`, `src/render/image.rs`, `src/render/buffer.rs`.
- RC probe synchronization: `src/render/rc_probe_buffer.rs`, `src/render/passes/radiance_cascade_trace.rs`, `src/render/passes/radiance_cascade_merge.rs`, `src/render/passes/lighting.rs`, `src/app.rs`.
- Shader/build reliability: `assets/shaders/passes/primary_ray.slang`, `assets/shaders/passes/lighting.slang`, `assets/shaders/passes/test_pattern.slang`, `build.rs`, `README.md`.

Do not edit files outside your assigned ownership without reporting a blocker.

---

### Task 1: CPU Voxel Bounds And Pool Safety

**Files:**
- Modify: `src/voxel/ucvh.rs`
- Modify: `src/voxel/occupancy.rs`
- Modify: `src/voxel/brick_pool.rs`

- [ ] **Step 1: Add failing tests**

Add tests that fail on the current implementation:

```rust
#[test]
fn non_aligned_world_size_can_write_last_valid_voxel() {
    let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(129, 9, 8)));
    let cell = VoxelCell::new(1, 0, [0; 3]);

    assert!(u.set_voxel(UVec3::new(128, 8, 7), cell));
    assert_eq!(u.get_voxel(UVec3::new(128, 8, 7)).material, 1);
}

#[test]
fn tiny_world_size_can_write_valid_voxel() {
    let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(1, 1, 1)));
    let cell = VoxelCell::new(1, 0, [0; 3]);

    assert!(u.set_voxel(UVec3::ZERO, cell));
    assert_eq!(u.get_voxel(UVec3::ZERO).material, 1);
}
```

```rust
#[test]
fn odd_grid_far_corner_propagates_to_highest_level() {
    let mut h = CascadedOccupancy::new(UVec3::splat(15));
    h.set_l0(UVec3::splat(14), 0, true);
    h.rebuild();

    assert!(h.levels[3].iter().any(|node| node.flags & 1 != 0));
}
```

```rust
#[test]
fn double_free_is_rejected_without_corrupting_free_list() {
    let mut pool = BrickPool::new(2);
    let id = pool.allocate().unwrap();

    assert!(pool.free(id));
    assert!(!pool.free(id));
    assert_eq!(pool.allocated_count(), 0);

    let a = pool.allocate().unwrap();
    let b = pool.allocate().unwrap();
    assert_ne!(a, b);
}
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test voxel::ucvh::tests::non_aligned_world_size_can_write_last_valid_voxel voxel::ucvh::tests::tiny_world_size_can_write_valid_voxel voxel::occupancy::tests::odd_grid_far_corner_propagates_to_highest_level voxel::brick_pool::tests::double_free_is_rejected_without_corrupting_free_list
```

Expected: at least the non-aligned/tiny world and odd-grid tests fail or panic on current code.

- [ ] **Step 3: Implement minimal fixes**

Use ceil division for brick grid sizes:

```rust
fn div_ceil_uvec3(value: UVec3, divisor: u32) -> UVec3 {
    UVec3::new(
        value.x.div_ceil(divisor),
        value.y.div_ceil(divisor),
        value.z.div_ceil(divisor),
    )
}
```

Use ceil halving for occupancy hierarchy dimensions and keep each level at least one node when the previous level is non-empty:

```rust
let dims = std::array::from_fn(|i| {
    let mut dim = brick_grid_size;
    for _ in 0..i {
        dim = UVec3::new(dim.x.div_ceil(2), dim.y.div_ceil(2), dim.z.div_ceil(2));
    }
    dim
});
```

Track allocated brick IDs in `BrickPool` and make `free` reject invalid, unallocated, and duplicate frees:

```rust
allocated: Vec<bool>,
```

Return `bool` from `free`, update tests and call sites.

- [ ] **Step 4: Verify green**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test voxel::
```

Expected: all voxel tests pass.

---

### Task 2: Vulkan Frame Fence And Resource Cleanup

**Files:**
- Modify: `src/render/device.rs`
- Modify: `src/render/pipeline.rs`
- Modify: `src/render/image.rs`
- Modify: `src/render/buffer.rs`

- [ ] **Step 1: Add regression hooks/tests where practical**

For resource cleanup, prefer small RAII guard helpers that can be unit-tested without a live Vulkan device only if the helper is pure. Do not invent fake Vulkan handles that call real destroy functions. If a live-device unit test is impractical, document the exact manual/validation scenario in comments near the helper and verify via build/clippy.

- [ ] **Step 2: Fix fence reset ordering**

In `RenderDevice::begin_frame`, wait for the frame fence first, acquire the swapchain image, then reset the fence only after acquire succeeds and the frame will record/submit work. The out-of-date skip path must leave the fence signaled.

- [ ] **Step 3: Add cleanup on construction failures**

Ensure each partial Vulkan resource is cleaned if a later step fails:

- `ComputePipeline::new`: destroy pipeline layout if compute pipeline creation fails.
- `GpuBuffer::new`: destroy buffer if allocation or bind fails; free allocation if bind fails.
- `GpuImage::new`: destroy image if allocation/bind/view creation fails; free allocation if bind/view creation fails; destroy view only after successful creation ownership transfer.

- [ ] **Step 4: Verify**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
```

Expected: render tests and clippy pass.

---

### Task 3: RC Probe Buffer Frame-Slot Isolation

**Files:**
- Modify: `src/render/rc_probe_buffer.rs`
- Modify: `src/render/passes/radiance_cascade_trace.rs`
- Modify: `src/render/passes/radiance_cascade_merge.rs`
- Modify: `src/render/passes/lighting.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Add failing tests**

Add a CPU-only test for index behavior:

```rust
#[test]
fn per_frame_slots_swap_independently() {
    let mut indices = ProbeFrameIndices::new(3);

    assert_eq!(indices.write_index(0), 0);
    assert_eq!(indices.read_index(0), 1);
    assert_eq!(indices.write_index(1), 0);

    indices.swap_slot(0);

    assert_eq!(indices.write_index(0), 1);
    assert_eq!(indices.read_index(0), 0);
    assert_eq!(indices.write_index(1), 0);
    assert_eq!(indices.read_index(1), 1);
}
```

- [ ] **Step 2: Verify test fails**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::rc_probe_buffer::tests::per_frame_slots_swap_independently
```

Expected: fails because `ProbeFrameIndices` or equivalent frame-slot isolation does not exist.

- [ ] **Step 3: Implement per-frame-slot RC buffers**

Replace global two-buffer state with per-frame-slot double buffers. Public behavior should be:

```rust
pub fn write_buffer(&self, frame_slot: usize) -> vk::Buffer
pub fn read_buffer(&self, frame_slot: usize) -> vk::Buffer
pub fn swap_slot(&mut self, frame_slot: usize)
```

Create `2 * frame_slot_count` buffers. Update descriptor writers in trace/merge/lighting to use the passed `frame_slot`. Update `app.rs` to call `RcProbeBuffer::new(..., renderer.frame_slot_count())`, record clear for all buffers, and call `swap_slot(frame.frame_slot)` only after `end_frame`.

- [ ] **Step 4: Verify**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::rc_probe_buffer
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::passes::
```

Expected: RC buffer and pass tests pass.

---

### Task 4: Shader ABI And Build Reliability

**Files:**
- Modify: `assets/shaders/passes/primary_ray.slang`
- Modify: `assets/shaders/passes/lighting.slang`
- Modify: `assets/shaders/passes/test_pattern.slang`
- Modify: `build.rs`
- Modify: `README.md`

- [ ] **Step 1: Add/strengthen source tests**

Extend existing shader source tests or add lightweight tests that assert required image format annotations are present:

```rust
assert!(source.contains("[[vk::image_format(\"rgba32f\")]]"));
assert!(source.contains("[[vk::image_format(\"rgba8\")]]"));
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::passes
```

Expected: format annotation test fails on current shader sources.

- [ ] **Step 3: Make shader image formats explicit**

Add:

```slang
[[vk::image_format("rgba32f")]]
```

to `gbuffer_pos` storage images, and:

```slang
[[vk::image_format("rgba8")]]
```

to RGBA8 `float4` output storage images. Keep existing `rgba8ui` annotations.

- [ ] **Step 4: Harden build script visibility**

Add directory rerun tracking:

```rust
println!("cargo:rerun-if-changed=assets/shaders");
println!("cargo:rerun-if-changed=assets/shaders/passes");
```

Keep `skip` available for CPU-only tests, but update README so CI/release validation uses strict shader compilation. Ensure RC trace/merge empty SPIR-V paths log warnings like primary/lighting.

- [ ] **Step 5: Verify**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo build --lib
```

Expected: strict shader compilation succeeds and tests pass.

---

## Final Verification

Run after all tasks are integrated:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo build --lib
```

If `cargo build` for the binary still fails with `os error 5` while `cargo build --lib` succeeds, report it separately as local executable lock/permission risk rather than source failure.
