# Swapchain Lifecycle Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RenderDevice` report swapchain recreation to `RenderRuntime` so runtime-owned VPT resources can be synchronized after implicit device recreation paths.

**Architecture:** `FrameContext` carries acquire-side recreation state, `FrameCompletion` carries present-side recreation state, and `RenderRuntime` funnels explicit and implicit resize synchronization through one private helper. The phase preserves existing Vulkan frame ordering and pass resize behavior.

**Tech Stack:** Rust, ash/Vulkan wrappers, existing source-level tests, existing `cargo test --lib` verification path.

---

## File Structure

- Modify `src/render/frame.rs`: add `FrameCompletion`, add `FrameContext::swapchain_recreated`, and add `skip_after_swapchain_recreate`.
- Modify `src/render/device.rs`: change `end_frame` return type to `Result<FrameCompletion>` and add device source/value tests.
- Modify `src/render/runtime.rs`: add `resize_pipeline_to_swapchain`, consume acquire/present recreation signals, and add runtime source tests.
- Modify docs under `docs/superpowers/specs` and `docs/superpowers/plans`.

---

### Task 1: Frame-Level Recreate Signals

**Files:**
- Modify: `src/render/frame.rs`
- Modify: `src/render/device.rs`

- [ ] **Step 1: Write failing frame tests**

Add tests in `src/render/frame.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skipped_frame_can_report_swapchain_recreation() {
        let frame = FrameContext::skip_after_swapchain_recreate(42);

        assert_eq!(frame.frame_index, 42);
        assert!(!frame.should_render);
        assert!(frame.swapchain_recreated);
    }

    #[test]
    fn frame_completion_defaults_to_no_swapchain_recreation() {
        assert!(!FrameCompletion::default().swapchain_recreated);
    }
}
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::frame::tests::skipped_frame_can_report_swapchain_recreation render::frame::tests::frame_completion_defaults_to_no_swapchain_recreation; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile failure because `FrameCompletion` and `skip_after_swapchain_recreate` do not exist.

- [ ] **Step 3: Implement frame values**

Add `FrameCompletion`, `swapchain_recreated`, and `skip_after_swapchain_recreate` in `src/render/frame.rs`.

- [ ] **Step 4: Update `RenderDevice::begin_frame`**

Change the acquire out-of-date path from:

```rust
return Ok(FrameContext::skip(self.frame_index));
```

to:

```rust
return Ok(FrameContext::skip_after_swapchain_recreate(self.frame_index));
```

- [ ] **Step 5: Verify GREEN**

Run the same focused frame tests. Expected: both pass.

---

### Task 2: Present-Side Completion Signal

**Files:**
- Modify: `src/render/device.rs`
- Modify: `src/render/runtime.rs`

- [ ] **Step 1: Write failing device source test**

Add to `src/render/device.rs` tests:

```rust
#[test]
fn device_reports_swapchain_recreation_from_frame_completion() {
    let source = crate::render::source_checks::read_source("src/render/device.rs");
    let end_frame = source
        .split("pub fn end_frame")
        .nth(1)
        .expect("RenderDevice::end_frame should exist")
        .split("pub fn wait_for_fence")
        .next()
        .expect("end_frame should end before wait_for_fence");

    assert!(end_frame.contains("Result<FrameCompletion>"));
    assert!(end_frame.contains("let mut completion = FrameCompletion::default();"));
    assert!(end_frame.contains("completion.swapchain_recreated = true;"));
    assert!(end_frame.contains("Ok(completion)"));
}
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::device::tests::device_reports_swapchain_recreation_from_frame_completion; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: fail because `end_frame` still returns `Result<()>`.

- [ ] **Step 3: Change `RenderDevice::end_frame`**

Import `FrameCompletion` and return it:

```rust
pub fn end_frame(&mut self, ctx: FrameContext) -> Result<FrameCompletion> {
    if !ctx.should_render {
        return Ok(FrameCompletion::default());
    }
    let mut completion = FrameCompletion::default();
    ...
    completion.swapchain_recreated = true;
    ...
    Ok(completion)
}
```

Set `completion.swapchain_recreated = true` in both present recreate branches.

- [ ] **Step 4: Update runtime call sites**

Temporarily bind the return value from `self.renderer.end_frame(frame)?` in `RenderRuntime::render_frame`; Task 3 will consume it.

- [ ] **Step 5: Verify GREEN**

Run the focused device test. Expected: pass.

---

### Task 3: Runtime Resize Synchronization

**Files:**
- Modify: `src/render/runtime.rs`

- [ ] **Step 1: Write failing runtime source test**

Add to `src/render/runtime.rs` tests:

```rust
#[test]
fn render_runtime_observes_swapchain_recreate_signals() {
    let source = crate::render::source_checks::read_source("src/render/runtime.rs");
    let runtime_impl = source
        .split("impl RenderRuntime")
        .nth(1)
        .expect("RenderRuntime impl should exist");

    assert!(runtime_impl.contains("fn resize_pipeline_to_swapchain("));
    assert!(runtime_impl.contains("self.renderer.swapchain_extent()"));
    assert!(runtime_impl.contains("if frame.swapchain_recreated"));
    assert!(runtime_impl.contains("if completion.swapchain_recreated"));
}
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::runtime::tests::render_runtime_observes_swapchain_recreate_signals; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: fail because the helper and checks do not exist.

- [ ] **Step 3: Add `resize_pipeline_to_swapchain`**

Implement:

```rust
fn resize_pipeline_to_swapchain(
    &mut self,
    ucvh: Option<&Ucvh>,
    settings: RuntimeSettings,
    restir_di_enabled: bool,
    area_restir_enabled: bool,
) -> Result<()> {
    self.ensure_passes(ucvh, settings, restir_di_enabled, area_restir_enabled);
    let extent = self.renderer.swapchain_extent();
    if let (Some(scene_ubo), Some(ucvh_gpu)) = (self.scene_ubo.as_ref(), self.ucvh_gpu.as_ref()) {
        self.vpt_pipeline.resize(
            &self.renderer,
            scene_ubo,
            ucvh_gpu,
            extent.width,
            extent.height,
            settings.lighting,
            restir_di_enabled,
            area_restir_enabled,
        )?;
    }
    Ok(())
}
```

- [ ] **Step 4: Reuse helper from explicit resize**

Change `RenderRuntime::resize` to call `handle_resize`, then `resize_pipeline_to_swapchain`.

- [ ] **Step 5: Consume acquire/present signals in `render_frame`**

After `begin_frame`, if `frame.swapchain_recreated`, call helper and return the skipped outcome.

After each `end_frame`, if `completion.swapchain_recreated`, call helper before returning or processing capture.

- [ ] **Step 6: Verify GREEN**

Run the focused runtime test. Expected: pass.

---

### Task 4: Full Verification

**Files:**
- All modified files above.

- [ ] **Step 1: Format**

Run:

```powershell
cargo fmt
```

- [ ] **Step 2: Whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors; existing LF/CRLF warnings are acceptable.

- [ ] **Step 3: Full library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all library tests pass.

---

## Execution Results

- Added frame-level swapchain recreation signals in `src/render/frame.rs`.
- Changed `RenderDevice::end_frame` to return `FrameCompletion`.
- Routed acquire-side and present-side swapchain recreation through `RenderRuntime::resize_pipeline_to_swapchain`.
- Verification command:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

- Result observed during execution: 446 passed, 0 failed.
