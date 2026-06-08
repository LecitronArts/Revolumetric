# Egui Editor Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a desktop `egui` editor shell that overlays the existing Vulkan VPT renderer and drives real-time render settings.

**Architecture:** `src/editor` owns pure egui UI state and setting mutation. `src/render/egui_renderer.rs` owns Vulkan resources needed to draw tessellated egui meshes as the final swapchain overlay pass. `app.rs` wires winit events into egui first, then gates existing camera input based on egui consumption.

**Tech Stack:** Rust 2024, winit 0.30, egui 0.30, egui-winit 0.30, ash/Vulkan 1.3, existing RenderGraph and render runtime.

---

## File Structure

- Modify `Cargo.toml`: add `egui` and `egui-winit` desktop dependencies.
- Modify `Cargo.lock`: update via Cargo.
- Create `src/editor/mod.rs`: editor module exports.
- Create `src/editor/fonts.rs`: optional asset/system font loading and egui font definition setup.
- Create `src/editor/ui.rs`: fixed editor shell layout and controls for runtime settings.
- Create `src/render/egui_renderer.rs`: custom Vulkan egui mesh renderer.
- Modify `src/render/mod.rs`: export `egui_renderer`.
- Modify `src/render/resource.rs`: add color attachment access.
- Modify `src/render/device.rs`: enable dynamic rendering and expose limits needed by egui renderer.
- Modify `src/render/frame.rs`: carry optional UI overlay data if needed by runtime.
- Modify `src/render/runtime.rs`: own optional `EguiRenderer` and pass `EguiFrame` into frame rendering.
- Modify `src/render/vpt_pipeline.rs`: add egui overlay pass after blit and before present.
- Modify `src/app.rs`: own egui context/state/editor, route events, build UI each frame, and gate camera input.
- Modify `README.md`: document editor UI and font assets.

---

### Task 1: Add Dependencies And CPU-Only Editor Modules

**Files:**
- Modify: `Cargo.toml`
- Create: `src/editor/mod.rs`
- Create: `src/editor/fonts.rs`
- Create: `src/editor/ui.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write failing source test for editor module export**

Add a source test in `src/lib.rs` test module or a new editor test module that asserts `src/lib.rs` exports `pub mod editor;`.

Run: `$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib editor_module_is_exported; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE`

Expected: FAIL because `editor` is not exported.

- [ ] **Step 2: Add egui dependencies**

Add:

```toml
egui = { version = "0.30", features = ["bytemuck"] }
egui-winit = { version = "0.30", default-features = false, features = ["bytemuck", "clipboard"] }
```

Keep them as normal dependencies because the library currently compiles desktop and Android code behind cfg gates. If Android compilation pulls in unsupported desktop-only pieces, gate editor modules with `#[cfg(not(target_os = "android"))]`.

- [ ] **Step 3: Add editor module shell**

Add `pub mod editor;` to `src/lib.rs`.

Create `src/editor/mod.rs`:

```rust
#[cfg(not(target_os = "android"))]
pub mod fonts;
#[cfg(not(target_os = "android"))]
pub mod ui;
```

- [ ] **Step 4: Run focused test**

Run: `$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib editor_module_is_exported; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE`

Expected: PASS.

### Task 2: Font Loading And Editor Theme

**Files:**
- Modify: `src/editor/fonts.rs`
- Modify: `src/editor/ui.rs`

- [ ] **Step 1: Write failing font fallback tests**

Tests:

- Missing optional font assets returns warnings and a usable `FontDefinitions`.
- Family priority includes Inter/Noto for proportional and JetBrains/Noto for monospace when assets are present.

Run focused editor font tests. Expected: FAIL because functions do not exist.

- [ ] **Step 2: Implement `EditorFontReport` and `configure_editor_fonts`**

Implement a pure function that accepts a list of asset paths, tries to load known names, inserts available fonts into `egui::FontDefinitions`, and returns warnings for missing optional fonts.

- [ ] **Step 3: Implement `configure_editor_style`**

Set graphite visual theme, compact spacing, panel colors, text style sizes, and no purple default accents.

- [ ] **Step 4: Run focused tests**

Run editor font/style tests. Expected: PASS.

### Task 3: Editor UI Settings Model

**Files:**
- Modify: `src/editor/ui.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Write failing settings-bound tests**

Add tests for pure clamp/helper functions:

- VPT bounce clamp stays in `1..=8`.
- Denoiser iterations stay in `0..=5`.
- Area ReSTIR debug bridge maps to VPT debug views through existing app helper.
- Enabling/disabling ReSTIR flags mutates the existing settings structs without separate shadow state.

Expected: FAIL because helpers do not exist.

- [ ] **Step 2: Implement editor UI state**

Create:

```rust
pub struct EditorUi {
    pub visible: bool,
    pub selected_panel: EditorPanel,
    pub console_lines: Vec<String>,
}
```

Create borrowed frame state:

```rust
pub struct EditorUiFrameState<'a> {
    pub lighting: &'a mut LightingSettings,
    pub restir_di: &'a mut RestirDiSettings,
    pub area_restir: &'a mut AreaRestirSettings,
    pub camera: VptCameraFrame,
    pub viewport_extent: [u32; 2],
    pub rendered_frames: u64,
}
```

- [ ] **Step 3: Implement fixed editor shell**

Implement top bar, left scene panel, right inspector, bottom console, and center overlay with `egui` panels.

- [ ] **Step 4: Run focused tests**

Run editor UI tests. Expected: PASS.

### Task 4: RenderGraph Access For UI Overlay

**Files:**
- Modify: `src/render/resource.rs`
- Modify: `src/render/graph.rs`

- [ ] **Step 1: Write failing RenderGraph access test**

Add a test that imports swapchain as `TransferWrite`, adds a graphics pass that writes it as `ColorAttachmentWrite`, and finishes as `Present`.

Expected barriers:

- `TransferWrite -> ColorAttachmentWrite`
- `ColorAttachmentWrite -> Present`

Expected: FAIL because `ColorAttachmentWrite` does not exist.

- [ ] **Step 2: Implement access kind**

Add `AccessKind::ColorAttachmentWrite` with:

- stage: `COLOR_ATTACHMENT_OUTPUT`
- access: `COLOR_ATTACHMENT_WRITE`
- layout: `COLOR_ATTACHMENT_OPTIMAL`

Update validation so present final access accepts either transfer write or color attachment write in the same pass.

- [ ] **Step 3: Run RenderGraph tests**

Run: `$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph -- --nocapture; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE`

Expected: PASS for relevant graph tests.

### Task 5: Vulkan Egui Renderer

**Files:**
- Create: `src/render/egui_renderer.rs`
- Modify: `src/render/mod.rs`
- Modify: `src/render/device.rs`
- Modify: `src/render/runtime.rs`

- [ ] **Step 1: Write failing source tests**

Add source tests that assert:

- `RenderDevice` enables dynamic rendering.
- `EguiRenderer` owns sampler, descriptor layout/pool, pipeline layout, pipeline, and per-frame buffers.
- `RenderRuntime` owns `Option<EguiRenderer>`.

Expected: FAIL because code does not exist.

- [ ] **Step 2: Enable dynamic rendering**

Add `vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true)` to device creation pNext chain and validate support when selecting devices.

- [ ] **Step 3: Implement `EguiFrame` and `EguiRenderer` resource shell**

Create:

```rust
pub struct EguiFrame {
    pub clipped_primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}
```

Implement renderer creation/destruction and no-op record path for empty primitives.

- [ ] **Step 4: Implement texture upload and mesh buffers**

Handle full font atlas upload first. Handle partial updates by writing changed rectangles into the existing texture. Allocate host-visible vertex/index buffers sized for current frame.

- [ ] **Step 5: Implement dynamic rendering draw**

Record `cmd_begin_rendering`, viewport, scissor per clipped primitive, bind graphics pipeline, descriptor set, vertex/index buffers, and draw indexed meshes with alpha blending.

- [ ] **Step 6: Run focused source/build tests**

Run relevant render tests and `cargo build --features desktop --bin revolumetric`.

### Task 6: App Event Integration

**Files:**
- Modify: `src/app.rs`
- Modify: `src/render/runtime.rs`
- Modify: `src/render/vpt_pipeline.rs`

- [ ] **Step 1: Write failing input gate tests**

Add tests for a pure helper that decides whether app camera input should receive an event based on `egui_winit::EventResponse`.

Expected: FAIL because helper does not exist.

- [ ] **Step 2: Add egui state to app**

Initialize `egui::Context`, `egui_winit::State`, font definitions, style, and `EditorUi` after window creation.

- [ ] **Step 3: Route events through egui first**

For `WindowEvent`s, call `egui_state.on_window_event(window, &event)` before app-specific event handling. Skip keyboard, pointer, wheel, and touch camera updates when consumed.

- [ ] **Step 4: Build egui frame before render frame**

In `tick_frame`, run editor UI before calling runtime render and pass resulting `EguiFrame` to render runtime.

- [ ] **Step 5: Integrate overlay in VPT pipeline**

When an `EguiFrame` is present and renderer is initialized, add the egui overlay graphics pass after the blit. The egui pass owns the final `Present` access.

- [ ] **Step 6: Run tests**

Run app/editor/render focused tests.

### Task 7: Documentation And Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-05-egui-editor-shell.md`

- [ ] **Step 1: Document runtime usage**

Add README notes for:

- editor UI enabled by default on desktop
- expected font asset names
- fallback behavior
- current first-phase editor limits

- [ ] **Step 2: Run verification**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
cargo build --features desktop --bin revolumetric
```

If desktop build needs `slangc`, use the configured environment or report the exact failure.

- [ ] **Step 3: Review diff**

Run:

```powershell
git diff -- Cargo.toml src README.md docs/superpowers
git status --short --branch
```

Confirm no unrelated user changes were reverted.

---

## Self-Review

- Spec coverage: The plan covers dependencies, fonts, UI layout, settings mutation, input gating, render graph access, Vulkan overlay rendering, runtime integration, and documentation.
- Placeholder scan: No task uses deferred TODO/TBD behavior.
- Type consistency: `EditorUi`, `EditorUiFrameState`, `EguiFrame`, `EguiRenderer`, and `AccessKind::ColorAttachmentWrite` are named consistently across tasks.
