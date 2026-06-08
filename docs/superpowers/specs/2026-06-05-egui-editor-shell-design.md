# Egui Editor Shell Design

## Goal

Add a high-quality editor-style UI shell to the desktop Revolumetric runtime using `egui`. The first version must be useful for real-time renderer tuning and must create a stable foundation for a future full editor without replacing the existing custom Vulkan renderer.

## Current Facts

- The desktop app uses `winit` and a custom `ash` Vulkan backend.
- Rendering is VPT-only and is orchestrated through `RenderRuntime` and `VptRuntimePipeline`.
- The final image path is compute postprocess to an RGBA8 image, then a transfer blit to the swapchain.
- There is no existing UI layer, graphics pipeline helper, sampled texture path, or dock layout system.
- Runtime knobs already exist as strongly typed settings:
  - `LightingSettings`
  - `RestirDiSettings`
  - `AreaRestirSettings`
- The app currently owns settings and feeds `RuntimeSettings` into each frame.
- The repository has no font assets. The local Windows install contains Noto Sans SC, but not Inter or JetBrains Mono.

## Scope

This phase builds a real editor shell, not only loose debug sliders.

Included:

- `egui` + `egui-winit` integration for the desktop app.
- A custom Vulkan `EguiRenderer` that draws `egui` meshes over the swapchain as the final render step.
- A fixed editor layout with top bar, left scene panel, central viewport overlay, right inspector, and bottom console/profiler-style panel.
- Real-time controls for existing renderer settings.
- Input capture so UI interactions do not leak into FPS camera controls.
- Font setup with Inter/Noto Sans SC/JetBrains Mono asset support and safe fallback.
- Unit/source tests for settings UI behavior, input gating, and render graph/access contracts where GPU execution is not practical in CI.

Excluded:

- Docking/tab persistence.
- Entity hierarchy editing beyond placeholder scene/viewport panels.
- File import/export.
- Saving settings to disk.
- GPU profiler UI backed by complete timing history. The bottom panel can show current runtime summaries and leave the data seam ready.

## Architecture

### App/UI Boundary

Add `src/editor/` as an app-level UI boundary. It owns pure editor state and the `egui` frame construction logic. It must not own Vulkan resources.

`RevolumetricApp` owns:

- `Option<EditorUi>`
- `Option<egui_winit::State>`
- `egui::Context`
- app settings already used by the renderer

Each redraw:

1. `egui_winit::State` produces `RawInput`.
2. `EditorUi` runs panels and mutates app-owned settings through a borrowed `EditorUiFrameState`.
3. `egui::Context` tessellates shapes.
4. `RenderRuntime::render_frame` receives an optional `EguiFrame` containing textures delta, clipped primitives, and pixels-per-point.
5. `EguiRenderer` draws the final overlay after the swapchain blit and transitions the swapchain to present.

### Render Boundary

Add a focused `src/render/egui_renderer.rs`.

It owns:

- Vulkan sampler.
- Font/user texture images currently required by egui.
- Per-frame host-visible vertex/index buffers.
- Descriptor set layout/pool/sets.
- Pipeline layout and graphics pipeline.

The renderer records into the existing frame command buffer. It does not create its own swapchain, window, or queue.

The render graph gains a color attachment access kind so the final pass can:

1. Read the postprocess output as transfer source.
2. Write swapchain as transfer destination for the existing blit.
3. Write swapchain as color attachment for `egui`.
4. Finish swapchain as present.

### Graphics Pipeline

Use Vulkan dynamic rendering rather than legacy render passes. This matches Vulkan 1.3 and avoids building a render-pass/framebuffer lifetime system only for UI. Device creation must explicitly enable dynamic rendering if required by `ash` feature structs on the target platform.

The egui shaders are small checked-in GLSL/SPIR-V assets or Rust-side included SPIR-V. They transform egui points to clip space, sample the egui atlas, multiply premultiplied vertex color, and alpha-blend over the existing swapchain image.

### Editor Layout

Use a fixed, intentional layout:

- Top command bar: app title, active renderer, denoiser mode, debug view, frame resolution.
- Left rail: scene summary, camera readout, future object/tool placeholders.
- Center: viewport overlay controls with camera hints and selected debug view.
- Right inspector: grouped tuning panels:
  - Lighting
  - VPT
  - Denoiser
  - ReSTIR-DI
  - Area ReSTIR
  - Debug Views
- Bottom panel: console/profiler-style summary of current settings and runtime state.

The visual style should be editor-grade: dark graphite base, amber/cyan accents, compact typography, strong panel separation, and no generic purple theme.

### Fonts

The preferred configuration is:

- Proportional family: Inter first, Noto Sans SC fallback, egui defaults after.
- Monospace family: JetBrains Mono first, Noto Sans SC fallback, egui defaults after.

Implementation must load fonts from `assets/fonts` when present:

- `Inter-Regular.ttf`
- `NotoSansSC-Regular.otf` or `NotoSansSC-VF.ttf`
- `JetBrainsMono-Regular.ttf`

If a font asset is missing, startup continues and logs a warning. Noto Sans SC may be attempted from known Windows font paths as a development fallback, but runtime correctness cannot depend on the user's OS font directory.

## Data Flow

UI controls mutate the existing settings structs directly before `runtime_settings()` is built for the frame. Existing VPT scene-key logic resets accumulation/history when relevant settings change.

Controls must clamp to existing parser/runtime ranges:

- `vpt_max_bounces`: `1..=8`
- `exposure`: finite non-negative, UI max can be a practical editor range
- `sun_angular_radius`: `0.0..=0.25`
- `denoiser_atrous_iterations`: `0..=5`
- ReSTIR candidate/history/sample counts: existing struct/parser ranges
- Area ReSTIR thresholds/radius: finite positive editor ranges

## Input

All `WindowEvent`s are first offered to `egui_winit::State`. If `egui` consumes pointer or keyboard input, app-level camera controls skip that event.

Right mouse camera look remains available when the pointer is not over an interactive egui area. Focus loss still resets movement axes and cursor grab.

## Error Handling

- If egui renderer initialization fails, the app logs the error and continues rendering the scene without UI.
- If a texture upload fails, the frame logs the error and skips only UI rendering for that frame.
- If font loading fails, the app logs warnings and uses egui default fonts.
- If the graphics pass cannot be recorded because no primitives exist, it still transitions the swapchain correctly when it owns the present final access.

## Testing

CPU/source tests cover:

- Editor settings controls preserve configured numeric bounds.
- Area ReSTIR debug view bridges to VPT debug view consistently.
- App event handling does not update camera input when egui consumes a key/mouse event.
- Render graph can plan `TransferWrite -> ColorAttachmentWrite -> Present`.
- `EguiFrame` with empty primitives is treated as optional overlay and does not block rendering.
- Font loader reports missing optional fonts without panicking.

Manual/runtime validation:

- `cargo test`
- `cargo build --features desktop --bin revolumetric`
- If local shader/compiler environment supports it, open the app and verify the editor shell renders and controls mutate the image.

## Risks

- Dynamic rendering support must be enabled correctly on Vulkan device creation.
- Blending onto UNORM swapchain must preserve the existing postprocess gamma expectations.
- `egui` font atlas uploads require careful staging and partial update handling.
- The current dirty worktree contains existing changes in the same files; integration must avoid reverting unrelated work.
