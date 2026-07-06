# RT Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the hardware RT path by making miss/background direction explicit, keeping editor overlay visible on RT fallback presentation, and adding final RT visual baseline coverage.

**Architecture:** Keep the current RT frame graph and pass structure, but strengthen contracts at the data and presentation boundaries. Extend `RtSurfacePixel` with a named primary/background direction field, make direct lighting consume that field for invalid surfaces, centralize RT swapchain overlay/present handling, and cover both debug and final RT outputs in local validation.

**Tech Stack:** Rust, Vulkan via `ash`, project RenderGraph, Slang RT shaders, PowerShell visual baseline runner, Cargo tests/clippy/build.

---

## File Structure

- Modify `src/render/rt_history.rs`: CPU ABI for `GpuRtSurfacePixel` and layout/source checks for the matching Slang struct.
- Modify `assets/shaders/shared/rt_history_common.slang`: shared `RtSurfacePixel` ABI with `view_direction_background`.
- Modify `assets/shaders/shared/rt_surface_common.slang`: payload-to-surface conversion writes the explicit background/view direction.
- Modify `assets/shaders/passes/rt_surface.rgen.slang`: primary rays continue to seed the payload with camera ray direction.
- Modify `assets/shaders/passes/rt_surface.rmiss.slang`: true misses keep miss semantics and the existing payload ray direction.
- Modify `assets/shaders/passes/rt_surface.rchit.slang`: brick-local voxel traversal misses preserve the original primary ray direction and do not rely on AABB normals for background.
- Modify `src/render/passes/rt_surface.rs`: source tests for payload and local-miss direction contracts.
- Modify `assets/shaders/passes/rt_direct_lighting.rgen.slang`: invalid surface background shading uses `surface.view_direction_background.xyz`.
- Modify `src/render/passes/rt_direct_lighting.rs`: source/layout tests for direct-lighting uniforms and background direction; rustfmt-clean update signature.
- Modify `src/render/rt_pipeline.rs`: helper-based RT swapchain present/egui composition for both blit and clear fallback paths, plus source contract tests.
- Modify `run/visual-baselines.json`: add `rt_final` case beside `rt_surface_debug`.
- Modify `src/render/source_checks.rs`: require `rt_final` and signal thresholds in the manifest check.

## Task 1: Explicit RT Surface Background Direction ABI

**Files:**
- Modify: `src/render/rt_history.rs`
- Modify: `assets/shaders/shared/rt_history_common.slang`
- Modify: `assets/shaders/shared/rt_surface_common.slang`
- Modify: `assets/shaders/passes/rt_surface.rgen.slang`
- Modify: `assets/shaders/passes/rt_surface.rmiss.slang`
- Modify: `assets/shaders/passes/rt_surface.rchit.slang`
- Modify: `src/render/passes/rt_surface.rs`

- [ ] **Step 1: Write failing CPU/Slang ABI tests**

In `src/render/rt_history.rs`, update `rt_surface_pixel_layout_is_stable` so it expects a named `view_direction_background` field after `motion_history`:

```rust
#[test]
fn rt_surface_pixel_layout_is_stable() {
    assert_eq!(std::mem::size_of::<GpuRtSurfacePixel>(), 112);
    assert_eq!(std::mem::offset_of!(GpuRtSurfacePixel, position_depth), 0);
    assert_eq!(
        std::mem::offset_of!(GpuRtSurfacePixel, history_confidence),
        88
    );
    assert_eq!(std::mem::offset_of!(GpuRtSurfacePixel, hit_kind), 92);
    assert_eq!(std::mem::offset_of!(GpuRtSurfacePixel, brick_id), 96);
    assert_eq!(std::mem::offset_of!(GpuRtSurfacePixel, local), 100);
}
```

In `rt_history_common_shader_declares_matching_abi`, add the token:

```rust
"float4 view_direction_background",
```

- [ ] **Step 2: Write failing RT surface shader source test**

In `src/render/passes/rt_surface.rs`, add a test in `shader_source_tests`:

```rust
#[test]
fn rt_surface_preserves_primary_ray_direction_for_background_misses() {
    let common = std::fs::read_to_string("assets/shaders/shared/rt_surface_common.slang")
        .expect("rt_surface common shader should be readable");
    let raygen = std::fs::read_to_string("assets/shaders/passes/rt_surface.rgen.slang")
        .expect("rt_surface raygen shader should be readable");
    let miss = std::fs::read_to_string("assets/shaders/passes/rt_surface.rmiss.slang")
        .expect("rt_surface miss shader should be readable");
    let closest_hit = std::fs::read_to_string("assets/shaders/passes/rt_surface.rchit.slang")
        .expect("rt_surface closest-hit shader should be readable");
    let compact_common = crate::render::source_checks::compact(&common);
    let compact_raygen = crate::render::source_checks::compact(&raygen);
    let compact_miss = crate::render::source_checks::compact(&miss);
    let compact_closest_hit = crate::render::source_checks::compact(&closest_hit);

    for token in [
        "float3ray_direction;",
        "pixel.view_direction_background=float4(normalize(payload.ray_direction),payload.hit_kind==RT_SURFACE_HIT_KIND_MISS?1.0:0.0);",
    ] {
        assert!(
            compact_common.contains(token),
            "RT surface payload/pixel must carry explicit background direction; missing {token}"
        );
    }

    assert!(
        compact_raygen.contains("RtSurfacePayloadpayload=make_rt_surface_payload(primary_ray.direction);"),
        "RT surface raygen must seed payload with primary camera ray direction"
    );
    assert!(
        compact_miss.contains("payload.normal=-payload.ray_direction;"),
        "true RT misses should keep a normal suitable for debug views"
    );
    assert!(
        compact_closest_hit.contains("payload.ray_direction=world_direction;"),
        "closest-hit local miss path must restore the original world ray direction for background shading"
    );
    assert!(
        !compact_closest_hit.contains("payload.normal=normalize(attributes.object_normal);"),
        "brick-local misses must not encode AABB normals as the background direction contract"
    );
}
```

- [ ] **Step 3: Run RED tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::rt_history::tests::rt_surface_pixel_layout_is_stable render::rt_history::tests::rt_history_common_shader_declares_matching_abi render::passes::rt_surface::shader_source_tests::rt_surface_preserves_primary_ray_direction_for_background_misses --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: FAIL because `GpuRtSurfacePixel.view_direction_background` does not exist and the Slang struct/source does not declare or write it.

- [ ] **Step 4: Implement the ABI and shader contract**

In `src/render/rt_history.rs`, insert this field after `motion_history`:

```rust
pub view_direction_background: [f32; 4],
```

Then add this assertion to `rt_surface_pixel_layout_is_stable`:

```rust
    assert_eq!(
        std::mem::offset_of!(GpuRtSurfacePixel, view_direction_background),
        64
    );
```

In `assets/shaders/shared/rt_history_common.slang`, insert this field after `motion_history`:

```slang
    // xyz is the primary/view ray direction. w is 1.0 for miss/background pixels.
    float4 view_direction_background;
```

In `assets/shaders/shared/rt_surface_common.slang`, keep `payload.ray_direction = ray_direction;` in `make_rt_surface_payload`, and write the new pixel field in `surface_pixel_from_payload`:

```slang
    pixel.view_direction_background = float4(
        normalize(payload.ray_direction),
        payload.hit_kind == RT_SURFACE_HIT_KIND_MISS ? 1.0 : 0.0
    );
```

In `assets/shaders/passes/rt_surface.rchit.slang`, make the local miss block preserve the world ray direction and use `-world_direction` as the miss debug normal:

```slang
    if (!hit.hit) {
        payload.hit_kind = RT_SURFACE_HIT_KIND_MISS;
        payload.hit_t = hit_t;
        payload.position = float3(0.0);
        payload.normal = -world_direction;
        payload.albedo = float3(0.0);
        payload.roughness = 1.0;
        payload.material_id = 0u;
        payload.brick_id = 0xffffffffu;
        payload.local = uint3(0u);
        payload.ray_direction = world_direction;
        return;
    }
```

- [ ] **Step 5: Run GREEN tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::rt_history::tests::rt_surface_pixel_layout_is_stable render::rt_history::tests::rt_history_common_shader_declares_matching_abi render::passes::rt_surface::shader_source_tests::rt_surface_preserves_primary_ray_direction_for_background_misses --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS for the three targeted tests.

- [ ] **Step 6: Commit**

Stage only Task 1 files:

```powershell
git add src/render/rt_history.rs assets/shaders/shared/rt_history_common.slang assets/shaders/shared/rt_surface_common.slang assets/shaders/passes/rt_surface.rgen.slang assets/shaders/passes/rt_surface.rmiss.slang assets/shaders/passes/rt_surface.rchit.slang src/render/passes/rt_surface.rs
git commit -m "feat: make RT background direction explicit"
```

## Task 2: Direct-Lighting Background Uses Explicit Direction

**Files:**
- Modify: `assets/shaders/passes/rt_direct_lighting.rgen.slang`
- Modify: `src/render/passes/rt_direct_lighting.rs`

- [ ] **Step 1: Write failing direct-lighting source test**

Replace the background-direction tokens in `rt_direct_lighting_primary_miss_uses_sky_ground_background` in `src/render/passes/rt_direct_lighting.rs` with:

```rust
for token in [
    "float3rt_direct_background_color(RtSurfacePixelsurface)",
    "float3miss_dir=normalize(surface.view_direction_background.xyz);",
    "floatground_ndotl=max(rt_direct.sun_direction_pad.y,0.0);",
    "float3finite_sun_irradiance=rt_direct.sun_intensity_pad.rgb*ground_ndotl*rt_direct_sun_disk_solid_angle();",
    "returnlerp(sunlit_ground,rt_direct.sky_color_sun_angular_radius.rgb,t);",
    "current_radiance[launch_id.xy]=float4(rt_direct_background_color(surface),1.0);",
] {
    assert!(
        compact.contains(token),
        "RT direct-lighting primary miss must resolve to scene background; missing {token}"
    );
}
assert!(
    !compact.contains("float3miss_dir=normalize(-surface.normal_roughness.xyz);"),
    "RT direct-lighting background must not infer miss ray direction from geometric normals"
);
assert!(
    !compact.contains("current_radiance[launch_id.xy]=float4(0.0,0.0,0.0,1.0);"),
    "RT direct-lighting must not turn primary miss pixels into a black bar"
);
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::passes::rt_direct_lighting::shader_source_tests::rt_direct_lighting_primary_miss_uses_sky_ground_background --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: FAIL because the shader currently uses `normalize(-surface.normal_roughness.xyz)`.

- [ ] **Step 3: Update direct-lighting shader**

In `assets/shaders/passes/rt_direct_lighting.rgen.slang`, change:

```slang
    float3 miss_dir = normalize(-surface.normal_roughness.xyz);
```

to:

```slang
    float3 miss_dir = normalize(surface.view_direction_background.xyz);
```

- [ ] **Step 4: Run GREEN test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::passes::rt_direct_lighting::shader_source_tests::rt_direct_lighting_primary_miss_uses_sky_ground_background --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS for the targeted test.

- [ ] **Step 5: Format the touched Rust file**

Run:

```powershell
cargo fmt -- src/render/passes/rt_direct_lighting.rs
```

Expected: exit code 0 and no rustfmt error.

- [ ] **Step 6: Commit**

```powershell
git add assets/shaders/passes/rt_direct_lighting.rgen.slang src/render/passes/rt_direct_lighting.rs
git commit -m "fix: use explicit RT background direction"
```

## Task 3: Unified RT Swapchain Overlay And Present Helper

**Files:**
- Modify: `src/render/rt_pipeline.rs`

- [ ] **Step 1: Write failing RT pipeline source contract**

Update `rt_pipeline_records_egui_overlay_after_swapchain_blit` or add a new test beside it requiring a unified helper. The new test should include:

```rust
#[test]
fn rt_pipeline_records_egui_overlay_after_blit_and_clear_fallback() {
    let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
    let implementation = source
        .split("#[cfg(test)]")
        .next()
        .expect("RT pipeline implementation should precede tests");
    let compact = crate::render::source_checks::compact(implementation);

    for token in [
        "fnadd_egui_overlay_present_pass<'a>(",
        "builder.write_as(swapchain_after_write,AccessKind::ColorAttachmentWrite",
        "builder.finish_as(swapchain_after_write,AccessKind::Present);",
        "fnadd_swapchain_clear_present_pass<'a>(",
        "has_egui_overlay:bool",
        "if!has_egui_overlay{builder.finish_as(swapchain,AccessKind::Present);}",
        "Ok(clear_writes[0])",
        "add_egui_overlay_present_pass(&mutgraph,renderer,frame,egui_renderer,egui_frame,swapchain_after_clear);",
    ] {
        assert!(
            compact.contains(token),
            "RT fallback presentation must keep egui overlay before present; missing {token}"
        );
    }

    assert!(
        !compact.contains("add_swapchain_clear_present_pass(&mutgraph,frame)?"),
        "RT fallback clear calls must pass overlay state and return the swapchain write handle"
    );
}
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::rt_pipeline::tests::rt_pipeline_records_egui_overlay_after_blit_and_clear_fallback --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: FAIL because there is no shared egui overlay/present helper and clear fallback still presents directly.

- [ ] **Step 3: Extract overlay present helper**

In `src/render/rt_pipeline.rs`, add a helper near the swapchain helpers:

```rust
#[cfg(not(target_os = "android"))]
fn add_egui_overlay_present_pass<'a>(
    graph: &mut RenderGraph<'a>,
    renderer: &'a RenderDevice,
    frame: &'a FrameContext,
    egui_renderer: Option<&'a mut EguiRenderer>,
    egui_frame: Option<&'a EguiFrame>,
    swapchain_after_write: ResourceHandle,
) {
    if let (Some(egui_renderer), Some(egui_frame)) = (egui_renderer, egui_frame) {
        graph.add_pass("egui_overlay", QueueType::Graphics, |builder| {
            builder.write_as(swapchain_after_write, AccessKind::ColorAttachmentWrite);
            builder.finish_as(swapchain_after_write, AccessKind::Present);
            Box::new(move |_ctx| {
                if let Err(error) = egui_renderer.record(renderer, frame, egui_frame) {
                    tracing::error!(%error, "failed to record egui overlay");
                }
            })
        });
    }
}
```

- [ ] **Step 4: Make clear fallback return the swapchain write**

Change `add_swapchain_clear_present_pass` signature and body:

```rust
fn add_swapchain_clear_present_pass<'a>(
    graph: &mut RenderGraph<'a>,
    frame: &'a FrameContext,
    has_egui_overlay: bool,
) -> Result<ResourceHandle> {
    let swapchain = graph.import_image_with_access(
        frame.swapchain_image,
        frame.swapchain_extent.width,
        frame.swapchain_extent.height,
        frame.swapchain_format,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        swapchain_access_from_layout(frame.swapchain_image_layout)?,
    );
    let clear_writes = graph.add_pass("rt_clear_fallback", QueueType::Transfer, |builder| {
        builder.write_as(swapchain, AccessKind::TransferWrite);
        if !has_egui_overlay {
            builder.finish_as(swapchain, AccessKind::Present);
        }
        Box::new(move |ctx| {
            let clear = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            };
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            unsafe {
                ctx.device.cmd_clear_color_image(
                    ctx.command_buffer,
                    frame.swapchain_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &clear,
                    &[range],
                );
            }
        })
    });
    Ok(clear_writes[0])
}
```

- [ ] **Step 5: Use the helper from success and fallback paths**

After successful blit:

```rust
#[cfg(not(target_os = "android"))]
if has_egui_overlay {
    add_egui_overlay_present_pass(
        &mut graph,
        renderer,
        frame,
        egui_renderer,
        egui_frame,
        swapchain_after_blit,
    );
}
```

For each fallback currently calling `add_swapchain_clear_present_pass(&mut graph, frame)?`, replace with:

```rust
let swapchain_after_clear = add_swapchain_clear_present_pass(&mut graph, frame, has_egui_overlay)?;
#[cfg(not(target_os = "android"))]
if has_egui_overlay {
    add_egui_overlay_present_pass(
        &mut graph,
        renderer,
        frame,
        egui_renderer,
        egui_frame,
        swapchain_after_clear,
    );
}
```

- [ ] **Step 6: Run GREEN tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::rt_pipeline::tests::rt_pipeline_records_egui_overlay_after_blit_and_clear_fallback render::rt_pipeline::tests::rt_pipeline_records_egui_overlay_after_swapchain_blit render::rt_pipeline::tests::rt_pipeline_skips_surface_trace_without_built_tlas_and_aabbs --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS for the three targeted tests.

- [ ] **Step 7: Commit**

```powershell
git add src/render/rt_pipeline.rs
git commit -m "fix: preserve egui overlay on RT fallback"
```

## Task 4: RT Final Visual Baseline

**Files:**
- Modify: `run/visual-baselines.json`
- Modify: `src/render/source_checks.rs`

- [ ] **Step 1: Write failing manifest source check**

In `visual_baseline_manifest_covers_svgf_and_reblur_debug_cases`, add these tokens to the manifest-level check:

```rust
"\"name\": \"rt_final\"",
"\"rtDebugView\": \"off\"",
```

Add a second RT case check:

```rust
let rt_final_case = visual_baseline_case(&manifest, "rt_final");
assert_contains_all(
    rt_final_case,
    &[
        "\"renderMode\": \"rt\"",
        "\"requiresRt\": true",
        "\"expectedRenderBackend\": \"rt\"",
        "\"expectedMinNonZeroPixelRatio\": 0.25",
        "\"expectedMinRgbRange\": 32",
        "\"rtDebugView\": \"off\"",
        "\"rtRestirDi\": true",
        "\"rtRestirDiSpatial\": true",
        "\"rtRestirDiSpatialSamples\": 4",
        "\"rtRestirGi\": true",
        "\"rtTemporalDenoise\": true",
        "\"expectedRtFrameRendered\": true",
        "\"expectedRtRestirDiRendered\": true",
        "\"expectedRtRestirGiRendered\": true",
        "\"expectedRtResolveReady\": true",
    ],
    "rt_final visual baseline case",
);
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::source_checks::tests::visual_baseline_manifest_covers_svgf_and_reblur_debug_cases --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: FAIL because `rt_final` is missing from `run/visual-baselines.json`.

- [ ] **Step 3: Add `rt_final` manifest case**

In `run/visual-baselines.json`, add this case after `rt_surface_debug`:

```json
    {
      "name": "rt_final",
      "renderMode": "rt",
      "denoiser": "svgf",
      "debugView": "final",
      "requiresNrd": false,
      "requiresRt": true,
      "expectedRenderBackend": "rt",
      "expectedEffectiveDenoiser": "svgf",
      "expectedMinNonZeroPixelRatio": 0.25,
      "expectedMinRgbRange": 32,
      "rtDebugView": "off",
      "rtRestirDi": true,
      "rtRestirDiSpatial": true,
      "rtRestirDiSpatialSamples": 4,
      "rtRestirGi": true,
      "rtTemporalDenoise": true,
      "expectedRtFrameRendered": true,
      "expectedRtRestirDiRendered": true,
      "expectedRtRestirGiRendered": true,
      "expectedRtResolveReady": true
    }
```

- [ ] **Step 4: Run GREEN test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::source_checks::tests::visual_baseline_manifest_covers_svgf_and_reblur_debug_cases --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS for the manifest source check.

- [ ] **Step 5: Commit**

```powershell
git add run/visual-baselines.json src/render/source_checks.rs
git commit -m "test: add RT final visual baseline"
```

## Task 5: Full Verification And Cleanup

**Files:**
- Inspect all files modified by Tasks 1-4.
- Do not stage or revert unrelated `Revolumetric.iml`, `.claudian/`, or `.obsidian/`.

- [ ] **Step 1: Format**

Run:

```powershell
cargo fmt --check
```

Expected: PASS.

- [ ] **Step 2: Run library tests with shader compilation skipped**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS with all library tests passing.

- [ ] **Step 3: Run library tests with strict shader compilation**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS with all library tests passing.

- [ ] **Step 4: Run clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS with no warnings promoted to errors.

- [ ] **Step 5: Build desktop binary with strict shader compilation**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo build --features desktop --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: PASS. A Cargo PDB filename collision warning is acceptable if the build exits 0.

- [ ] **Step 6: Run RT visual baseline**

Run:

```powershell
.\run\validate-visual-baseline.ps1 -Rt
```

Expected: PASS on RT-capable hardware. The output must include both `rt_surface_debug` and `rt_final`, with `render_backend = rt`, active RT frame/pass metadata, resolve readiness, and PPM signal checks passing for both RT cases.

- [ ] **Step 7: Check whitespace and final diff**

Run:

```powershell
git diff --check
git status --short
git diff --stat
```

Expected: `git diff --check` exits 0. `git status --short` shows only intentional RT files plus unrelated pre-existing user files if they were not committed.

- [ ] **Step 8: Final implementation commit if needed**

If Tasks 1-4 were not committed separately, stage only intentional RT stabilization files and commit:

```powershell
git add src/render/rt_history.rs assets/shaders/shared/rt_history_common.slang assets/shaders/shared/rt_surface_common.slang assets/shaders/passes/rt_surface.rgen.slang assets/shaders/passes/rt_surface.rmiss.slang assets/shaders/passes/rt_surface.rchit.slang src/render/passes/rt_surface.rs assets/shaders/passes/rt_direct_lighting.rgen.slang src/render/passes/rt_direct_lighting.rs src/render/rt_pipeline.rs run/visual-baselines.json src/render/source_checks.rs
git commit -m "feat: stabilize RT presentation and background output"
```

## Self-Review

- Spec coverage: Task 1 implements explicit RT background direction and ABI tests. Task 2 makes final direct lighting consume it. Task 3 covers overlay after success and fallback presentation. Task 4 adds final RT baseline while preserving `rt_surface_debug`. Task 5 covers required validation and diff review.
- Placeholder scan: no deferred-work markers or unspecified test-writing steps remain.
- Type consistency: the plan uses `view_direction_background` consistently across Rust ABI, Slang ABI, surface conversion, and direct-lighting consumption. The swapchain helper consistently uses `swapchain_after_write`, `swapchain_after_blit`, and `swapchain_after_clear` handles.
