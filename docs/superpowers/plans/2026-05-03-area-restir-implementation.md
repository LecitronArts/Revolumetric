# Area ReSTIR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Area ReSTIR as a first-class VPT sample-area subsystem, with 4D film/lens sample state, temporal and spatial reuse, debug views, and safe integration into the existing VPT-only graph.

**Architecture:** Area ReSTIR is separate from ReSTIR-DI and VPT temporal denoising. ReSTIR-DI chooses direct-light samples; Area ReSTIR chooses primary-ray sample-area state and evaluates whether subpixel/lens/path samples can be reused; VPT temporal denoise accumulates radiance after tracing. The final graph target is `vpt_surface -> restir_di_* -> area_restir_initial -> area_restir_temporal -> area_restir_spatial -> vpt -> vpt_temporal -> postprocess`.

**Tech Stack:** Rust 2024, Vulkan through `ash`, existing `RenderGraph`, Slang compute shaders, UCVH traversal buffers, `bytemuck` ABI checks, source tests, strict shader compilation.

---

## Research Basis And Constraints

Area ReSTIR references already inspected:

- Paper/project page: <https://dqlin.xyz/pubs/2024-sig-AREA/>
- NVIDIA page: <https://research.nvidia.com/labs/rtr/publication/zhang2024area/>
- Public implementation: <https://github.com/guiqi134/Area-ReSTIR>
- Local reference clone: `target/research/Area-ReSTIR`
- Existing local design note: `docs/superpowers/specs/2026-05-02-area-restir-reference-design.md`

Useful local reference files:

- `target/research/Area-ReSTIR/README.md`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/AreaReSTIR.h`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/Reservoir.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/PixelAreaSampleData.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/Resampling.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/TemporalResampling_FloatMotion.cs.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/SpatialResampling.cs.slang`
- `target/research/Area-ReSTIR/Source/RenderPasses/PathTracer/PathTracer.slang`

Do not copy source from the NVIDIA/Falcor implementation. Use it only for architecture, terminology, resource boundaries, validation ideas, and algorithmic shape. The reference headers include restrictive NVIDIA notices.

Hard constraints for this repo:

- Do not call emissive-area ReSTIR-DI "Area ReSTIR". That work is direct-light sampling, not sample-area reuse.
- Do not merge Area ReSTIR fields into `GpuRestirDiReservoir`.
- Do not fake full Area ReSTIR by storing only `subpixel_uv`; full implementation needs film area, lens area, previous sample state, validation, and debug views.
- Do not require DOF to be visually enabled by default. Thin-lens ABI and lens reuse must exist, but default aperture can stay zero so current pinhole behavior remains stable.
- Do not add pass-local barriers; RenderGraph owns all image/buffer transitions.
- Do not route Area ReSTIR around VPT temporal denoise. It feeds VPT ray generation; VPT temporal still accumulates radiance.

## Current Repo Facts

- VPT is the only active renderer in the dirty tree.
- `src/render/passes/vpt_surface.rs` and `assets/shaders/passes/vpt_surface.slang` write current and previous surface state plus motion history.
- `src/render/vpt_history.rs` and `assets/shaders/shared/vpt_history_common.slang` store current/previous view-projection matrices, resolution, jitter, frame index, reset generation, and flags.
- `src/scene/camera.rs` has pinhole camera fields only: `position`, `forward`, `up`, `fov_y_radians`.
- `assets/shaders/passes/vpt.slang` currently generates random subpixel jitter internally at trace time.
- `src/app.rs` graph order already has `vpt_surface`, optional ReSTIR-DI passes, `vpt`, `vpt_temporal`, history update, `postprocess`, and blit.
- No current `aperture_radius`, `focal_distance`, or `lens_uv` camera ABI exists.

## File Structure

- Modify `src/scene/camera.rs`: add thin-lens camera parameters and tests.
- Modify `src/render/camera.rs`: add camera basis and thin-lens ray data helpers if needed.
- Modify `src/render/scene_ubo.rs`: add camera basis / lens fields to `GpuSceneUniforms` and tests.
- Modify `assets/shaders/shared/scene_common.slang`: mirror scene ABI and add helper functions for sample-area primary rays.
- Create `src/render/area_restir.rs`: settings, parse warnings, GPU uniform ABI, reservoir ABI, sample/eval-context ABI, and source tests.
- Modify `src/render/mod.rs`: export `area_restir`.
- Create `assets/shaders/shared/area_restir_common.slang`: ABI mirror and helper functions.
- Create `src/render/passes/area_restir.rs`: pass/resource owner for initial, temporal, spatial, history update, and debug resources.
- Modify `src/render/passes/mod.rs`: export `area_restir`.
- Create `assets/shaders/passes/area_restir_initial.slang`.
- Create `assets/shaders/passes/area_restir_temporal.slang`.
- Create `assets/shaders/passes/area_restir_spatial.slang`.
- Modify `src/render/passes/vpt.rs` and `assets/shaders/passes/vpt.slang`: bind Area ReSTIR selected sample state and use it for primary ray generation.
- Modify `src/app.rs`: parse settings, initialize/resize/drop pass, wire graph dependencies, update VPT descriptors, reset history.
- Modify `src/render/gpu_profiler.rs`: add Area ReSTIR stages.
- Modify README/docs after behavior exists.

## Task 1: Camera And Scene ABI For 4D Sample Area

**Files:**
- Modify: `src/scene/camera.rs`
- Modify: `src/render/scene_ubo.rs`
- Modify: `assets/shaders/shared/scene_common.slang`
- Modify: `src/render/passes/vpt.rs`

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```rust
#[test]
fn camera_defaults_keep_pinhole_lens_disabled() {
    let cam = Camera::default();
    assert_eq!(cam.aperture_radius, 0.0);
    assert!(cam.focal_distance > 0.0);
}

#[test]
fn gpu_scene_uniforms_expose_camera_basis_and_lens_fields() {
    assert!(std::mem::size_of::<GpuSceneUniforms>() >= 224);
    assert_eq!(std::mem::offset_of!(GpuSceneUniforms, camera_right), 176);
    assert_eq!(std::mem::offset_of!(GpuSceneUniforms, camera_up), 192);
    assert_eq!(std::mem::offset_of!(GpuSceneUniforms, aperture_radius), 208);
    assert_eq!(std::mem::offset_of!(GpuSceneUniforms, focal_distance), 212);
}

#[test]
fn scene_common_declares_area_restir_camera_fields() {
    let source = std::fs::read_to_string("assets/shaders/shared/scene_common.slang").unwrap();
    assert!(source.contains("float3 camera_right"));
    assert!(source.contains("float3 camera_up"));
    assert!(source.contains("float aperture_radius"));
    assert!(source.contains("float focal_distance"));
    assert!(source.contains("sample_disk_for_lens"));
}
```

Expected RED: fields and helpers are missing.

- [ ] **Step 2: Implement camera fields**

Add to `Camera`:

```rust
pub aperture_radius: f32,
pub focal_distance: f32,
```

Default:

```rust
aperture_radius: 0.0,
focal_distance: 128.0,
```

Keep fly camera controls unchanged.

- [ ] **Step 3: Extend scene UBO**

Append fields to `GpuSceneUniforms` after `vpt_debug_view`:

```rust
pub camera_right: [f32; 3],
pub aperture_radius: f32,
pub camera_up: [f32; 3],
pub focal_distance: f32,
pub camera_forward: [f32; 3],
pub _pad4: f32,
```

Update `SceneUniformInputs` with camera basis and lens values. Compute `camera_right`, normalized `camera_up`, and normalized `camera_forward` in `src/app.rs` from the active `CameraRig`.

- [ ] **Step 4: Mirror Slang ABI**

Add the same fields to `SceneUniforms`. Add helpers:

```hlsl
float2 sample_disk_for_lens(float2 uv)
float3 scene_primary_ray_dir_from_sample(SceneUniforms scene, float2 pixel_sample)
Ray scene_primary_ray_from_area_sample(SceneUniforms scene, float2 pixel_sample, float2 lens_uv)
```

For `aperture_radius <= 0`, origin must remain `scene.pixel_to_ray[3].xyz` and the ray must match existing pinhole behavior.

- [ ] **Step 5: Run targeted tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib scene::camera render::scene_ubo render::passes::vpt::shader_source_tests::vpt_history_abi_declares_surface_and_reprojection_contract; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 2: Area ReSTIR Settings And ABI

**Files:**
- Create: `src/render/area_restir.rs`
- Modify: `src/render/mod.rs`
- Create: `assets/shaders/shared/area_restir_common.slang`

- [ ] **Step 1: Write failing ABI/settings tests**

Create tests for:

- Default disabled: `enabled=false`.
- Temporal/spatial/subpixel/lens enabled defaults for an enabled configuration.
- Env parsing:
  - `REVOLUMETRIC_AREA_RESTIR`
  - `REVOLUMETRIC_AREA_RESTIR_TEMPORAL`
  - `REVOLUMETRIC_AREA_RESTIR_SPATIAL`
  - `REVOLUMETRIC_AREA_RESTIR_SUBPIXEL`
  - `REVOLUMETRIC_AREA_RESTIR_LENS`
  - `REVOLUMETRIC_AREA_RESTIR_INITIAL_CANDIDATES`
  - `REVOLUMETRIC_AREA_RESTIR_SPATIAL_SAMPLES`
  - `REVOLUMETRIC_AREA_RESTIR_HISTORY_LENGTH`
  - `REVOLUMETRIC_AREA_RESTIR_DEBUG`
- Invalid values warn and keep defaults.
- ABI sizes and offsets for:
  - `GpuAreaRestirUniforms`
  - `GpuAreaRestirReservoir`
  - `GpuAreaRestirSampleState`
  - `GpuAreaRestirEvalContext`
- Source test proving `area_restir_common.slang` does not include `restir_di_common.slang`.

Expected RED: module and shader common file missing.

- [ ] **Step 2: Implement Rust ABI**

Required structs:

```rust
#[repr(C)]
pub struct GpuAreaRestirUniforms {
    pub enabled: u32,
    pub temporal_enabled: u32,
    pub spatial_enabled: u32,
    pub subpixel_enabled: u32,
    pub lens_enabled: u32,
    pub initial_candidate_count: u32,
    pub spatial_sample_count: u32,
    pub history_length: u32,
    pub frame_index: u32,
    pub reservoir_count: u32,
    pub width: u32,
    pub height: u32,
    pub normal_threshold_bits: f32,
    pub depth_threshold: f32,
    pub spatial_radius: f32,
    pub debug_view: u32,
}

#[repr(C)]
pub struct GpuAreaRestirSampleState {
    pub subpixel_uv: [f32; 2],
    pub lens_uv: [f32; 2],
    pub pixel_sample: [f32; 2],
    pub path_sample: u32,
    pub flags: u32,
}

#[repr(C)]
pub struct GpuAreaRestirEvalContext {
    pub position_depth: [f32; 4],
    pub normal_roughness: [f32; 4],
    pub albedo_material: [f32; 4],
    pub motion_history: [f32; 4],
}

#[repr(C)]
pub struct GpuAreaRestirReservoir {
    pub sample_state: GpuAreaRestirSampleState,
    pub sample_count_m: u32,
    pub pad0: u32,
    pub weight_sum: f32,
    pub target_pdf: f32,
    pub selected_weight: f32,
    pub confidence: f32,
    pub jacobian: f32,
    pub contribution_luma: f32,
    pub selected_radiance: [f32; 4],
}
```

All GPU structs must derive `Pod` and `Zeroable`.

- [ ] **Step 3: Implement Slang ABI mirror**

`area_restir_common.slang` must define matching structs and helpers:

- `AREA_RESTIR_DEBUG_*` constants.
- `area_restir_is_valid_reservoir`.
- `area_restir_invalid_reservoir`.
- `area_restir_sample_pixel_state`.
- `area_restir_surface_compatible`.
- `area_restir_finalize_reservoir`.

Do not include or reference `RestirDiReservoir`.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::area_restir; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 3: Area ReSTIR Pass Owner And Resource Lifecycle

**Files:**
- Create: `src/render/passes/area_restir.rs`
- Modify: `src/render/passes/mod.rs`
- Modify: `src/render/gpu_profiler.rs`

- [ ] **Step 1: Write failing source/lifecycle tests**

Tests must assert:

- `AreaRestirPass` owns initial, temporal, spatial, and history buffers.
- Hot fullscreen buffers are `MemoryLocation::GpuOnly`.
- Descriptor layouts bind:
  - uniforms
  - current reservoirs
  - previous/history reservoirs
  - current VPT surface images
  - previous VPT surface images
  - motion history
  - debug output
- No `cmd_pipeline_barrier`, `ImageMemoryBarrier`, or `BufferMemoryBarrier` in the pass.
- `destroy()` cleans every buffer/image/stage.
- `resize_buffers()` recreates resolution-dependent state and resets history.

Expected RED: pass does not exist.

- [ ] **Step 2: Implement pass owner**

Create buffers:

- `uniform_buffers`: per frame slot, CPU-to-GPU.
- `initial_reservoirs`: GPU-only storage buffer.
- `temporal_reservoirs`: GPU-only storage buffer.
- `spatial_reservoirs`: GPU-only storage buffer.
- `history_reservoirs`: GPU-only storage + transfer src/dst.
- `debug_image`: `rgba16f` or `rgba32f` storage image.

Use the same cleanup style as `RestirDiPass` and `VptTemporalPass`.

- [ ] **Step 3: Add profiler scopes**

Add:

- `AreaRestirInitial`
- `AreaRestirTemporal`
- `AreaRestirSpatial`

Update `COUNT`, `ALL`, `log_name`, and `csv_column` tests.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::area_restir render::gpu_profiler; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 4: Initial Candidate Generation Shader

**Files:**
- Create: `assets/shaders/passes/area_restir_initial.slang`
- Modify: `src/render/passes/area_restir.rs`

- [ ] **Step 1: Write failing shader source tests**

Assert the shader:

- Includes `area_restir_common.slang`, `scene_common.slang`, and `vpt_history_common.slang`.
- Reads current `vpt_surface_*`.
- Writes `RWStructuredBuffer<AreaRestirReservoir> output_reservoirs`.
- Generates `subpixel_uv` and `lens_uv`.
- Uses `scene_primary_ray_from_area_sample`.
- Writes invalid reservoirs for miss pixels.
- Does not reference `RestirDiReservoir`.

Expected RED.

- [ ] **Step 2: Implement initial shader**

For each valid pixel:

- Generate `initial_candidate_count` sample-area candidates from deterministic per-frame RNG.
- Generate film sample in `[0, 1)^2`; if subpixel reuse disabled, use `(0.5, 0.5)`.
- Generate lens sample in `[0, 1)^2`; if lens reuse disabled or aperture is zero, use `(0.5, 0.5)`.
- Compute `pixel_sample = pixel + subpixel_uv`.
- Build a reservoir selecting one candidate by a simple, finite target: valid surface and finite luminance proxy from current material/emissive/sun visibility.
- Store sample state, target PDF, weight sum, selected weight, confidence, and eval context compatibility data.

This milestone does not need full reconnection MIS yet, but it must store all data required for it.

- [ ] **Step 3: Run shader source test and strict compile**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib render::passes::area_restir::shader_source_tests::area_restir_initial_declares_full_sample_area_contract; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 5: Temporal Reuse Shader

**Files:**
- Create: `assets/shaders/passes/area_restir_temporal.slang`
- Modify: `src/render/passes/area_restir.rs`

- [ ] **Step 1: Write failing tests**

Tests must assert:

- Temporal shader reads current and history reservoirs.
- Reads current and previous VPT surface images.
- Reads `motion_history`.
- Rejects camera cut / resize / scene invalidation.
- Rejects incompatible history by miss state, normal dot, depth/position, and material id.
- Caps `sample_count_m` by `history_length`.
- Recomputes target PDF in current pixel measure.
- Mentions random replay and reconnection placeholders as explicit code paths, not comments only.

Expected RED.

- [ ] **Step 2: Implement temporal shader**

Implement conservative Area ReSTIR temporal reuse:

- Start from initial current reservoir.
- Use `motion_history.xy` to locate previous pixel.
- Bilinear/fractional motion can start with nearest valid previous pixel, but shader must retain fractional coordinates for later reconnection.
- Validate current and previous surface compatibility.
- Re-evaluate candidate in current pixel sample domain using stored subpixel/lens state.
- Combine current and previous reservoirs with finite weights.
- Clamp `M` and confidence to `history_length`.
- Set debug flags/rejection reason in reservoir flags or debug image.

- [ ] **Step 3: Run targeted tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib render::passes::area_restir::shader_source_tests::area_restir_temporal_reuses_history_with_surface_rejection; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 6: Spatial Reuse Shader

**Files:**
- Create: `assets/shaders/passes/area_restir_spatial.slang`
- Modify: `src/render/passes/area_restir.rs`

- [ ] **Step 1: Write failing tests**

Tests must assert:

- Spatial shader reads temporal reservoirs and writes spatial reservoirs.
- Samples multiple independent neighbor offsets, not one hard-coded neighbor.
- Uses `area_restir_surface_compatible`.
- Re-evaluates reused samples against current pixel.
- Does not add raw neighbor `weight_sum` without target PDF conversion.
- Uses `spatial_radius` and `spatial_sample_count`.

Expected RED.

- [ ] **Step 2: Implement spatial shader**

For each pixel:

- Load center temporal reservoir.
- Pick `spatial_sample_count` neighbors from a deterministic rotation/hash.
- Reject outside viewport, miss, incompatible normal/depth/material.
- Evaluate neighbor sample in current pixel domain.
- Combine using reservoir update rules and finite weights.
- Write final spatial reservoir.

- [ ] **Step 3: Run targeted tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib render::passes::area_restir::shader_source_tests::area_restir_spatial_reuses_multiple_compatible_neighbors; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 7: VPT Integration

**Files:**
- Modify: `src/render/passes/vpt.rs`
- Modify: `assets/shaders/passes/vpt.slang`
- Modify: `src/app.rs`

- [ ] **Step 1: Write failing tests**

Tests must assert:

- VPT descriptor set binds Area ReSTIR uniforms and selected reservoir buffer separately from ReSTIR-DI.
- Disabled fallback resources exist.
- VPT shader includes `area_restir_common.slang`.
- Primary ray generation reads selected `AreaRestirReservoir`.
- Internal random jitter is bypassed when Area ReSTIR is enabled and reservoir is valid.
- VPT still compiles and behaves safely when Area ReSTIR is disabled.

Expected RED.

- [ ] **Step 2: Implement VPT descriptor fallback**

Add disabled `GpuAreaRestirUniforms` and one invalid `GpuAreaRestirReservoir` fallback buffer to `VptPass`.

Add:

```rust
pub fn update_area_restir_descriptors(
    &self,
    device: &ash::Device,
    frame_slot: usize,
    uniforms: &GpuBuffer,
    reservoirs: &GpuBuffer,
)
```

- [ ] **Step 3: Update VPT shader**

Primary ray path:

- If Area ReSTIR disabled or invalid, keep existing stochastic jitter.
- If enabled and valid, generate ray using reservoir `pixel_sample`, `subpixel_uv`, and `lens_uv`.
- Use `scene_primary_ray_from_area_sample`.
- Keep direct/indirect/debug outputs intact.

- [ ] **Step 4: Wire app graph**

When `REVOLUMETRIC_AREA_RESTIR=on`:

- Run `area_restir_initial` after `vpt_surface`.
- Run `area_restir_temporal` after initial.
- Run `area_restir_spatial` after temporal if spatial enabled.
- Add explicit history update copy after selected reservoir.
- Pass selected reservoir to `vpt`.
- Reset Area ReSTIR history on resize/camera cut/scene invalidation.

When disabled:

- No Area ReSTIR graph passes.
- VPT reads fallback disabled resources.

- [ ] **Step 5: Run targeted tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib render::passes::vpt::shader_source_tests render::passes::area_restir::shader_source_tests; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 8: Debug Views And Documentation

**Files:**
- Modify: `src/render/scene_ubo.rs`
- Modify: `assets/shaders/shared/scene_common.slang`
- Modify: `assets/shaders/passes/vpt_temporal.slang`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-05-02-area-restir-reference-design.md`

- [ ] **Step 1: Add debug view tests**

Debug views:

- `area_subpixel`
- `area_lens`
- `area_weight`
- `area_history_valid`
- `area_rejection`
- `area_jacobian`

Tests must assert parsing and shader routing.

- [ ] **Step 2: Implement debug view routing**

Expose Area ReSTIR debug via either `REVOLUMETRIC_AREA_RESTIR_DEBUG` or VPT debug view extension. The selected debug output must be visible in postprocess path without temporal smoothing.

- [ ] **Step 3: Update docs**

Document:

- Full settings list.
- Pipeline order.
- Default disabled behavior.
- Lens/DOF default aperture behavior.
- Known limitations: reconnection MIS may start conservative, and final visual proof requires capture/manual scene validation.

## Task 9: Verification

Run:

```powershell
cargo fmt --check
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --lib --bin revolumetric -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
Get-Process | Where-Object { $_.ProcessName -like '*revolumetric*' } | Select-Object Id,ProcessName,Path
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo build --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
git diff --check
```

Runtime smoke only counts if it exits:

```powershell
$env:REVOLUMETRIC_AREA_RESTIR='on'; $env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; cargo run; Remove-Item Env:\REVOLUMETRIC_AREA_RESTIR; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
```

If `revolumetric.exe` stays running or the environment has no Vulkan/window support, record the exact limitation and do not claim runtime validation.

## Completion Criteria

- Area ReSTIR has a separate Rust and Slang ABI from ReSTIR-DI.
- Camera and scene ABI support both film/subpixel and lens sample domains.
- VPT can use Area ReSTIR-selected primary-ray samples when enabled.
- Temporal and spatial Area ReSTIR reuse validate surface compatibility.
- History is explicit and reset on resize/camera cut/scene invalidation.
- Strict shader compilation succeeds.
- Source tests prove disabled mode remains safe and enabled mode uses dedicated resources.
