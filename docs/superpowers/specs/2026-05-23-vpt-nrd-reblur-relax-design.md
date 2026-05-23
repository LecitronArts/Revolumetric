# VPT NRD ReBLUR/ReLAX Denoiser Design

## Decision

Replace the current custom SVGF-like temporal plus a-trous denoising path with
an NRD-shaped denoising architecture, then integrate the official NVIDIA NRD
SDK for ReLAX and ReBLUR quality.

This is not a parameter tuning pass for `vpt_temporal` or `vpt_atrous`. The
current denoiser is too light because the renderer does not yet provide the
inputs that ReBLUR/ReLAX need: true roughness, viewZ, NRD-compatible motion,
diffuse/specular radiance split, hit distance, material demodulation, and a
validation/debug path. The implementation must first create this input
contract, then wire official NRD dispatches.

The staged product target is:

```text
vpt_surface
-> optional ReSTIR-DI / Area ReSTIR
-> vpt
-> vpt_nrd_frontend
-> nrd_relax or nrd_reblur
-> vpt_nrd_resolve
-> postprocess
```

The existing path remains as fallback until NRD output is validated:

```text
vpt_surface
-> optional ReSTIR-DI / Area ReSTIR
-> vpt
-> vpt_temporal
-> vpt_atrous
-> postprocess
```

Initial runtime modes:

```text
REVOLUMETRIC_DENOISER=svgf   # current fallback path
REVOLUMETRIC_DENOISER=relax  # first official NRD target
REVOLUMETRIC_DENOISER=reblur # enabled after ReBLUR-specific inputs validate
REVOLUMETRIC_DENOISER=off    # raw VPT/postprocess debug path
```

`svgf` remains available for A/B comparison and as a non-NRD build fallback.

## Current Code Facts

The active VPT chain is owned by:

- `assets/shaders/passes/vpt_surface.slang`
- `assets/shaders/passes/vpt.slang`
- `assets/shaders/passes/vpt_temporal.slang`
- `assets/shaders/passes/vpt_atrous.slang`
- `src/render/vpt_pipeline.rs`
- `src/render/passes/vpt_surface.rs`
- `src/render/passes/vpt.rs`
- `src/render/passes/vpt_temporal.rs`
- `src/render/passes/vpt_atrous.rs`

Current graph shape:

```text
vpt_surface
-> optional restir_di / area_restir
-> vpt
-> vpt_temporal
-> vpt_atrous
-> postprocess
-> blit/capture
```

Current surface outputs:

- `surface_position_depth`: `rgba32f`
  - `xyz`: world position
  - `w`: voxel DDA ray distance `hit.t`, not NRD `viewZ`
- `surface_normal_roughness`: `rgba32f`
  - `xyz`: world normal
  - `w`: emissive luminance today, not roughness
- `surface_albedo_material`: `rgba32f`
  - `rgb`: material albedo
  - `w`: material id
- `motion_history`: `rgba32f`
  - `xy`: previous pixel center minus current pixel center
  - `z`: currently used for `motion_id`
  - `w`: validity/confidence mirror
- `motion_flags`: `r32ui`
- `surface_brick_generation`: `r32ui`
- previous-frame ping-pong images for surface state and brick generation

Current VPT noisy outputs:

- `noisy_radiance_image`: combined path radiance, already material-modulated
- `noisy_moments_image`: luminance moments for the custom temporal/a-trous path

The current renderer does not produce:

- standalone `viewZ`
- true material roughness
- specular lobe parameters or specular radiance
- diffuse/specular NRD radiance hit-distance inputs
- demodulated primary-hit diffuse/specular lighting
- NRD validation output
- official NRD permanent/transient resource pools

The motion guide Phase 4 work already provides useful prerequisites:

- pixel motion delta convention
- motion flags
- UCVH semantic move events
- per-brick generation comparison
- previous/current surface ping-pong

But it still does not satisfy NRD's recommended 2.5D motion because
`motion_history.z` is not `viewZprev - viewZ`.

## Reference Basis

Use these as architecture and API references. Do not copy source code into this
repository without the relevant license being accepted.

- NVIDIA NRD README, research snapshot on 2026-05-23:
  <https://github.com/NVIDIA-RTX/NRD>
  - NRD is a spatio-temporal denoising library focused on 1 path per pixel.
  - ReBLUR is recurrent-blur based.
  - ReLAX is a-trous based and designed around RTXDI-style signals.
  - NRD uses per-pixel guides: normal, roughness, viewZ, motion vector.
  - Diffuse and specular signals must be separated at the primary hit or PSR.
  - Material demodulation is required before denoising and remodulation after
    denoising.
  - ReBLUR expects normalized hit distances produced with NRD frontend helpers.
  - 2.5D motion is recommended: screen-space motion plus `viewZprev - viewZ`.
- NVIDIA NRD `NRDDescs.h`:
  <https://github.com/NVIDIA-RTX/NRD/blob/master/Include/NRDDescs.h>
  - Common inputs include `IN_MV`, `IN_NORMAL_ROUGHNESS`, and `IN_VIEWZ`.
  - ReBLUR/ReLAX diffuse/specular inputs are
    `IN_DIFF_RADIANCE_HITDIST` and `IN_SPEC_RADIANCE_HITDIST`.
  - Outputs are `OUT_DIFF_RADIANCE_HITDIST` and
    `OUT_SPEC_RADIANCE_HITDIST`.
  - Optional inputs include diffuse/specular confidence and disocclusion
    threshold mix.
  - `OUT_VALIDATION` is available when validation is enabled.
- NVIDIA NRD license:
  <https://github.com/NVIDIA-RTX/NRD/blob/master/LICENSE.txt>
  - The SDK is under the NVIDIA RTX SDK license.
  - Vendoring or distributing the SDK requires explicit license acceptance and
    compliance.
- Local RTXDI sample:
  `target/research/RTXDI/Samples/FullSample/Source/RenderPasses/DenoisingPasses/NrdIntegration.cpp`
  - Best local template for Vulkan-style NRD dispatching.
  - It creates NRD pipelines from embedded SPIR-V bytecode, uses the library
    SPIR-V binding offsets, builds shared constant/sampler bindings, maps
    per-dispatch resources, and dispatches the sequence returned by NRD.
- Local Area-ReSTIR sample:
  `target/research/Area-ReSTIR/Source/RenderPasses/NRDPass/NRDPass.cpp`
  and `target/research/Area-ReSTIR/Source/RenderPasses/NRDPass/PackRadiance.cs.slang`
  - Useful for resource naming and ReLAX/ReBLUR toggles.
  - Less useful as direct engineering template because it is Falcor/D3D12
    oriented.

## Approaches Considered

### Approach A: Tune Current SVGF/a-trous

Increase a-trous iterations, widen kernels, tweak luminance clamps, and change
history weights.

Rejected. This can make the image smoother, but it cannot become ReBLUR/ReLAX
quality because the current input signal lacks NRD's required guide/noisy
contract. Heavier filtering would blur voxel edges, smear content edits, and
hide root causes.

### Approach B: Implement a Native ReBLUR/ReLAX-inspired Clone

Keep everything in Slang/Rust and build internal passes that mimic public NRD
papers and samples.

Rejected as the primary path. This avoids NVIDIA SDK build and license work,
but it would not be official ReBLUR/ReLAX. It is large, easy to get subtly
wrong, hard to validate against expected behavior, and would likely reproduce
only a subset of NRD after significant effort. A small internal fallback can
exist, but it must not be marketed as ReBLUR/ReLAX.

### Approach C: Official NRD Integration After Frontend Contract

Recommended. First make the renderer emit NRD-shaped guide and noisy resources,
then add an official NRD adapter behind a feature/dependency boundary.

This is the only path that honestly addresses the user's requirement to replace
the lightweight SVGF denoiser with ReBLUR/ReLAX-quality denoising. It has higher
engineering cost, but the work is concentrated in explicit interfaces:

- guide generation
- noisy signal packing
- NRD adapter
- resolve/remodulation
- validation/debugging

### Approach D: Direct ReBLUR First

Wire ReBLUR immediately and fill missing resources with approximations.

Rejected for the first implementation phase. ReBLUR is more sensitive to hit
distance normalization, roughness, viewZ, and motion correctness. Feeding it
`emissive_luma` as roughness or primary hit distance as the lobe hit distance
would produce misleading results. ReBLUR remains a required target after ReLAX
validates the common NRD path.

## Target Architecture

### Runtime Denoiser Selection

Extend denoiser settings from a boolean to a mode:

```rust
pub enum VptDenoiserMode {
    Off,
    Svgf,
    Relax,
    Reblur,
}
```

Parsing rules:

- `REVOLUMETRIC_DENOISER=off|0|false`: raw VPT output to postprocess.
- `REVOLUMETRIC_DENOISER=svgf|on|1|true`: current temporal/a-trous path.
- `REVOLUMETRIC_DENOISER=relax`: official NRD ReLAX path if compiled and
  available; otherwise warn and fall back to `svgf`.
- `REVOLUMETRIC_DENOISER=reblur`: official NRD ReBLUR path if compiled and
  the ReBLUR readiness checks pass; otherwise warn and fall back to `relax` if
  available, then `svgf`.

The default for builds without NRD support remains `svgf`. The default for
builds with NRD support can move to `relax` only after validation captures and
runtime smoke tests pass.

### Guide Resource Contract

Create a named denoiser guide bundle instead of passing anonymous
`[ResourceHandle; N]` arrays through the graph:

```rust
pub struct VptDenoiserGuideResources {
    pub normal_roughness: ResourceHandle,
    pub view_z: ResourceHandle,
    pub motion: ResourceHandle,
    pub motion_flags: ResourceHandle,
    pub material_id: ResourceHandle,
    pub surface_brick_generation: ResourceHandle,
    pub previous_surface_brick_generation: ResourceHandle,
}
```

Renderer-facing resources:

- `vpt_denoise_normal_roughness`
  - format: start with `R16G16B16A16_SFLOAT` or `R32G32B32A32_SFLOAT`
    while validating; pack to the NRD-selected encoding in `vpt_nrd_frontend`
  - `xyz`: world-space normal
  - `w`: linear roughness
- `vpt_denoise_view_z`
  - format: `R32_SFLOAT` for validation, may be reduced after captures
  - linear view-space depth for the primary visible surface
  - miss pixels use a value greater than `CommonSettings::denoisingRange`
- `vpt_denoise_motion`
  - format: `R16G16B16A16_SFLOAT` or `R32G32B32A32_SFLOAT` during validation
  - `xy`: current pixel center to previous pixel center, in pixel units
  - `z`: `viewZprev - viewZ` for 2.5D motion
  - `w`: validity mirror for debug; official NRD consumes validity through
    depth/motion/confidence rather than relying on this lane
- `vpt_denoise_motion_flags`
  - existing `r32ui` flags
  - used by custom confidence/disocclusion input generation and debug views
- `vpt_denoise_material_id`
  - `r32ui`
  - keeps material identity separate from motion `.z`

NRD-facing conversion:

- `vpt_nrd_frontend` packs `normal_roughness` using the selected NRD normal and
  roughness encoding.
- `CommonSettings.motionVectorScale` maps pixel-space `motion.xy` to the NRD
  expected screen/UV scale. With pixel-space motion, initial values are
  `(1 / width, 1 / height)`.
- `motion.z` remains in viewZ units and must be verified by validation output.
- `CommonSettings` matrices must be non-jittered. Current jitter fields are
  zero; any future jitter must be routed consistently.

### Material Contract

Current material data is diffuse-only and fixed-table based. NRD needs at least
linear roughness immediately, and specular support later.

Add a GPU-visible material parameter table:

```rust
pub struct GpuMaterialParams {
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub specular: [f32; 3],
    pub flags: u32,
}
```

Initial defaults:

- roughness: `1.0` for existing diffuse voxel materials
- metallic: `0.0`
- specular: low dielectric F0 default value, not used for specular denoising
  until specular paths exist
- emissive: existing emissive data

Rules:

- Roughness must replace emissive luminance in the normal/roughness guide.
- Emissive remains separate residual lighting, not a roughness proxy.
- Specular denoising is disabled until the path tracer produces a real
  specular lobe and material factors.

### Noisy Signal Contract

Create NRD-shaped noisy outputs alongside or instead of the current
`noisy_radiance/noisy_moments` when NRD mode is active:

```rust
pub struct VptNrdNoisyResources {
    pub diff_radiance_hitdist: ResourceHandle,
    pub spec_radiance_hitdist: ResourceHandle,
    pub residual_radiance: ResourceHandle,
    pub material_factors: ResourceHandle,
}
```

Required semantics:

- `diff_radiance_hitdist.rgb`
  - demodulated diffuse lighting for the primary visible surface
  - albedo/BRDF factors removed before denoising
- `diff_radiance_hitdist.a`
  - lobe hit distance
  - not the primary hit distance
  - baseline for indirect diffuse: first bounce after the primary hit
  - direct-light distance must be represented separately if direct lighting is
    folded into this signal
- `spec_radiance_hitdist`
  - zero-filled until real specular sampling is implemented
  - must not contain fake data
- `residual_radiance`
  - emissive primary hits, sky, debug views, and other components that should
    bypass NRD
- `material_factors`
  - primary albedo and any BRDF factors required for remodulation

Phase 3 initial target is **demodulated indirect diffuse only**:

- denoise indirect diffuse radiance with `RELAX_DIFFUSE`
- keep analytic sun direct, ReSTIR-DI direct, sky, primary-hit emissive, and
  debug radiance in `residual_radiance`
- do not fold direct lighting into `diff_radiance_hitdist` until direct-light
  hit-distance semantics are defined and tested
- use the first bounce after the primary hit as the baseline indirect diffuse
  lobe hit distance

This choice keeps the first NRD path honest: it avoids assigning a single fake
hit distance to mixed direct/indirect lighting.

`RELAX_DIFFUSE_SPECULAR` is allowed only when specular is represented honestly,
even if the initial specular signal is intentionally zero and tested as such.

For ReBLUR:

- Use NRD frontend helpers for hit distance normalization.
- Pass the same hit distance parameters to NRD settings and the packing shader.
- Do not reuse Area-ReSTIR's local `PackRadiance.cs.slang` blindly. The local
  RTXDI reference shows the safer pattern: normalize hit distance, then pack
  radiance plus normalized hit distance with the ReBLUR-specific helper.

### Confidence And Content Invalidation

NRD has optional confidence and disocclusion-threshold inputs. The project needs
them because UCVH content edits can keep similar depth/normal while changing
radiance.

Add `vpt_nrd_confidence` after guide generation:

- reads current motion flags
- reads current and previous surface brick generation
- reconstructs previous pixel from motion
- emits diffuse/specular confidence textures
- emits disocclusion threshold mix if useful for content edits and
  disocclusions

Rules:

- History confidence is `0` if motion flags reject history.
- History confidence is `0` if previous/current brick generation mismatch.
- History confidence is reduced near disocclusion edges.
- Specular confidence stays `0` while specular signal is disabled.
- If NRD confidence inputs are unavailable in the selected build, any UCVH
  content edit falls back to a one-frame NRD history restart instead of risking
  stale per-pixel reuse.

### NRD Adapter

Add an official NRD adapter rather than reimplementing NRD algorithms.

Build and license boundary:

- Do not vendor NRD SDK files into this repository unless the repository owner
  explicitly accepts the NVIDIA RTX SDK license.
- Initial integration may use `REVOLUMETRIC_NRD_ROOT` to point at an external
  local NRD SDK checkout.
- Gate all SDK-dependent Rust/C++ build steps behind a Cargo feature such as
  `nrd`.
- Builds without `nrd` must keep working and use `svgf` fallback.
- Enabling the `nrd` feature without `REVOLUMETRIC_NRD_ROOT` must fail early
  with a clear message that names the missing environment variable and the
  NVIDIA RTX SDK license requirement.
- CI/default verification does not require the NRD SDK. NRD feature
  verification runs only when `REVOLUMETRIC_NRD_ROOT` is set and points to an
  accepted local SDK checkout.

Rust/C++ boundary:

- Use a small C ABI wrapper around NRD C++ APIs.
- Keep the Rust side free from direct C++ ABI assumptions.
- Expose only the data needed by the renderer:
  - library description
  - instance creation/destruction
  - pool texture descriptions
  - pipeline descriptions and SPIR-V bytecode
  - sampler descriptions
  - dispatch descriptions
  - settings upload blocks

Vulkan/ash responsibilities:

- map NRD formats to `vk::Format`
- create permanent and transient pool textures
- create samplers
- create compute pipelines from NRD SPIR-V bytecode
- create shared constant/sampler descriptor set layout
- create per-pipeline resource descriptor set layouts
- upload constant data per dispatch
- bind resources by `nrd::ResourceType`
- record all NRD dispatches inside RenderGraph-owned passes
- recreate the NRD instance and resources on resize

`RELAX_DIFFUSE` initial resource mapping:

```text
IN_MV                       -> vpt_nrd_motion
IN_NORMAL_ROUGHNESS         -> vpt_nrd_normal_roughness
IN_VIEWZ                    -> vpt_nrd_view_z
IN_DIFF_RADIANCE_HITDIST    -> vpt_nrd_diff_radiance_hitdist
IN_DIFF_CONFIDENCE          -> vpt_nrd_diff_confidence, when enabled
OUT_DIFF_RADIANCE_HITDIST   -> vpt_nrd_out_diff_radiance_hitdist
OUT_VALIDATION              -> vpt_nrd_validation, when enabled
TRANSIENT_POOL              -> adapter-owned transient textures
PERMANENT_POOL              -> adapter-owned permanent textures
```

Diffuse/specular method mapping, enabled only after specular support is real:

```text
IN_MV                       -> vpt_nrd_motion
IN_NORMAL_ROUGHNESS         -> vpt_nrd_normal_roughness
IN_VIEWZ                    -> vpt_nrd_view_z
IN_DIFF_RADIANCE_HITDIST    -> vpt_nrd_diff_radiance_hitdist
IN_SPEC_RADIANCE_HITDIST    -> vpt_nrd_spec_radiance_hitdist
IN_DIFF_CONFIDENCE          -> vpt_nrd_diff_confidence, when enabled
IN_SPEC_CONFIDENCE          -> vpt_nrd_spec_confidence, when enabled
OUT_DIFF_RADIANCE_HITDIST   -> vpt_nrd_out_diff_radiance_hitdist
OUT_SPEC_RADIANCE_HITDIST   -> vpt_nrd_out_spec_radiance_hitdist
OUT_VALIDATION              -> vpt_nrd_validation, when enabled
TRANSIENT_POOL              -> adapter-owned transient textures
PERMANENT_POOL              -> adapter-owned permanent textures
```

### Resolve And Remodulation

Add `vpt_nrd_resolve` after NRD dispatches:

- unpack NRD diffuse output using NRD backend helpers
- unpack specular output only when specular is enabled
- multiply denoised diffuse by primary material factors
- add residual radiance
- write final HDR radiance for postprocess

Do not tonemap before NRD. Exposure changes must not be baked into NRD inputs.

### Debug Views

Add debug views before visual tuning:

```text
nrd_normal_roughness
nrd_viewz
nrd_motion
nrd_motion_z
nrd_diff_noisy
nrd_spec_noisy
nrd_diff_hitdist
nrd_spec_hitdist
nrd_diff_confidence
nrd_spec_confidence
nrd_diff_output
nrd_spec_output
nrd_validation
nrd_residual
```

`nrd_validation` is required for ReBLUR enablement.

## Implementation Phasing

### Phase 1: Settings And Interfaces

- Replace denoiser bool-only settings with `VptDenoiserMode`.
- Preserve old boolean aliases as `svgf`.
- Add typed graph structs for surface guides, NRD guides, noisy resources, and
  denoiser outputs.
- Keep the existing SVGF path behavior unchanged.

### Phase 2: Guide Contract

- Replace emissive-luma roughness misuse with true linear roughness.
- Add material parameter table and shader helpers.
- Add `viewZ` output.
- Move `motion_id` out of `motion_history.z`.
- Write 2.5D motion `.z = viewZprev - viewZ`.
- Add guide debug views and source tests.

### Phase 3: Noisy NRD Frontend

- Refactor `vpt.slang` output so NRD mode emits demodulated indirect diffuse
  radiance.
- Add lobe hit distance tracking.
- Add zero-filled, tested specular input while specular is disabled.
- Add residual radiance and material factors.
- Keep analytic sun direct, ReSTIR-DI direct, sky, primary-hit emissive, and
  debug radiance in residual radiance for the first NRD path.
- Add ReLAX packing pass.
- Keep current `noisy_radiance/noisy_moments` for SVGF fallback.

### Phase 4: Confidence And Validation Inputs

- Add `vpt_nrd_confidence` pass.
- Use motion flags and brick generation mismatch to suppress stale history.
- Add NRD validation-output resource wiring.
- Add captures for guide/noisy/confidence outputs.

### Phase 5: Official NRD Adapter

- Add external NRD SDK feature gate and C ABI wrapper.
- Create NRD instance for `RELAX_DIFFUSE` first.
- Create Vulkan resources, pipelines, samplers, descriptor sets, and pool
  textures from NRD descriptors.
- Record NRD dispatches through RenderGraph.
- Recreate on resize.
- Validate `RELAX_DIFFUSE` output against debug views and runtime smoke.

### Phase 6: ReLAX Product Path

- Add `vpt_nrd_resolve`.
- Feed resolved HDR output into postprocess.
- Make `REVOLUMETRIC_DENOISER=relax` usable for normal runtime.
- Keep `svgf` fallback.

### Phase 7: ReBLUR Enablement

- Add ReBLUR packing with normalized hit distance.
- Add ReBLUR settings and hit-distance parameter plumbing.
- Enable validation output by default in ReBLUR debug runs.
- Promote `REVOLUMETRIC_DENOISER=reblur` only after motion/viewZ/hitDist
  validation captures are clean.

### Phase 8: Specular Follow-up

- Add real specular material parameters and sampling.
- Split diffuse/specular paths at the primary hit.
- Enable `RELAX_DIFFUSE_SPECULAR` and `REBLUR_DIFFUSE_SPECULAR`.
- Keep diffuse-only mode available for regression isolation.

## Testing Strategy

Required static and unit tests:

- settings parse tests for `off`, `svgf`, `relax`, `reblur`, and old boolean
  aliases
- ABI size/offset tests for new material parameters and NRD settings structs
- shader source tests proving `surface_normal_roughness.w` is roughness, not
  emissive luminance
- shader source tests proving `motion_history.z` is not `motion_id`
- shader source tests proving `viewZ` is written independently from `hit.t`
- shader source tests proving diffuse/specular NRD inputs are demodulated or
  explicitly zero-filled
- source tests for named graph resource structs instead of anonymous NRD input
  arrays
- NRD feature-gate tests proving builds without NRD still support `svgf`

Required shader/build checks:

```powershell
cargo fmt
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --lib; cargo build --bin revolumetric --features desktop; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
git diff --check
```

Required runtime smoke for fallback path:

```powershell
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; $env:REVOLUMETRIC_DENOISER='svgf'; cargo run --features desktop --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES; Remove-Item Env:\REVOLUMETRIC_DENOISER
```

Required runtime smoke for NRD feature builds:

```powershell
$env:REVOLUMETRIC_NRD_ROOT='<accepted-local-nrd-sdk>'; cargo build --features 'desktop nrd' --bin revolumetric
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; $env:REVOLUMETRIC_DENOISER='relax'; cargo run --features 'desktop nrd' --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES; Remove-Item Env:\REVOLUMETRIC_DENOISER
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; $env:REVOLUMETRIC_DENOISER='relax'; $env:REVOLUMETRIC_VPT_DEBUG_VIEW='nrd_validation'; cargo run --features 'desktop nrd' --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES; Remove-Item Env:\REVOLUMETRIC_DENOISER; Remove-Item Env:\REVOLUMETRIC_VPT_DEBUG_VIEW
Remove-Item Env:\REVOLUMETRIC_NRD_ROOT
```

ReBLUR runtime smoke is required only after Phase 7:

```powershell
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; $env:REVOLUMETRIC_DENOISER='reblur'; cargo run --features 'desktop nrd' --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES; Remove-Item Env:\REVOLUMETRIC_DENOISER
```

Capture validation scenes:

- static camera, static scene
- slow camera pan
- disocclusion at voxel silhouettes
- UCVH content edit
- UCVH semantic region move
- direct sun only
- emissive voxel scene
- high-intensity firefly-prone scene
- depth-of-field / Area ReSTIR primary sample scene, if enabled

Acceptance criteria:

- no NRD path may become default until `svgf`, `relax`, and debug captures are
  all reproducible
- ReBLUR cannot be enabled by default until `nrd_validation`, `nrd_viewz`, and
  `nrd_motion_z` views show coherent values during motion
- specular denoising cannot be claimed until the renderer traces a real
  specular lobe

## Non-Goals

- Do not delete `vpt_temporal` or `vpt_atrous` until official NRD fallback and
  validation paths are stable.
- Do not implement a home-grown algorithm and call it ReBLUR/ReLAX.
- Do not vendor NVIDIA SDK code without license acceptance.
- Do not claim volumetric or transparency denoising support. This design covers
  opaque primary voxel surfaces.
- Do not fabricate specular inputs from diffuse data.
- Do not use primary hit distance as the NRD lobe hit distance.
- Do not bake exposure or tonemapping into denoiser inputs.

## Risks

- **License and distribution**: NRD uses the NVIDIA RTX SDK license. The build
  must support a non-NRD fallback, and vendoring requires explicit acceptance.
- **Input contract drift**: NRD resource enums and packing helpers can change by
  version. The adapter must verify `LibraryDesc` and expected encodings at
  startup.
- **Motion sign or scale errors**: wrong `IN_MV` convention causes ghosting.
  Validation views and tests must pin `old = new + MV`.
- **2.5D motion misuse**: keeping `motion_id` in `.z` blocks correct ReBLUR
  behavior. Move identity to a separate guide.
- **Roughness default risk**: roughness defaults are acceptable for
  diffuse-only startup, but specular denoising needs real material data.
- **Hit distance ambiguity**: mixed direct and indirect radiance cannot share a
  meaningless hit distance. The first implementation must either provide a
  valid lobe hit distance or keep that component in residual radiance.
- **Fireflies and HDR range**: NRD expects positive sane-range inputs. Add
  pre-pack clamping/compression only as a documented frontend operation, not as
  a hidden tonemap.
- **Descriptor model complexity**: NRD dynamic dispatches require sampled image
  descriptors, samplers, constant buffers, storage images, and pool resources.
  Existing descriptor helpers are not sufficient by themselves.
- **RenderGraph integration**: NRD internal dispatches read/write permanent and
  transient textures repeatedly. The graph must own synchronization and resource
  lifetime; no hidden pass-local barriers.
- **Content edits**: UCVH generation mismatch must reach NRD through
  confidence or a conservative restart. Otherwise NRD may reuse stale history
  through visually similar geometry.

## Review Gate

This design is the convergence point for replacing the lightweight SVGF-like
denoiser with official NRD ReLAX/ReBLUR integration. No production code should
change until an implementation plan is written from this spec and reviewed.
