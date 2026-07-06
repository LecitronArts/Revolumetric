# Visual Baseline PPM Signal Gate Design

## Goal

Strengthen the local visual baseline gate so a capture must show meaningful RGB signal coverage and basic dynamic range, not merely one non-zero byte.

## Current Facts

- `run/validate-visual-baseline.ps1` validates PPM header, dimensions, byte length, metadata, and that at least one RGB byte is non-zero.
- The current non-zero-byte check would pass a mostly black image with one bright pixel.
- The latest local captures have broad signal:
  - `svgf_final`: non-zero pixel ratio about `0.999888`, RGB range `233`.
  - `rt_surface_debug`: non-zero pixel ratio about `0.857143`, RGB range `255`.
- Visual baseline validation is a smoke gate, not perceptual image comparison.

## Design

Add optional manifest thresholds:

```json
"expectedMinNonZeroPixelRatio": 0.25,
"expectedMinRgbRange": 32
```

Extend the PowerShell validator with a PPM statistics helper that computes:

- total pixel count from the PPM header
- number of pixels where at least one RGB channel is non-zero
- non-zero pixel ratio
- minimum RGB channel value
- maximum RGB channel value
- RGB range

Replace the one-byte non-zero assertion with `Assert-PpmSignal`. The new assertion should:

- always reject captures with zero non-black pixels
- when `expectedMinNonZeroPixelRatio` is present, reject captures below that ratio
- when `expectedMinRgbRange` is present, reject captures below that RGB range

Add conservative thresholds to the measured default and RT cases:

- `svgf_final`: `expectedMinNonZeroPixelRatio: 0.25`, `expectedMinRgbRange: 32`
- `rt_surface_debug`: `expectedMinNonZeroPixelRatio: 0.25`, `expectedMinRgbRange: 32`

Do not add thresholds to NRD-gated cases in this slice because those captures were not measured in this environment during this cycle. The helper remains optional so future NRD runs can add thresholds once measured.

## Non-Goals

- Do not add golden image byte comparisons.
- Do not add perceptual metrics or image diff storage.
- Do not change renderer output.
- Do not require RT hardware for the default visual baseline run.
- Do not claim visual quality from this gate; it only catches blank, almost blank, and flat captures.

## Testing Strategy

Use TDD with source-contract tests:

- Update `visual_baseline_script_validates_captures_metadata_and_nonblank_ppm` to require the new statistics helper, threshold fields, and signal assertion.
- Update `visual_baseline_manifest_covers_svgf_and_reblur_debug_cases` to require the new thresholds on measured cases.
- Run the focused source-check tests first.
- Run `.\run\validate-visual-baseline.ps1 -Rt` to confirm the real default and RT captures pass the new thresholds.

Run the normal verification set before committing:

- `cargo fmt --check`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings`
- `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib`
- `.\run\validate-visual-baseline.ps1 -Rt`
- `git diff --check`

## Risks

The thresholds are intentionally broad and scene-smoke-oriented. They can catch blank, nearly blank, or flat captures, but they cannot detect wrong lighting, wrong material colors, temporal artifacts, or subtle RT/VPT regressions. Future perceptual or golden-image validation should be a separate design.
