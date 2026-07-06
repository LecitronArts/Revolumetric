# RT ReSTIR Active Pass Status Design

## Goal

Expose whether optional RT ReSTIR-DI and RT ReSTIR-GI passes participated in the last hardware RT frame through the existing runtime status path and editor UI.

## Current Facts

- `RtRuntimePipeline::record_and_execute_frame` already computes local booleans:
  - `rt_restir_di_rendered`
  - `rt_restir_gi_rendered`
- Capture metadata already records those booleans as `rt_restir_di_rendered` and `rt_restir_gi_rendered`.
- `RtFrameStatus` currently exposes resource/history readiness, but not last-frame optional pass activity.
- The editor shows `di_history` and `gi_history`, which are useful but not identical to "this pass participated in the last frame."

## Design

Add two booleans to `RtFrameStatus` and `RtPipelineFrameState`:

```rust
pub restir_di_rendered: bool,
pub restir_gi_rendered: bool,
```

After `record_and_execute_frame` executes the graph, store the local active-pass values in frame state only when the RT graph rendered:

```rust
self.frame_state.restir_di_rendered = rt_graph_rendered && rt_restir_di_rendered;
self.frame_state.restir_gi_rendered = rt_graph_rendered && rt_restir_gi_rendered;
```

Clear both fields in `reset_history` and in fallback frames. `frame_status()` snapshots them through `RenderRuntimeStatus.rt_frame_status`.

Extend editor formatting with helper labels:

```rust
fn rt_frame_restir_di_pass_label(status: Option<RenderRuntimeStatus>) -> &'static str
fn rt_frame_restir_gi_pass_label(status: Option<RenderRuntimeStatus>) -> &'static str
```

Display the labels beside the existing RT ready row:

- render panel: `di_pass true|false|unknown`, `gi_pass true|false|unknown`
- console: `rt_restir_di_pass=<token>`, `rt_restir_gi_pass=<token>`

Do not put these labels in the top bar. The top bar is already dense and should keep only backend, support, frame readiness, and skip reason.

## Non-Goals

- Do not change RT graph registration or pass activation conditions.
- Do not change capture metadata; it already records these fields.
- Do not treat inactive optional ReSTIR passes as full-frame skip reasons.
- Do not add new debug views or shader behavior.

## Testing Strategy

Use TDD with source and helper tests:

- Add `RtFrameStatus` tests proving `frame_status()` snapshots `restir_di_rendered` and `restir_gi_rendered`.
- Add source-contract tests proving `record_and_execute_frame` writes active-pass state from `rt_graph_rendered && rt_restir_*_rendered`, and `reset_history` clears it.
- Add editor helper tests for `true`, `false`, and `unknown` active-pass labels.
- Extend editor source tests to require render panel and console tokens.

Verification commands:

- `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib rt_pipeline`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib rt_frame`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib top_bar_render_panel_and_console_report_rt_frame_status`
- `cargo fmt --check`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`
- `REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings`
- `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib`
- `.\run\validate-visual-baseline.ps1 -Rt`
- `git diff --check`

## Risks

The labels are a last-completed-frame snapshot, matching the existing runtime status architecture. They can lag UI controls by one frame and report `false` when optional ReSTIR is disabled, warming, or unavailable. The labels are diagnostic pass-activity signals, not image quality metrics.
