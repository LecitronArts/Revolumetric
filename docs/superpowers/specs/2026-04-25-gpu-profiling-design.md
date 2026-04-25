# Phase P1: GPU Profiling — Design Specification

## Goal

Add a production-quality GPU profiling layer for Revolumetric's Vulkan renderer. The profiler must measure the major GPU passes with low runtime overhead, provide immediately useful console summaries, and optionally export frame-level CSV data for optimization comparisons.

## Non-Goals

- No shader behavior changes.
- No visual output changes.
- No in-app HUD/debug overlay in P1.
- No CPU profiling or benchmark scene automation in P1.

## Requirements

- Debug builds enable GPU profiling by default; release builds disable it by default.
- `REVOLUMETRIC_GPU_PROFILER=1` forces profiling on.
- `REVOLUMETRIC_GPU_PROFILER=0` forces profiling off.
- `REVOLUMETRIC_GPU_PROFILE_CSV=<path>` enables CSV export.
- `REVOLUMETRIC_GPU_PROFILE_CSV_FLUSH_INTERVAL=<frames>` controls CSV flush cadence; default batches rows to reduce profiling-path disk I/O.
- Unsupported timestamp-capable queues must disable profiling safely with a single warning.
- Disabled profiling must avoid Vulkan query calls in the frame loop.
- Enabled profiling must read back delayed results rather than stalling on the current frame.
- Console summaries print every 60 completed GPU profile frames.

## Measured Scopes

The first implementation measures the current render path at pass granularity:

1. Primary Ray
2. RC Clear
3. RC Trace C0
4. RC Trace C1
5. RC Trace C2
6. RC Merge C2→C1
7. RC Merge C1→C0
8. Lighting
9. Blit To Swapchain

RC Clear is included because first-frame buffer initialization can hide a major one-off stall and should be visible rather than folded into unrelated passes.

## Architecture

Add `src/render/gpu_profiler.rs` with these responsibilities:

- Parse environment configuration into a small `GpuProfilerConfig` value.
- Define `GpuProfileScope` as the stable list of measured pass scopes.
- Own a Vulkan timestamp `vk::QueryPool` sized as `frames_in_flight * scope_count * 2`.
- Reset and write timestamps for the active frame slot.
- Read back the previous frame slot after the frame fence has been waited by `RenderDevice::begin_frame`.
- Convert timestamp ticks to milliseconds using `timestamp_period`.
- Accumulate rolling totals for console summaries.
- Write optional CSV rows with per-frame timings.

The app stores `Option<GpuProfiler>`. If profiler construction returns disabled/unsupported, the option remains `None`, so the hot path only pays an `if let Some` branch.

## Vulkan Timing Model

Each scope uses two timestamp queries:

- `begin_scope(scope)` writes `vkCmdWriteTimestamp2` or `vkCmdWriteTimestamp` at `COMPUTE_SHADER`/`BOTTOM_OF_PIPE` compatible stages depending on the available API wrapper.
- `end_scope(scope)` writes the matching end timestamp.

For portability with the current `ash` usage, P1 can use `cmd_write_timestamp` with conservative pipeline stages:

- Begin: `vk::PipelineStageFlags::TOP_OF_PIPE`
- End: `vk::PipelineStageFlags::BOTTOM_OF_PIPE`

This measures elapsed queued GPU work for each command region. Later phases can refine stage masks if the renderer moves to synchronization2.

## Readback Strategy

At the start of a reused frame slot, `RenderDevice::begin_frame` has already waited for that slot's fence. The profiler can therefore read that slot's previous query results without waiting on the GPU. The frame flow is:

1. `begin_frame` waits for the frame slot fence.
2. Profiler reads finished timestamps for that slot from the previous use.
3. Profiler resets that slot's query range for the new command buffer.
4. Render passes write begin/end timestamps.
5. `end_frame` submits normally.

This keeps profiling asynchronous and avoids a per-frame GPU stall.

## Console Output

Every 60 valid profiled frames, log one line through `tracing::info!`:

```text
[GPU] PrimaryRay: 1.23ms | RCClear: 0.00ms | RC-C0: 0.45ms | RC-C1: 0.89ms | RC-C2: 0.34ms | Merge C2→C1: 0.12ms | Merge C1→C0: 0.10ms | Lighting: 0.67ms | Blit: 0.05ms | Total: 3.85ms
```

The summary reports averages over the last 60 valid readbacks.

## CSV Output

If `REVOLUMETRIC_GPU_PROFILE_CSV` is set, create parent directories when needed and write:

```csv
frame,primary_ray_ms,rc_clear_ms,rc_trace_c0_ms,rc_trace_c1_ms,rc_trace_c2_ms,rc_merge_c2_to_c1_ms,rc_merge_c1_to_c0_ms,lighting_ms,blit_to_swapchain_ms,total_ms
```

Each subsequent row records one finished GPU frame. Rows are flushed in batches and all pending rows are flushed during profiler shutdown. CSV write or flush errors disable CSV output and emit one warning; they must not stop rendering.

## Integration Points

- `src/render/mod.rs`: export `gpu_profiler`.
- `src/render/device.rs`: expose physical-device timestamp properties and the frame slot lifecycle data needed by the profiler, or pass them through from `app.rs` using existing accessors if simpler.
- `src/app.rs`: construct profiler after `RenderDevice` creation, call readback/reset after `begin_frame`, and wrap GPU command regions in profiling scopes.

## Testing Strategy

Unit tests cover deterministic logic without requiring Vulkan:

- Environment parsing defaults and overrides.
- Scope names and CSV column order.
- Timestamp delta conversion using a fake timestamp period.
- Rolling 60-frame average aggregation.
- CSV row formatting.

Manual verification for Vulkan integration:

- `cargo test` passes.
- `cargo run` emits `[GPU]` summaries in debug builds.
- `REVOLUMETRIC_GPU_PROFILER=0 cargo run` emits no profiler output.
- `REVOLUMETRIC_GPU_PROFILE_CSV=target/gpu-profile.csv cargo run` creates CSV rows.

## Future Extensions

- Add CPU frame timing and CPU/GPU overlap reporting.
- Add in-app HUD once the project has a debug UI layer.
- Add benchmark scene presets and automated before/after comparison.
- Add nested scopes if future render graph execution becomes more generic.
