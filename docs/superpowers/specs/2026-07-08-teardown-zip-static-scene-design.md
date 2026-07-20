# Teardown Zip Static Scene Design

## Goal

Load a Teardown workshop zip directly into the existing UCVH scene as static voxel geometry, avoiding the current large intermediate `.vox` merge file.

## Scope

The first version restores static layout only. It reads `main.xml`, nested prefab `instance` XML, `MOD/...` zip resources, and `BUILT-IN/...` resources from a local Teardown install. It supports static `vox` and `voxbox` geometry and reports skipped or missing resources.

Out of scope for this phase: physics, joints, vehicles, scripts, water, lights, object selection, per-instance runtime transforms, and exact `hole` boolean CSG.

## Architecture

Add a Rust `src/voxel/teardown_zip_loader.rs` module. It owns Teardown zip/resource resolution, XML traversal, Teardown transform handling, and brick-batched UCVH writes. It reuses the existing MagicaVoxel parser in `src/voxel/vox_loader.rs` for referenced `.vox` assets, but it does not write or read an intermediate scene `.vox`.

Default scene generation changes from "load `run/Vintessa_Hills_static.vox` first" to "load `C:/Users/mc897/Downloads/Vintessa Hills.zip` first, then existing `.vox`, then checkerboard fallback".

## Data Flow

1. Detect the workshop zip root and read `main.xml`.
2. Resolve XML and `.vox` refs through `MOD/...` in the zip or `BUILT-IN/...` under the Teardown install.
3. Traverse XML transforms recursively, including `instance` prefab expansion.
4. First pass computes source bounds in Teardown voxel coordinates.
5. Second pass maps those bounds into the current UCVH world using the same bounded fit policy as the VOX loader and writes touched bricks in batches.

## Validation

Unit tests cover MOD voxel loading, BUILT-IN prefab expansion, voxbox rasterization, and fallback selection. Local validation may be blocked by unrelated render-tree compile errors; if so, the blocker is reported separately rather than hidden.
