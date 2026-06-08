# hecs ECS Migration Design

## Goal

Replace the custom ECS storage, query, and command scaffolding with `hecs` while keeping the existing app-level resource and stage scheduling model intact. The result should be a `hecs`-backed ECS boundary that preserves current startup flow, scene bootstrapping, and resource access patterns, without leaving two entity stores in the codebase.

## Current Facts

- `src/ecs/world.rs` currently owns `SlotMap<EntityId, ()>`, an `entity_cache: Vec<Entity>`, and the custom `Resources` map.
- `src/ecs/query.rs` only iterates the cached entity list; it does not query components.
- `src/ecs/commands.rs` only counts queued spawns; it does not apply deferred world mutations.
- `src/ecs/archetype.rs` and `src/ecs/column.rs` exist, but nothing outside `src/ecs/` references them.
- `src/scene/systems.rs::bootstrap_scene` only spawns one entity and inserts `CameraRig`, `DirectionalLight`, and `InputState` as resources.
- `src/app.rs` uses `World` for resource access and stage execution; it does not depend on any real component query behavior today.

## Scope

This phase replaces the entity/component backend with `hecs` and removes the custom ECS storage layers. It must not change:

- window creation or winit event handling
- render runtime ownership
- scene resource semantics
- schedule stage ordering
- render behavior, shaders, or voxel data flow

`Resources` stays custom, because `hecs` does not replace the app-wide typed resource table.

## Architecture

`src/ecs/world.rs` becomes the compatibility boundary around `hecs::World`:

- `World` owns a `hecs::World` for entities and components.
- `World` continues to own `Resources` for global typed values.
- `spawn` uses `hecs` bundle semantics and returns a `hecs::Entity`.
- component access is forwarded through `hecs`-native operations, not a custom entity cache or archetype table.
- deferred mutations, if needed later, should use `hecs::CommandBuffer` at the call site; this phase does not preserve the bespoke `Commands` counter.

The public ECS surface should therefore become:

- `World` for entity/component storage plus app resources
- `Resources` for global typed values
- `Schedule` for stage ordering and system execution

The following custom storage helpers are retired from the public module surface:

- `EntityId` / `Entity` wrapper
- `Query` cursor wrapper
- `Commands` counter
- `Archetype`
- `ColumnStorage`

If any of those files remain temporarily during implementation, they should only exist as short-lived migration scaffolding, not as a second ECS implementation.

## Public API Shape

`World` should expose hecs-backed operations in a form that keeps the rest of the codebase readable:

```rust
pub struct World {
    entities: hecs::World,
    resources: Resources,
}

impl World {
    pub fn new() -> Self;
    pub fn spawn<B: hecs::Bundle>(&mut self, bundle: B) -> hecs::Entity;
    pub fn despawn(&mut self, entity: hecs::Entity) -> bool;
    pub fn insert_resource<T>(&mut self, value: T)
        where T: Send + Sync + 'static;
    pub fn resource<T>(&self) -> Option<&T>
        where T: Send + Sync + 'static;
    pub fn resource_mut<T>(&mut self) -> Option<&mut T>
        where T: Send + Sync + 'static;
    pub fn entity_count(&self) -> usize;
}
```

If query helpers remain on `World`, they should forward to hecs query borrows directly. The old `Query` wrapper should not survive as a parallel abstraction.

`Schedule` stays as-is structurally:

```rust
pub type SystemFn = fn(&mut World) -> anyhow::Result<()>;
```

That keeps the app loop and startup systems stable while the storage backend changes underneath.

## Migration Plan

1. Add `hecs` to `Cargo.toml`.
2. Rebuild `src/ecs/world.rs` around `hecs::World`.
3. Update `src/scene/systems.rs` to use the hecs spawn path.
4. Remove or stop exporting the custom entity/query/command/storage modules.
5. Update any source-level guard tests that currently assert the old ECS shapes.
6. Add behavior tests that prove `spawn`, resource access, and simple hecs component query flow still work.

## Testing

Add focused regression tests around the new ECS boundary:

- `World::spawn` returns a live entity handle and supports component insertion/retrieval.
- `World::despawn` removes the entity cleanly.
- `Resources` still supports insert/get/get_mut unchanged.
- `bootstrap_scene` still populates the expected app resources.
- `Schedule` still runs startup systems in the same order.

Source-level guard tests should assert that:

- `src/ecs/world.rs` no longer imports `slotmap`
- `src/ecs/mod.rs` no longer exports the removed custom storage modules as public API
- `src/scene/systems.rs` uses the hecs-backed spawn path

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

## Acceptance Criteria

- The ECS backend is hecs-driven, not slotmap-driven.
- There is only one entity/component store in the codebase.
- App startup and scene bootstrapping still work without behavior changes.
- Resource access and stage scheduling continue to function as before.
- The full library test suite passes.

## Deferred Risks

- This phase does not introduce a full hecs system-parameter framework; systems remain `fn(&mut World) -> Result<()>`.
- If future code needs deferred mutation, it should adopt `hecs::CommandBuffer` directly rather than reviving a bespoke command counter.
- Any code that relied on the old custom `EntityId` or `Query` wrapper will need direct source updates during implementation.
