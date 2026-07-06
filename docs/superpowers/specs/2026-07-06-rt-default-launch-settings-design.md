# RT Default Launch Settings Design

## Goal

Make the default desktop startup exercise the completed hardware RT ReSTIR path on RT-capable devices.

The requested render mode stays `auto`: RT-capable devices use the hardware RT backend and unsupported devices fall back to VPT. The RT backend defaults change from opt-in ReSTIR features to a full RT ReSTIR startup profile:

- RT ReSTIR-DI enabled by default.
- RT ReSTIR-DI spatial reuse enabled by default.
- RT ReSTIR-GI enabled by default.
- RT temporal denoise, history length, compatibility thresholds, spatial sample count, and debug view keep their existing defaults.

Environment variables and editor UI controls remain authoritative overrides, so users can disable any RT ReSTIR component explicitly.

## Scope

Change only default startup settings, tests, and user-facing docs. Do not change window size, shader compilation mode, VPT ReSTIR defaults, Area ReSTIR defaults, RT fallback behavior, visual baseline case pinning, or pass scheduling semantics.

## Rationale

The RT pipeline now has UI status, capture metadata, visual baseline coverage, and tests for active RT pass state. Keeping the backend on `auto` preserves compatibility, while defaulting RT ReSTIR features on makes the normal RT-capable startup path validate the integrated RT pipeline without requiring manual environment setup.

## Testing

Update unit/source-check tests before production changes:

- `RtSettings::default()` should expose the new RT ReSTIR startup profile.
- `GpuRtSettings` built from defaults should encode enabled RT ReSTIR flags.
- README/source checks should document the default-on RT ReSTIR profile and the simplified RT smoke command.

Run focused failing tests first, then implement and run focused tests plus the standard local gates.

## Approval

The user previously gave `全部你自行决定，TRUSTED`; this small design is executed under that authorization.
