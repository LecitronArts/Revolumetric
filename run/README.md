# Local Run Assets

This directory is for machine-local runtime and SDK assets. The repository
tracks only this README and helper scripts; large or licensed third-party files
under this directory are ignored.

## NRD SDK Layout

Put the accepted NVIDIA NRD SDK checkout or prepared SDK bundle here:

```text
run/
  nrd/
    Include/
      NRD.h
      NRDDescs.h
      NRDSettings.h
    Lib/            # or _Bin/, lib/, Build/Release/, build/Release/, build/lib/
      NRD.lib       # or libNRD.a
      ShaderMakeBlob.lib  # needed by official static NRD builds
```

When building with `--features nrd`, `build.rs` uses this order:

1. `REVOLUMETRIC_NRD_ROOT`, if set.
2. `run/nrd`, if the environment variable is unset.

Use the checked-in validation wrapper from the repo root:

```powershell
.\run\validate-nrd.ps1 -Frames 3
```

If an IDE launch environment does not inherit the Vulkan SDK `PATH`, set an
explicit shader compiler path:

```text
REVOLUMETRIC_SLANGC=D:\VulkanSDK\Bin\slangc.exe
```

For an interactive visual run, keep the app open:

```powershell
.\run\validate-nrd.ps1 -Frames 0
```

For guide-buffer checks:

```powershell
.\run\validate-nrd.ps1 -DebugView nrd_normal_roughness
.\run\validate-nrd.ps1 -DebugView nrd_viewz
.\run\validate-nrd.ps1 -DebugView nrd_motion
.\run\validate-nrd.ps1 -DebugView nrd_motion_z
.\run\validate-nrd.ps1 -DebugView nrd_validation
```
