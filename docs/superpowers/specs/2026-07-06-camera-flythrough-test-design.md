# Camera Flythrough Test Design

## Goal

Add a deterministic camera flythrough mode for inspecting RT temporal accumulation, reprojection, noise stability, and stutter without manual input.

## Selected Approach

Use an app-level automatic camera path plus a lightweight PowerShell wrapper.

The app owns frame timing and camera state, so it can update the `CameraRig` deterministically before each render. A script-only wrapper cannot move the camera today because the renderer only exposes capture and exit-frame environment variables. A separate headless renderer is larger than this diagnostic need.

## Runtime Controls

- `REVOLUMETRIC_CAMERA_PATH=orbit` enables the test path.
- `REVOLUMETRIC_CAMERA_PATH_CENTER=x,y,z` overrides the map center. Default: `64,32,64`.
- `REVOLUMETRIC_CAMERA_PATH_RADIUS=value` overrides orbit radius. Default: `72`.
- `REVOLUMETRIC_CAMERA_PATH_HEIGHT=value` overrides camera height. Default: `44`.
- `REVOLUMETRIC_CAMERA_PATH_PERIOD_FRAMES=value` controls one loop duration. Default: `240`.

Invalid optional values fall back to defaults. Unknown path names disable the automatic path.

## Behavior

When enabled, each frame places the camera on a smooth horizontal orbit around the center and points it inward with a slight downward component. Manual fly-camera input is skipped while the automatic path is active so user input does not corrupt deterministic captures.

## Tooling

Add `tools/rt_flythrough_capture.ps1`.

The script starts the app with the automatic camera path and supports:

- real-time observation by default;
- optional exit after `-Frames`;
- optional single-frame capture with `-CaptureFrame`;
- optional same-run multi-frame capture with `-CaptureFrames "2,16,32"`;
- render mode selection with `-Mode auto|rt|vpt`;
- optional RT debug view.

It does not run multi-process capture loops by default; multi-frame capture uses
one deterministic app run so temporal history is preserved between captured
frames.

## Verification

Use targeted source/unit tests only:

- camera path config parsing;
- deterministic orbit position and inward direction;
- app update path skipping manual fly input;
- script contract includes expected environment variables.

No full render capture is required for this change.
