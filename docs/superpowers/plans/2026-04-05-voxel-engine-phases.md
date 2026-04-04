# Revolumetric — Implementation Phase Overview

Each phase produces working, testable software. Later phases build on earlier ones.

| Phase | Name | Deliverable | Depends On |
|-------|------|-------------|------------|
| 1 | Render Foundation | Render graph + GPU resource management + Slang pipeline + fullscreen compute | — |
| 2 | UCVH Core | Brick pool + occupancy hierarchy + upload pass + procedural demo scene | Phase 1 |
| 3 | Voxel Ray Tracing | Hierarchical DDA shader + primary ray pass + flat-color visualization | Phase 2 |
| 4 | Shadow & Lighting | Shadow map pass + sun/sky lighting + composite + tonemap | Phase 3 |
| 5 | Radiance Cascades | Probe storage + cascade trace + cascade merge + GI visualization | Phase 3 |
| 6 | Full Composite | Integrate RC GI + shadow + specular + emissive + AO into final image | Phase 4, 5 |
| 7 | Data Pipeline | .vox import + runtime editing + procedural generators | Phase 2 |
| 8 | Advanced GI | Multi-bounce + hemisphere overlap reduction + temporal reprojection | Phase 5 |
| 9 | Camera & Input | FPS camera controller + WASD + mouse look | Phase 3 |
| 10 | Polish | Hot reload + debug views + perf tuning + visual regression tests | Phase 6 |

Phase 1 plan: `2026-04-05-phase1-render-foundation.md`
