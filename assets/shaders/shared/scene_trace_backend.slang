#pragma once
// Active scene tracing backend selection.
//
// Render passes include this file, not a concrete backend. To switch tracing
// from voxels to mesh BVH, SDF, hardware RT, or a hybrid backend, provide the
// same SceneHit/trace_scene_ray/scene_outside_geometry contract here.

#include "scene_trace_voxel.slang"
