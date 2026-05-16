use crate::assets::shader_reflect::{DescriptorBinding, DescriptorKind, ShaderReflection};

use super::AreaRestirPass;

fn source(path: &str) -> String {
    crate::render::source_checks::read_source(path)
}

fn binding(binding: u32, kind: DescriptorKind, name: &str) -> DescriptorBinding {
    DescriptorBinding {
        set: 0,
        binding,
        kind,
        name: name.to_string(),
    }
}

fn shader_reflection(path: &str) -> ShaderReflection {
    ShaderReflection::from_slang_compiled_or_source("main", path, &source(path))
        .expect("shader reflection should parse")
}

fn shader_bindings(path: &str) -> Vec<DescriptorBinding> {
    shader_reflection(path).bindings
}

#[test]
fn area_restir_descriptor_specs_match_shader_manifests() {
    let initial_specs = AreaRestirPass::initial_descriptor_binding_specs();
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "Area ReSTIR initial",
        &initial_specs,
        &shader_reflection("assets/shaders/passes/area_restir_initial.slang"),
    );

    let temporal_specs = AreaRestirPass::temporal_descriptor_binding_specs();
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "Area ReSTIR temporal",
        &temporal_specs,
        &shader_reflection("assets/shaders/passes/area_restir_temporal.slang"),
    );

    let spatial_specs = AreaRestirPass::spatial_descriptor_binding_specs();
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "Area ReSTIR spatial",
        &spatial_specs,
        &shader_reflection("assets/shaders/passes/area_restir_spatial.slang"),
    );
}

#[test]
fn area_restir_initial_shader_binding_manifest_matches_expected_resources() {
    assert_eq!(
        shader_bindings("assets/shaders/passes/area_restir_initial.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "area_restir"),
            binding(1, DescriptorKind::StorageBuffer, "output_reservoirs"),
            binding(2, DescriptorKind::StorageImage, "surface_position_depth"),
            binding(3, DescriptorKind::StorageImage, "surface_normal_roughness"),
            binding(4, DescriptorKind::StorageImage, "surface_albedo_material"),
            binding(5, DescriptorKind::StorageImage, "area_restir_debug"),
            binding(6, DescriptorKind::UniformBuffer, "scene_ubo"),
            binding(7, DescriptorKind::StorageBuffer, "ucvh_config"),
            binding(8, DescriptorKind::StorageBuffer, "hierarchy_l0"),
            binding(9, DescriptorKind::StorageBuffer, "hierarchy_l1"),
            binding(10, DescriptorKind::StorageBuffer, "hierarchy_l2"),
            binding(11, DescriptorKind::StorageBuffer, "hierarchy_l3"),
            binding(12, DescriptorKind::StorageBuffer, "hierarchy_l4"),
            binding(13, DescriptorKind::StorageBuffer, "brick_occupancy"),
            binding(14, DescriptorKind::StorageBuffer, "brick_materials"),
        ]
    );
}

#[test]
fn area_restir_pass_owns_graph_registration_contract() {
    let pass = source("src/render/passes/area_restir.rs");
    let implementation = pass
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");
    assert!(implementation.contains("pub fn register_graph"));
    assert!(implementation.contains("vpt_surface.update_area_restir_descriptors"));
    assert!(implementation.contains("vpt.update_area_restir_descriptors"));
}

#[test]
fn area_restir_pass_declares_independent_resources_without_local_barriers() {
    let implementation = source("src/render/passes/area_restir.rs");
    let implementation = implementation
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    for token in [
        "pub struct AreaRestirPass",
        "initial_reservoirs",
        "temporal_reservoirs",
        "selected_reservoirs",
        "debug_image",
        "MemoryLocation::GpuOnly",
        "resize_buffers",
        "update_surface_descriptors",
        "selected_current_buffer",
        "selected_history_buffer",
        "update_frame_descriptors",
        "record_initial",
        "record_temporal",
        "record_spatial",
        "update_ucvh_descriptors",
        "write_ucvh_descriptors",
    ] {
        assert!(
            implementation.contains(token),
            "Area ReSTIR pass missing token {token}"
        );
    }

    assert!(!implementation.contains("cmd_pipeline_barrier"));
    assert!(!implementation.contains("cmd_copy_buffer"));
    assert!(!implementation.contains("ImageMemoryBarrier"));
    assert!(!implementation.contains("BufferMemoryBarrier"));
    assert!(!implementation.contains("GpuRestirDiReservoir"));
    assert!(!implementation.contains("spatial_reservoirs"));
    assert!(!implementation.contains("history_reservoirs"));
    assert!(!implementation.contains("AreaRestirHistorySource"));
}

#[test]
fn area_restir_resize_cleans_up_failed_resource_creation() {
    let implementation = source("src/render/passes/area_restir.rs");
    let implementation = implementation
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    for token in [
        "struct AreaRestirResizeResources",
        "impl AreaRestirResizeResources",
        "fn new(",
        "fn destroy(self, device: &ash::Device, allocator: &GpuAllocator)",
        "let resized = AreaRestirResizeResources::new(",
        "std::mem::replace(&mut self.initial_reservoirs, resized.initial_reservoirs)",
        "std::mem::replace(&mut self.selected_reservoirs, resized.selected_reservoirs)",
        "std::mem::replace(&mut self.debug_image, resized.debug_image)",
    ] {
        assert!(
            implementation.contains(token),
            "Area ReSTIR resize path missing cleanup token {token}"
        );
    }
}

#[test]
fn app_area_restir_dispatches_only_enabled_reuse_stages() {
    let app = source("src/app.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let pass = source("src/render/passes/area_restir.rs");
    let pass_impl = pass
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");
    let compact_pass = pass_impl.split_whitespace().collect::<String>();

    for token in [
        "letsettings=area_restir_effective_settings(settings,history_initialized);",
        "lettemporal_active=settings.temporal_enabled;",
        "letspatial_active=temporal_active&&settings.spatial_enabled;",
        "letinitial_output_resource=iftemporal_active{initial_resource}else{selected_current_resource};",
        "lettemporal_dep=iftemporal_active{",
        "graph.add_pass(\"area_restir_temporal\"",
        "}else{initial_dep};",
        "lettemporal_output_resource=ifspatial_active{temporal_resource}else{selected_current_resource};",
        ")=ifspatial_active{",
        "graph.add_pass(\"area_restir_spatial\"",
        "vpt.update_area_restir_descriptors(",
        "vpt_surface.update_area_restir_descriptors(",
    ] {
        assert!(
            compact_pass.contains(token),
            "Area ReSTIR pass graph missing conditional-dispatch token {token}"
        );
    }
    assert!(
        compact_pass.contains("if!history_initialized{settings.temporal_enabled=false;}")
            && compact_pass.contains(
                "settings.spatial_enabled=settings.temporal_enabled&&settings.spatial_enabled;"
            )
            && compact_pass
                .contains("area_restir_effective_settings(settings,history_initialized)")
            && compact_pass
                .contains("letspatial_active=temporal_active&&settings.spatial_enabled;"),
        "effective settings must disable spatial reuse until temporal history is usable"
    );
    assert!(app.contains("self.vpt_pipeline.ensure_passes("));
    assert!(
        compact_pipeline.contains("area_restir.register_graph(")
            && compact_pipeline.contains(
                "inputs.area_restir_settings,self.frame_state.area_restir_history_initialized,"
            )
            && compact_pipeline.contains("final_surface_writes=area_graph.final_surface_writes;"),
        "VPT pipeline must delegate Area ReSTIR graph registration while preserving settings and outputs"
    );
}

#[test]
fn app_uses_area_restir_selected_frame_ring_for_history_and_vpt_reads() {
    let app = source("src/app.rs");
    let pass = source("src/render/passes/area_restir.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let pass_impl = pass
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");
    let compact_pass = pass_impl.split_whitespace().collect::<String>();
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();

    for token in [
        "fnarea_restir_effective_settings(settings:AreaRestirSettings,history_initialized:bool,)->AreaRestirSettings{",
        "self.selected_current_buffer(frame_slot)",
        "self.selected_history_buffer(frame_slot)",
        "selected_current_resource",
        "selected_history_resource",
        "area_restir_effective_settings(settings,history_initialized)",
        "builder.read_as(selected_history_resource,",
        "builder.write_as(selected_current_resource,",
        "selected_current_resource:selected_resource,",
    ] {
        assert!(
            compact_pass.contains(token),
            "Area ReSTIR pass selected frame-ring policy missing token {token}"
        );
    }
    assert!(pipeline.contains("pub area_restir_pass: Option<AreaRestirPass>"));
    assert!(pipeline.contains("self.frame_state.reset_for_resize_or_camera_cut();"));
    assert!(app.contains("self.vpt_pipeline.ensure_passes("));
    assert!(
        compact_pipeline.contains(
            "vpt_area_restir_reads=Some((area_graph.uniform_resource,area_graph.selected_current_resource,));"
        ),
        "VPT pipeline must feed VPT the selected Area ReSTIR graph resource returned by the pass"
    );
    assert!(
        !compact_pass.contains("area_restir_selected_reservoir("),
        "Area ReSTIR final reservoir selection must not point VPT/history at intermediate buffers"
    );
    assert!(
        !pipeline.contains("area_restir_history_update")
            && !pass_impl.contains("area_restir_history_update"),
        "Area ReSTIR graph must not add a transfer history update pass"
    );
}

#[test]
fn area_restir_shaders_declare_expected_entry_points_and_resources() {
    let initial = source("assets/shaders/passes/area_restir_initial.slang");
    let temporal = source("assets/shaders/passes/area_restir_temporal.slang");
    let spatial = source("assets/shaders/passes/area_restir_spatial.slang");

    for shader in [&initial, &temporal, &spatial] {
        assert!(shader.contains("#include \"area_restir_common.slang\""));
        assert!(shader.contains("#include \"scene_common.slang\""));
        assert!(shader.contains("[shader(\"compute\")]"));
        assert!(shader.contains("AreaRestirUniforms"));
        assert!(shader.contains("AreaRestirReservoir"));
        assert!(!shader.contains("RestirDiReservoir"));
    }

    for token in [
        "RWStructuredBuffer<AreaRestirReservoir> output_reservoirs",
        "ConstantBuffer<SceneUniforms> scene_ubo",
        "StructuredBuffer<UcvhConfig> ucvh_config",
        "StructuredBuffer<NodeL0> hierarchy_l0",
        "StructuredBuffer<NodeLN> hierarchy_l1",
        "StructuredBuffer<NodeLN> hierarchy_l2",
        "StructuredBuffer<NodeLN> hierarchy_l3",
        "StructuredBuffer<NodeLN> hierarchy_l4",
        "StructuredBuffer<BrickOccupancy> brick_occupancy",
        "StructuredBuffer<VoxelCell> brick_materials",
        "area_restir_invalid_reservoir",
    ] {
        assert!(initial.contains(token), "initial shader missing {token}");
    }

    assert!(
        !initial.contains("scene_primary_ray_from_area_sample((SceneUniforms)0"),
        "initial shader must use real SceneUniforms instead of a zeroed placeholder"
    );

    for token in [
        "StructuredBuffer<AreaRestirReservoir> history_reservoirs",
        "motion_history",
        "area_restir_temporal_surface_compatible",
        "history_length",
    ] {
        assert!(temporal.contains(token), "temporal shader missing {token}");
    }

    for token in [
        "StructuredBuffer<AreaRestirReservoir> temporal_reservoirs",
        "spatial_sample_count",
        "spatial_radius",
        "neighbor_offsets",
        "area_restir_surface_compatible",
    ] {
        assert!(spatial.contains(token), "spatial shader missing {token}");
    }
}

#[test]
fn area_restir_surface_inputs_use_storage_images_with_cached_reads() {
    let pass = source("src/render/passes/area_restir.rs");
    let pass = pass
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");
    let initial = source("assets/shaders/passes/area_restir_initial.slang");
    let temporal = source("assets/shaders/passes/area_restir_temporal.slang");
    let spatial = source("assets/shaders/passes/area_restir_spatial.slang");

    for token in [
        "vk::DescriptorType::STORAGE_IMAGE",
        "storage_image_count",
        "write_image_descriptors",
    ] {
        assert!(
            pass.contains(token),
            "Area ReSTIR pass missing storage-image descriptor token {token}"
        );
    }
    assert!(
        !pass.contains("vk::DescriptorType::SAMPLED_IMAGE"),
        "Area ReSTIR sampled-image path regressed profiling and must not be reintroduced without new evidence"
    );

    for (name, shader) in [
        ("initial", initial.as_str()),
        ("temporal", temporal.as_str()),
        ("spatial", spatial.as_str()),
    ] {
        assert!(
            shader.contains("RWTexture2D<float4> surface_position_depth"),
            "{name} shader must bind surface_position_depth as storage image"
        );
        assert!(
            !shader.contains("\nTexture2D<float4> surface_position_depth"),
            "{name} shader must not use sampled surface inputs after profiling regression"
        );
        assert!(
            !shader.contains(".Load(int3("),
            "{name} shader must not use sampled texture Load for surface inputs"
        );
    }
    assert!(
        initial.contains("RWTexture2D<float4> area_restir_debug")
            && spatial.contains("RWTexture2D<float4> area_restir_debug"),
        "Area ReSTIR debug output remains a storage image"
    );
}

#[test]
fn area_restir_shaders_cache_per_pixel_surface_reads() {
    let initial = source("assets/shaders/passes/area_restir_initial.slang");
    let temporal = source("assets/shaders/passes/area_restir_temporal.slang");

    crate::render::source_checks::assert_contains_all(
        &initial,
        &[
            "float4 center_position_depth = surface_position_depth[pixel];",
            "AreaRestirCandidateSurface center_surface = read_center_surface(pixel);",
            "AreaRestirCandidateSurface candidate_surface = evaluate_area_restir_candidate_surface(",
            "uint2 pixel,\n    AreaRestirSampleState sample_state",
            "evaluate_area_restir_candidate_surface(scene_ubo, pixel, sample_state)",
            "ScenePrimaryRay primary_ray = scene_primary_ray_from_area_sample(",
            "HitResult hit = trace_primary_ray(",
            "make_ray(primary_ray.origin, primary_ray.direction)",
            "float target_pdf = area_restir_candidate_target_pdf(center_surface, candidate_surface);",
            "float2 pixel_sample = area_restir_pixel_sample(pixel, sample_state);",
            "surface.position_depth = float4(hit.position, hit.t);",
        ],
        "initial shader ray-evaluated candidate",
    );
    assert!(
        initial.contains("#include \"voxel_traverse.slang\"")
            && initial.contains("#include \"material_common.slang\"")
            && initial.contains("StructuredBuffer<UcvhConfig> ucvh_config")
            && initial.contains("StructuredBuffer<NodeL0> hierarchy_l0")
            && initial.contains("StructuredBuffer<NodeLN> hierarchy_l1")
            && initial.contains("StructuredBuffer<NodeLN> hierarchy_l2")
            && initial.contains("StructuredBuffer<NodeLN> hierarchy_l3")
            && initial.contains("StructuredBuffer<NodeLN> hierarchy_l4")
            && initial.contains("StructuredBuffer<BrickOccupancy> brick_occupancy")
            && initial.contains("StructuredBuffer<VoxelCell> brick_materials"),
        "initial shader must bind UCVH resources to evaluate each area candidate ray"
    );
    assert!(
        !initial.contains("float target_pdf = target_luma;")
            && !initial.contains("float target_luma = surface_target_luma("),
        "initial shader must not assign every candidate the same center-surface target"
    );
    assert!(
        !initial.contains("float2 pixel_sample = float2(pixel) + sample_state.subpixel_uv;"),
        "Area ReSTIR subpixel_uv is stored in [0,1) and must be converted to a pixel-center-relative sample before tracing"
    );
    assert!(
        !initial.contains("sample_state.pixel_sample")
            && !initial.contains("reservoir.selected_radiance")
            && !initial.contains("distance(primary_ray.origin, hit.position)"),
        "initial shader must not preserve unused reservoir payload or recompute hit depth"
    );
    assert!(
        temporal.contains("float4 motion = center_context.motion_history;")
            && !temporal.contains("float4 motion = motion_history.Load(int3(pixel, 0));"),
        "temporal shader must reuse motion already loaded into center_context"
    );
    assert!(
        temporal.contains("float2 history_sample = vpt_history_sample_from_motion(pixel, motion);"),
        "Area ReSTIR temporal must reconstruct the previous pixel center from motion delta"
    );
}

#[test]
fn area_restir_debug_writes_are_gated_by_debug_view() {
    let initial = source("assets/shaders/passes/area_restir_initial.slang");
    let spatial = source("assets/shaders/passes/area_restir_spatial.slang");

    for (name, shader) in [("initial", initial.as_str()), ("spatial", spatial.as_str())] {
        assert!(
            shader.contains("if (area_restir.debug_view != 0u)"),
            "{name} shader must not write the debug image on the default debug-off path"
        );
    }
}

#[test]
fn area_restir_common_declares_replay_target_and_weighted_reservoir_update() {
    let common = source("assets/shaders/shared/area_restir_common.slang");

    for token in [
        "float area_restir_replay_target_pdf",
        "bool area_restir_candidate_finite",
        "float area_restir_reservoir_stream_weight",
        "float area_restir_reservoir_reuse_weight",
        "void area_restir_finalize_reservoir",
        "void area_restir_reservoir_update",
        "candidate_target_pdf",
        "candidate_stream_weight",
        "candidate_weight_sum",
        "float keep_ratio = capped_m / original_m;",
        "reservoir.weight_sum *= keep_ratio;",
    ] {
        assert!(
            common.contains(token),
            "Area ReSTIR common missing robust resampling token {token}"
        );
    }
}

#[test]
fn area_restir_temporal_reuses_history_in_current_pixel_measure() {
    let temporal = source("assets/shaders/passes/area_restir_temporal.slang");
    let common = source("assets/shaders/shared/area_restir_common.slang");

    for token in [
        "float2 history_sample = vpt_history_sample_from_motion(pixel, motion);",
        "float2 history_fraction",
        "static const int2 area_restir_history_tap_offsets[4]",
        "static const float AREA_RESTIR_TEMPORAL_MIN_TAP_WEIGHT",
        "float area_restir_history_tap_weight",
        "bool area_restir_history_tap_inside",
        "for (uint tap = 0u; tap < 4u; tap++)",
        "int2 tap_pixel_i = previous_base_pixel + area_restir_history_tap_offsets[tap]",
        "float tap_weight = area_restir_history_tap_weight(tap, history_fraction)",
        "tap_weight < AREA_RESTIR_TEMPORAL_MIN_TAP_WEIGHT",
        "float center_target_luma = area_restir_context_target_luma(center_context);",
        "float current_target_pdf = center_target_luma;",
        "float history_target_pdf = center_target_luma;",
        "float history_candidate_weight_sum",
        "area_restir_reservoir_update(",
        "area_restir_temporal_surface_compatible(center_context, previous_pixel)",
        "AREA_RESTIR_SAMPLE_HISTORY_VALID",
        "history.rejection_reason = 0u",
        "history.debug_flags",
    ] {
        assert!(
            temporal.contains(token),
            "temporal shader missing current-measure reuse token {token}"
        );
    }

    assert!(
        !temporal.contains("float history_weight = history.weight_sum;"),
        "temporal reuse must not copy previous-frame raw weight_sum into the current pixel measure"
    );
    assert!(
        temporal.contains("area_restir_reservoir_reuse_weight(history)")
            && common.contains("float area_restir_reservoir_reuse_weight")
            && common.contains("reservoir.selected_weight")
            && !temporal.contains("area_restir_reservoir_stream_weight(history)"),
        "temporal reuse must derive bounded reuse weight from selected_weight, not weight_sum/M"
    );
    assert!(
        temporal.contains("area_restir_reservoir_reuse_weight(current)")
            && !temporal.contains("current_target_pdf * float(max(current.sample_count_m, 1u))"),
        "temporal current reservoir must use the same bounded reuse-weight path as reused history"
    );
    assert!(
        !temporal.contains("int2 previous_pixel_i = int2(floor(history_sample));")
            && !temporal.contains("uint2 previous_pixel = uint2(previous_pixel_i);"),
        "temporal reuse must not collapse fractional reprojection to a single previous-pixel reservoir"
    );
    assert!(
        !temporal
            .contains("area_restir_replay_target_pdf(current_context(pixel), history.sample_state"),
        "temporal history taps must reuse the already-loaded center context instead of rereading current surface textures"
    );
    assert!(
        !temporal.contains("area_restir_replay_target_pdf(center_context, history.sample_state")
            && !temporal
                .contains("area_restir_replay_target_pdf(center_context, current.sample_state"),
        "valid current/history reservoirs can reuse cached center target luma instead of repeating replay-domain checks"
    );
    assert!(
        !temporal.contains("if (combined > 0.0 && history_weight >= current_weight)"),
        "temporal reuse must use weighted reservoir update instead of max-weight replacement"
    );
}

#[test]
fn area_restir_temporal_rejects_history_with_staged_previous_surface_reads() {
    let temporal = source("assets/shaders/passes/area_restir_temporal.slang");

    for token in [
        "bool area_restir_temporal_surface_compatible(",
        "float4 previous_position = previous_surface_position_depth[previous_pixel];",
        "if (center.position_depth.w < 0.0 || previous_position.w < 0.0)",
        "float position_delta = distance(center.position_depth.xyz, previous_position.xyz);",
        "float4 previous_albedo = previous_surface_albedo_material[previous_pixel];",
        "float3 previous_normal = normalize(previous_surface_normal_roughness[previous_pixel].xyz);",
    ] {
        assert!(
            temporal.contains(token),
            "temporal shader missing staged previous-surface token {token}"
        );
    }
    assert!(
        !temporal.contains("previous_context(previous_pixel)"),
        "temporal reuse should not construct a full previous context before cheap rejection tests"
    );
}

#[test]
fn area_restir_temporal_preserves_current_reservoir_when_history_rejects() {
    let temporal = source("assets/shaders/passes/area_restir_temporal.slang");

    assert!(
        temporal.contains("uint history_rejection_reason = AREA_RESTIR_TEMPORAL_REJECTION_HISTORY_INVALID;")
            && temporal.contains(
                "history_rejection_reason = AREA_RESTIR_TEMPORAL_REJECTION_MOTION_INVALID;"
            )
            && temporal.contains("if (!accepted_history) {")
            && temporal.contains("reservoir.rejection_reason = history_rejection_reason;")
            && temporal.contains("reservoir.confidence = min(max(current.confidence, reservoir.confidence) + 1.0, float(area_restir.history_length));")
            && temporal.contains("area_restir_finalize_reservoir(reservoir, area_restir.history_length);"),
        "temporal history rejection must preserve the already-merged current reservoir, avoid history-confidence growth, and continue through finalization"
    );
    assert!(
        !temporal.contains(
            "if (motion.w == 0.0) {\n        reservoir.rejection_reason = 1u;\n        output_reservoirs[index] = reservoir;\n        return;\n    }"
        ),
        "temporal history rejection must not early-return before finalizing the current reservoir"
    );
    assert!(
        !temporal.contains("if (motion.w == 0.0) {\n        reservoir.rejection_reason = 1u;\n        output_reservoirs[index] = reservoir;\n        return;\n    }"),
        "missing motion history must not early-return before finalizing the current reservoir"
    );
}

#[test]
fn area_restir_temporal_shifts_history_subpixel_into_current_pixel_domain() {
    let temporal = source("assets/shaders/passes/area_restir_temporal.slang");

    for token in [
        "float2 shifted_subpixel_uv = float2(tap_pixel_i - previous_base_pixel) + history.sample_state.subpixel_uv - history_fraction;",
        "if (any(shifted_subpixel_uv < 0.0) || any(shifted_subpixel_uv >= 1.0))",
        "history.sample_state.subpixel_uv = shifted_subpixel_uv;",
    ] {
        assert!(
            temporal.contains(token),
            "temporal shader must shift reused history subpixel samples into the current pixel domain; missing {token}"
        );
    }
}

#[test]
fn area_restir_reuse_passes_let_finalize_apply_history_length_weight_scaling() {
    let temporal = source("assets/shaders/passes/area_restir_temporal.slang");
    let spatial = source("assets/shaders/passes/area_restir_spatial.slang");

    for (name, shader) in [
        ("temporal", temporal.as_str()),
        ("spatial", spatial.as_str()),
    ] {
        assert!(
            !shader.contains(
                "reservoir.sample_count_m = min(reservoir.sample_count_m, area_restir.history_length);"
            ),
            "{name} pass must not pre-cap sample_count_m before area_restir_finalize_reservoir scales weight_sum"
        );
        assert!(
            shader
                .contains("area_restir_finalize_reservoir(reservoir, area_restir.history_length);"),
            "{name} pass must leave history-length capping to area_restir_finalize_reservoir"
        );
    }
}

#[test]
fn area_restir_spatial_reuses_neighbors_in_current_pixel_measure() {
    let spatial = source("assets/shaders/passes/area_restir_spatial.slang");
    let common = source("assets/shaders/shared/area_restir_common.slang");

    for token in [
        "area_restir_spatial_hash",
        "uint rotated_tap",
        "if (area_restir.enabled == 0u || area_restir.spatial_enabled == 0u || area_restir.spatial_sample_count == 0u",
        "float center_target_pdf = area_restir_context_target_luma(center);",
        "float neighbor_target_pdf = center_target_pdf;",
        "float neighbor_candidate_weight_sum",
        "area_restir_reservoir_update(",
        "reservoir.jacobian",
        "reservoir.debug_flags",
    ] {
        assert!(
            spatial.contains(token),
            "spatial shader missing current-measure reuse token {token}"
        );
    }

    assert!(
        !spatial.contains("reservoir.weight_sum += neighbor.weight_sum;"),
        "spatial reuse must not add raw neighbor weight_sum without current-domain conversion"
    );
    assert!(
        spatial.contains("area_restir_reservoir_reuse_weight(neighbor)")
            && common.contains("float area_restir_reservoir_reuse_weight")
            && common.contains("reservoir.selected_weight")
            && !spatial.contains("area_restir_reservoir_stream_weight(neighbor)"),
        "spatial reuse must derive bounded reuse weight from selected_weight, not weight_sum/M"
    );
    assert!(
        spatial.contains("area_restir_reservoir_reuse_weight(center_reservoir)")
            && !spatial
                .contains("center_target_pdf * float(max(center_reservoir.sample_count_m, 1u))"),
        "spatial center reservoir must use the same bounded reuse-weight path as reused neighbors"
    );
    assert!(
        !spatial.contains("if (neighbor.weight_sum > reservoir.weight_sum)"),
        "spatial reuse must not select neighbors by stale raw weight_sum"
    );
    assert!(
        !spatial.contains("area_restir_replay_target_pdf(center, neighbor.sample_state"),
        "spatial reuse should not recompute a neighbor replay target when the current-pixel target is already known"
    );
    assert!(
        !spatial.contains("area_restir_replay_target_pdf(center, center_reservoir.sample_state"),
        "valid center reservoirs can reuse cached center target luma instead of repeating replay-domain checks"
    );
}
