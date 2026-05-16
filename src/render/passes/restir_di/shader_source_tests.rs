use crate::assets::shader_reflect::{DescriptorBinding, DescriptorKind, ShaderReflection};

use super::RestirDiPass;

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
fn restir_di_descriptor_specs_match_shader_manifests() {
    let initial_specs = RestirDiPass::initial_descriptor_binding_specs();
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "ReSTIR-DI initial",
        &initial_specs,
        &shader_reflection("assets/shaders/passes/restir_di_initial.slang"),
    );

    let temporal_specs = RestirDiPass::temporal_descriptor_binding_specs();
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "ReSTIR-DI temporal",
        &temporal_specs,
        &shader_reflection("assets/shaders/passes/restir_di_temporal.slang"),
    );

    let spatial_specs = RestirDiPass::spatial_descriptor_binding_specs();
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "ReSTIR-DI spatial",
        &spatial_specs,
        &shader_reflection("assets/shaders/passes/restir_di_spatial.slang"),
    );
}

#[test]
fn restir_di_initial_shader_binding_manifest_matches_expected_resources() {
    assert_eq!(
        shader_bindings("assets/shaders/passes/restir_di_initial.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "restir"),
            binding(1, DescriptorKind::StorageBuffer, "direct_lights"),
            binding(2, DescriptorKind::StorageBuffer, "output_reservoirs"),
            binding(
                3,
                DescriptorKind::StorageImage,
                "current_surface_position_depth",
            ),
            binding(
                4,
                DescriptorKind::StorageImage,
                "current_surface_normal_roughness",
            ),
            binding(
                5,
                DescriptorKind::StorageImage,
                "current_surface_albedo_material",
            ),
        ]
    );
}

#[test]
fn restir_di_pass_owns_graph_registration_contract() {
    let pass = source("src/render/passes/restir_di.rs");
    let implementation = pass
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");
    assert!(implementation.contains("pub fn register_graph"));
    assert!(
        implementation
            .contains("builder.read_as(final_surface_writes[0], AccessKind::ComputeShaderRead)")
    );
    assert!(implementation.contains("vpt.update_restir_di_descriptors"));
}

#[test]
fn restir_di_shaders_declare_expected_entry_points_and_resources() {
    let initial = source("assets/shaders/passes/restir_di_initial.slang");
    let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
    let spatial = source("assets/shaders/passes/restir_di_spatial.slang");
    for shader in [&initial, &temporal, &spatial] {
        assert!(shader.contains("#include \"restir_di_common.slang\""));
        assert!(shader.contains("[shader(\"compute\")]"));
        assert!(shader.contains("RestirDiUniforms"));
        assert!(shader.contains("RestirDiReservoir"));
    }
    assert!(initial.contains("StructuredBuffer<DirectLight>"));
    assert!(temporal.contains("history_reservoirs"));
    assert!(spatial.contains("temporal_reservoirs"));
}

#[test]
fn restir_di_initial_writes_valid_candidate_reservoirs() {
    let initial = source("assets/shaders/passes/restir_di_initial.slang");

    assert!(initial.contains("restir.light_count == 0u"));
    assert!(initial.contains("DirectLight light = direct_lights[light_id];"));
    assert!(initial.contains("sample_direct_light_id(rand01(rng_state))"));
    assert!(initial.contains("reservoir.sample_light_id = light_id;"));
    assert!(initial.contains("reservoir.sample_flags ="));
    assert!(initial.contains("reservoir.sample_count_m += 1u;"));
    assert!(initial.contains("reservoir.weight_sum ="));
    assert!(initial.contains("reservoir.selected_weight ="));
    assert!(initial.contains("reservoir.sample_position_pdf ="));
    assert!(initial.contains("reservoir.sample_radiance ="));
}

#[test]
fn restir_di_initial_computes_selected_weight_from_total_candidate_weight() {
    let initial = source("assets/shaders/passes/restir_di_initial.slang");

    assert!(
        initial.contains("reservoir.selected_weight = min(")
            && initial.contains("reservoir.weight_sum /"),
        "direct lighting resolve consumes selected_weight, so initial RIS must compute it from the total candidate weight"
    );
    assert!(initial.contains("reservoir.target_pdf * float(reservoir.sample_count_m)"));
    assert!(
        initial.contains("reservoir.confidence = reservoir.weight_sum;"),
        "initial debug confidence should reflect the bounded reservoir weight, not the unclamped pre-firefly-clamp stream sum"
    );
}

#[test]
fn restir_di_target_pdf_matches_vpt_direct_light_measure() {
    let common = source("assets/shaders/shared/restir_di_common.slang");

    for token in [
        "restir_di_emissive_distance_attenuation",
        "restir_di_sun_target_pdf",
        "restir_di_emissive_target_pdf",
        "distance_sq",
        "return light_power * albedo_luma * light_term * attenuation;",
    ] {
        assert!(
            common.contains(token),
            "ReSTIR target PDF must include the same cosine and attenuation measure as VPT direct resolve: {token}"
        );
    }
    assert!(
        !common.contains("max(dot(surface_normal, light_dir), 0.05)"),
        "target PDF must not assign positive probability to back-facing direct-light samples that resolve to zero"
    );
}

#[test]
fn restir_di_initial_evaluates_sun_as_direction_not_world_position() {
    let initial = source("assets/shaders/passes/restir_di_initial.slang");

    assert!(
        initial.contains("restir_di_target_pdf_for_light_sample("),
        "initial sampling should use the shared light target PDF evaluator"
    );
    assert!(
        !initial.contains("light.position_radius.xyz - current_surface_position_depth[pixel].xyz"),
        "sun lights store a direction in position_radius.xyz, not a world-space point"
    );
}

#[test]
fn restir_di_initial_samples_emissive_area_points_instead_of_centroids() {
    let initial = source("assets/shaders/passes/restir_di_initial.slang");
    let common = source("assets/shaders/shared/restir_di_common.slang");

    for token in [
        "restir_di_sample_emissive_area_point",
        "restir_di_emissive_area_pdf",
        "light.position_radius.w",
        "reservoir.sample_position_pdf = float4(sampled_position, sample_pdf)",
        "target_pdf = restir_di_target_pdf_for_light_sample(",
    ] {
        assert!(
            initial.contains(token) || common.contains(token),
            "Area ReSTIR direct-light candidate generation missing token {token}"
        );
    }

    assert!(
        !initial.contains("reservoir.sample_position_pdf = float4(light.position_radius.xyz, 1.0 / float(restir.light_count))"),
        "emissive reservoirs must store a sampled area point and its PDF, not the centroid"
    );
}

#[test]
fn restir_di_initial_weights_candidates_by_light_selection_pdf() {
    let initial = source("assets/shaders/passes/restir_di_initial.slang");

    for token in [
        "uint sample_direct_light_id(float random01)",
        "while (lo < hi)",
        "direct_lights[mid].sampling.x",
        "float light_selection_pdf = max(light.sampling.y, 1.0e-8);",
        "float target_pdf = restir_di_target_pdf_for_light_sample(",
        "float candidate_weight = target_pdf / max(light_selection_pdf, 1.0e-6);",
        "float next_weight_sum = weight_sum + candidate_weight;",
        "reservoir.target_pdf = target_pdf;",
    ] {
        assert!(
            initial.contains(token),
            "ReSTIR-DI initial RIS must compensate the discrete light-selection proposal PDF: {token}"
        );
    }
    assert!(
        !initial.contains("float candidate_weight = restir_di_target_pdf_for_light_sample("),
        "target PDF alone underestimates direct lighting by the light proposal probability"
    );
    assert!(
        !initial.contains("hash_u32(rng_state ^ candidate * 747796405u) % restir.light_count"),
        "uniform light-id sampling causes high-variance direct lighting when the light table is large"
    );
    assert!(
        !initial.contains("while (lo + 1u < hi)"),
        "CDF lower_bound must compare the final one-element interval instead of returning light 0 early"
    );
    assert!(
        !initial.contains("candidate_weight = target_pdf / max(sample_pdf"),
        "the current emissive cluster target is not an area-density measure; area PDF amplification reintroduces bright fireflies"
    );
}

#[test]
fn restir_di_shaders_are_surface_aware() {
    let initial = source("assets/shaders/passes/restir_di_initial.slang");
    let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
    let spatial = source("assets/shaders/passes/restir_di_spatial.slang");
    let pass = source("src/render/passes/restir_di.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact_pass = pass.split_whitespace().collect::<String>();

    for shader in [&initial, &temporal, &spatial] {
        assert!(shader.contains("current_surface_position_depth"));
        assert!(shader.contains("current_surface_normal_roughness"));
        assert!(shader.contains("current_surface_albedo_material"));
    }

    assert!(initial.contains("output_reservoirs[index] = invalid_reservoir();"));
    assert!(initial.contains("surface_is_valid(index)"));
    assert!(
        temporal.contains("float4 surface_position_depth = current_surface_position_depth[pixel];")
            && spatial
                .contains("float4 surface_position_depth = current_surface_position_depth[pixel];"),
        "temporal and spatial reuse should validate with the cached center surface load"
    );
    assert!(initial.contains("current_surface_albedo_material"));
    assert!(temporal.contains("compatible_temporal_surface"));
    assert!(temporal.contains("uint previous_index"));
    assert!(temporal.contains("RestirDiReservoir history_reservoir"));
    assert!(temporal.contains("uint capped_history_m"));
    assert!(temporal.contains("history_target_pdf"));
    assert!(spatial.contains("compatible_spatial_surface"));
    assert!(spatial.contains("normal_dot"));
    assert!(spatial.contains("position_delta"));

    assert!(pass.contains("update_surface_descriptors"));
    assert!(pass.contains("VptSurfacePass"));
    assert!(pass.contains("DescriptorType::STORAGE_IMAGE"));
    assert!(pipeline.contains("restir_di.update_surface_descriptors"));
    assert!(pipeline.contains("restir_di.register_graph("));
    assert!(
        compact_pass
            .contains("builder.read_as(final_surface_writes[0],AccessKind::ComputeShaderRead)")
            || compact_pass.contains(
                "builder.read_as(final_surface_writes[0],AccessKind::ComputeShaderRead,)"
            )
    );
}

#[test]
fn restir_di_temporal_uses_explicit_history_surface_and_selected_frame_ring() {
    let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
    let pass = source("src/render/passes/restir_di.rs");
    let pass_impl = pass
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();

    assert!(temporal.contains("previous_surface_position_depth"));
    assert!(temporal.contains("previous_surface_normal_roughness"));
    assert!(temporal.contains("previous_surface_albedo_material"));
    assert!(temporal.contains("motion_history"));
    assert!(
        temporal.contains("float2 history_sample = vpt_history_sample_from_motion(pixel, motion);")
    );
    assert!(temporal.contains("position_delta"));
    assert!(
        !temporal.contains("dot(normalize(normal_roughness.xyz), normalize(normal_roughness.xyz))")
    );

    assert!(pass_impl.contains("selected_reservoirs"));
    assert!(pass_impl.contains("selected_current_buffer"));
    assert!(pass_impl.contains("selected_history_buffer"));
    assert!(pass_impl.contains("update_frame_descriptors"));
    assert!(pass_impl.contains("update_surface_descriptors"));
    assert!(
        !pass_impl.contains("record_history_update"),
        "ReSTIR-DI selected history must be maintained by a selected reservoir frame ring, not a fullscreen copy"
    );
    assert!(
        !pass_impl.contains("cmd_copy_buffer"),
        "ReSTIR-DI pass must not issue a per-frame history copy"
    );

    assert!(!pipeline.contains("restir_di_history_update"));
    assert!(
        pass_impl.contains("selected_current_resource")
            && pass_impl.contains("selected_history_resource")
            && compact_pipeline.contains(
                "vpt_restir_reads=Some((restir_graph.uniform_resource,restir_graph.selected_current_resource,));"
            ),
        "ReSTIR-DI graph must write the current selected slot and feed that exact resource to VPT"
    );
    assert!(
        !compact_pipeline
            .contains("builder.read_as(selected_current_dep,AccessKind::TransferRead)")
            && !pass_impl
                .contains("builder.read_as(selected_current_dep, AccessKind::TransferRead)"),
        "ReSTIR-DI current selected resource must not be copied through the transfer queue"
    );
}

#[test]
fn app_restir_di_dispatches_only_enabled_reuse_stages() {
    let app = source("src/app.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let pass = source("src/render/passes/restir_di.rs");
    let pass_impl = pass
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");
    let compact_pass = pass_impl.split_whitespace().collect::<String>();

    for token in [
        "letsettings=restir_di_effective_settings(settings,history_initialized);",
        "lettemporal_active=settings.temporal_enabled;",
        "letspatial_active=temporal_active&&settings.spatial_enabled;",
        "self.update_frame_descriptors(device,frame_slot,selected_history_buffer,selected_current_buffer,temporal_active,spatial_active,);",
        "letinitial_output_resource=iftemporal_active{initial_resource}else{selected_current_resource};",
        "lettemporal_dep=iftemporal_active{",
        "letselected_current_dep=ifspatial_active",
    ] {
        assert!(
            compact_pass.contains(token),
            "ReSTIR-DI pass graph must gate disabled reuse stages and keep selected-current descriptors in sync: {token}"
        );
    }
    assert!(
        pass_impl.contains("temporal_enabled: bool")
            && pass.contains("let initial_output = if temporal_enabled")
            && pass.contains("self.initial_stage.write_storage_descriptors_for_frame(\n            device,\n            frame_slot,\n            2,\n            &[initial_output],\n        );"),
        "when temporal reuse is disabled, ReSTIR-DI initial must write the selected current slot that VPT reads"
    );
    assert!(app.contains("self.vpt_pipeline.ensure_passes("));
    assert!(
        compact_pipeline.contains("restir_di.register_graph(")
            && compact_pipeline.contains(
                "inputs.restir_di_settings,self.frame_state.restir_di_history_initialized,"
            ),
        "VPT pipeline must delegate ReSTIR-DI graph registration with the runtime settings"
    );
    assert!(
        !compact_pass.contains(
            "letinitial_dep=initial_writes[0];lettemporal_writes=graph.add_pass(\"restir_di_temporal\""
        ),
        "ReSTIR-DI must not run temporal or graph-read selected history when temporal reuse is disabled"
    );
}

#[test]
fn restir_di_surface_descriptors_are_refreshed_only_on_resize_not_every_frame() {
    let app = source("src/app.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");

    assert!(!app.contains("restir_di.update_surface_descriptors(&device,vpt_surface);"));
    assert!(
        pipeline.contains("restir_di.update_surface_descriptors(&device, vpt_surface);")
            || pipeline.contains("restir_di.update_surface_descriptors(&device,vpt_surface);")
    );
    assert!(app.contains("self.vpt_pipeline.resize("));
}

#[test]
fn restir_di_temporal_combines_history_in_current_surface_measure_without_unbounded_weight_sum() {
    let temporal = source("assets/shaders/passes/restir_di_temporal.slang");

    assert!(temporal.contains("current_target_pdf"));
    assert!(temporal.contains("history_target_pdf"));
    assert!(
        temporal.contains("restir_di_reservoir_stream_weight"),
        "temporal reuse should convert current/history reservoirs into the current pixel's stream measure"
    );
    assert!(
        temporal.contains("restir_di_finalize_reservoir_on_surface"),
        "temporal reuse should rebuild selected_weight from capped combined M and current-surface weight"
    );
    assert!(
        !temporal
            .contains("float weight_sum = reservoir.weight_sum + history_reservoir.weight_sum"),
        "temporal reuse must not keep adding historical weight_sum after sample_count_m is capped"
    );
    assert!(
        !temporal.contains("reservoir.selected_weight = reservoir.weight_sum /"),
        "temporal reuse should use the shared finalizer so weight_sum and selected_weight stay normalized together"
    );
}

#[test]
fn restir_di_reuse_recomputes_selected_target_pdf_on_current_surface() {
    let common = source("assets/shaders/shared/restir_di_common.slang");
    let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
    let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

    assert!(common.contains("restir_di_target_pdf_for_reservoir"));
    assert!(common.contains("restir_di_finalize_reservoir_on_surface_with_target"));
    for shader in [temporal, spatial] {
        assert!(
            shader.contains("selected_target_pdf")
                || shader.contains("restir_di_finalize_reservoir_on_surface("),
            "reused reservoirs must be renormalized against the current pixel surface"
        );
    }
}

#[test]
fn restir_di_temporal_reuses_selected_target_pdf_without_finalizer_recompute() {
    let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
    let common = source("assets/shaders/shared/restir_di_common.slang");

    for token in [
        "float selected_target_pdf = current_target_pdf;",
        "selected_target_pdf = history_target_pdf;",
        "restir_di_finalize_reservoir_on_surface_with_target(",
        "selected_target_pdf,",
        "if (reservoir.sample_light_id != 0xffffffffu && reservoir.sample_count_m > 0u && reservoir.target_pdf <= 0.0)",
    ] {
        assert!(
            temporal.contains(token),
            "temporal shader missing selected-target reuse token {token}"
        );
    }
    assert!(
        common.contains("void restir_di_finalize_reservoir_on_surface_with_target("),
        "shared ReSTIR-DI common must expose finalizer that accepts a precomputed selected target"
    );
    assert!(
        !temporal.contains("restir_di_finalize_reservoir_on_surface(\n                reservoir,"),
        "temporal pass must not call the recomputing finalizer after it already computed current/history target PDFs"
    );
}

#[test]
fn restir_di_spatial_reuses_selected_target_pdf_without_finalizer_recompute() {
    let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

    for token in [
        "float selected_target_pdf = center_target_pdf;",
        "selected_target_pdf = neighbor_target_pdf;",
        "restir_di_finalize_reservoir_on_surface_with_target(",
        "selected_target_pdf,",
    ] {
        assert!(
            spatial.contains(token),
            "spatial shader missing selected-target reuse token {token}"
        );
    }
    assert!(
        !spatial.contains("restir_di_finalize_reservoir_on_surface(\n        reservoir,"),
        "spatial pass must not recompute the selected target PDF after it already evaluated the chosen candidate"
    );
}

#[test]
fn restir_di_reuse_shaders_validate_with_cached_center_position() {
    let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
    let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

    for (name, shader) in [
        ("temporal", temporal.as_str()),
        ("spatial", spatial.as_str()),
    ] {
        let compact = shader.split_whitespace().collect::<String>();
        assert!(
            compact.contains(
                "float4surface_position_depth=current_surface_position_depth[pixel];if(surface_position_depth.w<0.0){"
            ),
            "{name} shader should validate the center surface using the cached position/depth load"
        );
        assert!(
            !shader.contains("if (!surface_is_valid(index))"),
            "{name} shader must not read current_surface_position_depth once for surface_is_valid and again for the center surface"
        );
    }
}

#[test]
fn restir_di_reuse_shaders_cache_center_surface_reads() {
    let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
    let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

    assert!(
        temporal.contains(
            "compatible_temporal_surface(surface_position_depth, surface_normal_roughness, surface_albedo_material, previous_pixel)"
        ),
        "temporal reuse should pass cached center surface values into compatibility checks"
    );
    assert!(
        spatial.contains(
            "compatible_spatial_surface(surface_position_depth, surface_normal_roughness, surface_albedo_material, neighbor_index)"
        ),
        "spatial reuse should pass cached center surface values into compatibility checks"
    );
    assert!(
        !temporal.contains("compatible_temporal_surface(index, previous_pixel)")
            && !spatial.contains("compatible_spatial_surface(index, neighbor_index)"),
        "reuse shaders must not reload the center surface in each compatibility check"
    );
}

#[test]
fn restir_di_spatial_combines_neighbors_with_current_surface_stream_weights() {
    let common = source("assets/shaders/shared/restir_di_common.slang");
    let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

    assert!(common.contains("restir_di_reservoir_stream_weight"));
    assert!(common.contains("restir_di_finalize_reservoir_on_surface"));
    assert!(spatial.contains("neighbor_target_pdf"));
    assert!(spatial.contains("accepted_neighbor_m"));
    assert!(
        !spatial.contains("float neighbor_weight = max(neighbor.weight_sum"),
        "spatial reuse must not sample a neighbor using the neighbor surface's weight sum"
    );
    assert!(
        !spatial.contains("float weight_sum = reservoir.weight_sum + neighbor.weight_sum"),
        "spatial reuse must rebuild the combined stream weight in the current pixel's measure"
    );
}

#[test]
fn vpt_restir_direct_resolve_tests_visibility_before_applying_selected_weight() {
    let vpt = source("assets/shaders/passes/vpt.slang");

    assert!(vpt.contains("restir_di_light_visible_from_hit"));
    assert!(
        vpt.contains("voxel_shadow_occluded_from_hit(hit, shadow_dir, max_light_t)")
            && vpt.contains("return !shadow_occluded;"),
        "ReSTIR direct resolve must reject occluded reservoirs with a source-voxel-skipping any-hit visibility query before selected_weight creates bright leaks"
    );
    assert!(
        !vpt.contains("HitResult occluder = trace_primary_ray(shadow_ray"),
        "ReSTIR visibility should not use full material-returning traversal for shadow rays"
    );
    assert!(
        vpt.find("restir_di_light_visible_from_hit")
            .expect("VPT ReSTIR resolve should test visibility")
            < vpt
                .find("sample.radiance = albedo * reservoir.sample_radiance.rgb")
                .expect("VPT ReSTIR resolve should assign radiance"),
        "visibility must be tested before applying reservoir.selected_weight"
    );
}

#[test]
fn vpt_restir_direct_resolve_keeps_occluded_reservoirs_unselected_for_fallback() {
    let vpt = source("assets/shaders/passes/vpt.slang");

    assert!(
        vpt.find("if (!restir_di_light_visible_from_hit(hit, reservoir, scene))")
            .expect("VPT ReSTIR resolve should reject occluded light samples")
            < vpt
                .find("sample.selected_weight = selected_weight;")
                .expect("VPT ReSTIR resolve should only mark usable samples as selected"),
        "occluded ReSTIR-DI reservoirs must keep selected_weight at zero so VPT falls back to analytic direct light instead of writing black direct-light blocks"
    );
}

#[test]
fn restir_di_light_table_excludes_analytic_sun_and_preserves_emissive_sampling_power() {
    let app = source("src/app.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let restir = source("src/render/restir_di.rs");
    let vpt = source("assets/shaders/passes/vpt.slang");

    assert!(
        pipeline.contains("build_direct_lights_from_ucvh(ucvh"),
        "ReSTIR-DI direct-light setup should build a finite-emissive light table; the analytic sun is evaluated separately in VPT"
    );
    assert!(
        !app.contains("light.intensity.max_element().max(0.0)"),
        "collapsing the directional light to max_element before ReSTIR-DI lets one reservoir sample replace the deterministic RGB sun and creates over-bright direct-light blocks"
    );
    assert!(app.contains("let (sun_direction, sun_intensity) ="));
    assert!(
        restir.contains("pub fn build_direct_lights_from_ucvh")
            && restir.contains("ucvh: &Ucvh")
            && restir.contains("max_lights: usize"),
        "ReSTIR-DI direct-light builders must not include the analytic sun in the stochastic finite-light reservoir table"
    );
    assert!(
        vpt.contains("float3 analytic_direct = analytic_sun_direct(hit, scene, rng_state);")
            && vpt.contains("float3 direct_radiance = analytic_direct + direct.radiance;"),
        "VPT must always keep deterministic sun direct light and add finite-light ReSTIR-DI only when a reservoir is usable"
    );
}

#[test]
fn restir_di_shaders_reject_nonfinite_reservoir_weights_before_vpt_resolve() {
    let common = source("assets/shaders/shared/restir_di_common.slang");
    let vpt = source("assets/shaders/passes/vpt.slang");

    for token in [
        "static const float RESTIR_DI_MAX_SELECTED_WEIGHT",
        "bool restir_di_candidate_finite(float value)",
        "restir_di_bounded_selected_weight",
        "restir_di_is_valid_reservoir(RestirDiReservoir reservoir)",
        "restir_di_candidate_finite(reservoir.target_pdf)",
        "restir_di_candidate_finite(reservoir.weight_sum)",
        "restir_di_candidate_finite(reservoir.selected_weight)",
    ] {
        assert!(
            common.contains(token),
            "ReSTIR-DI common shader must bound invalid/overlarge reservoir state before reuse: {token}"
        );
    }
    assert!(
        vpt.contains("float selected_weight = restir_di_bounded_selected_weight(hit_reservoir);")
            && vpt.contains("if (selected_weight <= 0.0)")
            && vpt.contains("sample.selected_weight = selected_weight;")
            && vpt.contains("* selected_weight"),
        "VPT direct resolve must consume a finite bounded ReSTIR-DI weight evaluated on the actual VPT hit, not raw reservoir.selected_weight"
    );
    assert!(
        !vpt.contains("* reservoir.selected_weight"),
        "raw selected_weight can turn bad reservoirs into persistent white fireflies"
    );
}

#[test]
fn vpt_restir_di_falls_back_to_analytic_direct_when_reservoir_unusable() {
    let vpt = source("assets/shaders/passes/vpt.slang");

    assert!(
        vpt.contains(
            "float3 analytic_sun_direct(HitResult hit, SceneUniforms scene, inout uint rng_state)"
        ) && vpt.contains("float3 analytic_direct = analytic_sun_direct(hit, scene, rng_state);")
            && vpt.contains("float3 direct_radiance = analytic_direct + direct.radiance;")
            && vpt.contains(
                "float3 contribution = throughput * analytic_sun_direct(hit, scene, rng_state);"
            ),
        "invalid or incompatible ReSTIR-DI reservoirs should still keep analytic sun direct instead of turning first-bounce direct light into black blocks"
    );
}

#[test]
fn restir_di_spatial_disabled_is_exact_temporal_passthrough() {
    let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

    assert!(spatial.contains("if (restir.spatial_enabled == 0u)"));
    assert!(spatial.contains("output_reservoirs[index] = reservoir;"));
    assert!(
        spatial.find("if (restir.spatial_enabled == 0u)")
            < spatial.find("restir_di_reservoir_stream_weight")
    );
}

#[test]
fn restir_di_hot_reservoir_buffers_are_gpu_only() {
    let source = source("src/render/passes/restir_di.rs");
    let reservoir_fn = source
        .split("fn create_reservoir_buffer")
        .nth(1)
        .expect("create_reservoir_buffer should exist")
        .split("fn write_mapped")
        .next()
        .expect("create_reservoir_buffer body should end before mapped writes");

    assert!(reservoir_fn.contains("MemoryLocation::GpuOnly"));
    assert!(
        !reservoir_fn.contains("TRANSFER_SRC") && !reservoir_fn.contains("TRANSFER_DST"),
        "ReSTIR-DI reservoirs are no longer copied on the transfer queue"
    );
    assert!(
        !reservoir_fn.contains("write_mapped_slice"),
        "fullscreen reservoir buffers are GPU hot resources and should not require host-visible initialization"
    );
}

#[test]
fn app_profiles_each_restir_di_compute_stage_separately() {
    let pass = source("src/render/passes/restir_di.rs");
    let pass_impl = pass
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    for scope in [
        "GpuProfileScope::RestirDiInitial",
        "GpuProfileScope::RestirDiTemporal",
        "GpuProfileScope::RestirDiSpatial",
    ] {
        assert!(
            pass_impl.contains(scope),
            "{scope} should be emitted around the matching ReSTIR graph pass"
        );
    }
    assert!(
        pass_impl.find("GpuProfileScope::RestirDiInitial") < pass_impl.find("self.record_initial(")
    );
    assert!(
        pass_impl.find("GpuProfileScope::RestirDiTemporal")
            < pass_impl.find("self.record_temporal(")
    );
    assert!(
        pass_impl.find("GpuProfileScope::RestirDiSpatial") < pass_impl.find("self.record_spatial(")
    );
}

#[test]
fn app_skips_spatial_passthrough_when_spatial_reuse_is_disabled() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let pass = source("src/render/passes/restir_di.rs");
    let pass_impl = pass
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");
    let compact_pass = pass_impl.split_whitespace().collect::<String>();

    assert!(
        compact_pass.contains("ifspatial_active{"),
        "the spatial graph pass should be created only when spatial reuse is enabled"
    );
    assert!(
        compact_pass.contains("lettemporal_output_resource=ifspatial_active{temporal_resource}else{selected_current_resource};")
            && compact_pass.contains("letselected_current_dep=ifspatial_active{")
            && compact_pass.contains("}else{temporal_dep};")
            && compact_pipeline.contains(
                "vpt_restir_reads=Some((restir_graph.uniform_resource,restir_graph.selected_current_resource,));"
            ),
        "spatial-disabled ReSTIR should write the temporal output directly into the current selected slot"
    );
}

#[test]
fn restir_di_spatial_samples_multiple_neighbor_offsets_instead_of_one_right_neighbor() {
    let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

    assert!(spatial.contains("spatial_sample_count"));
    assert!(spatial.contains("int2 spatial_offsets"));
    assert!(spatial.contains("sample < min(restir.spatial_sample_count, 8u)"));
    assert!(spatial.contains("neighbor_offset"));
    assert!(
        !spatial.contains("uint neighbor_index = index + 1u;"),
        "spatial reuse should not propagate only to the right-hand neighbor"
    );
}

#[test]
fn restir_di_pass_does_not_issue_pass_local_barriers() {
    let implementation = source("src/render/passes/restir_di.rs");
    let implementation = implementation
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");
    assert!(!implementation.contains("cmd_pipeline_barrier"));
    assert!(!implementation.contains("ImageMemoryBarrier"));
    assert!(!implementation.contains("BufferMemoryBarrier"));
}

#[test]
fn restir_di_pass_cleans_up_failed_construction_paths() {
    let implementation = source("src/render/passes/restir_di.rs");
    let implementation = implementation
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    assert!(implementation.contains("buffers.destroy(device, allocator);"));
    assert!(implementation.contains("initial_stage.destroy(device);"));
    assert!(implementation.contains("temporal_stage.destroy(device);"));
    assert!(implementation.contains("descriptor_pool.destroy(device);"));
    assert!(implementation.contains("device.destroy_descriptor_set_layout(descriptor_set_layout"));
}
