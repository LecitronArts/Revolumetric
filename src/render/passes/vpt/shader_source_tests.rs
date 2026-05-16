use crate::assets::shader_reflect::{DescriptorBinding, DescriptorKind, ShaderReflection};

use super::VptPass;

fn normalized_source(path_source: &str) -> String {
    crate::render::source_checks::normalize(path_source)
}

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
fn vpt_descriptor_specs_match_shader_manifest() {
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "VPT trace",
        &VptPass::descriptor_binding_specs(),
        &shader_reflection("assets/shaders/passes/vpt.slang"),
    );
}

#[test]
fn vpt_shader_binding_manifest_matches_expected_resources() {
    assert_eq!(
        shader_bindings("assets/shaders/passes/vpt.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "scene_ubo"),
            binding(1, DescriptorKind::StorageImage, "noisy_radiance_image"),
            binding(2, DescriptorKind::StorageBuffer, "ucvh_config"),
            binding(3, DescriptorKind::StorageBuffer, "hierarchy_l0"),
            binding(4, DescriptorKind::StorageBuffer, "hierarchy_l1"),
            binding(5, DescriptorKind::StorageBuffer, "hierarchy_l2"),
            binding(6, DescriptorKind::StorageBuffer, "hierarchy_l3"),
            binding(7, DescriptorKind::StorageBuffer, "hierarchy_l4"),
            binding(8, DescriptorKind::StorageBuffer, "brick_occupancy"),
            binding(9, DescriptorKind::StorageBuffer, "brick_materials"),
            binding(10, DescriptorKind::UniformBuffer, "restir"),
            binding(11, DescriptorKind::StorageBuffer, "restir_reservoirs"),
            binding(12, DescriptorKind::StorageImage, "noisy_moments_image"),
            binding(13, DescriptorKind::UniformBuffer, "area_restir"),
            binding(14, DescriptorKind::StorageBuffer, "area_restir_reservoirs"),
        ]
    );
}

#[test]
fn vpt_surface_shader_binding_manifest_matches_expected_resources() {
    assert_eq!(
        shader_bindings("assets/shaders/passes/vpt_surface.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "scene_ubo"),
            binding(1, DescriptorKind::StorageImage, "surface_position_depth"),
            binding(2, DescriptorKind::StorageImage, "surface_normal_roughness"),
            binding(3, DescriptorKind::StorageImage, "surface_albedo_material"),
            binding(4, DescriptorKind::StorageImage, "motion_history"),
            binding(5, DescriptorKind::StorageBuffer, "ucvh_config"),
            binding(6, DescriptorKind::StorageBuffer, "hierarchy_l0"),
            binding(7, DescriptorKind::StorageBuffer, "hierarchy_l1"),
            binding(8, DescriptorKind::StorageBuffer, "hierarchy_l2"),
            binding(9, DescriptorKind::StorageBuffer, "hierarchy_l3"),
            binding(10, DescriptorKind::StorageBuffer, "hierarchy_l4"),
            binding(11, DescriptorKind::StorageBuffer, "brick_occupancy"),
            binding(12, DescriptorKind::StorageBuffer, "brick_materials"),
            binding(13, DescriptorKind::UniformBuffer, "vpt_history"),
            binding(14, DescriptorKind::UniformBuffer, "area_restir"),
            binding(15, DescriptorKind::StorageBuffer, "area_restir_reservoirs"),
        ]
    );
}

#[test]
fn vpt_shader_declares_stochastic_accumulating_reference_path() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let primary_sample_common = normalized_source(include_str!(
        "../../../../assets/shaders/shared/vpt_primary_sample_common.slang"
    ));

    assert!(source.contains("RWTexture2D<float4> noisy_radiance_image;"));
    assert!(source.contains("RWTexture2D<float4> noisy_moments_image;"));
    assert!(source.contains("hash_u32("));
    assert!(source.contains("#include \"vpt_primary_sample_common.slang\""));
    assert!(primary_sample_common.contains("scene.vpt_sample_index"));
    assert!(source.contains("scene.vpt_max_bounces"));
    assert!(source.contains("trace_primary_ray("));
    assert!(source.contains("VptTraceSample sample = trace_path("));
    assert!(source.contains("float3 sample_radiance = sample.radiance;"));
    assert!(source.contains("float luminance = dot(sample_radiance"));
    assert!(source.contains("noisy_radiance_image[tid.xy] = float4(sample_radiance, 1.0);"));
}

#[test]
fn vpt_phase4_writes_noisy_radiance_and_moments_not_progressive_accumulation() {
    let shader = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let rust = std::fs::read_to_string("src/render/passes/vpt.rs").expect("vpt source is readable");

    assert!(
        shader.contains("RWTexture2D<float4> noisy_radiance_image;"),
        "VPT shader must write current-frame noisy HDR radiance"
    );
    assert!(
        shader.contains("RWTexture2D<float4> noisy_moments_image;"),
        "VPT shader must write current-frame luminance moments"
    );
    assert!(
        shader.contains("float luminance = dot(sample_radiance"),
        "VPT shader must compute luminance moments from the noisy sample"
    );
    assert!(
        !shader.contains("lerp(previous, sample_radiance, 1.0 / sample_count)"),
        "VPT must not use progressive accumulation as the interactive quality mechanism"
    );
    assert!(
        rust.contains("pub noisy_radiance_image: GpuImage")
            && rust.contains("pub noisy_moments_image: GpuImage"),
        "VptPass must own separate noisy radiance and moments images"
    );
    assert!(
        rust.contains("vpt_noisy_radiance") && rust.contains("vpt_noisy_moments"),
        "VPT image names must be explicit for captures and graph debugging"
    );
}

#[test]
fn vpt_temporal_pass_declares_reprojection_and_history_contract() {
    let rust = std::fs::read_to_string("src/render/passes/vpt_temporal.rs")
        .expect("VPT temporal pass source should exist");
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");
    let pipeline = source("src/render/vpt_pipeline.rs");

    for token in [
        "pub struct VptTemporalPass",
        "accumulated_radiance",
        "accumulated_moments_history",
        "history_length",
        "previous_accumulated_radiance",
        "previous_accumulated_moments_history",
        "pub fn record",
        "pub fn record_history_update",
    ] {
        assert!(rust.contains(token), "VPT temporal Rust missing {token}");
    }

    for token in [
        "RWTexture2D<float4> noisy_radiance_image",
        "RWTexture2D<float4> noisy_moments_image",
        "RWTexture2D<float4> surface_position_depth",
        "RWTexture2D<float4> previous_surface_position_depth",
        "RWTexture2D<float4> motion_history",
        "RWTexture2D<float4> accumulated_radiance_image",
        "RWTexture2D<float4> accumulated_moments_history_image",
        "compatible_history",
        "history_length",
        "bounded_alpha",
    ] {
        assert!(
            shader.contains(token),
            "VPT temporal shader missing {token}"
        );
    }

    assert!(pipeline.contains("vpt.register_graph("));
    assert!(pipeline.contains("vpt_temporal.register_graph("));
    assert!(pipeline.contains("vpt_temporal.register_history_update_graph("));
    assert!(pipeline.contains("postprocess.register_graph("));

    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT graph registration should exist");
    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("VPT temporal graph registration should exist");
    let history_update_idx = pipeline
        .find("vpt_temporal.register_history_update_graph(")
        .expect("VPT temporal history update graph registration should exist");
    let postprocess_idx = pipeline
        .find("postprocess.register_graph(")
        .expect("postprocess graph registration should exist");

    assert!(vpt_idx < temporal_idx);
    assert!(temporal_idx < history_update_idx);
    assert!(history_update_idx < postprocess_idx);
}

#[test]
fn vpt_atrous_pass_declares_svgf_edge_aware_filter_contract() {
    let rust = std::fs::read_to_string("src/render/passes/vpt_atrous.rs")
        .expect("VPT atrous pass source should exist");
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_atrous.slang")
        .expect("VPT atrous shader should exist");

    for token in [
        "pub struct VptAtrousPass",
        "pub filtered_radiance",
        "ping_radiance",
        "pong_radiance",
        "pub fn record",
        "pub fn output_image",
        "vpt_atrous_filtered",
        "vpt_atrous_ping",
        "vpt_atrous_pong",
    ] {
        assert!(rust.contains(token), "VPT atrous Rust missing {token}");
    }

    for token in [
        "RWTexture2D<float4> input_radiance_image",
        "RWTexture2D<float4> moments_history_image",
        "RWTexture2D<float4> surface_position_depth",
        "RWTexture2D<float4> surface_normal_roughness",
        "RWTexture2D<float4> surface_albedo_material",
        "RWTexture2D<float4> filtered_radiance_image",
        "atrous_step_width",
        "normal_weight",
        "depth_weight",
        "safe_normalize",
        "albedo_weight",
        "center_albedo_material.rgb",
        "material_weight",
        "variance_weight",
        "spatial_luminance_variance",
        "effective_luminance_variance",
        "history_length",
        "center_moments_history.z < 4.0",
        "vpt_atrous.iteration_index == 0u",
        "neighbor_moments_history",
        "clamp_radiance_to_luminance_interval",
        "struct AtrousLuminanceStats",
        "gather_atrous_luminance_stats",
        "atrous_luminance_limit",
        "clamp_center_radiance_to_spatial_interval",
        "float3 center_filtered_radiance",
        "scene.denoiser_atrous_iterations",
        "DENOISER_FLAG_ENABLED",
        "scene.vpt_debug_view != VPT_DEBUG_VIEW_FINAL",
    ] {
        assert!(shader.contains(token), "VPT atrous shader missing {token}");
    }
}

#[test]
fn app_routes_temporal_radiance_through_vpt_atrous_before_postprocess() {
    let app = source("src/app.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let atrous_pass = source("src/render/passes/vpt_atrous.rs");
    let postprocess_pass = source("src/render/passes/postprocess.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let compact_atrous = atrous_pass.split_whitespace().collect::<String>();
    let compact_postprocess = postprocess_pass.split_whitespace().collect::<String>();

    assert!(pipeline.contains("pub vpt_atrous_pass: Option<VptAtrousPass>"));
    assert!(pipeline.contains("pub postprocess_pass: Option<PostprocessPass>"));
    assert!(pipeline.contains("VptAtrousPass::new"));
    assert!(pipeline.contains("VptAtrousPassCreateInfo"));
    assert!(pipeline.contains("PostprocessPass::new"));
    assert!(app.contains("self.vpt_pipeline.ensure_passes("));
    assert!(app.contains("self.vpt_pipeline.record_and_execute_frame("));
    assert!(pipeline.contains("vpt_temporal.register_graph("));
    assert!(pipeline.contains("vpt_atrous.register_graph("));
    assert!(pipeline.contains("postprocess.register_graph("));
    assert!(pipeline.contains("vpt_atrous.output_image()"));
    assert!(atrous_pass.contains("graph.add_pass(\"vpt_atrous\""));
    assert!(atrous_pass.contains("GpuProfileScope::VptAtrous"));
    assert!(postprocess_pass.contains("graph.add_pass(\"postprocess\""));
    assert!(postprocess_pass.contains("GpuProfileScope::Postprocess"));

    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("VPT temporal graph registration should exist");
    let atrous_idx = pipeline
        .find("vpt_atrous.register_graph(")
        .expect("VPT atrous graph registration should exist");
    let postprocess_idx = pipeline
        .find("postprocess.register_graph(")
        .expect("postprocess graph registration should exist");

    assert!(temporal_idx < atrous_idx);
    assert!(atrous_idx < postprocess_idx);
    assert!(compact_atrous.contains("letmutatrous_input_dep=temporal_radiance;"));
    assert!(compact_atrous.contains("letmutatrous_ping_dep=atrous_ping_resource;"));
    assert!(compact_atrous.contains("letmutatrous_pong_dep=atrous_pong_resource;"));
    assert!(compact_atrous.contains("letoutput_is_final=iteration_index+1==atrous_pass_count;"));
    assert!(
        compact_atrous
            .contains("letoutput_is_ping=!output_is_final&&iteration_index.is_multiple_of(2);")
    );
    assert!(
        compact_atrous.contains("builder.read_as(atrous_input_dep,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_atrous.contains("builder.read_as(temporal_moments,AccessKind::ComputeShaderRead")
    );
    assert!(compact_atrous.contains("letatrous_filtered_resource=graph.import_image_with_access("));
    assert!(
        compact_atrous
            .contains("builder.write_as(atrous_output_resource,AccessKind::ComputeShaderWrite")
    );
    assert!(compact_pipeline.contains("letatrous_filtered_dep=atrous_outputs.filtered_radiance;"));
    assert!(compact_atrous.contains("atrous_ping_dep=atrous_writes[0];"));
    assert!(compact_atrous.contains("atrous_pong_dep=atrous_writes[0];"));
    assert!(compact_postprocess.contains("self.update_input_image(device,hdr_image,frame_slot);"));
    assert!(
        compact_postprocess
            .contains("builder.read_as(input_radiance,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_postprocess.contains(
            "builder.write_as(postprocess_output_resource,AccessKind::ComputeShaderWrite"
        )
    );
    assert!(compact_pipeline.contains("vpt_atrous.output_image()"));
}

#[test]
fn vpt_pipeline_uses_pass_owned_graph_registration_for_vpt_stages() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let pipeline_impl = pipeline
        .split("#[cfg(test)]")
        .next()
        .expect("pipeline implementation section should exist");
    let vpt_surface = source("src/render/passes/vpt_surface.rs");
    let vpt_trace = source("src/render/passes/vpt.rs");
    let vpt_temporal = source("src/render/passes/vpt_temporal.rs");
    let vpt_atrous = source("src/render/passes/vpt_atrous.rs");
    let postprocess = source("src/render/passes/postprocess.rs");

    for (name, implementation, api) in [
        (
            "VPT surface",
            vpt_surface.as_str(),
            "pub fn register_bootstrap_graph",
        ),
        ("VPT trace", vpt_trace.as_str(), "pub fn register_graph"),
        (
            "VPT temporal",
            vpt_temporal.as_str(),
            "pub fn register_graph",
        ),
        (
            "VPT temporal history",
            vpt_temporal.as_str(),
            "pub fn register_history_update_graph",
        ),
        ("VPT atrous", vpt_atrous.as_str(), "pub fn register_graph"),
        ("postprocess", postprocess.as_str(), "pub fn register_graph"),
    ] {
        assert!(
            implementation.contains(api),
            "{name} pass must expose pass-owned graph registration API {api}"
        );
    }

    for call in [
        "vpt_surface.register_bootstrap_graph(",
        "vpt.register_graph(",
        "vpt_temporal.register_graph(",
        "vpt_atrous.register_graph(",
        "vpt_temporal.register_history_update_graph(",
        "postprocess.register_graph(",
    ] {
        assert!(
            pipeline_impl.contains(call),
            "VPT pipeline must use pass-owned graph registration call {call}"
        );
    }

    for forbidden in [
        "graph.add_pass(\"vpt_surface_bootstrap\"",
        "graph.add_pass(\"vpt\"",
        "graph.add_pass(\"vpt_temporal\"",
        "graph.add_pass(\"vpt_atrous\"",
        "graph.add_pass(\"vpt_surface_history_update\"",
        "graph.add_pass(\"postprocess\"",
    ] {
        assert!(
            !pipeline_impl.contains(forbidden),
            "VPT pipeline must not manually register pass graph node {forbidden}"
        );
    }
}

#[test]
fn vpt_temporal_seeds_valid_history_when_no_previous_history_is_accepted() {
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    assert!(
        shader.contains("bool current_surface_valid = surface_position_depth[pixel].w >= 0.0;")
            && shader.contains("float history_valid = current_surface_valid ? 1.0 : 0.0;"),
        "VPT temporal must seed valid history from valid current surface samples, otherwise reuse can never start"
    );
    assert!(
        shader.contains("float history_length = current_surface_valid ? 1.0 : 0.0;"),
        "VPT temporal must seed history length only for valid current surface samples"
    );
    assert!(
        shader.contains("history_valid = 1.0;") && shader.contains("history.history_length > 0.0"),
        "accepted previous history should keep validity while increasing history length"
    );
    assert!(
        shader.contains("scene.vpt_debug_view == VPT_DEBUG_VIEW_HISTORY_VALID")
            && shader.contains("accumulated = float3(history_reused);"),
        "history_valid debug view must expose where previous-frame temporal reuse was accepted"
    );
}

#[test]
fn app_does_not_rewrite_all_vpt_temporal_descriptors_per_frame() {
    let app = std::fs::read_to_string("src/app.rs")
        .expect("app source should be readable for VPT temporal descriptor lifetime test");
    let render_loop_start = app
        .find("let frame = renderer.begin_frame()?")
        .expect("render loop should begin a Vulkan frame");
    let render_loop_end = app[render_loop_start..]
        .find("renderer.end_frame(frame)?")
        .map(|offset| render_loop_start + offset)
        .expect("render loop should end the Vulkan frame");
    let frame_body = &app[render_loop_start..render_loop_end];

    assert!(
        !frame_body.contains("vpt_temporal.update_input_images"),
        "VPT temporal descriptor rebinding must not rewrite all frame-slot descriptor sets from the per-frame render path"
    );
}

#[test]
fn app_declares_persistent_vpt_image_layouts_from_previous_frame_final_access() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact = pipeline.split_whitespace().collect::<String>();
    let temporal = source("src/render/passes/vpt_temporal.rs");
    let postprocess = source("src/render/passes/postprocess.rs");
    let compact_temporal = temporal.split_whitespace().collect::<String>();
    let compact_postprocess = postprocess.split_whitespace().collect::<String>();

    assert!(
        compact_temporal
            .contains("lettemporal_initial_access=ifhistory_initialized{AccessKind::TransferRead"),
        "temporal radiance must be imported from the previous frame's history-copy read layout"
    );
    assert!(
        compact_temporal.contains(
            "letprevious_temporal_access=ifhistory_initialized{AccessKind::TransferWrite"
        ),
        "temporal moments must be imported from the previous frame's history-copy read layout"
    );
    assert!(
        pipeline.contains("pub postprocess_output_initialized: bool")
            && pipeline.contains("self.frame_state.reset_for_resize_or_camera_cut();")
            && compact.contains("vpt_temporal.register_graph(")
            && compact.contains("postprocess.register_graph("),
        "pipeline must hand persistent temporal/postprocess history to pass-owned graph registration"
    );
    assert!(
        compact_postprocess
            .contains("letoutput_initial_access=ifoutput_initialized{AccessKind::TransferRead"),
        "postprocess output must be imported from the previous frame's history-copy read layout"
    );
    assert!(
        compact_postprocess
            .contains("letpostprocess_output_resource=graph.import_image_with_access("),
        "postprocess output is a persistent image and must be imported with its tracked previous-frame layout"
    );
}

#[test]
fn app_resets_vpt_temporal_state_on_resize_and_key_changes() {
    let source = std::fs::read_to_string("src/app.rs")
        .expect("app source should be readable for VPT reset test");
    let pipeline = std::fs::read_to_string("src/render/vpt_pipeline.rs")
        .expect("VPT pipeline source should be readable for VPT reset test");
    let temporal = std::fs::read_to_string("src/render/passes/vpt_temporal.rs")
        .expect("VPT temporal source should be readable for VPT reset test");
    let compact = pipeline.split_whitespace().collect::<String>();

    assert!(source.contains("self.vpt_pipeline.resize("));
    assert!(
        source
            .contains("fn resize_render_passes(&mut self, width: u32, height: u32) -> Result<()>")
    );
    assert!(pipeline.contains("self.frame_state.reset_for_resize_or_camera_cut();"));
    assert!(compact.contains("self.frame_state.vpt_temporal_history_initialized"));
    assert!(pipeline.contains("camera.fov_y_radians.to_bits()"));
    assert!(pipeline.contains("frame.swapchain_extent.width"));
    assert!(pipeline.contains("inputs.lighting_settings.vpt_max_bounces"));
    assert!(pipeline.contains("initialized postprocess pass from VPT output"));
    assert!(pipeline.contains("skipping VPT frame until required passes are initialized"));
    assert!(pipeline.contains("graph.has_final_access(AccessKind::Present)"));
    assert!(pipeline.contains("add_swapchain_clear_present_pass"));
    assert!(
        source
            .contains("fn resize_render_passes(&mut self, width: u32, height: u32) -> Result<()>")
    );
    assert!(!source.contains("graph.add_pass(\"primary_ray\""));
    assert!(!source.contains("primary_ray_writes"));
    assert!(pipeline.contains("graph.import_image_with_access("));
    assert!(pipeline.contains("vpt_temporal.register_graph("));
    assert!(temporal.contains("graph.add_pass(\"vpt_temporal\""));
    assert!(temporal.contains("AccessKind::ComputeShaderWrite"));
}

#[test]
fn app_keeps_vpt_first_use_sample_zero_until_noisy_radiance_is_written() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let compact = pipeline.split_whitespace().collect::<String>();
    let compact_pass = vpt_pass.split_whitespace().collect::<String>();

    assert!(
        compact.contains(
            "letscene_vpt_sample_index=ifself.frame_state.vpt_accumulation_needs_init{0}else{self.frame_state.vpt_sample_index};"
        ) && compact.contains("vpt_sample_index:scene_vpt_sample_index"),
        "scene UBO must see sample 0 while the accumulation image is still first-use"
    );
    assert!(
        compact_pass
            .contains("letnoisy_initial_access=ifaccumulation_needs_init{AccessKind::Undefined}else{AccessKind::ComputeShaderRead};"),
        "first-use VPT noisy images must start from Undefined even if internal sample state was advanced"
    );
    assert!(
        pipeline.contains("self.frame_state.last_vpt_camera_key = None;"),
        "skipped VPT frames must not advance reusable accumulation state"
    );
}

#[test]
fn app_supports_frame_limited_runtime_smoke_validation() {
    let source = std::fs::read_to_string("src/app.rs")
        .expect("app source should be readable for runtime smoke validation test");

    assert!(source.contains("REVOLUMETRIC_EXIT_AFTER_FRAMES"));
    assert!(source.contains("exit_after_frames: parse_exit_after_frames()"));
    assert!(source.contains("if let Some(limit) = self.exit_after_frames"));
    assert!(source.contains("event_loop.exit();"));
}

#[test]
fn app_registers_vpt_graph_without_primary_ray_pass() {
    let app = source("src/app.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");

    assert!(app.contains("self.vpt_pipeline.record_and_execute_frame("));
    assert!(!app.contains("graph.add_pass(\"vpt\""));
    assert!(pipeline.contains("vpt.register_graph("));
    assert!(pipeline.contains("postprocess.register_graph("));
    assert!(!pipeline.contains("graph.add_pass(\"primary_ray\""));
    assert!(vpt_pass.contains("graph.add_pass(\"vpt\""));
    assert!(!vpt_pass.contains("graph.add_pass(\"primary_ray\""));
    assert!(!app.contains("let use_vpt ="));
}

#[test]
fn active_runtime_has_no_vct_primary_or_lighting_path() {
    let app_source = std::fs::read_to_string("src/app.rs")
        .expect("app source should be readable for VPT-only runtime test");
    let passes_mod = std::fs::read_to_string("src/render/passes/mod.rs")
        .expect("passes module source should be readable for VPT-only runtime test");
    let readme = std::fs::read_to_string("README.md")
        .expect("README should be readable for VPT-only runtime test");
    let active_source = format!(
        "{}\n{}\n{}",
        app_source
            .split("#[cfg(test)]")
            .next()
            .unwrap_or(&app_source),
        passes_mod,
        readme,
    );

    for forbidden in [
        concat!("Lighting", "Pass"),
        concat!("PrimaryRay", "Pass"),
        concat!("graph.add_pass(\"", "lighting", "\""),
        concat!("graph.add_pass(\"", "primary_ray", "\""),
        concat!("lighting", ".spv"),
        concat!("primary_ray", ".spv"),
        "pub mod lighting;",
        "pub mod primary_ray;",
        concat!("REVOLUMETRIC_", "VCT"),
        concat!("REVOLUMETRIC_RENDER_MODE=", "vct"),
        concat!("Default is `", "vct", "`"),
        concat!("VCT", "-first"),
        concat!("vct", "_common", ".slang"),
    ] {
        assert!(
            !active_source.contains(forbidden),
            "forbidden active VCT token remains: {forbidden}"
        );
    }
    for deleted_file in [
        "src/render/passes/lighting.rs",
        "src/render/passes/primary_ray.rs",
        "assets/shaders/passes/lighting.slang",
        "assets/shaders/passes/primary_ray.slang",
        concat!("assets/shaders/shared/", "vct", "_common", ".slang"),
    ] {
        assert!(
            !std::path::Path::new(deleted_file).exists(),
            "deleted VCT artifact still exists: {deleted_file}"
        );
    }
}

#[test]
fn vpt_history_abi_declares_surface_and_reprojection_contract() {
    let rust = std::fs::read_to_string("src/render/vpt_history.rs")
        .expect("VPT history ABI source should exist");
    let slang = std::fs::read_to_string("assets/shaders/shared/vpt_history_common.slang")
        .expect("VPT history Slang ABI should exist");

    for token in [
        "GpuVptHistoryUniforms",
        "current_view_proj",
        "previous_view_proj",
        "current_resolution",
        "previous_resolution",
        "current_jitter",
        "previous_jitter",
        "history_reset_generation",
        "VPT_HISTORY_FLAG_CAMERA_CUT",
        "VPT_HISTORY_FLAG_RESIZE",
        "VPT_HISTORY_FLAG_SCENE_INVALIDATED",
        "VPT_HISTORY_FLAG_LIGHTS_INVALIDATED",
        "GpuVptSurfacePixel",
        "linear_depth",
        "material_id",
        "history_confidence",
    ] {
        assert!(rust.contains(token), "Rust VPT history ABI missing {token}");
    }
    for token in [
        "struct VptHistoryUniforms",
        "float4x4 current_view_proj",
        "float4x4 previous_view_proj",
        "uint2 current_resolution",
        "uint2 previous_resolution",
        "float2 current_jitter",
        "float2 previous_jitter",
        "uint history_reset_generation",
        "static const uint VPT_HISTORY_FLAG_CAMERA_CUT",
        "static const uint VPT_HISTORY_FLAG_LIGHTS_INVALIDATED",
        "float2 vpt_previous_pixel_center_from_motion",
        "float2 vpt_history_sample_from_motion",
        "struct VptSurfacePixel",
    ] {
        assert!(
            slang.contains(token),
            "Slang VPT history ABI missing {token}"
        );
    }
}

#[test]
fn vpt_surface_pass_declares_owned_surface_outputs() {
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let rust = std::fs::read_to_string("src/render/passes/vpt_surface.rs")
        .expect("VPT surface pass source should exist");

    for token in [
        "RWTexture2D<float4> surface_position_depth",
        "RWTexture2D<float4> surface_normal_roughness",
        "RWTexture2D<float4> surface_albedo_material",
        "RWTexture2D<float4> motion_history",
        "trace_primary_ray(",
        "position_depth = float4(0.0, 0.0, 0.0, -1.0)",
        "voxel_material(hit.cell)",
        "surface_position_depth[pixel] = float4(hit.position, hit.t);",
    ] {
        assert!(shader.contains(token), "VPT surface shader missing {token}");
    }
    assert!(
        !shader.contains("distance(origin, hit.position)"),
        "VPT surface pass should use DDA hit.t instead of recomputing hit distance"
    );

    for token in [
        "pub struct VptSurfacePass",
        "surface_position_depth",
        "surface_normal_roughness",
        "surface_albedo_material",
        "motion_history",
        "pub fn resize_images",
        "pub fn record",
        "pub fn destroy",
    ] {
        assert!(rust.contains(token), "VPT surface pass missing {token}");
    }
    assert!(!rust.contains("cmd_pipeline_barrier"));
    assert!(!rust.contains("ImageMemoryBarrier"));
}

#[test]
fn vpt_surface_motion_uses_history_uniform_reprojection() {
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let rust = std::fs::read_to_string("src/render/passes/vpt_surface.rs")
        .expect("VPT surface pass source should exist");
    let pipeline = source("src/render/vpt_pipeline.rs");

    assert!(shader.contains("ConstantBuffer<VptHistoryUniforms> vpt_history"));
    assert!(shader.contains("project_previous_pixel"));
    assert!(shader.contains("vpt_history.current_view_proj"));
    assert!(shader.contains("vpt_history.previous_view_proj"));
    assert!(shader.contains("vpt_history.previous_resolution"));
    assert!(shader.contains("VPT_HISTORY_FLAG_CAMERA_CUT"));
    assert!(shader.contains("current_clip.w <= 1.0e-5"));
    assert!(shader.contains("previous_clip.w <= 1.0e-5"));
    assert!(shader.contains("float2 current_pixel = float2(pixel) + 0.5"));
    assert!(shader.contains("float2 motion = previous_pixel - current_pixel"));
    assert!(!shader.contains("motion_history[pixel] = float4(float2(pixel), 0.0, 1.0);"));

    assert!(rust.contains("GpuVptHistoryUniforms"));
    assert!(rust.contains("history_uniform_buffers"));
    assert!(rust.contains("update_history_uniforms"));
    assert!(rust.contains("std::mem::size_of::<GpuVptHistoryUniforms>()"));

    assert!(pipeline.contains("update_history_uniforms"));
    assert!(pipeline.contains("previous_vpt_view_proj"));
    assert!(pipeline.contains("compute_view_proj"));
}

#[test]
fn vpt_hit_payload_carries_motion_id_without_changing_reprojection_consumers() {
    let ray = source("assets/shaders/shared/ray.slang");
    let traversal = source("assets/shaders/shared/voxel_traverse.slang");
    let surface = source("assets/shaders/passes/vpt_surface.slang");
    let temporal = source("assets/shaders/passes/vpt_temporal.slang");
    let restir_di = source("assets/shaders/passes/restir_di_temporal.slang");
    let area_restir = source("assets/shaders/passes/area_restir_temporal.slang");

    for token in [
        "static const uint VPT_MOTION_ID_INVALID = 0u;",
        "static const uint VPT_MOTION_ID_STATIC_UCVH = 1u;",
        "uint  motion_id",
        "VPT_MOTION_ID_INVALID",
    ] {
        assert!(ray.contains(token), "ray hit payload missing {token}");
    }
    assert!(
        traversal.contains("result.motion_id = VPT_MOTION_ID_STATIC_UCVH;"),
        "primary UCVH hits should carry an explicit static motion source id"
    );
    assert!(
        surface.contains("motion.z = float(hit.motion_id);")
            && surface.contains("motion_history[pixel] = motion;"),
        "surface motion guide should preserve the hit motion id in the unused z lane"
    );
    for consumer in [temporal, restir_di, area_restir] {
        assert!(
            consumer.contains("vpt_history_sample_from_motion(pixel, motion);"),
            "temporal consumers should continue using the shared xy/w reprojection contract"
        );
        assert!(
            !consumer.contains("motion.z"),
            "motion id plumbing must not change temporal reprojection acceptance yet"
        );
    }
}

#[test]
fn vpt_pipeline_keeps_camera_projection_unjittered_until_taa_resolve_exists() {
    let pipeline = source("src/render/vpt_pipeline.rs");

    assert!(
        !pipeline.contains("taa_frame_jitter(frame.frame_index)"),
        "VPT must not apply frame-varying camera jitter until temporal resolve is explicitly jitter-stable"
    );
    assert!(
        pipeline.contains("current_jitter: [0.0, 0.0]"),
        "history uniforms should declare zero jitter while the VPT camera path is unjittered"
    );
    assert!(
        pipeline.contains("previous_jitter: [0.0, 0.0]"),
        "history uniforms should declare zero previous jitter while the VPT camera path is unjittered"
    );
}

#[test]
fn vpt_temporal_bilinear_reprojection_converts_pixel_centers_to_texel_corners() {
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");
    let common = std::fs::read_to_string("assets/shaders/shared/vpt_history_common.slang")
        .expect("VPT history common shader should exist");

    assert!(
        surface.contains("float2 current_pixel = float2(")
            && surface.contains("float2 previous_pixel = float2("),
        "surface pass should compute current and previous pixel centers before differencing them"
    );
    assert!(
        common.contains("return float2(pixel) + 0.5 + motion.xy;"),
        "shared helper must define previous pixel centers from motion delta"
    );
    assert!(
        temporal.contains("float2 history_sample = vpt_history_sample_from_motion(pixel, motion);"),
        "4-tap temporal reprojection must reconstruct the previous pixel-center coordinate from the motion delta"
    );
    assert!(
        temporal.contains("int2 previous_base_pixel = int2(floor(history_sample));"),
        "bilinear reprojection must keep a signed base pixel so edge taps can be rejected per tap"
    );
    assert!(
        temporal.contains(
            "float2 history_fraction = saturate(history_sample - float2(previous_base_pixel));"
        ),
        "bilinear weights must be computed from the center-corrected base pixel"
    );
    assert!(
        !temporal.contains("uint2 previous_pixel = uint2(motion.xy);"),
        "motion.xy is now a delta, not an absolute previous pixel coordinate"
    );
    assert!(
        !temporal.contains("float2 history_sample = motion.xy - 0.5;"),
        "subtracting 0.5 from a delta motion vector would bias the history sample"
    );
}

#[test]
fn vpt_temporal_bilinearly_samples_reprojected_history_and_clamps_variance_outliers() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    assert!(temporal.contains("sample_reprojected_history("));
    assert!(temporal.contains("history_tap_offsets"));
    assert!(temporal.contains("history_fraction"));
    assert!(temporal.contains("history_reliability"));
    assert!(temporal.contains("clamp_noisy_radiance_to_history"));
    assert!(temporal.contains("history_variance"));
}

#[test]
fn vpt_temporal_applies_edge_aware_spatial_firefly_clamp_before_history() {
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    for token in [
        "static const int2 firefly_tap_offsets[8]",
        "struct FireflyClampStats",
        "compatible_current_surface",
        "gather_firefly_clamp_stats",
        "clamp_noisy_radiance_to_spatial_firefly_stats",
        "spatial_firefly_luma_limit",
    ] {
        assert!(
            temporal.contains(token),
            "VPT temporal shader missing firefly clamp token {token}"
        );
    }

    assert!(
        temporal.contains("if (surface_normal_roughness[pixel].w > 1.0e-4)"),
        "firefly clamp must skip visible emissive surfaces instead of crushing real light sources"
    );
    assert!(
        temporal.contains("noisy_radiance = float4(clamped_noisy.radiance, noisy_radiance.a);"),
        "spatial firefly clamp must run before temporal history blending consumes noisy_radiance"
    );
    assert!(
        temporal.contains("clamped.moments = float2(clamped_luma, clamped_luma * clamped_luma);"),
        "firefly clamp must keep luminance moments consistent with the clamped radiance"
    );
    assert!(
        surface.contains("float emissive_luma = dot(material_emissive(hit.cell)")
            && surface.contains(
                "surface_normal_roughness[pixel] = float4(normalize(hit.normal), emissive_luma);"
            ),
        "VPT surface pass must expose emissive luminance for high-quality firefly clamp rejection"
    );
}

#[test]
fn vpt_temporal_firefly_clamp_uses_robust_trimmed_neighbors_not_max_neighbor() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    for token in [
        "float sorted_luma[8]",
        "sort_firefly_luma",
        "stats.median_luma",
        "stats.trimmed_mean_luma",
        "stats.robust_sigma_luma",
        "FIREFLY_CLAMP_MEDIAN_SCALE",
        "FIREFLY_CLAMP_TRIMMED_MEAN_SCALE",
    ] {
        assert!(
            temporal.contains(token),
            "VPT temporal shader missing robust clamp token {token}"
        );
    }
    assert!(
        !temporal.contains("max_neighbor_luma * FIREFLY_CLAMP_NEIGHBOR_SCALE"),
        "firefly clamp must not let one bright neighbor raise the limit for nearby outliers"
    );
}

#[test]
fn vpt_temporal_firefly_clamp_sanitizes_history_blended_output() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    assert!(
        temporal.contains("FireflyClampStats firefly_stats = (FireflyClampStats)0;"),
        "firefly stats must outlive the raw clamp so they can sanitize temporal history output"
    );
    assert!(
        temporal.contains(
            "ClampedRadianceSample output_clamped = clamp_noisy_radiance_to_spatial_firefly_stats("
        ) && temporal.contains("accumulated = output_clamped.radiance;")
            && temporal.contains("accumulated_moments = output_clamped.moments;"),
        "firefly clamp must also be applied after temporal blending, otherwise old history fireflies persist"
    );
}

#[test]
fn vpt_temporal_motion_debug_view_encodes_reprojection_delta() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    assert!(
        temporal.contains("VPT_DEBUG_VIEW_MOTION"),
        "temporal shader should expose motion history for reprojection debugging"
    );
    assert!(
        temporal.contains("motion.xy"),
        "motion debug view should show the raw motion delta"
    );
}

#[test]
fn vpt_debug_views_are_actually_routed_to_temporal_output() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    for token in [
        "VPT_DEBUG_VIEW_RAW",
        "VPT_DEBUG_VIEW_RESERVOIR_WEIGHT",
        "VPT_DEBUG_VIEW_DIRECT",
        "VPT_DEBUG_VIEW_INDIRECT",
        "VPT_DEBUG_VIEW_NORMAL",
        "VPT_DEBUG_VIEW_DEPTH",
        "VPT_DEBUG_VIEW_VARIANCE",
    ] {
        assert!(
            temporal.contains(token),
            "parsed VPT debug view is not consumed by temporal output: {token}"
        );
    }

    assert!(
        temporal.contains("debug_view_bypasses_temporal(scene.vpt_debug_view)")
            && temporal
                .contains("accumulated_radiance_image[pixel] = float4(noisy_radiance.rgb, 1.0);")
            && temporal.contains("return;"),
        "raw/ReSTIR/direct/indirect debug views must bypass temporal reuse and firefly clamps"
    );
    assert!(
        temporal.contains("visualize_luminance_variance("),
        "variance debug view should expose temporal moment variance instead of silently showing final color"
    );
}

#[test]
fn vpt_trace_shader_emits_isolated_direct_indirect_and_reservoir_debug_samples() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));

    for token in [
        "struct VptTraceSample",
        "sample.direct_radiance",
        "sample.indirect_radiance",
        "sample.reservoir_weight",
        "VPT_DEBUG_VIEW_RESERVOIR_WEIGHT",
        "VPT_DEBUG_VIEW_DIRECT",
        "VPT_DEBUG_VIEW_INDIRECT",
        "visualize_restir_reservoir_weight",
    ] {
        assert!(
            source.contains(token),
            "VPT trace shader missing diagnostic token {token}"
        );
    }
}

#[test]
fn vpt_trace_shader_exposes_voxel_traversal_debug_views() {
    let scene_common = normalized_source(include_str!(
        "../../../../assets/shaders/shared/scene_common.slang"
    ));
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let temporal = normalized_source(include_str!(
        "../../../../assets/shaders/passes/vpt_temporal.slang"
    ));

    for token in [
        "VPT_DEBUG_VIEW_VOXEL_BRICK",
        "VPT_DEBUG_VIEW_VOXEL_LOCAL",
        "VPT_DEBUG_VIEW_VOXEL_HIT",
    ] {
        assert!(
            scene_common.contains(token),
            "scene_common missing voxel debug constant {token}"
        );
        assert!(
            source.contains(token),
            "VPT shader missing voxel debug routing token {token}"
        );
        assert!(
            temporal.contains(token),
            "temporal shader must bypass voxel traversal debug token {token}"
        );
    }

    for token in [
        "visualize_voxel_brick_id",
        "visualize_voxel_local_coord",
        "visualize_voxel_hit",
        "sample.first_hit",
        "first_hit.brick_id",
        "first_hit.local",
        "first_hit.normal",
    ] {
        assert!(
            source.contains(token),
            "VPT shader missing voxel traversal debug token {token}"
        );
    }
}

#[test]
fn app_runs_vpt_surface_before_restir_and_vpt_trace() {
    let app = source("src/app.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let surface_pass = source("src/render/passes/vpt_surface.rs");
    let area_pass = std::fs::read_to_string("src/render/passes/area_restir.rs")
        .expect("Area ReSTIR pass source should be readable");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let bootstrap_surface_idx = pipeline
        .find("vpt_surface.register_bootstrap_graph(")
        .expect("VPT bootstrap surface graph pass should exist");
    let area_register_idx = pipeline
        .find("area_restir.register_graph(")
        .expect("Area ReSTIR graph registration should exist");
    let selected_surface_idx = area_pass
        .find("\"vpt_surface_selected\"")
        .expect("VPT selected surface graph pass should exist");
    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT trace graph pass should exist after surface registration");

    assert!(app.contains("vpt_pipeline: VptRuntimePipeline"));
    assert!(app.contains("self.vpt_pipeline.ensure_passes("));
    assert!(pipeline.contains("pub vpt_surface_pass: Option<VptSurfacePass>"));
    assert!(pipeline.contains("VptSurfacePass::new"));
    assert!(surface_pass.contains("\"vpt_surface_bootstrap\""));
    assert!(bootstrap_surface_idx < area_register_idx);
    assert!(area_pass.find("\"area_restir_initial\"") < Some(selected_surface_idx));
    assert!(area_register_idx < vpt_idx);
    assert!(vpt_pass.contains("graph.add_pass(\"vpt\""));
    if let Some(restir_idx) = pipeline.find("restir_di.register_graph(") {
        assert!(area_register_idx < restir_idx);
    }
}

#[test]
fn app_profiles_bootstrap_and_selected_vpt_surface_with_distinct_query_scopes() {
    let surface_pass = source("src/render/passes/vpt_surface.rs");
    let area_pass = std::fs::read_to_string("src/render/passes/area_restir.rs")
        .expect("Area ReSTIR pass source should be readable");
    let profiler = std::fs::read_to_string("src/render/gpu_profiler.rs")
        .expect("GPU profiler source should be readable");

    assert!(profiler.contains("VptSurfaceBootstrap"));
    assert!(profiler.contains("VptSurfaceSelected"));
    assert!(surface_pass.contains("GpuProfileScope::VptSurfaceBootstrap"));
    assert!(area_pass.contains("GpuProfileScope::VptSurfaceSelected"));
    assert!(
        !surface_pass.contains("GpuProfileScope::VptSurface,")
            && !area_pass.contains("GpuProfileScope::VptSurface,"),
        "bootstrap and selected surface passes must not reuse one timestamp query scope"
    );
}

#[test]
fn app_keeps_restir_di_behind_vpt_setting() {
    let source = std::fs::read_to_string("src/app.rs")
        .expect("app source should be readable for ReSTIR-DI app wiring test");
    let pipeline = std::fs::read_to_string("src/render/vpt_pipeline.rs")
        .expect("VPT pipeline source should be readable for ReSTIR-DI app wiring test");
    let compact_source = source.split_whitespace().collect::<String>();

    assert!(source.contains("RestirDiSettings::from_env"));
    assert!(source.contains("restir_di_settings: RestirDiSettings"));
    assert!(pipeline.contains("pub restir_di_pass: Option<RestirDiPass>"));
    assert!(source.contains("fn restir_di_vpt_enabled(&self) -> bool"));
    assert!(source.contains("let restir_di_enabled = self.restir_di_vpt_enabled();"));
    assert!(source.contains("self.vpt_pipeline.ensure_passes("));
    assert!(
        compact_source.contains("self.restir_di_settings.enabled"),
        "ReSTIR-DI must stay disabled unless the explicit setting is enabled"
    );
    assert!(
        pipeline.contains("fn ensure_restir_di_pass")
            && pipeline.contains("if self.restir_di_pass.is_some() || !restir_di_enabled {"),
        "ReSTIR-DI pass creation must be nested behind the explicit setting guard"
    );
}

#[test]
fn vpt_shader_can_resolve_restir_di_direct_light_when_enabled() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));

    assert!(source.contains("#include \"restir_di_common.slang\""));
    assert!(source.contains("ConstantBuffer<RestirDiUniforms> restir;"));
    assert!(source.contains("StructuredBuffer<RestirDiReservoir> restir_reservoirs;"));
    assert!(source.contains("resolve_restir_di_direct_light"));
    assert!(source.contains("if (restir.enabled != 0u && bounce == 0u)"));
}

#[test]
fn vpt_restir_direct_resolve_uses_area_sample_position_without_pdf_amplification() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));

    for token in [
        "restir_di_emissive_geometry_term",
        "reservoir.sample_radiance.rgb * geometry_term",
        "restir_di_light_visible_from_hit(hit, reservoir, scene)",
        "float4 hit_position_depth = float4(hit.position, max(hit.t, 0.0));",
        "float4 hit_normal_roughness = float4(normalize(hit.normal), 0.0);",
        "float4 hit_albedo_material = float4(material_cell_albedo(hit.cell), float(voxel_material(hit.cell)));",
        "restir_di_target_pdf_for_reservoir(",
        "hit_reservoir.target_pdf = hit_target_pdf;",
        "float selected_weight = restir_di_bounded_selected_weight(hit_reservoir);",
    ] {
        assert!(
            source.contains(token),
            "VPT Area ReSTIR resolve missing token {token}"
        );
    }
    assert!(
        !source.contains("reservoir.sample_radiance.rgb / sample_pdf"),
        "cluster color_power already represents total emissive power; dividing final radiance by area PDF causes brightness explosions"
    );
}

#[test]
fn vpt_restir_di_visibility_uses_any_hit_shadow_traversal() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "bool trace_any_hit_ray(",
        "bool trace_any_hit_ray_skip_voxel(",
        "bool brick_any_hit(",
        "float max_t",
        "BrickOccupancy occ",
        "ray_axis_t_max(ray.origin.x, ray.direction.x, inv_dir.x",
        "ray_axis_t_max(ray.origin.y, ray.direction.y, inv_dir.y",
        "ray_axis_t_max(ray.origin.z, ray.direction.z, inv_dir.z",
        "StructuredBuffer<BrickOccupancy> occupancy_buf",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing any-hit token {token}"
        );
    }
    assert!(
        !traverse.contains(
            "bool trace_any_hit_ray(\n    Ray ray,\n    float max_t,\n    StructuredBuffer<UcvhConfig> config_buf,\n    StructuredBuffer<NodeL0> hierarchy_l0,\n    StructuredBuffer<BrickOccupancy> occupancy_buf,\n    StructuredBuffer<VoxelCell> material_buf"
        ),
        "any-hit shadow traversal must not bind or read material cells"
    );
    assert!(
        vpt.contains(
            "bool shadow_occluded = voxel_shadow_occluded_from_hit(hit, shadow_dir, max_light_t);"
        ) && vpt.contains("return !shadow_occluded;"),
        "VPT ReSTIR-DI visibility should use source-voxel-skipping boolean any-hit shadow traversal"
    );
    assert!(
        !vpt.contains("HitResult occluder = trace_primary_ray(shadow_ray"),
        "VPT ReSTIR-DI visibility must not run full material-returning primary traversal for shadow rays"
    );
}

#[test]
fn vpt_restir_di_visibility_respects_shadow_disable_flag() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "bool restir_di_light_visible_from_hit(HitResult hit, RestirDiReservoir reservoir, SceneUniforms scene)",
        "(scene.lighting_flags & LIGHTING_FLAG_SHADOWS_ENABLED) == 0u",
        "restir_di_light_visible_from_hit(hit, reservoir, scene)",
    ] {
        assert!(
            vpt.contains(token),
            "VPT ReSTIR-DI visibility should follow REVOLUMETRIC_LIGHTING_SHADOWS; missing token {token}"
        );
    }
}

#[test]
fn vpt_analytic_sun_direct_respects_voxel_shadow_visibility() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "bool voxel_shadow_occluded_from_hit(HitResult hit, float3 shadow_dir, float max_t)",
        "bool analytic_sun_visible_from_hit(HitResult hit, SceneUniforms scene, float3 sun_dir)",
        "(scene.lighting_flags & LIGHTING_FLAG_SHADOWS_ENABLED) == 0u",
        "Ray shadow_ray = make_ray(hit.position + hit.normal * VPT_RAY_SURFACE_BIAS, shadow_dir);",
        "bool sun_occluded = voxel_shadow_occluded_from_hit(hit, sun_dir, 1.0e20);",
        "return !sun_occluded;",
        "if (sun_term <= 0.0 || !analytic_sun_visible_from_hit(hit, scene, sun_dir))",
    ] {
        assert!(
            vpt.contains(token),
            "VPT analytic sun must test voxel shadow visibility; missing token {token}"
        );
    }

    assert!(
        !vpt.contains(
            "return material_cell_albedo(hit.cell) * scene.sun_intensity * sun_term * 0.2;"
        ),
        "VPT analytic sun must not unconditionally add direct sunlight through voxel occluders"
    );
}

#[test]
fn vpt_analytic_sun_samples_solar_disk_for_soft_shadow_edges() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "float3 sample_sun_direction(SceneUniforms scene, inout uint rng_state)",
        "scene.sun_angular_radius",
        "float cos_min = cos(sun_radius);",
        "float cos_theta = lerp(cos_min, 1.0, rand01(rng_state));",
        "float phi = 6.28318530718 * rand01(rng_state);",
        "float3 sun_dir = sample_sun_direction(scene, rng_state);",
        "float sun_term = max(dot(hit.normal, sun_dir), 0.0);",
        "analytic_sun_direct(hit, scene, rng_state)",
    ] {
        assert!(
            vpt.contains(token),
            "VPT analytic sun should sample a finite solar disk for soft shadows; missing token {token}"
        );
    }

    assert!(
        !vpt.contains("analytic_sun_direct(hit, scene);"),
        "analytic sun direct lighting must consume rng_state so penumbrae can converge across VPT samples"
    );
}

#[test]
fn voxel_ray_traversal_treats_parallel_axes_as_non_stepping_slabs() {
    let ray =
        std::fs::read_to_string("assets/shaders/shared/ray.slang").expect("ray shader exists");
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "static const float RAY_DIRECTION_EPSILON",
        "static const float RAY_PARALLEL_INV_DIR",
        "static const float RAY_T_MAX",
        "float ray_safe_rcp(float direction_component)",
        "float3 ray_safe_inv_dir(float3 direction)",
        "int ray_step_component(float direction_component)",
        "int3 ray_step_dir(float3 direction)",
        "float ray_axis_t_delta(",
        "float ray_axis_t_max(",
        "r.inv_dir = ray_safe_inv_dir(direction);",
    ] {
        assert!(ray.contains(token), "ray helper missing token {token}");
    }

    assert!(
        !ray.contains("sign(direction.x) *")
            && !ray.contains("sign(direction.y) *")
            && !ray.contains("sign(direction.z) *"),
        "parallel ray reciprocals must not use sign(0), which produces a zero reciprocal"
    );

    for token in [
        "float3 inv_dir = ray_safe_inv_dir(ray_dir);",
        "int3 step_dir = ray_step_dir(ray_dir);",
        "ray_axis_t_delta(ray_dir.x, inv_dir.x, 1.0)",
        "ray_axis_t_delta(ray.direction.x, inv_dir.x, 8.0)",
        "ray_axis_t_max(ray_origin.x, ray_dir.x, inv_dir.x",
        "ray_axis_t_max(ray.origin.x, ray.direction.x, inv_dir.x",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal missing robust parallel-axis token {token}"
        );
    }

    assert!(
        !traverse.contains("int3(sign(ray_dir))")
            && !traverse.contains("int3(sign(ray.direction))")
            && !traverse.contains("sign(ray_dir.x) *")
            && !traverse.contains("sign(ray.direction.x) *"),
        "DDA must not step a zero-direction axis or derive reciprocal from sign(0)"
    );
}

#[test]
fn ray_aabb_rejects_parallel_axes_that_start_outside_the_slab() {
    let math = std::fs::read_to_string("assets/shaders/shared/math.slang")
        .expect("math shader should be readable");

    for token in [
        "static const float AABB_PARALLEL_INV_DIR_THRESHOLD",
        "static const float AABB_T_MAX",
        "float2 aabb_axis_interval(",
        "origin_component < min_component || origin_component > max_component",
        "return float2(1.0, 0.0);",
        "return float2(-AABB_T_MAX, AABB_T_MAX);",
        "float2 x = aabb_axis_interval(origin.x, inv_dir.x, box.mn.x, box.mx.x);",
        "float2 y = aabb_axis_interval(origin.y, inv_dir.y, box.mn.y, box.mx.y);",
        "float2 z = aabb_axis_interval(origin.z, inv_dir.z, box.mn.z, box.mx.z);",
    ] {
        assert!(
            math.contains(token),
            "AABB slab intersection missing parallel-axis rejection token {token}"
        );
    }
}

#[test]
fn voxel_traversal_steps_all_tied_dda_axes_to_avoid_edge_seams() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "static const float DDA_TIE_EPSILON",
        "float dda_next_t(float3 t_max)",
        "bool dda_axis_is_tied(float axis_t, float next_t)",
        "float3 dda_step_normal(",
        "void dda_step_voxel(",
        "void dda_step_brick(",
        "dda_step_voxel(coord, t_max, t_delta, step_dir, current_t, hit_normal);",
        "dda_step_brick_with_t(brick_coord, t_max, t_delta, step_dir, current_t);",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing tie-aware DDA token {token}"
        );
    }

    assert!(
        !traverse.contains("if (t_max.x < t_max.y)"),
        "single-axis DDA tie breaking visits edge-touching cells and creates voxel/brick seams"
    );
    assert!(
        !traverse.contains("normal * rsqrt(len2)"),
        "tie-aware traversal must keep exported hit normals axis-aligned for lighting and reprojection"
    );
}

#[test]
fn voxel_traversal_uses_direction_aware_entry_cells_instead_of_fixed_nudge() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "static const float DDA_GRID_BOUNDARY_EPSILON",
        "float dda_adjust_boundary_position(",
        "int dda_start_coord(",
        "int3 dda_start_coord3(",
        "dda_start_coord3(entry_pos",
        "ray.origin.x, ray.direction.x, inv_dir.x",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing direction-aware DDA entry token {token}"
        );
    }

    assert!(
        !traverse.contains("t_enter + 0.001"),
        "fixed t-space entry nudges perturb cell selection at voxel and brick boundaries"
    );
}

#[test]
fn voxel_traversal_uses_uploaded_l1_l4_hierarchy_for_empty_space_skipping() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("surface shader should be readable");

    for token in [
        "StructuredBuffer<NodeLN> hierarchy_l1",
        "StructuredBuffer<NodeLN> hierarchy_l2",
        "StructuredBuffer<NodeLN> hierarchy_l3",
        "StructuredBuffer<NodeLN> hierarchy_l4",
        "bool hierarchy_level_empty_at_brick(",
        "bool try_skip_empty_hierarchy_block(",
        "void advance_brick_dda_to_t(",
        "node_ln_child_mask(parent)",
        "try_skip_empty_hierarchy_block(",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing hierarchy skip token {token}"
        );
    }

    assert!(
        traverse.matches("try_skip_empty_hierarchy_block(").count() >= 3,
        "primary and any-hit traversal loops must both use hierarchy empty-space skipping"
    );
    assert!(
        vpt.contains("hierarchy_l4") && surface.contains("hierarchy_l4"),
        "VPT trace and surface passes must pass all L1-L4 buffers into shared traversal"
    );
}

#[test]
fn voxel_hierarchy_skip_does_not_advance_past_block_exit_boundary() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "bool dda_t_exceeds_target(",
        "float next_t = dda_next_t(t_max);",
        "if (dda_t_exceeds_target(next_t, target_t))",
        "break;",
        "int3 original_brick_coord = skipped_brick_coord;",
        "return any(skipped_brick_coord != original_brick_coord);",
    ] {
        assert!(
            traverse.contains(token),
            "hierarchy skip DDA advance missing boundary guard token {token}"
        );
    }
}

#[test]
fn vpt_ray_biases_do_not_skip_voxel_faces_or_shadow_endpoints() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "static const float VPT_RAY_SURFACE_BIAS",
        "static const float VPT_LIGHT_ENDPOINT_BIAS",
        "hit.position + hit.normal * VPT_RAY_SURFACE_BIAS",
        "max_light_t = max(light_distance - VPT_LIGHT_ENDPOINT_BIAS, 0.0);",
    ] {
        assert!(vpt.contains(token), "VPT shader missing bias token {token}");
    }

    for forbidden in [
        "hit.normal * 0.75",
        "hit.normal * 0.05 + shadow_dir * 0.05",
        "light_distance - 1.0",
    ] {
        assert!(
            !vpt.contains(forbidden),
            "VPT shader uses leak-prone large ray bias token {forbidden}"
        );
    }
}

#[test]
fn vpt_shadow_visibility_skips_originating_voxel_to_avoid_surface_acne() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "static const uint VOXEL_TRAVERSAL_NO_SKIP_BRICK",
        "bool voxel_traversal_should_skip_shadow_hit(",
        "bool trace_any_hit_ray_skip_voxel(",
        "uint skip_brick_id",
        "uint3 skip_local",
        "node.brick_id",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing source-voxel shadow skip token {token}"
        );
    }

    for token in [
        "bool voxel_shadow_occluded_from_hit(HitResult hit, float3 shadow_dir, float max_t)",
        "trace_any_hit_ray_skip_voxel(",
        "hit.brick_id",
        "hit.local",
        "bool sun_occluded = voxel_shadow_occluded_from_hit(hit, sun_dir, 1.0e20);",
        "bool shadow_occluded = voxel_shadow_occluded_from_hit(hit, shadow_dir, max_light_t);",
    ] {
        assert!(
            vpt.contains(token),
            "VPT shader missing source-voxel shadow skip token {token}"
        );
    }
}

#[test]
fn vpt_pass_binds_restir_di_uniform_and_reservoir_resources() {
    let pass_source = source("src/render/passes/vpt.rs");
    let implementation = pass_source
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    crate::render::source_checks::assert_contains_all(
        implementation,
        &[
            "descriptor_binding_specs",
            "update_restir_di_descriptors",
            "GpuRestirDiUniforms",
            "GpuRestirDiReservoir",
        ],
        "VPT pass ReSTIR-DI descriptors",
    );
}

#[test]
fn vpt_pass_binds_area_restir_as_independent_sample_area_resources() {
    let pass_source = source("src/render/passes/vpt.rs");
    let implementation = pass_source
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    crate::render::source_checks::assert_contains_all(
        implementation,
        &[
            "descriptor_binding_specs",
            "disabled_area_restir_uniform_buffers",
            "disabled_area_restir_reservoir_buffer",
            "update_area_restir_descriptors",
            "GpuAreaRestirUniforms",
            "GpuAreaRestirReservoir",
            "create_disabled_area_restir_uniform_buffers",
            "create_disabled_area_restir_reservoir_buffer",
        ],
        "VPT pass Area ReSTIR descriptors",
    );

    assert!(
        implementation.find("update_restir_di_descriptors")
            < implementation.find("update_area_restir_descriptors"),
        "Area ReSTIR must be bound separately from ReSTIR-DI, not merged into DI descriptors"
    );
}

#[test]
fn vpt_shader_uses_area_restir_reservoir_to_override_primary_ray_when_valid() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let primary_sample_common = normalized_source(include_str!(
        "../../../../assets/shaders/shared/vpt_primary_sample_common.slang"
    ));
    let combined_source = format!("{source}\n{primary_sample_common}");

    for token in [
        "#include \"area_restir_common.slang\"",
        "#include \"vpt_primary_sample_common.slang\"",
        "ConstantBuffer<AreaRestirUniforms> area_restir;",
        "StructuredBuffer<AreaRestirReservoir> area_restir_reservoirs;",
        "resolve_area_restir_primary_ray",
        "area_restir_is_valid_reservoir",
        "scene_primary_ray_from_area_sample",
        "area_restir_pixel_sample(pixel, reservoir.sample_state)",
        "reservoir.sample_state.lens_uv",
        "if (area.enabled != 0u",
        "fallback jitter",
        "float4 hit_position_depth = float4(hit.position, max(hit.t, 0.0));",
        "hit_reservoir.target_pdf = hit_target_pdf;",
    ] {
        assert!(
            combined_source.contains(token),
            "VPT shader missing Area ReSTIR primary-ray token {token}"
        );
    }
    assert!(
        !source.contains("reservoir.sample_state.pixel_sample"),
        "VPT must not replay a history/neighbor reservoir's source pixel; only its subpixel/lens state is reusable"
    );
    assert!(
        !source.contains("float2(pixel) + reservoir.sample_state.subpixel_uv"),
        "VPT must not shift Area ReSTIR samples by half a pixel; use the shared pixel-sample conversion"
    );
    assert!(
        !source.contains("current_surface_position_depth")
            && !source.contains("restir_di_surface_compatible_with_hit"),
        "VPT must not reject Area ReSTIR subpixel/lens hits by comparing them against the center-pixel surface buffer"
    );
    let area_common = include_str!("../../../../assets/shaders/shared/area_restir_common.slang");
    assert!(
        !area_common.contains("float2 pixel_sample")
            && !area_common.contains("float4 selected_radiance"),
        "Area ReSTIR reservoir ABI must not carry unused pixel-sample or radiance payload"
    );
}

#[test]
fn vpt_surface_and_trace_share_area_restir_primary_sample_contract() {
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("VPT trace shader should exist");
    let common = std::fs::read_to_string("assets/shaders/shared/vpt_primary_sample_common.slang")
        .expect("shared VPT primary sample contract should exist");

    for (name, source) in [("surface", surface.as_str()), ("vpt", vpt.as_str())] {
        assert!(
            source.contains("#include \"vpt_primary_sample_common.slang\""),
            "{name} shader must include the shared VPT primary-sample contract"
        );
        assert!(
            !source.contains("float3 primary_ray_direction(SceneUniforms scene, uint2 pixel)"),
            "{name} shader must not keep a private pixel-center primary-ray path"
        );
    }
    assert!(
        surface.contains("vpt_resolve_surface_primary_ray"),
        "surface shader must replay selected Area ReSTIR samples through the stable surface resolver"
    );
    assert!(
        vpt.contains("vpt_resolve_area_restir_primary_ray"),
        "trace shader must replay selected Area ReSTIR samples while preserving stochastic VPT fallback"
    );

    for token in [
        "uint vpt_primary_rng_seed",
        "ScenePrimaryRay vpt_center_primary_ray",
        "ScenePrimaryRay vpt_fallback_primary_ray",
        "ScenePrimaryRay vpt_primary_ray_from_area_reservoir",
        "ScenePrimaryRay vpt_resolve_area_restir_primary_ray",
        "ScenePrimaryRay vpt_resolve_surface_primary_ray",
        "area_restir_pixel_sample(pixel, reservoir.sample_state)",
        "scene_primary_ray_from_area_sample",
    ] {
        assert!(
            common.contains(token),
            "shared VPT primary-sample contract missing {token}"
        );
    }
}

#[test]
fn vpt_surface_fallback_keeps_stable_center_ray_for_history_guides() {
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let common = std::fs::read_to_string("assets/shaders/shared/vpt_primary_sample_common.slang")
        .expect("shared VPT primary sample contract should exist");

    assert!(
        common.contains("ScenePrimaryRay vpt_center_primary_ray"),
        "shared contract must expose a deterministic center-ray guide for surface history"
    );
    assert!(
        common.contains("ScenePrimaryRay vpt_resolve_surface_primary_ray"),
        "surface pass must have a resolver that replays valid Area ReSTIR samples but does not jitter invalid fallback"
    );
    assert!(
        common.contains("return vpt_center_primary_ray(pixel, scene);"),
        "surface fallback must return the stable center ray, not a stochastic VPT sample"
    );
    assert!(
        surface.contains("vpt_resolve_surface_primary_ray"),
        "VPT surface shader must use the stable surface resolver"
    );
    assert!(
        !surface.contains("vpt_resolve_area_restir_primary_ray"),
        "VPT surface shader must not use the stochastic VPT trace fallback resolver"
    );
}

#[test]
fn vpt_surface_pass_binds_area_restir_selected_primary_sample() {
    let surface_shader = source("assets/shaders/passes/vpt_surface.slang");
    let pass_source = source("src/render/passes/vpt_surface.rs");
    let implementation = pass_source
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    for token in [
        "#include \"area_restir_common.slang\"",
        "#include \"vpt_primary_sample_common.slang\"",
        "ConstantBuffer<AreaRestirUniforms> area_restir;",
        "StructuredBuffer<AreaRestirReservoir> area_restir_reservoirs;",
        "vpt_resolve_surface_primary_ray",
        "make_ray(primary_ray.origin, primary_ray.direction)",
    ] {
        assert!(
            surface_shader.contains(token),
            "VPT surface shader missing Area ReSTIR selected-primary token {token}"
        );
    }

    crate::render::source_checks::assert_contains_all(
        implementation,
        &[
            "descriptor_binding_specs",
            "bootstrap_descriptor_sets",
            "selected_descriptor_sets",
            "disabled_area_restir_uniform_buffers",
            "disabled_area_restir_reservoir_buffer",
            "update_area_restir_descriptors",
            "record_bootstrap",
            "record_selected",
            "write_area_restir_descriptor_sets",
            "GpuAreaRestirUniforms",
            "GpuAreaRestirReservoir",
        ],
        "VPT surface pass selected-primary descriptors",
    );
}

#[test]
fn app_uses_selected_vpt_surface_after_area_restir_for_di_trace_and_temporal() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let surface_pass = source("src/render/passes/vpt_surface.rs");
    let area_pass = std::fs::read_to_string("src/render/passes/area_restir.rs")
        .expect("Area ReSTIR pass source should be readable");
    let restir_pass = source("src/render/passes/restir_di.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let temporal_pass = source("src/render/passes/vpt_temporal.rs");
    let compact_area_pass = area_pass.split_whitespace().collect::<String>();
    let compact_vpt = vpt_pass.split_whitespace().collect::<String>();
    let compact_temporal = temporal_pass.split_whitespace().collect::<String>();
    let compact_restir = restir_pass.split_whitespace().collect::<String>();

    for token in [
        "vpt_surface.register_bootstrap_graph(",
        "area_restir.register_graph(",
        "vpt_area_restir_reads",
        "vpt.register_graph(",
        "vpt_temporal.register_graph(",
        "vpt_temporal.register_history_update_graph(",
    ] {
        assert!(
            pipeline.contains(token),
            "VPT pipeline graph missing selected-surface token {token}"
        );
    }
    for token in [
        "\"vpt_surface_selected\"",
        "vpt_surface.record_selected",
        "vpt_surface.update_area_restir_descriptors",
    ] {
        assert!(
            area_pass.contains(token),
            "Area ReSTIR pass graph missing selected-surface token {token}"
        );
    }

    let bootstrap_idx = pipeline
        .find("vpt_surface.register_bootstrap_graph(")
        .expect("bootstrap surface pass should exist");
    let area_register_idx = pipeline
        .find("area_restir.register_graph(")
        .expect("Area ReSTIR graph registration should exist");
    let area_initial_idx = area_pass
        .find("\"area_restir_initial\"")
        .expect("Area ReSTIR initial pass should exist");
    let selected_idx = area_pass
        .find("\"vpt_surface_selected\"")
        .expect("selected surface pass should exist");
    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT pass should exist");
    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("VPT temporal pass should exist");
    let history_update_idx = pipeline
        .find("vpt_temporal.register_history_update_graph(")
        .expect("VPT surface history update pass should exist");
    let restir_idx = pipeline
        .find("restir_di.register_graph(")
        .unwrap_or(usize::MAX);

    assert!(bootstrap_idx < area_register_idx);
    assert!(area_initial_idx < selected_idx);
    if restir_idx != usize::MAX {
        assert!(area_register_idx < restir_idx);
    }
    assert!(area_register_idx < vpt_idx);
    assert!(area_register_idx < temporal_idx);
    assert!(area_register_idx < history_update_idx);
    assert!(surface_pass.contains("\"vpt_surface_bootstrap\""));
    assert!(
        compact_vpt
            .contains("builder.read_as(area_uniform_resource,AccessKind::ComputeShaderRead)")
    );
    assert!(compact_vpt.contains(
        "builder.read_as(area_selected_reservoir_resource,AccessKind::ComputeShaderRead"
    ));
    assert!(
        compact_temporal.contains("builder.read_as(surface_input,AccessKind::ComputeShaderRead)")
    );
    assert!(
        compact_temporal
            .contains("builder.read_as(previous_surface_input,AccessKind::ComputeShaderRead)")
    );
    assert!(
        compact_restir
            .contains("builder.read_as(final_surface_writes[0],AccessKind::ComputeShaderRead)")
    );
    assert!(
        compact_restir
            .contains("builder.read_as(final_surface_writes[1],AccessKind::ComputeShaderRead)")
    );
    assert!(
        compact_restir
            .contains("builder.read_as(final_surface_writes[2],AccessKind::ComputeShaderRead)")
    );
    assert!(
        compact_area_pass
            .contains("builder.read_as(uniform_resource,AccessKind::ComputeShaderRead)")
            && compact_area_pass
                .contains("builder.read_as(selected_resource,AccessKind::ComputeShaderRead)"),
        "Area ReSTIR selected-surface pass must read the selected reservoir and uniform resources"
    );
}

#[test]
fn app_declares_spatial_reservoir_as_vpt_read_dependency() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let compact_vpt = vpt_pass.split_whitespace().collect::<String>();
    let restir_pass = std::fs::read_to_string("src/render/passes/restir_di.rs")
        .expect("ReSTIR-DI source should be readable");

    assert!(pipeline.contains("vpt_restir_reads"));
    assert!(pipeline.contains("vpt.register_graph("));
    assert!(
        compact_pipeline.contains(
            "vpt_restir_reads=Some((restir_graph.uniform_resource,restir_graph.selected_current_resource,));"
        )
    );
    assert!(restir_pass.contains("vpt.update_restir_di_descriptors"));
    assert!(
        compact_vpt
            .contains("builder.read_as(restir_uniform_resource,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_vpt
            .contains("builder.read_as(restir_reservoir_resource,AccessKind::ComputeShaderRead")
    );
}

#[test]
fn app_wires_area_restir_between_surface_and_vpt_with_history_and_vpt_reads() {
    let app = source("src/app.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let surface_pass = source("src/render/passes/vpt_surface.rs");
    let area_pass = std::fs::read_to_string("src/render/passes/area_restir.rs")
        .expect("Area ReSTIR pass source should be readable");
    let restir_pass = source("src/render/passes/restir_di.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let temporal_pass = source("src/render/passes/vpt_temporal.rs");
    let area_pass_impl = area_pass
        .split("#[cfg(test)]")
        .next()
        .expect("Area ReSTIR implementation section should exist");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let compact_area_pass = area_pass.split_whitespace().collect::<String>();
    let compact_vpt = vpt_pass.split_whitespace().collect::<String>();

    for app_token in [
        "AreaRestirSettings::from_env",
        "area_restir_settings: AreaRestirSettings",
        "ucvh_gpu",
        "self.vpt_pipeline.ensure_passes(",
    ] {
        assert!(
            app.contains(app_token),
            "app missing Area ReSTIR setup token {app_token}"
        );
    }

    for pipeline_token in [
        "vpt_surface.register_bootstrap_graph(",
        "area_restir.register_graph(",
        "vpt_area_restir_reads",
        "vpt.register_graph(",
        "vpt_temporal.register_graph(",
        "vpt_temporal.register_history_update_graph(",
        "self.frame_state.area_restir_history_initialized",
    ] {
        assert!(
            if pipeline_token == "self.frame_state.area_restir_history_initialized" {
                compact_pipeline.contains(pipeline_token)
            } else {
                pipeline.contains(pipeline_token)
            },
            "VPT pipeline missing Area ReSTIR graph token {pipeline_token}"
        );
    }

    assert!(pipeline.contains("AreaRestirPass::new"));
    assert!(pipeline.contains("AreaRestirPassCreateInfo"));

    for pass_token in [
        "self.update_uniforms",
        "\"area_restir_initial\"",
        "\"area_restir_temporal\"",
        "\"area_restir_spatial\"",
        "self.selected_current_buffer",
        "self.selected_history_buffer",
        "update_surface_descriptors",
        "update_ucvh_descriptors",
        "vpt.update_area_restir_descriptors",
    ] {
        assert!(
            area_pass.contains(pass_token) || pipeline.contains(pass_token),
            "Area ReSTIR pass missing graph token {pass_token}"
        );
    }

    let bootstrap_surface_idx = pipeline
        .find("vpt_surface.register_bootstrap_graph(")
        .expect("bootstrap surface pass should exist");
    let area_register_idx = pipeline
        .find("area_restir.register_graph(")
        .expect("Area ReSTIR graph registration should exist");
    let area_initial_idx = area_pass
        .find("\"area_restir_initial\"")
        .expect("Area ReSTIR initial pass should exist");
    let selected_surface_idx = area_pass
        .find("\"vpt_surface_selected\"")
        .expect("selected surface pass should exist");
    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT pass should exist");
    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("VPT temporal pass should exist");
    let history_update_idx = pipeline
        .find("vpt_temporal.register_history_update_graph(")
        .expect("VPT temporal history update pass should exist");
    assert!(bootstrap_surface_idx < area_register_idx);
    assert!(area_initial_idx < selected_surface_idx);
    assert!(area_register_idx < vpt_idx);
    assert!(vpt_idx < temporal_idx);
    assert!(temporal_idx < history_update_idx);

    assert!(
        compact_area_pass
            .contains("builder.read_as(uniform_resource,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_area_pass
            .contains("builder.read_as(selected_resource,AccessKind::ComputeShaderRead")
    );
    assert!(surface_pass.contains("\"vpt_surface_bootstrap\""));
    assert!(
        compact_vpt
            .contains("builder.read_as(area_uniform_resource,AccessKind::ComputeShaderRead)")
    );
    assert!(compact_vpt.contains(
        "builder.read_as(area_selected_reservoir_resource,AccessKind::ComputeShaderRead"
    ));
    assert!(
        restir_pass
            .contains("builder.read_as(final_surface_writes[0], AccessKind::ComputeShaderRead)")
    );
    assert!(
        temporal_pass.contains("builder.read_as(surface_input, AccessKind::ComputeShaderRead)")
    );
    assert!(
        temporal_pass
            .contains("builder.read_as(previous_surface_input, AccessKind::ComputeShaderRead)")
    );
    assert!(
        !pipeline.contains("\"area_restir_history_update\"")
            && !area_pass_impl.contains("\"area_restir_history_update\""),
        "Area ReSTIR selected reservoirs must not be copied through a transfer history pass"
    );
}

#[test]
fn vpt_does_not_assume_primary_gbuffer_for_restir_di() {
    let design =
        std::fs::read_to_string("docs/superpowers/specs/2026-05-02-restir-di-vpt-design.md")
            .expect("ReSTIR-DI VPT design doc should be readable");

    assert!(
        design.contains("Current VPT mode does not register the primary-ray graph pass before VPT")
    );
    assert!(
        design.contains("must explicitly add a VPT-mode surface-state pass")
            || design.contains("Do not write a ReSTIR-DI pass that silently assumes `gbuffer_pos`")
    );
}

#[test]
fn vpt_temporal_barriers_are_owned_by_render_graph() {
    let vpt_source =
        std::fs::read_to_string("src/render/passes/vpt.rs").expect("vpt source is readable");
    let temporal_source = std::fs::read_to_string("src/render/passes/vpt_temporal.rs")
        .expect("vpt temporal source is readable");
    let postprocess_source = std::fs::read_to_string("src/render/passes/postprocess.rs")
        .expect("postprocess source is readable");
    let pipeline_source = source("src/render/vpt_pipeline.rs");
    let implementation = vpt_source
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    assert!(!implementation.contains("cmd_pipeline_barrier"));
    assert!(!implementation.contains("ImageMemoryBarrier"));
    assert!(temporal_source.contains("let temporal_initial_access = if history_initialized"));
    assert!(temporal_source.contains("let previous_temporal_access = if history_initialized"));
    assert!(postprocess_source.contains("let output_initial_access = if output_initialized"));
    assert!(pipeline_source.contains("vpt_temporal.register_graph("));
    assert!(pipeline_source.contains("postprocess.register_graph("));
}

#[test]
fn vpt_trace_shader_routes_area_restir_debug_views_to_noisy_output() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let temporal = normalized_source(include_str!(
        "../../../../assets/shaders/passes/vpt_temporal.slang"
    ));

    for token in [
        "VPT_DEBUG_VIEW_AREA_SUBPIXEL",
        "VPT_DEBUG_VIEW_AREA_LENS",
        "VPT_DEBUG_VIEW_AREA_WEIGHT",
        "VPT_DEBUG_VIEW_AREA_HISTORY_VALID",
        "VPT_DEBUG_VIEW_AREA_REJECTION",
        "VPT_DEBUG_VIEW_AREA_JACOBIAN",
        "visualize_area_restir_debug",
        "area_restir_reservoirs[index]",
        "reservoir.sample_state.subpixel_uv",
        "reservoir.sample_state.lens_uv",
        "reservoir.rejection_reason",
        "reservoir.jacobian",
    ] {
        assert!(
            source.contains(token),
            "VPT shader missing Area ReSTIR debug token {token}"
        );
    }

    for token in [
        "VPT_DEBUG_VIEW_AREA_SUBPIXEL",
        "VPT_DEBUG_VIEW_AREA_LENS",
        "VPT_DEBUG_VIEW_AREA_WEIGHT",
        "VPT_DEBUG_VIEW_AREA_HISTORY_VALID",
        "VPT_DEBUG_VIEW_AREA_REJECTION",
        "VPT_DEBUG_VIEW_AREA_JACOBIAN",
    ] {
        assert!(
            temporal.contains(token),
            "VPT temporal shader must consume and bypass Area ReSTIR debug token {token}"
        );
    }
}

#[test]
fn app_maps_area_restir_debug_setting_to_visible_vpt_debug_view() {
    let source = normalized_source(include_str!("../../../app.rs"));

    for token in [
        "fn area_restir_debug_to_vpt_debug_view",
        "AreaRestirDebugView::Subpixel => Some(VptDebugView::AreaSubpixel)",
        "AreaRestirDebugView::Lens => Some(VptDebugView::AreaLens)",
        "AreaRestirDebugView::Weight => Some(VptDebugView::AreaWeight)",
        "AreaRestirDebugView::HistoryValid => Some(VptDebugView::AreaHistoryValid)",
        "AreaRestirDebugView::Rejection => Some(VptDebugView::AreaRejection)",
        "AreaRestirDebugView::Jacobian => Some(VptDebugView::AreaJacobian)",
        "self.lighting_settings.vpt_debug_view = vpt_debug_view",
    ] {
        assert!(
            source.contains(token),
            "app missing Area ReSTIR debug bridge token {token}"
        );
    }
}
