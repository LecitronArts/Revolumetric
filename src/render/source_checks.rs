pub(crate) fn normalize(source: &str) -> String {
    source.replace("\r\n", "\n").replace('\r', "\n")
}

pub(crate) fn read_source(path: &str) -> String {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read source {path}: {error}"));
    normalize(&source)
}

pub(crate) fn compact(source: &str) -> String {
    source.split_whitespace().collect::<String>()
}

pub(crate) fn assert_contains_all(source: &str, tokens: &[&str], context: &str) {
    for token in tokens {
        assert!(source.contains(token), "{context} missing token {token}");
    }
}

pub(crate) fn assert_compact_contains_all(source: &str, tokens: &[&str], context: &str) {
    let compact_source = compact(source);
    for token in tokens {
        assert!(
            compact_source.contains(token),
            "{context} missing compact token {token}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visual_baseline_case<'a>(manifest: &'a str, name: &str) -> &'a str {
        let after_name = manifest
            .split(&format!("\"name\": \"{name}\""))
            .nth(1)
            .unwrap_or_else(|| panic!("visual baseline manifest should include {name}"));
        after_name
            .split("\n    },\n    {")
            .next()
            .expect("visual baseline case should end before the next case")
            .split("\n    }\n  ]")
            .next()
            .expect("visual baseline case should end before the cases array")
    }

    #[test]
    fn compact_ignores_line_endings_and_spacing() {
        let source = "builder\r\n    .add_binding(\r\n        6,\r\n    )";
        assert_eq!(compact(source), "builder.add_binding(6,)");
        assert_compact_contains_all(source, &["builder.add_binding(6,)"], "compact source");
    }

    #[test]
    fn normalize_converts_crlf_and_lone_cr_to_lf() {
        assert_eq!(normalize("a\r\nb\rc"), "a\nb\nc");
    }

    #[test]
    fn local_validation_script_exposes_cpu_strict_and_nrd_gates() {
        let script = read_source("run/validate-local.ps1");

        assert_contains_all(
            &script,
            &[
                "[switch]$StrictShaders",
                "[switch]$Nrd",
                "[switch]$NrdRuntime",
                "cargo fmt --check",
                "cargo test --lib",
                "cargo clippy --all-targets -- -D warnings",
                "REVOLUMETRIC_SHADER_COMPILE",
                "REVOLUMETRIC_NRD_ROOT",
                ".\\run\\validate-nrd.ps1",
                "-Denoiser reblur",
                "-Frames $NrdFrames",
            ],
            "local validation script",
        );

        assert!(
            script.contains("$previousEnv") && script.contains("SetEnvironmentVariable"),
            "local validation script must restore process environment variables"
        );
    }

    #[test]
    fn readme_documents_local_validation_and_native_reblur_status() {
        let readme = read_source("README.md");
        let compact_readme = compact(&readme);

        assert_contains_all(
            &readme,
            &[
                ".\\run\\validate-local.ps1",
                ".\\run\\validate-local.ps1 -StrictShaders",
                ".\\run\\validate-local.ps1 -StrictShaders -Nrd",
                ".\\run\\validate-nrd.ps1 -Denoiser reblur -Frames 3",
                "`relax` and `reblur` use the native NRD path when the `nrd` Cargo feature and NRD SDK are available",
            ],
            "README validation and ReBLUR status",
        );
        assert!(
            compact_readme.contains("REVOLUMETRIC_DENOISER=off|svgf|relax|reblur"),
            "README must keep the full denoiser mode list visible"
        );
        assert_contains_all(
            &readme,
            &[
                "RT ReSTIR defaults are enabled on the hardware RT backend",
                "`REVOLUMETRIC_RT_RESTIR_DI=on|off|1|0|true|false`: enables RT ReSTIR-DI direct-light reservoirs. Default is `on`.",
                "`REVOLUMETRIC_RT_RESTIR_DI_SPATIAL=on|off|1|0|true|false`: enables RT ReSTIR-DI spatial reservoir reuse after temporal history is valid. Default is `on`.",
                "`REVOLUMETRIC_RT_RESTIR_GI=on|off|1|0|true|false`: enables RT ReSTIR-GI after RT surface generation.",
                "RT path still does not enable NRD, SER, or path guiding. Default is `on`.",
                "$env:REVOLUMETRIC_RENDER_MODE='rt'",
                "$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='2'",
                "cargo run --features desktop --bin revolumetric # explicit RT ReSTIR default smoke",
                "default auto backend smoke",
            ],
            "README RT validation smoke docs",
        );
    }

    #[test]
    fn github_ci_uses_local_validation_entrypoint_for_cpu_gate() {
        let workflow = read_source(".github/workflows/ci.yml");

        assert_contains_all(
            &workflow,
            &[
                "windows-latest",
                ".\\run\\validate-local.ps1",
                "REVOLUMETRIC_SHADER_COMPILE",
                "cargo --version",
            ],
            "GitHub CI workflow",
        );
    }

    #[test]
    fn visual_baseline_script_validates_captures_metadata_and_nonblank_ppm() {
        let script = read_source("run/validate-visual-baseline.ps1");

        assert_contains_all(
            &script,
            &[
                "[string]$Manifest",
                "visual-baselines.json",
                "REVOLUMETRIC_CAPTURE_FRAME",
                "REVOLUMETRIC_CAPTURE_DIR",
                "REVOLUMETRIC_CAPTURE_PREFIX",
                "REVOLUMETRIC_RENDER_MODE",
                "ConvertFrom-Json",
                "Assert-CaptureMetadata",
                "Assert-MetadataBooleanField",
                "$Metadata.PSObject.Properties.Name -contains $FieldName",
                "Assert-PpmMatchesMetadata",
                "Measure-PpmSignal",
                "Assert-PpmSignal",
                "Assert-PpmSignal -PpmPath $ppmPath -Case $case",
                "NonZeroPixelRatio",
                "RgbRange",
                "Test-CaseProperty",
                "Test-CaseProperty -Case $Case -FieldName \"expectedMinNonZeroPixelRatio\"",
                "Test-CaseProperty -Case $Case -FieldName \"expectedMinRgbRange\"",
                "expectedMinNonZeroPixelRatio",
                "expectedMinRgbRange",
                "PPM signal threshold expectedMinNonZeroPixelRatio was null",
                "PPM signal threshold expectedMinRgbRange was null",
                "[switch]$Rt",
                "requiresRt",
                "REVOLUMETRIC_RT_DEBUG_VIEW",
                "REVOLUMETRIC_RT_RESTIR_DI",
                "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL",
                "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES",
                "REVOLUMETRIC_RT_RESTIR_GI",
                "REVOLUMETRIC_RT_TEMPORAL_DENOISE",
                "render_backend",
                "render_mode",
                "rt_debug_view",
                "rt_restir_di_enabled",
                "rt_restir_di_spatial_enabled",
                "rt_restir_di_spatial_sample_count",
                "rt_restir_gi_enabled",
                "rt_temporal_denoise_enabled",
                "rt_frame_rendered",
                "rt_restir_di_rendered",
                "rt_restir_gi_rendered",
                "rt_resolve_ready",
                ".\\run\\validate-nrd.ps1",
                "cargo run --features desktop --bin revolumetric",
            ],
            "visual baseline script",
        );

        assert!(
            !script.contains("Assert-PpmHasNonZeroRgb"),
            "visual baseline script must not keep the old one-byte non-zero PPM gate"
        );
    }

    #[test]
    fn visual_baseline_manifest_covers_svgf_and_reblur_debug_cases() {
        let manifest = read_source("run/visual-baselines.json");

        assert_contains_all(
            &manifest,
            &[
                "\"captureFrame\": 2",
                "\"frames\": 3",
                "\"name\": \"svgf_final\"",
                "\"renderMode\": \"vpt\"",
                "\"denoiser\": \"svgf\"",
                "\"expectedEffectiveDenoiser\": \"svgf\"",
                "\"name\": \"reblur_final\"",
                "\"denoiser\": \"reblur\"",
                "\"expectedEffectiveDenoiser\": \"reblur\"",
                "\"name\": \"reblur_nrd_validation\"",
                "\"debugView\": \"nrd_validation\"",
                "\"name\": \"rt_surface_debug\"",
                "\"renderMode\": \"rt\"",
                "\"requiresRt\": true",
                "\"expectedRenderBackend\": \"rt\"",
                "\"rtDebugView\": \"surface\"",
                "\"rtRestirDi\": true",
                "\"rtRestirDiSpatial\": true",
                "\"rtRestirDiSpatialSamples\": 4",
                "\"rtRestirGi\": true",
                "\"rtTemporalDenoise\": true",
                "\"expectedRtFrameRendered\": true",
                "\"expectedRtRestirDiRendered\": true",
                "\"expectedRtRestirGiRendered\": true",
                "\"expectedRtResolveReady\": true",
            ],
            "visual baseline manifest",
        );

        let svgf_case = visual_baseline_case(&manifest, "svgf_final");
        assert_contains_all(
            svgf_case,
            &[
                "\"expectedMinNonZeroPixelRatio\": 0.25",
                "\"expectedMinRgbRange\": 32",
            ],
            "svgf_final visual baseline case",
        );

        let rt_case = visual_baseline_case(&manifest, "rt_surface_debug");
        assert_contains_all(
            rt_case,
            &[
                "\"expectedMinNonZeroPixelRatio\": 0.25",
                "\"expectedMinRgbRange\": 32",
            ],
            "rt_surface_debug visual baseline case",
        );

        for nrd_case_name in ["reblur_final", "reblur_nrd_validation"] {
            let nrd_case = visual_baseline_case(&manifest, nrd_case_name);
            assert!(
                !nrd_case.contains("\"expectedMinNonZeroPixelRatio\""),
                "{nrd_case_name} should not require measured PPM signal coverage"
            );
            assert!(
                !nrd_case.contains("\"expectedMinRgbRange\""),
                "{nrd_case_name} should not require measured PPM RGB range"
            );
        }
    }

    #[test]
    fn docs_expose_visual_baseline_validation_entrypoint() {
        let readme = read_source("README.md");
        let run_readme = read_source("run/README.md");

        assert_contains_all(
            &readme,
            &[
                ".\\run\\validate-visual-baseline.ps1",
                ".\\run\\validate-visual-baseline.ps1 -Nrd",
                ".\\run\\validate-visual-baseline.ps1 -Rt",
                "visual regression baseline",
                "hardware RT capture",
                "render_backend",
                "render_mode",
                "rt_debug_view",
                "active RT frame/pass state",
            ],
            "README visual baseline docs",
        );
        assert_contains_all(
            &run_readme,
            &[
                ".\\run\\validate-visual-baseline.ps1",
                ".\\run\\validate-visual-baseline.ps1 -Rt",
                "run/visual-baselines.json",
                "PPM",
                "metadata",
                "requires RT-capable hardware",
                "render_backend",
                "RT metadata",
                "active RT frame/pass state",
            ],
            "run README visual baseline docs",
        );
    }

    #[test]
    fn traversal_stats_are_wired_to_vpt_shaders_passes_and_docs() {
        let scene_common = read_source("assets/shaders/shared/scene_common.slang");
        let traverse = read_source("assets/shaders/shared/voxel_traverse.slang");
        let vpt = read_source("assets/shaders/passes/vpt.slang");
        let vpt_surface = read_source("assets/shaders/passes/vpt_surface.slang");
        let area_initial = read_source("assets/shaders/passes/area_restir_initial.slang");
        let vpt_pass = read_source("src/render/passes/vpt.rs");
        let vpt_surface_pass = read_source("src/render/passes/vpt_surface.rs");
        let area_pass = read_source("src/render/passes/area_restir.rs");
        let pipeline = read_source("src/render/vpt_pipeline.rs");
        let runtime = read_source("src/render/runtime.rs");
        let readme = read_source("README.md");

        assert_contains_all(
            &scene_common,
            &[
                "LIGHTING_FLAG_VPT_TRAVERSAL_STATS",
                "bool scene_vpt_traversal_stats_enabled",
            ],
            "scene shader traversal stats flag",
        );
        assert_contains_all(
            &traverse,
            &[
                "VPT_TRAVERSAL_STAT_PRIMARY_RAYS",
                "VPT_TRAVERSAL_STAT_SHADOW_RAYS",
                "VPT_TRAVERSAL_STAT_HIERARCHY_SKIP_TESTS",
                "VPT_TRAVERSAL_STAT_HIERARCHY_SKIPS_ACCEPTED",
                "VPT_TRAVERSAL_STAT_BRICK_DDA_CALLS",
                "VPT_TRAVERSAL_STAT_BRICK_DDA_STEPS",
                "VPT_TRAVERSAL_STAT_BRICK_ANY_HIT_CALLS",
                "VPT_TRAVERSAL_STAT_BRICK_ANY_HIT_STEPS",
                "vpt_traversal_stats_add(",
                "RWStructuredBuffer<uint> stats",
                "InterlockedAdd(stats[counter_index], value",
            ],
            "shared traversal stats shader",
        );

        for (name, shader, binding) in [
            ("vpt", vpt.as_str(), "[[vk::binding(19, 0)]]"),
            (
                "vpt_surface",
                vpt_surface.as_str(),
                "[[vk::binding(23, 0)]]",
            ),
            (
                "area_restir_initial",
                area_initial.as_str(),
                "[[vk::binding(15, 0)]]",
            ),
        ] {
            assert!(
                shader.contains(binding),
                "{name} shader must bind traversal stats at {binding}"
            );
            assert!(
                shader.contains("RWStructuredBuffer<uint> traversal_stats"),
                "{name} shader must expose traversal stats storage buffer"
            );
            assert!(
                shader.contains("scene_vpt_traversal_stats_enabled("),
                "{name} shader must gate counter writes on SceneUniforms"
            );
        }

        assert_contains_all(
            &vpt_pass,
            &[
                "DescriptorBindingSpec::compute(19, vk::DescriptorType::STORAGE_BUFFER)",
                "update_traversal_stats_descriptor",
                "traversal_stats_resource",
                "AccessKind::ComputeShaderReadWrite",
            ],
            "VPT pass traversal stats binding",
        );
        assert_contains_all(
            &vpt_surface_pass,
            &[
                "DescriptorBindingSpec::compute(23, vk::DescriptorType::STORAGE_BUFFER)",
                "update_traversal_stats_descriptor",
                "traversal_stats_resource",
                "AccessKind::ComputeShaderReadWrite",
            ],
            "VPT surface pass traversal stats binding",
        );
        assert_contains_all(
            &area_pass,
            &[
                "DescriptorBindingSpec::compute(15, vk::DescriptorType::STORAGE_BUFFER)",
                "update_traversal_stats_descriptors",
                "traversal_stats_resource",
                "AccessKind::ComputeShaderReadWrite",
            ],
            "Area ReSTIR traversal stats binding",
        );
        assert_contains_all(
            &pipeline,
            &[
                "VptTraversalStatsBuffer",
                "traversal_stats_buffers",
                "traversal_stats_buffer(frame.frame_slot)",
                "VptFrameRecordResult",
                "traversal_stats: Option<VptTraversalStatsSnapshot>",
                "for buffer in self.traversal_stats_buffers",
                "buffer.destroy(device, allocator)",
            ],
            "VPT pipeline traversal stats readback",
        );
        assert_contains_all(
            &runtime,
            &[
                "record_result.traversal_stats",
                "wait_for_fence(submitted_fence)",
                "TraversalStats",
            ],
            "runtime traversal stats logging",
        );
        assert_contains_all(
            &readme,
            &[
                "REVOLUMETRIC_VPT_TRAVERSAL_STATS=on|off|1|0|true|false",
                "TraversalStats",
            ],
            "README traversal stats docs",
        );
    }

    #[test]
    fn vpt_nrd_adapter_keeps_frame_settings_in_focused_submodule() {
        let adapter = read_source("src/render/passes/vpt_nrd_adapter.rs");
        let frame_settings = read_source("src/render/passes/vpt_nrd_adapter/frame_settings.rs");

        assert_contains_all(
            &adapter,
            &[
                "mod frame_settings;",
                "pub use frame_settings::{",
                "VptNrdFrameSettings",
                "VptNrdFrameSettingsInputs",
            ],
            "VPT NRD adapter frame settings module wiring",
        );
        assert!(
            !adapter.contains("fn build_nrd_common_settings"),
            "VPT NRD adapter root should not own NRD frame settings construction"
        );
        assert_contains_all(
            &frame_settings,
            &[
                "pub struct VptNrdFrameSettingsInputs",
                "pub struct VptNrdFrameSettings",
                "pub(crate) fn build_initial_nrd_frame_settings",
                "pub(crate) fn validate_nrd_library_desc",
                "fn build_nrd_common_settings",
            ],
            "VPT NRD adapter frame settings module",
        );
    }

    #[test]
    fn workspace_exposes_dedicated_nrd_subcrates_and_root_render_module_stops_owning_them() {
        let cargo_toml = read_source("Cargo.toml");
        let render_mod = read_source("src/render/mod.rs");

        assert_contains_all(
            &cargo_toml,
            &["crates/revolumetric-nrd", "crates/revolumetric-nrd-sys"],
            "workspace Cargo manifest",
        );
        assert!(
            !render_mod.contains("pub mod nrd_adapter;"),
            "root render module must stop declaring nrd_adapter after the split"
        );
        assert!(
            !render_mod.contains("pub mod nrd_sys;"),
            "root render module must stop declaring nrd_sys after the split"
        );
    }

    #[test]
    fn rendergraph_exposes_graph_owned_transient_allocation_path() {
        let graph = read_source("src/render/graph.rs");
        let pipeline = read_source("src/render/vpt_pipeline.rs");
        let readme = read_source("README.md");

        assert_contains_all(
            &graph,
            &[
                "pub struct RenderGraphTransientResources",
                "compile_with_graph_owned_transients",
                "ensure_for_graph",
                "bind_transient_slot_images",
                "bind_transient_slot_buffers",
                "execute_with_transient_resources",
            ],
            "RenderGraph graph-owned transient allocation path",
        );
        assert_contains_all(
            &pipeline,
            &[
                "render_graph_transients",
                "ensure_render_graph_transients",
                "execute_with_transient_resources",
                "transients.destroy(device, allocator)",
            ],
            "VPT runtime graph-owned transient cache",
        );
        assert_contains_all(
            &readme,
            &[
                "graph-owned transient image and buffer allocation",
                "descriptor automation",
                "resource aliasing",
            ],
            "README RenderGraph transient allocation status",
        );
    }

    #[test]
    fn rendergraph_and_descriptor_regression_contracts_are_visible() {
        let graph = read_source("src/render/graph.rs");
        let descriptor = read_source("src/render/descriptor.rs");
        let shader_reflect = read_source("src/assets/shader_reflect.rs");
        let build_support = read_source("src/build_support.rs");

        assert_contains_all(
            &graph,
            &[
                "rendergraph_rejects_future_version_reads",
                "compile_rejects_stale_version_write_after_newer_write",
                "reads resource version",
                "writes stale resource version",
                "current version is",
            ],
            "RenderGraph version validation contract",
        );
        assert_contains_all(
            &descriptor,
            &[
                "assert_eq!(",
                "stage_flags",
                "count, 1",
                "descriptor binding {} must target compute or ray-tracing stages",
                "assert_specs_match_shader_bindings_rejects_wrong_stage_flags",
                "assert_specs_match_shader_bindings_rejects_wrong_descriptor_count",
                "assert_specs_match_shader_bindings_accepts_sampled_image_and_sampler_types",
            ],
            "descriptor validation contract",
        );
        assert_contains_all(
            &shader_reflect,
            &[
                "ui_shader_reflection_json_path_is_discovered",
                "parses_slang_ui_descriptor_bindings_from_source_fallback",
                "parses_slang_ui_descriptor_bindings_from_reflection_json_over_source",
                "parses_slang_ui_reflection_json_bindings",
                "Texture2D<",
                "SamplerState",
            ],
            "shader reflection UI contract",
        );
        assert_contains_all(
            &build_support,
            &[
                "pub fn ui_shader_jobs(shader_dir: &Path) -> Vec<ShaderJobSpec>",
                "egui.vert.slang",
                "egui.frag.slang",
                "ui_shader_jobs_only_collects_existing_ui_shaders",
            ],
            "build support UI shader job contract",
        );
    }

    #[test]
    fn build_script_compiles_rt_shader_jobs_instead_of_discarding_them() {
        let build = read_source("build.rs");
        let compact_build = compact(&build);

        assert!(
            !compact_build
                .contains("let_rt_shader_jobs=build_support::rt_shader_jobs(shader_dir);"),
            "build.rs must not discard RT shader job discovery"
        );
        assert!(
            compact_build.contains("build_support::pass_shader_jobs(shader_dir)")
                && !compact_build.contains("build_support::rt_shader_jobs(shader_dir)"),
            "pass shader discovery already includes RT suffixes, so build.rs must not duplicate RT jobs"
        );
    }

    #[test]
    fn hardware_rt_pipeline_source_contracts_are_visible() {
        let render_mod = read_source("src/render/mod.rs");
        let passes_mod = read_source("src/render/passes/mod.rs");
        let pipeline = read_source("src/render/rt_pipeline.rs");
        let rt_surface = read_source("src/render/passes/rt_surface.rs");
        let rt_temporal = read_source("src/render/passes/rt_temporal.rs");
        let rt_resolve = read_source("src/render/passes/rt_resolve.rs");
        let rt_common = read_source("assets/shaders/shared/rt_common.slang");
        let surface_shader = read_source("assets/shaders/passes/rt_surface.rgen.slang");
        let temporal_shader = read_source("assets/shaders/passes/rt_temporal.rgen.slang");
        let resolve_shader = read_source("assets/shaders/passes/rt_resolve.rgen.slang");

        assert_contains_all(
            &render_mod,
            &["pub mod rt_pipeline;"],
            "render RT pipeline module export",
        );
        assert_contains_all(
            &passes_mod,
            &[
                "pub mod rt_surface;",
                "pub mod rt_temporal;",
                "pub mod rt_resolve;",
            ],
            "RT pass module exports",
        );
        assert_contains_all(
            &pipeline,
            &[
                "pub struct RtRuntimePipeline",
                "rt_surface",
                "rt_temporal",
                "rt_resolve",
                "record_and_execute_frame",
                "history_reset_generation",
                "as_rebuild_generation",
            ],
            "RT runtime pipeline source",
        );
        let surface = pipeline
            .find("let rt_surface_outputs")
            .expect("RT surface pass should be registered");
        let temporal = pipeline
            .find("let rt_temporal_outputs")
            .expect("RT temporal pass should be registered");
        let resolve = pipeline
            .find("let rt_resolve_outputs")
            .expect("RT resolve pass should be registered");
        assert!(
            surface < temporal && temporal < resolve,
            "RT pipeline must register surface before temporal before resolve"
        );
        assert_contains_all(
            &rt_surface,
            &[
                "pub struct RtSurfacePass",
                "RayTracingShaderWrite",
                "rt_surface.rgen.slang",
            ],
            "RT surface pass shell",
        );
        assert_contains_all(
            &rt_temporal,
            &[
                "pub struct RtTemporalPass",
                "RayTracingShaderRead",
                "RayTracingShaderWrite",
            ],
            "RT temporal pass shell",
        );
        assert_contains_all(
            &rt_resolve,
            &["pub struct RtResolvePass", "TransferRead"],
            "RT resolve pass shell",
        );
        assert_contains_all(
            &rt_common,
            &[
                "rt_history_common.slang",
                "rt_surface_common.slang",
                "rt_shadow_common.slang",
            ],
            "RT shared common shader umbrella",
        );
        assert_contains_all(
            &surface_shader,
            &[
                "rt_history_common.slang",
                "AccelerationStructureKHR",
                "RtSurfacePixel",
                "RtHistoryUniforms",
            ],
            "RT surface raygen shader",
        );
        assert_contains_all(
            &temporal_shader,
            &["reproject", "clamp", "rt_history.flags"],
            "RT temporal raygen shader",
        );
        assert_contains_all(
            &resolve_shader,
            &["rt_history_common.slang", "temporal_radiance"],
            "RT resolve raygen shader",
        );
    }

    #[test]
    fn hardware_rt_restir_di_temporal_contract_is_visible() {
        let passes_mod = read_source("src/render/passes/mod.rs");
        let pass = read_source("src/render/passes/rt_restir_di.rs");
        let shader = read_source("assets/shaders/passes/rt_restir_di.rgen.slang");
        let spatial_shader = read_source("assets/shaders/passes/rt_restir_di_spatial.rgen.slang");
        let readme = read_source("README.md");

        assert_contains_all(
            &passes_mod,
            &["pub mod rt_restir_di;"],
            "RT ReSTIR-DI pass module export",
        );
        assert_contains_all(
            &pass,
            &[
                "pub struct RtRestirDiPass",
                "RayTracingPipeline",
                "ShaderBindingTable",
                "GpuRestirDiUniforms",
                "GpuDirectLight",
                "GpuRestirDiReservoir",
                "selected_reservoirs",
                "surface_history_buffers",
                "temporal_reservoirs",
                "spatial_pipeline",
                "spatial_shader_binding_table",
                "selected_current_buffer",
                "selected_history_buffer",
                "create_direct_light_buffer",
                "create_reservoir_buffer",
                "RayTracingPipeline::new_raygen_only",
                "cmd_trace_rays",
                "rt_restir_di_initial",
                "rt_restir_di_spatial",
            ],
            "RT ReSTIR-DI pass shell",
        );
        assert_contains_all(
            &pass,
            &[
                "DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::UNIFORM_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(3, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(4, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(5, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(6, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(7, vk::DescriptorType::UNIFORM_BUFFER)",
            ],
            "RT ReSTIR-DI descriptor bindings",
        );
        assert_contains_all(
            &shader,
            &[
                "#include \"restir_di_common.slang\"",
                "#include \"rt_history_common.slang\"",
                "[shader(\"raygeneration\")]",
                "ConstantBuffer<RestirDiUniforms> restir",
                "StructuredBuffer<DirectLight> direct_lights",
                "StructuredBuffer<RtSurfacePixel> surface_pixels",
                "StructuredBuffer<RestirDiReservoir> history_reservoirs",
                "RWStructuredBuffer<RestirDiReservoir> output_reservoirs",
                "StructuredBuffer<RtSurfacePixel> previous_surface_history",
                "RWStructuredBuffer<RtSurfacePixel> current_surface_history",
                "ConstantBuffer<RtHistoryUniforms> rt_history",
                "restir_di_target_pdf_for_light_sample",
                "output_reservoirs[index] = invalid_reservoir();",
                "current_surface_history[index] = surface;",
                "rt_restir_reproject",
                "rt_history.previous_view_proj",
                "rt_history.previous_resolution",
                "rt_restir_surfaces_compatible",
                "restir_di_reservoir_stream_weight",
                "restir_di_finalize_reservoir_on_surface_with_target",
                "surface.hit_kind != RT_SURFACE_HIT_KIND_VOXEL",
                "restir.light_count == 0u",
            ],
            "RT ReSTIR-DI temporal raygen shader",
        );
        assert!(
            !shader.contains("spatial_reservoir"),
            "RT ReSTIR-DI keeps spatial reuse out of this phase"
        );
        assert_contains_all(
            &spatial_shader,
            &[
                "#include \"restir_di_common.slang\"",
                "#include \"rt_history_common.slang\"",
                "StructuredBuffer<RestirDiReservoir> temporal_reservoirs",
                "RWStructuredBuffer<RestirDiReservoir> output_reservoirs",
                "StructuredBuffer<RtSurfacePixel> surface_pixels",
                "ConstantBuffer<RtHistoryUniforms> rt_history",
                "rt_restir_spatial_offsets",
                "rt_restir_spatial_surfaces_compatible",
                "rt_history.normal_threshold",
                "rt_history.depth_threshold",
                "restir_di_reservoir_stream_weight",
                "restir_di_finalize_reservoir_on_surface_with_target",
            ],
            "RT ReSTIR-DI spatial raygen shader",
        );
        assert_contains_all(
            &readme,
            &[
                "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL=on|off|1|0|true|false",
                "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES=0..8",
            ],
            "README RT ReSTIR-DI spatial docs",
        );
    }

    #[test]
    fn hardware_rt_restir_gi_contract_is_visible() {
        let render_mod = read_source("src/render/mod.rs");
        let passes_mod = read_source("src/render/passes/mod.rs");
        let settings = read_source("src/render/restir_gi.rs");
        let pipeline = read_source("src/render/rt_pipeline.rs");
        let pass = read_source("src/render/passes/rt_restir_gi.rs");
        let shader = read_source("assets/shaders/passes/rt_restir_gi.rgen.slang");
        let common = read_source("assets/shaders/shared/restir_gi_common.slang");
        let readme = read_source("README.md");

        assert_contains_all(&render_mod, &["pub mod restir_gi;"], "GI module export");
        assert_contains_all(
            &passes_mod,
            &["pub mod rt_restir_gi;"],
            "RT ReSTIR-GI pass module export",
        );
        assert_contains_all(
            &settings,
            &[
                "pub struct RestirGiSettings",
                "pub struct GpuRestirGiUniforms",
                "pub struct GpuRestirGiReservoir",
                "pub enum RestirGiDebugView",
            ],
            "RT ReSTIR-GI CPU/GPU ABI",
        );
        assert_contains_all(
            &pipeline,
            &[
                "RtRestirGiPass",
                "RtRestirGiShaders",
                "rt_restir_gi_pass",
                "restir_gi_history_initialized",
                "ensure_rt_restir_gi_pass",
                "rt_restir_gi.rgen.spv",
                "rt_restir_gi.rmiss.spv",
                "rt_restir_gi.rchit.spv",
                "rt_restir_gi.rint.spv",
                "restir_gi_enabled",
                "rt_restir_gi.update_tlas_descriptor",
                "rt_restir_gi.update_aabb_descriptor",
                "rt_restir_gi.update_ucvh_descriptors",
                "rt_restir_gi.register_graph",
            ],
            "RT pipeline GI wiring",
        );
        assert_contains_all(
            &pass,
            &[
                "pub struct RtRestirGiPass",
                "RayTracingPipeline",
                "ShaderBindingTable",
                "GpuRestirGiUniforms",
                "GpuRestirGiReservoir",
                "reservoirs",
                "surface_history_buffers",
                "current_reservoir_buffer",
                "history_reservoir_buffer",
                "RtRestirGiShaders",
                "RayTracingPipeline::new_surface_pipeline",
                "update_tlas_descriptor",
                "update_aabb_descriptor",
                "update_ucvh_descriptors",
                "write_traversal_stats_descriptor",
                "cmd_trace_rays",
                "rt_restir_gi",
            ],
            "RT ReSTIR-GI pass shell",
        );
        assert_contains_all(
            &shader,
            &[
                "#include \"restir_gi_common.slang\"",
                "#include \"rt_history_common.slang\"",
                "#include \"rt_surface_common.slang\"",
                "ConstantBuffer<RestirGiUniforms> restir_gi",
                "StructuredBuffer<RtSurfacePixel> surface_pixels",
                "StructuredBuffer<RestirGiReservoir> history_reservoirs",
                "RWStructuredBuffer<RestirGiReservoir> output_reservoirs",
                "StructuredBuffer<RtSurfacePixel> previous_surface_history",
                "RWStructuredBuffer<RtSurfacePixel> current_surface_history",
                "RaytracingAccelerationStructure scene_tlas",
                "TraceRay(",
                "rt_restir_gi_trace_indirect_surface",
                "restir_gi_finalize_reservoir",
                "rt_history.normal_threshold",
                "rt_history.depth_threshold",
            ],
            "RT ReSTIR-GI raygen shader",
        );
        assert_contains_all(
            &common,
            &[
                "struct RestirGiUniforms",
                "struct RestirGiReservoir",
                "restir_gi_is_valid_reservoir",
                "restir_gi_reservoir_stream_weight",
                "restir_gi_finalize_reservoir",
            ],
            "RT ReSTIR-GI common shader",
        );
        assert_contains_all(
            &readme,
            &[
                "REVOLUMETRIC_RT_RESTIR_GI=on|off",
                "traces a one-bounce hardware RT indirect sample",
                "The RT path still does not enable NRD, SER, or path guiding",
            ],
            "README RT ReSTIR-GI control",
        );
    }

    #[test]
    fn hardware_rt_direct_lighting_resolve_contract_is_visible() {
        let passes_mod = read_source("src/render/passes/mod.rs");
        let pass = read_source("src/render/passes/rt_direct_lighting.rs");
        let shader = read_source("assets/shaders/passes/rt_direct_lighting.rgen.slang");
        let miss = read_source("assets/shaders/passes/rt_direct_lighting.rmiss.slang");
        let closest_hit = read_source("assets/shaders/passes/rt_direct_lighting.rchit.slang");
        let intersection = read_source("assets/shaders/passes/rt_direct_lighting.rint.slang");

        assert_contains_all(
            &passes_mod,
            &["pub mod rt_direct_lighting;"],
            "RT direct-lighting pass module export",
        );
        assert_contains_all(
            &pass,
            &[
                "pub struct RtDirectLightingPass",
                "GpuRtDirectLightingUniforms",
                "RayTracingPipeline",
                "ShaderBindingTable",
                "fallback_direct_reservoirs",
                "fallback_indirect_reservoirs",
                "current_radiance",
                "RtDirectLightingShaders",
                "RayTracingPipeline::new_surface_pipeline",
                "cmd_trace_rays",
                "rt_direct_lighting",
                "update_tlas_descriptor",
                "update_aabb_descriptor",
                "update_ucvh_descriptors",
            ],
            "RT direct-lighting pass shell",
        );
        assert_contains_all(
            &pass,
            &[
                "DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::STORAGE_IMAGE)",
                "DescriptorBindingSpec::ray_tracing(3, vk::DescriptorType::UNIFORM_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(4, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)",
                "DescriptorBindingSpec::ray_tracing(5, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(6, vk::DescriptorType::UNIFORM_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(12, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(13, vk::DescriptorType::STORAGE_BUFFER)",
                "DescriptorBindingSpec::ray_tracing(14, vk::DescriptorType::STORAGE_BUFFER)",
            ],
            "RT direct-lighting descriptor bindings",
        );
        assert_contains_all(
            &shader,
            &[
                "#include \"restir_di_common.slang\"",
                "#include \"restir_gi_common.slang\"",
                "#include \"rt_history_common.slang\"",
                "struct RtDirectLightingUniforms",
                "StructuredBuffer<RtSurfacePixel> surface_pixels",
                "StructuredBuffer<RestirDiReservoir> direct_reservoirs",
                "StructuredBuffer<RestirGiReservoir> indirect_reservoirs",
                "RWTexture2D<float4> current_radiance",
                "RaytracingAccelerationStructure scene_tlas",
                "rt_direct.restir_gi_enabled",
                "RtShadowPayload",
                "TraceRay(",
                "rt_direct_light_visible",
                "restir_di_bounded_selected_weight",
                "restir_di_emissive_geometry_term",
                "RT_DEBUG_VIEW_DIRECT_RESERVOIR",
                "rt_direct_resolve_reservoir",
                "rt_direct_resolve_indirect_reservoir",
            ],
            "RT direct-lighting raygen shader",
        );
        assert_contains_all(
            &miss,
            &["[shader(\"miss\")]", "payload.occluded = 0u"],
            "RT direct-lighting shadow miss shader",
        );
        assert_contains_all(
            &closest_hit,
            &[
                "[shader(\"closesthit\")]",
                "trace_any_hit_ray_skip_voxel",
                "payload.skip_brick_id",
                "payload.skip_local",
            ],
            "RT direct-lighting shadow closest-hit shader",
        );
        assert_contains_all(
            &intersection,
            &["[shader(\"intersection\")]", "ReportHit(hit_t"],
            "RT direct-lighting shadow intersection shader",
        );
    }
}
