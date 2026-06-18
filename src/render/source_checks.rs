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
                "ConvertFrom-Json",
                "Assert-CaptureMetadata",
                "Assert-PpmMatchesMetadata",
                "Assert-PpmHasNonZeroRgb",
                ".\\run\\validate-nrd.ps1",
                "cargo run --features desktop --bin revolumetric",
            ],
            "visual baseline script",
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
                "\"denoiser\": \"svgf\"",
                "\"expectedEffectiveDenoiser\": \"svgf\"",
                "\"name\": \"reblur_final\"",
                "\"denoiser\": \"reblur\"",
                "\"expectedEffectiveDenoiser\": \"reblur\"",
                "\"name\": \"reblur_nrd_validation\"",
                "\"debugView\": \"nrd_validation\"",
            ],
            "visual baseline manifest",
        );
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
                "visual regression baseline",
            ],
            "README visual baseline docs",
        );
        assert_contains_all(
            &run_readme,
            &[
                ".\\run\\validate-visual-baseline.ps1",
                "run/visual-baselines.json",
                "PPM",
                "metadata",
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
}
