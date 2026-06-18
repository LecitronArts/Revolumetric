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
}
