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
}
