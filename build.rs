use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "src/build_support.rs"]
mod build_support;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShaderCompileMode {
    Auto,
    Strict,
    Skip,
}

#[derive(Clone, Debug)]
struct ShaderJob {
    path: PathBuf,
    stage: &'static str,
    output_stem: String,
}

fn main() {
    println!("cargo:rerun-if-env-changed=REVOLUMETRIC_SHADER_COMPILE");
    println!("cargo:rerun-if-env-changed=REVOLUMETRIC_SLANGC");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DESKTOP");
    println!("cargo:rerun-if-changed=assets/shaders");
    println!("cargo:rerun-if-changed=assets/shaders/passes");

    let shader_dir = Path::new("assets/shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shaders");
    std::fs::create_dir_all(&out_dir).unwrap();
    let shader_compile_mode = shader_compile_mode();
    let desktop_feature = desktop_feature_enabled();

    // Track every shader file individually so edits trigger recompilation on
    // Windows NTFS (directory mtime doesn't update when file contents change).
    for entry in walkdir::WalkDir::new(shader_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "slang"))
    {
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }

    // Find all .slang files in passes/
    let passes_dir = shader_dir.join("passes");
    if !passes_dir.exists() {
        return;
    }

    let mut shader_jobs = build_support::pass_shader_jobs(shader_dir)
        .into_iter()
        .map(|job| ShaderJob {
            path: job.path,
            stage: job.stage,
            output_stem: job.output_stem,
        })
        .collect::<Vec<_>>();
    for job in ui_shader_jobs(shader_dir) {
        shader_jobs.push(job);
    }

    if shader_compile_mode == ShaderCompileMode::Skip {
        if desktop_feature {
            panic!(
                "REVOLUMETRIC_SHADER_COMPILE=skip cannot be used with the desktop feature; \
                 desktop builds require real shaders, otherwise the app opens a black window. \
                 Install slangc and use REVOLUMETRIC_SHADER_COMPILE=strict for runtime validation"
            );
        }
        println!(
            "cargo:warning=REVOLUMETRIC_SHADER_COMPILE=skip, writing placeholder shader files"
        );
        write_placeholder_spirv_files(&shader_jobs, &out_dir);
        return;
    }

    for job in &shader_jobs {
        let spv_path = out_dir.join(format!("{}.spv", job.output_stem));
        let reflection_json_path = out_dir.join(format!("{}.reflection.json", job.output_stem));

        let status = slangc_command()
            .arg(&job.path)
            .arg("-target")
            .arg("spirv")
            .arg("-entry")
            .arg("main")
            .arg("-stage")
            .arg(job.stage)
            .arg("-o")
            .arg(&spv_path)
            .arg("-reflection-json")
            .arg(&reflection_json_path)
            .arg("-I")
            .arg(shader_dir.join("shared"))
            .status();

        match status {
            Ok(s) if s.success() => {
                println!(
                    "cargo:warning=Compiled {} ({})",
                    job.path.display(),
                    job.stage
                );
            }
            Ok(s) => {
                panic!(
                    "slangc failed for {} with exit code {:?}; set REVOLUMETRIC_SHADER_COMPILE=skip only for CPU-only test environments",
                    job.path.display(),
                    s.code()
                );
            }
            Err(e) => match shader_compile_mode {
                ShaderCompileMode::Auto => {
                    if desktop_feature {
                        panic!(
                            "slangc not found ({e}); desktop builds require real shaders. \
                             Put slangc on PATH, or set REVOLUMETRIC_SLANGC to the absolute \
                             slangc.exe path in your launch environment"
                        );
                    }
                    println!(
                        "cargo:warning=slangc not found ({e}), writing placeholder shader files"
                    );
                    write_placeholder_spirv_files(&shader_jobs, &out_dir);
                    return;
                }
                ShaderCompileMode::Strict => {
                    panic!(
                        "slangc not found ({e}); install slangc, put it on PATH, set \
                         REVOLUMETRIC_SLANGC to the absolute slangc.exe path, or set \
                         REVOLUMETRIC_SHADER_COMPILE=skip for CPU-only test environments"
                    );
                }
                ShaderCompileMode::Skip => unreachable!("skip mode returns before invoking slangc"),
            },
        }
    }
}

fn ui_shader_jobs(shader_dir: &Path) -> Vec<ShaderJob> {
    build_support::ui_shader_jobs(shader_dir)
        .into_iter()
        .map(|job| ShaderJob {
            path: job.path,
            stage: job.stage,
            output_stem: job.output_stem,
        })
        .collect()
}

fn slangc_command() -> Command {
    match env::var_os("REVOLUMETRIC_SLANGC") {
        Some(path) => Command::new(path),
        None => Command::new("slangc"),
    }
}

fn desktop_feature_enabled() -> bool {
    env::var_os("CARGO_FEATURE_DESKTOP").is_some()
}

fn shader_compile_mode() -> ShaderCompileMode {
    match env::var("REVOLUMETRIC_SHADER_COMPILE") {
        Ok(value) => parse_shader_compile_mode(&value).unwrap_or_else(|| {
            panic!(
                "invalid REVOLUMETRIC_SHADER_COMPILE={value:?}; expected one of: auto, strict, skip"
            )
        }),
        Err(env::VarError::NotPresent) => ShaderCompileMode::Auto,
        Err(env::VarError::NotUnicode(value)) => {
            panic!(
                "invalid REVOLUMETRIC_SHADER_COMPILE={value:?}; expected valid Unicode: auto, strict, skip"
            )
        }
    }
}

fn parse_shader_compile_mode(value: &str) -> Option<ShaderCompileMode> {
    match value {
        "auto" => Some(ShaderCompileMode::Auto),
        "strict" => Some(ShaderCompileMode::Strict),
        "skip" => Some(ShaderCompileMode::Skip),
        _ => None,
    }
}

fn write_placeholder_spirv_files(shader_jobs: &[ShaderJob], out_dir: &Path) {
    for job in shader_jobs {
        let spv_path = out_dir.join(format!("{}.spv", job.output_stem));
        let reflection_json_path = out_dir.join(format!("{}.reflection.json", job.output_stem));
        std::fs::write(spv_path, []).unwrap();
        let _ = std::fs::remove_file(reflection_json_path);
    }
}
