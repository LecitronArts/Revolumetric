use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShaderCompileMode {
    Auto,
    Strict,
    Skip,
}

fn main() {
    println!("cargo:rerun-if-env-changed=REVOLUMETRIC_SHADER_COMPILE");
    println!("cargo:rerun-if-changed=assets/shaders");
    println!("cargo:rerun-if-changed=assets/shaders/passes");

    let shader_dir = Path::new("assets/shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shaders");
    std::fs::create_dir_all(&out_dir).unwrap();
    let shader_compile_mode = shader_compile_mode();

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

    let pass_paths = walkdir::WalkDir::new(passes_dir.as_path())
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "slang"))
        .map(|entry| entry.path().to_path_buf())
        .collect::<Vec<_>>();

    if shader_compile_mode == ShaderCompileMode::Skip {
        println!(
            "cargo:warning=REVOLUMETRIC_SHADER_COMPILE=skip, writing placeholder shader files"
        );
        write_placeholder_spirv_files(&pass_paths, &out_dir);
        return;
    }

    for path in &pass_paths {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let spv_path = out_dir.join(format!("{stem}.spv"));

        let status = Command::new("slangc")
            .arg(path)
            .arg("-target")
            .arg("spirv")
            .arg("-entry")
            .arg("main")
            .arg("-stage")
            .arg("compute")
            .arg("-o")
            .arg(&spv_path)
            .arg("-I")
            .arg(shader_dir.join("shared"))
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Compiled {}", path.display());
            }
            Ok(s) => {
                panic!(
                    "slangc failed for {} with exit code {:?}; set REVOLUMETRIC_SHADER_COMPILE=skip only for CPU-only test environments",
                    path.display(),
                    s.code()
                );
            }
            Err(e) => match shader_compile_mode {
                ShaderCompileMode::Auto => {
                    println!(
                        "cargo:warning=slangc not found ({e}), writing placeholder shader files"
                    );
                    write_placeholder_spirv_files(&pass_paths, &out_dir);
                    return;
                }
                ShaderCompileMode::Strict => {
                    panic!(
                        "slangc not found ({e}); install slangc or set REVOLUMETRIC_SHADER_COMPILE=skip for CPU-only test environments"
                    );
                }
                ShaderCompileMode::Skip => unreachable!("skip mode returns before invoking slangc"),
            },
        }
    }
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

fn write_placeholder_spirv_files(pass_paths: &[PathBuf], out_dir: &Path) {
    for path in pass_paths {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let spv_path = out_dir.join(format!("{stem}.spv"));
        std::fs::write(spv_path, []).unwrap();
    }
}
