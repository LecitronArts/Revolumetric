use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_NRD");
    println!("cargo:rerun-if-env-changed=REVOLUMETRIC_NRD_ROOT");
    println!("cargo:rerun-if-changed=assets/shaders");
    println!("cargo:rerun-if-changed=assets/shaders/passes");

    if nrd_feature_enabled() {
        build_nrd_adapter();
    }

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

    let mut shader_jobs = walkdir::WalkDir::new(passes_dir.as_path())
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "slang"))
        .map(|entry| {
            let path = entry.path().to_path_buf();
            ShaderJob {
                output_stem: path.file_stem().unwrap().to_str().unwrap().to_owned(),
                path,
                stage: "compute",
            }
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
    [
        (
            "egui.vert",
            "vertex",
            shader_dir.join("ui").join("egui.vert.slang"),
        ),
        (
            "egui.frag",
            "fragment",
            shader_dir.join("ui").join("egui.frag.slang"),
        ),
    ]
    .into_iter()
    .filter(|(_, _, path)| path.exists())
    .map(|(output_stem, stage, path)| ShaderJob {
        path,
        stage,
        output_stem: output_stem.to_owned(),
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

fn nrd_feature_enabled() -> bool {
    env::var_os("CARGO_FEATURE_NRD").is_some()
}

fn build_nrd_adapter() {
    let root = validate_nrd_sdk_root();
    let link_dir = nrd_library_dir(&root);
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(root.join("Include"))
        .file("native/nrd_adapter.cpp")
        .compile("revolumetric_nrd_adapter");
    println!("cargo:rustc-link-search=native={}", link_dir.display());
    println!("cargo:rustc-link-lib=static=NRD");
    if nrd_shader_make_blob_library_exists(&link_dir) {
        println!("cargo:rustc-link-lib=static=ShaderMakeBlob");
    }
    println!("cargo:rerun-if-changed=native/nrd_adapter.h");
    println!("cargo:rerun-if-changed=native/nrd_adapter.cpp");
}

fn validate_nrd_sdk_root() -> PathBuf {
    let root = resolve_nrd_sdk_root();

    let required_files = [
        "Include/NRD.h",
        "Include/NRDDescs.h",
        "Include/NRDSettings.h",
    ];
    for relative_path in required_files {
        let full_path = root.join(relative_path);
        if !full_path.exists() {
            panic!(
                "REVOLUMETRIC_NRD_ROOT={:?} is missing required NRD SDK file {}; \
                 the nrd feature requires an accepted local NVIDIA RTX SDK checkout",
                root,
                full_path.display()
            );
        }
        println!("cargo:rerun-if-changed={}", full_path.display());
    }
    root
}

fn resolve_nrd_sdk_root() -> PathBuf {
    match env::var("REVOLUMETRIC_NRD_ROOT") {
        Ok(value) => PathBuf::from(value),
        Err(env::VarError::NotPresent) => {
            let local_root = PathBuf::from("run").join("nrd");
            println!("cargo:rerun-if-changed={}", local_root.display());
            if local_root.exists() {
                local_root
            } else {
                panic!(
                    "REVOLUMETRIC_NRD_ROOT is required when the nrd feature is enabled, \
                     or place an accepted local NVIDIA NRD SDK checkout under run/nrd"
                );
            }
        }
        Err(env::VarError::NotUnicode(value)) => {
            panic!(
                "invalid REVOLUMETRIC_NRD_ROOT={value:?}; expected a valid Unicode path \
                 to an accepted local NVIDIA NRD SDK checkout"
            );
        }
    }
}

fn nrd_library_dir(root: &Path) -> PathBuf {
    let candidates = [
        root.join("_Bin"),
        root.join("Lib"),
        root.join("lib"),
        root.join("Build/Release"),
        root.join("build/Release"),
        root.join("build/lib"),
    ];
    for candidate in candidates {
        if candidate.join("NRD.lib").exists() || candidate.join("libNRD.a").exists() {
            println!("cargo:rerun-if-changed={}", candidate.display());
            return candidate;
        }
    }
    panic!(
        "REVOLUMETRIC_NRD_ROOT={:?} contains NRD headers but no prebuilt static NRD library; \
         build the NVIDIA NRD SDK with NRD_STATIC_LIBRARY=ON, NRD_EMBEDS_SPIRV_SHADERS=ON, \
         NRD_SUPPORTS_HISTORY_CONFIDENCE=ON, then place NRD.lib or libNRD.a under _Bin, Lib, lib, \
         Build/Release, build/Release, or build/lib",
        root
    );
}

fn nrd_shader_make_blob_library_exists(link_dir: &Path) -> bool {
    let candidates = [
        link_dir.join("ShaderMakeBlob.lib"),
        link_dir.join("libShaderMakeBlob.a"),
    ];
    candidates.iter().any(|path| path.exists())
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
