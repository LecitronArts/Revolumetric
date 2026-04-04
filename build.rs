use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let shader_dir = Path::new("assets/shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shaders");
    std::fs::create_dir_all(&out_dir).unwrap();

    println!("cargo:rerun-if-changed=assets/shaders");

    // Find all .slang files in passes/
    let passes_dir = shader_dir.join("passes");
    if !passes_dir.exists() {
        return;
    }

    for entry in walkdir::WalkDir::new(&passes_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "slang"))
    {
        let path = entry.path();
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let spv_path = out_dir.join(format!("{stem}.spv"));

        let status = Command::new("slangc")
            .arg(path)
            .arg("-target").arg("spirv")
            .arg("-entry").arg("main")
            .arg("-stage").arg("compute")
            .arg("-o").arg(&spv_path)
            .arg("-I").arg(shader_dir.join("shared"))
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Compiled {}", path.display());
            }
            Ok(s) => {
                panic!("slangc failed for {} with exit code {:?}", path.display(), s.code());
            }
            Err(e) => {
                println!("cargo:warning=slangc not found ({e}), skipping shader compilation");
                // Write a placeholder so the build doesn't fail when slangc isn't installed
                std::fs::write(&spv_path, &[]).unwrap();
                break;
            }
        }
    }
}
