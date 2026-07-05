use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_NRD");
    println!("cargo:rerun-if-env-changed=REVOLUMETRIC_NRD_ROOT");
    println!("cargo:rerun-if-changed=native/nrd_adapter.h");
    println!("cargo:rerun-if-changed=native/nrd_adapter.cpp");

    if !nrd_feature_enabled() {
        return;
    }

    let root = validate_nrd_sdk_root();
    let link_dir = nrd_library_dir(&root);
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(root.join("Include"))
        .include("native")
        .file("native/nrd_adapter.cpp")
        .compile("revolumetric_nrd_adapter");
    println!("cargo:rustc-link-search=native={}", link_dir.display());
    println!("cargo:rustc-link-lib=static=NRD");
    if nrd_shader_make_blob_library_exists(&link_dir) {
        println!("cargo:rustc-link-lib=static=ShaderMakeBlob");
    }
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

fn nrd_feature_enabled() -> bool {
    env::var_os("CARGO_FEATURE_NRD").is_some()
}

fn resolve_nrd_sdk_root() -> PathBuf {
    match env::var("REVOLUMETRIC_NRD_ROOT") {
        Ok(value) => PathBuf::from(value),
        Err(env::VarError::NotPresent) => {
            let local_root = workspace_root().join("run").join("nrd");
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

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
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
