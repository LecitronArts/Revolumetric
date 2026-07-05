use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderJobSpec {
    pub path: PathBuf,
    pub stage: &'static str,
    pub output_stem: String,
}

pub fn ui_shader_jobs(shader_dir: &Path) -> Vec<ShaderJobSpec> {
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
    .map(|(output_stem, stage, path)| ShaderJobSpec {
        path,
        stage,
        output_stem: output_stem.to_owned(),
    })
    .collect()
}

pub fn pass_shader_jobs(shader_dir: &Path) -> Vec<ShaderJobSpec> {
    let mut jobs = Vec::new();
    collect_pass_shader_jobs(&shader_dir.join("passes"), &mut jobs);
    jobs
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn rt_shader_jobs(shader_dir: &Path) -> Vec<ShaderJobSpec> {
    pass_shader_jobs(shader_dir)
        .into_iter()
        .filter(|job| is_rt_shader_stage(job.stage))
        .collect()
}

fn collect_pass_shader_jobs(dir: &Path, jobs: &mut Vec<ShaderJobSpec>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_pass_shader_jobs(&path, jobs);
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.ends_with(".slang") {
            continue;
        }

        let stage = shader_stage_for_pass_shader(file_name);
        let output_stem = output_stem_for_shader(file_name);
        jobs.push(ShaderJobSpec {
            path,
            stage,
            output_stem,
        });
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn shader_stage_for_pass_shader(file_name: &str) -> &'static str {
    if file_name.ends_with(".rgen.slang") {
        "raygeneration"
    } else if file_name.ends_with(".rmiss.slang") {
        "miss"
    } else if file_name.ends_with(".rchit.slang") {
        "closesthit"
    } else if file_name.ends_with(".rahit.slang") {
        "anyhit"
    } else if file_name.ends_with(".rint.slang") {
        "intersection"
    } else if file_name.ends_with(".rcall.slang") {
        "callable"
    } else {
        "compute"
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn is_rt_shader_stage(stage: &str) -> bool {
    matches!(
        stage,
        "raygeneration" | "miss" | "closesthit" | "anyhit" | "intersection" | "callable"
    )
}

#[cfg_attr(not(test), allow(dead_code))]
fn output_stem_for_shader(file_name: &str) -> String {
    let stem = file_name.strip_suffix(".slang").unwrap_or(file_name);
    stem.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, time::SystemTime};

    fn unique_temp_dir() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system time should be after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("revolumetric_build_support_{suffix}"))
    }

    #[test]
    fn ui_shader_jobs_only_collects_existing_ui_shaders() {
        let root = unique_temp_dir();
        let ui_dir = root.join("ui");
        fs::create_dir_all(&ui_dir).expect("temp ui dir should be creatable");
        fs::write(ui_dir.join("egui.frag.slang"), "// fragment").expect("frag file should exist");

        let jobs = ui_shader_jobs(&root);

        assert_eq!(
            jobs,
            vec![ShaderJobSpec {
                path: ui_dir.join("egui.frag.slang"),
                stage: "fragment",
                output_stem: "egui.frag".to_string(),
            }]
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rt_shader_jobs_map_suffixes_to_ray_tracing_stages() {
        let root = unique_temp_dir();
        let passes_dir = root.join("passes");
        fs::create_dir_all(&passes_dir).expect("temp passes dir should be creatable");

        fs::write(passes_dir.join("rt_surface.rgen.slang"), "// raygen")
            .expect("raygen file should exist");
        fs::write(passes_dir.join("rt_surface.rmiss.slang"), "// miss")
            .expect("miss file should exist");
        fs::write(passes_dir.join("rt_surface.rchit.slang"), "// closest hit")
            .expect("closest-hit file should exist");
        fs::write(passes_dir.join("rt_surface.rint.slang"), "// intersection")
            .expect("intersection file should exist");
        fs::write(passes_dir.join("vpt.slang"), "// compute").expect("compute file should exist");

        let all_jobs = pass_shader_jobs(&root);
        assert!(all_jobs.iter().any(|job| {
            job.path.ends_with("vpt.slang") && job.stage == "compute" && job.output_stem == "vpt"
        }));

        let jobs = rt_shader_jobs(&root);

        assert_eq!(jobs.len(), 4);
        assert!(jobs.iter().any(|job| {
            job.path.ends_with("rt_surface.rgen.slang")
                && job.stage == "raygeneration"
                && job.output_stem == "rt_surface.rgen"
        }));
        assert!(jobs.iter().any(|job| {
            job.path.ends_with("rt_surface.rmiss.slang")
                && job.stage == "miss"
                && job.output_stem == "rt_surface.rmiss"
        }));
        assert!(jobs.iter().any(|job| {
            job.path.ends_with("rt_surface.rchit.slang")
                && job.stage == "closesthit"
                && job.output_stem == "rt_surface.rchit"
        }));
        assert!(jobs.iter().any(|job| {
            job.path.ends_with("rt_surface.rint.slang")
                && job.stage == "intersection"
                && job.output_stem == "rt_surface.rint"
        }));
        assert!(!jobs.iter().any(|job| job.path.ends_with("vpt.slang")));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rt_shader_jobs_keep_unique_output_stems_for_all_rt_stages() {
        let root = unique_temp_dir();
        let passes_dir = root.join("passes");
        fs::create_dir_all(&passes_dir).expect("temp passes dir should be creatable");

        for suffix in ["rgen", "rmiss", "rchit", "rint"] {
            fs::write(
                passes_dir.join(format!("rt_surface.{suffix}.slang")),
                format!("// {suffix}"),
            )
            .expect("rt shader stage file should exist");
        }

        let jobs = rt_shader_jobs(&root);
        let mut stems = jobs
            .iter()
            .map(|job| job.output_stem.as_str())
            .collect::<Vec<_>>();
        stems.sort_unstable();
        stems.dedup();

        assert_eq!(
            stems.len(),
            jobs.len(),
            "RT shader stage outputs must not overwrite each other"
        );
        assert!(stems.contains(&"rt_surface.rgen"));
        assert!(stems.contains(&"rt_surface.rmiss"));
        assert!(stems.contains(&"rt_surface.rchit"));
        assert!(stems.contains(&"rt_surface.rint"));

        let _ = fs::remove_dir_all(root);
    }
}
