use crate::voxel::brick::{BRICK_EDGE, BRICK_VOLUME, BrickData, BrickOccupancy, VoxelCell};
use crate::voxel::morton;
use crate::voxel::sparse_index::{U64IndexMap, new_u64_index_map};
use crate::voxel::ucvh::Ucvh;
use crate::voxel::vox_loader::{
    VoxBounds, VoxLoadError, VoxScene, VoxTargetBounds, material_for_color, parse_vox,
};
use glam::{IVec2, IVec3, Mat4, UVec3, Vec2, Vec3};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub const DEFAULT_TEARDOWN_ZIP_MAP_PATH: &str = "assets/scenes/Vintessa Hills.zip";
//pub const LEGACY_TEARDOWN_ZIP_MAP_PATH: &str = "assets/scenes/Vintessa Hills.zip";
const TEARDOWN_NATIVE_VOXELS_PER_UNIT: u32 = 10;
const DOWNSAMPLED_VOX_PHASE_BUCKETS: i32 = 1;
const DOWNSAMPLED_VOX_LINEAR_QUANTIZATION: f32 = 4096.0;
const DEFAULT_TEARDOWN_DIR_CANDIDATES: &[&str] = &[
    "D:/SteamLibrary/steamapps/common/Teardown",
    "D:/Steam/steamapps/common/Teardown",
    "D:/Games/SteamLibrary/steamapps/common/Teardown",
    "C:/Program Files (x86)/Steam/steamapps/common/Teardown",
];

#[derive(Debug, Clone)]
pub struct TeardownZipLoadOptions {
    pub teardown_dir: Option<PathBuf>,
    pub voxels_per_unit: u32,
}

impl Default for TeardownZipLoadOptions {
    fn default() -> Self {
        Self {
            teardown_dir: None,
            voxels_per_unit: TEARDOWN_NATIVE_VOXELS_PER_UNIT,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TeardownZipWriteStats {
    pub vox_nodes: u64,
    pub vox_nodes_exported: u64,
    pub voxbox_nodes: u64,
    pub voxbox_nodes_exported: u64,
    pub instance_nodes: u64,
    pub instance_nodes_exported: u64,
    pub voxels_per_unit: u32,
    pub input_voxels: u64,
    pub written_voxels: u64,
    pub out_of_bounds_voxels: u64,
    pub unique_written_voxels: u64,
    pub target_bounds: Option<VoxTargetBounds>,
    pub source_bounds: Option<TeardownSourceBounds>,
    pub target_scale_millis: u32,
    pub missing_mod_refs: Vec<String>,
    pub missing_builtin_refs: Vec<String>,
    pub recursive_instance_refs: Vec<String>,
    pub malformed_xml_refs: Vec<String>,
    pub malformed_vox_refs: Vec<String>,
    pub downsampled_vox_nodes: u64,
    pub downsampled_vox_plan_hits: u64,
    pub downsampled_vox_plan_misses: u64,
    pub downsampled_vox_plans_prebuilt: u64,
    pub vox_cache_hits: u64,
    pub vox_cache_misses: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TeardownSourceBounds {
    pub min: IVec3,
    pub max_exclusive: IVec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ColorTint {
    rgb: [u8; 3],
}

#[derive(Debug, Clone)]
struct TeardownDebugVoxPlacement {
    path: String,
    object: Option<String>,
    source_min: IVec3,
    source_max_exclusive: IVec3,
    target_min: UVec3,
    target_max_exclusive: UVec3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeardownZipLoadError {
    InvalidDensity,
    Io(String),
    Zip(String),
    MainXmlMissing,
    Xml(String),
    Vox(VoxLoadError),
    UcvhCapacityExceeded {
        failed_bricks: u32,
        dropped_voxels: u64,
    },
}

impl From<VoxLoadError> for TeardownZipLoadError {
    fn from(value: VoxLoadError) -> Self {
        Self::Vox(value)
    }
}

pub fn default_zip_map_path() -> PathBuf {
    select_default_zip_map_path(&[
        Path::new(DEFAULT_TEARDOWN_ZIP_MAP_PATH),
        //Path::new(LEGACY_TEARDOWN_ZIP_MAP_PATH),
    ])
}

fn select_default_zip_map_path(candidates: &[&Path]) -> PathBuf {
    candidates
        .iter()
        .copied()
        .find(|path| path.exists())
        .unwrap_or_else(|| candidates[0])
        .to_path_buf()
}

pub fn load_teardown_zip_into_ucvh(
    zip_path: impl AsRef<Path>,
    ucvh: &mut Ucvh,
    options: TeardownZipLoadOptions,
) -> Result<TeardownZipWriteStats, TeardownZipLoadError> {
    if options.voxels_per_unit == 0 {
        return Err(TeardownZipLoadError::InvalidDensity);
    }

    let profile = std::env::var_os("REVOLUMETRIC_TEARDOWN_PROFILE").is_some();
    let load_start = Instant::now();
    // Capture teardown_dir before it's moved into the resource source open — workers
    // reopen their own sources and need it for BUILT-IN/ resource resolution.
    let worker_teardown_dir = options.teardown_dir.clone();
    let mut source = TeardownResourceSource::open(zip_path.as_ref(), options.teardown_dir)?;
    profile_teardown_phase(profile, "open", load_start.elapsed());
    let mut cache = VoxCache::default();
    let mut stats = TeardownZipWriteStats {
        voxels_per_unit: options.voxels_per_unit,
        ..Default::default()
    };
    let main_xml = source.read_main_xml()?;
    let identity = Mat4::IDENTITY;
    let mut bounds = SourceBounds::default();

    let phase_start = Instant::now();
    walk_xml_bytes(
        &main_xml,
        identity,
        &mut source,
        &mut cache,
        &mut stats,
        &mut (),
        true,
        true,
        &mut |pos, _material, _sink| {
            bounds.include(pos);
        },
        &mut |_min, _max_exclusive, _material, _sink| false,
        &mut |_path, _object, _scene, _tint, _matrix, _sink| false,
    )?;
    profile_teardown_phase(profile, "bounds", phase_start.elapsed());

    let Some(source_bounds) = bounds.finish() else {
        stats.missing_mod_refs = source.sorted_missing_mod();
        stats.missing_builtin_refs = source.sorted_missing_builtin();
        return Ok(stats);
    };

    let target_mapping = TargetMapping::new(source_bounds, ucvh.config.world_size);
    stats.source_bounds = Some(source_bounds.into());
    stats.target_scale_millis = target_mapping.scale_millis();
    let mut staged_bricks = StagedBricks::new(ucvh.config.brick_grid_size);
    let write_stats = TargetWriteStatsTracker::default();
    let density = source_density(&stats);
    let mut downsampled_plan_cache = DownsampledVoxPlanCache::default();
    let downsampled_stats = RefCell::new(DownsampledVoxStats::default());
    let debug_placements = std::env::var_os("REVOLUMETRIC_TEARDOWN_DUMP_PLACEMENTS")
        .is_some()
        .then(|| RefCell::new(Vec::new()));
    // Downsampled plans only exist when the source is scaled down to fit the world
    // (scale < 1.0). At scale >= 1.0 `downsampled_vox_plan_spec` always returns None, so
    // the collect_plans XML pass and the parallel build both produce zero jobs and the
    // write pass never consults the plan cache. Skip both phases entirely in that case.
    let mut prebuilt_plan_count = 0u64;
    if target_mapping.scale < 1.0 {
        let mut downsampled_plan_jobs: HashMap<DownsampledVoxPlanKey, DownsampledVoxPlanBuildJob> =
            HashMap::new();

        let phase_start = Instant::now();
        walk_xml_bytes(
            &main_xml,
            identity,
            &mut source,
            &mut cache,
            &mut stats,
            &mut downsampled_plan_jobs,
            false,
            false,
            &mut |_pos, _material, _sink| {},
            &mut |_min, _max_exclusive, _material, _sink| true,
            &mut |path, object, scene, _tint, matrix, jobs| {
                collect_downsampled_vox_plan_job(
                    path,
                    object,
                    scene,
                    matrix,
                    density,
                    &target_mapping,
                    jobs,
                )
            },
        )?;
        profile_teardown_phase(profile, "collect_plans", phase_start.elapsed());
        prebuilt_plan_count = downsampled_plan_jobs.len() as u64;
        let phase_start = Instant::now();
        downsampled_plan_cache.plans =
            build_downsampled_vox_plans_parallel(downsampled_plan_jobs.into_values().collect());
        profile_teardown_phase(profile, "build_plans", phase_start.elapsed());
    }

    // Brick-slab parallel write (opt-in via REVOLUMETRIC_TEARDOWN_PARALLEL_WRITE):
    // N workers each walk the full document independently with their own
    // TeardownResourceSource + VoxCache, but only write bricks in their slab
    // (brick_z % N == worker_id). Bricks are disjoint across workers → merge is trivial
    // concatenation. No shared mutable state → no thread-safety refactor needed.
    // Transform work is replicated N× (runs in parallel → wall-clock 1×); write work
    // partitions 1/N. Only valid when scale >= 1.0 (downsampled-plan path inert there).
    let use_parallel_write = target_mapping.scale >= 1.0
        && std::env::var_os("REVOLUMETRIC_TEARDOWN_PARALLEL_WRITE").is_some();

    let phase_start = Instant::now();
    let (flush_bricks, write_stats): (Vec<(UVec3, StagedBrick)>, TargetWriteStats) =
        if use_parallel_write {
            let result = parallel_slab_write(
                zip_path.as_ref(),
                worker_teardown_dir,
                &main_xml,
                cache.clone(), // pre-filled from bounds pass — workers get cheap Arc-ptr clones
                ucvh,
                &target_mapping,
                density,
                profile,
            )?;
            profile_teardown_phase(profile, "write", phase_start.elapsed());
            result
        } else {
            walk_xml_bytes(
                &main_xml,
                identity,
                &mut source,
                &mut cache,
                &mut stats,
                &mut staged_bricks,
                false,
                true,
                &mut |pos, material, bricks| {
                    let Some(target) = target_mapping.map(pos) else {
                        write_stats.record_out_of_bounds(1);
                        return;
                    };
                    if target.x < ucvh.config.world_size.x
                        && target.y < ucvh.config.world_size.y
                        && target.z < ucvh.config.world_size.z
                    {
                        if write_target_voxel(ucvh, bricks, target, material) {
                            write_stats.record_written(target);
                        }
                    } else {
                        write_stats.record_out_of_bounds(1);
                    }
                },
                &mut |source_min, source_max_exclusive, material, bricks| {
                    let Some((source_min, source_max_exclusive)) =
                        target_mapping.clipped_source_box(source_min, source_max_exclusive)
                    else {
                        return false;
                    };
                    let Some(start) = target_mapping.map(source_min) else {
                        return false;
                    };
                    let Some(end) = target_mapping.map(source_max_exclusive - IVec3::ONE) else {
                        return false;
                    };
                    let min = start.min(end);
                    let max_exclusive = (start.max(end) + UVec3::ONE).min(ucvh.config.world_size);
                    if min.x >= max_exclusive.x
                        || min.y >= max_exclusive.y
                        || min.z >= max_exclusive.z
                    {
                        return true;
                    }
                    fill_target_box(ucvh, bricks, min, max_exclusive, material, &write_stats);
                    true
                },
                &mut |path, object, scene, tint, matrix, bricks| {
                    write_downsampled_vox_scene(
                        path,
                        object,
                        scene,
                        tint,
                        matrix,
                        density,
                        &target_mapping,
                        ucvh.config.world_size,
                        ucvh,
                        bricks,
                        &write_stats,
                        &mut downsampled_plan_cache,
                        &downsampled_stats,
                        debug_placements.as_ref(),
                    )
                },
            )?;
            profile_teardown_phase(profile, "write", phase_start.elapsed());
            (std::mem::take(&mut staged_bricks.bricks), write_stats.finish())
        };
    let downsampled_stats = downsampled_stats.into_inner();
    stats.written_voxels = write_stats.written_voxels;
    stats.out_of_bounds_voxels = write_stats.out_of_bounds_voxels;
    stats.downsampled_vox_nodes = downsampled_stats.nodes;
    stats.downsampled_vox_plan_hits = downsampled_stats.plan_hits;
    stats.downsampled_vox_plan_misses = downsampled_stats.plan_misses;
    stats.downsampled_vox_plans_prebuilt = prebuilt_plan_count;

    let phase_start = Instant::now();
    let failed_bricks = ucvh.write_static_bricks_bulk(
        flush_bricks
            .iter()
            .map(|(brick_pos, brick)| (*brick_pos, &brick.data)),
    );
    profile_teardown_phase(profile, "flush_bricks", phase_start.elapsed());
    if failed_bricks > 0 {
        return Err(TeardownZipLoadError::UcvhCapacityExceeded {
            failed_bricks,
            dropped_voxels: 0,
        });
    }
    stats.unique_written_voxels = flush_bricks
        .iter()
        .map(|(_, brick)| brick)
        .map(|brick| u64::from(brick.touched_count))
        .sum();
    if stats.unique_written_voxels > 0 {
        stats.target_bounds = Some(VoxTargetBounds {
            min: write_stats.target_min,
            max_exclusive: write_stats.target_max_exclusive,
        });
    }
    stats.missing_mod_refs = source.sorted_missing_mod();
    stats.missing_builtin_refs = source.sorted_missing_builtin();
    if let Some(debug_placements) = debug_placements {
        dump_debug_placements(&debug_placements.into_inner());
    }
    profile_teardown_phase(profile, "total", load_start.elapsed());
    Ok(stats)
}

fn profile_teardown_phase(enabled: bool, phase: &str, elapsed: Duration) {
    if enabled {
        eprintln!("teardown_zip_profile {phase}={:.3}s", elapsed.as_secs_f64());
    }
}

fn dump_debug_placements(placements: &[TeardownDebugVoxPlacement]) {
    let mut placements = placements.to_vec();
    placements.sort_by_key(|placement| {
        let center = (placement.target_min + placement.target_max_exclusive) / 2;
        let dx = center.x.abs_diff(512);
        let dy = center.y.abs_diff(256);
        let dz = center.z.abs_diff(512);
        std::cmp::Reverse(u64::from(dx) + u64::from(dy) + u64::from(dz))
    });
    for placement in placements.into_iter().take(40) {
        eprintln!(
            "teardown_zip_placement path={} object={} source={:?}..{:?} target={:?}..{:?}",
            placement.path,
            placement.object.as_deref().unwrap_or("<all>"),
            placement.source_min,
            placement.source_max_exclusive,
            placement.target_min,
            placement.target_max_exclusive,
        );
    }
}

#[derive(Default, Clone)]
struct VoxCache {
    scenes: HashMap<String, Option<Arc<CachedVoxScene>>>,
    brush_patterns: HashMap<(String, Option<String>), Option<Arc<BrushPattern>>>,
}

struct CachedVoxScene {
    scene: VoxScene,
    bounds: Option<VoxBounds>,
    visible_voxel_count: u64,
    materials: [u16; 256],
    objects: HashMap<String, CachedVoxObjectMetadata>,
}

#[derive(Clone, Copy)]
struct CachedVoxObjectMetadata {
    bounds: Option<VoxBounds>,
    origin: IVec3,
    visible_voxel_count: u64,
}

impl CachedVoxScene {
    fn new(scene: VoxScene) -> Self {
        let bounds = scene.bounds();
        let visible_voxel_count = scene.visible_voxel_count();
        let materials = scene.palette.map(material_for_color);
        let objects = cached_vox_object_metadata(&scene);
        Self {
            scene,
            bounds,
            visible_voxel_count,
            materials,
            objects,
        }
    }

    fn metadata(&self, object: Option<&str>) -> Option<CachedVoxObjectMetadata> {
        match object {
            Some(object) => self.objects.get(object).copied(),
            None => Some(CachedVoxObjectMetadata {
                bounds: self.bounds,
                origin: IVec3::ZERO,
                visible_voxel_count: self.visible_voxel_count,
            }),
        }
    }
}

fn selected_object_origin(scene: &CachedVoxScene, object: Option<&str>) -> IVec3 {
    scene
        .metadata(object)
        .map(|metadata| metadata.origin)
        .unwrap_or(IVec3::ZERO)
}

fn selected_object_bounds(scene: &CachedVoxScene, object: Option<&str>) -> Option<VoxBounds> {
    let bounds = scene.metadata(object)?.bounds?;
    let origin = selected_object_origin(scene, object);
    Some(VoxBounds {
        min: bounds.min - origin,
        max_exclusive: bounds.max_exclusive - origin,
    })
}

fn cached_vox_object_metadata(scene: &VoxScene) -> HashMap<String, CachedVoxObjectMetadata> {
    let mut objects: HashMap<String, CachedVoxObjectMetadata> = HashMap::new();
    for instance in &scene.instances {
        let Some(object_name) = &instance.object_name else {
            continue;
        };
        let Some(model) = scene.models.get(instance.model_index) else {
            continue;
        };
        let metadata = objects
            .entry(object_name.clone())
            .or_insert(CachedVoxObjectMetadata {
                bounds: None,
                origin: instance.object_origin.unwrap_or(IVec3::ZERO),
                visible_voxel_count: 0,
            });
        let mut bounds = metadata.bounds.unwrap_or(VoxBounds {
            min: IVec3::splat(i32::MAX),
            max_exclusive: IVec3::splat(i32::MIN),
        });
        for voxel in &model.voxels {
            let position = transform_vox_instance_point(instance, voxel.position.as_ivec3());
            bounds.min = bounds.min.min(position);
            bounds.max_exclusive = bounds.max_exclusive.max(position + IVec3::ONE);
            metadata.visible_voxel_count += 1;
        }
        metadata.bounds = (metadata.visible_voxel_count > 0).then_some(bounds);
    }
    objects
}

fn instance_matches_selected_object(
    instance: &crate::voxel::vox_loader::VoxInstance,
    object: Option<&str>,
) -> bool {
    match object {
        Some(object) => instance.object_name.as_deref() == Some(object),
        None => !instance.hidden,
    }
}

impl VoxCache {
    fn get(
        &mut self,
        source: &mut TeardownResourceSource,
        stats: &mut TeardownZipWriteStats,
        path: &str,
    ) -> Option<Arc<CachedVoxScene>> {
        if !self.scenes.contains_key(path) {
            stats.vox_cache_misses += 1;
            let scene = source
                .read_resource(path)
                .and_then(|bytes| match parse_vox(&bytes) {
                    Ok(scene) => Some(Arc::new(CachedVoxScene::new(scene))),
                    Err(_) => {
                        stats.malformed_vox_refs.push(path.to_owned());
                        None
                    }
                });
            self.scenes.insert(path.to_owned(), scene);
        } else {
            stats.vox_cache_hits += 1;
        }
        self.scenes
            .get(path)
            .and_then(|scene| scene.as_ref().cloned())
    }
}

struct TeardownResourceSource {
    zip: zip::ZipArchive<File>,
    root: String,
    lookup: HashMap<String, String>,
    teardown_dir: Option<PathBuf>,
    level_roots: Vec<LevelRoot>,
    missing_mod: HashSet<String>,
    missing_builtin: HashSet<String>,
}

#[derive(Clone)]
enum LevelRoot {
    Zip(String),
    TeardownData(String),
}

impl TeardownResourceSource {
    fn open(zip_path: &Path, teardown_dir: Option<PathBuf>) -> Result<Self, TeardownZipLoadError> {
        let file =
            File::open(zip_path).map_err(|error| TeardownZipLoadError::Io(error.to_string()))?;
        let mut zip = zip::ZipArchive::new(file)
            .map_err(|error| TeardownZipLoadError::Zip(error.to_string()))?;
        let mut names = Vec::new();
        for index in 0..zip.len() {
            let file = zip
                .by_index(index)
                .map_err(|error| TeardownZipLoadError::Zip(error.to_string()))?;
            let name = file.name().replace('\\', "/");
            if !name.ends_with('/') {
                names.push(name);
            }
        }
        let root = names
            .iter()
            .find_map(|name| {
                let lower = name.to_ascii_lowercase();
                (lower == "main.xml" || lower.ends_with("/main.xml")).then(|| {
                    name.rsplit_once('/')
                        .map(|(root, _)| root.to_owned())
                        .unwrap_or_default()
                })
            })
            .ok_or(TeardownZipLoadError::MainXmlMissing)?;
        let lookup = names
            .into_iter()
            .map(|name| (name.to_ascii_lowercase(), name))
            .collect();
        Ok(Self {
            zip,
            level_roots: vec![LevelRoot::Zip(root.clone())],
            root,
            lookup,
            teardown_dir: find_teardown_dir(teardown_dir),
            missing_mod: HashSet::new(),
            missing_builtin: HashSet::new(),
        })
    }

    fn read_main_xml(&mut self) -> Result<Vec<u8>, TeardownZipLoadError> {
        let name = zip_resource_name(&self.root, "main.xml");
        self.read_zip_entry(&name)
            .ok_or(TeardownZipLoadError::MainXmlMissing)
    }

    fn read_resource(&mut self, path: &str) -> Option<Vec<u8>> {
        let normalized = normalize_resource_path(path);
        if let Some(rel) = normalized.strip_prefix("MOD/") {
            let name = zip_resource_name(&self.root, rel);
            let Some(bytes) = self.read_zip_entry(&name) else {
                self.missing_mod.insert(normalized);
                return None;
            };
            return Some(bytes);
        }
        if let Some(rel) = normalized.strip_prefix("LEVEL/") {
            let Some(bytes) = self.read_level_resource(rel) else {
                self.missing_mod.insert(normalized);
                return None;
            };
            return Some(bytes);
        }
        if let Some(rel) = normalized.strip_prefix("BUILT-IN/") {
            let Some(teardown_dir) = &self.teardown_dir else {
                self.missing_builtin.insert(normalized);
                return None;
            };
            let Some(candidate) = safe_join(&teardown_dir.join("data").join("built-in"), rel)
            else {
                self.missing_builtin.insert(normalized);
                return None;
            };
            return match std::fs::read(candidate) {
                Ok(bytes) => Some(bytes),
                Err(_) => {
                    self.missing_builtin.insert(normalized);
                    None
                }
            };
        }
        if let Some(bytes) = self.read_teardown_data_relative(&normalized) {
            return Some(bytes);
        }
        self.missing_mod.insert(normalized);
        None
    }

    fn read_level_resource(&mut self, rel: &str) -> Option<Vec<u8>> {
        match self.current_level_root() {
            LevelRoot::Zip(root) => {
                let name = zip_resource_name(&root, rel);
                self.read_zip_entry(&name)
            }
            LevelRoot::TeardownData(level_root) => {
                let teardown_dir = self.teardown_dir.clone()?;
                let rel = format!("{level_root}/{rel}");
                let candidate = safe_join(&teardown_dir.join("data"), &rel)?;
                std::fs::read(candidate).ok()
            }
        }
    }

    fn current_level_root(&self) -> LevelRoot {
        self.level_roots
            .last()
            .cloned()
            .unwrap_or_else(|| LevelRoot::Zip(self.root.clone()))
    }

    fn level_root_for_resource_path(&self, path: &str) -> LevelRoot {
        let normalized = path.replace('\\', "/");
        if normalized.starts_with("MOD/") {
            return LevelRoot::Zip(self.root.clone());
        }
        if normalized.starts_with("LEVEL/") {
            return self.current_level_root();
        }
        if let Some(root) = teardown_data_level_root(&normalized) {
            return LevelRoot::TeardownData(root.to_owned());
        }
        self.current_level_root()
    }

    fn push_level_root(&mut self, root: LevelRoot) {
        self.level_roots.push(root);
    }

    fn pop_level_root(&mut self) {
        self.level_roots.pop();
    }

    fn read_teardown_data_relative(&mut self, normalized: &str) -> Option<Vec<u8>> {
        let teardown_dir = self.teardown_dir.clone()?;
        let data_root = teardown_dir.join("data");
        let data_relative = teardown_data_relative_path(normalized);
        let mut candidates = Vec::new();
        if data_relative.to_ascii_lowercase().ends_with(".vox")
            && !data_relative.starts_with("vox/")
        {
            candidates.push(data_root.join("vox"));
        }
        candidates.push(data_root);
        for root in candidates {
            let Some(candidate) = safe_join(&root, data_relative) else {
                continue;
            };
            if let Ok(bytes) = std::fs::read(candidate) {
                return Some(bytes);
            }
        }
        None
    }

    fn read_zip_entry(&mut self, name: &str) -> Option<Vec<u8>> {
        let zip_name = self.lookup.get(&name.to_ascii_lowercase())?.clone();
        let mut file = self.zip.by_name(&zip_name).ok()?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).ok()?;
        Some(bytes)
    }

    fn sorted_missing_mod(&self) -> Vec<String> {
        let mut refs: Vec<_> = self.missing_mod.iter().cloned().collect();
        refs.sort();
        refs
    }

    fn sorted_missing_builtin(&self) -> Vec<String> {
        let mut refs: Vec<_> = self.missing_builtin.iter().cloned().collect();
        refs.sort();
        refs
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_xml_bytes<S>(
    xml: &[u8],
    parent_matrix: Mat4,
    source: &mut TeardownResourceSource,
    cache: &mut VoxCache,
    stats: &mut TeardownZipWriteStats,
    sink: &mut S,
    record_stats: bool,
    emit_content: bool,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    fill_axis_aligned_box: &mut impl FnMut(IVec3, IVec3, u16, &mut S) -> bool,
    write_axis_aligned_vox_scene: &mut impl FnMut(
        &str,
        Option<&str>,
        &Arc<CachedVoxScene>,
        Option<ColorTint>,
        Mat4,
        &mut S,
    ) -> bool,
) -> Result<(), TeardownZipLoadError> {
    let xml =
        std::str::from_utf8(xml).map_err(|error| TeardownZipLoadError::Xml(error.to_string()))?;
    let document = roxmltree::Document::parse(xml)
        .map_err(|error| TeardownZipLoadError::Xml(error.to_string()))?;
    let root = document.root_element();
    let mut stack = Vec::new();
    walk_xml_node(
        root,
        parent_matrix,
        source,
        cache,
        stats,
        sink,
        record_stats,
        emit_content,
        emit,
        fill_axis_aligned_box,
        write_axis_aligned_vox_scene,
        &mut stack,
    )
}

#[allow(clippy::too_many_arguments)]
fn walk_xml_node<S>(
    node: roxmltree::Node,
    parent_matrix: Mat4,
    source: &mut TeardownResourceSource,
    cache: &mut VoxCache,
    stats: &mut TeardownZipWriteStats,
    sink: &mut S,
    record_stats: bool,
    emit_content: bool,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    fill_axis_aligned_box: &mut impl FnMut(IVec3, IVec3, u16, &mut S) -> bool,
    write_axis_aligned_vox_scene: &mut impl FnMut(
        &str,
        Option<&str>,
        &Arc<CachedVoxScene>,
        Option<ColorTint>,
        Mat4,
        &mut S,
    ) -> bool,
    stack: &mut Vec<String>,
) -> Result<(), TeardownZipLoadError> {
    if !node.is_element() {
        return Ok(());
    }
    let tag = node.tag_name().name();
    let matrix = if matches!(
        tag,
        "scene"
            | "prefab"
            | "group"
            | "body"
            | "compound"
            | "vehicle"
            | "wheel"
            | "shape"
            | "joint"
            | "rope"
            | "script"
            | "screen"
            | "trigger"
            | "boundary"
            | "location"
            | "light"
            | "spawnpoint"
            | "water"
            | "vox"
            | "voxbox"
            | "voxagon"
            | "voxscript"
            | "instance"
    ) {
        parent_matrix * node_transform(node)
    } else {
        parent_matrix
    };

    match tag {
        "vox" => {
            if record_stats {
                stats.vox_nodes += 1;
            }
            if let Some(path) = node.attribute("file") {
                if let Some(scene) = cache.get(source, stats, path) {
                    let object = node.attribute("object");
                    let tint = parse_color_tint(node.attribute("color"));
                    let mirror = parse_mirror_axes(node);
                    if record_stats {
                        let Some(metadata) = scene.metadata(object) else {
                            return Ok(());
                        };
                        let count = metadata.visible_voxel_count
                            * native_voxel_expansion_count(source_density(stats));
                        if count > 0 {
                            if let Some(bounds) = selected_object_bounds(scene.as_ref(), object) {
                                emit_transformed_native_volume_bounds(
                                    bounds.min,
                                    bounds.max_exclusive,
                                    matrix,
                                    source_density(stats),
                                    emit,
                                    sink,
                                );
                            }
                            stats.input_voxels += count;
                            stats.vox_nodes_exported += 1;
                        }
                    } else if (mirror.any()
                        || !write_axis_aligned_vox_scene(path, object, &scene, tint, matrix, sink))
                        && emit_content
                    {
                        let Some(bounds) = selected_object_bounds(scene.as_ref(), object) else {
                            return Ok(());
                        };
                        let tinted_materials = tinted_materials(scene.as_ref(), tint);
                        let materials = tinted_materials.as_ref().unwrap_or(&scene.materials);
                        let density_cell_emission =
                            density_cell_emission(matrix, source_density(stats));
                        visit_cached_visible_voxels(
                            scene.as_ref(),
                            object,
                            materials,
                            |native_pos, material| {
                                let native_pos = mirror_native_coord(native_pos, bounds, mirror);
                                for local in
                                    expanded_native_coords(native_pos, source_density(stats))
                                {
                                    emit_transformed_density_cell(
                                        local,
                                        matrix,
                                        source_density(stats),
                                        material,
                                        density_cell_emission,
                                        emit,
                                        sink,
                                    );
                                }
                            },
                        );
                    }
                }
            }
        }
        "voxbox" => {
            if record_stats {
                stats.voxbox_nodes += 1;
            }
            let is_hole = is_hole_brush(node.attribute("brush"));
            let color = parse_color(node.attribute("color"), [155, 155, 150, 255]);
            let material = if is_hole {
                0
            } else {
                material_for_color(color)
            };
            let tint = (!is_hole)
                .then(|| parse_color_tint(node.attribute("color")))
                .flatten();
            let size = parse_vec3(node.attribute("size"), Vec3::ONE);
            let counts = voxbox_axis_counts(size, source_density(stats));
            let (local_min, local_max_exclusive) = voxbox_local_bounds(counts);
            let brush = (!is_hole)
                .then(|| load_voxbox_brush_pattern(node, source, cache, stats))
                .flatten();
            let offset = parse_native_offset(node.attribute("offset"));
            let mut emitted = false;
            if let Some(brush) = brush.as_ref() {
                let count = emit_brushed_voxbox_points(
                    brush,
                    counts,
                    offset,
                    parse_mirror_axes(node),
                    matrix,
                    source_density(stats),
                    local_min,
                    tint,
                    emit,
                    sink,
                );
                emitted = count > 0;
                if record_stats {
                    stats.input_voxels += count;
                }
            } else if record_stats {
                if !is_hole && counts.x > 0 && counts.y > 0 && counts.z > 0 {
                    emit_transformed_native_volume_bounds(
                        local_min,
                        local_max_exclusive,
                        matrix,
                        source_density(stats),
                        emit,
                        sink,
                    );
                    stats.input_voxels += counts.x as u64 * counts.y as u64 * counts.z as u64;
                    emitted = true;
                }
            } else if emit_content
                && let Some((source_min, source_max_exclusive)) = axis_aligned_native_box(
                    matrix,
                    local_min,
                    local_max_exclusive,
                    source_density(stats),
                )
            {
                emitted = fill_axis_aligned_box(source_min, source_max_exclusive, material, sink);
            } else if emit_content {
                let count = emit_solid_voxbox_points(
                    local_min,
                    local_max_exclusive,
                    matrix,
                    source_density(stats),
                    material,
                    emit,
                    sink,
                );
                emitted = count > 0;
            }
            if emit_content && !record_stats && !emitted {
                let density_cell_emission = density_cell_emission(matrix, source_density(stats));
                for x in local_min.x..local_max_exclusive.x {
                    for y in local_min.y..local_max_exclusive.y {
                        for z in local_min.z..local_max_exclusive.z {
                            emit_transformed_density_cell(
                                IVec3::new(x, y, z),
                                matrix,
                                source_density(stats),
                                material,
                                density_cell_emission,
                                emit,
                                sink,
                            );
                            emitted = true;
                        }
                    }
                }
            }
            if emitted && record_stats && !is_hole {
                stats.voxbox_nodes_exported += 1;
            }
        }
        "voxagon" => {
            let is_hole = is_hole_brush(node.attribute("brush"));
            let color = parse_color(node.attribute("color"), [155, 155, 150, 255]);
            let material = if is_hole {
                0
            } else {
                material_for_color(color)
            };
            let tint = (!is_hole)
                .then(|| parse_color_tint(node.attribute("color")))
                .flatten();
            if record_stats {
                if !is_hole
                    && let Some((min, max_exclusive)) =
                        voxagon_native_bounds(node, matrix, source_density(stats))
                {
                    emit_transformed_native_bounds(
                        min,
                        max_exclusive,
                        Mat4::IDENTITY,
                        source_density(stats),
                        emit,
                        sink,
                    );
                    let extent = max_exclusive - min;
                    stats.input_voxels +=
                        extent.x.max(0) as u64 * extent.y.max(0) as u64 * extent.z.max(0) as u64;
                }
            } else if emit_content {
                let brush = (!is_hole)
                    .then(|| load_voxbox_brush_pattern(node, source, cache, stats))
                    .flatten();
                let offset = parse_native_offset(node.attribute("offset"));
                if let Some(brush) = brush.as_ref() {
                    emit_brushed_voxagon_points(
                        node,
                        brush,
                        offset,
                        parse_mirror_axes(node),
                        matrix,
                        source_density(stats),
                        tint,
                        emit,
                        sink,
                    );
                } else {
                    emit_voxagon_points(node, matrix, source_density(stats), material, emit, sink);
                }
            }
        }
        "voxscript" => {
            if (record_stats || emit_content)
                && let Some(file) = node.attribute("file")
            {
                if let Some(script) = source.read_resource(file) {
                    if let Some(heightmap) = load_voxscript_heightmap(node, &script, source) {
                        if record_stats {
                            if let Some((min, max_exclusive, count)) =
                                heightmap_native_bounds(&heightmap, matrix, source_density(stats))
                            {
                                emit_transformed_native_bounds(
                                    min,
                                    max_exclusive,
                                    Mat4::IDENTITY,
                                    source_density(stats),
                                    emit,
                                    sink,
                                );
                                stats.input_voxels += count;
                            }
                        } else if emit_content {
                            emit_heightmap_columns(
                                &heightmap,
                                matrix,
                                source_density(stats),
                                emit,
                                sink,
                            );
                        }
                    }
                }
            }
        }
        "water" => {}
        "instance" => {
            if record_stats {
                stats.instance_nodes += 1;
            }
            if let Some(path) = node.attribute("file") {
                if stack.iter().any(|item| item == path) {
                    if record_stats {
                        stats.recursive_instance_refs.push(path.to_owned());
                    }
                } else if let Some(xml) = source.read_resource(path) {
                    stack.push(path.to_owned());
                    let level_root = source.level_root_for_resource_path(path);
                    source.push_level_root(level_root);
                    let result = walk_xml_bytes(
                        &xml,
                        matrix,
                        source,
                        cache,
                        stats,
                        sink,
                        record_stats,
                        emit_content,
                        emit,
                        fill_axis_aligned_box,
                        write_axis_aligned_vox_scene,
                    );
                    source.pop_level_root();
                    stack.pop();
                    if result.is_ok() && record_stats {
                        stats.instance_nodes_exported += 1;
                    }
                    result?;
                }
            }
        }
        _ => {}
    }

    for child in node.children() {
        walk_xml_node(
            child,
            matrix,
            source,
            cache,
            stats,
            sink,
            record_stats,
            emit_content,
            emit,
            fill_axis_aligned_box,
            write_axis_aligned_vox_scene,
            stack,
        )?;
    }
    Ok(())
}

fn source_density(stats: &TeardownZipWriteStats) -> u32 {
    stats.voxels_per_unit.max(1)
}

fn node_transform(node: roxmltree::Node) -> Mat4 {
    let pos = parse_vec3(node.attribute("pos"), Vec3::ZERO);
    Mat4::from_translation(pos)
        * teardown_euler_rotation(parse_vec3(node.attribute("rot"), Vec3::ZERO))
        * Mat4::from_scale(parse_scale(node.attribute("scale")))
}

fn teardown_euler_rotation(rot: Vec3) -> Mat4 {
    let rot = Vec3::new(rot.x.to_radians(), rot.y.to_radians(), rot.z.to_radians());
    Mat4::from_rotation_y(rot.y) * Mat4::from_rotation_z(rot.z) * Mat4::from_rotation_x(rot.x)
}

fn native_voxel_expansion_count(voxels_per_unit: u32) -> u64 {
    let len = native_axis_range(0, voxels_per_unit).len() as u64;
    len * len * len
}

fn emit_transformed_native_bounds<S>(
    min: IVec3,
    max_exclusive: IVec3,
    matrix: Mat4,
    voxels_per_unit: u32,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) {
    if min.x >= max_exclusive.x || min.y >= max_exclusive.y || min.z >= max_exclusive.z {
        return;
    }
    let last = max_exclusive - IVec3::ONE;
    for x in [min.x, last.x] {
        for y in [min.y, last.y] {
            for z in [min.z, last.z] {
                let world = matrix.transform_point3(native_point(x, y, z, voxels_per_unit));
                emit(round_world_point(world, voxels_per_unit), 0, sink);
            }
        }
    }
}

fn emit_transformed_native_volume_bounds<S>(
    min: IVec3,
    max_exclusive: IVec3,
    matrix: Mat4,
    voxels_per_unit: u32,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) {
    let Some((start, end)) =
        transformed_native_volume_bounds(min, max_exclusive, matrix, voxels_per_unit)
    else {
        return;
    };
    for x in [start.x, end.x - 1] {
        for y in [start.y, end.y - 1] {
            for z in [start.z, end.z - 1] {
                emit(IVec3::new(x, y, z), 0, sink);
            }
        }
    }
}

fn transformed_native_volume_bounds(
    min: IVec3,
    max_exclusive: IVec3,
    matrix: Mat4,
    voxels_per_unit: u32,
) -> Option<(IVec3, IVec3)> {
    if min.x >= max_exclusive.x || min.y >= max_exclusive.y || min.z >= max_exclusive.z {
        return None;
    }
    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);
    for x in [min.x, max_exclusive.x] {
        for y in [min.y, max_exclusive.y] {
            for z in [min.z, max_exclusive.z] {
                let world = matrix.transform_point3(native_point(x, y, z, voxels_per_unit));
                world_min = world_min.min(world);
                world_max = world_max.max(world);
            }
        }
    }
    let start = floor_ivec3(world_min * voxels_per_unit as f32);
    let end = ceil_ivec3(world_max * voxels_per_unit as f32).max(start + IVec3::ONE);
    Some((start, end))
}

fn axis_aligned_native_box(
    matrix: Mat4,
    min: IVec3,
    max_exclusive: IVec3,
    voxels_per_unit: u32,
) -> Option<(IVec3, IVec3)> {
    if min.x >= max_exclusive.x
        || min.y >= max_exclusive.y
        || min.z >= max_exclusive.z
        || !is_axis_aligned_linear_transform(matrix)
    {
        return None;
    }
    axis_aligned_transformed_native_volume_bounds(min, max_exclusive, matrix, voxels_per_unit)
}

fn is_axis_aligned_linear_transform(matrix: Mat4) -> bool {
    let cols = matrix.to_cols_array_2d();
    let epsilon = 1.0e-5;
    let mut row_hits = [0u8; 3];
    for values in cols.iter().take(3) {
        let mut column_hits = 0u8;
        for row in 0..3 {
            if values[row].abs() > epsilon {
                column_hits += 1;
                row_hits[row] += 1;
            }
        }
        if column_hits != 1 {
            return false;
        }
    }
    row_hits.iter().all(|hits| *hits == 1)
}

fn axis_aligned_transformed_native_volume_bounds(
    min: IVec3,
    max_exclusive: IVec3,
    matrix: Mat4,
    voxels_per_unit: u32,
) -> Option<(IVec3, IVec3)> {
    if min.x >= max_exclusive.x || min.y >= max_exclusive.y || min.z >= max_exclusive.z {
        return None;
    }
    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);
    for x in [min.x, max_exclusive.x] {
        for y in [min.y, max_exclusive.y] {
            for z in [min.z, max_exclusive.z] {
                let world = matrix.transform_point3(native_point(x, y, z, voxels_per_unit));
                world_min = world_min.min(world);
                world_max = world_max.max(world);
            }
        }
    }
    let scale = voxels_per_unit as f32;
    let start = floor_ivec3_with_epsilon(world_min * scale);
    let end = ceil_ivec3_with_epsilon(world_max * scale).max(start + IVec3::ONE);
    Some((start, end))
}

fn floor_ivec3_with_epsilon(value: Vec3) -> IVec3 {
    let epsilon = 1.0e-4;
    IVec3::new(
        (value.x + epsilon).floor() as i32,
        (value.y + epsilon).floor() as i32,
        (value.z + epsilon).floor() as i32,
    )
}

fn ceil_ivec3_with_epsilon(value: Vec3) -> IVec3 {
    let epsilon = 1.0e-4;
    IVec3::new(
        (value.x - epsilon).ceil() as i32,
        (value.y - epsilon).ceil() as i32,
        (value.z - epsilon).ceil() as i32,
    )
}

fn transform_expands_density_cells(matrix: Mat4) -> bool {
    let cols = matrix.to_cols_array_2d();
    let epsilon = 1.0e-4;
    let mut row_occupancy = [0u8; 3];
    (0..3).any(|col| {
        let len_sq =
            cols[col][0] * cols[col][0] + cols[col][1] * cols[col][1] + cols[col][2] * cols[col][2];
        if len_sq > 1.0001 {
            return true;
        }
        let mut significant_rows = 0;
        for row in 0..3 {
            if cols[col][row].abs() > epsilon {
                significant_rows += 1;
                row_occupancy[row] += 1;
            }
        }
        significant_rows > 1
    }) || row_occupancy.iter().any(|count| *count > 1)
}

/// Integer affine map from a native source coord `c` to a target source coord:
/// `target = L·c + t`, where `L·(c/vpu) + translation` composes to integers.
///
/// Derivation: the float path computes `round((L·(c/vpu) + translation)·vpu)`
/// componentwise (round-half-away-from-zero). That equals `L·c + translation·vpu`.
/// This struct is only constructed when every `L` entry rounds to an integer and
/// every `translation·vpu` component rounds to an integer within a tight tolerance,
/// so the true (real-number) result is an exact integer for all `c`. The residual
/// float error is `|δ·c| + |δt|` with `|δ| ~ 1e-7`; for Teardown coords (`|c| < ~1e6`)
/// this stays far below 0.5, so the float path recovers the same integer — making the
/// integer path bit-identical. Any node that does not satisfy this (fractional scale,
/// non-axis-aligned rotation, half-integer translation) is rejected and falls back to
/// the float `Point` path unchanged.
#[derive(Clone, Copy)]
struct IntegerAffine {
    l: [[i32; 3]; 3],
    t: [i32; 3],
}

impl IntegerAffine {
    #[inline]
    fn apply(&self, coord: IVec3) -> IVec3 {
        let c = [coord.x, coord.y, coord.z];
        IVec3::new(
            self.l[0][0] * c[0] + self.l[0][1] * c[1] + self.l[0][2] * c[2] + self.t[0],
            self.l[1][0] * c[0] + self.l[1][1] * c[1] + self.l[1][2] * c[2] + self.t[1],
            self.l[2][0] * c[0] + self.l[2][1] * c[1] + self.l[2][2] * c[2] + self.t[2],
        )
    }
}

fn integer_affine(matrix: Mat4, voxels_per_unit: u32) -> Option<IntegerAffine> {
    const L_TOL: f32 = 1.0e-4;
    const T_TOL: f32 = 1.0e-3;
    let linear = linear_transform(matrix);
    let mut l = [[0i32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            let value = linear[row][col];
            let rounded = value.round();
            if (value - rounded).abs() > L_TOL {
                return None;
            }
            l[row][col] = rounded as i32;
        }
    }
    let translation = matrix.transform_point3(Vec3::ZERO);
    let vpu = voxels_per_unit as f32;
    let translation = [translation.x, translation.y, translation.z];
    let mut t = [0i32; 3];
    for axis in 0..3 {
        let scaled = translation[axis] * vpu;
        let rounded = round_voxel_coord(scaled);
        if (scaled - rounded as f32).abs() > T_TOL {
            return None;
        }
        t[axis] = rounded;
    }
    Some(IntegerAffine { l, t })
}

#[derive(Clone, Copy)]
enum DensityCellEmission {
    Point,
    IntegerPoint(IntegerAffine),
    OrientedVolume { inverse: Mat4 },
    AabbVolume,
}

fn density_cell_emission(matrix: Mat4, voxels_per_unit: u32) -> DensityCellEmission {
    if !transform_expands_density_cells(matrix) {
        if let Some(affine) = integer_affine(matrix, voxels_per_unit) {
            return DensityCellEmission::IntegerPoint(affine);
        }
        return DensityCellEmission::Point;
    }
    let inverse = matrix.inverse();
    if inverse.is_finite() {
        DensityCellEmission::OrientedVolume { inverse }
    } else {
        DensityCellEmission::AabbVolume
    }
}

fn emit_transformed_density_cell<S>(
    coord: IVec3,
    matrix: Mat4,
    voxels_per_unit: u32,
    material: u16,
    emission: DensityCellEmission,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) {
    match emission {
        DensityCellEmission::Point => {
            let world =
                matrix.transform_point3(native_point(coord.x, coord.y, coord.z, voxels_per_unit));
            emit(round_world_point(world, voxels_per_unit), material, sink);
        }
        DensityCellEmission::IntegerPoint(affine) => {
            emit(affine.apply(coord), material, sink);
        }
        DensityCellEmission::OrientedVolume { inverse } => emit_oriented_density_cell(
            coord,
            matrix,
            inverse,
            voxels_per_unit,
            material,
            emit,
            sink,
        ),
        DensityCellEmission::AabbVolume => {
            emit_aabb_density_cell(coord, matrix, voxels_per_unit, material, emit, sink)
        }
    }
}

fn emit_aabb_density_cell<S>(
    coord: IVec3,
    matrix: Mat4,
    voxels_per_unit: u32,
    material: u16,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for dx in [0, 1] {
        for dy in [0, 1] {
            for dz in [0, 1] {
                let world = matrix.transform_point3(native_point(
                    coord.x + dx,
                    coord.y + dy,
                    coord.z + dz,
                    voxels_per_unit,
                ));
                min = min.min(world);
                max = max.max(world);
            }
        }
    }

    emit_cells_in_bounds(min, max, voxels_per_unit, emit, sink, |_, _| Some(material));
}

fn emit_oriented_density_cell<S>(
    coord: IVec3,
    matrix: Mat4,
    inverse: Mat4,
    voxels_per_unit: u32,
    material: u16,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for dx in [0, 1] {
        for dy in [0, 1] {
            for dz in [0, 1] {
                let world = matrix.transform_point3(native_point(
                    coord.x + dx,
                    coord.y + dy,
                    coord.z + dz,
                    voxels_per_unit,
                ));
                min = min.min(world);
                max = max.max(world);
            }
        }
    }

    let min_local = coord.as_vec3();
    let max_local = min_local + Vec3::ONE;
    let epsilon = 1.0e-4;
    let mut emitted = false;
    emit_cells_in_bounds(min, max, voxels_per_unit, emit, sink, |source, _sink| {
        let world = (source.as_vec3() + Vec3::splat(0.5)) / voxels_per_unit as f32;
        let local = inverse.transform_point3(world) * voxels_per_unit as f32;
        if local.x >= min_local.x - epsilon
            && local.y >= min_local.y - epsilon
            && local.z >= min_local.z - epsilon
            && local.x < max_local.x + epsilon
            && local.y < max_local.y + epsilon
            && local.z < max_local.z + epsilon
        {
            emitted = true;
            Some(material)
        } else {
            None
        }
    });

    if !emitted {
        let world =
            matrix.transform_point3(native_point(coord.x, coord.y, coord.z, voxels_per_unit));
        emit(round_world_point(world, voxels_per_unit), material, sink);
    }
}

fn emit_cells_in_bounds<S>(
    min: Vec3,
    max: Vec3,
    voxels_per_unit: u32,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
    mut material_at: impl FnMut(IVec3, &mut S) -> Option<u16>,
) {
    let start = floor_ivec3(min * voxels_per_unit as f32);
    let end = ceil_ivec3(max * voxels_per_unit as f32);
    for x in start.x..end.x.max(start.x + 1) {
        for y in start.y..end.y.max(start.y + 1) {
            for z in start.z..end.z.max(start.z + 1) {
                let source = IVec3::new(x, y, z);
                if let Some(material) = material_at(source, sink) {
                    emit(source, material, sink);
                }
            }
        }
    }
}

#[derive(Default)]
struct DownsampledVoxPlanCache {
    plans: HashMap<DownsampledVoxPlanKey, DownsampledVoxPlan>,
    used: HashSet<DownsampledVoxPlanKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DownsampledVoxPlanKey {
    path: String,
    object: Option<String>,
    linear: [[i16; 3]; 3],
    phase: [i32; 3],
}

struct DownsampledVoxPlanBuildJob {
    key: DownsampledVoxPlanKey,
    scene: Arc<CachedVoxScene>,
    object: Option<String>,
    voxels_per_unit: u32,
    scale: f32,
    phase: Vec3,
    linear: [[f32; 3]; 3],
}

struct DownsampledVoxPlan {
    cells: Vec<(IVec3, u16)>,
    bounds: Option<DownsampledVoxPlanBounds>,
}

#[derive(Clone, Copy)]
struct DownsampledVoxPlanBounds {
    min: IVec3,
    max_exclusive: IVec3,
}

#[derive(Default)]
struct DownsampledVoxStats {
    nodes: u64,
    plan_hits: u64,
    plan_misses: u64,
}

fn collect_downsampled_vox_plan_job(
    path: &str,
    object: Option<&str>,
    scene: &Arc<CachedVoxScene>,
    matrix: Mat4,
    voxels_per_unit: u32,
    target_mapping: &TargetMapping,
    jobs: &mut HashMap<DownsampledVoxPlanKey, DownsampledVoxPlanBuildJob>,
) -> bool {
    if scene.metadata(object).is_none() {
        return true;
    }
    let Some((key, _base_target, phase, linear)) =
        downsampled_vox_plan_spec(path, object, matrix, voxels_per_unit, target_mapping)
    else {
        return false;
    };
    jobs.entry(key.clone())
        .or_insert_with(|| DownsampledVoxPlanBuildJob {
            key,
            scene: Arc::clone(scene),
            object: object.map(str::to_owned),
            voxels_per_unit,
            scale: target_mapping.scale,
            phase,
            linear,
        });
    true
}

#[allow(clippy::too_many_arguments)]
fn write_downsampled_vox_scene(
    path: &str,
    object: Option<&str>,
    scene: &Arc<CachedVoxScene>,
    tint: Option<ColorTint>,
    matrix: Mat4,
    voxels_per_unit: u32,
    target_mapping: &TargetMapping,
    world_size: UVec3,
    ucvh: &Ucvh,
    bricks: &mut StagedBricks,
    write_stats: &TargetWriteStatsTracker,
    plan_cache: &mut DownsampledVoxPlanCache,
    downsampled_stats: &RefCell<DownsampledVoxStats>,
    debug_placements: Option<&RefCell<Vec<TeardownDebugVoxPlacement>>>,
) -> bool {
    if scene.metadata(object).is_none() {
        return true;
    }
    let Some((key, base_target, phase, linear)) =
        downsampled_vox_plan_spec(path, object, matrix, voxels_per_unit, target_mapping)
    else {
        return false;
    };
    {
        let mut stats = downsampled_stats.borrow_mut();
        stats.nodes += 1;
        if plan_cache.used.insert(key.clone()) {
            stats.plan_misses += 1;
        } else {
            stats.plan_hits += 1;
        }
    }
    let plan = match plan_cache.plans.entry(key) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            let plan = build_downsampled_vox_plan(
                scene,
                object,
                voxels_per_unit,
                target_mapping.scale,
                phase,
                linear,
            );
            entry.insert(plan)
        }
    };

    if let (Some(debug_placements), Some(bounds)) = (debug_placements, plan.bounds) {
        let min = base_target + bounds.min;
        let max_exclusive = base_target + bounds.max_exclusive;
        if min.x >= 0 && min.y >= 0 && min.z >= 0 {
            debug_placements
                .borrow_mut()
                .push(TeardownDebugVoxPlacement {
                    path: path.to_owned(),
                    object: object.map(str::to_owned),
                    source_min: bounds.min,
                    source_max_exclusive: bounds.max_exclusive,
                    target_min: UVec3::new(min.x as u32, min.y as u32, min.z as u32),
                    target_max_exclusive: UVec3::new(
                        max_exclusive.x.max(0) as u32,
                        max_exclusive.y.max(0) as u32,
                        max_exclusive.z.max(0) as u32,
                    ),
                });
        }
    }

    if let Some(bounds) = plan.bounds {
        let min = base_target + bounds.min;
        let max_exclusive = base_target + bounds.max_exclusive;
        if min.x >= 0
            && min.y >= 0
            && min.z >= 0
            && max_exclusive.x <= world_size.x as i32
            && max_exclusive.y <= world_size.y as i32
            && max_exclusive.z <= world_size.z as i32
        {
            let min = UVec3::new(min.x as u32, min.y as u32, min.z as u32);
            let max_exclusive = UVec3::new(
                max_exclusive.x as u32,
                max_exclusive.y as u32,
                max_exclusive.z as u32,
            );
            write_stats.record_count_bounds(plan.cells.len() as u64, min, max_exclusive);
            for (local_target, material) in &plan.cells {
                let target = base_target + *local_target;
                write_target_voxel(
                    ucvh,
                    bricks,
                    UVec3::new(target.x as u32, target.y as u32, target.z as u32),
                    apply_material_tint(*material, tint),
                );
            }
            return true;
        }
    }

    let mut out_of_bounds = 0;
    for (local_target, material) in &plan.cells {
        let target = base_target + *local_target;
        if target.x < 0 || target.y < 0 || target.z < 0 {
            out_of_bounds += 1;
            continue;
        }
        let target = UVec3::new(target.x as u32, target.y as u32, target.z as u32);
        if target.x < world_size.x && target.y < world_size.y && target.z < world_size.z {
            write_stats.record_written(target);
            write_target_voxel(ucvh, bricks, target, apply_material_tint(*material, tint));
        } else {
            out_of_bounds += 1;
        }
    }
    write_stats.record_out_of_bounds(out_of_bounds);
    true
}

fn downsampled_vox_plan_spec(
    path: &str,
    object: Option<&str>,
    matrix: Mat4,
    voxels_per_unit: u32,
    target_mapping: &TargetMapping,
) -> Option<(DownsampledVoxPlanKey, IVec3, Vec3, [[f32; 3]; 3])> {
    if target_mapping.scale >= 1.0 {
        return None;
    }
    let linear = linear_transform(matrix);
    let translation = round_world_point(matrix.transform_point3(Vec3::ZERO), voxels_per_unit);
    let (base_target, phase, quantized_phase) = target_mapping.base_target_and_phase(translation);
    Some((
        DownsampledVoxPlanKey {
            path: path.to_owned(),
            object: object.map(str::to_owned),
            linear: quantized_linear_key(linear),
            phase: quantized_phase,
        },
        base_target,
        phase,
        linear,
    ))
}

fn build_downsampled_vox_plans_parallel(
    jobs: Vec<DownsampledVoxPlanBuildJob>,
) -> HashMap<DownsampledVoxPlanKey, DownsampledVoxPlan> {
    if jobs.is_empty() {
        return HashMap::new();
    }
    let worker_count = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(jobs.len());
    if worker_count <= 1 || jobs.len() <= 1 {
        return jobs
            .into_iter()
            .map(|job| {
                let plan = build_downsampled_vox_plan(
                    &job.scene,
                    job.object.as_deref(),
                    job.voxels_per_unit,
                    job.scale,
                    job.phase,
                    job.linear,
                );
                (job.key, plan)
            })
            .collect();
    }

    let mut output = HashMap::with_capacity(jobs.len());
    let next_job = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..worker_count)
            .map(|_| {
                let jobs = &jobs;
                let next_job = &next_job;
                scope.spawn(move || {
                    let mut plans = Vec::new();
                    loop {
                        let index = next_job.fetch_add(1, Ordering::Relaxed);
                        let Some(job) = jobs.get(index) else {
                            break;
                        };
                        let plan = build_downsampled_vox_plan(
                            &job.scene,
                            job.object.as_deref(),
                            job.voxels_per_unit,
                            job.scale,
                            job.phase,
                            job.linear,
                        );
                        plans.push((job.key.clone(), plan));
                    }
                    plans
                })
            })
            .collect();
        for handle in handles {
            for (key, plan) in handle
                .join()
                .expect("downsample plan worker should not panic")
            {
                output.insert(key, plan);
            }
        }
    });
    output
}

fn build_downsampled_vox_plan(
    scene: &CachedVoxScene,
    object: Option<&str>,
    voxels_per_unit: u32,
    scale: f32,
    phase: Vec3,
    linear: [[f32; 3]; 3],
) -> DownsampledVoxPlan {
    if let Some(plan) =
        build_dense_downsampled_vox_plan(scene, object, voxels_per_unit, scale, phase, linear)
    {
        return plan;
    }
    let visible_voxel_count = scene
        .metadata(object)
        .map(|metadata| metadata.visible_voxel_count)
        .unwrap_or(0);
    let capacity = visible_voxel_count.min(262_144) as usize;
    let mut target_materials = HashMap::with_capacity(capacity);
    visit_cached_visible_voxels(scene, object, &scene.materials, |native_pos, material| {
        if voxels_per_unit == TEARDOWN_NATIVE_VOXELS_PER_UNIT {
            let source = transform_density_coord(linear, native_pos, voxels_per_unit);
            let target = floor_ivec3(source.as_vec3() * scale + phase);
            target_materials.insert(target, material);
        } else {
            for coord in expanded_native_coords(native_pos, voxels_per_unit) {
                let source = transform_density_coord(linear, coord, voxels_per_unit);
                let target = floor_ivec3(source.as_vec3() * scale + phase);
                target_materials.insert(target, material);
            }
        }
    });
    downsampled_vox_plan_from_cells(target_materials.into_iter().collect())
}

fn build_dense_downsampled_vox_plan(
    scene: &CachedVoxScene,
    object: Option<&str>,
    voxels_per_unit: u32,
    scale: f32,
    phase: Vec3,
    linear: [[f32; 3]; 3],
) -> Option<DownsampledVoxPlan> {
    let bounds = selected_object_bounds(scene, object)?;
    let (min, max_exclusive) =
        dense_downsampled_target_bounds(bounds, voxels_per_unit, scale, phase, linear)?;
    let extent = max_exclusive - min;
    if extent.x <= 0 || extent.y <= 0 || extent.z <= 0 {
        return Some(downsampled_vox_plan_from_cells(Vec::new()));
    }
    let volume = i64::from(extent.x) * i64::from(extent.y) * i64::from(extent.z);
    let max_dense_volume = scene
        .metadata(object)
        .map(|metadata| metadata.visible_voxel_count)
        .unwrap_or(0)
        .saturating_mul(8)
        .clamp(4_096, 20_000_000);
    if volume <= 0 || volume as u64 > max_dense_volume {
        return None;
    }

    let extent = UVec3::new(extent.x as u32, extent.y as u32, extent.z as u32);
    let xy_stride = extent.x as usize * extent.y as usize;
    let mut materials = vec![0u16; volume as usize];
    let mut outside_bounds = false;
    visit_cached_visible_voxels(scene, object, &scene.materials, |native_pos, material| {
        if voxels_per_unit == TEARDOWN_NATIVE_VOXELS_PER_UNIT {
            write_dense_plan_material(
                &mut materials,
                extent,
                xy_stride,
                min,
                transform_density_coord(linear, native_pos, voxels_per_unit),
                scale,
                phase,
                material,
                &mut outside_bounds,
            );
        } else {
            for coord in expanded_native_coords(native_pos, voxels_per_unit) {
                write_dense_plan_material(
                    &mut materials,
                    extent,
                    xy_stride,
                    min,
                    transform_density_coord(linear, coord, voxels_per_unit),
                    scale,
                    phase,
                    material,
                    &mut outside_bounds,
                );
            }
        }
    });
    if outside_bounds {
        return None;
    }

    let mut cells = Vec::with_capacity(materials.iter().filter(|material| **material != 0).count());
    for (index, material) in materials.into_iter().enumerate() {
        if material == 0 {
            continue;
        }
        let z = index / xy_stride;
        let rem = index - z * xy_stride;
        let y = rem / extent.x as usize;
        let x = rem - y * extent.x as usize;
        cells.push((min + IVec3::new(x as i32, y as i32, z as i32), material));
    }
    Some(downsampled_vox_plan_from_cells(cells))
}

fn downsampled_vox_plan_from_cells(cells: Vec<(IVec3, u16)>) -> DownsampledVoxPlan {
    let mut min = IVec3::splat(i32::MAX);
    let mut max_exclusive = IVec3::splat(i32::MIN);
    for (target, _material) in &cells {
        min = min.min(*target);
        max_exclusive = max_exclusive.max(*target + IVec3::ONE);
    }
    let bounds = (!cells.is_empty()).then_some(DownsampledVoxPlanBounds { min, max_exclusive });
    DownsampledVoxPlan { cells, bounds }
}

#[allow(clippy::too_many_arguments)]
fn write_dense_plan_material(
    materials: &mut [u16],
    extent: UVec3,
    xy_stride: usize,
    min: IVec3,
    source: IVec3,
    scale: f32,
    phase: Vec3,
    material: u16,
    outside_bounds: &mut bool,
) {
    let target = floor_ivec3(source.as_vec3() * scale + phase);
    let local_target = target - min;
    if local_target.x < 0
        || local_target.y < 0
        || local_target.z < 0
        || local_target.x >= extent.x as i32
        || local_target.y >= extent.y as i32
        || local_target.z >= extent.z as i32
    {
        *outside_bounds = true;
        return;
    }
    let index = local_target.x as usize
        + local_target.y as usize * extent.x as usize
        + local_target.z as usize * xy_stride;
    materials[index] = material;
}

fn dense_downsampled_target_bounds(
    bounds: VoxBounds,
    voxels_per_unit: u32,
    scale: f32,
    phase: Vec3,
    linear: [[f32; 3]; 3],
) -> Option<(IVec3, IVec3)> {
    if bounds.min.x >= bounds.max_exclusive.x
        || bounds.min.y >= bounds.max_exclusive.y
        || bounds.min.z >= bounds.max_exclusive.z
    {
        return None;
    }
    let x_range =
        expanded_native_coord_range(bounds.min.x, bounds.max_exclusive.x, voxels_per_unit);
    let y_range =
        expanded_native_coord_range(bounds.min.y, bounds.max_exclusive.y, voxels_per_unit);
    let z_range =
        expanded_native_coord_range(bounds.min.z, bounds.max_exclusive.z, voxels_per_unit);
    let max = IVec3::new(x_range.end - 1, y_range.end - 1, z_range.end - 1);
    let min = IVec3::new(x_range.start, y_range.start, z_range.start);
    let mut target_min = IVec3::splat(i32::MAX);
    let mut target_max_exclusive = IVec3::splat(i32::MIN);
    for x in [min.x, max.x] {
        for y in [min.y, max.y] {
            for z in [min.z, max.z] {
                let source = transform_density_coord(linear, IVec3::new(x, y, z), voxels_per_unit);
                let target = floor_ivec3(source.as_vec3() * scale + phase);
                target_min = target_min.min(target);
                target_max_exclusive = target_max_exclusive.max(target + IVec3::ONE);
            }
        }
    }
    Some((target_min - IVec3::ONE, target_max_exclusive + IVec3::ONE))
}

fn linear_transform(matrix: Mat4) -> [[f32; 3]; 3] {
    let cols = matrix.to_cols_array_2d();
    [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ]
}

fn quantized_linear_key(linear: [[f32; 3]; 3]) -> [[i16; 3]; 3] {
    linear.map(|row| {
        row.map(|value| {
            (value * DOWNSAMPLED_VOX_LINEAR_QUANTIZATION)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
    })
}

fn mat3_vec3(matrix: [[f32; 3]; 3], vector: Vec3) -> Vec3 {
    let values = [vector.x, vector.y, vector.z];
    Vec3::new(
        matrix[0][0] * values[0] + matrix[0][1] * values[1] + matrix[0][2] * values[2],
        matrix[1][0] * values[0] + matrix[1][1] * values[1] + matrix[1][2] * values[2],
        matrix[2][0] * values[0] + matrix[2][1] * values[1] + matrix[2][2] * values[2],
    )
}

fn transform_density_coord(matrix: [[f32; 3]; 3], coord: IVec3, voxels_per_unit: u32) -> IVec3 {
    round_world_point(
        mat3_vec3(
            matrix,
            native_point(coord.x, coord.y, coord.z, voxels_per_unit),
        ),
        voxels_per_unit,
    )
}

fn visit_cached_visible_voxels(
    scene: &CachedVoxScene,
    object: Option<&str>,
    materials: &[u16; 256],
    mut visit: impl FnMut(IVec3, u16),
) -> bool {
    if scene.scene.instances.is_empty() {
        if object.is_some() {
            return false;
        }
        for model in &scene.scene.models {
            for voxel in &model.voxels {
                visit(
                    voxel.position.as_ivec3(),
                    materials[voxel.color_index as usize],
                );
            }
        }
        return true;
    }

    let mut any = false;
    let origin = selected_object_origin(scene, object);
    for instance in scene
        .scene
        .instances
        .iter()
        .filter(|instance| instance_matches_selected_object(instance, object))
    {
        let Some(model) = scene.scene.models.get(instance.model_index) else {
            continue;
        };
        for voxel in &model.voxels {
            any = true;
            visit(
                transform_vox_instance_point(instance, voxel.position.as_ivec3()) - origin,
                materials[voxel.color_index as usize],
            );
        }
    }
    any
}

fn transform_vox_instance_point(
    instance: &crate::voxel::vox_loader::VoxInstance,
    point: IVec3,
) -> IVec3 {
    IVec3::new(
        instance.rotation[0][0] * point.x
            + instance.rotation[0][1] * point.y
            + instance.rotation[0][2] * point.z,
        instance.rotation[1][0] * point.x
            + instance.rotation[1][1] * point.y
            + instance.rotation[1][2] * point.z,
        instance.rotation[2][0] * point.x
            + instance.rotation[2][1] * point.y
            + instance.rotation[2][2] * point.z,
    ) + instance.translation
}

fn floor_ivec3(value: Vec3) -> IVec3 {
    IVec3::new(
        value.x.floor() as i32,
        value.y.floor() as i32,
        value.z.floor() as i32,
    )
}

fn ceil_ivec3(value: Vec3) -> IVec3 {
    IVec3::new(
        value.x.ceil() as i32,
        value.y.ceil() as i32,
        value.z.ceil() as i32,
    )
}

fn expanded_native_coords(native_pos: IVec3, voxels_per_unit: u32) -> impl Iterator<Item = IVec3> {
    let x = native_axis_range(native_pos.x, voxels_per_unit);
    let y = native_axis_range(native_pos.y, voxels_per_unit);
    let z = native_axis_range(native_pos.z, voxels_per_unit);
    x.flat_map(move |x| {
        let z_for_x = z.clone();
        y.clone()
            .flat_map(move |y| z_for_x.clone().map(move |z| IVec3::new(x, y, z)))
    })
}

fn voxbox_axis_counts(size: Vec3, voxels_per_unit: u32) -> IVec3 {
    IVec3::new(
        native_axis_count(size.x, voxels_per_unit),
        native_axis_count(size.y, voxels_per_unit),
        native_axis_count(size.z, voxels_per_unit),
    )
}

fn voxbox_local_bounds(counts: IVec3) -> (IVec3, IVec3) {
    (IVec3::ZERO, counts)
}

struct BrushPattern {
    materials: HashMap<IVec3, u16>,
    bounds: VoxBounds,
    extent: IVec3,
}

#[derive(Clone, Copy, Default)]
struct MirrorAxes {
    x: bool,
    y: bool,
    z: bool,
}

impl BrushPattern {
    fn from_scene(scene: &CachedVoxScene, object: Option<&str>) -> Option<Self> {
        let bounds = brush_pattern_bounds(scene, object)?;
        let extent = bounds.max_exclusive - bounds.min;
        if extent.x <= 0 || extent.y <= 0 || extent.z <= 0 {
            return None;
        }
        let capacity = scene
            .metadata(object)
            .map(|metadata| metadata.visible_voxel_count.min(262_144) as usize)
            .unwrap_or(0);
        let mut materials = HashMap::with_capacity(capacity);
        visit_cached_visible_voxels(scene, object, &scene.materials, |position, material| {
            materials.insert(position, material);
        });
        (!materials.is_empty()).then_some(Self {
            materials,
            bounds,
            extent,
        })
    }

    fn sample(
        &self,
        local_coord: IVec3,
        offset: IVec3,
        mirror: MirrorAxes,
        voxels_per_unit: u32,
    ) -> Option<u16> {
        let scale = TEARDOWN_NATIVE_VOXELS_PER_UNIT as f32 / voxels_per_unit.max(1) as f32;
        let coord = IVec3::new(
            round_half_up(local_coord.x as f32 * scale),
            round_half_up(local_coord.y as f32 * scale),
            round_half_up(local_coord.z as f32 * scale),
        ) + offset;
        let wrapped = IVec3::new(
            mirror_wrapped_axis(coord.x, self.bounds.min.x, self.extent.x, mirror.x),
            mirror_wrapped_axis(coord.y, self.bounds.min.y, self.extent.y, mirror.y),
            mirror_wrapped_axis(coord.z, self.bounds.min.z, self.extent.z, mirror.z),
        );
        self.materials.get(&wrapped).copied()
    }
}

impl MirrorAxes {
    fn any(self) -> bool {
        self.x || self.y || self.z
    }
}

fn mirror_native_coord(coord: IVec3, bounds: VoxBounds, mirror: MirrorAxes) -> IVec3 {
    IVec3::new(
        mirror_axis_in_bounds(coord.x, bounds.min.x, bounds.max_exclusive.x, mirror.x),
        mirror_axis_in_bounds(coord.y, bounds.min.y, bounds.max_exclusive.y, mirror.y),
        mirror_axis_in_bounds(coord.z, bounds.min.z, bounds.max_exclusive.z, mirror.z),
    )
}

fn mirror_axis_in_bounds(coord: i32, min: i32, max_exclusive: i32, mirror: bool) -> i32 {
    if mirror {
        min + max_exclusive - 1 - coord
    } else {
        coord
    }
}

fn mirror_wrapped_axis(coord: i32, min: i32, extent: i32, mirror: bool) -> i32 {
    let wrapped = (coord - min).rem_euclid(extent);
    if mirror {
        min + extent - 1 - wrapped
    } else {
        min + wrapped
    }
}

fn brush_pattern_bounds(scene: &CachedVoxScene, object: Option<&str>) -> Option<VoxBounds> {
    if scene.scene.instances.is_empty() {
        if object.is_some() {
            return None;
        }
        let mut bounds = VoxBounds {
            min: IVec3::splat(i32::MAX),
            max_exclusive: IVec3::splat(i32::MIN),
        };
        for model in &scene.scene.models {
            let size = model.size.as_ivec3();
            if size.x <= 0 || size.y <= 0 || size.z <= 0 {
                continue;
            }
            bounds.min = bounds.min.min(IVec3::ZERO);
            bounds.max_exclusive = bounds.max_exclusive.max(size);
        }
        return (bounds.min.x != i32::MAX).then_some(bounds);
    }

    let origin = selected_object_origin(scene, object);
    let mut bounds = VoxBounds {
        min: IVec3::splat(i32::MAX),
        max_exclusive: IVec3::splat(i32::MIN),
    };
    for instance in scene
        .scene
        .instances
        .iter()
        .filter(|instance| instance_matches_selected_object(instance, object))
    {
        let Some(model) = scene.scene.models.get(instance.model_index) else {
            continue;
        };
        let size = model.size.as_ivec3();
        if size.x <= 0 || size.y <= 0 || size.z <= 0 {
            continue;
        }
        for x in [0, size.x - 1] {
            for y in [0, size.y - 1] {
                for z in [0, size.z - 1] {
                    let position =
                        transform_vox_instance_point(instance, IVec3::new(x, y, z)) - origin;
                    bounds.min = bounds.min.min(position);
                    bounds.max_exclusive = bounds.max_exclusive.max(position + IVec3::ONE);
                }
            }
        }
    }
    (bounds.min.x != i32::MAX).then_some(bounds)
}

fn load_voxbox_brush_pattern(
    node: roxmltree::Node,
    source: &mut TeardownResourceSource,
    cache: &mut VoxCache,
    stats: &mut TeardownZipWriteStats,
) -> Option<Arc<BrushPattern>> {
    let brush = node.attribute("brush")?;
    if !is_vox_brush_reference(brush) {
        return None;
    }
    let (path, object) = parse_brush_reference(brush, node.attribute("object"));
    let key = (path, object);
    if let Some(pattern) = cache.brush_patterns.get(&key) {
        return pattern.clone();
    }
    let (path, object) = (&key.0, key.1.as_deref());
    let scene = cache.get(source, stats, path)?;
    let pattern = BrushPattern::from_scene(&scene, object).map(Arc::new);
    cache.brush_patterns.insert(key, pattern.clone());
    pattern
}

fn is_hole_brush(brush: Option<&str>) -> bool {
    matches!(
        brush.map(|brush| brush.trim().to_ascii_lowercase()),
        Some(brush) if brush == "hole" || brush == "hoe"
    )
}

fn parse_mirror_axes(node: roxmltree::Node) -> MirrorAxes {
    MirrorAxes {
        x: parse_bool_attr(node.attribute("mirrorx")),
        y: parse_bool_attr(node.attribute("mirrory")),
        z: parse_bool_attr(node.attribute("mirrorz")),
    }
}

fn parse_bool_attr(value: Option<&str>) -> bool {
    matches!(
        value.map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if value == "true" || value == "1"
    )
}

fn is_vox_brush_reference(brush: &str) -> bool {
    let path = brush
        .rsplit_once(':')
        .map(|(path, _)| path)
        .unwrap_or(brush);
    path.to_ascii_lowercase().ends_with(".vox")
}

fn parse_brush_reference(brush: &str, object: Option<&str>) -> (String, Option<String>) {
    if let Some((path, brush_object)) = brush.rsplit_once(':') {
        if is_vox_brush_reference(path) {
            return (path.to_owned(), Some(brush_object.to_owned()));
        }
    }
    (brush.to_owned(), object.map(str::to_owned))
}

fn tinted_materials(scene: &CachedVoxScene, tint: Option<ColorTint>) -> Option<[u16; 256]> {
    tint.map(|tint| {
        scene
            .scene
            .palette
            .map(|color| material_for_color(apply_color_tint(color, tint)))
    })
}

fn parse_color_tint(value: Option<&str>) -> Option<ColorTint> {
    let color = parse_color(value, [255, 255, 255, 255]);
    let tint = ColorTint {
        rgb: [color[0], color[1], color[2]],
    };
    (tint.rgb != [255, 255, 255]).then_some(tint)
}

fn apply_color_tint(color: [u8; 4], tint: ColorTint) -> [u8; 4] {
    [
        tint_channel(color[0], tint.rgb[0]),
        tint_channel(color[1], tint.rgb[1]),
        tint_channel(color[2], tint.rgb[2]),
        color[3],
    ]
}

fn apply_material_tint(material: u16, tint: Option<ColorTint>) -> u16 {
    let Some(tint) = tint else {
        return material;
    };
    const ENCODED_COLOR_MATERIAL_FLAG: u16 = 0x8000;
    if material & ENCODED_COLOR_MATERIAL_FLAG == 0 {
        return material;
    }
    let r = expand_5bit_to_u8((material >> 10) & 0x1f);
    let g = expand_5bit_to_u8((material >> 5) & 0x1f);
    let b = expand_5bit_to_u8(material & 0x1f);
    material_for_color(apply_color_tint([r, g, b, 255], tint))
}

fn tint_channel(base: u8, tint: u8) -> u8 {
    ((u16::from(base) * u16::from(tint) + 127) / 255) as u8
}

fn expand_5bit_to_u8(value: u16) -> u8 {
    ((value * 255 + 15) / 31) as u8
}

fn parse_native_offset(value: Option<&str>) -> IVec3 {
    let Some(value) = value else {
        return IVec3::ZERO;
    };
    let parts: Vec<Option<f32>> = value
        .split_whitespace()
        .map(parse_float_token_prefix)
        .collect();
    let x = parts.first().and_then(|value| *value).unwrap_or(0.0);
    let y = parts.get(1).and_then(|value| *value).unwrap_or(0.0);
    let z = parts.get(2).and_then(|value| *value).unwrap_or(0.0);
    IVec3::new(
        round_voxel_coord(x),
        round_voxel_coord(y),
        round_voxel_coord(z),
    )
}

fn emit_solid_voxbox_points<S>(
    local_min: IVec3,
    max_exclusive: IVec3,
    matrix: Mat4,
    voxels_per_unit: u32,
    material: u16,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) -> u64 {
    if local_min.x >= max_exclusive.x
        || local_min.y >= max_exclusive.y
        || local_min.z >= max_exclusive.z
    {
        return 0;
    }
    let inverse = matrix.inverse();
    if !inverse.is_finite() {
        return 0;
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for x in [local_min.x, max_exclusive.x] {
        for y in [local_min.y, max_exclusive.y] {
            for z in [local_min.z, max_exclusive.z] {
                let world = matrix.transform_point3(native_point(x, y, z, voxels_per_unit));
                min = min.min(world);
                max = max.max(world);
            }
        }
    }

    let start = floor_ivec3(min * voxels_per_unit as f32) - IVec3::ONE;
    let end = ceil_ivec3(max * voxels_per_unit as f32) + IVec3::ONE;
    let local_min = local_min.as_vec3();
    let local_max = max_exclusive.as_vec3();
    let mut emitted = 0;
    for x in start.x..end.x.max(start.x + 1) {
        for y in start.y..end.y.max(start.y + 1) {
            for z in start.z..end.z.max(start.z + 1) {
                let source = IVec3::new(x, y, z);
                let world = (source.as_vec3() + Vec3::splat(0.5)) / voxels_per_unit as f32;
                let local = inverse.transform_point3(world) * voxels_per_unit as f32;
                if local.x >= local_min.x - 0.5
                    && local.y >= local_min.y - 0.5
                    && local.z >= local_min.z - 0.5
                    && local.x < local_max.x + 0.5
                    && local.y < local_max.y + 0.5
                    && local.z < local_max.z + 0.5
                {
                    emit(source, material, sink);
                    emitted += 1;
                }
            }
        }
    }
    emitted
}

#[allow(clippy::too_many_arguments)]
fn emit_brushed_voxbox_points<S>(
    brush: &BrushPattern,
    counts: IVec3,
    offset: IVec3,
    mirror: MirrorAxes,
    matrix: Mat4,
    voxels_per_unit: u32,
    local_min: IVec3,
    tint: Option<ColorTint>,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) -> u64 {
    if counts.x <= 0 || counts.y <= 0 || counts.z <= 0 {
        return 0;
    }
    let mut count = 0;
    let density_cell_emission = density_cell_emission(matrix, voxels_per_unit);
    for x in 0..counts.x {
        for y in 0..counts.y {
            for z in 0..counts.z {
                let sample_coord = IVec3::new(x, y, z);
                let Some(material) = brush.sample(sample_coord, offset, mirror, voxels_per_unit)
                else {
                    continue;
                };
                let local = local_min + sample_coord;
                emit_transformed_density_cell(
                    local,
                    matrix,
                    voxels_per_unit,
                    apply_material_tint(material, tint),
                    density_cell_emission,
                    emit,
                    sink,
                );
                count += 1;
            }
        }
    }
    count
}

struct HeightmapImage {
    width: u32,
    height: u32,
    pixels: Vec<[u8; 4]>,
    height_scale: i32,
    tiles: Vec<HeightmapTile>,
    fill_boundary_columns: bool,
}

#[derive(Clone, Copy)]
struct HeightmapTile {
    coord_offset: IVec3,
    x_start: i32,
    z_start: i32,
    x_end: i32,
    z_end: i32,
}

fn load_voxscript_heightmap(
    node: roxmltree::Node,
    script: &[u8],
    source: &mut TeardownResourceSource,
) -> Option<HeightmapImage> {
    let script = std::str::from_utf8(script).ok()?;
    if !script.contains("Heightmap") || !script.contains("LoadImage") {
        return None;
    }
    let image_path = voxscript_parameter(node, "file")?;
    let height_scale = voxscript_parameter(node, "scale")
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(64)
        .max(0);
    let requested_tile_size =
        voxscript_parameter(node, "tilesize").and_then(|value| value.parse::<i32>().ok());
    let bytes = source.read_resource(image_path)?;
    let (width, height, pixels) = decode_png_rgba8(&bytes)?;
    let plan = parse_voxscript_heightmap_plan(script, width, height, requested_tile_size)?;
    Some(HeightmapImage {
        width,
        height,
        pixels,
        height_scale,
        tiles: plan.tiles,
        fill_boundary_columns: plan.fill_boundary_columns,
    })
}

struct VoxscriptHeightmapPlan {
    tiles: Vec<HeightmapTile>,
    fill_boundary_columns: bool,
}

#[derive(Clone, Copy)]
struct VoxscriptHeightmapArgs {
    range: Option<(i32, i32, i32, i32)>,
    fill_boundary_columns: bool,
}

fn parse_voxscript_heightmap_plan(
    script: &str,
    width: u32,
    height: u32,
    requested_tile_size: Option<i32>,
) -> Option<VoxscriptHeightmapPlan> {
    if let Some(plan) =
        parse_voxscript_tiled_heightmap_plan(script, width, height, requested_tile_size)
    {
        return Some(plan);
    }

    let heightmap_args = parse_voxscript_heightmap_args(script, width, height)?;
    let (x_start, z_start, x_end, z_end) =
        heightmap_args
            .range
            .unwrap_or((0, 0, width as i32, height as i32));
    Some(VoxscriptHeightmapPlan {
        tiles: vec![HeightmapTile {
            coord_offset: parse_voxscript_constant_vox_offset(script).unwrap_or(IVec3::ZERO),
            x_start,
            z_start,
            x_end,
            z_end,
        }],
        fill_boundary_columns: heightmap_args.fill_boundary_columns,
    })
}

fn parse_voxscript_tiled_heightmap_plan(
    script: &str,
    width: u32,
    height: u32,
    requested_tile_size: Option<i32>,
) -> Option<VoxscriptHeightmapPlan> {
    let compact = compact_lua(script);
    if !compact.contains("w,h=GetImageSize()")
        || !compact.contains("whiley0<hdo")
        || !compact.contains("whilex0<wdo")
        || !compact.contains("Vox(x0,0,y0)")
        || !compact.contains("Heightmap(x0,y0,x1,y1,")
    {
        return None;
    }

    let heightmap_args = parse_voxscript_heightmap_args(script, width, height)?;
    let tile_size = requested_tile_size
        .or_else(|| parse_voxscript_getint_default(script, "tilesize"))
        .unwrap_or(128)
        .max(1);
    let mut tiles = Vec::new();
    let width = width as i32;
    let height = height as i32;
    let mut z0 = 0;
    while z0 < height {
        let z1 = (z0 + tile_size).min(height);
        let mut x0 = 0;
        while x0 < width {
            let x1 = (x0 + tile_size).min(width);
            tiles.push(HeightmapTile {
                coord_offset: IVec3::ZERO,
                x_start: x0,
                z_start: z0,
                x_end: x1,
                z_end: z1,
            });
            x0 = x1;
        }
        z0 = z1;
    }

    (!tiles.is_empty()).then_some(VoxscriptHeightmapPlan {
        tiles,
        fill_boundary_columns: heightmap_args.fill_boundary_columns,
    })
}

fn parse_voxscript_heightmap_args(
    script: &str,
    width: u32,
    height: u32,
) -> Option<VoxscriptHeightmapArgs> {
    let args = lua_call_args(script, "Heightmap")?;
    let mut parts = args.split(',').map(|part| part.trim());
    let x0 = parts.next().and_then(parse_float_token_prefix);
    let z0 = parts.next().and_then(parse_float_token_prefix);
    let x1 = parts.next().and_then(parse_float_token_prefix);
    let z1 = parts.next().and_then(parse_float_token_prefix);
    let _height_scale = parts.next();
    let fill_boundary_columns = parts
        .next()
        .and_then(parse_lua_heightmap_fill_boundary_flag)
        .unwrap_or(true);

    let range = if let (Some(x0), Some(z0), Some(x1), Some(z1)) = (x0, z0, x1, z1) {
        let width = width as i32;
        let height = height as i32;
        Some((
            round_voxel_coord(x0).clamp(0, width),
            round_voxel_coord(z0).clamp(0, height),
            round_voxel_coord(x1).clamp(0, width),
            round_voxel_coord(z1).clamp(0, height),
        ))
    } else {
        None
    };

    Some(VoxscriptHeightmapArgs {
        range,
        fill_boundary_columns,
    })
}

fn parse_lua_heightmap_fill_boundary_flag(value: &str) -> Option<bool> {
    let value = value.trim().replace(' ', "");
    match value.as_str() {
        "true" | "1" => Some(true),
        "false" | "nil" | "0" | "hollow==0" => Some(false),
        _ => None,
    }
}

fn parse_voxscript_constant_vox_offset(script: &str) -> Option<IVec3> {
    let args = lua_call_args(script, "Vox")?;
    let mut parts = args.split(',').map(|part| part.trim());
    let x = parse_optional_lua_number(parts.next())?;
    let y = parse_optional_lua_number(parts.next())?;
    let z = parse_optional_lua_number(parts.next())?;
    Some(IVec3::new(
        round_voxel_coord(x),
        round_voxel_coord(y),
        round_voxel_coord(z),
    ))
}

fn parse_optional_lua_number(value: Option<&str>) -> Option<f32> {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return Some(0.0);
    };
    parse_float_token_prefix(value)
}

fn parse_voxscript_getint_default(script: &str, name: &str) -> Option<i32> {
    let needle = format!("GetInt(\"{name}\"");
    let start = script.find(&needle)?;
    let rest = script.get(start + needle.len()..)?;
    let comma = rest.find(',')?;
    let after_comma = rest.get(comma + 1..)?;
    let end = after_comma.find(')')?;
    after_comma[..end].trim().parse::<i32>().ok()
}

fn compact_lua(script: &str) -> String {
    script
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect()
}

fn lua_call_args<'a>(script: &'a str, name: &str) -> Option<&'a str> {
    let mut search_start = 0;
    while let Some(relative) = script[search_start..].find(name) {
        let name_start = search_start + relative;
        let name_end = name_start + name.len();
        let before = script[..name_start].chars().next_back();
        let after = script[name_end..].chars().next();
        if before.is_some_and(is_lua_identifier_char)
            || after.is_some_and(is_lua_identifier_char)
            || after != Some('(')
        {
            search_start = name_end;
            continue;
        }
        let args_start = name_end + 1;
        let args_end = script[args_start..].find(')')? + args_start;
        return Some(&script[args_start..args_end]);
    }
    None
}

fn is_lua_identifier_char(value: char) -> bool {
    value == '_' || value.is_ascii_alphanumeric()
}

fn scaled_voxscript_offset(offset: IVec3, voxels_per_unit: u32) -> IVec3 {
    IVec3::new(
        scale_native_coord(offset.x, voxels_per_unit),
        scale_native_coord(offset.y, voxels_per_unit),
        scale_native_coord(offset.z, voxels_per_unit),
    )
}

fn scale_native_coord(coord: i32, voxels_per_unit: u32) -> i32 {
    let scale = voxels_per_unit as f32 / TEARDOWN_NATIVE_VOXELS_PER_UNIT as f32;
    round_half_up(coord as f32 * scale)
}

fn voxscript_parameter<'a>(node: roxmltree::Node<'a, 'a>, name: &str) -> Option<&'a str> {
    node.children()
        .find(|child| child.is_element() && child.tag_name().name() == "parameters")
        .and_then(|parameters| parameters.attribute(name))
        .or_else(|| node.attribute(name))
}

fn decode_png_rgba8(bytes: &[u8]) -> Option<(u32, u32, Vec<[u8; 4]>)> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer).ok()?;
    let data = &buffer[..info.buffer_size()];
    let pixels = match info.color_type {
        png::ColorType::Rgba => data
            .chunks_exact(4)
            .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
            .collect(),
        png::ColorType::Rgb => data
            .chunks_exact(3)
            .map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
            .collect(),
        png::ColorType::Grayscale => data
            .iter()
            .map(|value| [*value, *value, *value, 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => data
            .chunks_exact(2)
            .map(|pixel| [pixel[0], pixel[0], pixel[0], pixel[1]])
            .collect(),
        png::ColorType::Indexed => return None,
    };
    Some((info.width, info.height, pixels))
}

fn emit_heightmap_columns<S>(
    heightmap: &HeightmapImage,
    matrix: Mat4,
    voxels_per_unit: u32,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) {
    for tile in &heightmap.tiles {
        let offset = scaled_voxscript_offset(tile.coord_offset, voxels_per_unit);
        for z in tile.z_start..tile.z_end {
            for x in tile.x_start..tile.x_end {
                let pixel = heightmap_pixel(heightmap, x, z);
                let native_y = heightmap_native_height(pixel[0], heightmap.height_scale);
                let material = heightmap_material(pixel);
                for_each_expanded_heightmap_visible_coord(
                    heightmap,
                    tile,
                    x,
                    native_y,
                    z,
                    voxels_per_unit,
                    |local| {
                        let local = local + offset;
                        let world = matrix.transform_point3(native_point(
                            local.x,
                            local.y,
                            local.z,
                            voxels_per_unit,
                        ));
                        emit(round_world_point(world, voxels_per_unit), material, sink);
                    },
                );
            }
        }
    }
}

fn heightmap_native_bounds(
    heightmap: &HeightmapImage,
    matrix: Mat4,
    voxels_per_unit: u32,
) -> Option<(IVec3, IVec3, u64)> {
    if heightmap.width == 0 || heightmap.height == 0 || heightmap.tiles.is_empty() {
        return None;
    }
    let mut min = IVec3::splat(i32::MAX);
    let mut max_exclusive = IVec3::splat(i32::MIN);
    let mut count = 0;
    for tile in &heightmap.tiles {
        if tile.x_start >= tile.x_end || tile.z_start >= tile.z_end {
            continue;
        }
        let mut max_native_y = 0;
        for z in tile.z_start..tile.z_end {
            for x in tile.x_start..tile.x_end {
                let native_y = heightmap_native_height(
                    heightmap_pixel(heightmap, x, z)[0],
                    heightmap.height_scale,
                );
                max_native_y = max_native_y.max(native_y);
                count += heightmap_visible_column_count(
                    heightmap,
                    tile,
                    x,
                    native_y,
                    z,
                    voxels_per_unit,
                );
            }
        }
        let x_range = expanded_native_coord_range(tile.x_start, tile.x_end, voxels_per_unit);
        let y_range = expanded_native_coord_range(0, max_native_y + 1, voxels_per_unit);
        let z_range = expanded_native_coord_range(tile.z_start, tile.z_end, voxels_per_unit);
        let offset = scaled_voxscript_offset(tile.coord_offset, voxels_per_unit);
        for x in [x_range.start, x_range.end - 1] {
            for y in [y_range.start, y_range.end - 1] {
                for z in [z_range.start, z_range.end - 1] {
                    let local = IVec3::new(x, y, z) + offset;
                    let world = matrix.transform_point3(native_point(
                        local.x,
                        local.y,
                        local.z,
                        voxels_per_unit,
                    ));
                    let target = round_world_point(world, voxels_per_unit);
                    min = min.min(target);
                    max_exclusive = max_exclusive.max(target + IVec3::ONE);
                }
            }
        }
    }
    (count > 0).then_some((min, max_exclusive, count))
}

fn heightmap_pixel(heightmap: &HeightmapImage, x: i32, z: i32) -> [u8; 4] {
    heightmap.pixels[z as usize * heightmap.width as usize + x as usize]
}

fn heightmap_native_height(red: u8, height_scale: i32) -> i32 {
    round_half_up(f32::from(red) / 255.0 * height_scale as f32)
}

fn heightmap_material(pixel: [u8; 4]) -> u16 {
    let [red, green, blue, _] = pixel;
    let color = if blue == 10 {
        [153, 153, 153, 255]
    } else if blue == 20 {
        [51, 51, 51, 255]
    } else if blue != 0 {
        [90, 90, 90, 255]
    } else if green > 96 {
        [48, 61, 43, 255]
    } else if red < 48 {
        [66, 59, 51, 255]
    } else {
        [77, 77, 77, 255]
    };
    material_for_color(color)
}

fn for_each_brushed_voxagon_cell(
    node: roxmltree::Node,
    brush: &BrushPattern,
    offset: IVec3,
    mirror: MirrorAxes,
    voxels_per_unit: u32,
    mut visit: impl FnMut(IVec3, u16),
) -> u64 {
    let Some(vertices) = voxagon_vertices(node, voxels_per_unit) else {
        return 0;
    };
    let extrude = voxagon_extrude(node);
    let axis = voxagon_axis(node);
    let (min, max_exclusive) = voxagon_local_bounds(&vertices);
    let mut count = 0;
    for depth in extrude.depths() {
        for v in min.y..max_exclusive.y {
            for x in min.x..max_exclusive.x {
                let sample = Vec2::new(x as f32 + 0.5, v as f32 + 0.5);
                if !point_in_polygon(sample, &vertices) {
                    continue;
                }
                let local = axis.local_coord(x, v, depth);
                let Some(material) = brush.sample(local, offset, mirror, voxels_per_unit) else {
                    continue;
                };
                visit(local, material);
                count += 1;
            }
        }
    }
    count
}

#[allow(clippy::too_many_arguments)]
fn emit_brushed_voxagon_points<S>(
    node: roxmltree::Node,
    brush: &BrushPattern,
    offset: IVec3,
    mirror: MirrorAxes,
    matrix: Mat4,
    voxels_per_unit: u32,
    tint: Option<ColorTint>,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) -> u64 {
    let density_cell_emission = density_cell_emission(matrix, voxels_per_unit);
    for_each_brushed_voxagon_cell(
        node,
        brush,
        offset,
        mirror,
        voxels_per_unit,
        |local, material| {
            emit_transformed_density_cell(
                local,
                matrix,
                voxels_per_unit,
                apply_material_tint(material, tint),
                density_cell_emission,
                emit,
                sink,
            );
        },
    )
}

fn for_each_expanded_heightmap_visible_coord(
    heightmap: &HeightmapImage,
    tile: &HeightmapTile,
    x: i32,
    max_y: i32,
    z: i32,
    voxels_per_unit: u32,
    mut visit: impl FnMut(IVec3),
) {
    let xs = native_axis_range(x, voxels_per_unit);
    let zs = native_axis_range(z, voxels_per_unit);
    let start_y = heightmap_visible_column_start_in_tile(heightmap, tile, x, z, max_y.max(0));
    for native_y in start_y..=max_y.max(0) {
        for x in xs.clone() {
            for y in native_axis_range(native_y, voxels_per_unit) {
                for z in zs.clone() {
                    visit(IVec3::new(x, y, z));
                }
            }
        }
    }
}

fn heightmap_visible_column_range(
    heightmap: &HeightmapImage,
    tile: &HeightmapTile,
    x: i32,
    max_y: i32,
    z: i32,
    voxels_per_unit: u32,
) -> (IVec3, IVec3, u64) {
    let x_range = native_axis_range(x, voxels_per_unit);
    let y_range = expanded_native_coord_range(
        heightmap_visible_column_start_in_tile(heightmap, tile, x, z, max_y.max(0)),
        max_y.max(0) + 1,
        voxels_per_unit,
    );
    let z_range = native_axis_range(z, voxels_per_unit);
    let min = IVec3::new(x_range.start, y_range.start, z_range.start);
    let max_exclusive = IVec3::new(x_range.end, y_range.end, z_range.end);
    let extent = max_exclusive - min;
    let count = extent.x.max(0) as u64 * extent.y.max(0) as u64 * extent.z.max(0) as u64;
    (min, max_exclusive, count)
}

fn heightmap_visible_column_count(
    heightmap: &HeightmapImage,
    tile: &HeightmapTile,
    x: i32,
    max_y: i32,
    z: i32,
    voxels_per_unit: u32,
) -> u64 {
    heightmap_visible_column_range(heightmap, tile, x, max_y, z, voxels_per_unit).2
}

#[cfg(test)]
fn heightmap_visible_column_start(heightmap: &HeightmapImage, x: i32, z: i32, max_y: i32) -> i32 {
    let tile = HeightmapTile {
        coord_offset: IVec3::ZERO,
        x_start: 0,
        z_start: 0,
        x_end: heightmap.width as i32,
        z_end: heightmap.height as i32,
    };
    heightmap_visible_column_start_in_tile(heightmap, &tile, x, z, max_y)
}

fn heightmap_visible_column_start_in_tile(
    heightmap: &HeightmapImage,
    tile: &HeightmapTile,
    x: i32,
    z: i32,
    max_y: i32,
) -> i32 {
    if heightmap.fill_boundary_columns
        && (x == tile.x_start || z == tile.z_start || x + 1 == tile.x_end || z + 1 == tile.z_end)
    {
        return 0;
    }
    let mut neighbor_min = max_y.max(0);
    for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
        if nx < tile.x_start || nz < tile.z_start || nx >= tile.x_end || nz >= tile.z_end {
            continue;
        }
        let neighbor_y = heightmap_native_height(
            heightmap_pixel(heightmap, nx, nz)[0],
            heightmap.height_scale,
        );
        neighbor_min = neighbor_min.min(neighbor_y);
    }
    (neighbor_min + 1).min(max_y)
}

fn emit_voxagon_points<S>(
    node: roxmltree::Node,
    matrix: Mat4,
    voxels_per_unit: u32,
    material: u16,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) {
    let Some(vertices) = voxagon_vertices(node, voxels_per_unit) else {
        return;
    };
    let extrude = voxagon_extrude(node);
    let axis = voxagon_axis(node);
    let (min, max_exclusive) = voxagon_local_bounds(&vertices);
    let density_cell_emission = density_cell_emission(matrix, voxels_per_unit);
    for depth in extrude.depths() {
        for v in min.y..max_exclusive.y {
            for x in min.x..max_exclusive.x {
                let sample = Vec2::new(x as f32 + 0.5, v as f32 + 0.5);
                if point_in_polygon(sample, &vertices) {
                    let local = axis.local_coord(x, v, depth);
                    emit_transformed_density_cell(
                        local,
                        matrix,
                        voxels_per_unit,
                        material,
                        density_cell_emission,
                        emit,
                        sink,
                    );
                }
            }
        }
    }
}

fn voxagon_native_bounds(
    node: roxmltree::Node,
    matrix: Mat4,
    voxels_per_unit: u32,
) -> Option<(IVec3, IVec3)> {
    let (min, max_exclusive) = voxagon_local_native_bounds(node, voxels_per_unit)?;
    transformed_native_volume_bounds(min, max_exclusive, matrix, voxels_per_unit)
}

fn voxagon_local_native_bounds(
    node: roxmltree::Node,
    voxels_per_unit: u32,
) -> Option<(IVec3, IVec3)> {
    let vertices = voxagon_vertices(node, voxels_per_unit)?;
    let extrude = voxagon_extrude(node);
    let axis = voxagon_axis(node);
    let (min, max_exclusive) = voxagon_local_bounds(&vertices);
    let mut local_min = IVec3::splat(i32::MAX);
    let mut local_max_exclusive = IVec3::splat(i32::MIN);
    for u in [min.x, max_exclusive.x - 1] {
        for v in [min.y, max_exclusive.y - 1] {
            for depth in [0, extrude.last_depth()] {
                let local = axis.local_coord(u, v, depth);
                local_min = local_min.min(local);
                local_max_exclusive = local_max_exclusive.max(local + IVec3::ONE);
            }
        }
    }
    Some((local_min, local_max_exclusive))
}

fn voxagon_vertices(node: roxmltree::Node, voxels_per_unit: u32) -> Option<Vec<Vec2>> {
    let vertices: Vec<_> = node
        .children()
        .filter(|child| child.is_element() && child.tag_name().name() == "vertex")
        .filter_map(|child| parse_vec2(child.attribute("pos")))
        .map(|vertex| vertex * voxels_per_unit as f32)
        .collect();
    (vertices.len() >= 3).then_some(vertices)
}

#[derive(Debug, Clone, Copy)]
struct VoxagonExtrude {
    count: i32,
    sign: i32,
}

impl VoxagonExtrude {
    fn depths(self) -> impl Iterator<Item = i32> {
        (0..self.count).map(move |depth| depth * self.sign)
    }

    fn last_depth(self) -> i32 {
        (self.count - 1) * self.sign
    }
}

fn voxagon_extrude(node: roxmltree::Node) -> VoxagonExtrude {
    let value = node
        .attribute("extrude")
        .and_then(|value| value.parse::<f32>().ok())
        .map(round_half_up)
        .unwrap_or(1);
    VoxagonExtrude {
        count: value.abs().max(1),
        sign: if value < 0 { -1 } else { 1 },
    }
}

#[derive(Debug, Clone, Copy)]
enum VoxagonExtrudeAxis {
    X(i32),
    Y(i32),
    Z(i32),
}

impl VoxagonExtrudeAxis {
    fn local_coord(self, u: i32, v: i32, depth: i32) -> IVec3 {
        match self {
            Self::X(sign) => IVec3::new(depth * sign, u, v),
            Self::Y(sign) => IVec3::new(u, depth * sign, v),
            Self::Z(sign) => IVec3::new(u, v, depth * sign),
        }
    }
}

fn voxagon_axis(node: roxmltree::Node) -> VoxagonExtrudeAxis {
    if let Some(axis) = node.attribute("axis").and_then(parse_axis_letter) {
        return axis;
    }
    let axis = parse_vec3(node.attribute("axis"), Vec3::Y);
    let abs = axis.abs();
    if abs.x >= abs.y && abs.x >= abs.z && abs.x > 0.0 {
        VoxagonExtrudeAxis::X(axis_sign(axis.x))
    } else if abs.z >= abs.x && abs.z >= abs.y && abs.z > 0.0 {
        VoxagonExtrudeAxis::Z(axis_sign(axis.z))
    } else {
        VoxagonExtrudeAxis::Y(axis_sign(axis.y))
    }
}

fn parse_axis_letter(value: &str) -> Option<VoxagonExtrudeAxis> {
    let value = value.trim().to_ascii_lowercase();
    let (sign, axis) = value
        .strip_prefix('-')
        .map(|axis| (-1, axis))
        .unwrap_or((1, value.as_str()));
    match axis {
        "x" => Some(VoxagonExtrudeAxis::X(sign)),
        "y" => Some(VoxagonExtrudeAxis::Y(sign)),
        "z" => Some(VoxagonExtrudeAxis::Z(sign)),
        _ => None,
    }
}

fn axis_sign(value: f32) -> i32 {
    if value < 0.0 { -1 } else { 1 }
}

fn voxagon_local_bounds(vertices: &[Vec2]) -> (IVec2, IVec2) {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for vertex in vertices {
        min = min.min(*vertex);
        max = max.max(*vertex);
    }
    let min = IVec2::new(min.x.floor() as i32, min.y.floor() as i32);
    let max_exclusive = IVec2::new(max.x.ceil() as i32, max.y.ceil() as i32);
    (
        min,
        max_exclusive
            .max(min + IVec2::ONE)
            .max(IVec2::new(min.x + 1, min.y + 1)),
    )
}

fn point_in_polygon(point: Vec2, vertices: &[Vec2]) -> bool {
    let mut inside = false;
    let mut previous = vertices.len() - 1;
    for current in 0..vertices.len() {
        let a = vertices[current];
        let b = vertices[previous];
        let crosses = (a.y > point.y) != (b.y > point.y);
        if crosses {
            let x = (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
            if point.x < x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

fn native_point(x: i32, y: i32, z: i32, voxels_per_unit: u32) -> Vec3 {
    Vec3::new(
        x as f32 / voxels_per_unit as f32,
        y as f32 / voxels_per_unit as f32,
        z as f32 / voxels_per_unit as f32,
    )
}

fn native_axis_range(coord: i32, voxels_per_unit: u32) -> std::ops::Range<i32> {
    let scale = voxels_per_unit as f32 / TEARDOWN_NATIVE_VOXELS_PER_UNIT as f32;
    let start = round_half_up(coord as f32 * scale);
    let end = round_half_up((coord + 1) as f32 * scale);
    start..end.max(start + 1)
}

fn expanded_native_coord_range(
    min: i32,
    max_exclusive: i32,
    voxels_per_unit: u32,
) -> std::ops::Range<i32> {
    let start = native_axis_range(min, voxels_per_unit).start;
    let end = native_axis_range(max_exclusive - 1, voxels_per_unit).end;
    start..end.max(start + 1)
}

fn native_axis_count(size: f32, voxels_per_unit: u32) -> i32 {
    let scale = voxels_per_unit as f32 / TEARDOWN_NATIVE_VOXELS_PER_UNIT as f32;
    round_half_up(size.max(0.0) * scale).max(0)
}

fn round_world_point(point: Vec3, voxels_per_unit: u32) -> IVec3 {
    IVec3::new(
        round_voxel_coord(point.x * voxels_per_unit as f32),
        round_voxel_coord(point.y * voxels_per_unit as f32),
        round_voxel_coord(point.z * voxels_per_unit as f32),
    )
}

fn round_half_up(value: f32) -> i32 {
    (value + 0.5).floor() as i32
}

fn round_voxel_coord(value: f32) -> i32 {
    if value >= 0.0 {
        (value + 0.5).floor() as i32
    } else {
        (value - 0.5).ceil() as i32
    }
}

fn parse_vec3(value: Option<&str>, default: Vec3) -> Vec3 {
    let Some(value) = value else {
        return default;
    };
    let parts: Vec<Option<f32>> = value
        .split_whitespace()
        .map(parse_float_token_prefix)
        .collect();
    match parts.as_slice() {
        [Some(x)] => Vec3::splat(*x),
        [Some(x), Some(y)] => Vec3::new(*x, *y, *x),
        [] => default,
        _ => Vec3::new(
            parts.first().and_then(|value| *value).unwrap_or(default.x),
            parts.get(1).and_then(|value| *value).unwrap_or(default.y),
            parts.get(2).and_then(|value| *value).unwrap_or(default.z),
        ),
    }
}

fn parse_scale(value: Option<&str>) -> Vec3 {
    let scale = parse_vec3(value, Vec3::ONE);
    Vec3::new(scale.x.max(0.0), scale.y.max(0.0), scale.z.max(0.0))
}

fn parse_vec2(value: Option<&str>) -> Option<Vec2> {
    let value = value?;
    let parts: Vec<Option<f32>> = value
        .split_whitespace()
        .map(parse_float_token_prefix)
        .collect();
    let x = parts.first().and_then(|value| *value)?;
    let y = parts.get(1).and_then(|value| *value)?;
    Some(Vec2::new(x, y))
}

fn parse_color(value: Option<&str>, default: [u8; 4]) -> [u8; 4] {
    let Some(value) = value else {
        return default;
    };
    let parts: Vec<f32> = value
        .split_whitespace()
        .filter_map(parse_float_token_prefix)
        .collect();
    if parts.len() < 3 {
        return default;
    }
    if parts[0].max(parts[1]).max(parts[2]) <= 1.0 {
        [
            (parts[0] * 255.0).round() as u8,
            (parts[1] * 255.0).round() as u8,
            (parts[2] * 255.0).round() as u8,
            255,
        ]
    } else {
        [
            parts[0].round().clamp(0.0, 255.0) as u8,
            parts[1].round().clamp(0.0, 255.0) as u8,
            parts[2].round().clamp(0.0, 255.0) as u8,
            255,
        ]
    }
}

fn parse_float_token_prefix(token: &str) -> Option<f32> {
    if let Ok(value) = token.parse::<f32>() {
        return Some(value);
    }
    for (index, _) in token.char_indices().rev() {
        if index == 0 {
            break;
        }
        if let Ok(value) = token[..index].parse::<f32>() {
            return Some(value);
        }
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct SourceBounds {
    min: IVec3,
    max_exclusive: IVec3,
    any: bool,
}

impl Default for SourceBounds {
    fn default() -> Self {
        Self {
            min: IVec3::splat(i32::MAX),
            max_exclusive: IVec3::splat(i32::MIN),
            any: false,
        }
    }
}

impl SourceBounds {
    fn include(&mut self, pos: IVec3) {
        self.any = true;
        self.min = self.min.min(pos);
        self.max_exclusive = self.max_exclusive.max(pos + IVec3::ONE);
    }

    fn finish(self) -> Option<Self> {
        self.any.then_some(self)
    }
}

impl From<SourceBounds> for TeardownSourceBounds {
    fn from(bounds: SourceBounds) -> Self {
        Self {
            min: bounds.min,
            max_exclusive: bounds.max_exclusive,
        }
    }
}

struct TargetMapping {
    source_min: IVec3,
    source_max_exclusive: IVec3,
    scale: f32,
    padding: f32,
    scale_one_padding: Option<IVec3>,
}

impl TargetMapping {
    fn new(bounds: SourceBounds, world_size: UVec3) -> Self {
        let extent = (bounds.max_exclusive - bounds.min).as_vec3().max(Vec3::ONE);
        let world = world_size.as_vec3();
        let padding = if world.min_element() > 16.0 { 4.0 } else { 1.0 };
        let usable = (world - Vec3::splat(padding * 2.0)).max(Vec3::ONE);
        let scale = (usable / extent).min_element().min(1.0);
        Self {
            source_min: bounds.min,
            source_max_exclusive: bounds.max_exclusive,
            scale,
            padding,
            scale_one_padding: (scale == 1.0 && padding.fract() == 0.0)
                .then(|| IVec3::splat(padding as i32)),
        }
    }

    fn scale_millis(&self) -> u32 {
        (self.scale * 1000.0).round().clamp(0.0, u32::MAX as f32) as u32
    }

    fn map(&self, source: IVec3) -> Option<UVec3> {
        if let Some(padding) = self.scale_one_padding {
            return self.map_scale_one(source, padding);
        }
        let target =
            ((source - self.source_min).as_vec3() * self.scale + Vec3::splat(self.padding)).floor();
        if target.x < 0.0 || target.y < 0.0 || target.z < 0.0 {
            return None;
        }
        Some(UVec3::new(
            target.x as u32,
            target.y as u32,
            target.z as u32,
        ))
    }

    fn map_scale_one(&self, source: IVec3, padding: IVec3) -> Option<UVec3> {
        let target = source - self.source_min + padding;
        if target.x < 0 || target.y < 0 || target.z < 0 {
            return None;
        }
        Some(UVec3::new(
            target.x as u32,
            target.y as u32,
            target.z as u32,
        ))
    }

    fn clipped_source_box(
        &self,
        source_min: IVec3,
        source_max_exclusive: IVec3,
    ) -> Option<(IVec3, IVec3)> {
        let min = source_min.max(self.source_min);
        let max_exclusive = source_max_exclusive.min(self.source_max_exclusive);
        (min.x < max_exclusive.x && min.y < max_exclusive.y && min.z < max_exclusive.z)
            .then_some((min, max_exclusive))
    }

    fn base_target_and_phase(&self, source_translation: IVec3) -> (IVec3, Vec3, [i32; 3]) {
        let offset = (source_translation - self.source_min).as_vec3() * self.scale
            + Vec3::splat(self.padding);
        let base = floor_ivec3(offset);
        let phase = offset - base.as_vec3();
        let buckets = DOWNSAMPLED_VOX_PHASE_BUCKETS as f32;
        let quantized = [
            ((phase.x * buckets).floor() as i32).clamp(0, DOWNSAMPLED_VOX_PHASE_BUCKETS - 1),
            ((phase.y * buckets).floor() as i32).clamp(0, DOWNSAMPLED_VOX_PHASE_BUCKETS - 1),
            ((phase.z * buckets).floor() as i32).clamp(0, DOWNSAMPLED_VOX_PHASE_BUCKETS - 1),
        ];
        let quantized_phase = Vec3::new(
            quantized[0] as f32 / buckets,
            quantized[1] as f32 / buckets,
            quantized[2] as f32 / buckets,
        );
        (base, quantized_phase, quantized)
    }
}

#[derive(Debug, Clone, Copy)]
struct TargetWriteStats {
    written_voxels: u64,
    out_of_bounds_voxels: u64,
    target_min: UVec3,
    target_max_exclusive: UVec3,
}

impl Default for TargetWriteStats {
    fn default() -> Self {
        Self {
            written_voxels: 0,
            out_of_bounds_voxels: 0,
            target_min: UVec3::splat(u32::MAX),
            target_max_exclusive: UVec3::ZERO,
        }
    }
}

struct TargetWriteStatsTracker {
    written_voxels: Cell<u64>,
    out_of_bounds_voxels: Cell<u64>,
    target_min_x: Cell<u32>,
    target_min_y: Cell<u32>,
    target_min_z: Cell<u32>,
    target_max_x: Cell<u32>,
    target_max_y: Cell<u32>,
    target_max_z: Cell<u32>,
}

impl Default for TargetWriteStatsTracker {
    fn default() -> Self {
        Self {
            written_voxels: Cell::new(0),
            out_of_bounds_voxels: Cell::new(0),
            target_min_x: Cell::new(u32::MAX),
            target_min_y: Cell::new(u32::MAX),
            target_min_z: Cell::new(u32::MAX),
            target_max_x: Cell::new(0),
            target_max_y: Cell::new(0),
            target_max_z: Cell::new(0),
        }
    }
}

impl TargetWriteStatsTracker {
    fn record_written(&self, target: UVec3) {
        self.record_count_bounds(1, target, target + UVec3::ONE);
    }

    fn record_count_bounds(&self, count: u64, min: UVec3, max_exclusive: UVec3) {
        self.written_voxels
            .set(self.written_voxels.get().saturating_add(count));
        self.target_min_x.set(self.target_min_x.get().min(min.x));
        self.target_min_y.set(self.target_min_y.get().min(min.y));
        self.target_min_z.set(self.target_min_z.get().min(min.z));
        self.target_max_x
            .set(self.target_max_x.get().max(max_exclusive.x));
        self.target_max_y
            .set(self.target_max_y.get().max(max_exclusive.y));
        self.target_max_z
            .set(self.target_max_z.get().max(max_exclusive.z));
    }

    fn record_out_of_bounds(&self, count: u64) {
        self.out_of_bounds_voxels
            .set(self.out_of_bounds_voxels.get().saturating_add(count));
    }

    fn finish(&self) -> TargetWriteStats {
        TargetWriteStats {
            written_voxels: self.written_voxels.get(),
            out_of_bounds_voxels: self.out_of_bounds_voxels.get(),
            target_min: UVec3::new(
                self.target_min_x.get(),
                self.target_min_y.get(),
                self.target_min_z.get(),
            ),
            target_max_exclusive: UVec3::new(
                self.target_max_x.get(),
                self.target_max_y.get(),
                self.target_max_z.get(),
            ),
        }
    }
}

/// A single target-space write operation captured during the collection walk, in
/// document order (vector index == global order). Replayed per brick, sorted by index,
/// so overlapping writes resolve last-writer-wins identically to the serial path.
enum WriteOp {
    Point { target: UVec3, material: u16 },
    FillBox {
        min: UVec3,
        max_exclusive: UVec3,
        material: u16,
    },
    /// Deferred vox-scene transform (Step 2): expanded into Point ops in parallel.
    VoxScene(Box<VoxSceneJob>),
}

struct VoxSceneJob {
    scene: Arc<CachedVoxScene>,
    object: Option<String>,
    tint: Option<ColorTint>,
    mirror: MirrorAxes,
    matrix: Mat4,
    voxels_per_unit: u32,
}

/// Ordered op stream collected during the (serial) write walk when parallel write is
/// enabled. Points/boxes/jobs are interleaved in document order via `ops`.
#[derive(Default)]
struct WriteOpLog {
    ops: Vec<WriteOp>,
    out_of_bounds: u64,
}

/// Per-brick replay entry: `order` is the originating op's document-order index (for
/// jobs, the job op's index; ties broken by intra-job emission order in a stable sort),
/// enabling per-brick last-writer-wins that matches the serial single-threaded walk.
#[derive(Clone, Copy)]
struct BrickWriteEntry {
    order: u64,
    kind: BrickEntryKind,
}

#[derive(Clone, Copy)]
enum BrickEntryKind {
    Point { morton: u16, material: u16 },
    FillBox {
        local_min: UVec3,
        local_max_exclusive: UVec3,
        material: u16,
    },
}

fn write_target_voxel(
    ucvh: &Ucvh,
    bricks: &mut StagedBricks,
    target: UVec3,
    material: u16,
) -> bool {
    let brick_pos = target / BRICK_EDGE;
    let local_pos = target - brick_pos * BRICK_EDGE;
    let cell = if material == 0 {
        VoxelCell::AIR
    } else {
        VoxelCell::new(material, 1, [0; 3])
    };
    bricks
        .get_or_insert_with(ucvh, brick_pos)
        .write(local_pos, cell)
}

impl WriteOpLog {
    fn push_point(&mut self, target: UVec3, material: u16) {
        self.ops.push(WriteOp::Point { target, material });
    }

    fn push_fill_box(&mut self, min: UVec3, max_exclusive: UVec3, material: u16) {
        self.ops.push(WriteOp::FillBox {
            min,
            max_exclusive,
            material,
        });
    }

    fn push_vox_scene(
        &mut self,
        scene: Arc<CachedVoxScene>,
        object: Option<&str>,
        tint: Option<ColorTint>,
        mirror: MirrorAxes,
        matrix: Mat4,
        voxels_per_unit: u32,
    ) {
        self.ops.push(WriteOp::VoxScene(Box::new(VoxSceneJob {
            scene,
            object: object.map(str::to_owned),
            tint,
            mirror,
            matrix,
            voxels_per_unit,
        })));
    }
}

/// Emit the visible voxels of a cached vox scene (the exact body of the vox handler's
/// non-downsampled emit path) to a generic `emit` callback. Used by the parallel
/// write path (jobs) and the serial write path (mirrored/fallback nodes).
fn emit_vox_scene_points<S>(
    scene: &CachedVoxScene,
    object: Option<&str>,
    tint: Option<ColorTint>,
    mirror: MirrorAxes,
    matrix: Mat4,
    voxels_per_unit: u32,
    emit: &mut impl FnMut(IVec3, u16, &mut S),
    sink: &mut S,
) {
    let Some(bounds) = selected_object_bounds(scene, object) else {
        return;
    };
    let tinted_materials = tinted_materials(scene, tint);
    let materials = tinted_materials.as_ref().unwrap_or(&scene.materials);
    let emission = density_cell_emission(matrix, voxels_per_unit);
    visit_cached_visible_voxels(scene, object, materials, |native_pos, material| {
        let native_pos = mirror_native_coord(native_pos, bounds, mirror);
        for local in expanded_native_coords(native_pos, voxels_per_unit) {
            emit_transformed_density_cell(local, matrix, voxels_per_unit, material, emission, emit, sink);
        }
    });
}

/// Bucket collected ops into per-brick entry lists in document order, then replay each
/// brick in parallel via the existing `StagedBrick` methods. Because every brick applies
/// its ops in ascending document order using the same mutators as the serial walk, the
/// resulting bricks (content, occupancy, touched_count) and aggregate stats are
/// bit-identical to the single-threaded path — only the brick dimension is parallelized.
///
/// `WriteOp::VoxScene` entries are expanded in parallel across threads before bucketing,
/// allowing the expensive per-voxel transform work to run concurrently.
fn replay_write_op_log(
    ucvh: &Ucvh,
    op_log: WriteOpLog,
    grid_size: UVec3,
    target_mapping: &TargetMapping,
    world_size: UVec3,
) -> (Vec<(UVec3, StagedBrick)>, TargetWriteStats) {
    let profile = std::env::var_os("REVOLUMETRIC_TEARDOWN_PROFILE").is_some();

    // ── Phase 1: parallel VoxScene expansion ────────────────────────────────
    // Collect every VoxScene op with its op-index, expand them in parallel,
    // storing per-op expanded points indexed by op-index. The serial bucketing
    // phase (Phase 2) uses these pre-computed results instead of running the
    // per-voxel transform inline — combining parallel transforms with a simple
    // single-threaded scatter.
    let expand_start = Instant::now();
    let voxscene_indices: Vec<(usize, &VoxSceneJob)> = op_log
        .ops
        .iter()
        .enumerate()
        .filter_map(|(i, op)| match op {
            WriteOp::VoxScene(job) => Some((i, job.as_ref())),
            _ => None,
        })
        .collect();

    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // expanded[k] = (op_index, Vec<(UVec3 target, u16 material)>) for VoxScene job k.
    let mut expanded: Vec<(usize, Vec<(UVec3, u16)>)> =
        voxscene_indices.iter().map(|(i, _)| (*i, Vec::new())).collect();
    let worker_chunk = voxscene_indices.len().div_ceil(worker_count.max(1));
    let expanded_slice = &mut expanded;
    let jobs_slice = &voxscene_indices[..];

    std::thread::scope(|scope| {
        for chunk_result in expanded_slice
            .chunks_mut(worker_chunk)
            .zip(jobs_slice.chunks(worker_chunk))
        {
            let (out_chunk, job_chunk) = chunk_result;
            scope.spawn(move || {
                for ((_, points), (_, job)) in out_chunk.iter_mut().zip(job_chunk.iter()) {
                    emit_vox_scene_points(
                        &job.scene,
                        job.object.as_deref(),
                        job.tint,
                        job.mirror,
                        job.matrix,
                        job.voxels_per_unit,
                        &mut |source_pos: IVec3, material: u16, _: &mut ()| {
                            let Some(target) = target_mapping.map(source_pos) else {
                                return;
                            };
                            if target.x < world_size.x
                                && target.y < world_size.y
                                && target.z < world_size.z
                            {
                                points.push((target, material));
                            }
                        },
                        &mut (),
                    );
                }
            });
        }
    });

    // Build op-index → expanded-points lookup (only for VoxScene ops).
    // Vec index = same as expanded[], linear scan is fine (2529 jobs).
    let expanded = expanded;

    if profile {
        let total_pts: usize = expanded.iter().map(|(_, v)| v.len()).sum();
        eprintln!(
            "teardown_zip_profile expand={:.3}s jobs={} pts={}",
            expand_start.elapsed().as_secs_f64(),
            voxscene_indices.len(),
            total_pts,
        );
    }

    // ── Phase 2: serial bucketing ──────────────────────────────────────────
    // Single pass over op_log.ops in document order. All ops land in their
    // brick's entry list in ascending order — no sort needed.
    let bucket_start = Instant::now();
    let mut buckets: U64IndexMap<usize> = new_u64_index_map();
    let mut brick_entries: Vec<(UVec3, Vec<BrickWriteEntry>)> = Vec::new();
    let brick_key = |brick_pos: UVec3| -> u64 {
        u64::from(
            brick_pos.x + brick_pos.y * grid_size.x + brick_pos.z * grid_size.x * grid_size.y,
        )
    };
    let mut bucket_for = |buckets: &mut U64IndexMap<usize>,
                          entries: &mut Vec<(UVec3, Vec<BrickWriteEntry>)>,
                          brick_pos: UVec3|
     -> usize {
        let key = brick_key(brick_pos);
        match buckets.get(&key).copied() {
            Some(i) => i,
            None => {
                let i = entries.len();
                buckets.insert(key, i);
                entries.push((brick_pos, Vec::new()));
                i
            }
        }
    };

    let mut expanded_iter = expanded.iter().peekable();
    for (order, op) in op_log.ops.iter().enumerate() {
        let order = order as u64;
        match op {
            WriteOp::Point { target, material } => {
                let brick_pos = *target / BRICK_EDGE;
                let local = *target - brick_pos * BRICK_EDGE;
                let morton = morton::encode(local.x, local.y, local.z) as u16;
                let idx = bucket_for(&mut buckets, &mut brick_entries, brick_pos);
                brick_entries[idx].1.push(BrickWriteEntry {
                    order,
                    kind: BrickEntryKind::Point { morton, material: *material },
                });
            }
            WriteOp::FillBox { min, max_exclusive, material } => {
                let brick_min = *min / BRICK_EDGE;
                let brick_max = (*max_exclusive - UVec3::ONE) / BRICK_EDGE;
                for bz in brick_min.z..=brick_max.z {
                    for by in brick_min.y..=brick_max.y {
                        for bx in brick_min.x..=brick_max.x {
                            let brick_pos = UVec3::new(bx, by, bz);
                            let brick_base = brick_pos * BRICK_EDGE;
                            let local_min = min.saturating_sub(brick_base);
                            let local_max = (*max_exclusive - brick_base).min(UVec3::splat(BRICK_EDGE));
                            let idx = bucket_for(&mut buckets, &mut brick_entries, brick_pos);
                            brick_entries[idx].1.push(BrickWriteEntry {
                                order,
                                kind: BrickEntryKind::FillBox {
                                    local_min,
                                    local_max_exclusive: local_max,
                                    material: *material,
                                },
                            });
                        }
                    }
                }
            }
            WriteOp::VoxScene(_) => {
                // Use the pre-expanded points from Phase 1.
                if let Some((exp_op_idx, points)) = expanded_iter.peek().copied() {
                    if *exp_op_idx == order as usize {
                        expanded_iter.next();
                        for (intra, (target, material)) in points.iter().enumerate() {
                            let brick_pos = *target / BRICK_EDGE;
                            let local = *target - brick_pos * BRICK_EDGE;
                            let morton = morton::encode(local.x, local.y, local.z) as u16;
                            // Encode (job_order << 20 | intra) so multiple voxels from
                            // the same job hitting the same brick apply in visit order.
                            let entry_order = (order << 20) | intra as u64;
                            let idx = bucket_for(&mut buckets, &mut brick_entries, brick_pos);
                            brick_entries[idx].1.push(BrickWriteEntry {
                                order: entry_order,
                                kind: BrickEntryKind::Point { morton, material: *material },
                            });
                        }
                    }
                }
            }
        }
    }

    if profile {
        eprintln!(
            "teardown_zip_profile replay_bucket={:.3}s bricks={}",
            bucket_start.elapsed().as_secs_f64(),
            brick_entries.len(),
        );
    }

    // ── Phase 3: parallel brick apply ─────────────────────────────────────
    let apply_start = Instant::now();
    let apply_workers = worker_count.min(brick_entries.len().max(1));
    let next_index = AtomicUsize::new(0);
    let brick_entries_ref = &brick_entries;
    let results: Vec<(Vec<(UVec3, StagedBrick)>, TargetWriteStats)> = std::thread::scope(|scope| {
        (0..apply_workers)
            .map(|_| {
                let ni = &next_index;
                scope.spawn(move || {
                    let mut local_bricks = Vec::new();
                    let mut local_stats = TargetWriteStats::default();
                    loop {
                        let index = ni.fetch_add(1, Ordering::Relaxed);
                        let Some((brick_pos, entries)) = brick_entries_ref.get(index) else {
                            break;
                        };
                        let mut brick = StagedBrick::new_seeded(ucvh, *brick_pos);
                        let brick_base = *brick_pos * BRICK_EDGE;
                        for entry in entries {
                            match entry.kind {
                                BrickEntryKind::Point { morton, material } => {
                                    let cell = material_cell(material);
                                    if brick.write_morton(morton as usize, cell) {
                                        let world = brick_base + morton_to_local(morton);
                                        record_target_write_stats(&mut local_stats, 1, world, world + UVec3::ONE);
                                    }
                                }
                                BrickEntryKind::FillBox { local_min, local_max_exclusive, material } => {
                                    let cell = material_cell(material);
                                    let changed = brick.fill_box(local_min, local_max_exclusive, cell);
                                    if changed > 0 {
                                        record_target_write_stats(
                                            &mut local_stats,
                                            u64::from(changed),
                                            brick_base + local_min,
                                            brick_base + local_max_exclusive,
                                        );
                                    }
                                }
                            }
                        }
                        local_bricks.push((*brick_pos, brick));
                    }
                    (local_bricks, local_stats)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("write replay worker should not panic"))
            .collect()
    });

    if profile {
        eprintln!(
            "teardown_zip_profile replay_apply={:.3}s workers={}",
            apply_start.elapsed().as_secs_f64(),
            apply_workers,
        );
    }

    let mut bricks = Vec::new();
    let mut stats = TargetWriteStats::default();
    stats.out_of_bounds_voxels = op_log.out_of_bounds;
    for (local_bricks, local_stats) in results {
        bricks.extend(local_bricks);
        merge_target_write_stats(&mut stats, &local_stats);
    }
    (bricks, stats)
}

fn material_cell(material: u16) -> VoxelCell {
    if material == 0 {
        VoxelCell::AIR
    } else {
        VoxelCell::new(material, 1, [0; 3])
    }
}

fn morton_to_local(morton: u16) -> UVec3 {
    let (x, y, z) = morton::decode(morton as u32);
    UVec3::new(x, y, z)
}

fn record_target_write_stats(
    stats: &mut TargetWriteStats,
    count: u64,
    min: UVec3,
    max_exclusive: UVec3,
) {
    stats.written_voxels = stats.written_voxels.saturating_add(count);
    stats.target_min = stats.target_min.min(min);
    stats.target_max_exclusive = stats.target_max_exclusive.max(max_exclusive);
}

fn merge_target_write_stats(into: &mut TargetWriteStats, other: &TargetWriteStats) {
    into.written_voxels = into.written_voxels.saturating_add(other.written_voxels);
    if other.written_voxels > 0 {
        into.target_min = into.target_min.min(other.target_min);
        into.target_max_exclusive = into.target_max_exclusive.max(other.target_max_exclusive);
    }
}

/// Brick-slab parallel write: N workers each walk the full document with their own
/// source/cache (no shared mutable state) and write only the bricks whose
/// `brick_z % num_workers == worker_id`. The slab sets are disjoint so the merge is a
/// trivial concatenation. Transform work is replicated N× but runs in parallel, so
/// wall-clock transform ≈ 1×; write work is partitioned ≈ 1/N.
///
/// Only valid when `scale >= 1.0` (the downsampled-plan closure is inert there —
/// it only gate-checks vox-node metadata without writing).
fn parallel_slab_write(
    zip_path: &Path,
    teardown_dir: Option<PathBuf>,
    main_xml: &[u8],
    prefilled_cache: VoxCache, // cloned from bounds pass — workers start with all scenes cached
    ucvh: &Ucvh,
    target_mapping: &TargetMapping,
    density: u32,
    profile: bool,
) -> Result<(Vec<(UVec3, StagedBrick)>, TargetWriteStats), TeardownZipLoadError> {
    let world_size = ucvh.config.world_size;
    let grid_size = ucvh.config.brick_grid_size;
    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        // Cap at the number of brick layers — avoid idle workers on tiny maps.
        .min(grid_size.z as usize)
        .max(1);

    let results: Vec<Result<(Vec<(UVec3, StagedBrick)>, TargetWriteStats), TeardownZipLoadError>> =
        std::thread::scope(|scope| {
            (0..num_workers)
                .map(|worker_id| {
                    let td_dir = teardown_dir.clone();
                    // Clone the pre-filled cache — only Arc pointer clones, no scene data copied.
                    // Workers start with all .vox scenes already parsed → zero zip re-reads.
                    let mut worker_cache = prefilled_cache.clone();
                    scope.spawn(move || -> Result<(Vec<(UVec3, StagedBrick)>, TargetWriteStats), TeardownZipLoadError> {
                        let mut source = TeardownResourceSource::open(zip_path, td_dir)?;
                        // Minimal stats — only voxels_per_unit is consulted by source_density()
                        // during the walk. Other counters are discarded (they were populated by
                        // the bounds pass in the outer function).
                        let mut worker_stats = TeardownZipWriteStats {
                            voxels_per_unit: density,
                            ..Default::default()
                        };
                        let mut staged_bricks = StagedBricks::new(grid_size);
                        let write_stats = TargetWriteStatsTracker::default();
                        // At scale >= 1.0 the downsampled path is never reached — each
                        // worker gets its own empty plan cache to satisfy the type.
                        let mut plan_cache = DownsampledVoxPlanCache::default();
                        let downsampled_stats = RefCell::new(DownsampledVoxStats::default());

                        walk_xml_bytes(
                            main_xml,
                            Mat4::IDENTITY,
                            &mut source,
                            &mut worker_cache,
                            &mut worker_stats,
                            &mut staged_bricks,
                            false,
                            true,
                            &mut |pos, material, bricks| {
                                let Some(target) = target_mapping.map(pos) else {
                                    // Only worker 0 counts OOB to avoid N× inflation.
                                    if worker_id == 0 {
                                        write_stats.record_out_of_bounds(1);
                                    }
                                    return;
                                };
                                if target.x >= world_size.x
                                    || target.y >= world_size.y
                                    || target.z >= world_size.z
                                {
                                    if worker_id == 0 {
                                        write_stats.record_out_of_bounds(1);
                                    }
                                    return;
                                }
                                // Slab filter: only handle bricks that belong to this worker.
                                if (target.z / BRICK_EDGE) as usize % num_workers != worker_id {
                                    return;
                                }
                                if write_target_voxel(ucvh, bricks, target, material) {
                                    write_stats.record_written(target);
                                }
                            },
                            &mut |source_min, source_max_exclusive, material, bricks| {
                                let Some((source_min, source_max_exclusive)) =
                                    target_mapping.clipped_source_box(source_min, source_max_exclusive)
                                else {
                                    return false;
                                };
                                let Some(start) = target_mapping.map(source_min) else {
                                    return false;
                                };
                                let Some(end) = target_mapping.map(source_max_exclusive - IVec3::ONE) else {
                                    return false;
                                };
                                let min = start.min(end);
                                let max_exclusive = (start.max(end) + UVec3::ONE).min(world_size);
                                if min.x >= max_exclusive.x
                                    || min.y >= max_exclusive.y
                                    || min.z >= max_exclusive.z
                                {
                                    return true;
                                }
                                fill_target_box_slab(
                                    ucvh,
                                    bricks,
                                    min,
                                    max_exclusive,
                                    material,
                                    &write_stats,
                                    worker_id,
                                    num_workers,
                                );
                                true
                            },
                            &mut |_path, object, scene, _tint, _matrix, _bricks| {
                                // At scale >= 1.0, downsampled_vox_plan_spec returns None for
                                // every node so this closure only needs to report "handled"
                                // (true) for metadata-None nodes (skips the inline emit path,
                                // which would produce nothing anyway) and "not handled" (false)
                                // for metadata-Some nodes (lets the inline emit path run).
                                scene.metadata(object).is_none()
                            },
                        )?;

                        let _ = &mut plan_cache; // suppress unused warning
                        let _ = downsampled_stats;
                        let ws = write_stats.finish();
                        Ok((std::mem::take(&mut staged_bricks.bricks), ws))
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("slab write worker should not panic"))
                .collect()
        });

    let mut all_bricks: Vec<(UVec3, StagedBrick)> = Vec::new();
    let mut merged = TargetWriteStats::default();
    for result in results {
        let (worker_bricks, worker_stats) = result?;
        all_bricks.extend(worker_bricks);
        merge_target_write_stats(&mut merged, &worker_stats);
        merged.out_of_bounds_voxels = merged
            .out_of_bounds_voxels
            .saturating_add(worker_stats.out_of_bounds_voxels);
    }
    if profile {
        eprintln!(
            "teardown_zip_profile slab_workers={} bricks={}",
            num_workers,
            all_bricks.len()
        );
    }
    Ok((all_bricks, merged))
}

/// Like `fill_target_box` but skips bricks whose `brick_z % num_workers != worker_id`,
/// so each worker only fills its own disjoint slab region.
fn fill_target_box_slab(
    ucvh: &Ucvh,
    bricks: &mut StagedBricks,
    min: UVec3,
    max_exclusive: UVec3,
    material: u16,
    stats: &TargetWriteStatsTracker,
    worker_id: usize,
    num_workers: usize,
) {
    if min.x >= max_exclusive.x || min.y >= max_exclusive.y || min.z >= max_exclusive.z {
        return;
    }
    let cell = if material == 0 {
        VoxelCell::AIR
    } else {
        VoxelCell::new(material, 1, [0; 3])
    };
    let brick_min = min / BRICK_EDGE;
    let brick_max = (max_exclusive - UVec3::ONE) / BRICK_EDGE;
    for bz in brick_min.z..=brick_max.z {
        if bz as usize % num_workers != worker_id {
            continue; // skip bricks not in this worker's slab
        }
        for by in brick_min.y..=brick_max.y {
            for bx in brick_min.x..=brick_max.x {
                let brick_pos = UVec3::new(bx, by, bz);
                let brick_base = brick_pos * BRICK_EDGE;
                let local_min = min.saturating_sub(brick_base);
                let local_max_exclusive =
                    (max_exclusive - brick_base).min(UVec3::splat(BRICK_EDGE));
                if cell.is_air()
                    && !bricks.contains(brick_pos)
                    && ucvh.brick_id_at(brick_pos).is_none()
                {
                    continue;
                }
                let brick = bricks.get_or_insert_with(ucvh, brick_pos);
                let changed = brick.fill_box(local_min, local_max_exclusive, cell);
                if changed > 0 {
                    stats.record_count_bounds(
                        u64::from(changed),
                        brick_base + local_min,
                        brick_base + local_max_exclusive,
                    );
                }
            }
        }
    }
}

fn fill_target_box(
    ucvh: &Ucvh,
    bricks: &mut StagedBricks,
    min: UVec3,
    max_exclusive: UVec3,
    material: u16,
    stats: &TargetWriteStatsTracker,
) {
    if min.x >= max_exclusive.x || min.y >= max_exclusive.y || min.z >= max_exclusive.z {
        return;
    }
    let cell = if material == 0 {
        VoxelCell::AIR
    } else {
        VoxelCell::new(material, 1, [0; 3])
    };
    let brick_min = min / BRICK_EDGE;
    let brick_max = (max_exclusive - UVec3::ONE) / BRICK_EDGE;
    for bz in brick_min.z..=brick_max.z {
        for by in brick_min.y..=brick_max.y {
            for bx in brick_min.x..=brick_max.x {
                let brick_pos = UVec3::new(bx, by, bz);
                let brick_base = brick_pos * BRICK_EDGE;
                let local_min = min.saturating_sub(brick_base);
                let local_max_exclusive =
                    (max_exclusive - brick_base).min(UVec3::splat(BRICK_EDGE));
                if cell.is_air()
                    && !bricks.contains(brick_pos)
                    && ucvh.brick_id_at(brick_pos).is_none()
                {
                    continue;
                }
                let brick = bricks.get_or_insert_with(ucvh, brick_pos);
                let changed = brick.fill_box(local_min, local_max_exclusive, cell);
                if changed > 0 {
                    stats.record_count_bounds(
                        u64::from(changed),
                        brick_base + local_min,
                        brick_base + local_max_exclusive,
                    );
                }
            }
        }
    }
}

struct StagedBricks {
    grid_size: UVec3,
    indices: U64IndexMap<usize>,
    bricks: Vec<(UVec3, StagedBrick)>,
    last_lookup_key: Option<u64>,
    last_lookup_index: usize,
}

impl StagedBricks {
    fn new(grid_size: UVec3) -> Self {
        Self {
            grid_size,
            indices: new_u64_index_map(),
            bricks: Vec::new(),
            last_lookup_key: None,
            last_lookup_index: 0,
        }
    }

    fn get_or_insert_with(&mut self, ucvh: &Ucvh, brick_pos: UVec3) -> &mut StagedBrick {
        let key = self.key(brick_pos);
        let brick_index = if self.last_lookup_key == Some(key) {
            self.last_lookup_index
        } else {
            let brick_index = match self.indices.get(&key).copied() {
                Some(brick_index) => brick_index,
                None => {
                    let brick_index = self.bricks.len();
                    self.indices.insert(key, brick_index);
                    self.bricks
                        .push((brick_pos, StagedBrick::new_seeded(ucvh, brick_pos)));
                    brick_index
                }
            };
            self.last_lookup_key = Some(key);
            self.last_lookup_index = brick_index;
            brick_index
        };
        &mut self.bricks[brick_index].1
    }

    fn iter(&self) -> impl Iterator<Item = (UVec3, &StagedBrick)> {
        self.bricks
            .iter()
            .map(|(brick_pos, brick)| (*brick_pos, brick))
    }

    fn contains(&self, brick_pos: UVec3) -> bool {
        self.indices.contains_key(&self.key(brick_pos))
    }

    fn key(&self, brick_pos: UVec3) -> u64 {
        u64::from(
            brick_pos.x
                + brick_pos.y * self.grid_size.x
                + brick_pos.z * self.grid_size.x * self.grid_size.y,
        )
    }
}

struct StagedBrick {
    data: BrickData,
    touched: [bool; BRICK_VOLUME],
    touched_count: u32,
}

impl StagedBrick {
    fn new_seeded(ucvh: &Ucvh, brick_pos: UVec3) -> Self {
        let mut data = BrickData::new();
        if ucvh.brick_id_at(brick_pos).is_some() {
            let base = brick_pos * BRICK_EDGE;
            for z in 0..BRICK_EDGE {
                for y in 0..BRICK_EDGE {
                    for x in 0..BRICK_EDGE {
                        let world = base + UVec3::new(x, y, z);
                        data.set_voxel(x, y, z, ucvh.get_voxel(world));
                    }
                }
            }
        }
        Self {
            data,
            touched: [false; BRICK_VOLUME],
            touched_count: 0,
        }
    }

    fn write(&mut self, local_pos: UVec3, cell: VoxelCell) -> bool {
        let morton = morton::encode(local_pos.x, local_pos.y, local_pos.z) as usize;
        self.write_morton(morton, cell)
    }

    fn write_morton(&mut self, morton: usize, cell: VoxelCell) -> bool {
        if self.data.materials[morton] == cell {
            return false;
        }
        if !self.touched[morton] {
            self.touched[morton] = true;
            self.touched_count += 1;
        }
        self.data.materials[morton] = cell;
        let word = morton / 32;
        let bit = 1u32 << (morton % 32);
        let occupied = self.data.occupancy.bits[word] & bit != 0;
        if cell.is_air() {
            if occupied {
                self.data.occupancy.bits[word] &= !bit;
                self.data.occupancy.count -= 1;
            }
        } else if !occupied {
            self.data.occupancy.bits[word] |= bit;
            self.data.occupancy.count += 1;
        }
        true
    }

    fn fill_box(&mut self, min: UVec3, max_exclusive: UVec3, cell: VoxelCell) -> u32 {
        if min == UVec3::ZERO && max_exclusive == UVec3::splat(BRICK_EDGE) {
            let mut changed = 0;
            for index in 0..BRICK_VOLUME {
                if self.data.materials[index] != cell {
                    changed += 1;
                    if !self.touched[index] {
                        self.touched[index] = true;
                        self.touched_count += 1;
                    }
                }
            }
            if changed == 0 {
                return 0;
            }
            self.data.materials.fill(cell);
            self.data.occupancy = if cell.is_air() {
                BrickOccupancy {
                    bits: [0; 16],
                    count: 0,
                    _pad: [0; 3],
                }
            } else {
                BrickOccupancy {
                    bits: [u32::MAX; 16],
                    count: BRICK_VOLUME as u32,
                    _pad: [0; 3],
                }
            };
            return changed;
        }
        let mut changed = 0;
        for z in min.z..max_exclusive.z {
            for y in min.y..max_exclusive.y {
                for x in min.x..max_exclusive.x {
                    if self.write(UVec3::new(x, y, z), cell) {
                        changed += 1;
                    }
                }
            }
        }
        changed
    }
}

fn find_teardown_dir(explicit: Option<PathBuf>) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(explicit) = explicit {
        candidates.push(explicit);
    }
    if let Ok(env_dir) = std::env::var("TEARDOWN_DIR") {
        candidates.push(PathBuf::from(env_dir));
    }
    candidates.extend(DEFAULT_TEARDOWN_DIR_CANDIDATES.iter().map(PathBuf::from));
    candidates
        .into_iter()
        .find(|path| path.join("data").join("built-in").is_dir())
}

fn safe_join(base: &Path, rel: &str) -> Option<PathBuf> {
    let mut out = base.to_path_buf();
    for component in Path::new(rel).components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => return None,
        }
    }
    Some(out)
}

fn normalize_resource_path(path: &str) -> String {
    let mut normalized = path.replace('\\', "/");
    while normalized.contains("//") {
        normalized = normalized.replace("//", "/");
    }
    normalized
}

fn teardown_data_relative_path(path: &str) -> &str {
    let mut path = path;
    while let Some(rest) = path.strip_prefix("./") {
        path = rest;
    }
    while let Some(rest) = path.strip_prefix("../") {
        path = rest;
    }
    path
}

fn teardown_data_level_root(path: &str) -> Option<&str> {
    let path = teardown_data_relative_path(path);
    let first_slash = path.find('/')?;
    if !path[..first_slash].eq_ignore_ascii_case("level") {
        return None;
    }
    let rest = &path[first_slash + 1..];
    let second_slash = rest.find('/')?;
    Some(&path[..first_slash + 1 + second_slash])
}

fn zip_resource_name(root: &str, rel: &str) -> String {
    if root.is_empty() {
        rel.to_owned()
    } else {
        format!("{root}/{rel}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::ucvh::{Ucvh, UcvhConfig};
    use glam::UVec3;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use zip::CompressionMethod;
    use zip::write::SimpleFileOptions;

    /// The integer transform fast path must be bit-identical to the float `Point` path
    /// for every node it accepts. When `integer_affine` returns `Some`, applying it must
    /// equal `round_world_point(matrix.transform_point3(native_point(c)))` for all coords.
    fn assert_integer_affine_matches_float(matrix: Mat4, vpu: u32, coords: &[IVec3]) {
        let Some(affine) = integer_affine(matrix, vpu) else {
            return; // node rejected → float path used, nothing to compare
        };
        for &c in coords {
            let float_target = round_world_point(
                matrix.transform_point3(native_point(c.x, c.y, c.z, vpu)),
                vpu,
            );
            assert_eq!(
                affine.apply(c),
                float_target,
                "integer affine diverged from float path at {c:?} (matrix={matrix:?})"
            );
        }
    }

    fn integer_affine_test_coords() -> Vec<IVec3> {
        let mut coords = Vec::new();
        for &v in &[0, 1, 2, 7, 8, 15, 63, 100, 511, 1000, 4095, 30000, -1, -100, -4096] {
            coords.push(IVec3::new(v, v, v));
            coords.push(IVec3::new(v, 0, 0));
            coords.push(IVec3::new(0, v, 0));
            coords.push(IVec3::new(0, 0, v));
        }
        coords.push(IVec3::new(1234, -5678, 9012));
        coords
    }

    #[test]
    fn integer_affine_bit_identical_for_integer_translation() {
        let coords = integer_affine_test_coords();
        for pos in [
            Vec3::ZERO,
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(-7.0, 12.0, 41.0),
            Vec3::new(100.0, 200.0, 300.0),
        ] {
            let matrix = Mat4::from_translation(pos);
            assert!(
                integer_affine(matrix, TEARDOWN_NATIVE_VOXELS_PER_UNIT).is_some(),
                "pure integer translation {pos:?} should take the integer fast path"
            );
            assert_integer_affine_matches_float(matrix, TEARDOWN_NATIVE_VOXELS_PER_UNIT, &coords);
        }
    }

    #[test]
    fn integer_affine_bit_identical_for_axis_aligned_rotations() {
        let coords = integer_affine_test_coords();
        for rot in [
            Vec3::new(0.0, 90.0, 0.0),
            Vec3::new(0.0, 180.0, 0.0),
            Vec3::new(0.0, 270.0, 0.0),
            Vec3::new(90.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 90.0),
            Vec3::new(90.0, 90.0, 0.0),
        ] {
            let matrix = Mat4::from_translation(Vec3::new(5.0, -3.0, 8.0))
                * teardown_euler_rotation(rot);
            assert!(
                integer_affine(matrix, TEARDOWN_NATIVE_VOXELS_PER_UNIT).is_some(),
                "axis-aligned rotation {rot:?} should take the integer fast path"
            );
            assert_integer_affine_matches_float(matrix, TEARDOWN_NATIVE_VOXELS_PER_UNIT, &coords);
        }
    }

    #[test]
    fn integer_affine_rejects_non_integer_transforms() {
        // Fractional scale, non-axis rotation, and half-voxel translation must all be
        // rejected so they fall back to the exact float path.
        let vpu = TEARDOWN_NATIVE_VOXELS_PER_UNIT;
        assert!(integer_affine(Mat4::from_scale(Vec3::splat(1.5)), vpu).is_none());
        assert!(
            integer_affine(teardown_euler_rotation(Vec3::new(0.0, 45.0, 0.0)), vpu).is_none()
        );
        // Translation of 0.05 world units = 0.5 native voxels → half-integer, must reject.
        assert!(integer_affine(Mat4::from_translation(Vec3::new(0.05, 0.0, 0.0)), vpu).is_none());
    }

    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "revolumetric_teardown_zip_loader_{name}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test dir should be creatable");
        dir
    }

    fn chunk(id: &[u8; 4], content: &[u8], children: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(id);
        bytes.extend_from_slice(&(content.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(children.len() as u32).to_le_bytes());
        bytes.extend_from_slice(content);
        bytes.extend_from_slice(children);
        bytes
    }

    fn single_voxel_vox(color: [u8; 4]) -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        children.extend(chunk(b"XYZI", &[1, 0, 0, 0, 0, 0, 0, 1], &[]));
        let mut palette = Vec::new();
        palette.extend_from_slice(&color);
        for _ in 1..256 {
            palette.extend_from_slice(&[0, 0, 0, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    #[test]
    fn default_zip_map_path_selects_first_existing_candidate() {
        let temp = test_dir("default_zip_candidate");
        let missing = temp.join("missing.zip");
        let existing = temp.join("Vintessa Hills.zip");
        std::fs::write(&existing, b"zip placeholder").expect("candidate zip should be writable");

        let candidates = [missing.as_path(), existing.as_path()];
        let selected = select_default_zip_map_path(&candidates);

        assert_eq!(selected, existing);
    }

    #[test]
    fn default_zip_map_path_falls_back_to_first_candidate_when_none_exist() {
        let temp = test_dir("default_zip_fallback");
        let first = temp.join("first.zip");
        let second = temp.join("second.zip");

        let candidates = [first.as_path(), second.as_path()];
        let selected = select_default_zip_map_path(&candidates);

        assert_eq!(selected, first);
    }

    fn line_vox(count: u8) -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(
            b"SIZE",
            &[count, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            &[],
        ));
        let mut xyzi = Vec::new();
        xyzi.extend_from_slice(&(count as u32).to_le_bytes());
        for x in 0..count {
            xyzi.extend_from_slice(&[x, 0, 0, 1 + x % 3]);
        }
        children.extend(chunk(b"XYZI", &xyzi, &[]));
        let mut palette = Vec::new();
        for i in 0..256u16 {
            palette.extend_from_slice(&[i as u8, 64, 180, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    fn two_color_line_vox(left: [u8; 4], right: [u8; 4]) -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        children.extend(chunk(b"XYZI", &[2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2], &[]));
        let mut palette = Vec::new();
        palette.extend_from_slice(&left);
        palette.extend_from_slice(&right);
        for _ in 2..256 {
            palette.extend_from_slice(&[0, 0, 0, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    fn dict(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (key, value) in entries {
            bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        bytes
    }

    fn transform_node_with_name(node_id: u32, child_id: u32, name: &str) -> Vec<u8> {
        transform_node_with_name_and_frame(node_id, child_id, name, &[])
    }

    fn transform_node_with_name_and_frame(
        node_id: u32,
        child_id: u32,
        name: &str,
        frame_entries: &[(&str, &str)],
    ) -> Vec<u8> {
        transform_node_with_name_attrs_and_frame(
            node_id,
            child_id,
            &[("_name", name)],
            frame_entries,
        )
    }

    fn transform_node_with_name_attrs_and_frame(
        node_id: u32,
        child_id: u32,
        node_entries: &[(&str, &str)],
        frame_entries: &[(&str, &str)],
    ) -> Vec<u8> {
        let mut transform = Vec::new();
        transform.extend_from_slice(&node_id.to_le_bytes());
        transform.extend(dict(node_entries));
        transform.extend_from_slice(&(child_id as i32).to_le_bytes());
        transform.extend_from_slice(&(-1i32).to_le_bytes());
        transform.extend_from_slice(&(-1i32).to_le_bytes());
        transform.extend_from_slice(&1u32.to_le_bytes());
        transform.extend(dict(frame_entries));
        chunk(b"nTRN", &transform, &[])
    }

    fn shape_node(node_id: u32, model_id: u32) -> Vec<u8> {
        let mut shape = Vec::new();
        shape.extend_from_slice(&node_id.to_le_bytes());
        shape.extend(dict(&[]));
        shape.extend_from_slice(&1u32.to_le_bytes());
        shape.extend_from_slice(&model_id.to_le_bytes());
        shape.extend(dict(&[]));
        chunk(b"nSHP", &shape, &[])
    }

    fn named_two_shape_vox() -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"PACK", &2u32.to_le_bytes(), &[]));
        for (x, color) in [(0u8, 1u8), (10, 2)] {
            children.extend(chunk(b"SIZE", &[16, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
            let xyzi = [1, 0, 0, 0, x, 0, 0, color];
            children.extend(chunk(b"XYZI", &xyzi, &[]));
        }
        children.extend(transform_node_with_name(0, 2, "wanted"));
        children.extend(transform_node_with_name(1, 3, "other"));
        children.extend(shape_node(2, 0));
        children.extend(shape_node(3, 1));
        let mut palette = Vec::new();
        palette.extend_from_slice(&[10, 20, 30, 255]);
        palette.extend_from_slice(&[200, 210, 220, 255]);
        for _ in 2..256 {
            palette.extend_from_slice(&[0, 0, 0, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    fn hidden_named_object_vox() -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"PACK", &1u32.to_le_bytes(), &[]));
        children.extend(chunk(b"SIZE", &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        children.extend(chunk(b"XYZI", &[1, 0, 0, 0, 0, 0, 0, 1], &[]));
        children.extend(transform_node_with_name_attrs_and_frame(
            0,
            1,
            &[("_name", "part"), ("_hidden", "1")],
            &[],
        ));
        children.extend(shape_node(1, 0));
        let mut palette = Vec::new();
        palette.extend_from_slice(&[10, 20, 30, 255]);
        for _ in 1..256 {
            palette.extend_from_slice(&[0, 0, 0, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    fn sparse_brush_vox() -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        children.extend(chunk(b"XYZI", &[1, 0, 0, 0, 1, 0, 0, 1], &[]));
        let mut palette = Vec::new();
        palette.extend_from_slice(&[30, 180, 70, 255]);
        for _ in 1..256 {
            palette.extend_from_slice(&[0, 0, 0, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    fn test_rgba_png(width: u32, height: u32, pixels: &[[u8; 4]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("png header");
            let data: Vec<_> = pixels
                .iter()
                .flat_map(|pixel| pixel.iter().copied())
                .collect();
            writer.write_image_data(&data).expect("png data");
        }
        bytes
    }

    fn translated_named_two_shape_vox() -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"PACK", &2u32.to_le_bytes(), &[]));
        for color in [1u8, 2] {
            children.extend(chunk(b"SIZE", &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
            let xyzi = [1, 0, 0, 0, 0, 0, 0, color];
            children.extend(chunk(b"XYZI", &xyzi, &[]));
        }
        children.extend(transform_node_with_name_and_frame(0, 2, "trunk", &[]));
        children.extend(transform_node_with_name_and_frame(
            1,
            3,
            "crown",
            &[("_t", "0 0 10")],
        ));
        children.extend(shape_node(2, 0));
        children.extend(shape_node(3, 1));
        let mut palette = Vec::new();
        palette.extend_from_slice(&[10, 20, 30, 255]);
        palette.extend_from_slice(&[200, 210, 220, 255]);
        for _ in 2..256 {
            palette.extend_from_slice(&[0, 0, 0, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    fn write_workshop_zip(path: &Path, files: &[(&str, Vec<u8>)]) {
        let file = File::create(path).expect("test zip should be creatable");
        let mut zip = zip::ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        for (name, content) in files {
            zip.start_file(format!("root/{name}"), options)
                .expect("zip entry should start");
            zip.write_all(content)
                .expect("zip entry should be writable");
        }
        zip.finish().expect("zip should finish");
    }

    fn write_flat_workshop_zip(path: &Path, files: &[(&str, Vec<u8>)]) {
        let file = File::create(path).expect("test zip should be creatable");
        let mut zip = zip::ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        for (name, content) in files {
            zip.start_file(*name, options)
                .expect("zip entry should start");
            zip.write_all(content)
                .expect("zip entry should be writable");
        }
        zip.finish().expect("zip should finish");
    }

    #[test]
    fn native_offset_missing_components_default_to_zero() {
        assert_eq!(parse_native_offset(Some("4")), IVec3::new(4, 0, 0));
        assert_eq!(parse_native_offset(Some("50 1")), IVec3::new(50, 1, 0));
        assert_eq!(parse_native_offset(Some("-2 3 7")), IVec3::new(-2, 3, 7));
    }

    #[test]
    fn color_parser_keeps_numeric_prefix_before_garbled_suffix() {
        assert_eq!(
            parse_color(Some("0.22 0.27 0.21ű"), [155, 155, 150, 255]),
            [56, 69, 54, 255]
        );
    }

    #[test]
    fn vec3_parser_keeps_numeric_prefix_before_editor_garbage() {
        assert_eq!(
            parse_vec3(Some("4' 23 2"), Vec3::ZERO),
            Vec3::new(4.0, 23.0, 2.0)
        );
        assert_eq!(parse_vec3(Some("3u"), Vec3::ONE), Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn vec3_parser_does_not_shift_axes_after_invalid_placeholders() {
        assert_eq!(
            parse_vec3(Some("-90.0 90.0 -"), Vec3::ZERO),
            Vec3::new(-90.0, 90.0, 0.0)
        );
        assert_eq!(
            parse_vec3(Some("- 180"), Vec3::ZERO),
            Vec3::new(0.0, 180.0, 0.0)
        );
    }

    #[test]
    fn arbitrary_rotation_expands_density_cells_to_preserve_coverage() {
        assert!(transform_expands_density_cells(Mat4::from_rotation_y(
            45.0_f32.to_radians()
        )));
        assert!(!transform_expands_density_cells(Mat4::from_rotation_y(
            90.0_f32.to_radians()
        )));
    }

    #[test]
    fn arbitrary_rotation_rasterizes_density_cell_as_oriented_volume() {
        let matrix = Mat4::from_rotation_y(45.0_f32.to_radians());
        let mut emitted = HashSet::new();

        emit_transformed_density_cell(
            IVec3::ZERO,
            matrix,
            TEARDOWN_NATIVE_VOXELS_PER_UNIT,
            material_for_color([255, 0, 0, 255]),
            density_cell_emission(matrix, TEARDOWN_NATIVE_VOXELS_PER_UNIT),
            &mut |pos, _material, seen: &mut HashSet<IVec3>| {
                seen.insert(pos);
            },
            &mut emitted,
        );

        assert_eq!(
            emitted,
            HashSet::from([IVec3::new(0, 0, -1), IVec3::new(0, 0, 0)]),
            "rotated native voxels should be rasterized by their oriented volume, not by filling the full world AABB"
        );
    }

    #[test]
    fn staged_voxel_write_skips_same_cell_duplicates() {
        let ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));
        let mut bricks = StagedBricks::new(ucvh.config.brick_grid_size);
        let target = UVec3::new(4, 4, 4);
        let red = material_for_color([255, 0, 0, 255]);
        let green = material_for_color([0, 255, 0, 255]);

        assert!(write_target_voxel(&ucvh, &mut bricks, target, red));
        assert!(!write_target_voxel(&ucvh, &mut bricks, target, red));
        assert!(write_target_voxel(&ucvh, &mut bricks, target, green));
        assert!(write_target_voxel(&ucvh, &mut bricks, target, 0));
        assert!(!write_target_voxel(
            &ucvh,
            &mut bricks,
            UVec3::new(8, 8, 8),
            0
        ));
    }

    #[test]
    fn fill_target_box_skips_air_over_already_empty_bricks() {
        let ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        let mut bricks = StagedBricks::new(ucvh.config.brick_grid_size);
        let stats = TargetWriteStatsTracker::default();

        fill_target_box(
            &ucvh,
            &mut bricks,
            UVec3::ZERO,
            UVec3::splat(BRICK_EDGE),
            0,
            &stats,
        );

        let stats = stats.finish();
        assert_eq!(stats.written_voxels, 0);
        assert_eq!(bricks.iter().count(), 0);
    }

    #[test]
    fn transformed_solid_voxbox_emits_oriented_box_without_per_cell_expansion() {
        let counts = IVec3::new(160, 2, 80);
        let (local_min, local_max_exclusive) = voxbox_local_bounds(counts);
        let matrix = Mat4::from_rotation_y(33.0_f32.to_radians());
        let mut emitted = HashSet::new();

        let visited = emit_solid_voxbox_points(
            local_min,
            local_max_exclusive,
            matrix,
            TEARDOWN_NATIVE_VOXELS_PER_UNIT,
            material_for_color([255, 0, 0, 255]),
            &mut |pos, _material, seen: &mut HashSet<IVec3>| {
                seen.insert(pos);
            },
            &mut emitted,
        );

        assert!(visited > 0);
        assert_eq!(visited as usize, emitted.len());
        assert!(visited < (counts.x * counts.y * counts.z * 3) as u64);
    }

    #[test]
    fn quarter_turn_voxbox_uses_axis_aligned_box_fast_path() {
        let bounds = axis_aligned_native_box(
            Mat4::from_rotation_z(90.0_f32.to_radians()),
            IVec3::ZERO,
            IVec3::new(10, 2, 3),
            TEARDOWN_NATIVE_VOXELS_PER_UNIT,
        )
        .expect("quarter-turn boxes remain world-axis aligned");

        assert_eq!(bounds.0, IVec3::new(-2, 0, 0));
        assert_eq!(bounds.1, IVec3::new(0, 10, 3));
    }

    #[test]
    fn full_brick_box_fill_uses_bulk_material_path() {
        let source = crate::render::source_checks::read_source("src/voxel/teardown_zip_loader.rs");
        let body = source
            .split("fn fill_box(&mut self")
            .nth(1)
            .expect("StagedBrick::fill_box should exist")
            .split("fn find_teardown_dir")
            .next()
            .expect("fill_box body should end before teardown dir helpers");

        assert!(
            body.contains("self.data.materials.fill(cell)"),
            "full-brick fill should bulk-fill materials instead of routing every voxel through write_morton"
        );
    }

    #[test]
    fn loads_mod_vox_node_from_workshop_zip_into_ucvh() {
        let temp = test_dir("mod_vox");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/block.vox"/></scene>"#.to_vec(),
                ),
                ("vox/block.vox", single_voxel_vox([220, 20, 30, 255])),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("teardown zip should load");

        assert_eq!(stats.vox_nodes_exported, 1);
        assert_eq!(stats.unique_written_voxels, 1);
        assert_ne!(ucvh.get_voxel(UVec3::splat(4)).material, 0);
    }

    #[test]
    fn loads_flat_workshop_zip_without_outer_folder() {
        let temp = test_dir("flat_zip");
        let zip_path = temp.join("map.zip");
        write_flat_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/block.vox"/></scene>"#.to_vec(),
                ),
                ("vox/block.vox", single_voxel_vox([220, 20, 30, 255])),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("flat teardown zip should load");

        assert_eq!(stats.vox_nodes_exported, 1);
        assert!(stats.missing_mod_refs.is_empty());
        assert_eq!(stats.unique_written_voxels, 1);
    }

    #[test]
    fn vox_node_scale_attribute_scales_loaded_geometry() {
        let temp = test_dir("scaled_vox");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox scale="2" file="MOD/vox/block.vox"/></scene>"#.to_vec(),
                ),
                ("vox/block.vox", single_voxel_vox([220, 20, 30, 255])),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("scaled vox should load");

        assert_eq!(stats.vox_nodes_exported, 1);
        assert!(stats.unique_written_voxels > 1);
    }

    #[test]
    fn scaled_voxbox_expands_source_bounds_before_target_mapping() {
        let temp = test_dir("scaled_voxbox_bounds");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxbox scale="40" size="1 1 1" color="1 0 0"/></scene>"#.to_vec(),
            )],
        );
        let world_size = UVec3::splat(32);
        let mut ucvh = Ucvh::new(UcvhConfig::new(world_size));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("scaled voxbox should load");

        assert_eq!(stats.voxbox_nodes_exported, 1);
        assert_eq!(stats.out_of_bounds_voxels, 0);
        let bounds = stats
            .target_bounds
            .expect("scaled voxbox should write bounds");
        assert!(bounds.max_exclusive.cmple(world_size).all());
    }

    #[test]
    fn vox_node_mirrorx_flips_loaded_geometry() {
        let temp = test_dir("vox_mirrorx");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/line.vox" mirrorx="true"/></scene>"#.to_vec(),
                ),
                (
                    "vox/line.vox",
                    two_color_line_vox([220, 20, 30, 255], [20, 200, 240, 255]),
                ),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("mirrored vox should load");

        assert_eq!(stats.vox_nodes_exported, 1);
        assert_eq!(stats.unique_written_voxels, 2);
        assert_eq!(
            ucvh.get_voxel(UVec3::new(4, 4, 4)).material,
            material_for_color([20, 200, 240, 255])
        );
        assert_eq!(
            ucvh.get_voxel(UVec3::new(5, 4, 4)).material,
            material_for_color([220, 20, 30, 255])
        );
    }

    #[test]
    fn expands_builtin_prefab_instance_from_teardown_install() {
        let temp = test_dir("builtin_prefab");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><instance pos="1 0 0" file="BUILT-IN/prefab/prop/crate.xml"/></scene>"#
                    .to_vec(),
            )],
        );
        let install = temp.join("Teardown");
        let prefab = install
            .join("data")
            .join("built-in")
            .join("prefab")
            .join("prop")
            .join("crate.xml");
        std::fs::create_dir_all(prefab.parent().unwrap()).expect("prefab dir");
        std::fs::write(
            &prefab,
            br#"<prefab><group pos="1 0 0"><vox file="BUILT-IN/vox/prop/crate.vox"/></group></prefab>"#,
        )
        .expect("prefab xml");
        let builtin_vox = install
            .join("data")
            .join("built-in")
            .join("vox")
            .join("prop")
            .join("crate.vox");
        std::fs::create_dir_all(builtin_vox.parent().unwrap()).expect("vox dir");
        std::fs::write(&builtin_vox, single_voxel_vox([80, 180, 90, 255])).expect("builtin vox");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats = load_teardown_zip_into_ucvh(
            &zip_path,
            &mut ucvh,
            TeardownZipLoadOptions {
                teardown_dir: Some(install),
                ..Default::default()
            },
        )
        .expect("teardown zip should load");

        assert_eq!(stats.instance_nodes_exported, 1);
        assert_eq!(stats.vox_nodes_exported, 1);
        assert!(stats.missing_builtin_refs.is_empty());
        assert_eq!(stats.unique_written_voxels, 1);
    }

    #[test]
    fn builtin_prefab_resolves_bare_vox_paths_from_teardown_data_vox() {
        let temp = test_dir("builtin_prefab_bare_vox");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><instance file="BUILT-IN/prefab/tool/booster.xml"/></scene>"#.to_vec(),
            )],
        );
        let install = temp.join("Teardown");
        let prefab = install
            .join("data")
            .join("built-in")
            .join("prefab")
            .join("tool")
            .join("booster.xml");
        std::fs::create_dir_all(prefab.parent().unwrap()).expect("prefab dir");
        std::fs::write(
            &prefab,
            br#"<prefab><vox file="tool/booster.vox"/></prefab>"#,
        )
        .expect("prefab xml");
        let bare_vox = install
            .join("data")
            .join("vox")
            .join("tool")
            .join("booster.vox");
        std::fs::create_dir_all(bare_vox.parent().unwrap()).expect("vox dir");
        std::fs::write(&bare_vox, single_voxel_vox([80, 180, 90, 255])).expect("bare vox");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats = load_teardown_zip_into_ucvh(
            &zip_path,
            &mut ucvh,
            TeardownZipLoadOptions {
                teardown_dir: Some(install),
                ..Default::default()
            },
        )
        .expect("teardown zip should load");

        assert_eq!(stats.instance_nodes_exported, 1);
        assert_eq!(stats.vox_nodes_exported, 1);
        assert!(stats.missing_mod_refs.is_empty());
        assert!(stats.missing_builtin_refs.is_empty());
        assert_eq!(stats.unique_written_voxels, 1);
    }

    #[test]
    fn builtin_prefab_resolves_duplicate_slash_builtin_vox_paths() {
        let temp = test_dir("builtin_prefab_duplicate_slash");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><instance file="BUILT-IN/prefab/prop/container.xml"/></scene>"#.to_vec(),
            )],
        );
        let install = temp.join("Teardown");
        let prefab = install
            .join("data")
            .join("built-in")
            .join("prefab")
            .join("prop")
            .join("container.xml");
        std::fs::create_dir_all(prefab.parent().unwrap()).expect("prefab dir");
        std::fs::write(
            &prefab,
            br#"<prefab><vox file="BUILT-IN//vox/prop/container.vox"/></prefab>"#,
        )
        .expect("prefab xml");
        let builtin_vox = install
            .join("data")
            .join("built-in")
            .join("vox")
            .join("prop")
            .join("container.vox");
        std::fs::create_dir_all(builtin_vox.parent().unwrap()).expect("vox dir");
        std::fs::write(&builtin_vox, single_voxel_vox([80, 180, 90, 255])).expect("builtin vox");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats = load_teardown_zip_into_ucvh(
            &zip_path,
            &mut ucvh,
            TeardownZipLoadOptions {
                teardown_dir: Some(install),
                ..Default::default()
            },
        )
        .expect("teardown zip should load");

        assert_eq!(stats.vox_nodes_exported, 1);
        assert!(stats.missing_builtin_refs.is_empty());
        assert_eq!(stats.unique_written_voxels, 1);
    }

    #[test]
    fn level_paths_resolve_from_workshop_level_root() {
        let temp = test_dir("level_paths");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><instance file="LEVEL/prefab/camera.xml"/></scene>"#.to_vec(),
                ),
                (
                    "prefab/camera.xml",
                    br#"<prefab><vox file="LEVEL/vox/camera.vox"/></prefab>"#.to_vec(),
                ),
                ("vox/camera.vox", single_voxel_vox([80, 180, 90, 255])),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("level paths should load");

        assert_eq!(stats.instance_nodes_exported, 1);
        assert_eq!(stats.vox_nodes_exported, 1);
        assert!(stats.missing_mod_refs.is_empty());
        assert!(stats.unique_written_voxels > 0);
    }

    #[test]
    fn level_paths_inside_teardown_data_level_resolve_from_level_folder() {
        let temp = test_dir("teardown_data_level_paths");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><instance file="level/factory/prefab/camera.xml"/></scene>"#.to_vec(),
            )],
        );
        let install = temp.join("Teardown");
        let prefab = install
            .join("data")
            .join("level")
            .join("factory")
            .join("prefab")
            .join("camera.xml");
        std::fs::create_dir_all(prefab.parent().unwrap()).expect("prefab dir");
        std::fs::write(
            &prefab,
            br#"<prefab><vox file="LEVEL/vox/camera.vox"/></prefab>"#,
        )
        .expect("prefab xml");
        let vox = install
            .join("data")
            .join("level")
            .join("factory")
            .join("vox")
            .join("camera.vox");
        std::fs::create_dir_all(vox.parent().unwrap()).expect("vox dir");
        std::fs::write(&vox, single_voxel_vox([80, 180, 90, 255])).expect("level vox");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats = load_teardown_zip_into_ucvh(
            &zip_path,
            &mut ucvh,
            TeardownZipLoadOptions {
                teardown_dir: Some(install),
                ..Default::default()
            },
        )
        .expect("teardown data level paths should load");

        assert_eq!(stats.instance_nodes_exported, 1);
        assert_eq!(stats.vox_nodes_exported, 1);
        assert!(stats.missing_mod_refs.is_empty());
        assert!(stats.unique_written_voxels > 0);
    }

    #[test]
    fn rasterizes_voxbox_as_native_teardown_voxels() {
        let temp = test_dir("voxbox");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxbox size="2 1 1" color="1 0 0"/></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("teardown zip should load");

        assert_eq!(stats.voxbox_nodes_exported, 1);
        assert_eq!(stats.input_voxels, 2);
        assert_eq!(stats.unique_written_voxels, 2);
    }

    #[test]
    fn load_fails_when_staged_bricks_exceed_ucvh_capacity() {
        let temp = test_dir("capacity_exceeded");
        let zip_path = temp.join("map.zip");
        let mut xml = String::from("<scene>");
        for x in 0..65 {
            xml.push_str(&format!(
                r#"<voxbox pos="{x} 0 0" size="1 1 1" color="1 0 0"/>"#
            ));
        }
        xml.push_str("</scene>");
        write_workshop_zip(&zip_path, &[("main.xml", xml.into_bytes())]);
        let mut ucvh = Ucvh::new(UcvhConfig::with_brick_capacity(UVec3::new(520, 8, 8), 64));

        let error =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect_err("capacity overflow should fail the zip load");

        assert!(matches!(
            error,
            TeardownZipLoadError::UcvhCapacityExceeded { .. }
        ));
    }

    #[test]
    fn voxbox_scale_attribute_scales_box_geometry() {
        let temp = test_dir("scaled_voxbox");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxbox scale="2" size="1 1 1" color="1 0 0"/></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("scaled voxbox should load");

        assert_eq!(stats.voxbox_nodes_exported, 1);
        assert!(stats.unique_written_voxels > 1);
    }

    #[test]
    fn voxbox_brush_uses_vox_pattern_instead_of_solid_fill() {
        let temp = test_dir("voxbox_brush");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><voxbox size="4 1 1" brush="MOD/brush/pattern.vox"/></scene>"#
                        .to_vec(),
                ),
                ("brush/pattern.vox", sparse_brush_vox()),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("brushed voxbox should load");

        assert_eq!(stats.voxbox_nodes_exported, 1);
        assert_eq!(stats.input_voxels, 2);
        assert_eq!(stats.unique_written_voxels, 2);
    }

    #[test]
    fn voxbox_brush_mirrorx_flips_pattern_sampling() {
        let temp = test_dir("voxbox_brush_mirrorx");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><voxbox size="2 1 1" brush="MOD/brush/pattern.vox" mirrorx="true"/></scene>"#
                        .to_vec(),
                ),
                (
                    "brush/pattern.vox",
                    two_color_line_vox([220, 20, 30, 255], [20, 200, 240, 255]),
                ),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("mirrored brushed voxbox should load");

        assert_eq!(stats.input_voxels, 2);
        assert_eq!(
            ucvh.get_voxel(UVec3::new(4, 4, 4)).material,
            material_for_color([20, 200, 240, 255])
        );
        assert_eq!(
            ucvh.get_voxel(UVec3::new(5, 4, 4)).material,
            material_for_color([220, 20, 30, 255])
        );
    }

    #[test]
    fn builtin_prefab_resolves_relative_voxbox_brush_paths() {
        let temp = test_dir("builtin_relative_voxbox_brush");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><instance file="BUILT-IN/prefab/ammo/planks.xml"/></scene>"#.to_vec(),
            )],
        );
        let install = temp.join("Teardown");
        let prefab = install
            .join("data")
            .join("built-in")
            .join("prefab")
            .join("ammo")
            .join("planks.xml");
        std::fs::create_dir_all(prefab.parent().unwrap()).expect("prefab dir");
        std::fs::write(
            &prefab,
            br#"<prefab><voxbox size="4 1 1" brush="../vox/tool/plank.vox"/></prefab>"#,
        )
        .expect("prefab xml");
        let brush = install
            .join("data")
            .join("vox")
            .join("tool")
            .join("plank.vox");
        std::fs::create_dir_all(brush.parent().unwrap()).expect("brush dir");
        std::fs::write(&brush, sparse_brush_vox()).expect("brush vox");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats = load_teardown_zip_into_ucvh(
            &zip_path,
            &mut ucvh,
            TeardownZipLoadOptions {
                teardown_dir: Some(install),
                ..Default::default()
            },
        )
        .expect("teardown zip should load");

        assert_eq!(stats.instance_nodes_exported, 1);
        assert_eq!(stats.voxbox_nodes_exported, 1);
        assert!(stats.missing_mod_refs.is_empty());
        assert!(stats.missing_builtin_refs.is_empty());
        assert_eq!(stats.input_voxels, 2);
        assert_eq!(stats.unique_written_voxels, 2);
    }

    #[test]
    fn voxagon_brush_uses_sparse_vox_pattern_instead_of_solid_fill() {
        let temp = test_dir("voxagon_brush");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><voxagon brush="MOD/brush/pattern.vox" extrude="1"><vertex pos="0 0"/><vertex pos="0.4 0"/><vertex pos="0.4 0.1"/><vertex pos="0 0.1"/></voxagon></scene>"#
                        .to_vec(),
                ),
                ("brush/pattern.vox", sparse_brush_vox()),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("brushed voxagon should load");

        assert_eq!(stats.input_voxels, 4);
        assert_eq!(stats.unique_written_voxels, 2);
    }

    #[test]
    fn vox_color_attribute_tints_loaded_vox_palette() {
        let temp = test_dir("vox_color_tint");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/block.vox" color="0.5 1 0.25"/></scene>"#
                        .to_vec(),
                ),
                ("vox/block.vox", single_voxel_vox([200, 100, 80, 255])),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
            .expect("tinted vox should load");

        assert_eq!(
            ucvh.get_voxel(UVec3::new(4, 4, 4)).material,
            material_for_color([100, 100, 20, 255])
        );
    }

    #[test]
    fn voxscript_heightmap_adds_visible_ground_columns_from_png() {
        let temp = test_dir("voxscript_heightmap");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><voxscript file="MOD/voxscript/terrain.lua"><parameters file="MOD/height.png" scale="3"/></voxscript></scene>"#
                        .to_vec(),
                ),
                (
                    "voxscript/terrain.lua",
                    br#"function init()
                        LoadImage(file)
                        Vox(0, 0, 0)
                        Heightmap(0, 0, 2, 2, scale)
                    end"#
                    .to_vec(),
                ),
                (
                    "height.png",
                    test_rgba_png(
                        2,
                        2,
                        &[
                            [0, 0, 0, 255],
                            [255, 255, 0, 255],
                            [128, 0, 0, 255],
                            [64, 255, 0, 255],
                        ],
                    ),
                ),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("heightmap voxscript should load");

        assert_eq!(stats.unique_written_voxels, 10);
        assert_ne!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(5, 7, 4)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(4, 6, 5)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(5, 5, 5)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(5, 4, 4)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(5, 5, 4)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(5, 6, 4)).material, 0);
    }

    #[test]
    fn voxscript_heightmap_does_not_fill_hidden_plateau_interior() {
        let temp = test_dir("voxscript_heightmap_shell");
        let zip_path = temp.join("map.zip");
        let pixels = vec![[255, 0, 0, 255]; 16];
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><voxscript file="MOD/voxscript/terrain.lua"><parameters file="MOD/height.png" scale="3"/></voxscript></scene>"#
                        .to_vec(),
                ),
                (
                    "voxscript/terrain.lua",
                    br#"function init()
                        LoadImage(file)
                        Vox(0, 0, 0)
                        Heightmap(0, 0, 4, 4, scale)
                    end"#
                    .to_vec(),
                ),
                ("height.png", test_rgba_png(4, 4, &pixels)),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("heightmap voxscript should load");

        assert_eq!(stats.unique_written_voxels, 52);
        assert_ne!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, 0);
        assert_eq!(ucvh.get_voxel(UVec3::new(5, 5, 5)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(5, 7, 5)).material, 0);
    }

    #[test]
    fn voxscript_heightmap_respects_constant_vox_shape_offset() {
        let temp = test_dir("voxscript_heightmap_vox_offset");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene>
                        <voxbox size="1 1 1" color="1 0 0"/>
                        <voxscript file="MOD/voxscript/terrain.lua">
                            <parameters file="MOD/height.png" scale="0"/>
                        </voxscript>
                    </scene>"#
                        .to_vec(),
                ),
                (
                    "voxscript/terrain.lua",
                    br#"function init()
                        LoadImage(file)
                        Vox(2, 0, 0)
                        Heightmap(0, 0, 1, 1, scale)
                    end"#
                        .to_vec(),
                ),
                ("height.png", test_rgba_png(1, 1, &[[255, 0, 0, 255]])),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("offset heightmap voxscript should load");

        assert_eq!(stats.unique_written_voxels, 2);
        let red = material_for_color([255, 0, 0, 255]);
        let terrain = material_for_color([77, 77, 77, 255]);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, red);
        assert_eq!(ucvh.get_voxel(UVec3::new(6, 4, 4)).material, terrain);
    }

    #[test]
    fn voxscript_heightmap_respects_constant_heightmap_crop_range() {
        let temp = test_dir("voxscript_heightmap_crop_range");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><voxscript file="MOD/voxscript/terrain.lua"><parameters file="MOD/height.png" scale="0"/></voxscript></scene>"#
                        .to_vec(),
                ),
                (
                    "voxscript/terrain.lua",
                    br#"function init()
                        LoadImage(file)
                        Vox(0, 0, 0)
                        Heightmap(1, 0, 2, 1, scale)
                    end"#
                    .to_vec(),
                ),
                (
                    "height.png",
                    test_rgba_png(
                        3,
                        1,
                        &[
                            [255, 0, 0, 255],
                            [255, 0, 0, 255],
                            [255, 0, 0, 255],
                        ],
                    ),
                ),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("cropped heightmap voxscript should load");

        assert_eq!(stats.unique_written_voxels, 1);
    }

    #[test]
    fn voxscript_heightmap_recognizes_vintessa_tiled_loop_ranges() {
        let script = r#"
            file = GetString("file", "testground.png", "script png")
            heightScale = GetInt("scale", 64)
            tileSize = GetInt("tilesize", 2)

            function init()
                LoadImage(file)
                w,h = GetImageSize()
                local maxSize = tileSize

                local y0 = 0
                while y0 < h do
                    local y1 = y0 + maxSize
                    if y1 > h then y1 = h end

                    local x0 = 0
                    while x0 < w do
                        local x1 = x0 + maxSize
                        if x1 > w then x1 = w end
                        Vox(x0, 0, y0)
                        Heightmap(x0, y0, x1, y1, heightScale, hollow==0)
                        x0 = x1
                    end
                    y0 = y1
                end
            end
        "#;

        let plan = parse_voxscript_heightmap_plan(script, 5, 3, None)
            .expect("Vintessa tiled terrain loop should parse");

        assert!(!plan.fill_boundary_columns);
        assert_eq!(plan.tiles.len(), 6);
        assert_eq!((plan.tiles[0].x_start, plan.tiles[0].z_start), (0, 0));
        assert_eq!((plan.tiles[0].x_end, plan.tiles[0].z_end), (2, 2));
        assert_eq!((plan.tiles[2].x_start, plan.tiles[2].z_start), (4, 0));
        assert_eq!((plan.tiles[2].x_end, plan.tiles[2].z_end), (5, 2));
        assert_eq!((plan.tiles[5].x_start, plan.tiles[5].z_start), (4, 2));
        assert_eq!((plan.tiles[5].x_end, plan.tiles[5].z_end), (5, 3));
    }

    #[test]
    fn voxscript_heightmap_false_hollow_flag_does_not_fill_outer_edges_to_ground() {
        let heightmap = HeightmapImage {
            width: 2,
            height: 2,
            pixels: vec![[255, 0, 0, 255]; 4],
            height_scale: 3,
            tiles: vec![HeightmapTile {
                coord_offset: IVec3::ZERO,
                x_start: 0,
                z_start: 0,
                x_end: 2,
                z_end: 2,
            }],
            fill_boundary_columns: false,
        };

        assert_eq!(
            heightmap_visible_column_start(&heightmap, 0, 0, 3),
            3,
            "Vintessa-style Heightmap(..., hollow==0) must not create full-height terrain border walls"
        );
    }

    #[test]
    fn voxscript_heightmap_without_hollow_flag_keeps_legacy_solid_outer_edges() {
        let heightmap = HeightmapImage {
            width: 2,
            height: 2,
            pixels: vec![[255, 0, 0, 255]; 4],
            height_scale: 3,
            tiles: vec![HeightmapTile {
                coord_offset: IVec3::ZERO,
                x_start: 0,
                z_start: 0,
                x_end: 2,
                z_end: 2,
            }],
            fill_boundary_columns: true,
        };

        assert_eq!(heightmap_visible_column_start(&heightmap, 0, 0, 3), 0);
    }

    #[test]
    fn heightmap_blue_channel_preserves_teardown_road_specials_over_grass() {
        assert_eq!(
            heightmap_material([80, 180, 10, 255]),
            material_for_color([153, 153, 153, 255])
        );
        assert_eq!(
            heightmap_material([80, 180, 20, 255]),
            material_for_color([51, 51, 51, 255])
        );
        assert_eq!(
            heightmap_material([80, 180, 255, 255]),
            material_for_color([90, 90, 90, 255])
        );
    }

    #[test]
    fn downscaled_axis_aligned_voxbox_writes_target_voxels_once() {
        let temp = test_dir("downscaled_voxbox");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxbox size="64 1 1" color="1 0 0"/></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("teardown zip should load");

        assert_eq!(stats.input_voxels, 64);
        assert!(stats.written_voxels < stats.input_voxels);
        assert_eq!(stats.written_voxels, stats.unique_written_voxels);
    }

    #[test]
    fn stats_report_source_bounds_and_target_scale() {
        let temp = test_dir("source_bounds_scale");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxbox size="64 1 1" color="1 0 0"/></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("teardown zip should load");

        let source_bounds = stats
            .source_bounds
            .expect("source bounds should be reported");
        assert_eq!(source_bounds.min, IVec3::ZERO);
        assert_eq!(source_bounds.max_exclusive, IVec3::new(64, 1, 1));
        assert!(
            stats.target_scale_millis < 1000,
            "oversized maps should report that they were downscaled"
        );
    }

    #[test]
    fn voxbox_position_is_min_corner_so_adjacent_boxes_tile_like_teardown() {
        let temp = test_dir("voxbox_min_corner_position");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxbox pos="0 0 0" size="10 10 10" color="1 0 0"/><voxbox pos="1 0 0" size="10 10 10" color="0 1 0"/></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("teardown zip should load");

        let source_bounds = stats
            .source_bounds
            .expect("source bounds should be reported");
        assert_eq!(source_bounds.min, IVec3::ZERO);
        assert_eq!(source_bounds.max_exclusive, IVec3::new(20, 10, 10));
    }

    #[test]
    fn default_scene_world_preserves_vintessa_native_scale() {
        let bounds = SourceBounds {
            min: IVec3::new(-1903, -17, -1096),
            max_exclusive: IVec3::new(1943, 504, 1451),
            any: true,
        };
        let mapping = TargetMapping::new(
            bounds,
            crate::voxel::generator::default_scene_ucvh_config().world_size,
        );

        assert_eq!(
            mapping.scale_millis(),
            1000,
            "default Vintessa import must preserve Teardown native voxel scale"
        );
    }

    #[test]
    fn target_mapping_scale_one_maps_large_integer_coordinates_exactly() {
        let mapping = TargetMapping {
            source_min: IVec3::new(16_777_217, -32, 9),
            source_max_exclusive: IVec3::new(16_777_300, 64, 20),
            scale: 1.0,
            padding: 4.0,
            scale_one_padding: Some(IVec3::splat(4)),
        };

        assert_eq!(
            mapping.map(IVec3::new(16_777_218, -31, 10)),
            Some(UVec3::new(5, 5, 5))
        );
    }

    #[test]
    fn target_mapping_scale_one_uses_integer_fast_path_for_vintessa_writes() {
        let source = crate::render::source_checks::read_source("src/voxel/teardown_zip_loader.rs");
        let mapping = source
            .split("impl TargetMapping")
            .nth(1)
            .expect("TargetMapping impl should exist")
            .split("#[derive(Debug, Clone, Copy)]")
            .next()
            .expect("TargetMapping impl should end before TargetWriteStats");

        assert!(
            mapping.contains("map_scale_one"),
            "Vintessa-scale 1:1 mapping must avoid per-voxel floating-point map work"
        );
    }

    #[test]
    fn target_write_stats_avoid_refcell_borrow_in_per_voxel_hot_path() {
        let source = crate::render::source_checks::read_source("src/voxel/teardown_zip_loader.rs");
        let forbidden = concat!("RefCell::new(", "TargetWriteStats::default())");

        assert!(
            !source.contains(forbidden),
            "Vintessa write hot path must not borrow a RefCell for every target voxel"
        );
        assert!(
            source.contains("TargetWriteStatsTracker"),
            "Teardown loader should use a low-overhead interior-mutable stats tracker"
        );
    }

    #[test]
    fn rasterizes_voxagon_polygon_prism() {
        let temp = test_dir("voxagon");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxagon pos="0 0 0" extrude="2" color="0 1 0"><vertex pos="0 0"/><vertex pos="2 0"/><vertex pos="2 2"/><vertex pos="0 2"/></voxagon></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("voxagon zip should load");

        assert!(stats.input_voxels > 0);
        assert!(stats.unique_written_voxels > 0);
    }

    #[test]
    fn voxagon_node_transform_places_polygon_in_world() {
        let temp = test_dir("voxagon_transform");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxbox size="1 1 1" color="1 0 0"/><voxagon pos="1 0 0" extrude="1" color="0 1 0"><vertex pos="0 0"/><vertex pos="0.1 0"/><vertex pos="0.1 0.1"/><vertex pos="0 0.1"/></voxagon></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("translated voxagon should load");

        let bounds = stats
            .target_bounds
            .expect("translated voxagon should write target bounds");
        assert!(bounds.max_exclusive.x > 10);
        assert_eq!(stats.unique_written_voxels, 2);
    }

    #[test]
    fn voxagon_axis_controls_extrusion_direction() {
        let temp = test_dir("voxagon_axis");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxagon axis="0 0 1" extrude="2" color="0 1 0"><vertex pos="0 0"/><vertex pos="0.1 0"/><vertex pos="0.1 0.1"/><vertex pos="0 0.1"/></voxagon></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
            .expect("axis voxagon should load");

        let material = material_for_color([0, 255, 0, 255]);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, material);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 4, 5)).material, material);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 5, 4)).material, 0);
    }

    #[test]
    fn voxagon_axis_accepts_teardown_axis_letters() {
        let temp = test_dir("voxagon_axis_letters");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxagon axis="x" extrude="2" color="0 1 0"><vertex pos="0 0"/><vertex pos="0.1 0"/><vertex pos="0.1 0.1"/><vertex pos="0 0.1"/></voxagon></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
            .expect("axis-letter voxagon should load");

        let material = material_for_color([0, 255, 0, 255]);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, material);
        assert_eq!(ucvh.get_voxel(UVec3::new(5, 4, 4)).material, material);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 5, 4)).material, 0);
    }

    #[test]
    fn voxagon_negative_extrude_extends_against_axis_direction() {
        let temp = test_dir("voxagon_negative_extrude");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxagon extrude="-2" color="0 1 0"><vertex pos="0 0"/><vertex pos="1 0"/><vertex pos="1 1"/><vertex pos="0 1"/></voxagon></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
            .expect("negative extrude voxagon should load");

        let material = material_for_color([0, 255, 0, 255]);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, material);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 5, 4)).material, material);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 6, 4)).material, 0);
    }

    #[test]
    fn voxagon_scale_attribute_expands_polygon_cells() {
        let temp = test_dir("voxagon_scale");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene><voxagon scale="2" extrude="1" color="0 1 0"><vertex pos="0 0"/><vertex pos="1 0"/><vertex pos="1 1"/><vertex pos="0 1"/></voxagon></scene>"#.to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("scaled voxagon should load");

        assert!(stats.unique_written_voxels > 1);
        assert_eq!(stats.out_of_bounds_voxels, 0);
        let material = material_for_color([0, 255, 0, 255]);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, material);
        assert_eq!(ucvh.get_voxel(UVec3::new(5, 5, 5)).material, material);
    }

    #[test]
    fn voxagon_vertices_are_in_world_units_not_native_voxel_units() {
        let xml = roxmltree::Document::parse(
            r#"<voxagon extrude="1"><vertex pos="0 0"/><vertex pos="1 0"/><vertex pos="1 1"/><vertex pos="0 1"/></voxagon>"#,
        )
        .expect("voxagon xml");
        let voxagon = xml.root_element();

        let (min, max_exclusive) =
            voxagon_local_native_bounds(voxagon, 10).expect("voxagon bounds");

        assert_eq!(min, IVec3::ZERO);
        assert_eq!(max_exclusive, IVec3::new(10, 1, 10));
    }

    #[test]
    fn vehicle_and_wheel_nodes_transform_child_geometry() {
        let temp = test_dir("vehicle_transform");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene>
                    <voxbox size="1 1 1" color="1 0 0"/>
                    <vehicle pos="2 0 0">
                        <body pos="1 0 0">
                            <voxbox size="1 1 1" color="0 1 0"/>
                        </body>
                        <wheel pos="3 0 0">
                            <voxbox size="1 1 1" color="0 0 1"/>
                        </wheel>
                        <shape pos="3.5 0 0">
                            <voxbox size="1 1 1" color="1 1 0"/>
                        </shape>
                    </vehicle>
                </scene>"#
                    .to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
            .expect("vehicle transform scene should load");

        assert_eq!(
            ucvh.get_voxel(UVec3::new(34, 4, 4)).material,
            material_for_color([0, 255, 0, 255])
        );
        assert_eq!(
            ucvh.get_voxel(UVec3::new(54, 4, 4)).material,
            material_for_color([0, 0, 255, 255])
        );
        assert_eq!(
            ucvh.get_voxel(UVec3::new(59, 4, 4)).material,
            material_for_color([255, 255, 0, 255])
        );
        assert_eq!(ucvh.get_voxel(UVec3::new(14, 4, 4)).material, 0);
        assert!(ucvh.get_voxel(UVec3::new(34, 4, 4)).material != 0);
    }

    #[test]
    fn script_nodes_transform_child_geometry() {
        let temp = test_dir("script_transform");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"
                <scene>
                    <voxbox size="1 1 1" color="1 0 0"/>
                    <script pos="2 0 0" file="alarm.lua">
                        <voxbox size="1 1 1" color="0 1 0"/>
                    </script>
                </scene>
                "#
                .to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("script child geometry should load");

        assert_eq!(stats.voxbox_nodes_exported, 2);
        assert_eq!(stats.unique_written_voxels, 2);
        assert_eq!(
            ucvh.get_voxel(UVec3::new(4, 4, 4)).material,
            material_for_color([255, 0, 0, 255])
        );
        assert_eq!(
            ucvh.get_voxel(UVec3::new(24, 4, 4)).material,
            material_for_color([0, 255, 0, 255])
        );
    }

    #[test]
    fn voxbox_hole_brush_erases_previous_geometry() {
        let temp = test_dir("voxbox_hole");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene>
                    <voxbox size="3 1 1" color="1 0 0"/>
                    <voxbox pos="0.1 0 0" size="1 1 1" brush="hole"/>
                </scene>"#
                    .to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
            .expect("hole brush scene should load");

        assert_ne!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, 0);
        assert_eq!(ucvh.get_voxel(UVec3::new(5, 4, 4)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(6, 4, 4)).material, 0);
    }

    #[test]
    fn voxbox_hoe_brush_erases_previous_geometry_like_teardown_typo() {
        let temp = test_dir("voxbox_hoe");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene>
                    <voxbox size="3 1 1" color="1 0 0"/>
                    <voxbox pos="0.1 0 0" size="1 1 1" brush="hoe"/>
                </scene>"#
                    .to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
            .expect("hoe brush scene should load");

        assert_ne!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, 0);
        assert_eq!(ucvh.get_voxel(UVec3::new(5, 4, 4)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(6, 4, 4)).material, 0);
    }

    #[test]
    fn water_nodes_are_not_exported_as_opaque_static_voxels() {
        let temp = test_dir("water_plane");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[(
                "main.xml",
                br#"<scene>
                    <voxbox size="1 1 1" color="1 0 0"/>
                    <water pos="0 0.1 0" size="0.1 0.1" color="0 0 1"/>
                </scene>"#
                    .to_vec(),
            )],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("finite water scene should load");

        assert_eq!(stats.input_voxels, 1);
        assert_eq!(
            ucvh.get_voxel(UVec3::new(4, 5, 4)).material,
            0,
            "Teardown water should not become opaque static UCVH geometry"
        );
    }

    #[test]
    fn downscaled_axis_aligned_vox_writes_target_voxels_once() {
        let temp = test_dir("downscaled_vox");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/line.vox"/></scene>"#.to_vec(),
                ),
                ("vox/line.vox", line_vox(64)),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("teardown zip should load");

        assert_eq!(stats.input_voxels, 64);
        assert!(stats.written_voxels < stats.input_voxels);
        assert_eq!(stats.written_voxels, stats.unique_written_voxels);
    }

    #[test]
    fn repeated_downscaled_vox_reuses_cached_target_plan() {
        let temp = test_dir("downscaled_vox_plan_cache");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/line.vox"/><vox file="MOD/vox/line.vox"/></scene>"#
                        .to_vec(),
                ),
                ("vox/line.vox", line_vox(64)),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("teardown zip should load");

        assert_eq!(stats.downsampled_vox_nodes, 2);
        assert_eq!(stats.downsampled_vox_plan_misses, 1);
        assert_eq!(stats.downsampled_vox_plan_hits, 1);
        assert_eq!(stats.downsampled_vox_plans_prebuilt, 1);
        assert_eq!(stats.vox_cache_misses, 1);
        assert!(stats.vox_cache_hits >= 5);
    }

    #[test]
    fn repeated_axis_aligned_rotated_vox_reuses_cached_target_plan() {
        let temp = test_dir("rotated_downscaled_vox_plan_cache");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox rot="0 90 0" file="MOD/vox/line.vox"/><vox rot="0 90 0" file="MOD/vox/line.vox"/></scene>"#
                        .to_vec(),
                ),
                ("vox/line.vox", line_vox(64)),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("teardown zip should load");

        assert_eq!(stats.downsampled_vox_nodes, 2);
        assert_eq!(stats.downsampled_vox_plan_misses, 1);
        assert_eq!(stats.downsampled_vox_plan_hits, 1);
        assert_eq!(stats.downsampled_vox_plans_prebuilt, 1);
    }

    #[test]
    fn repeated_arbitrary_rotated_vox_reuses_cached_target_plan() {
        let temp = test_dir("arbitrary_rotated_downscaled_vox_plan_cache");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox rot="0 17 0" file="MOD/vox/line.vox"/><vox rot="0 17 0" file="MOD/vox/line.vox"/></scene>"#
                        .to_vec(),
                ),
                ("vox/line.vox", line_vox(64)),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("teardown zip should load");

        assert_eq!(stats.downsampled_vox_nodes, 2);
        assert_eq!(stats.downsampled_vox_plan_misses, 1);
        assert_eq!(stats.downsampled_vox_plan_hits, 1);
        assert_eq!(stats.downsampled_vox_plans_prebuilt, 1);
    }

    #[test]
    fn vox_object_attribute_loads_only_named_shape() {
        let temp = test_dir("named_object_vox");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/named.vox" object="wanted"/></scene>"#.to_vec(),
                ),
                ("vox/named.vox", named_two_shape_vox()),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("named-object teardown zip should load");

        assert_eq!(stats.input_voxels, 1);
        assert_eq!(stats.unique_written_voxels, 1);
        assert_ne!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, 0);
    }

    #[test]
    fn vox_object_attribute_uses_selected_object_transform_as_local_origin() {
        let temp = test_dir("named_object_transform_origin");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/tree.vox" object="trunk"/><vox pos="0 1 0" file="MOD/vox/tree.vox" object="crown"/></scene>"#
                        .to_vec(),
                ),
                ("vox/tree.vox", translated_named_two_shape_vox()),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("named object teardown zip should load");

        assert_eq!(stats.input_voxels, 2);
        assert_eq!(stats.unique_written_voxels, 2);
        assert_ne!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, 0);
        assert_ne!(ucvh.get_voxel(UVec3::new(4, 14, 4)).material, 0);
        assert_eq!(ucvh.get_voxel(UVec3::new(4, 24, 4)).material, 0);
    }

    #[test]
    fn vox_object_attribute_can_select_hidden_named_library_object() {
        let temp = test_dir("hidden_named_object");
        let zip_path = temp.join("map.zip");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/hidden.vox" object="part"/></scene>"#.to_vec(),
                ),
                ("vox/hidden.vox", hidden_named_object_vox()),
            ],
        );
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("hidden named object should load when explicitly selected");

        assert_eq!(stats.input_voxels, 1);
        assert_eq!(stats.unique_written_voxels, 1);
        assert_ne!(ucvh.get_voxel(UVec3::new(4, 4, 4)).material, 0);
    }

    #[test]
    fn teardown_euler_rotation_matches_teardown_quat_euler_order() {
        let rot = Vec3::new(10.0, 20.0, 30.0);
        let point = Vec3::new(1.0, 2.0, 3.0);
        let expected = Vec3::new(1.2623996, 1.7545859, 3.0541408);

        assert!(
            teardown_euler_rotation(rot)
                .transform_point3(point)
                .abs_diff_eq(expected, 1.0e-5)
        );
    }

    #[test]
    fn cached_visible_voxel_materials_match_scene_palette_quantization() {
        let scene = parse_vox(&line_vox(3)).expect("test vox should parse");
        let cached = CachedVoxScene::new(scene);
        let mut cached_voxels = Vec::new();
        let mut reference_voxels = Vec::new();

        visit_cached_visible_voxels(&cached, None, &cached.materials, |position, material| {
            cached_voxels.push((position, material));
        });
        cached.scene.visit_visible_voxels(|position, color| {
            reference_voxels.push((position, material_for_color(color)));
        });

        assert_eq!(cached.bounds, cached.scene.bounds());
        assert_eq!(
            cached.visible_voxel_count,
            cached.scene.visible_voxel_count()
        );
        assert_eq!(cached_voxels, reference_voxels);
    }

    #[test]
    #[ignore = "requires local Vintessa workshop zip and Teardown install"]
    fn local_vintessa_teardown_zip_loads_when_present() {
        let zip_path = default_zip_map_path();
        if !zip_path.exists() {
            return;
        }
        let mut ucvh = Ucvh::new(crate::voxel::generator::default_scene_ucvh_config());

        let load_start = std::time::Instant::now();
        let stats =
            load_teardown_zip_into_ucvh(&zip_path, &mut ucvh, TeardownZipLoadOptions::default())
                .expect("local Vintessa zip should load");
        let load_elapsed = load_start.elapsed();
        let rebuild_start = std::time::Instant::now();
        ucvh.rebuild_hierarchy();
        let rebuild_elapsed = rebuild_start.elapsed();

        println!(
            "vintessa_zip unique={} written={} input={} source={:?} target={:?} scale_millis={} vox={} voxbox={} instances={} missing_mod={} missing_builtin={} downsampled_vox={} plan_hits={} plan_misses={} plans_prebuilt={} vox_cache_hits={} vox_cache_misses={} load={:?} rebuild={:?} total_with_rebuild={:?}",
            stats.unique_written_voxels,
            stats.written_voxels,
            stats.input_voxels,
            stats.source_bounds,
            stats.target_bounds,
            stats.target_scale_millis,
            stats.vox_nodes_exported,
            stats.voxbox_nodes_exported,
            stats.instance_nodes_exported,
            stats.missing_mod_refs.len(),
            stats.missing_builtin_refs.len(),
            stats.downsampled_vox_nodes,
            stats.downsampled_vox_plan_hits,
            stats.downsampled_vox_plan_misses,
            stats.downsampled_vox_plans_prebuilt,
            stats.vox_cache_hits,
            stats.vox_cache_misses,
            load_elapsed,
            rebuild_elapsed,
            load_elapsed + rebuild_elapsed,
        );
        assert!(stats.unique_written_voxels > 100_000);
    }

    #[test]
    fn staged_bricks_use_sparse_index_map_not_full_brick_grid_vector() {
        let source = crate::render::source_checks::read_source("src/voxel/teardown_zip_loader.rs");
        let staged = source
            .split("struct StagedBricks")
            .nth(1)
            .expect("StagedBricks should exist")
            .split("struct StagedBrick")
            .next()
            .expect("StagedBricks implementation should end before StagedBrick");

        assert!(
            staged.contains("indices: U64IndexMap<usize>"),
            "Teardown zip staging must use a sparse brick index map"
        );
        assert!(
            !staged.contains("vec![None; len]") && !staged.contains("Vec<Option<usize>>"),
            "Vintessa-scale zip staging must not allocate one index slot per brick-grid coordinate"
        );
    }

    #[test]
    fn staged_bricks_cache_recent_brick_lookup_for_sequential_voxel_writes() {
        let source = crate::render::source_checks::read_source("src/voxel/teardown_zip_loader.rs");
        let staged = source
            .split("struct StagedBricks")
            .nth(1)
            .expect("StagedBricks should exist")
            .split("struct StagedBrick")
            .next()
            .expect("StagedBricks implementation should end before StagedBrick");

        for token in [
            "last_lookup_key: Option<u64>",
            "last_lookup_index: usize",
            "self.last_lookup_key == Some(key)",
        ] {
            assert!(
                staged.contains(token),
                "Teardown staging must avoid repeated HashMap lookups for sequential voxels; missing {token}"
            );
        }
    }
}
