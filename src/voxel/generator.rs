// src/voxel/generator.rs
use crate::voxel::brick::{BRICK_EDGE, BrickData, VoxelCell};
use crate::voxel::teardown_zip_loader::{
    TeardownZipLoadError, TeardownZipLoadOptions, TeardownZipWriteStats,
    load_teardown_zip_into_ucvh,
};
use crate::voxel::ucvh::{Ucvh, UcvhConfig};
use crate::voxel::vox_loader::{VoxLoadError, VoxWriteStats, load_vox_file_into_ucvh};
use glam::{UVec3, Vec3};
use std::path::Path;

pub const DEFAULT_VOX_MAP_PATH: &str = "run/Vintessa_Hills_static.vox";
pub use crate::voxel::teardown_zip_loader::{DEFAULT_TEARDOWN_ZIP_MAP_PATH, default_zip_map_path};
pub const DEFAULT_SCENE_WORLD_SIZE: UVec3 = UVec3::new(4096, 768, 3072);
pub const DEFAULT_SCENE_BRICK_CAPACITY: u32 = 524_288;
pub const MAT_CHECKER_WHITE: u16 = 1;
pub const MAT_CHECKER_BLACK: u16 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultSceneKind {
    TeardownZip,
    VoxFile,
    CheckerboardFallback,
}

#[derive(Debug, Clone)]
pub struct DefaultSceneLoadResult {
    pub kind: DefaultSceneKind,
    pub brick_count: u32,
    pub teardown_zip_stats: Option<TeardownZipWriteStats>,
    pub teardown_zip_error: Option<TeardownZipLoadError>,
    pub vox_stats: Option<VoxWriteStats>,
    pub vox_error: Option<VoxLoadError>,
}

pub trait VoxelGenerator {
    /// Generate brick data at the given brick grid coordinate.
    /// Returns None if the brick would be entirely empty.
    fn generate_brick(&self, brick_pos: UVec3, config: &UcvhConfig) -> Option<BrickData>;
}

pub struct SphereGenerator {
    pub center: Vec3,
    pub radius: f32,
    pub material: u16,
}

impl VoxelGenerator for SphereGenerator {
    fn generate_brick(&self, brick_pos: UVec3, _config: &UcvhConfig) -> Option<BrickData> {
        let base = brick_pos * BRICK_EDGE;
        let mut data = BrickData::new();
        let mut any_solid = false;

        for lz in 0..BRICK_EDGE {
            for ly in 0..BRICK_EDGE {
                for lx in 0..BRICK_EDGE {
                    let world = Vec3::new(
                        (base.x + lx) as f32 + 0.5,
                        (base.y + ly) as f32 + 0.5,
                        (base.z + lz) as f32 + 0.5,
                    );
                    if world.distance(self.center) <= self.radius {
                        data.set_voxel(
                            lx,
                            ly,
                            lz,
                            VoxelCell {
                                material: self.material,
                                flags: 1, // solid
                                emissive: [0; 3],
                                _pad: 0,
                            },
                        );
                        any_solid = true;
                    }
                }
            }
        }

        if any_solid { Some(data) } else { None }
    }
}

/// Generate a Sponza-inspired architectural scene.
pub fn generate_sponza_scene(ucvh: &mut Ucvh) -> u32 {
    let generator = crate::voxel::sponza_generator::SponzaGenerator;
    let bgs = ucvh.config.brick_grid_size;
    let mut count = 0u32;
    for bz in 0..bgs.z {
        for by in 0..bgs.y {
            for bx in 0..bgs.x {
                let bp = UVec3::new(bx, by, bz);
                if let Some(data) = generator.generate_brick(bp, &ucvh.config) {
                    if ucvh.write_brick(bp, &data) {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

pub fn generate_default_scene(ucvh: &mut Ucvh) -> DefaultSceneLoadResult {
    let zip_path = default_zip_map_path();
    generate_default_scene_from_paths(ucvh, zip_path.as_path(), Path::new(DEFAULT_VOX_MAP_PATH))
}

pub fn default_scene_ucvh_config() -> UcvhConfig {
    UcvhConfig::with_brick_capacity(DEFAULT_SCENE_WORLD_SIZE, DEFAULT_SCENE_BRICK_CAPACITY)
}

pub fn generate_default_scene_from_paths(
    ucvh: &mut Ucvh,
    zip_path: &Path,
    vox_path: &Path,
) -> DefaultSceneLoadResult {
    let mut teardown_zip_error = None;
    let mut zip_ucvh = None;
    if zip_path.exists() {
        let mut candidate_ucvh = Ucvh::new(UcvhConfig::with_brick_capacity(
            ucvh.config.world_size,
            ucvh.config.brick_capacity,
        ));
        match load_teardown_zip_into_ucvh(
            zip_path,
            &mut candidate_ucvh,
            TeardownZipLoadOptions::default(),
        ) {
            Ok(stats) if stats.unique_written_voxels > 0 => {
                let brick_count = candidate_ucvh.allocated_brick_count();
                *ucvh = candidate_ucvh;
                return DefaultSceneLoadResult {
                    kind: DefaultSceneKind::TeardownZip,
                    brick_count,
                    teardown_zip_stats: Some(stats),
                    teardown_zip_error: None,
                    vox_stats: None,
                    vox_error: None,
                };
            }
            Ok(stats) => {
                zip_ucvh = Some(stats);
            }
            Err(error) => {
                teardown_zip_error = Some(error);
            }
        }
    }

    let mut result = generate_default_scene_from_path(ucvh, vox_path);
    result.teardown_zip_stats = zip_ucvh;
    result.teardown_zip_error = teardown_zip_error;
    result
}

pub fn generate_default_scene_from_path(ucvh: &mut Ucvh, path: &Path) -> DefaultSceneLoadResult {
    match load_vox_file_into_ucvh(path, ucvh) {
        Ok(stats) if stats.unique_written_voxels > 0 => DefaultSceneLoadResult {
            kind: DefaultSceneKind::VoxFile,
            brick_count: ucvh.allocated_brick_count(),
            teardown_zip_stats: None,
            teardown_zip_error: None,
            vox_stats: Some(stats),
            vox_error: None,
        },
        Ok(stats) => {
            let brick_count = generate_checkerboard_platform_scene(ucvh);
            DefaultSceneLoadResult {
                kind: DefaultSceneKind::CheckerboardFallback,
                brick_count,
                teardown_zip_stats: None,
                teardown_zip_error: None,
                vox_stats: Some(stats),
                vox_error: None,
            }
        }
        Err(error) => {
            let brick_count = generate_checkerboard_platform_scene(ucvh);
            DefaultSceneLoadResult {
                kind: DefaultSceneKind::CheckerboardFallback,
                brick_count,
                teardown_zip_stats: None,
                teardown_zip_error: None,
                vox_stats: None,
                vox_error: Some(error),
            }
        }
    }
}

pub fn generate_checkerboard_platform_scene(ucvh: &mut Ucvh) -> u32 {
    let world = ucvh.config.world_size;
    let tile_size = 8;
    for z in 0..world.z {
        for x in 0..world.x {
            let tile = (x / tile_size) + (z / tile_size);
            let material = if tile % 2 == 0 {
                MAT_CHECKER_WHITE
            } else {
                MAT_CHECKER_BLACK
            };
            ucvh.set_voxel(UVec3::new(x, 0, z), VoxelCell::new(material, 1, [0; 3]));
        }
    }
    ucvh.allocated_brick_count()
}

/// Generate a demo scene: solid sphere in center of world.
pub fn generate_demo_scene(ucvh: &mut Ucvh) -> u32 {
    let world = ucvh.config.world_size.as_vec3();
    let sphere = SphereGenerator {
        center: world * 0.5,
        radius: world.x.min(world.y).min(world.z) * 0.35,
        material: 1,
    };

    let bgs = ucvh.config.brick_grid_size;
    let mut count = 0u32;
    for bz in 0..bgs.z {
        for by in 0..bgs.y {
            for bx in 0..bgs.x {
                let bp = UVec3::new(bx, by, bz);
                if let Some(data) = sphere.generate_brick(bp, &ucvh.config) {
                    if ucvh.write_brick(bp, &data) {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use zip::CompressionMethod;
    use zip::write::SimpleFileOptions;

    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "revolumetric_generator_{name}_{}",
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

    #[test]
    fn sphere_generates_bricks() {
        let config = UcvhConfig::new(UVec3::splat(64)); // 8^3 brick grid
        let sphere = SphereGenerator {
            center: Vec3::splat(32.0),
            radius: 20.0,
            material: 1,
        };
        // Center brick at (4,4,4) in brick coords should be fully inside
        let data = sphere.generate_brick(UVec3::splat(4), &config);
        assert!(data.is_some());
        let data = data.unwrap();
        assert!(data.occupancy.count > 0);
    }

    #[test]
    fn sphere_empty_outside() {
        let config = UcvhConfig::new(UVec3::splat(64));
        let sphere = SphereGenerator {
            center: Vec3::splat(32.0),
            radius: 10.0,
            material: 1,
        };
        // Corner brick at (0,0,0) should be empty (far from sphere center)
        let data = sphere.generate_brick(UVec3::ZERO, &config);
        assert!(data.is_none());
    }

    #[test]
    fn demo_scene_populates_ucvh() {
        // Use 128^3 world so brick_grid_size = 16^3, giving dims[4]=1 (levels[3] is valid).
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(128)));
        let count = generate_demo_scene(&mut ucvh);
        assert!(count > 0, "should have allocated some bricks");
        ucvh.rebuild_hierarchy();
        // Root should be non-empty
        assert_ne!(ucvh.hierarchy.levels[3][0].child_mask, 0);
    }

    #[test]
    fn sponza_emissive_voxels_survive_ucvh_generation() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(128)));

        let count = generate_sponza_scene(&mut ucvh);
        assert!(count > 0, "sponza should allocate bricks");

        let sconce = ucvh.get_voxel(UVec3::new(20, 24, 19));
        assert_ne!(sconce.material, 0, "wall sconce should be written");
        assert_eq!(sconce.emissive, [255, 180, 80]);

        let chandelier = ucvh.get_voxel(UVec3::new(64, 82, 64));
        assert_ne!(chandelier.material, 0, "chandelier should be written");
        assert_eq!(chandelier.emissive, [255, 220, 140]);
    }

    #[test]
    fn default_scene_missing_vox_file_uses_checkerboard_fallback() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let result = generate_default_scene_from_path(
            &mut ucvh,
            std::path::Path::new("run/definitely_missing_test_map.vox"),
        );

        assert_eq!(result.kind, DefaultSceneKind::CheckerboardFallback);
        assert!(result.brick_count > 0);
        assert_eq!(
            ucvh.get_voxel(UVec3::new(1, 0, 1)).material,
            MAT_CHECKER_WHITE
        );
        assert_eq!(
            ucvh.get_voxel(UVec3::new(9, 0, 1)).material,
            MAT_CHECKER_BLACK
        );
    }

    #[test]
    fn default_scene_prefers_teardown_zip_over_vox_fallback() {
        let temp = test_dir("zip_first");
        let zip_path = temp.join("map.zip");
        let vox_path = temp.join("fallback.vox");
        write_workshop_zip(
            &zip_path,
            &[
                (
                    "main.xml",
                    br#"<scene><vox file="MOD/vox/block.vox"/></scene>"#.to_vec(),
                ),
                ("vox/block.vox", single_voxel_vox([10, 220, 30, 255])),
            ],
        );
        std::fs::write(&vox_path, single_voxel_vox([220, 20, 30, 255])).expect("fallback vox");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let result = generate_default_scene_from_paths(&mut ucvh, &zip_path, &vox_path);

        assert_eq!(result.kind, DefaultSceneKind::TeardownZip);
        assert!(result.teardown_zip_stats.is_some());
        assert!(result.vox_stats.is_none());
        assert_eq!(
            result
                .teardown_zip_stats
                .as_ref()
                .unwrap()
                .unique_written_voxels,
            1
        );
    }

    #[test]
    fn default_scene_uses_vox_when_zip_is_missing() {
        let temp = test_dir("vox_second");
        let zip_path = temp.join("missing.zip");
        let vox_path = temp.join("fallback.vox");
        std::fs::write(&vox_path, single_voxel_vox([220, 20, 30, 255])).expect("fallback vox");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let result = generate_default_scene_from_paths(&mut ucvh, &zip_path, &vox_path);

        assert_eq!(result.kind, DefaultSceneKind::VoxFile);
        assert!(result.teardown_zip_error.is_none());
        assert!(result.vox_stats.is_some());
    }

    #[test]
    fn default_scene_discards_partial_zip_load_before_vox_fallback() {
        let temp = test_dir("zip_capacity_then_vox");
        let zip_path = temp.join("map.zip");
        let vox_path = temp.join("fallback.vox");
        let mut xml = String::from("<scene>");
        for x in 0..65 {
            xml.push_str(&format!(
                r#"<voxbox pos="{x} 0 0" size="1 1 1" color="1 0 0"/>"#
            ));
        }
        xml.push_str("</scene>");
        write_workshop_zip(&zip_path, &[("main.xml", xml.into_bytes())]);
        std::fs::write(&vox_path, single_voxel_vox([220, 20, 30, 255])).expect("fallback vox");
        let mut ucvh = Ucvh::new(UcvhConfig::with_brick_capacity(UVec3::new(520, 8, 8), 64));

        let result = generate_default_scene_from_paths(&mut ucvh, &zip_path, &vox_path);

        assert_eq!(result.kind, DefaultSceneKind::VoxFile);
        assert!(matches!(
            result.teardown_zip_error,
            Some(TeardownZipLoadError::UcvhCapacityExceeded { .. })
        ));
        let vox_stats = result.vox_stats.expect("vox fallback should load");
        assert!(vox_stats.unique_written_voxels > 0);
        assert_eq!(ucvh.allocated_brick_count(), result.brick_count);
        assert_eq!(ucvh.allocated_brick_count(), 1);
    }

    #[test]
    fn checkerboard_platform_alternates_by_tile() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));

        let bricks = generate_checkerboard_platform_scene(&mut ucvh);

        assert!(bricks > 0);
        assert_eq!(
            ucvh.get_voxel(UVec3::new(2, 0, 2)).material,
            MAT_CHECKER_WHITE
        );
        assert_eq!(
            ucvh.get_voxel(UVec3::new(10, 0, 2)).material,
            MAT_CHECKER_BLACK
        );
        assert_eq!(
            ucvh.get_voxel(UVec3::new(10, 0, 10)).material,
            MAT_CHECKER_WHITE
        );
        assert_eq!(ucvh.get_voxel(UVec3::new(2, 1, 2)).material, 0);
    }

    #[test]
    fn default_scene_config_uses_high_detail_world_with_bounded_capacity() {
        let config = default_scene_ucvh_config();

        assert_eq!(config.world_size, UVec3::new(4096, 768, 3072));
        assert_eq!(config.brick_capacity, DEFAULT_SCENE_BRICK_CAPACITY);
        assert!(config.brick_capacity >= 524_288);
        assert!(
            config.brick_capacity < UcvhConfig::new(UVec3::new(4096, 768, 3072)).brick_capacity
        );
    }

    #[test]
    #[ignore = "requires local generated run/Vintessa_Hills_static.vox"]
    fn local_default_vintessa_vox_file_loads_when_present() {
        if !std::path::Path::new(DEFAULT_VOX_MAP_PATH).exists() {
            return;
        }
        let mut ucvh = Ucvh::new(default_scene_ucvh_config());

        let result = generate_default_scene(&mut ucvh);

        assert_eq!(result.kind, DefaultSceneKind::VoxFile);
        assert!(result.brick_count > 0);
        assert!(result.brick_count < DEFAULT_SCENE_BRICK_CAPACITY);
        let stats = result.vox_stats.expect("vox stats should be present");
        assert!(stats.unique_written_voxels > 100_000);
    }
}
