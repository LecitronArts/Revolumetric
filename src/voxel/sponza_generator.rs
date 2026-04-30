use crate::voxel::brick::{BRICK_EDGE, BrickData, VoxelCell};
use crate::voxel::generator::VoxelGenerator;
use crate::voxel::ucvh::UcvhConfig;
use glam::{UVec3, Vec3};

/// Material IDs matching the shader-side MATERIAL_ALBEDO LUT.
const MAT_STONE: u16 = 1;
const MAT_RED_CLOTH: u16 = 2;
const MAT_GREEN_CLOTH: u16 = 3;
const MAT_BLUE: u16 = 4;
const MAT_BRICK: u16 = 5;
const MAT_DARK_STONE: u16 = 6;
const MAT_WOOD: u16 = 7;

/// Sponza-inspired architectural scene for 128³ world.
///
/// Layout (world coords, offset from (16, 0, 4)):
///   x: 0..96  (building width)
///   y: 0..96  (height: ground floor + second floor + roof)
///   z: 0..120 (building length)
pub struct SponzaGenerator;

impl SponzaGenerator {
    fn eval_voxel(p: Vec3) -> Option<(u16, [u8; 3])> {
        let offset = Vec3::new(16.0, 0.0, 4.0);
        let v = p - offset;

        // Bounds check
        if v.x < 0.0 || v.x > 96.0 || v.y < 0.0 || v.y > 96.0 || v.z < 0.0 || v.z > 120.0 {
            return None;
        }

        // === FLOOR (y < 2): checkerboard stone/dark_stone ===
        if v.y < 2.0 {
            let cx = ((v.x / 8.0).floor() as i32) & 1;
            let cz = ((v.z / 8.0).floor() as i32) & 1;
            return if cx ^ cz == 0 {
                Some((MAT_STONE, [0; 3]))
            } else {
                Some((MAT_DARK_STONE, [0; 3]))
            };
        }

        // === CEILING (y 88-92): stone slab with courtyard opening ===
        if v.y > 88.0 && v.y < 92.0 {
            if v.x > 30.0 && v.x < 66.0 && v.z > 20.0 && v.z < 100.0 {
                // Open to sky — no ceiling here
            } else {
                return Some((MAT_STONE, [0; 3]));
            }
        }

        // === SECOND FLOOR (y 44-46): wood planks, full width ===
        if v.y > 44.0 && v.y < 46.0 && v.x > 2.0 && v.x < 94.0 && v.z > 2.0 && v.z < 118.0 {
            // Balcony opening in the center
            if v.x > 30.0 && v.x < 66.0 && v.z > 20.0 && v.z < 100.0 {
                // Open atrium — no floor here
            } else {
                return Some((MAT_WOOD, [0; 3]));
            }
        }

        // === BALCONY RAILING (y 46-50) around the atrium opening ===
        if v.y > 46.0 && v.y < 50.0 {
            let on_x_edge = (v.x > 29.0 && v.x < 31.0) || (v.x > 65.0 && v.x < 67.0);
            let on_z_edge = (v.z > 19.0 && v.z < 21.0) || (v.z > 99.0 && v.z < 101.0);
            let in_x_range = v.x > 29.0 && v.x < 67.0;
            let in_z_range = v.z > 19.0 && v.z < 101.0;
            if (on_x_edge && in_z_range) || (on_z_edge && in_x_range) {
                // Railing posts every 8 units, or top rail
                if v.y > 49.0 || ((v.x as i32) % 8 < 2) || ((v.z as i32) % 8 < 2) {
                    return Some((MAT_WOOD, [0; 3]));
                }
            }
        }

        // === OUTER WALLS ===
        let wall_inner = v.x > 3.0 && v.x < 93.0 && v.z > 3.0 && v.z < 117.0;
        if !wall_inner && v.y < 92.0 {
            // Left wall (x < 3): brick pattern with mortar lines
            if v.x < 3.0 {
                // Window openings (2 rows, every 20 z-units)
                let wz = (v.z - 10.0) % 20.0;
                let is_window = wz > 4.0 && wz < 14.0;
                let is_lower_window = is_window && v.y > 16.0 && v.y < 36.0;
                let is_upper_window = is_window && v.y > 54.0 && v.y < 74.0;
                if is_lower_window || is_upper_window {
                    return None; // window opening
                }
                // Brick pattern with mortar
                let by = ((v.y / 4.0).floor() as i32) & 1;
                let bz_off = if by == 0 { 0.0 } else { 2.0 };
                if ((v.z + bz_off) % 4.0) > 3.5 || (v.y % 4.0) > 3.5 {
                    return Some((MAT_DARK_STONE, [0; 3])); // mortar
                }
                return Some((MAT_BRICK, [0; 3]));
            }
            // Right wall (x > 93): stone with window openings
            if v.x > 93.0 {
                let wz = (v.z - 10.0) % 20.0;
                let is_window = wz > 4.0 && wz < 14.0;
                let is_lower_window = is_window && v.y > 16.0 && v.y < 36.0;
                let is_upper_window = is_window && v.y > 54.0 && v.y < 74.0;
                if is_lower_window || is_upper_window {
                    return None;
                }
                return Some((MAT_STONE, [0; 3]));
            }
            // Front wall (z < 3) and back wall (z > 117)
            if v.z < 3.0 || v.z > 117.0 {
                // Entrance archway in front wall
                if v.z < 3.0 && v.x > 38.0 && v.x < 58.0 && v.y < 40.0 {
                    let arch_cx = 48.0;
                    let arch_top = 40.0;
                    let arch_r = 10.0;
                    if v.y > arch_top - arch_r {
                        let dy = v.y - (arch_top - arch_r);
                        let dx = (v.x - arch_cx).abs();
                        if dx * dx + dy * dy < arch_r * arch_r {
                            return None; // arch opening
                        }
                    } else {
                        return None; // rectangular part of entrance
                    }
                }
                return Some((MAT_STONE, [0; 3]));
            }
        }

        // === COLUMNS: two rows (x=24, x=72) every 20z, with base and capital ===
        for &cx in &[24.0_f32, 72.0] {
            let mut cz = 15.0;
            while cz < 110.0 {
                let dx = v.x - cx;
                let dz = v.z - cz;
                let dist_sq = dx * dx + dz * dz;

                if v.y < 88.0 {
                    // Column base (y 2-5): wider, radius 6
                    if v.y < 5.0 && v.y >= 2.0 && dist_sq < 36.0 {
                        return Some((MAT_DARK_STONE, [0; 3]));
                    }
                    // Column shaft (y 5-80): radius 4
                    if v.y >= 5.0 && v.y < 80.0 && dist_sq < 16.0 {
                        return Some((MAT_STONE, [0; 3]));
                    }
                    // Column capital (y 80-88): wider, radius 6
                    if v.y >= 80.0 && dist_sq < 36.0 {
                        return Some((MAT_DARK_STONE, [0; 3]));
                    }
                }
                cz += 20.0;
            }
        }

        // === ARCHES between columns ===
        for &cx in &[24.0_f32, 72.0] {
            let mut cz = 15.0;
            while cz + 20.0 < 110.0 {
                let mid_z = cz + 10.0;
                if v.x > cx - 5.0 && v.x < cx + 5.0 && v.z > cz && v.z < cz + 20.0 {
                    if v.y > 72.0 && v.y < 82.0 {
                        let arch_r = 10.0;
                        let dy = v.y - 72.0;
                        let dz = (v.z - mid_z).abs();
                        if dy * dy + dz * dz > arch_r * arch_r {
                            return Some((MAT_STONE, [0; 3]));
                        }
                    }
                }
                cz += 20.0;
            }
        }

        // === CLOTH BANNERS: red and green, hanging from second floor ===
        if (v.z - 35.0).abs() < 0.6 && v.x > 36.0 && v.x < 44.0 && v.y > 14.0 && v.y < 44.0 {
            return Some((MAT_RED_CLOTH, [0; 3]));
        }
        if (v.z - 55.0).abs() < 0.6 && v.x > 36.0 && v.x < 44.0 && v.y > 14.0 && v.y < 44.0 {
            return Some((MAT_GREEN_CLOTH, [0; 3]));
        }
        // Mirror side
        if (v.z - 65.0).abs() < 0.6 && v.x > 52.0 && v.x < 60.0 && v.y > 14.0 && v.y < 44.0 {
            return Some((MAT_RED_CLOTH, [0; 3]));
        }
        if (v.z - 85.0).abs() < 0.6 && v.x > 52.0 && v.x < 60.0 && v.y > 14.0 && v.y < 44.0 {
            return Some((MAT_GREEN_CLOTH, [0; 3]));
        }

        // === FOUNTAIN at center-back ===
        {
            let dx = v.x - 48.0;
            let dz = v.z - 100.0;
            let dist_sq = dx * dx + dz * dz;
            // Basin (wide, shallow)
            if v.y < 6.0 && dist_sq < 64.0 && (v.y < 3.0 || dist_sq > 49.0) {
                return Some((MAT_DARK_STONE, [0; 3]));
            }
            // Central pedestal
            if v.y < 20.0 && dist_sq < 4.0 {
                return Some((MAT_STONE, [0; 3]));
            }
            // Upper basin
            if v.y > 18.0 && v.y < 22.0 && dist_sq < 16.0 && (v.y < 19.0 || dist_sq > 9.0) {
                return Some((MAT_DARK_STONE, [0; 3]));
            }
            // Water (emissive blue tint)
            if v.y > 3.0 && v.y < 5.0 && dist_sq < 49.0 {
                return Some((MAT_BLUE, [30, 50, 80]));
            }
        }

        // === EMISSIVE WALL SCONCES ===
        // Left wall sconces (x ≈ 4): every 20 z-units, at y=24 and y=64
        {
            let mut cz = 15.0;
            while cz < 110.0 {
                for &cy in &[24.0_f32, 64.0] {
                    let d = (p - Vec3::new(offset.x + 4.0, cy, offset.z + cz)).length();
                    if d < 1.8 {
                        return Some((MAT_STONE, [255, 180, 80]));
                    }
                }
                cz += 20.0;
            }
        }
        // Right wall sconces (x ≈ 92)
        {
            let mut cz = 15.0;
            while cz < 110.0 {
                for &cy in &[24.0_f32, 64.0] {
                    let d = (p - Vec3::new(offset.x + 92.0, cy, offset.z + cz)).length();
                    if d < 1.8 {
                        return Some((MAT_STONE, [255, 180, 80]));
                    }
                }
                cz += 20.0;
            }
        }
        // Chandelier: central emissive cluster
        {
            let d = (p - Vec3::new(offset.x + 48.0, 82.0, offset.z + 60.0)).length();
            if d < 3.5 {
                return Some((MAT_STONE, [255, 220, 140]));
            }
        }

        // === WOOD BENCHES between columns (ground floor only) ===
        for &cx in &[24.0_f32, 72.0] {
            let bench_x_near = if cx < 48.0 { cx + 8.0 } else { cx - 10.0 };
            let mut cz = 25.0;
            while cz < 100.0 {
                if v.x > bench_x_near
                    && v.x < bench_x_near + 4.0
                    && v.z > cz - 2.0
                    && v.z < cz + 2.0
                    && v.y >= 2.0
                    && v.y < 6.0
                {
                    return Some((MAT_WOOD, [0; 3]));
                }
                cz += 20.0;
            }
        }

        // === BLUE SPHERE (decorative) ===
        {
            let d = (p - Vec3::new(88.0, 58.0, 34.0)).length();
            if d < 6.0 {
                return Some((MAT_BLUE, [0; 3]));
            }
        }

        // === BACK WALL decorative bumps ===
        if v.z > 115.0 && v.z < 118.0 {
            let mut sx = 12.0;
            while sx < 90.0 {
                let mut sy = 12.0;
                while sy < 80.0 {
                    let dx = v.x - sx;
                    let dy = v.y - sy;
                    if dx * dx + dy * dy < 9.0 {
                        return Some((MAT_DARK_STONE, [0; 3]));
                    }
                    sy += 16.0;
                }
                sx += 16.0;
            }
        }

        // === STEPS at entrance (z < 6) ===
        if v.z < 6.0 && v.x > 36.0 && v.x < 60.0 {
            let step = ((6.0 - v.z) / 2.0).floor();
            if v.y < 2.0 - step * 1.0 && v.y >= -step * 1.0 {
                return Some((MAT_STONE, [0; 3]));
            }
        }

        None
    }
}

impl VoxelGenerator for SponzaGenerator {
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
                    if let Some((material, emissive)) = Self::eval_voxel(world) {
                        data.set_voxel(lx, ly, lz, VoxelCell::new(material, 1, emissive));
                        any_solid = true;
                    }
                }
            }
        }
        if any_solid { Some(data) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::ucvh::UcvhConfig;

    #[test]
    fn floor_is_solid() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(64.0, 0.5, 64.0));
        assert!(result.is_some(), "floor voxel should be solid");
        let (mat, _) = result.unwrap();
        assert!(
            mat == MAT_STONE || mat == MAT_DARK_STONE,
            "floor should be stone or dark_stone"
        );
    }

    #[test]
    fn floor_checkerboard_alternates() {
        let a = SponzaGenerator::eval_voxel(Vec3::new(20.5, 0.5, 8.5));
        let b = SponzaGenerator::eval_voxel(Vec3::new(28.5, 0.5, 8.5));
        assert!(a.is_some() && b.is_some());
        assert_ne!(
            a.unwrap().0,
            b.unwrap().0,
            "adjacent floor tiles should differ"
        );
    }

    #[test]
    fn center_air_is_empty() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(64.0, 30.0, 64.0));
        assert!(result.is_none(), "center should be air");
    }

    #[test]
    fn wall_sconce_is_emissive() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(20.0, 24.0, 19.0));
        assert!(result.is_some(), "wall sconce position should be solid");
        assert!(result.unwrap().1[0] > 0, "sconce should have emissive");
    }

    #[test]
    fn chandelier_is_emissive() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(64.0, 82.0, 64.0));
        assert!(result.is_some(), "chandelier should be solid");
        assert!(result.unwrap().1[0] > 0, "chandelier should have emissive");
    }

    #[test]
    fn red_cloth_has_correct_material() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(56.0, 30.0, 39.4));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, MAT_RED_CLOTH);
    }

    #[test]
    fn second_floor_is_wood() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(20.0, 45.0, 10.0));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, MAT_WOOD);
    }

    #[test]
    fn side_wall_windows_are_thin_openings_with_solid_frames() {
        assert!(
            SponzaGenerator::eval_voxel(Vec3::new(16.5, 20.5, 18.5)).is_none(),
            "left lower window should be open through the wall"
        );
        assert!(
            SponzaGenerator::eval_voxel(Vec3::new(18.5, 35.5, 27.5)).is_none(),
            "left lower window should stay open through all three wall voxels"
        );
        assert!(
            SponzaGenerator::eval_voxel(Vec3::new(109.5, 60.5, 38.5)).is_none(),
            "right upper window should be open through the wall"
        );

        assert!(
            SponzaGenerator::eval_voxel(Vec3::new(16.5, 20.5, 17.5)).is_some(),
            "z-adjacent window frame voxel should be solid"
        );
        assert!(
            SponzaGenerator::eval_voxel(Vec3::new(16.5, 48.5, 18.5)).is_some(),
            "y-adjacent wall band between window rows should be solid"
        );
    }

    #[test]
    fn sponza_generates_bricks() {
        let config = UcvhConfig::new(UVec3::splat(128));
        let generator = SponzaGenerator;
        let data = generator.generate_brick(UVec3::new(2, 0, 8), &config);
        assert!(data.is_some(), "floor brick should be non-empty");
    }
}
