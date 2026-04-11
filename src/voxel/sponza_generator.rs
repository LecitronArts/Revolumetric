use glam::{UVec3, Vec3};
use crate::voxel::brick::{BrickData, VoxelCell, BRICK_EDGE};
use crate::voxel::ucvh::UcvhConfig;
use crate::voxel::generator::VoxelGenerator;

/// Material IDs matching the shader-side MATERIAL_ALBEDO LUT.
const MAT_STONE: u16 = 1;
const MAT_RED_CLOTH: u16 = 2;
const MAT_GREEN_CLOTH: u16 = 3;
const MAT_BLUE: u16 = 4;
const MAT_BRICK: u16 = 5;

/// Sponza-inspired architectural scene for 128³ world.
pub struct SponzaGenerator;

impl SponzaGenerator {
    fn eval_voxel(p: Vec3) -> Option<(u16, [u8; 3])> {
        let offset = Vec3::new(16.0, 0.0, 4.0);
        let v = p - offset;

        if v.x < 0.0 || v.x > 96.0 || v.y < 0.0 || v.y > 96.0 || v.z < 0.0 || v.z > 120.0 {
            return None;
        }

        if v.y < 2.0 {
            let cx = ((v.x / 4.0).floor() as i32) & 1;
            let cz = ((v.z / 4.0).floor() as i32) & 1;
            if cx ^ cz == 0 { return Some((MAT_STONE, [0; 3])); }
            return Some((MAT_STONE, [0; 3]));
        }

        if v.y > 88.0 && v.y < 92.0 { return Some((MAT_STONE, [0; 3])); }

        let wall_inner = v.x > 2.0 && v.x < 94.0 && v.z > 2.0 && v.z < 118.0;
        if !wall_inner && v.y < 92.0 {
            if v.x < 2.0 || v.x > 94.0 {
                if v.x < 2.0 {
                    let by = ((v.y / 4.0).floor() as i32) & 1;
                    let bz_off = if by == 0 { 0.0 } else { 2.0 };
                    if ((v.z + bz_off) % 4.0) > 3.0 || (v.y % 4.0) > 3.0 { return Some((MAT_STONE, [0; 3])); }
                    return Some((MAT_BRICK, [0; 3]));
                }
                return Some((MAT_STONE, [0; 3]));
            }
            if v.z < 2.0 || v.z > 118.0 { return Some((MAT_STONE, [0; 3])); }
        }

        for &cx in &[24.0_f32, 72.0] {
            let mut cz = 18.0;
            while cz < 110.0 {
                let dx = v.x - cx; let dz = v.z - cz;
                if dx * dx + dz * dz < 25.0 && v.y < 88.0 { return Some((MAT_STONE, [0; 3])); }
                cz += 24.0;
            }
        }

        for &cx in &[24.0_f32, 72.0] {
            let mut cz = 18.0;
            while cz + 24.0 < 110.0 {
                let mid_z = cz + 12.0;
                if v.x > cx - 6.0 && v.x < cx + 6.0 && v.z > cz - 1.0 && v.z < cz + 25.0 {
                    if v.y > 75.0 && v.y < 85.0 {
                        let arch_r = 12.0;
                        let dy = v.y - 75.0; let dz = (v.z - mid_z).abs();
                        if dy * dy + dz * dz > arch_r * arch_r { return Some((MAT_STONE, [0; 3])); }
                    }
                }
                cz += 24.0;
            }
        }

        if v.x < 48.0 && v.y > 46.0 && v.y < 48.0 { return Some((MAT_STONE, [0; 3])); }

        if (v.z - 40.0).abs() < 0.6 && v.x > 30.0 && v.x < 36.0 && v.y > 20.0 && v.y < 70.0 {
            return Some((MAT_RED_CLOTH, [0; 3]));
        }
        if (v.z - 72.0).abs() < 0.6 && v.x > 30.0 && v.x < 36.0 && v.y > 20.0 && v.y < 70.0 {
            return Some((MAT_GREEN_CLOTH, [0; 3]));
        }

        { let dx = v.x - 48.0; let dz = v.z - 100.0; let dist_sq = dx * dx + dz * dz;
          if v.y < 8.0 && dist_sq < 64.0 && (v.y < 3.0 || dist_sq > 36.0) { return Some((MAT_STONE, [0; 3])); }
          if v.y < 14.0 && dist_sq < 4.0 { return Some((MAT_STONE, [0; 3])); }
        }

        { let d = (p - Vec3::new(64.0, 20.0, 104.0)).length(); if d < 2.5 { return Some((MAT_STONE, [255, 150, 50])); } }
        { let d = (p - Vec3::new(16.5, 30.0, 64.0)).length(); if d < 2.0 { return Some((MAT_STONE, [200, 140, 60])); } }
        { let d = (p - Vec3::new(110.5, 30.0, 64.0)).length(); if d < 2.0 { return Some((MAT_STONE, [200, 140, 60])); } }

        { let d = (p - Vec3::new(88.0, 58.0, 34.0)).length(); if d < 8.0 { return Some((MAT_BLUE, [0; 3])); } }

        if v.z > 116.0 && v.z < 118.0 {
            let mut sx = 12.0;
            while sx < 90.0 {
                let mut sy = 12.0;
                while sy < 80.0 {
                    let dx = v.x - sx; let dy = v.y - sy;
                    if dx * dx + dy * dy < 9.0 { return Some((MAT_STONE, [0; 3])); }
                    sy += 16.0;
                }
                sx += 16.0;
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
                    let world = Vec3::new((base.x + lx) as f32 + 0.5, (base.y + ly) as f32 + 0.5, (base.z + lz) as f32 + 0.5);
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
        assert_eq!(result.unwrap().0, MAT_STONE);
    }

    #[test]
    fn center_air_is_empty() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(64.0, 40.0, 64.0));
        assert!(result.is_none(), "center should be air");
    }

    #[test]
    fn lamp_is_emissive() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(64.0, 20.0, 104.0));
        assert!(result.is_some());
        assert!(result.unwrap().1[0] > 0, "lamp should have emissive");
    }

    #[test]
    fn red_cloth_has_correct_material() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(49.0, 40.0, 44.0));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, MAT_RED_CLOTH);
    }

    #[test]
    fn sponza_generates_bricks() {
        let config = UcvhConfig::new(UVec3::splat(128));
        let generator = SponzaGenerator;
        let data = generator.generate_brick(UVec3::new(2, 0, 8), &config);
        assert!(data.is_some(), "floor brick should be non-empty");
    }
}
