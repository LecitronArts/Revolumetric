// src/voxel/generator.rs
use glam::{UVec3, Vec3};
use crate::voxel::brick::{BrickData, VoxelCell, BRICK_EDGE};
use crate::voxel::ucvh::{Ucvh, UcvhConfig};

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
                        data.set_voxel(lx, ly, lz, VoxelCell {
                            material: self.material,
                            flags: 1, // solid
                            emissive: [0; 3],
                            _pad: 0,
                        });
                        any_solid = true;
                    }
                }
            }
        }

        if any_solid { Some(data) } else { None }
    }
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
}
