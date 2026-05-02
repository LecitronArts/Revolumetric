use bytemuck::{Pod, Zeroable};
use std::collections::BTreeMap;

use crate::voxel::brick::BRICK_VOLUME;
use crate::voxel::ucvh::Ucvh;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestirDiDebugView {
    Off,
    ReservoirWeight,
    LightId,
    Visibility,
    TemporalValid,
    SpatialNeighbors,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestirDiSettings {
    pub enabled: bool,
    pub temporal_enabled: bool,
    pub spatial_enabled: bool,
    pub initial_candidate_count: u32,
    pub spatial_sample_count: u32,
    pub history_length: u32,
    pub debug_view: RestirDiDebugView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestirDiParseWarning {
    pub variable: &'static str,
    pub expected: &'static str,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestirDiSettingsParse {
    pub settings: RestirDiSettings,
    pub warnings: Vec<RestirDiParseWarning>,
}

impl Default for RestirDiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            temporal_enabled: true,
            spatial_enabled: true,
            initial_candidate_count: 1,
            spatial_sample_count: 4,
            history_length: 20,
            debug_view: RestirDiDebugView::Off,
        }
    }
}

impl RestirDiSettings {
    pub fn from_env() -> RestirDiSettingsParse {
        Self::from_values(
            std::env::var("REVOLUMETRIC_VPT_RESTIR_DI").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_TEMPORAL")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_SPATIAL")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_INITIAL_CANDIDATES")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_HISTORY_LENGTH")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_DEBUG")
                .ok()
                .as_deref(),
        )
    }

    pub fn from_values(
        enabled: Option<&str>,
        temporal: Option<&str>,
        spatial: Option<&str>,
        initial_candidates: Option<&str>,
        spatial_samples: Option<&str>,
        history_length: Option<&str>,
        debug_view: Option<&str>,
    ) -> RestirDiSettingsParse {
        let mut settings = Self::default();
        let mut warnings = Vec::new();

        parse_bool(
            "REVOLUMETRIC_VPT_RESTIR_DI",
            enabled,
            &mut settings.enabled,
            &mut warnings,
        );
        parse_bool(
            "REVOLUMETRIC_RESTIR_DI_TEMPORAL",
            temporal,
            &mut settings.temporal_enabled,
            &mut warnings,
        );
        parse_bool(
            "REVOLUMETRIC_RESTIR_DI_SPATIAL",
            spatial,
            &mut settings.spatial_enabled,
            &mut warnings,
        );
        parse_u32_range(
            "REVOLUMETRIC_RESTIR_DI_INITIAL_CANDIDATES",
            initial_candidates,
            1,
            16,
            &mut settings.initial_candidate_count,
            &mut warnings,
        );
        parse_u32_range(
            "REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES",
            spatial_samples,
            0,
            8,
            &mut settings.spatial_sample_count,
            &mut warnings,
        );
        parse_u32_range(
            "REVOLUMETRIC_RESTIR_DI_HISTORY_LENGTH",
            history_length,
            1,
            64,
            &mut settings.history_length,
            &mut warnings,
        );
        parse_debug_view(debug_view, &mut settings.debug_view, &mut warnings);

        RestirDiSettingsParse { settings, warnings }
    }

    pub fn gpu_uniforms(
        self,
        frame_index: u32,
        reservoir_count: u32,
        light_count: u32,
        width: u32,
        height: u32,
    ) -> GpuRestirDiUniforms {
        GpuRestirDiUniforms {
            enabled: self.enabled as u32,
            temporal_enabled: self.temporal_enabled as u32,
            spatial_enabled: self.spatial_enabled as u32,
            debug_view: self.debug_view.as_gpu_value(),
            initial_candidate_count: self.initial_candidate_count,
            spatial_sample_count: self.spatial_sample_count,
            history_length: self.history_length,
            frame_index,
            reservoir_count,
            light_count,
            width,
            height,
        }
    }
}

impl RestirDiDebugView {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::ReservoirWeight => 1,
            Self::LightId => 2,
            Self::Visibility => 3,
            Self::TemporalValid => 4,
            Self::SpatialNeighbors => 5,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRestirDiUniforms {
    pub enabled: u32,
    pub temporal_enabled: u32,
    pub spatial_enabled: u32,
    pub debug_view: u32,
    pub initial_candidate_count: u32,
    pub spatial_sample_count: u32,
    pub history_length: u32,
    pub frame_index: u32,
    pub reservoir_count: u32,
    pub light_count: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuDirectLight {
    pub position_radius: [f32; 4],
    pub normal_type: [f32; 4],
    pub color_power: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRestirDiReservoir {
    pub sample_light_id: u32,
    pub sample_flags: u32,
    pub sample_count_m: u32,
    pub pad0: u32,
    pub target_pdf: f32,
    pub weight_sum: f32,
    pub selected_weight: f32,
    pub confidence: f32,
    pub sample_position_pdf: [f32; 4],
    pub sample_radiance: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct EmissiveVoxelForTest {
    pub brick_id: u32,
    pub world_position: [f32; 3],
    pub emissive: [u8; 3],
}

pub fn build_direct_lights_for_test(
    sun_direction: [f32; 3],
    sun_intensity: f32,
    emissive_voxels: &[EmissiveVoxelForTest],
) -> Vec<GpuDirectLight> {
    build_direct_lights_from_emissive_iter(
        sun_direction,
        sun_intensity,
        emissive_voxels.iter().copied(),
    )
}

pub fn build_direct_lights_from_ucvh(
    ucvh: &Ucvh,
    sun_direction: [f32; 3],
    sun_intensity: f32,
    max_lights: usize,
) -> Vec<GpuDirectLight> {
    let mut emissive_voxels = Vec::new();
    let grid = ucvh.config.brick_grid_size;
    let materials = ucvh.pool.material_pool();

    for z in 0..grid.z {
        for y in 0..grid.y {
            for x in 0..grid.x {
                let brick_pos = glam::UVec3::new(x, y, z);
                let node = ucvh.hierarchy.get_l0(brick_pos);
                if node.flags & 1 == 0 || node.brick_id == u32::MAX {
                    continue;
                }

                let base = node.brick_id as usize * BRICK_VOLUME;
                for morton_index in 0..BRICK_VOLUME {
                    let cell = materials[base + morton_index];
                    if cell.emissive == [0; 3] {
                        continue;
                    }

                    let (lx, ly, lz) = crate::voxel::morton::decode(morton_index as u32);
                    let world =
                        brick_pos * crate::voxel::brick::BRICK_EDGE + glam::UVec3::new(lx, ly, lz);
                    emissive_voxels.push(EmissiveVoxelForTest {
                        brick_id: node.brick_id,
                        world_position: [world.x as f32, world.y as f32, world.z as f32],
                        emissive: cell.emissive,
                    });
                }
            }
        }
    }

    let mut lights =
        build_direct_lights_from_emissive_iter(sun_direction, sun_intensity, emissive_voxels);
    lights.sort_by(|a, b| b.color_power[3].total_cmp(&a.color_power[3]));
    lights.truncate(max_lights);
    lights
}

fn build_direct_lights_from_emissive_iter(
    sun_direction: [f32; 3],
    sun_intensity: f32,
    emissive_voxels: impl IntoIterator<Item = EmissiveVoxelForTest>,
) -> Vec<GpuDirectLight> {
    let mut lights = Vec::new();
    if sun_intensity > 0.0 {
        lights.push(GpuDirectLight {
            position_radius: [sun_direction[0], sun_direction[1], sun_direction[2], 0.0],
            normal_type: [sun_direction[0], sun_direction[1], sun_direction[2], 0.0],
            color_power: [sun_intensity, sun_intensity, sun_intensity, sun_intensity],
        });
    }

    #[derive(Clone, Copy)]
    struct Cluster {
        position_sum: [f32; 3],
        radiance_sum: [f32; 3],
        count: u32,
    }

    let mut clusters: BTreeMap<u32, Cluster> = BTreeMap::new();
    for voxel in emissive_voxels {
        let radiance = [
            f32::from(voxel.emissive[0]) / 255.0 * 3.0,
            f32::from(voxel.emissive[1]) / 255.0 * 3.0,
            f32::from(voxel.emissive[2]) / 255.0 * 3.0,
        ];
        if radiance[0] + radiance[1] + radiance[2] <= 0.0 {
            continue;
        }
        clusters
            .entry(voxel.brick_id)
            .and_modify(|cluster| {
                for (i, value) in radiance.iter().enumerate() {
                    cluster.position_sum[i] += voxel.world_position[i];
                    cluster.radiance_sum[i] += *value;
                }
                cluster.count += 1;
            })
            .or_insert(Cluster {
                position_sum: voxel.world_position,
                radiance_sum: radiance,
                count: 1,
            });
    }

    for cluster in clusters.values() {
        let inv_count = 1.0 / cluster.count as f32;
        let power = cluster.radiance_sum[0] + cluster.radiance_sum[1] + cluster.radiance_sum[2];
        lights.push(GpuDirectLight {
            position_radius: [
                cluster.position_sum[0] * inv_count,
                cluster.position_sum[1] * inv_count,
                cluster.position_sum[2] * inv_count,
                1.0,
            ],
            normal_type: [0.0, 0.0, 0.0, 1.0],
            color_power: [
                cluster.radiance_sum[0],
                cluster.radiance_sum[1],
                cluster.radiance_sum[2],
                power,
            ],
        });
    }

    lights
}

fn parse_bool(
    variable: &'static str,
    value: Option<&str>,
    target: &mut bool,
    warnings: &mut Vec<RestirDiParseWarning>,
) {
    let Some(value) = value else {
        return;
    };

    if value == "1"
        || value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
    {
        *target = true;
    } else if value == "0"
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("no")
    {
        *target = false;
    } else {
        warnings.push(RestirDiParseWarning {
            variable,
            expected: "on|off|1|0|true|false|yes|no",
            value: value.to_string(),
        });
    }
}

fn parse_u32_range(
    variable: &'static str,
    value: Option<&str>,
    min: u32,
    max: u32,
    target: &mut u32,
    warnings: &mut Vec<RestirDiParseWarning>,
) {
    let Some(value) = value else {
        return;
    };

    match value
        .parse::<u32>()
        .ok()
        .filter(|parsed| (min..=max).contains(parsed))
    {
        Some(parsed) => *target = parsed,
        None => warnings.push(RestirDiParseWarning {
            variable,
            expected: "integer in configured range",
            value: value.to_string(),
        }),
    }
}

fn parse_debug_view(
    value: Option<&str>,
    target: &mut RestirDiDebugView,
    warnings: &mut Vec<RestirDiParseWarning>,
) {
    let Some(value) = value else {
        return;
    };

    if value.eq_ignore_ascii_case("off") {
        *target = RestirDiDebugView::Off;
    } else if value.eq_ignore_ascii_case("reservoir_weight") {
        *target = RestirDiDebugView::ReservoirWeight;
    } else if value.eq_ignore_ascii_case("light_id") {
        *target = RestirDiDebugView::LightId;
    } else if value.eq_ignore_ascii_case("visibility") {
        *target = RestirDiDebugView::Visibility;
    } else if value.eq_ignore_ascii_case("temporal_valid") {
        *target = RestirDiDebugView::TemporalValid;
    } else if value.eq_ignore_ascii_case("spatial_neighbors") {
        *target = RestirDiDebugView::SpatialNeighbors;
    } else {
        warnings.push(RestirDiParseWarning {
            variable: "REVOLUMETRIC_RESTIR_DI_DEBUG",
            expected: "off|reservoir_weight|light_id|visibility|temporal_valid|spatial_neighbors",
            value: value.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_defaults_keep_restir_disabled() {
        let settings = RestirDiSettings::default();
        assert!(!settings.enabled);
        assert!(settings.temporal_enabled);
        assert!(settings.spatial_enabled);
        assert_eq!(settings.initial_candidate_count, 1);
        assert_eq!(settings.spatial_sample_count, 4);
        assert_eq!(settings.history_length, 20);
        assert_eq!(settings.debug_view, RestirDiDebugView::Off);
    }

    #[test]
    fn settings_parse_valid_overrides() {
        let parsed = RestirDiSettings::from_values(
            Some("on"),
            Some("off"),
            Some("true"),
            Some("8"),
            Some("3"),
            Some("32"),
            Some("reservoir_weight"),
        );
        assert!(parsed.settings.enabled);
        assert!(!parsed.settings.temporal_enabled);
        assert!(parsed.settings.spatial_enabled);
        assert_eq!(parsed.settings.initial_candidate_count, 8);
        assert_eq!(parsed.settings.spatial_sample_count, 3);
        assert_eq!(parsed.settings.history_length, 32);
        assert_eq!(
            parsed.settings.debug_view,
            RestirDiDebugView::ReservoirWeight
        );
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn settings_reject_invalid_values_without_changing_defaults() {
        let parsed = RestirDiSettings::from_values(
            Some("maybe"),
            Some("later"),
            Some("sometimes"),
            Some("0"),
            Some("99"),
            Some("0"),
            Some("heatmap"),
        );
        assert_eq!(parsed.settings, RestirDiSettings::default());
        assert_eq!(parsed.warnings.len(), 7);
    }

    #[test]
    fn gpu_struct_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuRestirDiUniforms>(), 48);
        assert_eq!(std::mem::size_of::<GpuDirectLight>(), 48);
        assert_eq!(std::mem::size_of::<GpuRestirDiReservoir>(), 64);
        assert_eq!(std::mem::offset_of!(GpuRestirDiUniforms, enabled), 0);
        assert_eq!(std::mem::offset_of!(GpuRestirDiUniforms, light_count), 36);
        assert_eq!(std::mem::offset_of!(GpuDirectLight, color_power), 32);
        assert_eq!(
            std::mem::offset_of!(GpuRestirDiReservoir, sample_position_pdf),
            32
        );
    }

    #[test]
    fn slang_restir_di_common_declares_matching_abi() {
        let source = std::fs::read_to_string("assets/shaders/shared/restir_di_common.slang")
            .expect("restir_di_common.slang should be readable");
        assert!(source.contains("struct RestirDiUniforms"));
        assert!(source.contains("uint enabled;"));
        assert!(source.contains("uint light_count;"));
        assert!(source.contains("struct DirectLight"));
        assert!(source.contains("float4 color_power;"));
        assert!(source.contains("struct RestirDiReservoir"));
        assert!(source.contains("uint sample_count_m;"));
        assert!(source.contains("float4 sample_radiance;"));
    }

    #[test]
    fn direct_light_table_includes_sun_when_power_is_positive() {
        let lights = build_direct_lights_for_test([0.0, -1.0, 0.0], 2.0, &[]);
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0].normal_type[3], 0.0);
        assert!(lights[0].color_power[3] > 0.0);
    }

    #[test]
    fn direct_light_table_clusters_emissive_voxels_by_brick() {
        let voxels = [
            EmissiveVoxelForTest {
                brick_id: 7,
                world_position: [1.0, 2.0, 3.0],
                emissive: [255, 128, 0],
            },
            EmissiveVoxelForTest {
                brick_id: 7,
                world_position: [3.0, 2.0, 1.0],
                emissive: [255, 128, 0],
            },
            EmissiveVoxelForTest {
                brick_id: 8,
                world_position: [8.0, 0.0, 0.0],
                emissive: [0, 0, 0],
            },
        ];
        let lights = build_direct_lights_for_test([0.0, -1.0, 0.0], 0.0, &voxels);
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0].normal_type[3], 1.0);
        assert_eq!(lights[0].position_radius[0], 2.0);
        assert_eq!(lights[0].position_radius[1], 2.0);
        assert_eq!(lights[0].position_radius[2], 2.0);
    }

    #[test]
    fn direct_light_table_can_be_built_from_ucvh_sponza_emissives() {
        let mut ucvh = crate::voxel::ucvh::Ucvh::new(crate::voxel::ucvh::UcvhConfig::new(
            glam::UVec3::splat(128),
        ));
        let count = crate::voxel::generator::generate_sponza_scene(&mut ucvh);
        assert!(count > 0, "sponza should allocate bricks");
        ucvh.rebuild_hierarchy();
        let lights = build_direct_lights_from_ucvh(&ucvh, [0.0, -1.0, 0.0], 0.0, 4096);
        assert!(
            lights
                .iter()
                .any(|light| light.normal_type[3] == 1.0 && light.color_power[3] > 0.0),
            "sponza emissive voxels should produce at least one emissive direct-light cluster"
        );
    }
}
