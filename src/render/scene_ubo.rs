use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

pub const LIGHTING_FLAG_SHADOWS_ENABLED: u32 = 1 << 0;
pub const LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS: u32 = 1 << 1;
pub const LIGHTING_FLAG_VCT_ENABLED: u32 = 1 << 2;
pub const LIGHTING_DEBUG_VIEW_SHIFT: u32 = 28;
pub const LIGHTING_DEBUG_VIEW_MASK: u32 = 0xF << LIGHTING_DEBUG_VIEW_SHIFT;
pub const LIGHTING_DEBUG_VIEW_FINAL: u32 = 0;
pub const LIGHTING_DEBUG_VIEW_DIRECT_DIFFUSE: u32 = 1;
pub const LIGHTING_DEBUG_VIEW_NORMAL: u32 = 2;
pub const RENDER_MODE_VCT: u32 = 0;
pub const RENDER_MODE_VPT: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightingDebugView {
    Final,
    DirectDiffuse,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Vct,
    Vpt,
}

impl RenderMode {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Vct => RENDER_MODE_VCT,
            Self::Vpt => RENDER_MODE_VPT,
        }
    }
}

impl LightingDebugView {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Final => LIGHTING_DEBUG_VIEW_FINAL,
            Self::DirectDiffuse => LIGHTING_DEBUG_VIEW_DIRECT_DIFFUSE,
            Self::Normal => LIGHTING_DEBUG_VIEW_NORMAL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightingSettings {
    pub shadows_enabled: bool,
    pub skip_backface_shadows: bool,
    pub vct_enabled: bool,
    pub render_mode: RenderMode,
    pub vpt_max_bounces: u32,
    pub debug_view: LightingDebugView,
    pub exposure: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightingSettingsParseWarning {
    pub variable: &'static str,
    pub value: String,
    pub expected: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LightingSettingsParseResult {
    pub settings: LightingSettings,
    pub warnings: Vec<LightingSettingsParseWarning>,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            shadows_enabled: true,
            skip_backface_shadows: false,
            vct_enabled: true,
            render_mode: RenderMode::Vct,
            vpt_max_bounces: 2,
            debug_view: LightingDebugView::Final,
            exposure: 1.0,
        }
    }
}

impl LightingSettings {
    pub fn from_env() -> Self {
        Self::from_env_report().settings
    }

    pub fn from_env_report() -> LightingSettingsParseResult {
        let shadows = std::env::var("REVOLUMETRIC_LIGHTING_SHADOWS").ok();
        let skip_backface = std::env::var("REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS").ok();
        let vct = std::env::var("REVOLUMETRIC_VCT").ok();
        let render_mode = std::env::var("REVOLUMETRIC_RENDER_MODE").ok();
        let vpt_max_bounces = std::env::var("REVOLUMETRIC_VPT_MAX_BOUNCES").ok();
        let debug_view = std::env::var("REVOLUMETRIC_LIGHTING_DEBUG_VIEW").ok();
        let exposure = std::env::var("REVOLUMETRIC_EXPOSURE").ok();
        Self::from_values_report(
            shadows.as_deref(),
            skip_backface.as_deref(),
            vct.as_deref(),
            render_mode.as_deref(),
            vpt_max_bounces.as_deref(),
            debug_view.as_deref(),
            exposure.as_deref(),
        )
    }

    pub fn from_values(
        shadows: Option<&str>,
        skip_backface_shadows: Option<&str>,
        vct: Option<&str>,
        render_mode: Option<&str>,
        vpt_max_bounces: Option<&str>,
        debug_view: Option<&str>,
        exposure: Option<&str>,
    ) -> Self {
        Self::from_values_report(
            shadows,
            skip_backface_shadows,
            vct,
            render_mode,
            vpt_max_bounces,
            debug_view,
            exposure,
        )
        .settings
    }

    pub fn from_values_report(
        shadows: Option<&str>,
        skip_backface_shadows: Option<&str>,
        vct: Option<&str>,
        render_mode: Option<&str>,
        vpt_max_bounces: Option<&str>,
        debug_view: Option<&str>,
        exposure: Option<&str>,
    ) -> LightingSettingsParseResult {
        let mut settings = Self::default();
        let mut warnings = Vec::new();

        apply_optional_override(
            &mut settings.shadows_enabled,
            shadows,
            "REVOLUMETRIC_LIGHTING_SHADOWS",
            "on|off|1|0|true|false",
            parse_bool_value,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.skip_backface_shadows,
            skip_backface_shadows,
            "REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS",
            "on|off|1|0|true|false",
            parse_bool_value,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.vct_enabled,
            vct,
            "REVOLUMETRIC_VCT",
            "on|off|1|0|true|false",
            parse_bool_value,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.render_mode,
            render_mode,
            "REVOLUMETRIC_RENDER_MODE",
            "vct|vpt",
            parse_render_mode,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.vpt_max_bounces,
            vpt_max_bounces,
            "REVOLUMETRIC_VPT_MAX_BOUNCES",
            "integer in 1..=8",
            parse_vpt_max_bounces,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.debug_view,
            debug_view,
            "REVOLUMETRIC_LIGHTING_DEBUG_VIEW",
            "final|off|diffuse|direct|normal",
            parse_lighting_debug_view,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.exposure,
            exposure,
            "REVOLUMETRIC_EXPOSURE",
            "finite non-negative float",
            parse_exposure,
            &mut warnings,
        );

        LightingSettingsParseResult { settings, warnings }
    }

    pub fn gpu_flags(self) -> u32 {
        let mut flags = 0;
        if self.shadows_enabled {
            flags |= LIGHTING_FLAG_SHADOWS_ENABLED;
        }
        if self.skip_backface_shadows {
            flags |= LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS;
        }
        if self.vct_enabled {
            flags |= LIGHTING_FLAG_VCT_ENABLED;
        }
        flags |= (self.debug_view.as_gpu_value() << LIGHTING_DEBUG_VIEW_SHIFT)
            & LIGHTING_DEBUG_VIEW_MASK;
        flags
    }
}

fn parse_bool_value(value: &str) -> Option<bool> {
    let value = value.trim();
    if value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("on") {
        Some(true)
    } else if value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
    {
        Some(false)
    } else {
        None
    }
}

fn parse_lighting_debug_view(value: &str) -> Option<LightingDebugView> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("final") || value.eq_ignore_ascii_case("off") {
        Some(LightingDebugView::Final)
    } else if value.eq_ignore_ascii_case("diffuse") || value.eq_ignore_ascii_case("direct") {
        Some(LightingDebugView::DirectDiffuse)
    } else if value.eq_ignore_ascii_case("normal") {
        Some(LightingDebugView::Normal)
    } else {
        None
    }
}

fn parse_render_mode(value: &str) -> Option<RenderMode> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("vct") {
        Some(RenderMode::Vct)
    } else if value.eq_ignore_ascii_case("vpt") {
        Some(RenderMode::Vpt)
    } else {
        None
    }
}

fn parse_vpt_max_bounces(value: &str) -> Option<u32> {
    let parsed = value.trim().parse::<u32>().ok()?;
    (1..=8).contains(&parsed).then_some(parsed)
}

fn parse_exposure(value: &str) -> Option<f32> {
    let parsed = value.trim().parse::<f32>().ok()?;
    (parsed.is_finite() && parsed >= 0.0).then_some(parsed)
}

fn apply_optional_override<T: Copy>(
    target: &mut T,
    raw_value: Option<&str>,
    variable: &'static str,
    expected: &'static str,
    parser: impl Fn(&str) -> Option<T>,
    warnings: &mut Vec<LightingSettingsParseWarning>,
) {
    let Some(raw_value) = raw_value else {
        return;
    };
    if let Some(parsed) = parser(raw_value) {
        *target = parsed;
    } else {
        warnings.push(LightingSettingsParseWarning {
            variable,
            value: raw_value.to_owned(),
            expected,
        });
    }
}

/// GPU-side scene uniforms. Must match Slang `SceneUniforms` in scene_common.slang exactly.
/// 176 bytes, std140-compatible (all float3 fields padded to 16-byte alignment).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSceneUniforms {
    pub pixel_to_ray: [[f32; 4]; 4], // 64B — col 0-2: direction matrix, col 3: camera origin
    pub resolution: [u32; 2],        // 8B
    pub _pad0: [u32; 2],             // 8B
    pub sun_direction: [f32; 3],     // 12B — normalized, world space, points TOWARD sun
    pub _pad1: f32,                  // 4B
    pub sun_intensity: [f32; 3],     // 12B — HDR color * intensity
    pub _pad2: f32,                  // 4B
    pub sky_color: [f32; 3],         // 12B — hemisphere ambient upper
    pub _pad3: f32,                  // 4B
    pub ground_color: [f32; 3],      // 12B — hemisphere ambient lower
    pub time: f32,                   // 4B
    pub lighting_flags: u32,         // 4B
    pub exposure: f32,               // 4B
    pub render_mode: u32,            // 4B
    pub vpt_sample_index: u32,       // 4B
    pub vpt_max_bounces: u32,        // 4B
    pub _pad4: [u32; 3],             // 12B
}

impl GpuSceneUniforms {
    pub fn apply_lighting_settings(&mut self, settings: LightingSettings) {
        self.lighting_flags = settings.gpu_flags();
        self.exposure = settings.exposure;
        self.render_mode = settings.render_mode.as_gpu_value();
        self.vpt_max_bounces = settings.vpt_max_bounces;
    }
}

pub struct SceneUniformInputs {
    pub pixel_to_ray: glam::Mat4,
    pub resolution: [u32; 2],
    pub sun_direction: glam::Vec3,
    pub sun_intensity: glam::Vec3,
    pub sky_color: [f32; 3],
    pub ground_color: [f32; 3],
    pub time: f32,
    pub lighting_settings: LightingSettings,
    pub vpt_sample_index: u32,
}

pub fn build_scene_uniforms(inputs: SceneUniformInputs) -> GpuSceneUniforms {
    let mut uniforms = GpuSceneUniforms {
        pixel_to_ray: inputs.pixel_to_ray.transpose().to_cols_array_2d(),
        resolution: inputs.resolution,
        _pad0: [0; 2],
        sun_direction: inputs.sun_direction.to_array(),
        _pad1: 0.0,
        sun_intensity: inputs.sun_intensity.to_array(),
        _pad2: 0.0,
        sky_color: inputs.sky_color,
        _pad3: 0.0,
        ground_color: inputs.ground_color,
        time: inputs.time,
        lighting_flags: 0,
        exposure: 1.0,
        render_mode: RENDER_MODE_VCT,
        vpt_sample_index: inputs.vpt_sample_index,
        vpt_max_bounces: 2,
        _pad4: [0; 3],
    };
    uniforms.apply_lighting_settings(inputs.lighting_settings);
    uniforms
}

/// Manages per-frame-slot uniform buffers for SceneUniforms.
/// One buffer per frame slot to prevent CPU/GPU write-after-read hazards.
pub struct SceneUniformBuffer {
    buffers: Vec<GpuBuffer>,
}

impl SceneUniformBuffer {
    /// Create N uniform buffers (one per frame slot).
    pub fn new(device: &ash::Device, allocator: &GpuAllocator, frame_count: usize) -> Result<Self> {
        let size = std::mem::size_of::<GpuSceneUniforms>() as vk::DeviceSize;
        let mut buffers = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            let buf = GpuBuffer::new(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                &format!("scene_ubo_frame_{i}"),
            )?;
            buffers.push(buf);
        }
        Ok(Self { buffers })
    }

    /// Write scene uniforms to the buffer for the given frame slot.
    pub fn update(&self, frame_slot: usize, data: &GpuSceneUniforms) {
        let buf = &self.buffers[frame_slot];
        if let Some(ptr) = buf.mapped_ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data as *const GpuSceneUniforms as *const u8,
                    ptr,
                    std::mem::size_of::<GpuSceneUniforms>(),
                );
            }
        }
    }

    /// Get the VkBuffer handle for a specific frame slot (for descriptor writes).
    pub fn buffer_handle(&self, frame_slot: usize) -> vk::Buffer {
        self.buffers[frame_slot].handle
    }

    /// Number of frame slots.
    pub fn frame_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        for buf in self.buffers {
            buf.destroy(device, allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_scene_uniforms_size_is_176_bytes() {
        assert_eq!(std::mem::size_of::<GpuSceneUniforms>(), 176);
    }

    #[test]
    fn gpu_scene_uniforms_offsets_match_slang_abi() {
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, pixel_to_ray), 0);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, resolution), 64);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, sun_direction), 80);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, ground_color), 128);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, lighting_flags), 144);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, exposure), 148);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, render_mode), 152);
        assert_eq!(
            std::mem::offset_of!(GpuSceneUniforms, vpt_sample_index),
            156
        );
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, vpt_max_bounces), 160);
    }

    #[test]
    fn gpu_scene_uniforms_is_zeroable() {
        let u = GpuSceneUniforms::zeroed();
        assert_eq!(u.resolution, [0, 0]);
        assert_eq!(u.time, 0.0);
    }

    #[test]
    fn lighting_settings_default_preserves_direct_lighting_path() {
        let settings = LightingSettings::default();

        assert!(settings.shadows_enabled);
        assert!(!settings.skip_backface_shadows);
        assert!(settings.vct_enabled);
        assert_eq!(settings.render_mode, RenderMode::Vct);
        assert_eq!(settings.vpt_max_bounces, 2);
        assert_eq!(settings.debug_view, LightingDebugView::Final);
        assert_eq!(settings.exposure, 1.0);
    }

    #[test]
    fn lighting_settings_can_disable_vct_explicitly() {
        let settings =
            LightingSettings::from_values(None, None, Some("off"), None, None, None, None);
        let mut uniforms = GpuSceneUniforms::zeroed();

        uniforms.apply_lighting_settings(settings);

        assert_eq!(uniforms.lighting_flags & LIGHTING_FLAG_VCT_ENABLED, 0);
    }

    #[test]
    fn lighting_settings_parse_vpt_render_mode_and_bounce_limit() {
        let settings =
            LightingSettings::from_values(None, None, None, Some("vpt"), Some("4"), None, None);
        let mut uniforms = GpuSceneUniforms::zeroed();

        uniforms.apply_lighting_settings(settings);

        assert_eq!(settings.render_mode, RenderMode::Vpt);
        assert_eq!(settings.vpt_max_bounces, 4);
        assert_eq!(uniforms.render_mode, RENDER_MODE_VPT);
        assert_eq!(uniforms.vpt_max_bounces, 4);
    }

    #[test]
    fn lighting_settings_encode_to_stable_gpu_fields() {
        let settings = LightingSettings::default();
        let mut uniforms = GpuSceneUniforms::zeroed();

        uniforms.apply_lighting_settings(settings);

        assert_eq!(
            uniforms.lighting_flags & LIGHTING_FLAG_SHADOWS_ENABLED,
            LIGHTING_FLAG_SHADOWS_ENABLED
        );
        assert_eq!(
            uniforms.lighting_flags & LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS,
            0
        );
    }

    #[test]
    fn lighting_settings_can_enable_backface_shadow_skip_explicitly() {
        let settings =
            LightingSettings::from_values(None, Some("on"), None, None, None, None, None);
        let mut uniforms = GpuSceneUniforms::zeroed();

        uniforms.apply_lighting_settings(settings);

        assert_eq!(
            uniforms.lighting_flags & LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS,
            LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS
        );
    }

    #[test]
    fn build_scene_uniforms_copies_scene_inputs() {
        let settings = LightingSettings {
            shadows_enabled: true,
            skip_backface_shadows: true,
            vct_enabled: true,
            render_mode: RenderMode::Vct,
            vpt_max_bounces: 2,
            debug_view: LightingDebugView::Final,
            exposure: 1.0,
        };

        let uniforms = build_scene_uniforms(SceneUniformInputs {
            pixel_to_ray: glam::Mat4::IDENTITY,
            resolution: [800, 600],
            sun_direction: glam::Vec3::X,
            sun_intensity: glam::Vec3::splat(2.0),
            sky_color: [0.1, 0.2, 0.3],
            ground_color: [0.4, 0.5, 0.6],
            time: 12.5,
            lighting_settings: settings,
            vpt_sample_index: 9,
        });

        assert_eq!(uniforms.resolution, [800, 600]);
        assert_eq!(uniforms.sun_direction, [1.0, 0.0, 0.0]);
        assert_eq!(uniforms.sun_intensity, [2.0, 2.0, 2.0]);
        assert_eq!(uniforms.sky_color, [0.1, 0.2, 0.3]);
        assert_eq!(uniforms.ground_color, [0.4, 0.5, 0.6]);
        assert_eq!(uniforms.time, 12.5);
        assert_eq!(uniforms.lighting_flags, 7);
        assert_eq!(uniforms.exposure, 1.0);
        assert_eq!(uniforms.render_mode, RENDER_MODE_VCT);
        assert_eq!(uniforms.vpt_sample_index, 9);
        assert_eq!(uniforms.vpt_max_bounces, 2);
    }

    #[test]
    fn lighting_settings_parse_exposure_override() {
        let settings =
            LightingSettings::from_values(None, None, None, None, None, None, Some("2.5"));

        assert_eq!(settings.exposure, 2.5);
    }

    #[test]
    fn lighting_settings_parse_bool_overrides_case_insensitively() {
        let settings =
            LightingSettings::from_values(Some("Off"), Some("FALSE"), None, None, None, None, None);

        assert!(!settings.shadows_enabled);
        assert!(!settings.skip_backface_shadows);
    }

    #[test]
    fn lighting_settings_parse_debug_view_aliases() {
        let final_view =
            LightingSettings::from_values(None, None, None, None, None, Some("final"), None);
        let off = LightingSettings::from_values(None, None, None, None, None, Some("off"), None);
        let diffuse =
            LightingSettings::from_values(None, None, None, None, None, Some("diffuse"), None);
        let direct =
            LightingSettings::from_values(None, None, None, None, None, Some("direct"), None);
        let normal =
            LightingSettings::from_values(None, None, None, None, None, Some("normal"), None);

        assert_eq!(final_view.debug_view, LightingDebugView::Final);
        assert_eq!(off.debug_view, LightingDebugView::Final);
        assert_eq!(diffuse.debug_view, LightingDebugView::DirectDiffuse);
        assert_eq!(direct.debug_view, LightingDebugView::DirectDiffuse);
        assert_eq!(normal.debug_view, LightingDebugView::Normal);
    }

    #[test]
    fn lighting_settings_encode_debug_view_without_colliding_with_boolean_flags() {
        let cases = [
            (LightingDebugView::Final, LIGHTING_DEBUG_VIEW_FINAL),
            (
                LightingDebugView::DirectDiffuse,
                LIGHTING_DEBUG_VIEW_DIRECT_DIFFUSE,
            ),
            (LightingDebugView::Normal, LIGHTING_DEBUG_VIEW_NORMAL),
        ];

        for (debug_view, expected_gpu_value) in cases {
            let settings = LightingSettings {
                shadows_enabled: true,
                skip_backface_shadows: true,
                vct_enabled: true,
                render_mode: RenderMode::Vct,
                vpt_max_bounces: 2,
                debug_view,
                exposure: 1.0,
            };

            assert_eq!(
                settings.gpu_flags() & LIGHTING_DEBUG_VIEW_MASK,
                expected_gpu_value << LIGHTING_DEBUG_VIEW_SHIFT
            );
            assert_eq!(
                settings.gpu_flags() & LIGHTING_FLAG_SHADOWS_ENABLED,
                LIGHTING_FLAG_SHADOWS_ENABLED
            );
            assert_eq!(
                settings.gpu_flags() & LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS,
                LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS
            );
        }
    }

    #[test]
    fn lighting_settings_reports_invalid_overrides_without_changing_defaults() {
        let result = LightingSettings::from_values_report(
            Some("maybe"),
            Some("sometimes"),
            Some("perhaps"),
            Some("raster"),
            Some("128"),
            Some("beauty"),
            Some("bright"),
        );

        assert_eq!(result.settings, LightingSettings::default());
        assert_eq!(result.warnings.len(), 7);
        assert_eq!(result.warnings[0].variable, "REVOLUMETRIC_LIGHTING_SHADOWS");
        assert_eq!(
            result.warnings[1].variable,
            "REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS"
        );
        assert_eq!(result.warnings[2].variable, "REVOLUMETRIC_VCT");
        assert_eq!(result.warnings[3].variable, "REVOLUMETRIC_RENDER_MODE");
        assert_eq!(result.warnings[4].variable, "REVOLUMETRIC_VPT_MAX_BOUNCES");
        assert_eq!(
            result.warnings[5].variable,
            "REVOLUMETRIC_LIGHTING_DEBUG_VIEW"
        );
        assert_eq!(result.warnings[6].variable, "REVOLUMETRIC_EXPOSURE");
    }
}
