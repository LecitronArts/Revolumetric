use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

pub const LIGHTING_FLAG_SHADOWS_ENABLED: u32 = 1 << 0;
pub const LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS: u32 = 1 << 1;
pub const LIGHTING_DEBUG_VIEW_SHIFT: u32 = 28;
pub const LIGHTING_DEBUG_VIEW_MASK: u32 = 0xF << LIGHTING_DEBUG_VIEW_SHIFT;
pub const LIGHTING_DEBUG_VIEW_FINAL: u32 = 0;
pub const LIGHTING_DEBUG_VIEW_RC_INDIRECT: u32 = 1;
pub const LIGHTING_DEBUG_VIEW_DIRECT_DIFFUSE: u32 = 2;
pub const LIGHTING_DEBUG_VIEW_NORMAL: u32 = 3;

pub const RC_NORMAL_STRATEGY_AXIS_NORMAL: u32 = 0;
pub const RC_NORMAL_STRATEGY_OCCUPANCY_GRADIENT: u32 = 1;

pub const RC_PROBE_QUALITY_FULL: u32 = 0;
pub const RC_PROBE_QUALITY_FAST: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightingDebugView {
    Final,
    RcIndirect,
    DirectDiffuse,
    Normal,
}

impl LightingDebugView {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Final => LIGHTING_DEBUG_VIEW_FINAL,
            Self::RcIndirect => LIGHTING_DEBUG_VIEW_RC_INDIRECT,
            Self::DirectDiffuse => LIGHTING_DEBUG_VIEW_DIRECT_DIFFUSE,
            Self::Normal => LIGHTING_DEBUG_VIEW_NORMAL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcNormalStrategy {
    AxisNormal,
    OccupancyGradient,
}

impl RcNormalStrategy {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::AxisNormal => RC_NORMAL_STRATEGY_AXIS_NORMAL,
            Self::OccupancyGradient => RC_NORMAL_STRATEGY_OCCUPANCY_GRADIENT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcProbeQuality {
    Full,
    Fast,
}

impl RcProbeQuality {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Full => RC_PROBE_QUALITY_FULL,
            Self::Fast => RC_PROBE_QUALITY_FAST,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightingSettings {
    pub shadows_enabled: bool,
    pub skip_backface_shadows: bool,
    pub debug_view: LightingDebugView,
    pub rc_normal_strategy: RcNormalStrategy,
    pub rc_probe_quality: RcProbeQuality,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightingSettingsParseWarning {
    pub variable: &'static str,
    pub value: String,
    pub expected: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightingSettingsParseResult {
    pub settings: LightingSettings,
    pub warnings: Vec<LightingSettingsParseWarning>,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            shadows_enabled: true,
            skip_backface_shadows: false,
            debug_view: LightingDebugView::Final,
            rc_normal_strategy: RcNormalStrategy::AxisNormal,
            rc_probe_quality: RcProbeQuality::Full,
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
        let debug_view = std::env::var("REVOLUMETRIC_LIGHTING_DEBUG_VIEW").ok();
        let normal_strategy = std::env::var("REVOLUMETRIC_RC_NORMAL_STRATEGY").ok();
        let probe_quality = std::env::var("REVOLUMETRIC_RC_PROBE_QUALITY").ok();
        Self::from_values_report(
            shadows.as_deref(),
            skip_backface.as_deref(),
            debug_view.as_deref(),
            normal_strategy.as_deref(),
            probe_quality.as_deref(),
        )
    }

    pub fn from_values(
        shadows: Option<&str>,
        skip_backface_shadows: Option<&str>,
        debug_view: Option<&str>,
        rc_normal_strategy: Option<&str>,
        rc_probe_quality: Option<&str>,
    ) -> Self {
        Self::from_values_report(
            shadows,
            skip_backface_shadows,
            debug_view,
            rc_normal_strategy,
            rc_probe_quality,
        )
        .settings
    }

    pub fn from_values_report(
        shadows: Option<&str>,
        skip_backface_shadows: Option<&str>,
        debug_view: Option<&str>,
        rc_normal_strategy: Option<&str>,
        rc_probe_quality: Option<&str>,
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
            &mut settings.debug_view,
            debug_view,
            "REVOLUMETRIC_LIGHTING_DEBUG_VIEW",
            "final|off|rc|indirect|ambient|diffuse|direct|normal",
            parse_lighting_debug_view,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.rc_normal_strategy,
            rc_normal_strategy,
            "REVOLUMETRIC_RC_NORMAL_STRATEGY",
            "occupancy-gradient|gradient|axis|axis-normal",
            parse_rc_normal_strategy,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.rc_probe_quality,
            rc_probe_quality,
            "REVOLUMETRIC_RC_PROBE_QUALITY",
            "full|fast|cheap|low",
            parse_rc_probe_quality,
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
    } else if value.eq_ignore_ascii_case("rc")
        || value.eq_ignore_ascii_case("indirect")
        || value.eq_ignore_ascii_case("ambient")
    {
        Some(LightingDebugView::RcIndirect)
    } else if value.eq_ignore_ascii_case("diffuse") || value.eq_ignore_ascii_case("direct") {
        Some(LightingDebugView::DirectDiffuse)
    } else if value.eq_ignore_ascii_case("normal") {
        Some(LightingDebugView::Normal)
    } else {
        None
    }
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

fn parse_rc_normal_strategy(value: &str) -> Option<RcNormalStrategy> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("axis") || value.eq_ignore_ascii_case("axis-normal") {
        Some(RcNormalStrategy::AxisNormal)
    } else if value.eq_ignore_ascii_case("gradient")
        || value.eq_ignore_ascii_case("occupancy-gradient")
    {
        Some(RcNormalStrategy::OccupancyGradient)
    } else {
        None
    }
}

fn parse_rc_probe_quality(value: &str) -> Option<RcProbeQuality> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("full") {
        Some(RcProbeQuality::Full)
    } else if value.eq_ignore_ascii_case("fast")
        || value.eq_ignore_ascii_case("cheap")
        || value.eq_ignore_ascii_case("low")
    {
        Some(RcProbeQuality::Fast)
    } else {
        None
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
    // --- RC fields (new) ---
    pub rc_c0_grid: [u32; 3],    // 12B
    pub rc_c0_offset: u32,       // 4B
    pub rc_enabled: u32,         // 4B
    pub lighting_flags: u32,     // 4B
    pub rc_normal_strategy: u32, // 4B
    pub rc_probe_quality: u32,   // 4B
}

impl GpuSceneUniforms {
    pub fn apply_lighting_settings(&mut self, settings: LightingSettings) {
        self.lighting_flags = settings.gpu_flags();
        self.rc_normal_strategy = settings.rc_normal_strategy.as_gpu_value();
        self.rc_probe_quality = settings.rc_probe_quality.as_gpu_value();
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
    pub rc_enabled: bool,
    pub lighting_settings: LightingSettings,
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
        rc_c0_grid: [16, 16, 16],
        rc_c0_offset: 0,
        rc_enabled: u32::from(inputs.rc_enabled),
        lighting_flags: 0,
        rc_normal_strategy: 0,
        rc_probe_quality: 0,
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
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, rc_c0_grid), 144);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, rc_c0_offset), 156);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, rc_enabled), 160);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, lighting_flags), 164);
        assert_eq!(
            std::mem::offset_of!(GpuSceneUniforms, rc_normal_strategy),
            168
        );
        assert_eq!(
            std::mem::offset_of!(GpuSceneUniforms, rc_probe_quality),
            172
        );
    }

    #[test]
    fn gpu_scene_uniforms_is_zeroable() {
        let u = GpuSceneUniforms::zeroed();
        assert_eq!(u.resolution, [0, 0]);
        assert_eq!(u.time, 0.0);
    }

    #[test]
    fn lighting_settings_default_preserves_high_quality_path() {
        let settings = LightingSettings::default();

        assert!(settings.shadows_enabled);
        assert!(!settings.skip_backface_shadows);
        assert_eq!(settings.debug_view, LightingDebugView::Final);
        assert_eq!(settings.rc_normal_strategy, RcNormalStrategy::AxisNormal);
        assert_eq!(settings.rc_probe_quality, RcProbeQuality::Full);
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
        assert_eq!(uniforms.rc_normal_strategy, RC_NORMAL_STRATEGY_AXIS_NORMAL);
        assert_eq!(uniforms.rc_probe_quality, RC_PROBE_QUALITY_FULL);
    }

    #[test]
    fn lighting_settings_can_enable_backface_shadow_skip_explicitly() {
        let settings = LightingSettings::from_values(None, Some("on"), None, None, None);
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
            debug_view: LightingDebugView::Final,
            rc_normal_strategy: RcNormalStrategy::AxisNormal,
            rc_probe_quality: RcProbeQuality::Full,
        };

        let uniforms = build_scene_uniforms(SceneUniformInputs {
            pixel_to_ray: glam::Mat4::IDENTITY,
            resolution: [800, 600],
            sun_direction: glam::Vec3::X,
            sun_intensity: glam::Vec3::splat(2.0),
            sky_color: [0.1, 0.2, 0.3],
            ground_color: [0.4, 0.5, 0.6],
            time: 12.5,
            rc_enabled: true,
            lighting_settings: settings,
        });

        assert_eq!(uniforms.resolution, [800, 600]);
        assert_eq!(uniforms.sun_direction, [1.0, 0.0, 0.0]);
        assert_eq!(uniforms.sun_intensity, [2.0, 2.0, 2.0]);
        assert_eq!(uniforms.sky_color, [0.1, 0.2, 0.3]);
        assert_eq!(uniforms.ground_color, [0.4, 0.5, 0.6]);
        assert_eq!(uniforms.time, 12.5);
        assert_eq!(uniforms.rc_enabled, 1);
        assert_eq!(uniforms.lighting_flags, 3);
        assert_eq!(uniforms.rc_normal_strategy, RC_NORMAL_STRATEGY_AXIS_NORMAL);
    }

    #[test]
    fn lighting_settings_parse_bool_overrides_case_insensitively() {
        let settings = LightingSettings::from_values(Some("Off"), Some("FALSE"), None, None, None);

        assert!(!settings.shadows_enabled);
        assert!(!settings.skip_backface_shadows);
    }

    #[test]
    fn lighting_settings_parse_strategy_overrides() {
        let axis = LightingSettings::from_values(None, None, None, Some("axis"), None);
        let gradient = LightingSettings::from_values(
            None,
            None,
            None,
            Some("occupancy-gradient"),
            Some("full"),
        );

        assert_eq!(axis.rc_normal_strategy, RcNormalStrategy::AxisNormal);
        assert_eq!(
            gradient.rc_normal_strategy,
            RcNormalStrategy::OccupancyGradient
        );
        assert_eq!(gradient.rc_probe_quality, RcProbeQuality::Full);
    }

    #[test]
    fn lighting_settings_parse_fast_probe_quality() {
        let settings = LightingSettings::from_values(None, None, None, None, Some("fast"));

        assert_eq!(settings.rc_probe_quality, RcProbeQuality::Fast);
    }

    #[test]
    fn lighting_settings_encode_fast_probe_quality() {
        let settings = LightingSettings {
            shadows_enabled: true,
            skip_backface_shadows: false,
            debug_view: LightingDebugView::Final,
            rc_normal_strategy: RcNormalStrategy::AxisNormal,
            rc_probe_quality: RcProbeQuality::Fast,
        };

        let uniforms = build_scene_uniforms(SceneUniformInputs {
            pixel_to_ray: glam::Mat4::IDENTITY,
            resolution: [640, 480],
            sun_direction: glam::Vec3::Y,
            sun_intensity: glam::Vec3::ONE,
            sky_color: [0.4, 0.5, 0.7],
            ground_color: [0.15, 0.1, 0.08],
            time: 0.0,
            rc_enabled: true,
            lighting_settings: settings,
        });

        assert_eq!(uniforms.rc_probe_quality, RC_PROBE_QUALITY_FAST);
    }

    #[test]
    fn lighting_settings_parse_debug_view_aliases() {
        let final_view = LightingSettings::from_values(None, None, Some("final"), None, None);
        let rc = LightingSettings::from_values(None, None, Some("rc"), None, None);
        let ambient = LightingSettings::from_values(None, None, Some("ambient"), None, None);
        let indirect = LightingSettings::from_values(None, None, Some("indirect"), None, None);
        let diffuse = LightingSettings::from_values(None, None, Some("diffuse"), None, None);
        let direct = LightingSettings::from_values(None, None, Some("direct"), None, None);
        let normal = LightingSettings::from_values(None, None, Some("normal"), None, None);

        assert_eq!(final_view.debug_view, LightingDebugView::Final);
        assert_eq!(rc.debug_view, LightingDebugView::RcIndirect);
        assert_eq!(ambient.debug_view, LightingDebugView::RcIndirect);
        assert_eq!(indirect.debug_view, LightingDebugView::RcIndirect);
        assert_eq!(diffuse.debug_view, LightingDebugView::DirectDiffuse);
        assert_eq!(direct.debug_view, LightingDebugView::DirectDiffuse);
        assert_eq!(normal.debug_view, LightingDebugView::Normal);
    }

    #[test]
    fn lighting_settings_encode_debug_view_without_colliding_with_boolean_flags() {
        let cases = [
            (LightingDebugView::Final, LIGHTING_DEBUG_VIEW_FINAL),
            (
                LightingDebugView::RcIndirect,
                LIGHTING_DEBUG_VIEW_RC_INDIRECT,
            ),
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
                debug_view,
                rc_normal_strategy: RcNormalStrategy::AxisNormal,
                rc_probe_quality: RcProbeQuality::Full,
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
            Some("beauty"),
            Some("smooth"),
            Some("invalid-quality"),
        );

        assert_eq!(result.settings, LightingSettings::default());
        assert_eq!(result.warnings.len(), 5);
        assert_eq!(result.warnings[0].variable, "REVOLUMETRIC_LIGHTING_SHADOWS");
        assert_eq!(
            result.warnings[1].variable,
            "REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS"
        );
        assert_eq!(
            result.warnings[2].variable,
            "REVOLUMETRIC_LIGHTING_DEBUG_VIEW"
        );
        assert_eq!(
            result.warnings[3].variable,
            "REVOLUMETRIC_RC_NORMAL_STRATEGY"
        );
        assert_eq!(result.warnings[4].variable, "REVOLUMETRIC_RC_PROBE_QUALITY");
    }
}
