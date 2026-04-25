use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

pub const LIGHTING_FLAG_SHADOWS_ENABLED: u32 = 1 << 0;
pub const LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS: u32 = 1 << 1;

pub const RC_NORMAL_STRATEGY_AXIS_NORMAL: u32 = 0;
pub const RC_NORMAL_STRATEGY_OCCUPANCY_GRADIENT: u32 = 1;

pub const RC_PROBE_QUALITY_FULL: u32 = 0;

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
}

impl RcProbeQuality {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Full => RC_PROBE_QUALITY_FULL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightingSettings {
    pub shadows_enabled: bool,
    pub skip_backface_shadows: bool,
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
            rc_normal_strategy: RcNormalStrategy::OccupancyGradient,
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
        let normal_strategy = std::env::var("REVOLUMETRIC_RC_NORMAL_STRATEGY").ok();
        let probe_quality = std::env::var("REVOLUMETRIC_RC_PROBE_QUALITY").ok();
        Self::from_values_report(
            shadows.as_deref(),
            skip_backface.as_deref(),
            normal_strategy.as_deref(),
            probe_quality.as_deref(),
        )
    }

    pub fn from_values(
        shadows: Option<&str>,
        skip_backface_shadows: Option<&str>,
        rc_normal_strategy: Option<&str>,
        rc_probe_quality: Option<&str>,
    ) -> Self {
        Self::from_values_report(
            shadows,
            skip_backface_shadows,
            rc_normal_strategy,
            rc_probe_quality,
        )
        .settings
    }

    pub fn from_values_report(
        shadows: Option<&str>,
        skip_backface_shadows: Option<&str>,
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
            "full",
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
    if value.trim().eq_ignore_ascii_case("full") {
        Some(RcProbeQuality::Full)
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
        assert_eq!(
            settings.rc_normal_strategy,
            RcNormalStrategy::OccupancyGradient
        );
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
        assert_eq!(
            uniforms.rc_normal_strategy,
            RC_NORMAL_STRATEGY_OCCUPANCY_GRADIENT
        );
        assert_eq!(uniforms.rc_probe_quality, RC_PROBE_QUALITY_FULL);
    }

    #[test]
    fn lighting_settings_can_enable_backface_shadow_skip_explicitly() {
        let settings = LightingSettings::from_values(None, Some("on"), None, None);
        let mut uniforms = GpuSceneUniforms::zeroed();

        uniforms.apply_lighting_settings(settings);

        assert_eq!(
            uniforms.lighting_flags & LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS,
            LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS
        );
    }

    #[test]
    fn lighting_settings_parse_bool_overrides_case_insensitively() {
        let settings = LightingSettings::from_values(Some("Off"), Some("FALSE"), None, None);

        assert!(!settings.shadows_enabled);
        assert!(!settings.skip_backface_shadows);
    }

    #[test]
    fn lighting_settings_parse_strategy_overrides() {
        let axis = LightingSettings::from_values(None, None, Some("axis"), None);
        let gradient =
            LightingSettings::from_values(None, None, Some("occupancy-gradient"), Some("full"));

        assert_eq!(axis.rc_normal_strategy, RcNormalStrategy::AxisNormal);
        assert_eq!(
            gradient.rc_normal_strategy,
            RcNormalStrategy::OccupancyGradient
        );
        assert_eq!(gradient.rc_probe_quality, RcProbeQuality::Full);
    }

    #[test]
    fn lighting_settings_reports_invalid_overrides_without_changing_defaults() {
        let result = LightingSettings::from_values_report(
            Some("maybe"),
            Some("sometimes"),
            Some("smooth"),
            Some("cheap"),
        );

        assert_eq!(result.settings, LightingSettings::default());
        assert_eq!(result.warnings.len(), 4);
        assert_eq!(result.warnings[0].variable, "REVOLUMETRIC_LIGHTING_SHADOWS");
        assert_eq!(
            result.warnings[1].variable,
            "REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS"
        );
        assert_eq!(
            result.warnings[2].variable,
            "REVOLUMETRIC_RC_NORMAL_STRATEGY"
        );
        assert_eq!(result.warnings[3].variable, "REVOLUMETRIC_RC_PROBE_QUALITY");
    }
}
