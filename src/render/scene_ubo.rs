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
pub const LIGHTING_DEBUG_VIEW_DIRECT_DIFFUSE: u32 = 1;
pub const LIGHTING_DEBUG_VIEW_NORMAL: u32 = 2;
pub const RENDER_MODE_VPT: u32 = 1;
pub const DENOISER_FLAG_ENABLED: u32 = 1 << 0;
pub const VPT_DEBUG_VIEW_FINAL: u32 = 0;
pub const VPT_DEBUG_VIEW_RAW: u32 = 1;
pub const VPT_DEBUG_VIEW_TEMPORAL: u32 = 2;
pub const VPT_DEBUG_VIEW_VARIANCE: u32 = 3;
pub const VPT_DEBUG_VIEW_HISTORY_VALID: u32 = 4;
pub const VPT_DEBUG_VIEW_MOTION: u32 = 5;
pub const VPT_DEBUG_VIEW_NORMAL: u32 = 6;
pub const VPT_DEBUG_VIEW_DEPTH: u32 = 7;
pub const VPT_DEBUG_VIEW_RESERVOIR_WEIGHT: u32 = 8;
pub const VPT_DEBUG_VIEW_DIRECT: u32 = 9;
pub const VPT_DEBUG_VIEW_INDIRECT: u32 = 10;
pub const VPT_DEBUG_VIEW_AREA_SUBPIXEL: u32 = 11;
pub const VPT_DEBUG_VIEW_AREA_LENS: u32 = 12;
pub const VPT_DEBUG_VIEW_AREA_WEIGHT: u32 = 13;
pub const VPT_DEBUG_VIEW_AREA_HISTORY_VALID: u32 = 14;
pub const VPT_DEBUG_VIEW_AREA_REJECTION: u32 = 15;
pub const VPT_DEBUG_VIEW_AREA_JACOBIAN: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightingDebugView {
    Final,
    DirectDiffuse,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Vpt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VptDebugView {
    Final,
    Raw,
    Temporal,
    Variance,
    HistoryValid,
    Motion,
    Normal,
    Depth,
    ReservoirWeight,
    Direct,
    Indirect,
    AreaSubpixel,
    AreaLens,
    AreaWeight,
    AreaHistoryValid,
    AreaRejection,
    AreaJacobian,
}

impl RenderMode {
    pub fn as_gpu_value(self) -> u32 {
        match self {
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

impl VptDebugView {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Final => VPT_DEBUG_VIEW_FINAL,
            Self::Raw => VPT_DEBUG_VIEW_RAW,
            Self::Temporal => VPT_DEBUG_VIEW_TEMPORAL,
            Self::Variance => VPT_DEBUG_VIEW_VARIANCE,
            Self::HistoryValid => VPT_DEBUG_VIEW_HISTORY_VALID,
            Self::Motion => VPT_DEBUG_VIEW_MOTION,
            Self::Normal => VPT_DEBUG_VIEW_NORMAL,
            Self::Depth => VPT_DEBUG_VIEW_DEPTH,
            Self::ReservoirWeight => VPT_DEBUG_VIEW_RESERVOIR_WEIGHT,
            Self::Direct => VPT_DEBUG_VIEW_DIRECT,
            Self::Indirect => VPT_DEBUG_VIEW_INDIRECT,
            Self::AreaSubpixel => VPT_DEBUG_VIEW_AREA_SUBPIXEL,
            Self::AreaLens => VPT_DEBUG_VIEW_AREA_LENS,
            Self::AreaWeight => VPT_DEBUG_VIEW_AREA_WEIGHT,
            Self::AreaHistoryValid => VPT_DEBUG_VIEW_AREA_HISTORY_VALID,
            Self::AreaRejection => VPT_DEBUG_VIEW_AREA_REJECTION,
            Self::AreaJacobian => VPT_DEBUG_VIEW_AREA_JACOBIAN,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightingSettings {
    pub shadows_enabled: bool,
    pub skip_backface_shadows: bool,
    pub render_mode: RenderMode,
    pub vpt_max_bounces: u32,
    pub sun_angular_radius: f32,
    pub debug_view: LightingDebugView,
    pub exposure: f32,
    pub denoiser_enabled: bool,
    pub denoiser_atrous_iterations: u32,
    pub vpt_debug_view: VptDebugView,
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
            render_mode: RenderMode::Vpt,
            vpt_max_bounces: 2,
            sun_angular_radius: 0.02,
            debug_view: LightingDebugView::Final,
            exposure: 1.0,
            denoiser_enabled: true,
            denoiser_atrous_iterations: 4,
            vpt_debug_view: VptDebugView::Final,
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
        let render_mode = std::env::var("REVOLUMETRIC_RENDER_MODE").ok();
        let vpt_max_bounces = std::env::var("REVOLUMETRIC_VPT_MAX_BOUNCES").ok();
        let debug_view = std::env::var("REVOLUMETRIC_LIGHTING_DEBUG_VIEW").ok();
        let exposure = std::env::var("REVOLUMETRIC_EXPOSURE").ok();
        let denoiser = std::env::var("REVOLUMETRIC_DENOISER").ok();
        let denoiser_atrous_iterations =
            std::env::var("REVOLUMETRIC_DENOISER_ATROUS_ITERATIONS").ok();
        let vpt_debug_view = std::env::var("REVOLUMETRIC_VPT_DEBUG_VIEW").ok();
        let sun_angular_radius = std::env::var("REVOLUMETRIC_SUN_ANGULAR_RADIUS").ok();
        Self::from_values_report_with_denoiser(
            shadows.as_deref(),
            skip_backface.as_deref(),
            render_mode.as_deref(),
            vpt_max_bounces.as_deref(),
            debug_view.as_deref(),
            exposure.as_deref(),
            denoiser.as_deref(),
            denoiser_atrous_iterations.as_deref(),
            vpt_debug_view.as_deref(),
            sun_angular_radius.as_deref(),
        )
    }

    pub fn from_values(
        shadows: Option<&str>,
        skip_backface_shadows: Option<&str>,
        render_mode: Option<&str>,
        vpt_max_bounces: Option<&str>,
        debug_view: Option<&str>,
        exposure: Option<&str>,
    ) -> Self {
        Self::from_values_report(
            shadows,
            skip_backface_shadows,
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
        render_mode: Option<&str>,
        vpt_max_bounces: Option<&str>,
        debug_view: Option<&str>,
        exposure: Option<&str>,
    ) -> LightingSettingsParseResult {
        Self::from_values_report_with_denoiser(
            shadows,
            skip_backface_shadows,
            render_mode,
            vpt_max_bounces,
            debug_view,
            exposure,
            None,
            None,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_values_report_with_denoiser(
        shadows: Option<&str>,
        skip_backface_shadows: Option<&str>,
        render_mode: Option<&str>,
        vpt_max_bounces: Option<&str>,
        debug_view: Option<&str>,
        exposure: Option<&str>,
        denoiser: Option<&str>,
        denoiser_atrous_iterations: Option<&str>,
        vpt_debug_view: Option<&str>,
        sun_angular_radius: Option<&str>,
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
            &mut settings.render_mode,
            render_mode,
            "REVOLUMETRIC_RENDER_MODE",
            "vpt",
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
        apply_optional_override(
            &mut settings.denoiser_enabled,
            denoiser,
            "REVOLUMETRIC_DENOISER",
            "on|off|1|0|true|false",
            parse_bool_value,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.denoiser_atrous_iterations,
            denoiser_atrous_iterations,
            "REVOLUMETRIC_DENOISER_ATROUS_ITERATIONS",
            "integer in 0..=5",
            parse_denoiser_atrous_iterations,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.vpt_debug_view,
            vpt_debug_view,
            "REVOLUMETRIC_VPT_DEBUG_VIEW",
            "final|raw|temporal|variance|history_valid|motion|normal|depth|reservoir_weight|direct|indirect|area_subpixel|area_lens|area_weight|area_history_valid|area_rejection|area_jacobian",
            parse_vpt_debug_view,
            &mut warnings,
        );
        apply_optional_override(
            &mut settings.sun_angular_radius,
            sun_angular_radius,
            "REVOLUMETRIC_SUN_ANGULAR_RADIUS",
            "finite float in 0.0..=0.25 radians",
            parse_sun_angular_radius,
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

    pub fn denoiser_flags(self) -> u32 {
        if self.denoiser_enabled {
            DENOISER_FLAG_ENABLED
        } else {
            0
        }
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

fn parse_vpt_debug_view(value: &str) -> Option<VptDebugView> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("final") || value.eq_ignore_ascii_case("off") {
        Some(VptDebugView::Final)
    } else if value.eq_ignore_ascii_case("raw") {
        Some(VptDebugView::Raw)
    } else if value.eq_ignore_ascii_case("temporal") {
        Some(VptDebugView::Temporal)
    } else if value.eq_ignore_ascii_case("variance") {
        Some(VptDebugView::Variance)
    } else if value.eq_ignore_ascii_case("history_valid") {
        Some(VptDebugView::HistoryValid)
    } else if value.eq_ignore_ascii_case("motion") {
        Some(VptDebugView::Motion)
    } else if value.eq_ignore_ascii_case("normal") {
        Some(VptDebugView::Normal)
    } else if value.eq_ignore_ascii_case("depth") {
        Some(VptDebugView::Depth)
    } else if value.eq_ignore_ascii_case("reservoir_weight") {
        Some(VptDebugView::ReservoirWeight)
    } else if value.eq_ignore_ascii_case("direct") {
        Some(VptDebugView::Direct)
    } else if value.eq_ignore_ascii_case("indirect") {
        Some(VptDebugView::Indirect)
    } else if value.eq_ignore_ascii_case("area_subpixel") {
        Some(VptDebugView::AreaSubpixel)
    } else if value.eq_ignore_ascii_case("area_lens") {
        Some(VptDebugView::AreaLens)
    } else if value.eq_ignore_ascii_case("area_weight") {
        Some(VptDebugView::AreaWeight)
    } else if value.eq_ignore_ascii_case("area_history_valid") {
        Some(VptDebugView::AreaHistoryValid)
    } else if value.eq_ignore_ascii_case("area_rejection") {
        Some(VptDebugView::AreaRejection)
    } else if value.eq_ignore_ascii_case("area_jacobian") {
        Some(VptDebugView::AreaJacobian)
    } else {
        None
    }
}

fn parse_render_mode(value: &str) -> Option<RenderMode> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("vpt") {
        Some(RenderMode::Vpt)
    } else {
        None
    }
}

fn parse_denoiser_atrous_iterations(value: &str) -> Option<u32> {
    let parsed = value.trim().parse::<u32>().ok()?;
    (0..=5).contains(&parsed).then_some(parsed)
}

fn parse_vpt_max_bounces(value: &str) -> Option<u32> {
    let parsed = value.trim().parse::<u32>().ok()?;
    (1..=8).contains(&parsed).then_some(parsed)
}

fn parse_sun_angular_radius(value: &str) -> Option<f32> {
    let parsed = value.trim().parse::<f32>().ok()?;
    (parsed.is_finite() && (0.0..=0.25).contains(&parsed)).then_some(parsed)
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
/// 224 bytes, std140-compatible (all float3 fields padded to 16-byte alignment).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSceneUniforms {
    pub pixel_to_ray: [[f32; 4]; 4], // 64B — col 0-2: direction matrix, col 3: camera origin
    pub resolution: [u32; 2],        // 8B
    pub _pad0: [u32; 2],             // 8B
    pub sun_direction: [f32; 3],     // 12B — normalized, world space, points TOWARD sun
    pub sun_angular_radius: f32,     // 4B
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
    pub denoiser_flags: u32,         // 4B
    pub denoiser_atrous_iterations: u32, // 4B
    pub vpt_debug_view: u32,         // 4B
    pub camera_right: [f32; 3],      // 12B
    pub aperture_radius: f32,        // 4B
    pub camera_up: [f32; 3],         // 12B
    pub focal_distance: f32,         // 4B
    pub camera_forward: [f32; 3],    // 12B
    pub _pad4: f32,                  // 4B
}

impl GpuSceneUniforms {
    pub fn apply_lighting_settings(&mut self, settings: LightingSettings) {
        self.lighting_flags = settings.gpu_flags();
        self.exposure = settings.exposure;
        self.render_mode = settings.render_mode.as_gpu_value();
        self.vpt_max_bounces = settings.vpt_max_bounces;
        self.sun_angular_radius = settings.sun_angular_radius;
        self.denoiser_flags = settings.denoiser_flags();
        self.denoiser_atrous_iterations = settings.denoiser_atrous_iterations;
        self.vpt_debug_view = settings.vpt_debug_view.as_gpu_value();
    }
}

pub struct SceneUniformInputs {
    pub pixel_to_ray: glam::Mat4,
    pub resolution: [u32; 2],
    pub camera_right: glam::Vec3,
    pub camera_up: glam::Vec3,
    pub camera_forward: glam::Vec3,
    pub aperture_radius: f32,
    pub focal_distance: f32,
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
        sun_angular_radius: inputs.lighting_settings.sun_angular_radius,
        sun_intensity: inputs.sun_intensity.to_array(),
        _pad2: 0.0,
        sky_color: inputs.sky_color,
        _pad3: 0.0,
        ground_color: inputs.ground_color,
        time: inputs.time,
        lighting_flags: 0,
        exposure: 1.0,
        render_mode: RENDER_MODE_VPT,
        vpt_sample_index: inputs.vpt_sample_index,
        vpt_max_bounces: 2,
        denoiser_flags: 0,
        denoiser_atrous_iterations: 0,
        vpt_debug_view: VPT_DEBUG_VIEW_FINAL,
        camera_right: inputs.camera_right.normalize_or_zero().to_array(),
        aperture_radius: inputs.aperture_radius.max(0.0),
        camera_up: inputs.camera_up.normalize_or_zero().to_array(),
        focal_distance: inputs.focal_distance.max(1.0e-3),
        camera_forward: inputs.camera_forward.normalize_or_zero().to_array(),
        _pad4: 0.0,
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
    fn gpu_scene_uniforms_size_is_224_bytes() {
        assert_eq!(std::mem::size_of::<GpuSceneUniforms>(), 224);
    }

    #[test]
    fn gpu_scene_uniforms_offsets_match_slang_abi() {
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, pixel_to_ray), 0);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, resolution), 64);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, sun_direction), 80);
        assert_eq!(
            std::mem::offset_of!(GpuSceneUniforms, sun_angular_radius),
            92
        );
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, ground_color), 128);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, lighting_flags), 144);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, exposure), 148);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, render_mode), 152);
        assert_eq!(
            std::mem::offset_of!(GpuSceneUniforms, vpt_sample_index),
            156
        );
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, vpt_max_bounces), 160);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, camera_right), 176);
    }

    #[test]
    fn gpu_scene_uniforms_expose_area_restir_camera_basis_and_lens_fields() {
        assert_eq!(std::mem::size_of::<GpuSceneUniforms>(), 224);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, camera_right), 176);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, aperture_radius), 188);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, camera_up), 192);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, focal_distance), 204);
        assert_eq!(std::mem::offset_of!(GpuSceneUniforms, camera_forward), 208);
    }

    #[test]
    fn scene_common_declares_area_restir_camera_fields() {
        let source = std::fs::read_to_string("assets/shaders/shared/scene_common.slang")
            .expect("scene_common.slang should be readable");

        for token in [
            "float3   camera_right",
            "float    aperture_radius",
            "float3   camera_up",
            "float    focal_distance",
            "float3   camera_forward",
            "sample_disk_for_lens",
            "scene_primary_ray_from_area_sample",
        ] {
            assert!(
                source.contains(token),
                "scene_common.slang missing Area ReSTIR camera token {token}"
            );
        }
    }

    #[test]
    fn scene_uniforms_expose_sun_angular_radius_without_growing_abi() {
        let source = std::fs::read_to_string("assets/shaders/shared/scene_common.slang")
            .expect("scene_common.slang should be readable");

        assert_eq!(std::mem::size_of::<GpuSceneUniforms>(), 224);
        assert_eq!(
            std::mem::offset_of!(GpuSceneUniforms, sun_angular_radius),
            92
        );
        assert!(source.contains("float    sun_angular_radius;"));

        let settings = LightingSettings::from_values_report_with_denoiser(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("0.02"),
        )
        .settings;
        let mut uniforms = GpuSceneUniforms::zeroed();
        uniforms.apply_lighting_settings(settings);

        assert!((settings.sun_angular_radius - 0.02).abs() < f32::EPSILON);
        assert!((uniforms.sun_angular_radius - 0.02).abs() < f32::EPSILON);
    }

    #[test]
    fn gpu_scene_uniforms_is_zeroable() {
        let u = GpuSceneUniforms::zeroed();
        assert_eq!(u.resolution, [0, 0]);
        assert_eq!(u.time, 0.0);
    }

    #[test]
    fn lighting_settings_default_preserves_direct_lighting_controls() {
        let settings = LightingSettings::default();

        assert!(settings.shadows_enabled);
        assert!(!settings.skip_backface_shadows);
        assert_eq!(settings.render_mode, RenderMode::Vpt);
        assert_eq!(settings.vpt_max_bounces, 2);
        assert_eq!(settings.sun_angular_radius, 0.02);
        assert_eq!(settings.debug_view, LightingDebugView::Final);
        assert_eq!(settings.exposure, 1.0);
    }

    #[test]
    fn lighting_settings_default_is_vpt_only() {
        let settings = LightingSettings::default();

        assert_eq!(settings.render_mode, RenderMode::Vpt);
        assert_eq!(
            settings.gpu_flags() & LIGHTING_FLAG_SHADOWS_ENABLED,
            LIGHTING_FLAG_SHADOWS_ENABLED
        );
    }

    #[test]
    fn legacy_vct_render_mode_is_rejected() {
        let result =
            LightingSettings::from_values_report(None, None, Some("vct"), None, None, None);

        assert_eq!(result.settings.render_mode, RenderMode::Vpt);
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.variable == "REVOLUMETRIC_RENDER_MODE")
        );
    }

    #[test]
    fn lighting_settings_parse_vpt_render_mode_and_bounce_limit() {
        let settings =
            LightingSettings::from_values(None, None, Some("vpt"), Some("4"), None, None);
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
        let settings = LightingSettings::from_values(None, Some("on"), None, None, None, None);
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
            render_mode: RenderMode::Vpt,
            vpt_max_bounces: 2,
            sun_angular_radius: 0.02,
            debug_view: LightingDebugView::Final,
            exposure: 1.0,
            denoiser_enabled: true,
            denoiser_atrous_iterations: 4,
            vpt_debug_view: VptDebugView::Final,
        };

        let uniforms = build_scene_uniforms(SceneUniformInputs {
            pixel_to_ray: glam::Mat4::IDENTITY,
            resolution: [800, 600],
            camera_right: glam::Vec3::X,
            camera_up: glam::Vec3::Y,
            camera_forward: glam::Vec3::Z,
            aperture_radius: 0.25,
            focal_distance: 64.0,
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
        assert_eq!(uniforms.lighting_flags, 3);
        assert_eq!(uniforms.exposure, 1.0);
        assert_eq!(uniforms.render_mode, RENDER_MODE_VPT);
        assert_eq!(uniforms.vpt_sample_index, 9);
        assert_eq!(uniforms.vpt_max_bounces, 2);
        assert_eq!(uniforms.camera_right, [1.0, 0.0, 0.0]);
        assert_eq!(uniforms.camera_up, [0.0, 1.0, 0.0]);
        assert_eq!(uniforms.camera_forward, [0.0, 0.0, 1.0]);
        assert_eq!(uniforms.aperture_radius, 0.25);
        assert_eq!(uniforms.focal_distance, 64.0);
    }

    #[test]
    fn lighting_settings_parse_exposure_override() {
        let settings = LightingSettings::from_values(None, None, None, None, None, Some("2.5"));

        assert_eq!(settings.exposure, 2.5);
    }

    #[test]
    fn lighting_settings_parse_bool_overrides_case_insensitively() {
        let settings =
            LightingSettings::from_values(Some("Off"), Some("FALSE"), None, None, None, None);

        assert!(!settings.shadows_enabled);
        assert!(!settings.skip_backface_shadows);
    }

    #[test]
    fn lighting_settings_parse_debug_view_aliases() {
        let final_view = LightingSettings::from_values(None, None, None, None, Some("final"), None);
        let off = LightingSettings::from_values(None, None, None, None, Some("off"), None);
        let diffuse = LightingSettings::from_values(None, None, None, None, Some("diffuse"), None);
        let direct = LightingSettings::from_values(None, None, None, None, Some("direct"), None);
        let normal = LightingSettings::from_values(None, None, None, None, Some("normal"), None);

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
                render_mode: RenderMode::Vpt,
                vpt_max_bounces: 2,
                sun_angular_radius: 0.02,
                debug_view,
                exposure: 1.0,
                denoiser_enabled: true,
                denoiser_atrous_iterations: 4,
                vpt_debug_view: VptDebugView::Final,
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
            Some("raster"),
            Some("128"),
            Some("beauty"),
            Some("bright"),
        );

        assert_eq!(result.settings, LightingSettings::default());
        assert_eq!(result.warnings.len(), 6);
        assert_eq!(result.warnings[0].variable, "REVOLUMETRIC_LIGHTING_SHADOWS");
        assert_eq!(
            result.warnings[1].variable,
            "REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS"
        );
        assert_eq!(result.warnings[2].variable, "REVOLUMETRIC_RENDER_MODE");
        assert_eq!(result.warnings[3].variable, "REVOLUMETRIC_VPT_MAX_BOUNCES");
        assert_eq!(
            result.warnings[4].variable,
            "REVOLUMETRIC_LIGHTING_DEBUG_VIEW"
        );
        assert_eq!(result.warnings[5].variable, "REVOLUMETRIC_EXPOSURE");
    }

    #[test]
    fn lighting_settings_parse_vpt_denoiser_and_debug_controls() {
        let result = LightingSettings::from_values_report_with_denoiser(
            None,
            None,
            None,
            None,
            None,
            None,
            Some("off"),
            Some("4"),
            Some("history_valid"),
            None,
        );
        let settings = result.settings;
        let mut uniforms = GpuSceneUniforms::zeroed();

        uniforms.apply_lighting_settings(settings);

        assert!(!settings.denoiser_enabled);
        assert_eq!(settings.denoiser_atrous_iterations, 4);
        assert_eq!(settings.vpt_debug_view, VptDebugView::HistoryValid);
        assert_eq!(uniforms.denoiser_flags & DENOISER_FLAG_ENABLED, 0);
        assert_eq!(uniforms.denoiser_atrous_iterations, 4);
        assert_eq!(uniforms.vpt_debug_view, VPT_DEBUG_VIEW_HISTORY_VALID);
    }

    #[test]
    fn lighting_settings_parse_area_restir_vpt_debug_view_aliases() {
        let cases = [
            ("area_subpixel", 11),
            ("area_lens", 12),
            ("area_weight", 13),
            ("area_history_valid", 14),
            ("area_rejection", 15),
            ("area_jacobian", 16),
        ];

        for (raw, expected_gpu_value) in cases {
            let result = LightingSettings::from_values_report_with_denoiser(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(raw),
                None,
            );
            let mut uniforms = GpuSceneUniforms::zeroed();
            uniforms.apply_lighting_settings(result.settings);

            assert!(
                result.warnings.is_empty(),
                "area debug view alias {raw} should parse without warnings"
            );
            assert_eq!(uniforms.vpt_debug_view, expected_gpu_value);
        }
    }

    #[test]
    fn lighting_settings_warn_invalid_vpt_denoiser_controls() {
        let result = LightingSettings::from_values_report_with_denoiser(
            None,
            None,
            None,
            None,
            None,
            None,
            Some("maybe"),
            Some("9"),
            Some("beauty"),
            None,
        );

        assert_eq!(result.settings, LightingSettings::default());
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.variable == "REVOLUMETRIC_DENOISER")
        );
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.variable == "REVOLUMETRIC_DENOISER_ATROUS_ITERATIONS")
        );
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.variable == "REVOLUMETRIC_VPT_DEBUG_VIEW")
        );
    }

    #[test]
    fn lighting_settings_warn_invalid_sun_angular_radius() {
        let result = LightingSettings::from_values_report_with_denoiser(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("0.5"),
        );

        assert_eq!(result.settings.sun_angular_radius, 0.02);
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.variable == "REVOLUMETRIC_SUN_ANGULAR_RADIUS")
        );
    }
}
