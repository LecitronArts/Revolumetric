use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

const DEFAULT_CAPTURE_DIR: &str = "target/captures";
const DEFAULT_CAPTURE_PREFIX: &str = "capture";
const RGBA8_BYTES_PER_PIXEL: u64 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureConfig {
    pub target_frame: Option<u64>,
    pub output_dir: PathBuf,
    pub prefix: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturePaths {
    pub ppm_path: PathBuf,
    pub json_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureMetadata {
    pub frame_index: u64,
    pub vpt_sample_index: u32,
    pub width: u32,
    pub height: u32,
    pub source: &'static str,
    pub ppm_path: PathBuf,
    pub json_path: PathBuf,
    pub render_backend: &'static str,
    pub render_mode: &'static str,
    pub rt_debug_view: &'static str,
    pub rt_restir_di_enabled: bool,
    pub rt_restir_di_spatial_enabled: bool,
    pub rt_restir_di_spatial_sample_count: u32,
    pub rt_restir_gi_enabled: bool,
    pub rt_temporal_denoise_enabled: bool,
    pub restir_di_enabled: bool,
    pub restir_di_temporal_enabled: bool,
    pub restir_di_spatial_enabled: bool,
    pub area_restir_enabled: bool,
    pub area_restir_temporal_enabled: bool,
    pub area_restir_spatial_enabled: bool,
    pub vpt_debug_view: &'static str,
    pub denoiser_enabled: bool,
    pub denoiser_mode: &'static str,
    pub effective_denoiser_mode: &'static str,
}

pub struct RenderCapture {
    config: CaptureConfig,
    readback: Option<GpuBuffer>,
    readback_extent: Option<[u32; 2]>,
}

impl CaptureConfig {
    pub fn from_env() -> Result<Option<Self>> {
        let frame = std::env::var("REVOLUMETRIC_CAPTURE_FRAME").ok();
        if frame.is_none() {
            return Ok(None);
        }
        let output_dir = std::env::var("REVOLUMETRIC_CAPTURE_DIR").ok();
        let prefix = std::env::var("REVOLUMETRIC_CAPTURE_PREFIX").ok();

        Self::from_values(frame.as_deref(), output_dir.as_deref(), prefix.as_deref()).map(Some)
    }

    pub fn from_values(
        target_frame: Option<&str>,
        output_dir: Option<&str>,
        prefix: Option<&str>,
    ) -> Result<Self> {
        let target_frame = match target_frame {
            Some(value) => Some(parse_target_frame(value)?),
            None => None,
        };
        let output_dir = output_dir
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map_or_else(|| PathBuf::from(DEFAULT_CAPTURE_DIR), PathBuf::from);
        let prefix = prefix
            .map(sanitize_prefix)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_CAPTURE_PREFIX.to_string());

        Ok(Self {
            target_frame,
            output_dir,
            prefix,
        })
    }

    pub fn paths_for_frame(&self, frame_index: u64) -> CapturePaths {
        let stem = format!("{}_{frame_index:06}", self.prefix);
        CapturePaths {
            ppm_path: self.output_dir.join(format!("{stem}.ppm")),
            json_path: self.output_dir.join(format!("{stem}.json")),
        }
    }
}

impl CaptureMetadata {
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"frame_index\": {},\n",
                "  \"vpt_sample_index\": {},\n",
                "  \"width\": {},\n",
                "  \"height\": {},\n",
                "  \"source\": \"{}\",\n",
                "  \"ppm_path\": \"{}\",\n",
                "  \"json_path\": \"{}\",\n",
                "  \"render_backend\": \"{}\",\n",
                "  \"render_mode\": \"{}\",\n",
                "  \"rt_debug_view\": \"{}\",\n",
                "  \"rt_restir_di_enabled\": {},\n",
                "  \"rt_restir_di_spatial_enabled\": {},\n",
                "  \"rt_restir_di_spatial_sample_count\": {},\n",
                "  \"rt_restir_gi_enabled\": {},\n",
                "  \"rt_temporal_denoise_enabled\": {},\n",
                "  \"restir_di_enabled\": {},\n",
                "  \"restir_di_temporal_enabled\": {},\n",
                "  \"restir_di_spatial_enabled\": {},\n",
                "  \"area_restir_enabled\": {},\n",
                "  \"area_restir_temporal_enabled\": {},\n",
                "  \"area_restir_spatial_enabled\": {},\n",
                "  \"vpt_debug_view\": \"{}\",\n",
                "  \"denoiser_enabled\": {},\n",
                "  \"denoiser_mode\": \"{}\",\n",
                "  \"effective_denoiser_mode\": \"{}\"\n",
                "}}\n"
            ),
            self.frame_index,
            self.vpt_sample_index,
            self.width,
            self.height,
            json_escape(self.source),
            json_escape_path(&self.ppm_path),
            json_escape_path(&self.json_path),
            json_escape(self.render_backend),
            json_escape(self.render_mode),
            json_escape(self.rt_debug_view),
            self.rt_restir_di_enabled,
            self.rt_restir_di_spatial_enabled,
            self.rt_restir_di_spatial_sample_count,
            self.rt_restir_gi_enabled,
            self.rt_temporal_denoise_enabled,
            self.restir_di_enabled,
            self.restir_di_temporal_enabled,
            self.restir_di_spatial_enabled,
            self.area_restir_enabled,
            self.area_restir_temporal_enabled,
            self.area_restir_spatial_enabled,
            json_escape(self.vpt_debug_view),
            self.denoiser_enabled,
            json_escape(self.denoiser_mode),
            json_escape(self.effective_denoiser_mode)
        )
    }
}

impl RenderCapture {
    pub fn from_env() -> Result<Option<Self>> {
        CaptureConfig::from_env().map(|config| config.map(Self::new))
    }

    pub fn new(config: CaptureConfig) -> Self {
        Self {
            config,
            readback: None,
            readback_extent: None,
        }
    }

    pub fn config(&self) -> &CaptureConfig {
        &self.config
    }

    pub fn should_capture(&self, frame_index: u64) -> bool {
        self.config.target_frame == Some(frame_index)
    }

    pub fn ensure_readback(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<&GpuBuffer> {
        let byte_size = rgba8_byte_size(width, height)?;
        let extent = [width, height];
        let needs_recreate = self.readback_extent != Some(extent)
            || self
                .readback
                .as_ref()
                .is_none_or(|buffer| buffer.size < byte_size);

        if needs_recreate {
            if let Some(buffer) = self.readback.take() {
                buffer.destroy(device, allocator);
            }
            self.readback = Some(GpuBuffer::new(
                device,
                allocator,
                byte_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "postprocess_capture_readback",
            )?);
            self.readback_extent = Some(extent);
        }

        self.readback
            .as_ref()
            .ok_or_else(|| anyhow!("capture readback buffer was not created"))
    }

    pub fn write_rgba8_capture(&self, metadata: &CaptureMetadata) -> Result<()> {
        let buffer = self
            .readback
            .as_ref()
            .ok_or_else(|| anyhow!("capture readback buffer is missing"))?;
        let needed = rgba8_byte_size(metadata.width, metadata.height)? as usize;
        let mapped = buffer
            .mapped_slice()
            .ok_or_else(|| anyhow!("capture readback buffer is not host-visible"))?;
        if mapped.len() < needed {
            bail!(
                "capture readback buffer is too small: has {} bytes, needs {} bytes",
                mapped.len(),
                needed
            );
        }

        let ppm = encode_ppm_rgba8(metadata.width, metadata.height, &mapped[..needed])?;
        create_parent_dir(&metadata.ppm_path)?;
        fs::write(&metadata.ppm_path, ppm).with_context(|| {
            format!(
                "failed to write capture PPM to {}",
                metadata.ppm_path.display()
            )
        })?;
        fs::write(&metadata.json_path, metadata.to_json()).with_context(|| {
            format!(
                "failed to write capture metadata to {}",
                metadata.json_path.display()
            )
        })?;
        Ok(())
    }

    pub fn destroy(mut self, device: &ash::Device, allocator: &GpuAllocator) {
        if let Some(buffer) = self.readback.take() {
            buffer.destroy(device, allocator);
        }
    }
}

pub fn cmd_copy_image_to_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_extent: vk::Extent3D,
    dst_buffer: vk::Buffer,
) {
    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(src_extent);

    unsafe {
        device.cmd_copy_image_to_buffer(
            command_buffer,
            src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_buffer,
            &[region],
        );
    }
}

pub fn encode_ppm_rgba8(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>> {
    let expected = rgba8_byte_size(width, height)? as usize;
    if rgba.len() != expected {
        bail!(
            "RGBA8 capture size mismatch: got {} bytes, expected {} bytes",
            rgba.len(),
            expected
        );
    }

    let mut out = Vec::with_capacity(ppm_byte_size(width, height)?);
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    for pixel in rgba.chunks_exact(4) {
        out.extend_from_slice(&pixel[0..3]);
    }
    Ok(out)
}

fn rgba8_byte_size(width: u32, height: u32) -> Result<vk::DeviceSize> {
    if width == 0 || height == 0 {
        bail!("capture dimensions must be non-zero");
    }
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(RGBA8_BYTES_PER_PIXEL))
        .ok_or_else(|| anyhow!("capture dimensions overflow RGBA8 byte size"))
}

fn ppm_byte_size(width: u32, height: u32) -> Result<usize> {
    let header_len = format!("P6\n{width} {height}\n255\n").len();
    let rgb_len = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| anyhow!("capture dimensions overflow PPM byte size"))?;
    usize::try_from(rgb_len)
        .ok()
        .and_then(|rgb_len| rgb_len.checked_add(header_len))
        .ok_or_else(|| anyhow!("capture PPM byte size does not fit usize"))
}

fn parse_target_frame(value: &str) -> Result<u64> {
    let value = value.trim();
    value
        .parse::<u64>()
        .with_context(|| "REVOLUMETRIC_CAPTURE_FRAME must be a non-negative integer")
}

fn sanitize_prefix(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn create_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create capture directory {}", parent.display()))?;
    }
    Ok(())
}

fn json_escape_path(path: &Path) -> String {
    json_escape(&path.to_string_lossy())
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\u{:04x}", ch as u32);
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn capture_config_parses_env_values() {
        let config = CaptureConfig::from_values(
            Some("42"),
            Some("target/debug-captures"),
            Some("restir_on"),
        )
        .expect("valid capture config should parse");

        assert_eq!(config.target_frame, Some(42));
        assert_eq!(config.output_dir, PathBuf::from("target/debug-captures"));
        assert_eq!(config.prefix, "restir_on");
    }

    #[test]
    fn capture_config_rejects_invalid_frame() {
        let error = CaptureConfig::from_values(Some("soon"), None, None).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("REVOLUMETRIC_CAPTURE_FRAME must be a non-negative integer")
        );
    }

    #[test]
    fn ppm_encoder_strips_rgba_alpha() {
        let rgba = [
            1, 2, 3, 255, //
            4, 5, 6, 128,
        ];

        let encoded = encode_ppm_rgba8(2, 1, &rgba).expect("valid rgba8 should encode");

        assert_eq!(&encoded[..11], b"P6\n2 1\n255\n");
        assert_eq!(&encoded[11..], &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn ppm_encoder_rejects_wrong_rgba_size() {
        let error = encode_ppm_rgba8(2, 1, &[1, 2, 3]).unwrap_err();

        assert!(error.to_string().contains("RGBA8 capture size mismatch"));
    }

    #[test]
    fn metadata_json_records_frame_settings_and_paths() {
        let metadata = CaptureMetadata {
            frame_index: 7,
            vpt_sample_index: 3,
            width: 320,
            height: 180,
            source: "postprocess_output",
            ppm_path: PathBuf::from("target/captures/restir_000007.ppm"),
            json_path: PathBuf::from("target/captures/restir_000007.json"),
            render_backend: "vpt",
            render_mode: "rt",
            rt_debug_view: "surface",
            rt_restir_di_enabled: true,
            rt_restir_di_spatial_enabled: true,
            rt_restir_di_spatial_sample_count: 4,
            rt_restir_gi_enabled: true,
            rt_temporal_denoise_enabled: true,
            restir_di_enabled: true,
            restir_di_temporal_enabled: true,
            restir_di_spatial_enabled: false,
            area_restir_enabled: true,
            area_restir_temporal_enabled: false,
            area_restir_spatial_enabled: false,
            vpt_debug_view: "final",
            denoiser_enabled: true,
            denoiser_mode: "relax",
            effective_denoiser_mode: "svgf",
        };

        let json = metadata.to_json();

        assert!(json.contains("\"frame_index\": 7"));
        assert!(json.contains("\"vpt_sample_index\": 3"));
        assert!(json.contains("\"source\": \"postprocess_output\""));
        assert!(json.contains("\"render_backend\": \"vpt\""));
        assert!(json.contains("\"render_mode\": \"rt\""));
        assert!(json.contains("\"rt_debug_view\": \"surface\""));
        assert!(json.contains("\"rt_restir_di_enabled\": true"));
        assert!(json.contains("\"rt_restir_di_spatial_enabled\": true"));
        assert!(json.contains("\"rt_restir_di_spatial_sample_count\": 4"));
        assert!(json.contains("\"rt_restir_gi_enabled\": true"));
        assert!(json.contains("\"rt_temporal_denoise_enabled\": true"));
        assert!(json.contains("\"restir_di_enabled\": true"));
        assert!(json.contains("\"area_restir_enabled\": true"));
        assert!(json.contains("\"vpt_debug_view\": \"final\""));
        assert!(json.contains("\"denoiser_enabled\": true"));
        assert!(json.contains("\"denoiser_mode\": \"relax\""));
        assert!(json.contains("\"effective_denoiser_mode\": \"svgf\""));
    }
}
