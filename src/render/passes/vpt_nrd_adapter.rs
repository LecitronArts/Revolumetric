use anyhow::{Context, Result};
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::nrd_adapter::{
    NrdDispatchSnapshot, NrdInstance, NrdInstanceSnapshot, NrdLibraryDesc, NrdTextureImageDesc,
};
use crate::render::passes::vpt_nrd_confidence::VptNrdConfidenceResources;
use crate::render::passes::vpt_nrd_frontend::VptNrdPackedResources;
use crate::render::passes::vpt_surface::VptCurrentSurfaceResources;
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::SceneUniformBuffer;

pub struct VptNrdAdapterPass {
    pub nrd_diff_radiance_hitdist: GpuImage,
    pub nrd_validation: GpuImage,
    backend: VptNrdAdapterBackend,
}

pub enum VptNrdAdapterBackend {
    Ready(VptNrdReadyBackend),
    Unavailable(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VptNrdAdapterBackendMetadata {
    pub library_desc: NrdLibraryDesc,
    pub constant_buffer_max_data_size: u32,
    pub sampler_count: usize,
    pub pipeline_count: usize,
    pub permanent_pool_size: usize,
    pub transient_pool_size: usize,
    pub dispatch_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdTexturePoolPlan {
    pub permanent: Vec<VptNrdTexturePoolImagePlan>,
    pub transient: Vec<VptNrdTexturePoolImagePlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdTexturePoolImagePlan {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub downsample_factor: u16,
}

pub struct VptNrdReadyBackend {
    _instance: NrdInstance,
    library_desc: NrdLibraryDesc,
    instance_snapshot: NrdInstanceSnapshot,
    texture_pool_plan: VptNrdTexturePoolPlan,
    dispatches: Vec<NrdDispatchSnapshot>,
}

pub struct VptNrdAdapterGraphInputs<'a> {
    pub frame_slot: usize,
    pub packed: VptNrdPackedResources,
    pub confidence: VptNrdConfidenceResources,
    pub surface_inputs: VptCurrentSurfaceResources,
    pub profiler: Option<&'a GpuProfiler>,
}

#[derive(Clone, Copy)]
pub struct VptNrdAdapterGraphOutputs {
    pub resources: VptNrdAdapterResources,
}

#[derive(Clone, Copy)]
pub struct VptNrdAdapterResources {
    pub diff_radiance_hitdist: ResourceHandle,
    pub validation: ResourceHandle,
}

pub struct VptNrdAdapterPassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
}

pub struct VptNrdAdapterPassResizeInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
}

impl VptNrdAdapterPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdAdapterPassCreateInfo<'_>,
    ) -> Result<Self> {
        let _ = info.scene_ubo.frame_count();
        let backend = VptNrdAdapterBackend::initialize_relax(info.width, info.height);
        if let Some(reason) = backend.unavailable_reason() {
            tracing::warn!(
                reason,
                "VPT NRD adapter backend unavailable; RELAX dispatch remains disabled"
            );
        }
        let images = create_adapter_images(device, allocator, info.width, info.height)?;
        Ok(Self {
            nrd_diff_radiance_hitdist: images.nrd_diff_radiance_hitdist,
            nrd_validation: images.nrd_validation,
            backend,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdAdapterPassResizeInfo<'_>,
    ) -> Result<()> {
        let _ = info.scene_ubo.frame_count();
        let new_images = create_adapter_images(device, allocator, info.width, info.height)?;
        let old_images = VptNrdAdapterImages {
            nrd_diff_radiance_hitdist: std::mem::replace(
                &mut self.nrd_diff_radiance_hitdist,
                new_images.nrd_diff_radiance_hitdist,
            ),
            nrd_validation: std::mem::replace(&mut self.nrd_validation, new_images.nrd_validation),
        };
        old_images.destroy(device, allocator);
        Ok(())
    }

    pub fn record(&self, _device: &ash::Device, _cmd: vk::CommandBuffer, _frame_slot: usize) {
        let _ = (
            self.backend.is_ready(),
            self.backend.dispatch_count(),
            self.backend.unavailable_reason(),
            self.backend.ready_metadata(),
            self.backend.texture_pool_plan(),
        );
    }

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        inputs: VptNrdAdapterGraphInputs<'a>,
    ) -> VptNrdAdapterGraphOutputs {
        let VptNrdAdapterGraphInputs {
            frame_slot,
            packed,
            confidence,
            surface_inputs,
            profiler,
        } = inputs;
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let diff_resource = graph.import_image_with_access(
            self.nrd_diff_radiance_hitdist.handle,
            self.nrd_diff_radiance_hitdist.extent.width,
            self.nrd_diff_radiance_hitdist.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );
        let validation_resource = graph.import_image_with_access(
            self.nrd_validation.handle,
            self.nrd_validation.extent.width,
            self.nrd_validation.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );

        let adapter_writes = graph.add_pass("vpt_nrd_adapter", QueueType::Compute, |builder| {
            builder.read_as(packed.diff_radiance_hitdist, AccessKind::ComputeShaderRead);
            builder.read_as(packed.spec_radiance_hitdist, AccessKind::ComputeShaderRead);
            builder.read_as(packed.residual_radiance, AccessKind::ComputeShaderRead);
            builder.read_as(packed.material_factors, AccessKind::ComputeShaderRead);
            builder.read_as(confidence.diff_confidence, AccessKind::ComputeShaderRead);
            builder.read_as(confidence.spec_confidence, AccessKind::ComputeShaderRead);
            builder.read_as(
                surface_inputs.normal_roughness,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(
                surface_inputs.material_roughness,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(surface_inputs.view_z, AccessKind::ComputeShaderRead);
            builder.read_as(surface_inputs.motion_history, AccessKind::ComputeShaderRead);
            builder.write_as(diff_resource, AccessKind::ComputeShaderWrite);
            builder.write_as(validation_resource, AccessKind::ComputeShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptNrdAdapter,
                    );
                }
                self.record(ctx.device, ctx.command_buffer, frame_slot);
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptNrdAdapter,
                    );
                }
            })
        });

        VptNrdAdapterGraphOutputs {
            resources: VptNrdAdapterResources {
                diff_radiance_hitdist: adapter_writes[0],
                validation: adapter_writes[1],
            },
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.nrd_diff_radiance_hitdist.destroy(device, allocator);
        self.nrd_validation.destroy(device, allocator);
    }
}

impl VptNrdAdapterBackend {
    pub fn initialize_relax(width: u32, height: u32) -> Self {
        match VptNrdReadyBackend::initialize_relax(width, height) {
            Ok(backend) => Self::Ready(backend),
            Err(error) => Self::Unavailable(error.to_string()),
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }

    pub fn dispatch_count(&self) -> usize {
        match self {
            Self::Ready(backend) => backend.dispatches.len(),
            Self::Unavailable(_) => 0,
        }
    }

    pub fn ready_metadata(&self) -> Option<VptNrdAdapterBackendMetadata> {
        match self {
            Self::Ready(backend) => Some(backend.metadata()),
            Self::Unavailable(_) => None,
        }
    }

    pub fn texture_pool_plan(&self) -> Option<&VptNrdTexturePoolPlan> {
        match self {
            Self::Ready(backend) => Some(&backend.texture_pool_plan),
            Self::Unavailable(_) => None,
        }
    }

    pub fn unavailable_reason(&self) -> Option<&str> {
        match self {
            Self::Ready(_) => None,
            Self::Unavailable(reason) => Some(reason.as_str()),
        }
    }
}

impl VptNrdReadyBackend {
    fn initialize_relax(width: u32, height: u32) -> Result<Self> {
        let instance = NrdInstance::relax_diffuse(width, height)?;
        let library_desc = NrdInstance::library_desc()?;
        let instance_snapshot = instance.instance_snapshot()?;
        let texture_pool_plan =
            VptNrdTexturePoolPlan::from_instance_snapshot(width, height, &instance_snapshot)
                .context("failed to build VPT NRD texture pool plan")?;
        Ok(Self {
            _instance: instance,
            library_desc,
            instance_snapshot,
            texture_pool_plan,
            dispatches: Vec::new(),
        })
    }

    fn metadata(&self) -> VptNrdAdapterBackendMetadata {
        VptNrdAdapterBackendMetadata {
            library_desc: self.library_desc,
            constant_buffer_max_data_size: self.instance_snapshot.constant_buffer_max_data_size,
            sampler_count: self.instance_snapshot.samplers.len(),
            pipeline_count: self.instance_snapshot.pipelines.len(),
            permanent_pool_size: self.instance_snapshot.permanent_pool.len(),
            transient_pool_size: self.instance_snapshot.transient_pool.len(),
            dispatch_count: self.dispatches.len(),
        }
    }
}

impl VptNrdTexturePoolPlan {
    pub fn from_instance_snapshot(
        width: u32,
        height: u32,
        snapshot: &NrdInstanceSnapshot,
    ) -> Result<Self> {
        Ok(Self {
            permanent: build_pool_plan(
                "vpt_nrd_permanent",
                width,
                height,
                &snapshot.permanent_pool,
            )?,
            transient: build_pool_plan(
                "vpt_nrd_transient",
                width,
                height,
                &snapshot.transient_pool,
            )?,
        })
    }
}

fn build_pool_plan(
    prefix: &str,
    width: u32,
    height: u32,
    textures: &[crate::render::nrd_adapter::NrdTextureDesc],
) -> Result<Vec<VptNrdTexturePoolImagePlan>> {
    textures
        .iter()
        .enumerate()
        .map(|(index, texture)| {
            let image_desc = texture
                .image_desc()
                .with_context(|| format!("failed to map {prefix}_{index} texture format"))?;
            Ok(texture_pool_image_plan(
                prefix, index, width, height, image_desc,
            ))
        })
        .collect()
}

fn texture_pool_image_plan(
    prefix: &str,
    index: usize,
    width: u32,
    height: u32,
    image_desc: NrdTextureImageDesc,
) -> VptNrdTexturePoolImagePlan {
    let downsample_factor = image_desc.downsample_factor.max(1);
    VptNrdTexturePoolImagePlan {
        name: format!("{prefix}_{index}"),
        width: divide_round_up(width, downsample_factor),
        height: divide_round_up(height, downsample_factor),
        format: image_desc.format,
        downsample_factor,
    }
}

fn divide_round_up(value: u32, divisor: u16) -> u32 {
    value.div_ceil(u32::from(divisor))
}

struct VptNrdAdapterImages {
    nrd_diff_radiance_hitdist: GpuImage,
    nrd_validation: GpuImage,
}

impl VptNrdAdapterImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.nrd_diff_radiance_hitdist.destroy(device, allocator);
        self.nrd_validation.destroy(device, allocator);
    }
}

fn create_adapter_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptNrdAdapterImages> {
    let nrd_diff_radiance_hitdist = create_adapter_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_adapter_diff_radiance_hitdist",
    )?;
    let nrd_validation = match create_adapter_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_adapter_validation",
    ) {
        Ok(image) => image,
        Err(error) => {
            nrd_diff_radiance_hitdist.destroy(device, allocator);
            return Err(error);
        }
    };
    Ok(VptNrdAdapterImages {
        nrd_diff_radiance_hitdist,
        nrd_validation,
    })
}

fn create_adapter_image(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
    name: &'static str,
) -> Result<GpuImage> {
    GpuImage::new(
        device,
        allocator,
        &GpuImageDesc {
            width,
            height,
            depth: 1,
            format: vk::Format::R16G16B16A16_SFLOAT,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            aspect: vk::ImageAspectFlags::COLOR,
            name,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::nrd_adapter::{NrdPipelineSnapshot, NrdSamplerDesc, NrdTextureFormat};

    #[cfg(not(feature = "nrd"))]
    #[test]
    fn default_backend_reports_nrd_unavailable_without_dispatch_readiness() {
        let backend = VptNrdAdapterBackend::initialize_relax(1280, 720);

        assert!(!backend.is_ready());
        assert_eq!(backend.dispatch_count(), 0);
        assert_eq!(backend.ready_metadata(), None);
        assert!(
            backend
                .unavailable_reason()
                .expect("default build should report why NRD is unavailable")
                .contains("nrd Cargo feature is disabled")
        );
    }

    #[test]
    fn texture_pool_plan_maps_nrd_pool_descriptors_to_image_extents() {
        let snapshot = instance_snapshot_with_pools(
            &[
                texture(NrdTextureFormat::Rgba16Sfloat, 1),
                texture(NrdTextureFormat::R16Sfloat, 2),
            ],
            &[texture(NrdTextureFormat::Rg16Sfloat, 4)],
        );

        let plan = VptNrdTexturePoolPlan::from_instance_snapshot(1920, 1080, &snapshot).unwrap();

        assert_eq!(plan.permanent[0].width, 1920);
        assert_eq!(plan.permanent[0].height, 1080);
        assert_eq!(plan.permanent[0].format, vk::Format::R16G16B16A16_SFLOAT);
        assert_eq!(plan.permanent[1].width, 960);
        assert_eq!(plan.permanent[1].height, 540);
        assert_eq!(plan.permanent[1].format, vk::Format::R16_SFLOAT);
        assert_eq!(plan.transient[0].width, 480);
        assert_eq!(plan.transient[0].height, 270);
        assert_eq!(plan.transient[0].format, vk::Format::R16G16_SFLOAT);
    }

    #[test]
    fn texture_pool_plan_ceil_divides_downsampled_extents_and_rejects_unknown_formats() {
        let snapshot = instance_snapshot_with_pools(
            &[texture(NrdTextureFormat::R16Sfloat, 2)],
            &[texture(NrdTextureFormat::Unknown, 1)],
        );

        let error =
            VptNrdTexturePoolPlan::from_instance_snapshot(1919, 1079, &snapshot).unwrap_err();

        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains("unsupported NRD texture format"))
        );

        let snapshot =
            instance_snapshot_with_pools(&[texture(NrdTextureFormat::R16Sfloat, 2)], &[]);
        let plan = VptNrdTexturePoolPlan::from_instance_snapshot(1919, 1079, &snapshot).unwrap();
        assert_eq!(plan.permanent[0].width, 960);
        assert_eq!(plan.permanent[0].height, 540);
    }

    fn texture(
        format: NrdTextureFormat,
        downsample_factor: u16,
    ) -> crate::render::nrd_adapter::NrdTextureDesc {
        crate::render::nrd_adapter::NrdTextureDesc {
            format: format as u32,
            downsample_factor,
            reserved0: 0,
        }
    }

    fn instance_snapshot_with_pools(
        permanent_pool: &[crate::render::nrd_adapter::NrdTextureDesc],
        transient_pool: &[crate::render::nrd_adapter::NrdTextureDesc],
    ) -> NrdInstanceSnapshot {
        NrdInstanceSnapshot {
            constant_buffer_and_samplers_space_index: 0,
            resources_space_index: 0,
            constant_buffer_register_index: 0,
            samplers_base_register_index: 0,
            resources_base_register_index: 0,
            constant_buffer_max_data_size: 0,
            samplers: vec![NrdSamplerDesc { mode: 0 }],
            pipelines: vec![NrdPipelineSnapshot {
                spirv_bytecode: vec![0x0723_0203],
                resource_ranges: Vec::new(),
                has_constant_data: false,
                shader_identifier: "test".to_owned(),
            }],
            permanent_pool: permanent_pool.to_vec(),
            transient_pool: transient_pool.to_vec(),
        }
    }
}
