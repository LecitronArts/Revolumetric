use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::nrd_adapter::{
    NrdCommonSettings, NrdDescriptorType, NrdDispatchSnapshot, NrdInstance, NrdInstanceSnapshot,
    NrdLibraryDesc, NrdNormalEncoding, NrdReblurDiffuseSettings, NrdRelaxDiffuseSettings,
    NrdResourceType, NrdRoughnessEncoding, NrdSamplerDesc, NrdTextureImageDesc,
};
use crate::render::pass_context::PassBuilder;
use crate::render::passes::vpt_nrd_confidence::{VptNrdConfidencePass, VptNrdConfidenceResources};
use crate::render::passes::vpt_nrd_frontend::{VptNrdFrontendPass, VptNrdPackedResources};
use crate::render::passes::vpt_surface::{VptCurrentSurfaceResources, VptSurfacePass};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::{SceneUniformBuffer, VptDenoiserMode};

pub struct VptNrdAdapterPass {
    pub nrd_diff_radiance_hitdist: GpuImage,
    pub nrd_validation: GpuImage,
    denoiser_mode: VptDenoiserMode,
    constant_buffer_alignment: vk::DeviceSize,
    backend: VptNrdAdapterBackend,
    texture_pools: Option<VptNrdTexturePools>,
    shared_descriptor_resources: Option<VptNrdSharedDescriptorResources>,
    descriptor_resources: Option<VptNrdDescriptorResources>,
    pipeline_resources: Option<VptNrdPipelineResources>,
    descriptor_update_plan: Option<VptNrdDescriptorUpdatePlan>,
}

pub enum VptNrdAdapterBackend {
    Ready(Box<VptNrdReadyBackend>),
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
    pub dispatch_resource_plan_count: usize,
    pub pipeline_layout_plan_count: usize,
    pub pipeline_shader_plan_count: usize,
    pub pipeline_create_plan_count: usize,
    pub pipeline_descriptor_binding_plan_count: usize,
    pub descriptor_pool_size_count: usize,
    pub dispatch_descriptor_write_plan_count: usize,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdAccumulationMode {
    Continue = 0,
    Restart = 1,
    ClearAndRestart = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdCheckerboardMode {
    Off = 0,
    Black = 1,
    White = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdHitDistanceReconstructionMode {
    Off = 0,
    Area3x3 = 1,
    Area5x5 = 2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VptNrdFrameSettingsInputs {
    pub current_world_to_view: glam::Mat4,
    pub previous_world_to_view: glam::Mat4,
    pub current_view_to_clip: glam::Mat4,
    pub previous_view_to_clip: glam::Mat4,
    pub current_resolution: [u32; 2],
    pub previous_resolution: [u32; 2],
    pub frame_index: u32,
    pub time_delta_seconds: f32,
    pub reset_history: bool,
    pub history_confidence_available: bool,
    pub relax_atrous_iteration_num: u32,
    pub enable_validation: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VptNrdFrameSettings {
    pub common: NrdCommonSettings,
    pub relax_diffuse: NrdRelaxDiffuseSettings,
    pub reblur_diffuse: NrdReblurDiffuseSettings,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdDispatchResourcePlan {
    pub bindings: Vec<VptNrdDispatchResourceBindingPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdPipelineLayoutPlan {
    pub resource_ranges: Vec<VptNrdPipelineResourceRangePlan>,
    pub has_constant_data: bool,
    pub shader_identifier: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdPipelineShaderPlan {
    pub spirv_words: Vec<u32>,
    pub shader_identifier: String,
    pub has_constant_data: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdPipelineCreatePlan {
    pub pipeline_index: usize,
    pub shader_plan_index: usize,
    pub descriptor_set_layout_index: usize,
    pub descriptor_set_index: usize,
    pub shader_identifier: String,
    pub has_constant_data: bool,
    pub descriptor_binding_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdPipelineResourcesPlan {
    pub pipeline_index: usize,
    pub shader_identifier: String,
    pub spirv_bytes: Vec<u8>,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub shared_set_index: u32,
    pub resource_set_index: u32,
    pub resource_descriptor_set: vk::DescriptorSet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VptNrdPipelineResourceRangePlan {
    pub descriptor_type: NrdDescriptorType,
    pub descriptors_num: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdPipelineDescriptorBindingPlan {
    pub bindings: Vec<VptNrdPipelineDescriptorBinding>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VptNrdPipelineDescriptorBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdSharedDescriptorBindingPlan {
    pub bindings: Vec<VptNrdSharedDescriptorBinding>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VptNrdSharedDescriptorBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdPipelineSetLayoutPlan {
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub shared_set_index: u32,
    pub resources_set_index: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdDescriptorPoolPlan {
    pub max_sets: u32,
    pub pool_sizes: Vec<VptNrdDescriptorPoolSizePlan>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VptNrdDescriptorPoolSizePlan {
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VptNrdDispatchResourceBindingPlan {
    pub descriptor_type: NrdDescriptorType,
    pub resource: VptNrdDispatchResource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdDispatchDescriptorWritePlan {
    pub pipeline_index: usize,
    pub writes: Vec<VptNrdDispatchDescriptorWrite>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VptNrdDispatchDescriptorWrite {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_type: vk::DescriptorType,
    pub resource: VptNrdDispatchResource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdDispatchConstantUploadPlan {
    pub dispatches: Vec<VptNrdDispatchConstantUpload>,
    pub max_data_size: vk::DeviceSize,
    pub slot_stride: vk::DeviceSize,
    pub frame_count: usize,
    pub total_size: vk::DeviceSize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdDispatchConstantUpload {
    pub dispatch_index: usize,
    pub data: Vec<u8>,
    pub matches_previous_dispatch: bool,
}

pub struct VptNrdAdapterImageInputs<'a> {
    pub motion: &'a GpuImage,
    pub normal_roughness: &'a GpuImage,
    pub view_z: &'a GpuImage,
    pub diff_confidence: &'a GpuImage,
    pub spec_confidence: &'a GpuImage,
    pub diff_radiance_hitdist: &'a GpuImage,
    pub output_diff_radiance_hitdist: &'a GpuImage,
    pub validation: &'a GpuImage,
    pub permanent_pool: &'a [GpuImage],
    pub transient_pool: &'a [GpuImage],
}

pub struct VptNrdAdapterPassImageRefs<'a> {
    pub frontend: &'a VptNrdFrontendPass,
    pub confidence: &'a VptNrdConfidencePass,
    pub surface: &'a VptSurfacePass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdResolvedDispatchDescriptorWritePlan {
    pub pipeline_index: usize,
    pub writes: Vec<VptNrdResolvedDescriptorImageWrite>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VptNrdResolvedDescriptorImageWrite {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_type: vk::DescriptorType,
    pub image_view: vk::ImageView,
    pub image_layout: vk::ImageLayout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdDescriptorUpdatePlan {
    pub dispatches: Vec<VptNrdDispatchDescriptorUpdatePlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VptNrdDispatchDescriptorUpdatePlan {
    pub pipeline_index: usize,
    pub descriptor_set: vk::DescriptorSet,
    pub writes: Vec<VptNrdDescriptorImageUpdate>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VptNrdDescriptorImageUpdate {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_type: vk::DescriptorType,
    pub image_view: vk::ImageView,
    pub image_layout: vk::ImageLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VptNrdDispatchResource {
    Motion,
    NormalRoughness,
    ViewZ,
    DiffConfidence,
    SpecConfidence,
    DiffRadianceHitdist,
    OutputDiffRadianceHitdist,
    Validation,
    PermanentPool { index: usize },
    TransientPool { index: usize },
}

pub struct VptNrdReadyBackend {
    denoiser_mode: VptDenoiserMode,
    instance: NrdInstance,
    state: VptNrdReadyBackendState,
}

struct VptNrdReadyBackendState {
    library_desc: NrdLibraryDesc,
    instance_snapshot: NrdInstanceSnapshot,
    texture_pool_plan: VptNrdTexturePoolPlan,
    pipeline_layout_plans: Vec<VptNrdPipelineLayoutPlan>,
    pipeline_shader_plans: Vec<VptNrdPipelineShaderPlan>,
    pipeline_create_plans: Vec<VptNrdPipelineCreatePlan>,
    pipeline_descriptor_binding_plans: Vec<VptNrdPipelineDescriptorBindingPlan>,
    descriptor_pool_plan: VptNrdDescriptorPoolPlan,
    dispatches: Vec<NrdDispatchSnapshot>,
    dispatch_resource_plans: Vec<VptNrdDispatchResourcePlan>,
    dispatch_descriptor_write_plans: Vec<VptNrdDispatchDescriptorWritePlan>,
}

struct VptNrdTexturePools {
    permanent_pool: Vec<GpuImage>,
    transient_pool: Vec<GpuImage>,
}

impl VptNrdTexturePoolGraphResources {
    fn declare_read_write(&self, builder: &mut PassBuilder) {
        for &image in &self.images {
            builder.write_as(image, AccessKind::ComputeShaderReadWrite);
        }
    }
}

struct VptNrdSharedDescriptorResources {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    samplers: Vec<vk::Sampler>,
    constant_binding: u32,
    sampler_bindings: Vec<u32>,
    constant_buffer: GpuBuffer,
    constant_upload_plan: VptNrdDispatchConstantUploadPlan,
}

struct VptNrdDescriptorResources {
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    frame_count: usize,
    dispatch_slot_count: usize,
    pipeline_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VptNrdTexturePoolGraphResources {
    images: Vec<ResourceHandle>,
}

struct VptNrdPipelineResources {
    pipelines: Vec<VptNrdComputePipeline>,
    empty_descriptor_set_layout: vk::DescriptorSetLayout,
}

struct VptNrdComputePipeline {
    pipeline: ComputePipeline,
    shared_set_index: u32,
    resource_set_index: u32,
}

pub struct VptNrdAdapterGraphInputs<'a> {
    pub frame_slot: usize,
    pub packed: VptNrdPackedResources,
    pub confidence: VptNrdConfidenceResources,
    pub surface_inputs: VptCurrentSurfaceResources,
    pub texture_pools_initialized: bool,
    pub profiler: Option<&'a GpuProfiler>,
}

#[derive(Clone, Copy)]
pub struct VptNrdAdapterGraphOutputs<'a> {
    pub resources: VptNrdAdapterResources,
    pub validation_image: &'a GpuImage,
}

#[derive(Clone, Copy)]
pub struct VptNrdAdapterResources {
    pub diff_radiance_hitdist: ResourceHandle,
    pub validation: ResourceHandle,
}

pub struct VptNrdAdapterPassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub denoiser_mode: VptDenoiserMode,
    pub relax_atrous_iteration_num: u32,
    pub constant_buffer_alignment: vk::DeviceSize,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub image_refs: VptNrdAdapterPassImageRefs<'a>,
}

pub struct VptNrdAdapterPassResizeInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub denoiser_mode: VptDenoiserMode,
    pub relax_atrous_iteration_num: u32,
    pub constant_buffer_alignment: vk::DeviceSize,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub image_refs: VptNrdAdapterPassImageRefs<'a>,
}

impl VptNrdAdapterPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdAdapterPassCreateInfo<'_>,
    ) -> Result<Self> {
        let _ = info.scene_ubo.frame_count();
        let backend = VptNrdAdapterBackend::initialize(
            info.denoiser_mode,
            info.width,
            info.height,
            info.relax_atrous_iteration_num,
        );
        if let Some(reason) = backend.unavailable_reason() {
            tracing::warn!(
                reason,
                denoiser_mode = info.denoiser_mode.as_config_value(),
                "VPT NRD adapter backend unavailable; NRD dispatch remains disabled"
            );
        }
        let texture_pools = create_texture_pools(device, allocator, &backend)?;
        let shared_descriptor_resources = match create_shared_descriptor_resources(
            device,
            allocator,
            &backend,
            info.scene_ubo.frame_count(),
            info.constant_buffer_alignment,
        ) {
            Ok(resources) => resources,
            Err(error) => {
                if let Some(texture_pools) = texture_pools {
                    texture_pools.destroy(device, allocator);
                }
                return Err(error);
            }
        };
        let descriptor_resources =
            match create_descriptor_resources(device, &backend, info.scene_ubo.frame_count()) {
                Ok(resources) => resources,
                Err(error) => {
                    if let Some(shared_descriptor_resources) = shared_descriptor_resources {
                        shared_descriptor_resources.destroy(device, allocator);
                    }
                    if let Some(texture_pools) = texture_pools {
                        texture_pools.destroy(device, allocator);
                    }
                    return Err(error);
                }
            };
        let pipeline_resources = match create_pipeline_resources(
            device,
            &backend,
            shared_descriptor_resources.as_ref(),
            descriptor_resources.as_ref(),
        ) {
            Ok(resources) => resources,
            Err(error) => {
                if let Some(descriptor_resources) = descriptor_resources {
                    descriptor_resources.destroy(device);
                }
                if let Some(shared_descriptor_resources) = shared_descriptor_resources {
                    shared_descriptor_resources.destroy(device, allocator);
                }
                if let Some(texture_pools) = texture_pools {
                    texture_pools.destroy(device, allocator);
                }
                return Err(error);
            }
        };
        let images = match create_adapter_images(device, allocator, info.width, info.height) {
            Ok(images) => images,
            Err(error) => {
                if let Some(pipeline_resources) = pipeline_resources {
                    pipeline_resources.destroy(device);
                }
                if let Some(descriptor_resources) = descriptor_resources {
                    descriptor_resources.destroy(device);
                }
                if let Some(shared_descriptor_resources) = shared_descriptor_resources {
                    shared_descriptor_resources.destroy(device, allocator);
                }
                if let Some(texture_pools) = texture_pools {
                    texture_pools.destroy(device, allocator);
                }
                return Err(error);
            }
        };
        let descriptor_update_plan = match build_descriptor_update_plan(
            &backend,
            descriptor_resources.as_ref(),
            texture_pools.as_ref(),
            &images.nrd_diff_radiance_hitdist,
            &images.nrd_validation,
            info.image_refs,
            0,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                images.destroy(device, allocator);
                if let Some(pipeline_resources) = pipeline_resources {
                    pipeline_resources.destroy(device);
                }
                if let Some(descriptor_resources) = descriptor_resources {
                    descriptor_resources.destroy(device);
                }
                if let Some(shared_descriptor_resources) = shared_descriptor_resources {
                    shared_descriptor_resources.destroy(device, allocator);
                }
                if let Some(texture_pools) = texture_pools {
                    texture_pools.destroy(device, allocator);
                }
                return Err(error).context("failed to build VPT NRD descriptor update plan");
            }
        };
        Ok(Self {
            nrd_diff_radiance_hitdist: images.nrd_diff_radiance_hitdist,
            nrd_validation: images.nrd_validation,
            denoiser_mode: info.denoiser_mode,
            constant_buffer_alignment: info.constant_buffer_alignment,
            backend,
            texture_pools,
            shared_descriptor_resources,
            descriptor_resources,
            pipeline_resources,
            descriptor_update_plan,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdAdapterPassResizeInfo<'_>,
    ) -> Result<()> {
        let _ = info.scene_ubo.frame_count();
        let new_backend = VptNrdAdapterBackend::initialize(
            info.denoiser_mode,
            info.width,
            info.height,
            info.relax_atrous_iteration_num,
        );
        if let Some(reason) = new_backend.unavailable_reason() {
            tracing::warn!(
                reason,
                denoiser_mode = info.denoiser_mode.as_config_value(),
                "VPT NRD adapter backend unavailable after resize; NRD dispatch remains disabled"
            );
        }
        let new_images = create_adapter_images(device, allocator, info.width, info.height)?;
        let new_texture_pools = match create_texture_pools(device, allocator, &new_backend) {
            Ok(texture_pools) => texture_pools,
            Err(error) => {
                new_images.destroy(device, allocator);
                return Err(error);
            }
        };
        let new_shared_descriptor_resources = match create_shared_descriptor_resources(
            device,
            allocator,
            &new_backend,
            info.scene_ubo.frame_count(),
            info.constant_buffer_alignment,
        ) {
            Ok(resources) => resources,
            Err(error) => {
                if let Some(new_texture_pools) = new_texture_pools {
                    new_texture_pools.destroy(device, allocator);
                }
                new_images.destroy(device, allocator);
                return Err(error);
            }
        };
        let new_descriptor_resources =
            match create_descriptor_resources(device, &new_backend, info.scene_ubo.frame_count()) {
                Ok(resources) => resources,
                Err(error) => {
                    if let Some(new_shared_descriptor_resources) = new_shared_descriptor_resources {
                        new_shared_descriptor_resources.destroy(device, allocator);
                    }
                    if let Some(new_texture_pools) = new_texture_pools {
                        new_texture_pools.destroy(device, allocator);
                    }
                    new_images.destroy(device, allocator);
                    return Err(error);
                }
            };
        let new_pipeline_resources = match create_pipeline_resources(
            device,
            &new_backend,
            new_shared_descriptor_resources.as_ref(),
            new_descriptor_resources.as_ref(),
        ) {
            Ok(resources) => resources,
            Err(error) => {
                if let Some(new_descriptor_resources) = new_descriptor_resources {
                    new_descriptor_resources.destroy(device);
                }
                if let Some(new_shared_descriptor_resources) = new_shared_descriptor_resources {
                    new_shared_descriptor_resources.destroy(device, allocator);
                }
                if let Some(new_texture_pools) = new_texture_pools {
                    new_texture_pools.destroy(device, allocator);
                }
                new_images.destroy(device, allocator);
                return Err(error);
            }
        };
        let descriptor_update_plan = match build_descriptor_update_plan(
            &new_backend,
            new_descriptor_resources.as_ref(),
            new_texture_pools.as_ref(),
            &new_images.nrd_diff_radiance_hitdist,
            &new_images.nrd_validation,
            info.image_refs,
            0,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                if let Some(new_pipeline_resources) = new_pipeline_resources {
                    new_pipeline_resources.destroy(device);
                }
                if let Some(new_descriptor_resources) = new_descriptor_resources {
                    new_descriptor_resources.destroy(device);
                }
                if let Some(new_shared_descriptor_resources) = new_shared_descriptor_resources {
                    new_shared_descriptor_resources.destroy(device, allocator);
                }
                if let Some(new_texture_pools) = new_texture_pools {
                    new_texture_pools.destroy(device, allocator);
                }
                new_images.destroy(device, allocator);
                return Err(error).context("failed to rebuild VPT NRD descriptor update plan");
            }
        };
        let old_images = VptNrdAdapterImages {
            nrd_diff_radiance_hitdist: std::mem::replace(
                &mut self.nrd_diff_radiance_hitdist,
                new_images.nrd_diff_radiance_hitdist,
            ),
            nrd_validation: std::mem::replace(&mut self.nrd_validation, new_images.nrd_validation),
        };
        let old_texture_pools = std::mem::replace(&mut self.texture_pools, new_texture_pools);
        let old_shared_descriptor_resources = std::mem::replace(
            &mut self.shared_descriptor_resources,
            new_shared_descriptor_resources,
        );
        let old_descriptor_resources =
            std::mem::replace(&mut self.descriptor_resources, new_descriptor_resources);
        let old_pipeline_resources =
            std::mem::replace(&mut self.pipeline_resources, new_pipeline_resources);
        let old_descriptor_update_plan =
            std::mem::replace(&mut self.descriptor_update_plan, descriptor_update_plan);
        self.denoiser_mode = info.denoiser_mode;
        let old_backend = std::mem::replace(&mut self.backend, new_backend);
        if let Some(old_pipeline_resources) = old_pipeline_resources {
            old_pipeline_resources.destroy(device);
        }
        if let Some(old_descriptor_resources) = old_descriptor_resources {
            old_descriptor_resources.destroy(device);
        }
        if let Some(old_shared_descriptor_resources) = old_shared_descriptor_resources {
            old_shared_descriptor_resources.destroy(device, allocator);
        }
        let _ = old_descriptor_update_plan;
        old_images.destroy(device, allocator);
        if let Some(old_texture_pools) = old_texture_pools {
            old_texture_pools.destroy(device, allocator);
        }
        drop(old_backend);
        Ok(())
    }

    pub fn update_frame_settings(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        settings: VptNrdFrameSettings,
        image_refs: VptNrdAdapterPassImageRefs<'_>,
        frame_slot: usize,
    ) -> Result<()> {
        let VptNrdAdapterBackend::Ready(_) = &self.backend else {
            return Ok(());
        };
        let frame_count = self
            .descriptor_resources
            .as_ref()
            .context("VPT NRD descriptor resources are missing for ready backend")?
            .frame_count;
        let current_dispatch_count = self
            .descriptor_resources
            .as_ref()
            .context("VPT NRD descriptor resources are missing for ready backend")?
            .dispatch_slot_count;
        let new_dispatch_count = {
            let VptNrdAdapterBackend::Ready(backend) = &mut self.backend else {
                unreachable!("backend was verified as ready above")
            };
            backend
                .update_frame_settings(&settings)
                .context("failed to refresh VPT NRD backend dispatches")?;
            backend.state.dispatches.len()
        };
        if new_dispatch_count != current_dispatch_count {
            self.rebuild_descriptor_resources(
                device,
                allocator,
                frame_count,
                image_refs,
                frame_slot,
            )
        } else {
            let descriptor_update_plan = build_descriptor_update_plan(
                &self.backend,
                self.descriptor_resources.as_ref(),
                self.texture_pools.as_ref(),
                &self.nrd_diff_radiance_hitdist,
                &self.nrd_validation,
                image_refs,
                frame_slot,
            )?;
            {
                let VptNrdAdapterBackend::Ready(backend) = &self.backend else {
                    unreachable!("backend was verified as ready above")
                };
                let shared_descriptor_resources = self
                    .shared_descriptor_resources
                    .as_mut()
                    .context("VPT NRD shared descriptor resources are missing for ready backend")?;
                shared_descriptor_resources
                    .refresh_constant_upload_plan(&backend.state.dispatches)
                    .context("failed to refresh VPT NRD constant upload plan")?;
            }
            self.descriptor_update_plan = descriptor_update_plan;
            Ok(())
        }
    }

    pub fn denoiser_mode(&self) -> VptDenoiserMode {
        self.denoiser_mode
    }

    pub fn is_ready(&self) -> bool {
        self.backend.is_ready()
    }

    fn rebuild_descriptor_resources(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        frame_count: usize,
        image_refs: VptNrdAdapterPassImageRefs<'_>,
        frame_slot: usize,
    ) -> Result<()> {
        let new_shared_descriptor_resources = create_shared_descriptor_resources(
            device,
            allocator,
            &self.backend,
            frame_count,
            self.constant_buffer_alignment,
        )?
        .context("VPT NRD shared descriptor resources are missing for ready backend")?;
        let new_descriptor_resources =
            create_descriptor_resources(device, &self.backend, frame_count)?
                .context("VPT NRD descriptor resources are missing for ready backend")?;
        let descriptor_update_plan = match build_descriptor_update_plan(
            &self.backend,
            Some(&new_descriptor_resources),
            self.texture_pools.as_ref(),
            &self.nrd_diff_radiance_hitdist,
            &self.nrd_validation,
            image_refs,
            frame_slot,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                new_descriptor_resources.destroy(device);
                new_shared_descriptor_resources.destroy(device, allocator);
                return Err(error).context("failed to rebuild VPT NRD descriptor update plan");
            }
        };
        let new_pipeline_resources = match create_pipeline_resources(
            device,
            &self.backend,
            Some(&new_shared_descriptor_resources),
            Some(&new_descriptor_resources),
        ) {
            Ok(Some(resources)) => resources,
            Ok(None) => {
                new_descriptor_resources.destroy(device);
                new_shared_descriptor_resources.destroy(device, allocator);
                anyhow::bail!("VPT NRD pipeline resources are missing for ready backend");
            }
            Err(error) => {
                new_descriptor_resources.destroy(device);
                new_shared_descriptor_resources.destroy(device, allocator);
                return Err(error).context("failed to rebuild VPT NRD pipeline resources");
            }
        };

        let old_pipeline_resources = self.pipeline_resources.replace(new_pipeline_resources);
        let old_shared_descriptor_resources = self
            .shared_descriptor_resources
            .replace(new_shared_descriptor_resources);
        let old_descriptor_resources = self.descriptor_resources.replace(new_descriptor_resources);
        let old_descriptor_update_plan =
            std::mem::replace(&mut self.descriptor_update_plan, descriptor_update_plan);

        if let Some(old_pipeline_resources) = old_pipeline_resources {
            old_pipeline_resources.destroy(device);
        }
        if let Some(old_descriptor_resources) = old_descriptor_resources {
            old_descriptor_resources.destroy(device);
        }
        if let Some(old_shared_descriptor_resources) = old_shared_descriptor_resources {
            old_shared_descriptor_resources.destroy(device, allocator);
        }
        let _ = old_descriptor_update_plan;
        Ok(())
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let VptNrdAdapterBackend::Ready(backend) = &self.backend else {
            return;
        };
        let (
            Some(shared_descriptor_resources),
            Some(pipeline_resources),
            Some(descriptor_update_plan),
        ) = (
            self.shared_descriptor_resources.as_ref(),
            self.pipeline_resources.as_ref(),
            self.descriptor_update_plan.as_ref(),
        )
        else {
            return;
        };
        if backend.state.dispatches.len() != descriptor_update_plan.dispatches.len() {
            tracing::error!(
                dispatch_count = backend.state.dispatches.len(),
                descriptor_update_count = descriptor_update_plan.dispatches.len(),
                "VPT NRD dispatch descriptor update count mismatch"
            );
            return;
        }

        if let Err(error) = shared_descriptor_resources.upload_frame_constants(frame_slot) {
            tracing::error!(%error, "failed to upload VPT NRD dispatch constants");
            return;
        }
        write_descriptor_updates(device, descriptor_update_plan);

        unsafe {
            for (dispatch_index, (dispatch, update)) in backend
                .state
                .dispatches
                .iter()
                .zip(&descriptor_update_plan.dispatches)
                .enumerate()
            {
                if usize::from(dispatch.pipeline_index) != update.pipeline_index {
                    tracing::error!(
                        dispatch_pipeline_index = dispatch.pipeline_index,
                        update_pipeline_index = update.pipeline_index,
                        "VPT NRD dispatch pipeline index mismatch"
                    );
                    return;
                }
                let shared_descriptor_set = match shared_descriptor_resources
                    .descriptor_set(frame_slot, dispatch_index)
                {
                    Ok(descriptor_set) => descriptor_set,
                    Err(error) => {
                        tracing::error!(%error, "failed to select VPT NRD shared descriptor set");
                        return;
                    }
                };
                let Some(pipeline) = pipeline_resources.pipelines.get(update.pipeline_index) else {
                    tracing::error!(
                        pipeline_index = update.pipeline_index,
                        pipeline_count = pipeline_resources.pipelines.len(),
                        "VPT NRD dispatch pipeline index is out of bounds"
                    );
                    return;
                };
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline.handle,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline.layout,
                    pipeline.shared_set_index,
                    &[shared_descriptor_set],
                    &[],
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline.layout,
                    pipeline.resource_set_index,
                    &[update.descriptor_set],
                    &[],
                );
                device.cmd_dispatch(
                    cmd,
                    u32::from(dispatch.grid_width),
                    u32::from(dispatch.grid_height),
                    1,
                );
            }
        }
    }

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        inputs: VptNrdAdapterGraphInputs<'a>,
    ) -> VptNrdAdapterGraphOutputs<'a> {
        let VptNrdAdapterGraphInputs {
            frame_slot,
            packed,
            confidence,
            surface_inputs,
            texture_pools_initialized,
            profiler,
        } = inputs;
        let texture_pool_graph_resources = import_texture_pool_graph_resources(
            graph,
            self.texture_pools.as_ref(),
            texture_pools_initialized,
        );
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
            builder.read_as(packed.normal_roughness, AccessKind::ComputeShaderRead);
            builder.read_as(confidence.diff_confidence, AccessKind::ComputeShaderRead);
            builder.read_as(confidence.spec_confidence, AccessKind::ComputeShaderRead);
            builder.read_as(surface_inputs.view_z, AccessKind::ComputeShaderRead);
            builder.read_as(surface_inputs.motion_history, AccessKind::ComputeShaderRead);
            texture_pool_graph_resources.declare_read_write(builder);
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
            validation_image: &self.nrd_validation,
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        if let Some(pipeline_resources) = self.pipeline_resources {
            pipeline_resources.destroy(device);
        }
        if let Some(descriptor_resources) = self.descriptor_resources {
            descriptor_resources.destroy(device);
        }
        if let Some(shared_descriptor_resources) = self.shared_descriptor_resources {
            shared_descriptor_resources.destroy(device, allocator);
        }
        if let Some(texture_pools) = self.texture_pools {
            texture_pools.destroy(device, allocator);
        }
        self.nrd_diff_radiance_hitdist.destroy(device, allocator);
        self.nrd_validation.destroy(device, allocator);
    }
}

impl VptNrdAdapterBackend {
    pub fn initialize(
        denoiser_mode: VptDenoiserMode,
        width: u32,
        height: u32,
        relax_atrous_iteration_num: u32,
    ) -> Self {
        match VptNrdReadyBackend::initialize(
            denoiser_mode,
            width,
            height,
            relax_atrous_iteration_num,
        ) {
            Ok(backend) => Self::Ready(Box::new(backend)),
            Err(error) => Self::Unavailable(error.to_string()),
        }
    }

    pub fn initialize_relax(width: u32, height: u32, relax_atrous_iteration_num: u32) -> Self {
        Self::initialize(
            VptDenoiserMode::Relax,
            width,
            height,
            relax_atrous_iteration_num,
        )
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }

    pub fn dispatch_count(&self) -> usize {
        match self {
            Self::Ready(backend) => backend.state.dispatches.len(),
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
            Self::Ready(backend) => Some(&backend.state.texture_pool_plan),
            Self::Unavailable(_) => None,
        }
    }

    fn pipeline_descriptor_binding_plans(&self) -> Option<&[VptNrdPipelineDescriptorBindingPlan]> {
        match self {
            Self::Ready(backend) => Some(&backend.state.pipeline_descriptor_binding_plans),
            Self::Unavailable(_) => None,
        }
    }

    fn descriptor_pool_plan(&self) -> Option<&VptNrdDescriptorPoolPlan> {
        match self {
            Self::Ready(backend) => Some(&backend.state.descriptor_pool_plan),
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

fn build_descriptor_update_plan(
    backend: &VptNrdAdapterBackend,
    descriptor_resources: Option<&VptNrdDescriptorResources>,
    texture_pools: Option<&VptNrdTexturePools>,
    output_diff_radiance_hitdist: &GpuImage,
    validation: &GpuImage,
    image_refs: VptNrdAdapterPassImageRefs<'_>,
    frame_slot: usize,
) -> Result<Option<VptNrdDescriptorUpdatePlan>> {
    let VptNrdAdapterBackend::Ready(backend) = backend else {
        return Ok(None);
    };
    let Some(descriptor_resources) = descriptor_resources else {
        anyhow::bail!("VPT NRD descriptor resources are missing for ready backend");
    };
    let Some(texture_pools) = texture_pools else {
        anyhow::bail!("VPT NRD texture pools are missing for ready backend");
    };

    let inputs = VptNrdAdapterImageInputs::from_pass_refs(
        image_refs,
        texture_pools,
        output_diff_radiance_hitdist,
        validation,
    );
    let resolved_plans = backend
        .state
        .dispatch_descriptor_write_plans
        .iter()
        .map(|plan| VptNrdResolvedDispatchDescriptorWritePlan::from_write_plan(plan, &inputs))
        .collect::<Result<Vec<_>>>()?;
    VptNrdDescriptorUpdatePlan::from_resolved_plans(
        &resolved_plans,
        descriptor_resources,
        frame_slot,
    )
    .map(Some)
}

impl VptNrdReadyBackend {
    fn initialize(
        denoiser_mode: VptDenoiserMode,
        width: u32,
        height: u32,
        relax_atrous_iteration_num: u32,
    ) -> Result<Self> {
        let mut instance = match denoiser_mode {
            VptDenoiserMode::Relax => NrdInstance::relax_diffuse(width, height)?,
            VptDenoiserMode::Reblur => NrdInstance::reblur_diffuse(width, height)?,
            other => anyhow::bail!("unsupported VPT NRD denoiser mode {other:?}"),
        };
        let initial_settings =
            build_initial_nrd_frame_settings(width, height, relax_atrous_iteration_num)
                .context("failed to build initial VPT NRD frame settings")?;
        match denoiser_mode {
            VptDenoiserMode::Relax => instance
                .set_relax_diffuse_settings(&initial_settings.relax_diffuse)
                .context("failed to upload initial VPT NRD RELAX_DIFFUSE settings")?,
            VptDenoiserMode::Reblur => instance
                .set_reblur_diffuse_settings(&initial_settings.reblur_diffuse)
                .context("failed to upload initial VPT NRD REBLUR_DIFFUSE settings")?,
            _ => unreachable!("unsupported mode was rejected before instance creation"),
        }
        instance
            .set_common_settings(&initial_settings.common)
            .context("failed to upload initial VPT NRD common settings")?;
        let library_desc = NrdInstance::library_desc()?;
        validate_nrd_library_desc(library_desc)?;
        let instance_snapshot = instance.instance_snapshot()?;
        let dispatches = instance.dispatch_snapshot()?;
        Self::from_instance(
            denoiser_mode,
            instance,
            library_desc,
            instance_snapshot,
            dispatches,
            width,
            height,
        )
    }

    fn from_instance(
        denoiser_mode: VptDenoiserMode,
        instance: NrdInstance,
        library_desc: NrdLibraryDesc,
        instance_snapshot: NrdInstanceSnapshot,
        dispatches: Vec<NrdDispatchSnapshot>,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let state = VptNrdReadyBackendState::from_snapshots(
            library_desc,
            instance_snapshot,
            dispatches,
            width,
            height,
        )?;
        Ok(Self {
            denoiser_mode,
            instance,
            state,
        })
    }

    fn update_frame_settings(&mut self, settings: &VptNrdFrameSettings) -> Result<()> {
        match self.denoiser_mode {
            VptDenoiserMode::Relax => self
                .instance
                .set_relax_diffuse_settings(&settings.relax_diffuse)
                .context("failed to upload VPT NRD RELAX_DIFFUSE settings")?,
            VptDenoiserMode::Reblur => self
                .instance
                .set_reblur_diffuse_settings(&settings.reblur_diffuse)
                .context("failed to upload VPT NRD REBLUR_DIFFUSE settings")?,
            _ => anyhow::bail!("unsupported VPT NRD denoiser mode {:?}", self.denoiser_mode),
        }
        self.instance
            .set_common_settings(&settings.common)
            .context("failed to upload VPT NRD common settings")?;
        let dispatches = self
            .instance
            .dispatch_snapshot()
            .context("failed to query refreshed VPT NRD dispatches")?;
        self.state.refresh_dispatches(dispatches)
    }

    fn metadata(&self) -> VptNrdAdapterBackendMetadata {
        self.state.metadata()
    }
}

impl VptNrdReadyBackendState {
    fn from_snapshots(
        library_desc: NrdLibraryDesc,
        instance_snapshot: NrdInstanceSnapshot,
        dispatches: Vec<NrdDispatchSnapshot>,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let texture_pool_plan =
            VptNrdTexturePoolPlan::from_instance_snapshot(width, height, &instance_snapshot)
                .context("failed to build VPT NRD texture pool plan")?;
        let pipeline_layout_plans =
            VptNrdPipelineLayoutPlan::from_instance_snapshot(&instance_snapshot)
                .context("failed to build VPT NRD pipeline layout plans")?;
        let pipeline_shader_plans =
            VptNrdPipelineShaderPlan::from_instance_snapshot(&instance_snapshot)
                .context("failed to build VPT NRD pipeline shader plans")?;
        let pipeline_descriptor_binding_plans =
            VptNrdPipelineDescriptorBindingPlan::from_layout_plans(
                &pipeline_layout_plans,
                library_desc,
                &instance_snapshot,
            )
            .context("failed to build VPT NRD pipeline descriptor binding plans")?;
        let pipeline_create_plans = VptNrdPipelineCreatePlan::from_plans(
            &pipeline_layout_plans,
            &pipeline_shader_plans,
            &pipeline_descriptor_binding_plans,
        )
        .context("failed to build VPT NRD pipeline create plans")?;
        let descriptor_pool_plan =
            VptNrdDescriptorPoolPlan::from_binding_plans(&pipeline_descriptor_binding_plans)
                .context("failed to build VPT NRD descriptor pool plan")?;
        let dispatch_resource_plans =
            VptNrdDispatchResourcePlan::from_dispatches(&dispatches, &texture_pool_plan)
                .context("failed to build VPT NRD dispatch resource plans")?;
        let dispatch_descriptor_write_plans = VptNrdDispatchDescriptorWritePlan::from_dispatches(
            &dispatches,
            &dispatch_resource_plans,
            &pipeline_descriptor_binding_plans,
        )
        .context("failed to build VPT NRD dispatch descriptor write plans")?;
        Ok(Self {
            library_desc,
            instance_snapshot,
            texture_pool_plan,
            pipeline_layout_plans,
            pipeline_shader_plans,
            pipeline_create_plans,
            pipeline_descriptor_binding_plans,
            descriptor_pool_plan,
            dispatches,
            dispatch_resource_plans,
            dispatch_descriptor_write_plans,
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
            dispatch_resource_plan_count: self.dispatch_resource_plans.len(),
            pipeline_layout_plan_count: self.pipeline_layout_plans.len(),
            pipeline_shader_plan_count: self.pipeline_shader_plans.len(),
            pipeline_create_plan_count: self.pipeline_create_plans.len(),
            pipeline_descriptor_binding_plan_count: self.pipeline_descriptor_binding_plans.len(),
            descriptor_pool_size_count: self.descriptor_pool_plan.pool_sizes.len(),
            dispatch_descriptor_write_plan_count: self.dispatch_descriptor_write_plans.len(),
        }
    }

    fn refresh_dispatches(&mut self, dispatches: Vec<NrdDispatchSnapshot>) -> Result<()> {
        let dispatch_resource_plans =
            VptNrdDispatchResourcePlan::from_dispatches(&dispatches, &self.texture_pool_plan)
                .context("failed to refresh VPT NRD dispatch resource plans")?;
        let dispatch_descriptor_write_plans = VptNrdDispatchDescriptorWritePlan::from_dispatches(
            &dispatches,
            &dispatch_resource_plans,
            &self.pipeline_descriptor_binding_plans,
        )
        .context("failed to refresh VPT NRD dispatch descriptor write plans")?;

        self.dispatches = dispatches;
        self.dispatch_resource_plans = dispatch_resource_plans;
        self.dispatch_descriptor_write_plans = dispatch_descriptor_write_plans;
        Ok(())
    }
}

impl VptNrdFrameSettings {
    pub fn from_inputs(inputs: VptNrdFrameSettingsInputs) -> Result<Self> {
        Ok(Self {
            common: build_nrd_common_settings(inputs)?,
            relax_diffuse: build_nrd_relax_diffuse_settings(inputs.relax_atrous_iteration_num),
            reblur_diffuse: build_nrd_reblur_diffuse_settings(),
        })
    }
}

fn build_nrd_common_settings(inputs: VptNrdFrameSettingsInputs) -> Result<NrdCommonSettings> {
    let current_resolution = nrd_resolution_to_u16(inputs.current_resolution)?;
    let previous_resolution = nrd_resolution_to_u16(inputs.previous_resolution)?;
    let current_width = inputs.current_resolution[0].max(1) as f32;
    let current_height = inputs.current_resolution[1].max(1) as f32;
    let time_delta_ms = inputs.time_delta_seconds.max(0.0) * 1000.0;

    Ok(NrdCommonSettings {
        view_to_clip_matrix: inputs.current_view_to_clip.to_cols_array(),
        view_to_clip_matrix_prev: inputs.previous_view_to_clip.to_cols_array(),
        world_to_view_matrix: inputs.current_world_to_view.to_cols_array(),
        world_to_view_matrix_prev: inputs.previous_world_to_view.to_cols_array(),
        camera_jitter: [0.0, 0.0],
        camera_jitter_prev: [0.0, 0.0],
        motion_vector_scale: [1.0 / current_width, 1.0 / current_height, 1.0],
        resource_size: current_resolution,
        resource_size_prev: previous_resolution,
        rect_size: current_resolution,
        rect_size_prev: previous_resolution,
        denoising_range: 10_000.0,
        disocclusion_threshold: 0.01,
        disocclusion_threshold_alternate: 0.05,
        split_screen: 0.0,
        time_delta_between_frames: time_delta_ms,
        view_z_scale: 1.0,
        frame_index: inputs.frame_index,
        accumulation_mode: if inputs.reset_history {
            NrdAccumulationMode::Restart as u32
        } else {
            NrdAccumulationMode::Continue as u32
        },
        is_motion_vector_in_world_space: 0,
        is_history_confidence_available: inputs.history_confidence_available as u32,
        is_disocclusion_threshold_mix_available: 0,
        enable_validation: inputs.enable_validation as u32,
    })
}

fn build_initial_nrd_frame_settings(
    width: u32,
    height: u32,
    relax_atrous_iteration_num: u32,
) -> Result<VptNrdFrameSettings> {
    VptNrdFrameSettings::from_inputs(VptNrdFrameSettingsInputs {
        current_world_to_view: glam::Mat4::IDENTITY,
        previous_world_to_view: glam::Mat4::IDENTITY,
        current_view_to_clip: glam::Mat4::IDENTITY,
        previous_view_to_clip: glam::Mat4::IDENTITY,
        current_resolution: [width, height],
        previous_resolution: [width, height],
        frame_index: 0,
        time_delta_seconds: 0.0,
        reset_history: true,
        history_confidence_available: true,
        relax_atrous_iteration_num,
        enable_validation: false,
    })
}

fn build_nrd_relax_diffuse_settings(atrous_iteration_num: u32) -> NrdRelaxDiffuseSettings {
    NrdRelaxDiffuseSettings {
        antilag_acceleration_amount: 0.0,
        antilag_spatial_sigma_scale: 0.0,
        antilag_temporal_sigma_scale: 0.0,
        antilag_reset_amount: 0.5,
        diffuse_max_accumulated_frame_num: 30,
        diffuse_max_fast_accumulated_frame_num: 6,
        history_fix_frame_num: 3,
        history_fix_base_pixel_stride: 14,
        history_fix_alternate_pixel_stride: 14,
        history_fix_edge_stopping_normal_power: 8.0,
        fast_history_clamping_sigma_scale: 2.0,
        diffuse_prepass_blur_radius: 0.0,
        min_hit_distance_weight: 0.1,
        spatial_variance_estimation_history_threshold: 3,
        diffuse_phi_luminance: 2.0,
        atrous_iteration_num: atrous_iteration_num.clamp(2, 8),
        diffuse_min_luminance_weight: 0.0,
        depth_threshold: 0.003,
        confidence_driven_relaxation_multiplier: 0.0,
        confidence_driven_luminance_edge_stopping_relaxation: 0.0,
        confidence_driven_normal_edge_stopping_relaxation: 0.0,
        luminance_edge_stopping_relaxation: 0.5,
        normal_edge_stopping_relaxation: 0.3,
        roughness_edge_stopping_relaxation: 1.0,
        checkerboard_mode: NrdCheckerboardMode::Off as u32,
        hit_distance_reconstruction_mode: NrdHitDistanceReconstructionMode::Off as u32,
        min_material_for_diffuse: 4.0,
        enable_anti_firefly: 1,
        enable_roughness_edge_stopping: 1,
    }
}

fn build_nrd_reblur_diffuse_settings() -> NrdReblurDiffuseSettings {
    NrdReblurDiffuseSettings {
        diffuse_prepass_blur_radius: 0.0,
        specular_prepass_blur_radius: 0.0,
        firefly_suppressor_min_relative_scale: 2.0,
        enable_anti_firefly: 1,
        use_prepass_only_for_specular_motion_estimation: 1,
        ..NrdReblurDiffuseSettings::default()
    }
}

fn validate_nrd_library_desc(library_desc: NrdLibraryDesc) -> Result<()> {
    anyhow::ensure!(
        library_desc.normal_encoding == NrdNormalEncoding::R10G10B10A2Unorm as u8,
        "unsupported NRD normal encoding {}; expected R10_G10_B10_A2_UNORM",
        library_desc.normal_encoding
    );
    anyhow::ensure!(
        library_desc.roughness_encoding == NrdRoughnessEncoding::Linear as u8,
        "unsupported NRD roughness encoding {}; expected LINEAR",
        library_desc.roughness_encoding
    );
    Ok(())
}

fn nrd_resolution_to_u16(resolution: [u32; 2]) -> Result<[u16; 2]> {
    let width = u16::try_from(resolution[0]).context("NRD frame resolution exceeds u16")?;
    let height = u16::try_from(resolution[1]).context("NRD frame resolution exceeds u16")?;
    Ok([width.max(1), height.max(1)])
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

impl VptNrdPipelineLayoutPlan {
    pub fn from_instance_snapshot(snapshot: &NrdInstanceSnapshot) -> Result<Vec<Self>> {
        snapshot.pipelines.iter().map(Self::from_pipeline).collect()
    }

    fn from_pipeline(pipeline: &crate::render::nrd_adapter::NrdPipelineSnapshot) -> Result<Self> {
        let resource_ranges = pipeline
            .resource_ranges
            .iter()
            .map(|range| {
                let binding = range
                    .binding_desc()
                    .context("failed to map NRD pipeline resource range")?;
                Ok(VptNrdPipelineResourceRangePlan {
                    descriptor_type: binding.descriptor_type,
                    descriptors_num: binding.descriptors_num,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            resource_ranges,
            has_constant_data: pipeline.has_constant_data,
            shader_identifier: pipeline.shader_identifier.clone(),
        })
    }
}

impl VptNrdPipelineShaderPlan {
    pub fn from_instance_snapshot(snapshot: &NrdInstanceSnapshot) -> Result<Vec<Self>> {
        snapshot.pipelines.iter().map(Self::from_pipeline).collect()
    }

    pub fn spirv_bytes(&self) -> Vec<u8> {
        self.spirv_words
            .iter()
            .flat_map(|word| word.to_ne_bytes())
            .collect()
    }

    fn from_pipeline(pipeline: &crate::render::nrd_adapter::NrdPipelineSnapshot) -> Result<Self> {
        anyhow::ensure!(
            !pipeline.spirv_bytecode.is_empty(),
            "NRD pipeline shader bytecode is empty"
        );
        Ok(Self {
            spirv_words: pipeline.spirv_bytecode.clone(),
            shader_identifier: pipeline.shader_identifier.clone(),
            has_constant_data: pipeline.has_constant_data,
        })
    }
}

impl VptNrdPipelineCreatePlan {
    pub fn from_plans(
        layout_plans: &[VptNrdPipelineLayoutPlan],
        shader_plans: &[VptNrdPipelineShaderPlan],
        binding_plans: &[VptNrdPipelineDescriptorBindingPlan],
    ) -> Result<Vec<Self>> {
        anyhow::ensure!(
            layout_plans.len() == shader_plans.len(),
            "NRD pipeline layout plan count does not match shader plan count"
        );
        anyhow::ensure!(
            layout_plans.len() == binding_plans.len(),
            "NRD pipeline layout plan count does not match descriptor binding plan count"
        );

        layout_plans
            .iter()
            .zip(shader_plans)
            .zip(binding_plans)
            .enumerate()
            .map(
                |(pipeline_index, ((layout_plan, shader_plan), binding_plan))| {
                    anyhow::ensure!(
                        layout_plan.shader_identifier == shader_plan.shader_identifier
                            && layout_plan.has_constant_data == shader_plan.has_constant_data,
                        "NRD pipeline shader plan does not match layout plan"
                    );
                    Ok(Self {
                        pipeline_index,
                        shader_plan_index: pipeline_index,
                        descriptor_set_layout_index: pipeline_index,
                        descriptor_set_index: pipeline_index,
                        shader_identifier: shader_plan.shader_identifier.clone(),
                        has_constant_data: shader_plan.has_constant_data,
                        descriptor_binding_count: binding_plan.bindings.len(),
                    })
                },
            )
            .collect()
    }
}

impl VptNrdPipelineResourcesPlan {
    fn from_plans(
        create_plans: &[VptNrdPipelineCreatePlan],
        shader_plans: &[VptNrdPipelineShaderPlan],
        shared_set_layout: vk::DescriptorSetLayout,
        empty_set_layout: vk::DescriptorSetLayout,
        shared_set_index: u32,
        resources_set_index: u32,
        descriptor_resources: &VptNrdDescriptorResources,
    ) -> Result<Vec<Self>> {
        create_plans
            .iter()
            .map(|plan| {
                let shader_plan = shader_plans.get(plan.shader_plan_index).ok_or_else(|| {
                    anyhow::anyhow!(
                        "NRD pipeline shader plan index {} is out of bounds",
                        plan.shader_plan_index
                    )
                })?;
                let resource_set_layout = descriptor_resources
                    .descriptor_set_layouts
                    .get(plan.descriptor_set_layout_index)
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "NRD descriptor set layout index {} is out of bounds",
                            plan.descriptor_set_layout_index
                        )
                    })?;
                let resource_descriptor_set = descriptor_resources
                    .descriptor_set(0, 0, plan.descriptor_set_index)
                    .with_context(|| {
                        format!(
                            "failed to select NRD pipeline resource descriptor set {}",
                            plan.descriptor_set_index
                        )
                    })?;
                anyhow::ensure!(
                    shader_plan.shader_identifier == plan.shader_identifier
                        && shader_plan.has_constant_data == plan.has_constant_data,
                    "NRD pipeline resource plan shader metadata does not match create plan"
                );
                let set_layout_plan = VptNrdPipelineSetLayoutPlan::from_nrd_spaces(
                    shared_set_index,
                    resources_set_index,
                    shared_set_layout,
                    resource_set_layout,
                    empty_set_layout,
                )?;
                Ok(Self {
                    pipeline_index: plan.pipeline_index,
                    shader_identifier: plan.shader_identifier.clone(),
                    spirv_bytes: shader_plan.spirv_bytes(),
                    descriptor_set_layouts: set_layout_plan.descriptor_set_layouts,
                    shared_set_index: set_layout_plan.shared_set_index,
                    resource_set_index: set_layout_plan.resources_set_index,
                    resource_descriptor_set,
                })
            })
            .collect()
    }
}

impl VptNrdPipelineDescriptorBindingPlan {
    pub fn from_layout_plans(
        layout_plans: &[VptNrdPipelineLayoutPlan],
        library_desc: NrdLibraryDesc,
        snapshot: &NrdInstanceSnapshot,
    ) -> Result<Vec<Self>> {
        layout_plans
            .iter()
            .map(|plan| Self::from_layout_plan(plan, library_desc, snapshot))
            .collect()
    }

    pub fn descriptor_binding_specs(&self) -> Vec<DescriptorBindingSpec> {
        self.bindings
            .iter()
            .map(|binding| DescriptorBindingSpec {
                binding: binding.binding,
                descriptor_type: binding.descriptor_type,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                count: binding.descriptor_count,
            })
            .collect()
    }

    fn from_layout_plan(
        layout_plan: &VptNrdPipelineLayoutPlan,
        library_desc: NrdLibraryDesc,
        snapshot: &NrdInstanceSnapshot,
    ) -> Result<Self> {
        let mut sampled_index = 0u32;
        let mut storage_index = 0u32;
        let mut bindings = Vec::new();
        for range in &layout_plan.resource_ranges {
            anyhow::ensure!(
                range.descriptors_num > 0,
                "NRD pipeline resource range has zero descriptors"
            );
            for _ in 0..range.descriptors_num {
                let (binding, descriptor_type) = match range.descriptor_type {
                    NrdDescriptorType::Texture => {
                        let binding = library_desc
                            .texture_offset
                            .checked_add(snapshot.resources_base_register_index)
                            .and_then(|binding| binding.checked_add(sampled_index))
                            .context("NRD sampled image binding index overflow")?;
                        sampled_index = sampled_index
                            .checked_add(1)
                            .context("NRD sampled image binding count overflow")?;
                        (binding, vk::DescriptorType::SAMPLED_IMAGE)
                    }
                    NrdDescriptorType::StorageTexture => {
                        let binding = library_desc
                            .storage_texture_and_buffer_offset
                            .checked_add(snapshot.resources_base_register_index)
                            .and_then(|binding| binding.checked_add(storage_index))
                            .context("NRD storage image binding index overflow")?;
                        storage_index = storage_index
                            .checked_add(1)
                            .context("NRD storage image binding count overflow")?;
                        (binding, vk::DescriptorType::STORAGE_IMAGE)
                    }
                    other => anyhow::bail!("unsupported NRD pipeline descriptor type {other:?}"),
                };
                bindings.push(VptNrdPipelineDescriptorBinding {
                    binding,
                    descriptor_type,
                    descriptor_count: 1,
                });
            }
        }
        Ok(Self { bindings })
    }
}

impl VptNrdSharedDescriptorBindingPlan {
    pub fn from_instance(
        library_desc: NrdLibraryDesc,
        snapshot: &NrdInstanceSnapshot,
    ) -> Result<Self> {
        let mut bindings = Vec::with_capacity(1 + snapshot.samplers.len());
        bindings.push(VptNrdSharedDescriptorBinding {
            binding: library_desc
                .constant_buffer_offset
                .checked_add(snapshot.constant_buffer_register_index)
                .context("NRD constant buffer binding index overflow")?,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        });
        for sampler_index in 0..snapshot.samplers.len() {
            bindings.push(VptNrdSharedDescriptorBinding {
                binding: library_desc
                    .sampler_offset
                    .checked_add(snapshot.samplers_base_register_index)
                    .and_then(|binding| binding.checked_add(sampler_index as u32))
                    .context("NRD sampler binding index overflow")?,
                descriptor_type: vk::DescriptorType::SAMPLER,
                descriptor_count: 1,
            });
        }
        Ok(Self { bindings })
    }

    pub fn descriptor_binding_specs(&self) -> Vec<DescriptorBindingSpec> {
        self.bindings
            .iter()
            .map(|binding| DescriptorBindingSpec {
                binding: binding.binding,
                descriptor_type: binding.descriptor_type,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                count: binding.descriptor_count,
            })
            .collect()
    }
}

impl VptNrdPipelineSetLayoutPlan {
    pub fn from_nrd_spaces(
        shared_set_index: u32,
        resources_set_index: u32,
        shared_set_layout: vk::DescriptorSetLayout,
        resource_set_layout: vk::DescriptorSetLayout,
        empty_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        anyhow::ensure!(
            shared_set_index != resources_set_index,
            "NRD shared and resource descriptor spaces overlap"
        );
        let set_count = shared_set_index.max(resources_set_index) as usize + 1;
        let mut descriptor_set_layouts = vec![empty_set_layout; set_count];
        descriptor_set_layouts[shared_set_index as usize] = shared_set_layout;
        descriptor_set_layouts[resources_set_index as usize] = resource_set_layout;
        Ok(Self {
            descriptor_set_layouts,
            shared_set_index,
            resources_set_index,
        })
    }
}

impl VptNrdDescriptorPoolPlan {
    pub fn from_binding_plans(
        binding_plans: &[VptNrdPipelineDescriptorBindingPlan],
    ) -> Result<Self> {
        anyhow::ensure!(
            !binding_plans.is_empty(),
            "NRD pipeline descriptor binding plans are empty"
        );
        let max_sets = u32::try_from(binding_plans.len())
            .context("NRD pipeline descriptor set count exceeds u32")?;
        let mut sampled_image_count = 0u32;
        let mut storage_image_count = 0u32;
        for plan in binding_plans {
            for binding in &plan.bindings {
                match binding.descriptor_type {
                    vk::DescriptorType::SAMPLED_IMAGE => {
                        sampled_image_count = sampled_image_count
                            .checked_add(binding.descriptor_count)
                            .context("NRD sampled image descriptor pool size overflow")?;
                    }
                    vk::DescriptorType::STORAGE_IMAGE => {
                        storage_image_count = storage_image_count
                            .checked_add(binding.descriptor_count)
                            .context("NRD storage image descriptor pool size overflow")?;
                    }
                    other => {
                        anyhow::bail!("unsupported NRD descriptor pool type {other:?}");
                    }
                }
            }
        }

        let mut pool_sizes = Vec::new();
        if sampled_image_count > 0 {
            pool_sizes.push(VptNrdDescriptorPoolSizePlan {
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: sampled_image_count,
            });
        }
        if storage_image_count > 0 {
            pool_sizes.push(VptNrdDescriptorPoolSizePlan {
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: storage_image_count,
            });
        }

        Ok(Self {
            max_sets,
            pool_sizes,
        })
    }

    pub fn vk_pool_sizes(&self) -> Vec<vk::DescriptorPoolSize> {
        self.pool_sizes
            .iter()
            .map(|pool_size| vk::DescriptorPoolSize {
                ty: pool_size.descriptor_type,
                descriptor_count: pool_size.descriptor_count,
            })
            .collect()
    }
}

impl VptNrdDispatchResourcePlan {
    pub fn from_dispatches(
        dispatches: &[NrdDispatchSnapshot],
        pool_plan: &VptNrdTexturePoolPlan,
    ) -> Result<Vec<Self>> {
        dispatches
            .iter()
            .map(|dispatch| Self::from_dispatch(dispatch, pool_plan))
            .collect()
    }

    pub fn from_dispatch(
        dispatch: &NrdDispatchSnapshot,
        pool_plan: &VptNrdTexturePoolPlan,
    ) -> Result<Self> {
        let bindings = dispatch
            .resources
            .iter()
            .map(|resource| {
                let binding = resource
                    .binding_desc()
                    .context("failed to map NRD dispatch resource binding")?;
                Ok(VptNrdDispatchResourceBindingPlan {
                    descriptor_type: binding.descriptor_type,
                    resource: map_dispatch_resource(
                        binding.resource_type,
                        binding.index_in_pool,
                        pool_plan,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { bindings })
    }
}

impl VptNrdDispatchDescriptorWritePlan {
    pub fn from_dispatches(
        dispatches: &[NrdDispatchSnapshot],
        dispatch_plans: &[VptNrdDispatchResourcePlan],
        pipeline_binding_plans: &[VptNrdPipelineDescriptorBindingPlan],
    ) -> Result<Vec<Self>> {
        anyhow::ensure!(
            dispatches.len() == dispatch_plans.len(),
            "NRD dispatch count does not match dispatch resource plan count"
        );
        dispatches
            .iter()
            .zip(dispatch_plans.iter())
            .map(|(dispatch, dispatch_plan)| {
                Self::from_dispatch_plan(dispatch, dispatch_plan, pipeline_binding_plans)
            })
            .collect()
    }

    pub fn from_dispatch_plan(
        dispatch: &NrdDispatchSnapshot,
        dispatch_plan: &VptNrdDispatchResourcePlan,
        pipeline_binding_plans: &[VptNrdPipelineDescriptorBindingPlan],
    ) -> Result<Self> {
        let pipeline_index = usize::from(dispatch.pipeline_index);
        let pipeline_binding_plan =
            pipeline_binding_plans
                .get(pipeline_index)
                .with_context(|| {
                    format!("NRD dispatch pipeline index {pipeline_index} is out of bounds")
                })?;
        let expected_resource_count: usize = pipeline_binding_plan
            .bindings
            .iter()
            .map(|binding| binding.descriptor_count as usize)
            .sum();
        anyhow::ensure!(
            dispatch_plan.bindings.len() == expected_resource_count,
            "NRD dispatch resource count does not match pipeline descriptor binding count"
        );

        let mut resource_index = 0usize;
        let mut writes = Vec::with_capacity(dispatch_plan.bindings.len());
        for pipeline_binding in &pipeline_binding_plan.bindings {
            for array_element in 0..pipeline_binding.descriptor_count {
                let dispatch_binding = &dispatch_plan.bindings[resource_index];
                let expected_descriptor_type =
                    nrd_descriptor_type_to_vk(dispatch_binding.descriptor_type)?;
                anyhow::ensure!(
                    pipeline_binding.descriptor_type == expected_descriptor_type,
                    "NRD dispatch descriptor type does not match pipeline binding"
                );
                writes.push(VptNrdDispatchDescriptorWrite {
                    binding: pipeline_binding.binding,
                    array_element,
                    descriptor_type: pipeline_binding.descriptor_type,
                    resource: dispatch_binding.resource,
                });
                resource_index += 1;
            }
        }

        Ok(Self {
            pipeline_index,
            writes,
        })
    }
}

impl VptNrdDispatchConstantUploadPlan {
    pub fn from_dispatches(
        dispatches: &[NrdDispatchSnapshot],
        max_data_size: vk::DeviceSize,
        frame_count: usize,
        slot_alignment: vk::DeviceSize,
    ) -> Result<Self> {
        anyhow::ensure!(frame_count > 0, "NRD constant upload frame count is zero");
        let max_data_size = max_data_size.max(1);
        let slot_stride = align_up(max_data_size, slot_alignment.max(1))
            .context("NRD constant upload slot stride overflow")?;
        let slot_count = (dispatches.len())
            .checked_mul(frame_count)
            .context("NRD constant upload slot count overflow")?;
        let total_size = slot_stride
            .checked_mul(slot_count as vk::DeviceSize)
            .context("NRD constant upload buffer size overflow")?;
        let dispatches = dispatches
            .iter()
            .enumerate()
            .map(|(dispatch_index, dispatch)| {
                anyhow::ensure!(
                    dispatch.constant_buffer_data.len() as vk::DeviceSize <= max_data_size,
                    "NRD dispatch constant data exceeds max constant buffer size"
                );
                Ok(VptNrdDispatchConstantUpload {
                    dispatch_index,
                    data: dispatch.constant_buffer_data.clone(),
                    matches_previous_dispatch: dispatch
                        .constant_buffer_data_matches_previous_dispatch,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dispatches,
            max_data_size,
            slot_stride,
            frame_count,
            total_size,
        })
    }

    pub fn slot_offset(&self, frame_slot: usize, dispatch_index: usize) -> Result<vk::DeviceSize> {
        anyhow::ensure!(
            frame_slot < self.frame_count,
            "NRD constant upload frame slot {frame_slot} is out of bounds"
        );
        anyhow::ensure!(
            dispatch_index < self.dispatches.len(),
            "NRD constant upload dispatch index {dispatch_index} is out of bounds"
        );
        let slot_index = frame_slot
            .checked_mul(self.dispatches.len())
            .and_then(|base| base.checked_add(dispatch_index))
            .context("NRD constant upload slot index overflow")?;
        self.slot_stride
            .checked_mul(slot_index as vk::DeviceSize)
            .context("NRD constant upload slot offset overflow")
    }

    fn descriptor_buffer_info(
        &self,
        buffer: vk::Buffer,
        frame_slot: usize,
        dispatch_index: usize,
    ) -> Result<vk::DescriptorBufferInfo> {
        Ok(vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(self.slot_offset(frame_slot, dispatch_index)?)
            .range(self.max_data_size))
    }

    fn write_frame_constants_into_slice(&self, bytes: &mut [u8], frame_slot: usize) -> Result<()> {
        anyhow::ensure!(
            bytes.len() as vk::DeviceSize >= self.total_size,
            "NRD constant upload destination slice is too small"
        );
        let mut previous_data = Vec::<u8>::new();
        for upload in &self.dispatches {
            let offset = self.slot_offset(frame_slot, upload.dispatch_index)? as usize;
            let size = self.max_data_size as usize;
            let slot = bytes
                .get_mut(offset..offset + size)
                .context("NRD constant upload slot range is out of bounds")?;
            slot.fill(0);
            let data = if upload.matches_previous_dispatch && upload.data.is_empty() {
                previous_data.as_slice()
            } else {
                upload.data.as_slice()
            };
            slot[..data.len()].copy_from_slice(data);
            previous_data.clear();
            previous_data.extend_from_slice(slot);
        }
        Ok(())
    }
}

impl VptNrdResolvedDispatchDescriptorWritePlan {
    pub fn from_write_plan(
        plan: &VptNrdDispatchDescriptorWritePlan,
        inputs: &VptNrdAdapterImageInputs<'_>,
    ) -> Result<Self> {
        let writes = plan
            .writes
            .iter()
            .map(|write| write.resolve_image(inputs))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            pipeline_index: plan.pipeline_index,
            writes,
        })
    }
}

impl VptNrdDescriptorUpdatePlan {
    fn from_resolved_plans(
        resolved_plans: &[VptNrdResolvedDispatchDescriptorWritePlan],
        descriptor_resources: &VptNrdDescriptorResources,
        frame_slot: usize,
    ) -> Result<Self> {
        let dispatches = resolved_plans
            .iter()
            .enumerate()
            .map(|(dispatch_index, resolved_plan)| {
                let descriptor_set = descriptor_resources.descriptor_set(
                    frame_slot,
                    dispatch_index,
                    resolved_plan.pipeline_index,
                )?;
                Ok(VptNrdDispatchDescriptorUpdatePlan {
                    pipeline_index: resolved_plan.pipeline_index,
                    descriptor_set,
                    writes: resolved_plan
                        .writes
                        .iter()
                        .copied()
                        .map(VptNrdDescriptorImageUpdate::from)
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { dispatches })
    }
}

impl VptNrdDispatchDescriptorWrite {
    fn resolve_image(
        &self,
        inputs: &VptNrdAdapterImageInputs<'_>,
    ) -> Result<VptNrdResolvedDescriptorImageWrite> {
        let image = inputs.image_for_resource(self.resource)?;
        Ok(VptNrdResolvedDescriptorImageWrite {
            binding: self.binding,
            array_element: self.array_element,
            descriptor_type: self.descriptor_type,
            image_view: image.view,
            image_layout: descriptor_image_layout(self.descriptor_type)?,
        })
    }
}

impl<'a> VptNrdAdapterImageInputs<'a> {
    fn from_pass_refs(
        image_refs: VptNrdAdapterPassImageRefs<'a>,
        texture_pools: &'a VptNrdTexturePools,
        output_diff_radiance_hitdist: &'a GpuImage,
        validation: &'a GpuImage,
    ) -> Self {
        Self {
            motion: &image_refs.surface.motion_history,
            normal_roughness: &image_refs.frontend.packed_normal_roughness,
            view_z: &image_refs.surface.surface_view_z,
            diff_confidence: &image_refs.confidence.diff_confidence,
            spec_confidence: &image_refs.confidence.spec_confidence,
            diff_radiance_hitdist: &image_refs.frontend.packed_diff_radiance_hitdist,
            output_diff_radiance_hitdist,
            validation,
            permanent_pool: &texture_pools.permanent_pool,
            transient_pool: &texture_pools.transient_pool,
        }
    }

    fn image_for_resource(&self, resource: VptNrdDispatchResource) -> Result<&'a GpuImage> {
        match resource {
            VptNrdDispatchResource::Motion => Ok(self.motion),
            VptNrdDispatchResource::NormalRoughness => Ok(self.normal_roughness),
            VptNrdDispatchResource::ViewZ => Ok(self.view_z),
            VptNrdDispatchResource::DiffConfidence => Ok(self.diff_confidence),
            VptNrdDispatchResource::SpecConfidence => Ok(self.spec_confidence),
            VptNrdDispatchResource::DiffRadianceHitdist => Ok(self.diff_radiance_hitdist),
            VptNrdDispatchResource::OutputDiffRadianceHitdist => {
                Ok(self.output_diff_radiance_hitdist)
            }
            VptNrdDispatchResource::Validation => Ok(self.validation),
            VptNrdDispatchResource::PermanentPool { index } => {
                self.permanent_pool.get(index).with_context(|| {
                    format!("NRD permanent texture pool image index {index} is out of bounds")
                })
            }
            VptNrdDispatchResource::TransientPool { index } => {
                self.transient_pool.get(index).with_context(|| {
                    format!("NRD transient texture pool image index {index} is out of bounds")
                })
            }
        }
    }
}

impl VptNrdDescriptorImageUpdate {
    fn descriptor_image_info(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .image_view(self.image_view)
            .image_layout(self.image_layout)
    }
}

fn write_descriptor_updates(device: &ash::Device, update_plan: &VptNrdDescriptorUpdatePlan) {
    for update in &update_plan.dispatches {
        let image_infos = update
            .writes
            .iter()
            .map(VptNrdDescriptorImageUpdate::descriptor_image_info)
            .collect::<Vec<_>>();
        let writes = update
            .writes
            .iter()
            .zip(image_infos.iter())
            .map(|(image_update, image_info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(update.descriptor_set)
                    .dst_binding(image_update.binding)
                    .dst_array_element(image_update.array_element)
                    .descriptor_type(image_update.descriptor_type)
                    .image_info(std::slice::from_ref(image_info))
            })
            .collect::<Vec<_>>();
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}

fn write_shared_descriptor_updates(
    device: &ash::Device,
    resources: &VptNrdSharedDescriptorResources,
) -> Result<()> {
    for frame_slot in 0..resources.constant_upload_plan.frame_count {
        for dispatch_index in 0..resources.constant_upload_plan.dispatches.len() {
            let descriptor_set = resources.descriptor_set(frame_slot, dispatch_index)?;
            let buffer_info = resources.constant_upload_plan.descriptor_buffer_info(
                resources.constant_buffer.handle,
                frame_slot,
                dispatch_index,
            )?;
            let sampler_infos = resources
                .samplers
                .iter()
                .map(|sampler| vk::DescriptorImageInfo::default().sampler(*sampler))
                .collect::<Vec<_>>();
            let mut writes = Vec::with_capacity(1 + sampler_infos.len());
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(resources.constant_binding)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_info)),
            );
            for (sampler_index, sampler_info) in sampler_infos.iter().enumerate() {
                let binding = resources
                    .sampler_bindings
                    .get(sampler_index)
                    .copied()
                    .with_context(|| {
                        format!("NRD sampler binding index {sampler_index} is out of bounds")
                    })?;
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(std::slice::from_ref(sampler_info)),
                );
            }
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }
    Ok(())
}

fn shared_descriptor_pool_sizes(
    shared_set_count: u32,
    sampler_count: usize,
) -> Result<Vec<vk::DescriptorPoolSize>> {
    let mut pool_sizes = vec![vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: shared_set_count,
    }];
    if sampler_count > 0 {
        let sampler_descriptor_count = shared_set_count
            .checked_mul(u32::try_from(sampler_count).context("NRD sampler count exceeds u32")?)
            .context("NRD sampler descriptor pool size overflow")?;
        pool_sizes.push(vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLER,
            descriptor_count: sampler_descriptor_count,
        });
    }
    Ok(pool_sizes)
}

fn create_nrd_samplers(
    device: &ash::Device,
    sampler_descs: &[NrdSamplerDesc],
) -> Result<Vec<vk::Sampler>> {
    let mut samplers = Vec::with_capacity(sampler_descs.len());
    for desc in sampler_descs {
        match create_nrd_sampler(device, *desc) {
            Ok(sampler) => samplers.push(sampler),
            Err(error) => {
                destroy_samplers(device, samplers);
                return Err(error);
            }
        }
    }
    Ok(samplers)
}

fn create_nrd_sampler(device: &ash::Device, desc: NrdSamplerDesc) -> Result<vk::Sampler> {
    let filter = match desc.mode {
        0 => vk::Filter::NEAREST,
        1 => vk::Filter::LINEAR,
        other => anyhow::bail!("unsupported NRD sampler mode {other}"),
    };
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(0.0);
    unsafe { device.create_sampler(&create_info, None) }.context("failed to create NRD sampler")
}

fn destroy_samplers(device: &ash::Device, samplers: Vec<vk::Sampler>) {
    for sampler in samplers {
        unsafe { device.destroy_sampler(sampler, None) };
    }
}

impl From<VptNrdResolvedDescriptorImageWrite> for VptNrdDescriptorImageUpdate {
    fn from(value: VptNrdResolvedDescriptorImageWrite) -> Self {
        Self {
            binding: value.binding,
            array_element: value.array_element,
            descriptor_type: value.descriptor_type,
            image_view: value.image_view,
            image_layout: value.image_layout,
        }
    }
}

fn descriptor_image_layout(descriptor_type: vk::DescriptorType) -> Result<vk::ImageLayout> {
    match descriptor_type {
        vk::DescriptorType::SAMPLED_IMAGE => Ok(AccessKind::ComputeShaderRead.image_layout()),
        vk::DescriptorType::STORAGE_IMAGE => Ok(AccessKind::ComputeShaderWrite.image_layout()),
        other => anyhow::bail!(
            "NRD dispatch descriptor image write uses unsupported descriptor type {other:?}"
        ),
    }
}

fn nrd_descriptor_type_to_vk(descriptor_type: NrdDescriptorType) -> Result<vk::DescriptorType> {
    match descriptor_type {
        NrdDescriptorType::Texture => Ok(vk::DescriptorType::SAMPLED_IMAGE),
        NrdDescriptorType::StorageTexture => Ok(vk::DescriptorType::STORAGE_IMAGE),
        other => anyhow::bail!("unsupported NRD descriptor type {other:?}"),
    }
}

fn align_up(value: vk::DeviceSize, alignment: vk::DeviceSize) -> Option<vk::DeviceSize> {
    if alignment <= 1 {
        return Some(value);
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn map_dispatch_resource(
    resource_type: NrdResourceType,
    index_in_pool: u16,
    pool_plan: &VptNrdTexturePoolPlan,
) -> Result<VptNrdDispatchResource> {
    let resource = match resource_type {
        NrdResourceType::InMv => VptNrdDispatchResource::Motion,
        NrdResourceType::InNormalRoughness => VptNrdDispatchResource::NormalRoughness,
        NrdResourceType::InViewZ => VptNrdDispatchResource::ViewZ,
        NrdResourceType::InDiffConfidence => VptNrdDispatchResource::DiffConfidence,
        NrdResourceType::InSpecConfidence => VptNrdDispatchResource::SpecConfidence,
        NrdResourceType::InDiffRadianceHitdist => VptNrdDispatchResource::DiffRadianceHitdist,
        NrdResourceType::OutDiffRadianceHitdist => {
            VptNrdDispatchResource::OutputDiffRadianceHitdist
        }
        NrdResourceType::OutValidation => VptNrdDispatchResource::Validation,
        NrdResourceType::PermanentPool => {
            let index = usize::from(index_in_pool);
            anyhow::ensure!(
                index < pool_plan.permanent.len(),
                "NRD permanent texture pool index {index} is out of bounds"
            );
            VptNrdDispatchResource::PermanentPool { index }
        }
        NrdResourceType::TransientPool => {
            let index = usize::from(index_in_pool);
            anyhow::ensure!(
                index < pool_plan.transient.len(),
                "NRD transient texture pool index {index} is out of bounds"
            );
            VptNrdDispatchResource::TransientPool { index }
        }
        other => anyhow::bail!("unsupported NRD RELAX_DIFFUSE resource {other:?}"),
    };
    Ok(resource)
}

impl VptNrdTexturePools {
    fn create(
        device: &ash::Device,
        allocator: &GpuAllocator,
        plan: &VptNrdTexturePoolPlan,
    ) -> Result<Self> {
        let permanent_pool = create_texture_pool_images(
            device,
            allocator,
            &plan.permanent,
            "vpt_nrd_permanent_texture_pool",
        )?;
        let transient_pool = match create_texture_pool_images(
            device,
            allocator,
            &plan.transient,
            "vpt_nrd_transient_texture_pool",
        ) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_texture_pool_images(permanent_pool, device, allocator);
                return Err(error);
            }
        };

        Ok(Self {
            permanent_pool,
            transient_pool,
        })
    }

    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        destroy_texture_pool_images(self.transient_pool, device, allocator);
        destroy_texture_pool_images(self.permanent_pool, device, allocator);
    }
}

impl VptNrdSharedDescriptorResources {
    fn create(
        device: &ash::Device,
        allocator: &GpuAllocator,
        library_desc: NrdLibraryDesc,
        snapshot: &NrdInstanceSnapshot,
        dispatches: &[NrdDispatchSnapshot],
        frame_count: usize,
        constant_buffer_alignment: vk::DeviceSize,
    ) -> Result<Self> {
        anyhow::ensure!(
            !dispatches.is_empty(),
            "NRD shared descriptor resources require at least one dispatch"
        );
        let binding_plan =
            VptNrdSharedDescriptorBindingPlan::from_instance(library_desc, snapshot)?;
        let constant_binding = binding_plan
            .bindings
            .first()
            .map(|binding| binding.binding)
            .context("NRD shared descriptor binding plan is missing constant buffer")?;
        let sampler_bindings = binding_plan
            .bindings
            .iter()
            .skip(1)
            .map(|binding| binding.binding)
            .collect::<Vec<_>>();
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding_specs(&binding_plan.descriptor_binding_specs())
            .build(device)
            .context("failed to create NRD shared descriptor set layout")?;
        let constant_upload_plan = match VptNrdDispatchConstantUploadPlan::from_dispatches(
            dispatches,
            snapshot.constant_buffer_max_data_size.into(),
            frame_count,
            constant_buffer_alignment,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let constant_buffer = match GpuBuffer::new(
            device,
            allocator,
            constant_upload_plan.total_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "vpt_nrd_dispatch_constants",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error).context("failed to create NRD dispatch constant buffer");
            }
        };
        let samplers = match create_nrd_samplers(device, &snapshot.samplers) {
            Ok(samplers) => samplers,
            Err(error) => {
                constant_buffer.destroy(device, allocator);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let shared_set_count = u32::try_from(
            frame_count
                .checked_mul(dispatches.len())
                .context("NRD shared descriptor set count overflow")?,
        )
        .context("NRD shared descriptor set count exceeds u32")?;
        let pool_sizes = shared_descriptor_pool_sizes(shared_set_count, samplers.len())?;
        let descriptor_pool = match DescriptorPool::new(device, shared_set_count, &pool_sizes) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_samplers(device, samplers);
                constant_buffer.destroy(device, allocator);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error).context("failed to create NRD shared descriptor pool");
            }
        };
        let layouts = vec![descriptor_set_layout; shared_set_count as usize];
        let descriptor_sets = match descriptor_pool.allocate(device, &layouts) {
            Ok(sets) => sets,
            Err(error) => {
                descriptor_pool.destroy(device);
                destroy_samplers(device, samplers);
                constant_buffer.destroy(device, allocator);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error).context("failed to allocate NRD shared descriptor sets");
            }
        };

        let resources = Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            samplers,
            constant_binding,
            sampler_bindings,
            constant_buffer,
            constant_upload_plan,
        };
        write_shared_descriptor_updates(device, &resources)
            .context("failed to write NRD shared descriptor sets")?;
        Ok(resources)
    }

    fn descriptor_set(
        &self,
        frame_slot: usize,
        dispatch_index: usize,
    ) -> Result<vk::DescriptorSet> {
        let offset = self
            .constant_upload_plan
            .slot_offset(frame_slot, dispatch_index)?;
        let slot_index = usize::try_from(offset / self.constant_upload_plan.slot_stride)
            .context("NRD shared descriptor slot index exceeds usize")?;
        self.descriptor_sets
            .get(slot_index)
            .copied()
            .with_context(|| {
                format!("NRD shared descriptor set index {slot_index} is out of bounds")
            })
    }

    fn upload_frame_constants(&self, frame_slot: usize) -> Result<()> {
        let ptr = self
            .constant_buffer
            .mapped_ptr()
            .context("NRD dispatch constant buffer is not CPU mapped")?;
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(ptr, self.constant_upload_plan.total_size as usize)
        };
        self.constant_upload_plan
            .write_frame_constants_into_slice(bytes, frame_slot)
    }

    fn refresh_constant_upload_plan(&mut self, dispatches: &[NrdDispatchSnapshot]) -> Result<()> {
        let refreshed_plan = VptNrdDispatchConstantUploadPlan::from_dispatches(
            dispatches,
            self.constant_upload_plan.max_data_size,
            self.constant_upload_plan.frame_count,
            self.constant_upload_plan.slot_stride,
        )?;
        ensure_refreshed_constant_upload_plan_fits_existing_resources(
            &refreshed_plan,
            &self.constant_upload_plan,
            self.constant_buffer.size,
        )?;
        self.constant_upload_plan = refreshed_plan;
        Ok(())
    }

    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.descriptor_pool.destroy(device);
        destroy_samplers(device, self.samplers);
        self.constant_buffer.destroy(device, allocator);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}

impl VptNrdDescriptorResources {
    fn create(
        device: &ash::Device,
        binding_plans: &[VptNrdPipelineDescriptorBindingPlan],
        pool_plan: &VptNrdDescriptorPoolPlan,
        frame_count: usize,
        dispatch_slot_count: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            frame_count > 0,
            "NRD descriptor resource frame count is zero"
        );
        anyhow::ensure!(
            dispatch_slot_count > 0,
            "NRD descriptor resource dispatch slot count is zero"
        );
        let pipeline_count = binding_plans.len();
        let descriptor_set_layouts = create_descriptor_set_layouts(device, binding_plans)?;
        let descriptor_set_multiplier = frame_count
            .checked_mul(dispatch_slot_count)
            .context("NRD descriptor resource set multiplier overflow")?;
        let descriptor_set_multiplier_u32 = u32::try_from(descriptor_set_multiplier)
            .context("NRD descriptor resource set multiplier exceeds u32")?;
        let max_sets = pool_plan
            .max_sets
            .checked_mul(descriptor_set_multiplier_u32)
            .context("NRD descriptor resource max set count overflow")?;
        let pool_sizes = scaled_descriptor_pool_sizes(pool_plan, descriptor_set_multiplier_u32)?;
        let descriptor_pool = match DescriptorPool::new(device, max_sets, &pool_sizes) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_descriptor_set_layouts(device, descriptor_set_layouts);
                return Err(error);
            }
        };
        let mut descriptor_set_layout_slots =
            Vec::with_capacity(descriptor_set_multiplier * descriptor_set_layouts.len());
        for _ in 0..descriptor_set_multiplier {
            descriptor_set_layout_slots.extend(descriptor_set_layouts.iter().copied());
        }
        let descriptor_sets = match descriptor_pool.allocate(device, &descriptor_set_layout_slots) {
            Ok(sets) => sets,
            Err(error) => {
                descriptor_pool.destroy(device);
                destroy_descriptor_set_layouts(device, descriptor_set_layouts);
                return Err(error);
            }
        };
        Ok(Self {
            descriptor_set_layouts,
            descriptor_pool,
            descriptor_sets,
            frame_count,
            dispatch_slot_count,
            pipeline_count,
        })
    }

    fn layout_specs(
        binding_plans: &[VptNrdPipelineDescriptorBindingPlan],
    ) -> Vec<Vec<DescriptorBindingSpec>> {
        binding_plans
            .iter()
            .map(VptNrdPipelineDescriptorBindingPlan::descriptor_binding_specs)
            .collect()
    }

    fn destroy(self, device: &ash::Device) {
        let _ = self.descriptor_sets;
        self.descriptor_pool.destroy(device);
        destroy_descriptor_set_layouts(device, self.descriptor_set_layouts);
    }

    fn descriptor_set(
        &self,
        frame_slot: usize,
        dispatch_slot: usize,
        pipeline_index: usize,
    ) -> Result<vk::DescriptorSet> {
        anyhow::ensure!(
            pipeline_index < self.pipeline_count,
            "NRD descriptor update pipeline index {pipeline_index} is out of bounds"
        );
        anyhow::ensure!(
            frame_slot < self.frame_count,
            "NRD descriptor update frame slot {frame_slot} is out of bounds"
        );
        anyhow::ensure!(
            dispatch_slot < self.dispatch_slot_count,
            "NRD descriptor update dispatch slot {dispatch_slot} is out of bounds"
        );
        let slot_index = frame_slot
            .checked_mul(self.dispatch_slot_count)
            .and_then(|base| base.checked_add(dispatch_slot))
            .and_then(|slot| slot.checked_mul(self.pipeline_count))
            .and_then(|base| base.checked_add(pipeline_index))
            .context("NRD descriptor update set index overflow")?;
        self.descriptor_sets
            .get(slot_index)
            .copied()
            .with_context(|| {
                format!("NRD descriptor update set index {slot_index} is out of bounds")
            })
    }
}

fn scaled_descriptor_pool_sizes(
    pool_plan: &VptNrdDescriptorPoolPlan,
    descriptor_set_multiplier: u32,
) -> Result<Vec<vk::DescriptorPoolSize>> {
    pool_plan
        .pool_sizes
        .iter()
        .map(|pool_size| {
            Ok(vk::DescriptorPoolSize {
                ty: pool_size.descriptor_type,
                descriptor_count: pool_size
                    .descriptor_count
                    .checked_mul(descriptor_set_multiplier)
                    .context("NRD descriptor pool size overflow")?,
            })
        })
        .collect()
}

fn ensure_refreshed_constant_upload_plan_fits_existing_resources(
    refreshed_plan: &VptNrdDispatchConstantUploadPlan,
    current_plan: &VptNrdDispatchConstantUploadPlan,
    constant_buffer_size: vk::DeviceSize,
) -> Result<()> {
    anyhow::ensure!(
        refreshed_plan.slot_stride == current_plan.slot_stride,
        "NRD refreshed dispatch constants require shared descriptor resource recreation"
    );
    anyhow::ensure!(
        refreshed_plan.total_size <= constant_buffer_size,
        "NRD refreshed dispatch constants exceed shared descriptor resource capacity"
    );
    Ok(())
}

impl VptNrdPipelineResources {
    fn create(
        device: &ash::Device,
        create_plans: &[VptNrdPipelineCreatePlan],
        shader_plans: &[VptNrdPipelineShaderPlan],
        shared_descriptor_resources: &VptNrdSharedDescriptorResources,
        descriptor_resources: &VptNrdDescriptorResources,
        shared_set_index: u32,
        resources_set_index: u32,
    ) -> Result<Self> {
        let empty_descriptor_set_layout = DescriptorLayoutBuilder::new()
            .build(device)
            .context("failed to create NRD empty descriptor set layout")?;
        let resource_plans = VptNrdPipelineResourcesPlan::from_plans(
            create_plans,
            shader_plans,
            shared_descriptor_resources.descriptor_set_layout,
            empty_descriptor_set_layout,
            shared_set_index,
            resources_set_index,
            descriptor_resources,
        )
        .inspect_err(|_| {
            unsafe { device.destroy_descriptor_set_layout(empty_descriptor_set_layout, None) };
        })?;
        let mut pipelines = Vec::with_capacity(resource_plans.len());

        for plan in resource_plans {
            let shader_module = match create_shader_module(device, &plan.spirv_bytes) {
                Ok(module) => module,
                Err(error) => {
                    destroy_compute_pipelines(device, pipelines);
                    unsafe {
                        device.destroy_descriptor_set_layout(empty_descriptor_set_layout, None)
                    };
                    return Err(error).with_context(|| {
                        format!(
                            "failed to create NRD shader module for {}",
                            plan.shader_identifier
                        )
                    });
                }
            };
            let pipeline = match ComputePipeline::new(
                device,
                shader_module,
                c"main",
                &plan.descriptor_set_layouts,
                &[],
            ) {
                Ok(pipeline) => pipeline,
                Err(error) => {
                    unsafe { device.destroy_shader_module(shader_module, None) };
                    destroy_compute_pipelines(device, pipelines);
                    unsafe {
                        device.destroy_descriptor_set_layout(empty_descriptor_set_layout, None)
                    };
                    return Err(error).with_context(|| {
                        format!(
                            "failed to create NRD compute pipeline for {}",
                            plan.shader_identifier
                        )
                    });
                }
            };
            unsafe { device.destroy_shader_module(shader_module, None) };
            pipelines.push(VptNrdComputePipeline {
                pipeline,
                shared_set_index: plan.shared_set_index,
                resource_set_index: plan.resource_set_index,
            });
        }

        Ok(Self {
            pipelines,
            empty_descriptor_set_layout,
        })
    }

    fn destroy(self, device: &ash::Device) {
        destroy_compute_pipelines(device, self.pipelines);
        unsafe { device.destroy_descriptor_set_layout(self.empty_descriptor_set_layout, None) };
    }
}

fn create_descriptor_resources(
    device: &ash::Device,
    backend: &VptNrdAdapterBackend,
    frame_count: usize,
) -> Result<Option<VptNrdDescriptorResources>> {
    let (Some(binding_plans), Some(pool_plan)) = (
        backend.pipeline_descriptor_binding_plans(),
        backend.descriptor_pool_plan(),
    ) else {
        return Ok(None);
    };
    VptNrdDescriptorResources::create(
        device,
        binding_plans,
        pool_plan,
        frame_count,
        backend.dispatch_count(),
    )
    .map(Some)
}

fn create_shared_descriptor_resources(
    device: &ash::Device,
    allocator: &GpuAllocator,
    backend: &VptNrdAdapterBackend,
    frame_count: usize,
    constant_buffer_alignment: vk::DeviceSize,
) -> Result<Option<VptNrdSharedDescriptorResources>> {
    let VptNrdAdapterBackend::Ready(backend) = backend else {
        return Ok(None);
    };
    VptNrdSharedDescriptorResources::create(
        device,
        allocator,
        backend.state.library_desc,
        &backend.state.instance_snapshot,
        &backend.state.dispatches,
        frame_count,
        constant_buffer_alignment,
    )
    .map(Some)
}

fn create_pipeline_resources(
    device: &ash::Device,
    backend: &VptNrdAdapterBackend,
    shared_descriptor_resources: Option<&VptNrdSharedDescriptorResources>,
    descriptor_resources: Option<&VptNrdDescriptorResources>,
) -> Result<Option<VptNrdPipelineResources>> {
    let VptNrdAdapterBackend::Ready(backend) = backend else {
        return Ok(None);
    };
    let Some(shared_descriptor_resources) = shared_descriptor_resources else {
        return Ok(None);
    };
    let Some(descriptor_resources) = descriptor_resources else {
        return Ok(None);
    };
    VptNrdPipelineResources::create(
        device,
        &backend.state.pipeline_create_plans,
        &backend.state.pipeline_shader_plans,
        shared_descriptor_resources,
        descriptor_resources,
        backend
            .state
            .instance_snapshot
            .constant_buffer_and_samplers_space_index,
        backend.state.instance_snapshot.resources_space_index,
    )
    .map(Some)
}

fn create_descriptor_set_layouts(
    device: &ash::Device,
    binding_plans: &[VptNrdPipelineDescriptorBindingPlan],
) -> Result<Vec<vk::DescriptorSetLayout>> {
    let mut layouts = Vec::with_capacity(binding_plans.len());
    for specs in VptNrdDescriptorResources::layout_specs(binding_plans) {
        match DescriptorLayoutBuilder::new()
            .add_binding_specs(&specs)
            .build(device)
        {
            Ok(layout) => layouts.push(layout),
            Err(error) => {
                destroy_descriptor_set_layouts(device, layouts);
                return Err(error);
            }
        }
    }
    Ok(layouts)
}

fn destroy_descriptor_set_layouts(device: &ash::Device, layouts: Vec<vk::DescriptorSetLayout>) {
    for layout in layouts {
        unsafe { device.destroy_descriptor_set_layout(layout, None) };
    }
}

fn destroy_compute_pipelines(device: &ash::Device, pipelines: Vec<VptNrdComputePipeline>) {
    for pipeline in pipelines {
        pipeline.pipeline.destroy(device);
    }
}

fn create_texture_pools(
    device: &ash::Device,
    allocator: &GpuAllocator,
    backend: &VptNrdAdapterBackend,
) -> Result<Option<VptNrdTexturePools>> {
    backend
        .texture_pool_plan()
        .map(|plan| VptNrdTexturePools::create(device, allocator, plan))
        .transpose()
}

fn import_texture_pool_graph_resources<'a>(
    graph: &mut RenderGraph<'a>,
    texture_pools: Option<&VptNrdTexturePools>,
    initialized: bool,
) -> VptNrdTexturePoolGraphResources {
    let Some(texture_pools) = texture_pools else {
        return VptNrdTexturePoolGraphResources::default();
    };
    let initial_access = if initialized {
        AccessKind::ComputeShaderReadWrite
    } else {
        AccessKind::Undefined
    };
    let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
    let mut images =
        Vec::with_capacity(texture_pools.permanent_pool.len() + texture_pools.transient_pool.len());
    for image in &texture_pools.permanent_pool {
        images.push(graph.import_image_with_access(
            image.handle,
            image.extent.width,
            image.extent.height,
            image.format,
            usage,
            initial_access,
        ));
    }
    for image in &texture_pools.transient_pool {
        images.push(graph.import_image_with_access(
            image.handle,
            image.extent.width,
            image.extent.height,
            image.format,
            usage,
            initial_access,
        ));
    }
    VptNrdTexturePoolGraphResources { images }
}

fn create_texture_pool_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    plans: &[VptNrdTexturePoolImagePlan],
    name: &'static str,
) -> Result<Vec<GpuImage>> {
    let mut images = Vec::with_capacity(plans.len());
    for plan in plans {
        match create_texture_pool_image(device, allocator, plan, name) {
            Ok(image) => images.push(image),
            Err(error) => {
                destroy_texture_pool_images(images, device, allocator);
                return Err(error);
            }
        }
    }
    Ok(images)
}

fn create_texture_pool_image(
    device: &ash::Device,
    allocator: &GpuAllocator,
    plan: &VptNrdTexturePoolImagePlan,
    name: &'static str,
) -> Result<GpuImage> {
    GpuImage::new(
        device,
        allocator,
        &GpuImageDesc {
            width: plan.width,
            height: plan.height,
            depth: 1,
            format: plan.format,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
            name,
        },
    )
    .with_context(|| format!("failed to create NRD texture pool image {}", plan.name))
}

fn destroy_texture_pool_images(
    images: Vec<GpuImage>,
    device: &ash::Device,
    allocator: &GpuAllocator,
) {
    for pool in images {
        pool.destroy(device, allocator);
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
    use crate::render::graph::BarrierTiming;
    use crate::render::nrd_adapter::{NrdPipelineSnapshot, NrdSamplerDesc, NrdTextureFormat};
    use ash::vk::Handle;

    #[cfg(not(feature = "nrd"))]
    #[test]
    fn default_backend_reports_nrd_unavailable_without_dispatch_readiness() {
        let backend = VptNrdAdapterBackend::initialize_relax(1280, 720, 4);

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
    fn shared_descriptor_binding_plan_uses_nrd_constant_and_sampler_bindings() {
        let library_desc = NrdLibraryDesc {
            texture_offset: 10,
            sampler_offset: 20,
            constant_buffer_offset: 30,
            storage_texture_and_buffer_offset: 40,
            normal_encoding: NrdNormalEncoding::R10G10B10A2Unorm as u8,
            roughness_encoding: NrdRoughnessEncoding::Linear as u8,
            reserved0: 0,
        };
        let mut snapshot = instance_snapshot_with_pipelines(vec![pipeline(
            "relax_prepass",
            &[resource_range(NrdDescriptorType::Texture, 1)],
            true,
        )]);
        snapshot.constant_buffer_register_index = 2;
        snapshot.samplers_base_register_index = 4;
        snapshot.samplers = vec![NrdSamplerDesc { mode: 0 }, NrdSamplerDesc { mode: 1 }];

        let plan =
            VptNrdSharedDescriptorBindingPlan::from_instance(library_desc, &snapshot).unwrap();

        assert_eq!(
            plan.descriptor_binding_specs(),
            vec![
                DescriptorBindingSpec {
                    binding: 32,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    count: 1,
                },
                DescriptorBindingSpec {
                    binding: 24,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    count: 1,
                },
                DescriptorBindingSpec {
                    binding: 25,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    count: 1,
                },
            ]
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

    #[test]
    fn pipeline_layout_plan_maps_resource_ranges_for_each_nrd_pipeline() {
        let snapshot = instance_snapshot_with_pipelines(vec![
            pipeline(
                "relax_prepass",
                &[
                    resource_range(NrdDescriptorType::Texture, 6),
                    resource_range(NrdDescriptorType::StorageTexture, 2),
                ],
                true,
            ),
            pipeline(
                "relax_temporal",
                &[resource_range(NrdDescriptorType::Texture, 4)],
                false,
            ),
        ]);

        let plans = VptNrdPipelineLayoutPlan::from_instance_snapshot(&snapshot).unwrap();

        assert_eq!(
            plans,
            vec![
                VptNrdPipelineLayoutPlan {
                    resource_ranges: vec![
                        VptNrdPipelineResourceRangePlan {
                            descriptor_type: NrdDescriptorType::Texture,
                            descriptors_num: 6,
                        },
                        VptNrdPipelineResourceRangePlan {
                            descriptor_type: NrdDescriptorType::StorageTexture,
                            descriptors_num: 2,
                        },
                    ],
                    has_constant_data: true,
                    shader_identifier: "relax_prepass".to_owned(),
                },
                VptNrdPipelineLayoutPlan {
                    resource_ranges: vec![VptNrdPipelineResourceRangePlan {
                        descriptor_type: NrdDescriptorType::Texture,
                        descriptors_num: 4,
                    },],
                    has_constant_data: false,
                    shader_identifier: "relax_temporal".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn pipeline_layout_plan_rejects_unknown_resource_range_descriptor_type() {
        let snapshot = instance_snapshot_with_pipelines(vec![pipeline_with_raw_range(
            "bad_descriptor",
            &[crate::render::nrd_adapter::NrdResourceRangeDesc {
                descriptor_type: NrdDescriptorType::Unsupported as u32,
                descriptors_num: 1,
            }],
            false,
        )]);

        let error = VptNrdPipelineLayoutPlan::from_instance_snapshot(&snapshot).unwrap_err();

        assert!(error.chain().any(|cause| {
            cause
                .to_string()
                .contains("unsupported NRD descriptor type")
        }));
    }

    #[test]
    fn pipeline_shader_plan_copies_spirv_and_pipeline_metadata() {
        let snapshot = instance_snapshot_with_pipelines(vec![
            NrdPipelineSnapshot {
                spirv_bytecode: vec![0x0723_0203, 0x0001_0000],
                resource_ranges: vec![resource_range(NrdDescriptorType::Texture, 1)],
                has_constant_data: true,
                shader_identifier: "relax_prepass".to_owned(),
            },
            NrdPipelineSnapshot {
                spirv_bytecode: vec![0x0723_0203],
                resource_ranges: Vec::new(),
                has_constant_data: false,
                shader_identifier: "relax_postpass".to_owned(),
            },
        ]);

        let plans = VptNrdPipelineShaderPlan::from_instance_snapshot(&snapshot).unwrap();

        assert_eq!(
            plans,
            vec![
                VptNrdPipelineShaderPlan {
                    spirv_words: vec![0x0723_0203, 0x0001_0000],
                    shader_identifier: "relax_prepass".to_owned(),
                    has_constant_data: true,
                },
                VptNrdPipelineShaderPlan {
                    spirv_words: vec![0x0723_0203],
                    shader_identifier: "relax_postpass".to_owned(),
                    has_constant_data: false,
                },
            ]
        );
    }

    #[test]
    fn pipeline_shader_plan_rejects_empty_spirv() {
        let snapshot = instance_snapshot_with_pipelines(vec![NrdPipelineSnapshot {
            spirv_bytecode: Vec::new(),
            resource_ranges: Vec::new(),
            has_constant_data: false,
            shader_identifier: "empty_shader".to_owned(),
        }]);

        let error = VptNrdPipelineShaderPlan::from_instance_snapshot(&snapshot).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD pipeline shader bytecode is empty")
        );
    }

    #[test]
    fn pipeline_create_plan_pairs_shader_layout_and_descriptor_slots() {
        let layout_plans = vec![
            VptNrdPipelineLayoutPlan {
                resource_ranges: vec![VptNrdPipelineResourceRangePlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    descriptors_num: 2,
                }],
                has_constant_data: false,
                shader_identifier: "relax_prepass".to_owned(),
            },
            VptNrdPipelineLayoutPlan {
                resource_ranges: vec![VptNrdPipelineResourceRangePlan {
                    descriptor_type: NrdDescriptorType::StorageTexture,
                    descriptors_num: 1,
                }],
                has_constant_data: true,
                shader_identifier: "relax_temporal".to_owned(),
            },
        ];
        let shader_plans = vec![
            VptNrdPipelineShaderPlan {
                spirv_words: vec![0x0723_0203],
                shader_identifier: "relax_prepass".to_owned(),
                has_constant_data: false,
            },
            VptNrdPipelineShaderPlan {
                spirv_words: vec![0x0723_0203, 0x0001_0000],
                shader_identifier: "relax_temporal".to_owned(),
                has_constant_data: true,
            },
        ];
        let binding_plans = VptNrdPipelineDescriptorBindingPlan::from_layout_plans(
            &layout_plans,
            nrd_library_desc_for_test(),
            &instance_snapshot_with_pipelines(Vec::new()),
        )
        .unwrap();

        let plans =
            VptNrdPipelineCreatePlan::from_plans(&layout_plans, &shader_plans, &binding_plans)
                .unwrap();

        assert_eq!(
            plans,
            vec![
                VptNrdPipelineCreatePlan {
                    pipeline_index: 0,
                    shader_plan_index: 0,
                    descriptor_set_layout_index: 0,
                    descriptor_set_index: 0,
                    shader_identifier: "relax_prepass".to_owned(),
                    has_constant_data: false,
                    descriptor_binding_count: 2,
                },
                VptNrdPipelineCreatePlan {
                    pipeline_index: 1,
                    shader_plan_index: 1,
                    descriptor_set_layout_index: 1,
                    descriptor_set_index: 1,
                    shader_identifier: "relax_temporal".to_owned(),
                    has_constant_data: true,
                    descriptor_binding_count: 1,
                },
            ]
        );
    }

    #[test]
    fn pipeline_create_plan_rejects_mismatched_shader_metadata() {
        let layout_plans = vec![VptNrdPipelineLayoutPlan {
            resource_ranges: vec![VptNrdPipelineResourceRangePlan {
                descriptor_type: NrdDescriptorType::Texture,
                descriptors_num: 1,
            }],
            has_constant_data: false,
            shader_identifier: "relax_prepass".to_owned(),
        }];
        let shader_plans = vec![VptNrdPipelineShaderPlan {
            spirv_words: vec![0x0723_0203],
            shader_identifier: "relax_temporal".to_owned(),
            has_constant_data: false,
        }];
        let binding_plans = VptNrdPipelineDescriptorBindingPlan::from_layout_plans(
            &layout_plans,
            nrd_library_desc_for_test(),
            &instance_snapshot_with_pipelines(Vec::new()),
        )
        .unwrap();

        let error =
            VptNrdPipelineCreatePlan::from_plans(&layout_plans, &shader_plans, &binding_plans)
                .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD pipeline shader plan does not match layout plan")
        );
    }

    #[test]
    fn pipeline_create_plan_rejects_mismatched_plan_counts() {
        let layout_plans = vec![VptNrdPipelineLayoutPlan {
            resource_ranges: vec![VptNrdPipelineResourceRangePlan {
                descriptor_type: NrdDescriptorType::Texture,
                descriptors_num: 1,
            }],
            has_constant_data: false,
            shader_identifier: "relax_prepass".to_owned(),
        }];
        let shader_plans = Vec::new();
        let binding_plans = VptNrdPipelineDescriptorBindingPlan::from_layout_plans(
            &layout_plans,
            nrd_library_desc_for_test(),
            &instance_snapshot_with_pipelines(Vec::new()),
        )
        .unwrap();

        let error =
            VptNrdPipelineCreatePlan::from_plans(&layout_plans, &shader_plans, &binding_plans)
                .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD pipeline layout plan count does not match shader plan count")
        );

        let shader_plans = vec![VptNrdPipelineShaderPlan {
            spirv_words: vec![0x0723_0203],
            shader_identifier: "relax_prepass".to_owned(),
            has_constant_data: false,
        }];

        let error =
            VptNrdPipelineCreatePlan::from_plans(&layout_plans, &shader_plans, &[]).unwrap_err();

        assert!(error.to_string().contains(
            "NRD pipeline layout plan count does not match descriptor binding plan count"
        ));
    }

    #[test]
    fn pipeline_shader_plan_exposes_spirv_as_native_endian_bytes() {
        let shader_plan = VptNrdPipelineShaderPlan {
            spirv_words: vec![0x0723_0203, 0x0102_0304],
            shader_identifier: "relax_prepass".to_owned(),
            has_constant_data: false,
        };

        assert_eq!(
            shader_plan.spirv_bytes(),
            vec![0x03, 0x02, 0x23, 0x07, 0x04, 0x03, 0x02, 0x01]
        );
    }

    #[test]
    fn pipeline_resources_plan_resolves_descriptor_layout_and_set_slots() {
        let create_plans = vec![
            VptNrdPipelineCreatePlan {
                pipeline_index: 0,
                shader_plan_index: 0,
                descriptor_set_layout_index: 0,
                descriptor_set_index: 0,
                shader_identifier: "relax_prepass".to_owned(),
                has_constant_data: false,
                descriptor_binding_count: 1,
            },
            VptNrdPipelineCreatePlan {
                pipeline_index: 1,
                shader_plan_index: 1,
                descriptor_set_layout_index: 1,
                descriptor_set_index: 1,
                shader_identifier: "relax_temporal".to_owned(),
                has_constant_data: true,
                descriptor_binding_count: 2,
            },
        ];
        let shader_plans = vec![
            VptNrdPipelineShaderPlan {
                spirv_words: vec![0x0723_0203],
                shader_identifier: "relax_prepass".to_owned(),
                has_constant_data: false,
            },
            VptNrdPipelineShaderPlan {
                spirv_words: vec![0x0723_0203, 0x0102_0304],
                shader_identifier: "relax_temporal".to_owned(),
                has_constant_data: true,
            },
        ];
        let descriptor_resources = descriptor_resources_for_test(2, 1, 1);

        let snapshots = VptNrdPipelineResourcesPlan::from_plans(
            &create_plans,
            &shader_plans,
            vk::DescriptorSetLayout::from_raw(90),
            vk::DescriptorSetLayout::from_raw(99),
            0,
            1,
            &descriptor_resources,
        )
        .unwrap();

        assert_eq!(
            snapshots,
            vec![
                VptNrdPipelineResourcesPlan {
                    pipeline_index: 0,
                    shader_identifier: "relax_prepass".to_owned(),
                    spirv_bytes: vec![0x03, 0x02, 0x23, 0x07],
                    descriptor_set_layouts: vec![
                        vk::DescriptorSetLayout::from_raw(90),
                        vk::DescriptorSetLayout::from_raw(101),
                    ],
                    shared_set_index: 0,
                    resource_set_index: 1,
                    resource_descriptor_set: vk::DescriptorSet::from_raw(201),
                },
                VptNrdPipelineResourcesPlan {
                    pipeline_index: 1,
                    shader_identifier: "relax_temporal".to_owned(),
                    spirv_bytes: vec![0x03, 0x02, 0x23, 0x07, 0x04, 0x03, 0x02, 0x01],
                    descriptor_set_layouts: vec![
                        vk::DescriptorSetLayout::from_raw(90),
                        vk::DescriptorSetLayout::from_raw(102),
                    ],
                    shared_set_index: 0,
                    resource_set_index: 1,
                    resource_descriptor_set: vk::DescriptorSet::from_raw(202),
                },
            ]
        );
    }

    #[test]
    fn pipeline_resources_plan_rejects_out_of_bounds_resource_slots() {
        let create_plans = vec![VptNrdPipelineCreatePlan {
            pipeline_index: 0,
            shader_plan_index: 1,
            descriptor_set_layout_index: 0,
            descriptor_set_index: 1,
            shader_identifier: "relax_prepass".to_owned(),
            has_constant_data: false,
            descriptor_binding_count: 1,
        }];
        let shader_plans = vec![VptNrdPipelineShaderPlan {
            spirv_words: vec![0x0723_0203],
            shader_identifier: "relax_prepass".to_owned(),
            has_constant_data: false,
        }];
        let descriptor_resources = descriptor_resources_for_test(1, 1, 1);

        let error = VptNrdPipelineResourcesPlan::from_plans(
            &create_plans,
            &shader_plans,
            vk::DescriptorSetLayout::from_raw(90),
            vk::DescriptorSetLayout::from_raw(99),
            0,
            1,
            &descriptor_resources,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD pipeline shader plan index 1 is out of bounds")
        );

        let create_plans = vec![VptNrdPipelineCreatePlan {
            pipeline_index: 0,
            shader_plan_index: 0,
            descriptor_set_layout_index: 1,
            descriptor_set_index: 0,
            shader_identifier: "relax_prepass".to_owned(),
            has_constant_data: false,
            descriptor_binding_count: 1,
        }];
        let error = VptNrdPipelineResourcesPlan::from_plans(
            &create_plans,
            &shader_plans,
            vk::DescriptorSetLayout::from_raw(90),
            vk::DescriptorSetLayout::from_raw(99),
            0,
            1,
            &descriptor_resources,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD descriptor set layout index 1 is out of bounds")
        );
    }

    #[test]
    fn pipeline_descriptor_binding_plan_expands_layout_ranges_to_vulkan_bindings() {
        let layout_plans = vec![
            VptNrdPipelineLayoutPlan {
                resource_ranges: vec![
                    VptNrdPipelineResourceRangePlan {
                        descriptor_type: NrdDescriptorType::Texture,
                        descriptors_num: 6,
                    },
                    VptNrdPipelineResourceRangePlan {
                        descriptor_type: NrdDescriptorType::StorageTexture,
                        descriptors_num: 2,
                    },
                ],
                has_constant_data: true,
                shader_identifier: "relax_prepass".to_owned(),
            },
            VptNrdPipelineLayoutPlan {
                resource_ranges: vec![VptNrdPipelineResourceRangePlan {
                    descriptor_type: NrdDescriptorType::StorageTexture,
                    descriptors_num: 1,
                }],
                has_constant_data: false,
                shader_identifier: "relax_temporal".to_owned(),
            },
        ];

        let library_desc = NrdLibraryDesc {
            texture_offset: 10,
            sampler_offset: 20,
            constant_buffer_offset: 30,
            storage_texture_and_buffer_offset: 40,
            normal_encoding: NrdNormalEncoding::R10G10B10A2Unorm as u8,
            roughness_encoding: NrdRoughnessEncoding::Linear as u8,
            reserved0: 0,
        };
        let mut snapshot = instance_snapshot_with_pipelines(Vec::new());
        snapshot.resources_base_register_index = 3;

        let binding_plans = VptNrdPipelineDescriptorBindingPlan::from_layout_plans(
            &layout_plans,
            library_desc,
            &snapshot,
        )
        .unwrap();

        assert_eq!(
            binding_plans,
            vec![
                VptNrdPipelineDescriptorBindingPlan {
                    bindings: vec![
                        VptNrdPipelineDescriptorBinding {
                            binding: 13,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 1,
                        },
                        VptNrdPipelineDescriptorBinding {
                            binding: 14,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 1,
                        },
                        VptNrdPipelineDescriptorBinding {
                            binding: 15,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 1,
                        },
                        VptNrdPipelineDescriptorBinding {
                            binding: 16,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 1,
                        },
                        VptNrdPipelineDescriptorBinding {
                            binding: 17,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 1,
                        },
                        VptNrdPipelineDescriptorBinding {
                            binding: 18,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 1,
                        },
                        VptNrdPipelineDescriptorBinding {
                            binding: 43,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                        },
                        VptNrdPipelineDescriptorBinding {
                            binding: 44,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                        },
                    ],
                },
                VptNrdPipelineDescriptorBindingPlan {
                    bindings: vec![VptNrdPipelineDescriptorBinding {
                        binding: 43,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                    },],
                },
            ]
        );
    }

    #[test]
    fn pipeline_descriptor_binding_plan_rejects_zero_descriptor_ranges() {
        let layout_plans = vec![VptNrdPipelineLayoutPlan {
            resource_ranges: vec![VptNrdPipelineResourceRangePlan {
                descriptor_type: NrdDescriptorType::Texture,
                descriptors_num: 0,
            }],
            has_constant_data: false,
            shader_identifier: "empty_range".to_owned(),
        }];

        let error = VptNrdPipelineDescriptorBindingPlan::from_layout_plans(
            &layout_plans,
            nrd_library_desc_for_test(),
            &instance_snapshot_with_pipelines(Vec::new()),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD pipeline resource range has zero descriptors")
        );
    }

    #[test]
    fn pipeline_descriptor_binding_plan_exports_descriptor_binding_specs() {
        let binding_plan = VptNrdPipelineDescriptorBindingPlan {
            bindings: vec![
                VptNrdPipelineDescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 3,
                },
                VptNrdPipelineDescriptorBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 2,
                },
            ],
        };

        let specs = binding_plan.descriptor_binding_specs();

        assert_eq!(
            specs,
            vec![
                crate::render::descriptor::DescriptorBindingSpec {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    count: 3,
                },
                crate::render::descriptor::DescriptorBindingSpec {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    count: 2,
                },
            ]
        );
    }

    #[test]
    fn pipeline_set_layout_plan_places_shared_and_resource_layouts_at_nrd_spaces() {
        let shared_layout = vk::DescriptorSetLayout::from_raw(10);
        let resource_layout = vk::DescriptorSetLayout::from_raw(20);
        let empty_layout = vk::DescriptorSetLayout::from_raw(30);

        let plan = VptNrdPipelineSetLayoutPlan::from_nrd_spaces(
            2,
            4,
            shared_layout,
            resource_layout,
            empty_layout,
        )
        .unwrap();

        assert_eq!(plan.shared_set_index, 2);
        assert_eq!(plan.resources_set_index, 4);
        assert_eq!(
            plan.descriptor_set_layouts,
            vec![
                empty_layout,
                empty_layout,
                shared_layout,
                empty_layout,
                resource_layout
            ]
        );
    }

    #[test]
    fn pipeline_set_layout_plan_rejects_overlapping_nrd_spaces() {
        let error = VptNrdPipelineSetLayoutPlan::from_nrd_spaces(
            1,
            1,
            vk::DescriptorSetLayout::from_raw(10),
            vk::DescriptorSetLayout::from_raw(20),
            vk::DescriptorSetLayout::from_raw(30),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD shared and resource descriptor spaces overlap")
        );
    }

    #[test]
    fn dispatch_constant_upload_plan_copies_each_dispatch_into_frame_slot() {
        let dispatches = vec![
            dispatch_snapshot_with_constants(&[1, 2, 3], false),
            dispatch_snapshot_with_constants(&[4, 5], false),
        ];
        let plan =
            VptNrdDispatchConstantUploadPlan::from_dispatches(&dispatches, 8, 2, 16).unwrap();

        assert_eq!(plan.slot_stride, 16);
        assert_eq!(plan.total_size, 64);
        assert_eq!(plan.slot_offset(0, 0).unwrap(), 0);
        assert_eq!(plan.slot_offset(0, 1).unwrap(), 16);
        assert_eq!(plan.slot_offset(1, 0).unwrap(), 32);
        assert_eq!(plan.slot_offset(1, 1).unwrap(), 48);

        let mut bytes = vec![0xFF; plan.total_size as usize];
        plan.write_frame_constants_into_slice(&mut bytes, 1)
            .unwrap();

        assert_eq!(&bytes[32..40], &[1, 2, 3, 0, 0, 0, 0, 0]);
        assert_eq!(&bytes[40..48], &[0xFF; 8]);
        assert_eq!(&bytes[48..56], &[4, 5, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&bytes[56..64], &[0xFF; 8]);
    }

    #[test]
    fn dispatch_constant_upload_plan_reuses_previous_data_when_marked_matching() {
        let dispatches = vec![
            dispatch_snapshot_with_constants(&[9, 8, 7], false),
            dispatch_snapshot_with_constants(&[], true),
        ];
        let plan = VptNrdDispatchConstantUploadPlan::from_dispatches(&dispatches, 4, 1, 4).unwrap();
        let mut bytes = vec![0; plan.total_size as usize];

        plan.write_frame_constants_into_slice(&mut bytes, 0)
            .unwrap();

        assert_eq!(&bytes[0..4], &[9, 8, 7, 0]);
        assert_eq!(&bytes[4..8], &[9, 8, 7, 0]);
    }

    #[test]
    fn dispatch_constant_upload_plan_rejects_oversized_constant_data() {
        let dispatches = vec![dispatch_snapshot_with_constants(&[1, 2, 3, 4, 5], false)];

        let error =
            VptNrdDispatchConstantUploadPlan::from_dispatches(&dispatches, 4, 1, 4).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD dispatch constant data exceeds max constant buffer size")
        );
    }

    #[test]
    fn refreshed_constant_upload_plan_allows_smaller_dispatch_list_with_existing_capacity() {
        let current_plan = VptNrdDispatchConstantUploadPlan::from_dispatches(
            &[
                dispatch_snapshot_with_constants(&[1], false),
                dispatch_snapshot_with_constants(&[2], false),
            ],
            4,
            2,
            4,
        )
        .unwrap();
        let refreshed_plan = VptNrdDispatchConstantUploadPlan::from_dispatches(
            &[dispatch_snapshot_with_constants(&[3], false)],
            4,
            2,
            4,
        )
        .unwrap();

        ensure_refreshed_constant_upload_plan_fits_existing_resources(
            &refreshed_plan,
            &current_plan,
            current_plan.total_size,
        )
        .unwrap();
    }

    #[test]
    fn refreshed_constant_upload_plan_rejects_dispatch_list_exceeding_existing_capacity() {
        let current_plan = VptNrdDispatchConstantUploadPlan::from_dispatches(
            &[dispatch_snapshot_with_constants(&[1], false)],
            4,
            2,
            4,
        )
        .unwrap();
        let refreshed_plan = VptNrdDispatchConstantUploadPlan::from_dispatches(
            &[
                dispatch_snapshot_with_constants(&[2], false),
                dispatch_snapshot_with_constants(&[3], false),
            ],
            4,
            2,
            4,
        )
        .unwrap();

        let error = ensure_refreshed_constant_upload_plan_fits_existing_resources(
            &refreshed_plan,
            &current_plan,
            current_plan.total_size,
        )
        .unwrap_err();

        assert!(error.to_string().contains(
            "NRD refreshed dispatch constants exceed shared descriptor resource capacity"
        ));
    }

    #[test]
    fn ready_backend_state_refreshes_dispatch_dependent_plans() {
        let library_desc = nrd_library_desc_for_test();
        let snapshot = instance_snapshot_with_pipelines(vec![pipeline(
            "relax_prepass",
            &[resource_range(NrdDescriptorType::Texture, 1)],
            true,
        )]);
        let initial_dispatches = vec![dispatch_snapshot_with_resources_and_constants(
            &[resource(
                NrdDescriptorType::Texture,
                NrdResourceType::InMv,
                0,
            )],
            &[1, 2, 3, 4],
            false,
        )];
        let mut state = VptNrdReadyBackendState::from_snapshots(
            library_desc,
            snapshot,
            initial_dispatches,
            1280,
            720,
        )
        .unwrap();
        let refreshed_dispatches = vec![dispatch_snapshot_with_resources_and_constants(
            &[resource(
                NrdDescriptorType::Texture,
                NrdResourceType::InNormalRoughness,
                0,
            )],
            &[9, 8, 7, 6],
            false,
        )];

        state.refresh_dispatches(refreshed_dispatches).unwrap();

        assert_eq!(state.dispatches[0].constant_buffer_data, vec![9, 8, 7, 6]);
        assert_eq!(
            state.dispatch_resource_plans[0].bindings[0].resource,
            VptNrdDispatchResource::NormalRoughness
        );
        assert_eq!(
            state.dispatch_descriptor_write_plans[0].writes[0].resource,
            VptNrdDispatchResource::NormalRoughness
        );
    }

    #[test]
    fn ready_backend_state_refreshes_dispatch_count_changes() {
        let library_desc = nrd_library_desc_for_test();
        let snapshot = instance_snapshot_with_pipelines(vec![pipeline(
            "relax_prepass",
            &[resource_range(NrdDescriptorType::Texture, 1)],
            true,
        )]);
        let initial_dispatches = vec![dispatch_snapshot_with_resources_and_constants(
            &[resource(
                NrdDescriptorType::Texture,
                NrdResourceType::InMv,
                0,
            )],
            &[1],
            false,
        )];
        let mut state = VptNrdReadyBackendState::from_snapshots(
            library_desc,
            snapshot,
            initial_dispatches,
            1280,
            720,
        )
        .unwrap();
        let refreshed_dispatches = vec![
            dispatch_snapshot_with_resources_and_constants(
                &[resource(
                    NrdDescriptorType::Texture,
                    NrdResourceType::InMv,
                    0,
                )],
                &[2],
                false,
            ),
            dispatch_snapshot_with_resources_and_constants(
                &[resource(
                    NrdDescriptorType::Texture,
                    NrdResourceType::InMv,
                    0,
                )],
                &[3],
                false,
            ),
        ];

        state.refresh_dispatches(refreshed_dispatches).unwrap();

        assert_eq!(state.dispatches.len(), 2);
        assert_eq!(state.dispatch_resource_plans.len(), 2);
        assert_eq!(state.dispatch_descriptor_write_plans.len(), 2);
        assert_eq!(state.dispatches[0].constant_buffer_data, vec![2]);
        assert_eq!(state.dispatches[1].constant_buffer_data, vec![3]);
    }

    #[test]
    fn descriptor_resource_rebuild_refreshes_dependent_pipeline_layouts_before_destroying_layouts()
    {
        let source =
            crate::render::source_checks::read_source("src/render/passes/vpt_nrd_adapter.rs");
        let rebuild_start = source
            .find("    fn rebuild_descriptor_resources(")
            .expect("rebuild_descriptor_resources should exist");
        let record_start = source[rebuild_start..]
            .find("    pub fn record(")
            .map(|offset| rebuild_start + offset)
            .expect("record should follow rebuild_descriptor_resources");
        let rebuild_body = &source[rebuild_start..record_start];

        assert!(
            rebuild_body.contains("create_pipeline_resources("),
            "descriptor resource rebuild must recreate pipelines because their pipeline layouts reference the rebuilt descriptor set layouts"
        );
        let old_pipeline_destroy = rebuild_body
            .find("old_pipeline_resources.destroy(device)")
            .expect("old pipeline resources must be explicitly destroyed");
        let old_descriptor_destroy = rebuild_body
            .find("old_descriptor_resources.destroy(device)")
            .expect("old descriptor resources must be explicitly destroyed");
        let old_shared_destroy = rebuild_body
            .find("old_shared_descriptor_resources.destroy(device, allocator)")
            .expect("old shared descriptor resources must be explicitly destroyed");

        assert!(
            old_pipeline_destroy < old_descriptor_destroy,
            "old pipelines must be destroyed before old resource descriptor set layouts"
        );
        assert!(
            old_pipeline_destroy < old_shared_destroy,
            "old pipelines must be destroyed before old shared descriptor set layouts"
        );
    }

    #[test]
    fn frame_settings_builder_maps_runtime_state_to_nrd_common_settings() {
        let current_world_to_view = glam::Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let previous_world_to_view = glam::Mat4::from_cols_array(&[
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ]);
        let current_view_to_clip = glam::Mat4::from_cols_array(&[
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
            47.0, 48.0,
        ]);
        let previous_view_to_clip = glam::Mat4::from_cols_array(&[
            49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0,
            63.0, 64.0,
        ]);

        let settings = VptNrdFrameSettings::from_inputs(VptNrdFrameSettingsInputs {
            current_world_to_view,
            previous_world_to_view,
            current_view_to_clip,
            previous_view_to_clip,
            current_resolution: [1280, 720],
            previous_resolution: [640, 360],
            frame_index: 42,
            time_delta_seconds: 1.0 / 60.0,
            reset_history: true,
            history_confidence_available: true,
            relax_atrous_iteration_num: 4,
            enable_validation: true,
        })
        .unwrap();

        assert_eq!(
            settings.common.view_to_clip_matrix,
            current_view_to_clip.to_cols_array()
        );
        assert_eq!(
            settings.common.view_to_clip_matrix_prev,
            previous_view_to_clip.to_cols_array()
        );
        assert_eq!(
            settings.common.world_to_view_matrix,
            current_world_to_view.to_cols_array()
        );
        assert_eq!(
            settings.common.world_to_view_matrix_prev,
            previous_world_to_view.to_cols_array()
        );
        assert_eq!(settings.common.camera_jitter, [0.0, 0.0]);
        assert_eq!(settings.common.camera_jitter_prev, [0.0, 0.0]);
        assert_eq!(
            settings.common.motion_vector_scale,
            [1.0 / 1280.0, 1.0 / 720.0, 1.0]
        );
        assert_eq!(settings.common.resource_size, [1280, 720]);
        assert_eq!(settings.common.resource_size_prev, [640, 360]);
        assert_eq!(settings.common.rect_size, [1280, 720]);
        assert_eq!(settings.common.rect_size_prev, [640, 360]);
        assert_eq!(settings.common.denoising_range, 10_000.0);
        assert_eq!(settings.common.disocclusion_threshold, 0.01);
        assert_eq!(settings.common.disocclusion_threshold_alternate, 0.05);
        assert_eq!(settings.common.split_screen, 0.0);
        assert!((settings.common.time_delta_between_frames - 1000.0 / 60.0).abs() < 1.0e-4);
        assert_eq!(settings.common.view_z_scale, 1.0);
        assert_eq!(settings.common.frame_index, 42);
        assert_eq!(
            settings.common.accumulation_mode,
            NrdAccumulationMode::Restart as u32
        );
        assert_eq!(settings.common.is_motion_vector_in_world_space, 0);
        assert_eq!(settings.common.is_history_confidence_available, 1);
        assert_eq!(settings.common.is_disocclusion_threshold_mix_available, 0);
        assert_eq!(settings.common.enable_validation, 1);
    }

    #[test]
    fn frame_settings_builder_rejects_resolution_outside_nrd_u16_limits() {
        let error = VptNrdFrameSettings::from_inputs(VptNrdFrameSettingsInputs {
            current_world_to_view: glam::Mat4::IDENTITY,
            previous_world_to_view: glam::Mat4::IDENTITY,
            current_view_to_clip: glam::Mat4::IDENTITY,
            previous_view_to_clip: glam::Mat4::IDENTITY,
            current_resolution: [70_000, 720],
            previous_resolution: [1280, 720],
            frame_index: 0,
            time_delta_seconds: 0.0,
            reset_history: false,
            history_confidence_available: true,
            relax_atrous_iteration_num: 4,
            enable_validation: false,
        })
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD frame resolution exceeds u16")
        );
    }

    #[test]
    fn frame_settings_builder_builds_relax_diffuse_settings_for_vpt() {
        let settings = VptNrdFrameSettings::from_inputs(VptNrdFrameSettingsInputs {
            current_world_to_view: glam::Mat4::IDENTITY,
            previous_world_to_view: glam::Mat4::IDENTITY,
            current_view_to_clip: glam::Mat4::IDENTITY,
            previous_view_to_clip: glam::Mat4::IDENTITY,
            current_resolution: [1280, 720],
            previous_resolution: [1280, 720],
            frame_index: 7,
            time_delta_seconds: -1.0,
            reset_history: false,
            history_confidence_available: false,
            relax_atrous_iteration_num: 1,
            enable_validation: false,
        })
        .unwrap();

        assert_eq!(settings.common.time_delta_between_frames, 0.0);
        assert_eq!(
            settings.common.accumulation_mode,
            NrdAccumulationMode::Continue as u32
        );
        assert_eq!(settings.common.is_history_confidence_available, 0);
        assert_eq!(settings.relax_diffuse.diffuse_max_accumulated_frame_num, 30);
        assert_eq!(
            settings
                .relax_diffuse
                .diffuse_max_fast_accumulated_frame_num,
            6
        );
        assert_eq!(settings.relax_diffuse.history_fix_frame_num, 3);
        assert_eq!(settings.relax_diffuse.diffuse_prepass_blur_radius, 0.0);
        assert_eq!(settings.relax_diffuse.min_hit_distance_weight, 0.1);
        assert_eq!(settings.relax_diffuse.diffuse_phi_luminance, 2.0);
        assert_eq!(settings.relax_diffuse.atrous_iteration_num, 2);
        assert_eq!(settings.relax_diffuse.depth_threshold, 0.003);
        assert_eq!(
            settings.relax_diffuse.checkerboard_mode,
            NrdCheckerboardMode::Off as u32
        );
        assert_eq!(
            settings.relax_diffuse.hit_distance_reconstruction_mode,
            NrdHitDistanceReconstructionMode::Off as u32
        );
        assert_eq!(settings.relax_diffuse.enable_anti_firefly, 1);
        assert_eq!(settings.relax_diffuse.enable_roughness_edge_stopping, 1);
    }

    #[test]
    fn frame_settings_builder_keeps_reblur_firefly_suppression_enabled_for_vpt() {
        let settings = VptNrdFrameSettings::from_inputs(VptNrdFrameSettingsInputs {
            current_world_to_view: glam::Mat4::IDENTITY,
            previous_world_to_view: glam::Mat4::IDENTITY,
            current_view_to_clip: glam::Mat4::IDENTITY,
            previous_view_to_clip: glam::Mat4::IDENTITY,
            current_resolution: [1920, 1080],
            previous_resolution: [1920, 1080],
            frame_index: 0,
            time_delta_seconds: 0.0,
            reset_history: false,
            history_confidence_available: false,
            relax_atrous_iteration_num: 1,
            enable_validation: false,
        })
        .unwrap();

        assert_eq!(settings.reblur_diffuse.diffuse_prepass_blur_radius, 0.0);
        assert_eq!(settings.reblur_diffuse.specular_prepass_blur_radius, 0.0);
        assert_eq!(
            settings
                .reblur_diffuse
                .firefly_suppressor_min_relative_scale,
            2.0
        );
        assert_eq!(settings.reblur_diffuse.enable_anti_firefly, 1);
        assert_eq!(
            settings
                .reblur_diffuse
                .use_prepass_only_for_specular_motion_estimation,
            1
        );
    }

    #[test]
    #[cfg(feature = "nrd")]
    fn native_relax_backend_refreshes_dispatches_after_frame_settings() {
        let mut backend =
            VptNrdReadyBackend::initialize(VptDenoiserMode::Relax, 64, 64, 4).unwrap();
        let settings = VptNrdFrameSettings::from_inputs(VptNrdFrameSettingsInputs {
            current_world_to_view: glam::Mat4::IDENTITY,
            previous_world_to_view: glam::Mat4::IDENTITY,
            current_view_to_clip: glam::Mat4::IDENTITY,
            previous_view_to_clip: glam::Mat4::IDENTITY,
            current_resolution: [64, 64],
            previous_resolution: [64, 64],
            frame_index: 1,
            time_delta_seconds: 1.0 / 60.0,
            reset_history: false,
            history_confidence_available: true,
            relax_atrous_iteration_num: 4,
            enable_validation: false,
        })
        .unwrap();

        backend.update_frame_settings(&settings).unwrap();
    }

    #[test]
    #[cfg(feature = "nrd")]
    fn native_reblur_backend_refreshes_dispatches_after_frame_settings() {
        let mut backend =
            VptNrdReadyBackend::initialize(VptDenoiserMode::Reblur, 64, 64, 4).unwrap();
        assert_eq!(backend.denoiser_mode, VptDenoiserMode::Reblur);
        assert!(!backend.state.dispatches.is_empty());

        let settings = VptNrdFrameSettings::from_inputs(VptNrdFrameSettingsInputs {
            current_world_to_view: glam::Mat4::IDENTITY,
            previous_world_to_view: glam::Mat4::IDENTITY,
            current_view_to_clip: glam::Mat4::IDENTITY,
            previous_view_to_clip: glam::Mat4::IDENTITY,
            current_resolution: [64, 64],
            previous_resolution: [64, 64],
            frame_index: 1,
            time_delta_seconds: 1.0 / 60.0,
            reset_history: false,
            history_confidence_available: true,
            relax_atrous_iteration_num: 4,
            enable_validation: false,
        })
        .unwrap();

        backend.update_frame_settings(&settings).unwrap();
    }

    #[test]
    fn descriptor_pool_plan_sums_pipeline_binding_descriptor_counts() {
        let binding_plans = vec![
            VptNrdPipelineDescriptorBindingPlan {
                bindings: vec![
                    VptNrdPipelineDescriptorBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 6,
                    },
                    VptNrdPipelineDescriptorBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 2,
                    },
                ],
            },
            VptNrdPipelineDescriptorBindingPlan {
                bindings: vec![
                    VptNrdPipelineDescriptorBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 3,
                    },
                    VptNrdPipelineDescriptorBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 4,
                    },
                ],
            },
        ];

        let pool_plan = VptNrdDescriptorPoolPlan::from_binding_plans(&binding_plans).unwrap();

        assert_eq!(
            pool_plan,
            VptNrdDescriptorPoolPlan {
                max_sets: 2,
                pool_sizes: vec![
                    VptNrdDescriptorPoolSizePlan {
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 9,
                    },
                    VptNrdDescriptorPoolSizePlan {
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 6,
                    },
                ],
            }
        );
    }

    #[test]
    fn descriptor_pool_plan_exports_vulkan_pool_sizes() {
        let pool_plan = VptNrdDescriptorPoolPlan {
            max_sets: 3,
            pool_sizes: vec![
                VptNrdDescriptorPoolSizePlan {
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 9,
                },
                VptNrdDescriptorPoolSizePlan {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 6,
                },
            ],
        };

        let pool_sizes = pool_plan.vk_pool_sizes();

        assert_eq!(pool_sizes.len(), 2);
        assert_eq!(pool_sizes[0].ty, vk::DescriptorType::SAMPLED_IMAGE);
        assert_eq!(pool_sizes[0].descriptor_count, 9);
        assert_eq!(pool_sizes[1].ty, vk::DescriptorType::STORAGE_IMAGE);
        assert_eq!(pool_sizes[1].descriptor_count, 6);
    }

    #[test]
    fn descriptor_pool_plan_rejects_empty_pipeline_binding_plans() {
        let error = VptNrdDescriptorPoolPlan::from_binding_plans(&[]).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD pipeline descriptor binding plans are empty")
        );
    }

    #[test]
    fn descriptor_pool_plan_allows_pipelines_without_resource_bindings() {
        let binding_plans = vec![
            VptNrdPipelineDescriptorBindingPlan {
                bindings: Vec::new(),
            },
            VptNrdPipelineDescriptorBindingPlan {
                bindings: vec![VptNrdPipelineDescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 2,
                }],
            },
        ];

        let pool_plan = VptNrdDescriptorPoolPlan::from_binding_plans(&binding_plans).unwrap();

        assert_eq!(
            pool_plan,
            VptNrdDescriptorPoolPlan {
                max_sets: 2,
                pool_sizes: vec![VptNrdDescriptorPoolSizePlan {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 2,
                }],
            }
        );
    }

    #[test]
    fn descriptor_pool_plan_allows_all_pipelines_without_resource_bindings() {
        let binding_plans = vec![VptNrdPipelineDescriptorBindingPlan {
            bindings: Vec::new(),
        }];

        let pool_plan = VptNrdDescriptorPoolPlan::from_binding_plans(&binding_plans).unwrap();

        assert_eq!(
            pool_plan,
            VptNrdDescriptorPoolPlan {
                max_sets: 1,
                pool_sizes: Vec::new(),
            }
        );
    }

    #[test]
    fn descriptor_resources_plan_layout_specs_follow_pipeline_binding_plans() {
        let binding_plans = vec![
            VptNrdPipelineDescriptorBindingPlan {
                bindings: vec![
                    VptNrdPipelineDescriptorBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 3,
                    },
                    VptNrdPipelineDescriptorBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                    },
                ],
            },
            VptNrdPipelineDescriptorBindingPlan {
                bindings: Vec::new(),
            },
        ];

        let layout_specs = VptNrdDescriptorResources::layout_specs(&binding_plans);

        assert_eq!(
            layout_specs,
            vec![
                vec![
                    DescriptorBindingSpec {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        count: 3,
                    },
                    DescriptorBindingSpec {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        count: 1,
                    },
                ],
                Vec::new(),
            ]
        );
    }

    #[test]
    fn dispatch_resource_plan_maps_relax_diffuse_resources_to_adapter_bindings() {
        let dispatch = dispatch_snapshot(&[
            resource(NrdDescriptorType::Texture, NrdResourceType::InMv, 0),
            resource(
                NrdDescriptorType::Texture,
                NrdResourceType::InNormalRoughness,
                0,
            ),
            resource(NrdDescriptorType::Texture, NrdResourceType::InViewZ, 0),
            resource(
                NrdDescriptorType::Texture,
                NrdResourceType::InDiffRadianceHitdist,
                0,
            ),
            resource(
                NrdDescriptorType::Texture,
                NrdResourceType::InDiffConfidence,
                0,
            ),
            resource(
                NrdDescriptorType::StorageTexture,
                NrdResourceType::OutDiffRadianceHitdist,
                0,
            ),
            resource(
                NrdDescriptorType::StorageTexture,
                NrdResourceType::OutValidation,
                0,
            ),
            resource(
                NrdDescriptorType::Texture,
                NrdResourceType::PermanentPool,
                1,
            ),
            resource(
                NrdDescriptorType::StorageTexture,
                NrdResourceType::TransientPool,
                2,
            ),
        ]);
        let pool_plan = VptNrdTexturePoolPlan {
            permanent: vec![pool_image("permanent_0"), pool_image("permanent_1")],
            transient: vec![
                pool_image("transient_0"),
                pool_image("transient_1"),
                pool_image("transient_2"),
            ],
        };

        let plan = VptNrdDispatchResourcePlan::from_dispatch(&dispatch, &pool_plan).unwrap();

        assert_eq!(
            plan.bindings,
            vec![
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::Motion,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::NormalRoughness,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::ViewZ,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::DiffRadianceHitdist,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::DiffConfidence,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::StorageTexture,
                    resource: VptNrdDispatchResource::OutputDiffRadianceHitdist,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::StorageTexture,
                    resource: VptNrdDispatchResource::Validation,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::PermanentPool { index: 1 },
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::StorageTexture,
                    resource: VptNrdDispatchResource::TransientPool { index: 2 },
                },
            ]
        );
    }

    #[test]
    fn dispatch_resource_plan_rejects_unsupported_and_out_of_bounds_pool_resources() {
        let pool_plan = VptNrdTexturePoolPlan {
            permanent: vec![],
            transient: vec![],
        };
        let unsupported = dispatch_snapshot(&[resource(
            NrdDescriptorType::Texture,
            NrdResourceType::InSpecRadianceHitdist,
            0,
        )]);
        let out_of_bounds = dispatch_snapshot(&[resource(
            NrdDescriptorType::Texture,
            NrdResourceType::PermanentPool,
            0,
        )]);

        let unsupported_error =
            VptNrdDispatchResourcePlan::from_dispatch(&unsupported, &pool_plan).unwrap_err();
        let out_of_bounds_error =
            VptNrdDispatchResourcePlan::from_dispatch(&out_of_bounds, &pool_plan).unwrap_err();

        assert!(
            unsupported_error
                .to_string()
                .contains("unsupported NRD RELAX_DIFFUSE resource")
        );
        assert!(
            out_of_bounds_error
                .to_string()
                .contains("NRD permanent texture pool index 0 is out of bounds")
        );
    }

    #[test]
    fn dispatch_resource_plans_are_built_for_each_snapshot_dispatch() {
        let dispatches = vec![
            dispatch_snapshot(&[resource(
                NrdDescriptorType::Texture,
                NrdResourceType::InViewZ,
                0,
            )]),
            dispatch_snapshot(&[resource(
                NrdDescriptorType::StorageTexture,
                NrdResourceType::OutValidation,
                0,
            )]),
        ];
        let pool_plan = VptNrdTexturePoolPlan {
            permanent: Vec::new(),
            transient: Vec::new(),
        };

        let plans = VptNrdDispatchResourcePlan::from_dispatches(&dispatches, &pool_plan).unwrap();

        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].bindings[0].resource, VptNrdDispatchResource::ViewZ);
        assert_eq!(
            plans[1].bindings[0].resource,
            VptNrdDispatchResource::Validation
        );
    }

    #[test]
    fn dispatch_descriptor_write_plan_pairs_dispatch_resources_with_pipeline_bindings() {
        let dispatch = NrdDispatchSnapshot {
            pipeline_index: 1,
            ..dispatch_snapshot(&[
                resource(NrdDescriptorType::Texture, NrdResourceType::InViewZ, 0),
                resource(
                    NrdDescriptorType::StorageTexture,
                    NrdResourceType::OutValidation,
                    0,
                ),
            ])
        };
        let dispatch_plan = VptNrdDispatchResourcePlan {
            bindings: vec![
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::ViewZ,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::StorageTexture,
                    resource: VptNrdDispatchResource::Validation,
                },
            ],
        };
        let pipeline_binding_plans = vec![
            VptNrdPipelineDescriptorBindingPlan {
                bindings: vec![VptNrdPipelineDescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                }],
            },
            VptNrdPipelineDescriptorBindingPlan {
                bindings: vec![
                    VptNrdPipelineDescriptorBinding {
                        binding: 3,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 1,
                    },
                    VptNrdPipelineDescriptorBinding {
                        binding: 4,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                    },
                ],
            },
        ];

        let write_plan = VptNrdDispatchDescriptorWritePlan::from_dispatch_plan(
            &dispatch,
            &dispatch_plan,
            &pipeline_binding_plans,
        )
        .unwrap();

        assert_eq!(
            write_plan,
            VptNrdDispatchDescriptorWritePlan {
                pipeline_index: 1,
                writes: vec![
                    VptNrdDispatchDescriptorWrite {
                        binding: 3,
                        array_element: 0,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        resource: VptNrdDispatchResource::ViewZ,
                    },
                    VptNrdDispatchDescriptorWrite {
                        binding: 4,
                        array_element: 0,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        resource: VptNrdDispatchResource::Validation,
                    },
                ],
            }
        );
    }

    #[test]
    fn dispatch_descriptor_write_plan_expands_descriptor_arrays() {
        let dispatch = dispatch_snapshot(&[
            resource(NrdDescriptorType::Texture, NrdResourceType::InViewZ, 0),
            resource(
                NrdDescriptorType::Texture,
                NrdResourceType::InDiffConfidence,
                0,
            ),
            resource(
                NrdDescriptorType::StorageTexture,
                NrdResourceType::OutValidation,
                0,
            ),
        ]);
        let dispatch_plan = VptNrdDispatchResourcePlan {
            bindings: vec![
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::ViewZ,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::DiffConfidence,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::StorageTexture,
                    resource: VptNrdDispatchResource::Validation,
                },
            ],
        };
        let pipeline_binding_plans = vec![VptNrdPipelineDescriptorBindingPlan {
            bindings: vec![
                VptNrdPipelineDescriptorBinding {
                    binding: 2,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 2,
                },
                VptNrdPipelineDescriptorBinding {
                    binding: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                },
            ],
        }];

        let write_plan = VptNrdDispatchDescriptorWritePlan::from_dispatch_plan(
            &dispatch,
            &dispatch_plan,
            &pipeline_binding_plans,
        )
        .unwrap();

        assert_eq!(
            write_plan.writes,
            vec![
                VptNrdDispatchDescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    resource: VptNrdDispatchResource::ViewZ,
                },
                VptNrdDispatchDescriptorWrite {
                    binding: 2,
                    array_element: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    resource: VptNrdDispatchResource::DiffConfidence,
                },
                VptNrdDispatchDescriptorWrite {
                    binding: 3,
                    array_element: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    resource: VptNrdDispatchResource::Validation,
                },
            ]
        );
    }

    #[test]
    fn dispatch_descriptor_image_write_plan_resolves_named_resources_to_views_and_layouts() {
        let images = AdapterImageInputFixture::new();
        let inputs = images.inputs();
        let plan = VptNrdDispatchDescriptorWritePlan {
            pipeline_index: 3,
            writes: vec![
                descriptor_write(
                    0,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    VptNrdDispatchResource::Motion,
                ),
                descriptor_write(
                    1,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    VptNrdDispatchResource::NormalRoughness,
                ),
                descriptor_write(
                    2,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    VptNrdDispatchResource::ViewZ,
                ),
                descriptor_write(
                    3,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    VptNrdDispatchResource::DiffConfidence,
                ),
                descriptor_write(
                    4,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    VptNrdDispatchResource::SpecConfidence,
                ),
                descriptor_write(
                    5,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    VptNrdDispatchResource::DiffRadianceHitdist,
                ),
                descriptor_write(
                    6,
                    0,
                    vk::DescriptorType::STORAGE_IMAGE,
                    VptNrdDispatchResource::OutputDiffRadianceHitdist,
                ),
                descriptor_write(
                    7,
                    0,
                    vk::DescriptorType::STORAGE_IMAGE,
                    VptNrdDispatchResource::Validation,
                ),
                descriptor_write(
                    8,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    VptNrdDispatchResource::PermanentPool { index: 0 },
                ),
                descriptor_write(
                    9,
                    1,
                    vk::DescriptorType::STORAGE_IMAGE,
                    VptNrdDispatchResource::TransientPool { index: 0 },
                ),
            ],
        };

        let resolved =
            VptNrdResolvedDispatchDescriptorWritePlan::from_write_plan(&plan, &inputs).unwrap();

        assert_eq!(resolved.pipeline_index, 3);
        assert_eq!(
            resolved.writes,
            vec![
                resolved_image_write(0, 0, vk::DescriptorType::SAMPLED_IMAGE, 11),
                resolved_image_write(1, 0, vk::DescriptorType::SAMPLED_IMAGE, 12),
                resolved_image_write(2, 0, vk::DescriptorType::SAMPLED_IMAGE, 13),
                resolved_image_write(3, 0, vk::DescriptorType::SAMPLED_IMAGE, 14),
                resolved_image_write(4, 0, vk::DescriptorType::SAMPLED_IMAGE, 15),
                resolved_image_write(5, 0, vk::DescriptorType::SAMPLED_IMAGE, 16),
                resolved_image_write(6, 0, vk::DescriptorType::STORAGE_IMAGE, 17),
                resolved_image_write(7, 0, vk::DescriptorType::STORAGE_IMAGE, 18),
                resolved_image_write(8, 0, vk::DescriptorType::SAMPLED_IMAGE, 21),
                resolved_image_write(9, 1, vk::DescriptorType::STORAGE_IMAGE, 31),
            ]
        );
        assert_eq!(
            AccessKind::ComputeShaderRead.image_layout(),
            vk::ImageLayout::GENERAL,
            "descriptor layouts must match the current RenderGraph compute image-read layout"
        );
    }

    #[test]
    fn dispatch_descriptor_image_write_plan_rejects_runtime_pool_index_mismatches() {
        let images = AdapterImageInputFixture::new();
        let inputs = images.inputs();
        let permanent_plan = VptNrdDispatchDescriptorWritePlan {
            pipeline_index: 0,
            writes: vec![descriptor_write(
                0,
                0,
                vk::DescriptorType::SAMPLED_IMAGE,
                VptNrdDispatchResource::PermanentPool { index: 1 },
            )],
        };

        let error =
            VptNrdResolvedDispatchDescriptorWritePlan::from_write_plan(&permanent_plan, &inputs)
                .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD permanent texture pool image index 1 is out of bounds")
        );

        let transient_plan = VptNrdDispatchDescriptorWritePlan {
            pipeline_index: 0,
            writes: vec![descriptor_write(
                0,
                0,
                vk::DescriptorType::STORAGE_IMAGE,
                VptNrdDispatchResource::TransientPool { index: 1 },
            )],
        };

        let error =
            VptNrdResolvedDispatchDescriptorWritePlan::from_write_plan(&transient_plan, &inputs)
                .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD transient texture pool image index 1 is out of bounds")
        );
    }

    #[test]
    fn dispatch_descriptor_image_write_plan_rejects_non_image_descriptor_types() {
        let images = AdapterImageInputFixture::new();
        let inputs = images.inputs();
        let plan = VptNrdDispatchDescriptorWritePlan {
            pipeline_index: 0,
            writes: vec![descriptor_write(
                0,
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                VptNrdDispatchResource::Motion,
            )],
        };

        let error =
            VptNrdResolvedDispatchDescriptorWritePlan::from_write_plan(&plan, &inputs).unwrap_err();

        assert!(error.to_string().contains(
            "NRD dispatch descriptor image write uses unsupported descriptor type UNIFORM_BUFFER"
        ));
    }

    #[test]
    fn descriptor_update_plan_pairs_resolved_image_writes_with_descriptor_sets() {
        let descriptor_resources = descriptor_resources_for_test(2, 2, 2);
        let resolved_plans = vec![
            VptNrdResolvedDispatchDescriptorWritePlan {
                pipeline_index: 1,
                writes: vec![
                    resolved_image_write(2, 0, vk::DescriptorType::SAMPLED_IMAGE, 41),
                    resolved_image_write(3, 1, vk::DescriptorType::STORAGE_IMAGE, 42),
                ],
            },
            VptNrdResolvedDispatchDescriptorWritePlan {
                pipeline_index: 0,
                writes: vec![resolved_image_write(
                    4,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    43,
                )],
            },
        ];

        let update_plan = VptNrdDescriptorUpdatePlan::from_resolved_plans(
            &resolved_plans,
            &descriptor_resources,
            1,
        )
        .unwrap();

        assert_eq!(
            update_plan.dispatches,
            vec![
                VptNrdDispatchDescriptorUpdatePlan {
                    pipeline_index: 1,
                    descriptor_set: vk::DescriptorSet::from_raw(206),
                    writes: vec![
                        descriptor_image_update(2, 0, vk::DescriptorType::SAMPLED_IMAGE, 41),
                        descriptor_image_update(3, 1, vk::DescriptorType::STORAGE_IMAGE, 42),
                    ],
                },
                VptNrdDispatchDescriptorUpdatePlan {
                    pipeline_index: 0,
                    descriptor_set: vk::DescriptorSet::from_raw(207),
                    writes: vec![descriptor_image_update(
                        4,
                        0,
                        vk::DescriptorType::SAMPLED_IMAGE,
                        43,
                    )],
                },
            ]
        );

        let image_info = update_plan.dispatches[0].writes[0].descriptor_image_info();
        assert_eq!(image_info.image_view, vk::ImageView::from_raw(41));
        assert_eq!(
            image_info.image_layout,
            AccessKind::ComputeShaderRead.image_layout()
        );
    }

    #[test]
    fn descriptor_update_plan_uses_distinct_sets_for_repeated_pipeline_dispatches() {
        let descriptor_resources = descriptor_resources_for_test(2, 1, 2);
        let resolved_plans = vec![
            VptNrdResolvedDispatchDescriptorWritePlan {
                pipeline_index: 1,
                writes: vec![resolved_image_write(
                    2,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    41,
                )],
            },
            VptNrdResolvedDispatchDescriptorWritePlan {
                pipeline_index: 1,
                writes: vec![resolved_image_write(
                    2,
                    0,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    42,
                )],
            },
        ];

        let update_plan = VptNrdDescriptorUpdatePlan::from_resolved_plans(
            &resolved_plans,
            &descriptor_resources,
            0,
        )
        .unwrap();

        assert_eq!(
            update_plan.dispatches[0].descriptor_set,
            vk::DescriptorSet::from_raw(202)
        );
        assert_eq!(
            update_plan.dispatches[1].descriptor_set,
            vk::DescriptorSet::from_raw(204)
        );
    }

    #[test]
    fn descriptor_update_plan_rejects_pipeline_indices_without_descriptor_sets() {
        let descriptor_resources = descriptor_resources_for_test(1, 1, 1);
        let resolved_plans = vec![VptNrdResolvedDispatchDescriptorWritePlan {
            pipeline_index: 2,
            writes: vec![resolved_image_write(
                0,
                0,
                vk::DescriptorType::SAMPLED_IMAGE,
                41,
            )],
        }];

        let error = VptNrdDescriptorUpdatePlan::from_resolved_plans(
            &resolved_plans,
            &descriptor_resources,
            0,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD descriptor update pipeline index 2 is out of bounds")
        );
    }

    #[test]
    fn texture_pool_graph_resources_transition_pool_images_to_general() {
        let texture_pools = VptNrdTexturePools {
            permanent_pool: vec![fake_image_with_view(21)],
            transient_pool: vec![fake_image_with_view(31)],
        };
        let mut graph = RenderGraph::new();
        let pool_resources =
            import_texture_pool_graph_resources(&mut graph, Some(&texture_pools), false);

        graph.add_pass("nrd_pool_test", QueueType::Compute, |builder| {
            pool_resources.declare_read_write(builder);
            Box::new(|_| {})
        });

        graph.compile().unwrap();

        assert_eq!(graph.barrier_plan().len(), 2);
        for barrier in graph.barrier_plan() {
            assert_eq!(barrier.timing, BarrierTiming::BeforePass);
            assert_eq!(barrier.from, AccessKind::Undefined);
            assert_eq!(barrier.to, AccessKind::ComputeShaderReadWrite);
        }
    }

    #[test]
    fn dispatch_descriptor_write_plan_rejects_out_of_bounds_pipeline_index() {
        let dispatch = NrdDispatchSnapshot {
            pipeline_index: 4,
            ..dispatch_snapshot(&[resource(
                NrdDescriptorType::Texture,
                NrdResourceType::InViewZ,
                0,
            )])
        };
        let dispatch_plan = VptNrdDispatchResourcePlan {
            bindings: vec![VptNrdDispatchResourceBindingPlan {
                descriptor_type: NrdDescriptorType::Texture,
                resource: VptNrdDispatchResource::ViewZ,
            }],
        };
        let pipeline_binding_plans = vec![VptNrdPipelineDescriptorBindingPlan {
            bindings: vec![VptNrdPipelineDescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
            }],
        }];

        let error = VptNrdDispatchDescriptorWritePlan::from_dispatch_plan(
            &dispatch,
            &dispatch_plan,
            &pipeline_binding_plans,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD dispatch pipeline index 4 is out of bounds")
        );
    }

    #[test]
    fn dispatch_descriptor_write_plan_rejects_descriptor_type_mismatch() {
        let dispatch = dispatch_snapshot(&[resource(
            NrdDescriptorType::Texture,
            NrdResourceType::InViewZ,
            0,
        )]);
        let dispatch_plan = VptNrdDispatchResourcePlan {
            bindings: vec![VptNrdDispatchResourceBindingPlan {
                descriptor_type: NrdDescriptorType::Texture,
                resource: VptNrdDispatchResource::ViewZ,
            }],
        };
        let pipeline_binding_plans = vec![VptNrdPipelineDescriptorBindingPlan {
            bindings: vec![VptNrdPipelineDescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
            }],
        }];

        let error = VptNrdDispatchDescriptorWritePlan::from_dispatch_plan(
            &dispatch,
            &dispatch_plan,
            &pipeline_binding_plans,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD dispatch descriptor type does not match pipeline binding")
        );
    }

    #[test]
    fn dispatch_descriptor_write_plan_rejects_mismatched_resource_count() {
        let dispatch = dispatch_snapshot(&[resource(
            NrdDescriptorType::Texture,
            NrdResourceType::InViewZ,
            0,
        )]);
        let dispatch_plan = VptNrdDispatchResourcePlan {
            bindings: vec![VptNrdDispatchResourceBindingPlan {
                descriptor_type: NrdDescriptorType::Texture,
                resource: VptNrdDispatchResource::ViewZ,
            }],
        };
        let pipeline_binding_plans = vec![VptNrdPipelineDescriptorBindingPlan {
            bindings: vec![
                VptNrdPipelineDescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                },
                VptNrdPipelineDescriptorBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                },
            ],
        }];

        let error = VptNrdDispatchDescriptorWritePlan::from_dispatch_plan(
            &dispatch,
            &dispatch_plan,
            &pipeline_binding_plans,
        )
        .unwrap_err();

        assert!(error.to_string().contains(
            "NRD dispatch resource count does not match pipeline descriptor binding count"
        ));

        let extra_dispatch_plan = VptNrdDispatchResourcePlan {
            bindings: vec![
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::Texture,
                    resource: VptNrdDispatchResource::ViewZ,
                },
                VptNrdDispatchResourceBindingPlan {
                    descriptor_type: NrdDescriptorType::StorageTexture,
                    resource: VptNrdDispatchResource::Validation,
                },
            ],
        };
        let shorter_pipeline_binding_plans = vec![VptNrdPipelineDescriptorBindingPlan {
            bindings: vec![VptNrdPipelineDescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
            }],
        }];

        let error = VptNrdDispatchDescriptorWritePlan::from_dispatch_plan(
            &dispatch,
            &extra_dispatch_plan,
            &shorter_pipeline_binding_plans,
        )
        .unwrap_err();

        assert!(error.to_string().contains(
            "NRD dispatch resource count does not match pipeline descriptor binding count"
        ));
    }

    #[test]
    fn dispatch_descriptor_write_plans_reject_mismatched_dispatch_plan_count() {
        let dispatches = vec![dispatch_snapshot(&[resource(
            NrdDescriptorType::Texture,
            NrdResourceType::InViewZ,
            0,
        )])];
        let pipeline_binding_plans = vec![VptNrdPipelineDescriptorBindingPlan {
            bindings: vec![VptNrdPipelineDescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
            }],
        }];

        let error = VptNrdDispatchDescriptorWritePlan::from_dispatches(
            &dispatches,
            &[],
            &pipeline_binding_plans,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD dispatch count does not match dispatch resource plan count")
        );
    }

    #[test]
    fn ready_backend_metadata_counts_dispatch_snapshots_and_resource_plans() {
        let library_desc = NrdLibraryDesc {
            texture_offset: 10,
            sampler_offset: 20,
            constant_buffer_offset: 30,
            storage_texture_and_buffer_offset: 40,
            normal_encoding: NrdNormalEncoding::R10G10B10A2Unorm as u8,
            roughness_encoding: NrdRoughnessEncoding::Linear as u8,
            reserved0: 0,
        };
        let instance_snapshot = instance_snapshot_with_pools(
            &[texture(NrdTextureFormat::R16Sfloat, 1)],
            &[texture(NrdTextureFormat::R16Sfloat, 1)],
        );
        let dispatches = vec![
            dispatch_snapshot(&[resource(
                NrdDescriptorType::Texture,
                NrdResourceType::PermanentPool,
                0,
            )]),
            dispatch_snapshot(&[resource(
                NrdDescriptorType::Texture,
                NrdResourceType::TransientPool,
                0,
            )]),
        ];

        let state = VptNrdReadyBackendState::from_snapshots(
            library_desc,
            instance_snapshot,
            dispatches,
            32,
            16,
        )
        .unwrap();
        let metadata = state.metadata();

        assert_eq!(metadata.library_desc, library_desc);
        assert_eq!(metadata.dispatch_count, 2);
        assert_eq!(metadata.dispatch_resource_plan_count, 2);
        assert_eq!(metadata.pipeline_layout_plan_count, 1);
        assert_eq!(metadata.pipeline_shader_plan_count, 1);
        assert_eq!(metadata.pipeline_descriptor_binding_plan_count, 1);
        assert_eq!(metadata.descriptor_pool_size_count, 1);
        assert_eq!(metadata.dispatch_descriptor_write_plan_count, 2);
    }

    #[test]
    fn ready_backend_state_rejects_empty_pipeline_spirv() {
        let library_desc = NrdLibraryDesc {
            texture_offset: 10,
            sampler_offset: 20,
            constant_buffer_offset: 30,
            storage_texture_and_buffer_offset: 40,
            normal_encoding: NrdNormalEncoding::R10G10B10A2Unorm as u8,
            roughness_encoding: NrdRoughnessEncoding::Linear as u8,
            reserved0: 0,
        };
        let mut instance_snapshot = instance_snapshot_with_pipelines(vec![NrdPipelineSnapshot {
            spirv_bytecode: Vec::new(),
            resource_ranges: vec![resource_range(NrdDescriptorType::Texture, 1)],
            has_constant_data: false,
            shader_identifier: "empty_shader".to_owned(),
        }]);
        instance_snapshot.permanent_pool = vec![texture(NrdTextureFormat::R16Sfloat, 1)];
        let dispatches = vec![dispatch_snapshot(&[resource(
            NrdDescriptorType::Texture,
            NrdResourceType::PermanentPool,
            0,
        )])];

        let error = match VptNrdReadyBackendState::from_snapshots(
            library_desc,
            instance_snapshot,
            dispatches,
            32,
            16,
        ) {
            Ok(_) => panic!("ready backend state should reject empty NRD pipeline SPIR-V"),
            Err(error) => error,
        };

        assert!(error.chain().any(|cause| {
            cause
                .to_string()
                .contains("NRD pipeline shader bytecode is empty")
        }));
    }

    #[test]
    fn ready_backend_metadata_counts_each_pipeline_artifact_plan() {
        let library_desc = NrdLibraryDesc {
            texture_offset: 10,
            sampler_offset: 20,
            constant_buffer_offset: 30,
            storage_texture_and_buffer_offset: 40,
            normal_encoding: NrdNormalEncoding::R10G10B10A2Unorm as u8,
            roughness_encoding: NrdRoughnessEncoding::Linear as u8,
            reserved0: 0,
        };
        let instance_snapshot = instance_snapshot_with_pipelines(vec![
            pipeline(
                "relax_prepass",
                &[resource_range(NrdDescriptorType::Texture, 1)],
                false,
            ),
            pipeline(
                "relax_temporal",
                &[resource_range(NrdDescriptorType::StorageTexture, 1)],
                true,
            ),
        ]);
        let dispatches = vec![dispatch_snapshot(&[resource(
            NrdDescriptorType::Texture,
            NrdResourceType::InMv,
            0,
        )])];

        let state = VptNrdReadyBackendState::from_snapshots(
            library_desc,
            instance_snapshot,
            dispatches,
            32,
            16,
        )
        .unwrap();
        let metadata = state.metadata();

        assert_eq!(metadata.pipeline_count, 2);
        assert_eq!(metadata.pipeline_layout_plan_count, 2);
        assert_eq!(metadata.pipeline_shader_plan_count, 2);
        assert_eq!(metadata.pipeline_create_plan_count, 2);
        assert_eq!(metadata.pipeline_descriptor_binding_plan_count, 2);
    }

    #[test]
    fn dispatch_resource_plan_rejects_out_of_bounds_transient_pool_resources() {
        let dispatch = dispatch_snapshot(&[resource(
            NrdDescriptorType::Texture,
            NrdResourceType::TransientPool,
            0,
        )]);
        let pool_plan = VptNrdTexturePoolPlan {
            permanent: Vec::new(),
            transient: Vec::new(),
        };

        let error = VptNrdDispatchResourcePlan::from_dispatch(&dispatch, &pool_plan).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("NRD transient texture pool index 0 is out of bounds")
        );
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
                resource_ranges: vec![resource_range(NrdDescriptorType::Texture, 1)],
                has_constant_data: false,
                shader_identifier: "test".to_owned(),
            }],
            permanent_pool: permanent_pool.to_vec(),
            transient_pool: transient_pool.to_vec(),
        }
    }

    fn instance_snapshot_with_pipelines(
        pipelines: Vec<NrdPipelineSnapshot>,
    ) -> NrdInstanceSnapshot {
        NrdInstanceSnapshot {
            constant_buffer_and_samplers_space_index: 0,
            resources_space_index: 0,
            constant_buffer_register_index: 0,
            samplers_base_register_index: 0,
            resources_base_register_index: 0,
            constant_buffer_max_data_size: 0,
            samplers: vec![NrdSamplerDesc { mode: 0 }],
            pipelines,
            permanent_pool: Vec::new(),
            transient_pool: Vec::new(),
        }
    }

    fn pipeline(
        shader_identifier: &str,
        resource_ranges: &[crate::render::nrd_adapter::NrdResourceRangeDesc],
        has_constant_data: bool,
    ) -> NrdPipelineSnapshot {
        pipeline_with_raw_range(shader_identifier, resource_ranges, has_constant_data)
    }

    fn pipeline_with_raw_range(
        shader_identifier: &str,
        resource_ranges: &[crate::render::nrd_adapter::NrdResourceRangeDesc],
        has_constant_data: bool,
    ) -> NrdPipelineSnapshot {
        NrdPipelineSnapshot {
            spirv_bytecode: vec![0x0723_0203],
            resource_ranges: resource_ranges.to_vec(),
            has_constant_data,
            shader_identifier: shader_identifier.to_owned(),
        }
    }

    fn resource_range(
        descriptor_type: NrdDescriptorType,
        descriptors_num: u32,
    ) -> crate::render::nrd_adapter::NrdResourceRangeDesc {
        crate::render::nrd_adapter::NrdResourceRangeDesc {
            descriptor_type: descriptor_type as u32,
            descriptors_num,
        }
    }

    fn pool_image(name: &str) -> VptNrdTexturePoolImagePlan {
        VptNrdTexturePoolImagePlan {
            name: name.to_owned(),
            width: 16,
            height: 16,
            format: vk::Format::R16_SFLOAT,
            downsample_factor: 1,
        }
    }

    struct AdapterImageInputFixture {
        motion: GpuImage,
        normal_roughness: GpuImage,
        view_z: GpuImage,
        diff_confidence: GpuImage,
        spec_confidence: GpuImage,
        diff_radiance_hitdist: GpuImage,
        output_diff_radiance_hitdist: GpuImage,
        validation: GpuImage,
        permanent_pool: Vec<GpuImage>,
        transient_pool: Vec<GpuImage>,
    }

    impl AdapterImageInputFixture {
        fn new() -> Self {
            Self {
                motion: fake_image_with_view(11),
                normal_roughness: fake_image_with_view(12),
                view_z: fake_image_with_view(13),
                diff_confidence: fake_image_with_view(14),
                spec_confidence: fake_image_with_view(15),
                diff_radiance_hitdist: fake_image_with_view(16),
                output_diff_radiance_hitdist: fake_image_with_view(17),
                validation: fake_image_with_view(18),
                permanent_pool: vec![fake_image_with_view(21)],
                transient_pool: vec![fake_image_with_view(31)],
            }
        }

        fn inputs(&self) -> VptNrdAdapterImageInputs<'_> {
            VptNrdAdapterImageInputs {
                motion: &self.motion,
                normal_roughness: &self.normal_roughness,
                view_z: &self.view_z,
                diff_confidence: &self.diff_confidence,
                spec_confidence: &self.spec_confidence,
                diff_radiance_hitdist: &self.diff_radiance_hitdist,
                output_diff_radiance_hitdist: &self.output_diff_radiance_hitdist,
                validation: &self.validation,
                permanent_pool: &self.permanent_pool,
                transient_pool: &self.transient_pool,
            }
        }
    }

    fn fake_image_with_view(id: u64) -> GpuImage {
        GpuImage {
            handle: vk::Image::from_raw(1000 + id),
            view: vk::ImageView::from_raw(id),
            extent: vk::Extent3D {
                width: 16,
                height: 16,
                depth: 1,
            },
            format: vk::Format::R16G16B16A16_SFLOAT,
            allocation: None,
            current_layout: vk::ImageLayout::UNDEFINED,
        }
    }

    fn descriptor_write(
        binding: u32,
        array_element: u32,
        descriptor_type: vk::DescriptorType,
        resource: VptNrdDispatchResource,
    ) -> VptNrdDispatchDescriptorWrite {
        VptNrdDispatchDescriptorWrite {
            binding,
            array_element,
            descriptor_type,
            resource,
        }
    }

    fn resolved_image_write(
        binding: u32,
        array_element: u32,
        descriptor_type: vk::DescriptorType,
        image_view: u64,
    ) -> VptNrdResolvedDescriptorImageWrite {
        let image_layout = match descriptor_type {
            vk::DescriptorType::SAMPLED_IMAGE => AccessKind::ComputeShaderRead.image_layout(),
            vk::DescriptorType::STORAGE_IMAGE => AccessKind::ComputeShaderWrite.image_layout(),
            _ => unreachable!("test helper only resolves image descriptors"),
        };
        VptNrdResolvedDescriptorImageWrite {
            binding,
            array_element,
            descriptor_type,
            image_view: vk::ImageView::from_raw(image_view),
            image_layout,
        }
    }

    fn descriptor_image_update(
        binding: u32,
        array_element: u32,
        descriptor_type: vk::DescriptorType,
        image_view: u64,
    ) -> VptNrdDescriptorImageUpdate {
        let image_layout = match descriptor_type {
            vk::DescriptorType::SAMPLED_IMAGE => AccessKind::ComputeShaderRead.image_layout(),
            vk::DescriptorType::STORAGE_IMAGE => AccessKind::ComputeShaderWrite.image_layout(),
            _ => unreachable!("test helper only resolves image descriptors"),
        };
        VptNrdDescriptorImageUpdate {
            binding,
            array_element,
            descriptor_type,
            image_view: vk::ImageView::from_raw(image_view),
            image_layout,
        }
    }

    fn descriptor_resources_for_test(
        pipeline_count: usize,
        frame_count: usize,
        dispatch_slot_count: usize,
    ) -> VptNrdDescriptorResources {
        let set_count = pipeline_count * frame_count * dispatch_slot_count;
        VptNrdDescriptorResources {
            descriptor_set_layouts: (0..pipeline_count)
                .map(|index| vk::DescriptorSetLayout::from_raw(101 + index as u64))
                .collect(),
            descriptor_pool: DescriptorPool {
                handle: vk::DescriptorPool::from_raw(301),
            },
            descriptor_sets: (0..set_count)
                .map(|index| vk::DescriptorSet::from_raw(201 + index as u64))
                .collect(),
            frame_count,
            dispatch_slot_count,
            pipeline_count,
        }
    }

    fn nrd_library_desc_for_test() -> NrdLibraryDesc {
        NrdLibraryDesc {
            texture_offset: 0,
            sampler_offset: 0,
            constant_buffer_offset: 0,
            storage_texture_and_buffer_offset: 100,
            normal_encoding: NrdNormalEncoding::R10G10B10A2Unorm as u8,
            roughness_encoding: NrdRoughnessEncoding::Linear as u8,
            reserved0: 0,
        }
    }

    #[test]
    fn validate_nrd_library_desc_accepts_expected_pack_encoding() {
        validate_nrd_library_desc(nrd_library_desc_for_test()).unwrap();
    }

    #[test]
    fn validate_nrd_library_desc_rejects_unexpected_pack_encoding() {
        let mut library_desc = nrd_library_desc_for_test();
        library_desc.normal_encoding = NrdNormalEncoding::Rgba8Unorm as u8;

        let error = validate_nrd_library_desc(library_desc).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("unsupported NRD normal encoding")
        );
    }

    fn resource(
        descriptor_type: NrdDescriptorType,
        resource_type: NrdResourceType,
        index_in_pool: u16,
    ) -> crate::render::nrd_adapter::NrdResourceDesc {
        crate::render::nrd_adapter::NrdResourceDesc {
            descriptor_type: descriptor_type as u32,
            resource_type: resource_type as u32,
            index_in_pool,
            reserved0: 0,
        }
    }

    fn dispatch_snapshot(
        resources: &[crate::render::nrd_adapter::NrdResourceDesc],
    ) -> NrdDispatchSnapshot {
        NrdDispatchSnapshot {
            name: "Relax Diffuse".to_owned(),
            identifier: 0,
            resources: resources.to_vec(),
            constant_buffer_data: Vec::new(),
            constant_buffer_data_matches_previous_dispatch: false,
            pipeline_index: 0,
            grid_width: 16,
            grid_height: 16,
        }
    }

    fn dispatch_snapshot_with_constants(
        constant_buffer_data: &[u8],
        matches_previous_dispatch: bool,
    ) -> NrdDispatchSnapshot {
        let mut dispatch = dispatch_snapshot(&[]);
        dispatch.constant_buffer_data = constant_buffer_data.to_vec();
        dispatch.constant_buffer_data_matches_previous_dispatch = matches_previous_dispatch;
        dispatch
    }

    fn dispatch_snapshot_with_resources_and_constants(
        resources: &[crate::render::nrd_adapter::NrdResourceDesc],
        constant_buffer_data: &[u8],
        matches_previous_dispatch: bool,
    ) -> NrdDispatchSnapshot {
        let mut dispatch = dispatch_snapshot(resources);
        dispatch.constant_buffer_data = constant_buffer_data.to_vec();
        dispatch.constant_buffer_data_matches_previous_dispatch = matches_previous_dispatch;
        dispatch
    }
}
