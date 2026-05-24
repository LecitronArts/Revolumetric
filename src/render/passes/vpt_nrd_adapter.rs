use anyhow::{Context, Result};
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::nrd_adapter::{
    NrdDescriptorType, NrdDispatchSnapshot, NrdInstance, NrdInstanceSnapshot, NrdLibraryDesc,
    NrdResourceType, NrdTextureImageDesc,
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
    texture_pools: Option<VptNrdTexturePools>,
    descriptor_resources: Option<VptNrdDescriptorResources>,
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
    pub pipeline_descriptor_binding_plan_count: usize,
    pub descriptor_pool_size_count: usize,
    pub dispatch_descriptor_write_plan_count: usize,
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
    _instance: NrdInstance,
    state: VptNrdReadyBackendState,
}

struct VptNrdReadyBackendState {
    library_desc: NrdLibraryDesc,
    instance_snapshot: NrdInstanceSnapshot,
    texture_pool_plan: VptNrdTexturePoolPlan,
    pipeline_layout_plans: Vec<VptNrdPipelineLayoutPlan>,
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

struct VptNrdDescriptorResources {
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
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
        let texture_pools = create_texture_pools(device, allocator, &backend)?;
        let descriptor_resources = match create_descriptor_resources(device, &backend) {
            Ok(resources) => resources,
            Err(error) => {
                if let Some(texture_pools) = texture_pools {
                    texture_pools.destroy(device, allocator);
                }
                return Err(error);
            }
        };
        let images = match create_adapter_images(device, allocator, info.width, info.height) {
            Ok(images) => images,
            Err(error) => {
                if let Some(descriptor_resources) = descriptor_resources {
                    descriptor_resources.destroy(device);
                }
                if let Some(texture_pools) = texture_pools {
                    texture_pools.destroy(device, allocator);
                }
                return Err(error);
            }
        };
        Ok(Self {
            nrd_diff_radiance_hitdist: images.nrd_diff_radiance_hitdist,
            nrd_validation: images.nrd_validation,
            backend,
            texture_pools,
            descriptor_resources,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdAdapterPassResizeInfo<'_>,
    ) -> Result<()> {
        let _ = info.scene_ubo.frame_count();
        let new_backend = VptNrdAdapterBackend::initialize_relax(info.width, info.height);
        if let Some(reason) = new_backend.unavailable_reason() {
            tracing::warn!(
                reason,
                "VPT NRD adapter backend unavailable after resize; RELAX dispatch remains disabled"
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
        let new_descriptor_resources = match create_descriptor_resources(device, &new_backend) {
            Ok(resources) => resources,
            Err(error) => {
                if let Some(new_texture_pools) = new_texture_pools {
                    new_texture_pools.destroy(device, allocator);
                }
                new_images.destroy(device, allocator);
                return Err(error);
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
        let old_descriptor_resources =
            std::mem::replace(&mut self.descriptor_resources, new_descriptor_resources);
        let old_backend = std::mem::replace(&mut self.backend, new_backend);
        if let Some(old_descriptor_resources) = old_descriptor_resources {
            old_descriptor_resources.destroy(device);
        }
        old_images.destroy(device, allocator);
        if let Some(old_texture_pools) = old_texture_pools {
            old_texture_pools.destroy(device, allocator);
        }
        drop(old_backend);
        Ok(())
    }

    pub fn record(&self, _device: &ash::Device, _cmd: vk::CommandBuffer, _frame_slot: usize) {
        let _ = (
            self.backend.is_ready(),
            self.backend.dispatch_count(),
            self.backend.unavailable_reason(),
            self.backend.ready_metadata(),
            self.backend.texture_pool_plan(),
            self.descriptor_resources.as_ref(),
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
        if let Some(descriptor_resources) = self.descriptor_resources {
            descriptor_resources.destroy(device);
        }
        if let Some(texture_pools) = self.texture_pools {
            texture_pools.destroy(device, allocator);
        }
        self.nrd_diff_radiance_hitdist.destroy(device, allocator);
        self.nrd_validation.destroy(device, allocator);
    }
}

impl VptNrdAdapterBackend {
    pub fn initialize_relax(width: u32, height: u32) -> Self {
        match VptNrdReadyBackend::initialize_relax(width, height) {
            Ok(backend) => Self::Ready(Box::new(backend)),
            Err(error) => Self::Unavailable(error.to_string()),
        }
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

impl VptNrdReadyBackend {
    fn initialize_relax(width: u32, height: u32) -> Result<Self> {
        let mut instance = NrdInstance::relax_diffuse(width, height)?;
        let library_desc = NrdInstance::library_desc()?;
        let instance_snapshot = instance.instance_snapshot()?;
        let dispatches = instance.dispatch_snapshot()?;
        Self::from_instance(
            instance,
            library_desc,
            instance_snapshot,
            dispatches,
            width,
            height,
        )
    }

    fn from_instance(
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
            _instance: instance,
            state,
        })
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
        let pipeline_descriptor_binding_plans =
            VptNrdPipelineDescriptorBindingPlan::from_layout_plans(&pipeline_layout_plans)
                .context("failed to build VPT NRD pipeline descriptor binding plans")?;
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
            pipeline_descriptor_binding_plan_count: self.pipeline_descriptor_binding_plans.len(),
            descriptor_pool_size_count: self.descriptor_pool_plan.pool_sizes.len(),
            dispatch_descriptor_write_plan_count: self.dispatch_descriptor_write_plans.len(),
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

impl VptNrdPipelineDescriptorBindingPlan {
    pub fn from_layout_plans(layout_plans: &[VptNrdPipelineLayoutPlan]) -> Result<Vec<Self>> {
        layout_plans.iter().map(Self::from_layout_plan).collect()
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

    fn from_layout_plan(layout_plan: &VptNrdPipelineLayoutPlan) -> Result<Self> {
        let bindings = layout_plan
            .resource_ranges
            .iter()
            .enumerate()
            .map(|(binding, range)| {
                anyhow::ensure!(
                    range.descriptors_num > 0,
                    "NRD pipeline resource range has zero descriptors"
                );
                let descriptor_type = match range.descriptor_type {
                    NrdDescriptorType::Texture => vk::DescriptorType::SAMPLED_IMAGE,
                    NrdDescriptorType::StorageTexture => vk::DescriptorType::STORAGE_IMAGE,
                    other => anyhow::bail!("unsupported NRD pipeline descriptor type {other:?}"),
                };
                Ok(VptNrdPipelineDescriptorBinding {
                    binding: binding as u32,
                    descriptor_type,
                    descriptor_count: range.descriptors_num,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { bindings })
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

fn nrd_descriptor_type_to_vk(descriptor_type: NrdDescriptorType) -> Result<vk::DescriptorType> {
    match descriptor_type {
        NrdDescriptorType::Texture => Ok(vk::DescriptorType::SAMPLED_IMAGE),
        NrdDescriptorType::StorageTexture => Ok(vk::DescriptorType::STORAGE_IMAGE),
        other => anyhow::bail!("unsupported NRD descriptor type {other:?}"),
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

impl VptNrdDescriptorResources {
    fn create(
        device: &ash::Device,
        binding_plans: &[VptNrdPipelineDescriptorBindingPlan],
        pool_plan: &VptNrdDescriptorPoolPlan,
    ) -> Result<Self> {
        let descriptor_set_layouts = create_descriptor_set_layouts(device, binding_plans)?;
        let pool_sizes = pool_plan.vk_pool_sizes();
        let descriptor_pool = match DescriptorPool::new(device, pool_plan.max_sets, &pool_sizes) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_descriptor_set_layouts(device, descriptor_set_layouts);
                return Err(error);
            }
        };
        let descriptor_sets = match descriptor_pool.allocate(device, &descriptor_set_layouts) {
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
}

fn create_descriptor_resources(
    device: &ash::Device,
    backend: &VptNrdAdapterBackend,
) -> Result<Option<VptNrdDescriptorResources>> {
    let (Some(binding_plans), Some(pool_plan)) = (
        backend.pipeline_descriptor_binding_plans(),
        backend.descriptor_pool_plan(),
    ) else {
        return Ok(None);
    };
    VptNrdDescriptorResources::create(device, binding_plans, pool_plan).map(Some)
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

        let binding_plans =
            VptNrdPipelineDescriptorBindingPlan::from_layout_plans(&layout_plans).unwrap();

        assert_eq!(
            binding_plans,
            vec![
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
                    bindings: vec![VptNrdPipelineDescriptorBinding {
                        binding: 0,
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

        let error =
            VptNrdPipelineDescriptorBindingPlan::from_layout_plans(&layout_plans).unwrap_err();

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
        assert_eq!(metadata.pipeline_descriptor_binding_plan_count, 1);
        assert_eq!(metadata.descriptor_pool_size_count, 1);
        assert_eq!(metadata.dispatch_descriptor_write_plan_count, 2);
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
}
