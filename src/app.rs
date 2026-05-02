use anyhow::{Context, Result};
use ash::vk;
use tracing_subscriber::{EnvFilter, fmt};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::platform::input::InputState;
use crate::scene::camera::update_fly_camera;
use crate::scene::components::CameraRig;

use crate::ecs::schedule::{Schedule, Stage};
use crate::ecs::world::World;
use crate::platform::time::Time;
use crate::platform::window::WindowDescriptor;
use crate::render::camera::compute_pixel_to_ray;
use crate::render::device::RenderDevice;
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler, GpuProfilerConfig};
use crate::render::graph::RenderGraph;
use crate::render::passes::blit_to_swapchain;
use crate::render::passes::lighting::LightingPass;
use crate::render::passes::postprocess::PostprocessPass;
use crate::render::passes::primary_ray::PrimaryRayPass;
use crate::render::passes::restir_di::{RestirDiPass, RestirDiPassCreateInfo};
use crate::render::passes::vpt::VptPass;
use crate::render::resource::{AccessKind, QueueType};
use crate::render::restir_di::{RestirDiSettings, build_direct_lights_from_ucvh};
use crate::render::scene_ubo::{
    LightingSettings, RenderMode, SceneUniformBuffer, SceneUniformInputs, build_scene_uniforms,
};
use crate::scene::light::DirectionalLight;
use crate::scene::systems;
use crate::voxel::generator;
use crate::voxel::gpu_upload::UcvhGpuResources;
use crate::voxel::ucvh::{Ucvh, UcvhConfig};

pub fn run() -> Result<()> {
    init_tracing();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = RevolumetricApp::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn swapchain_access_from_layout(layout: vk::ImageLayout) -> Result<AccessKind> {
    AccessKind::from_swapchain_layout(layout)
        .with_context(|| format!("unsupported tracked swapchain image layout: {layout:?}"))
}

fn add_swapchain_clear_present_pass(
    graph: &mut RenderGraph<'_>,
    dst_image: vk::Image,
    dst_extent: vk::Extent2D,
    dst_format: vk::Format,
    current_layout: vk::ImageLayout,
) -> Result<()> {
    let current_access = swapchain_access_from_layout(current_layout)?;
    let swapchain = graph.import_image_with_access(
        dst_image,
        dst_extent.width,
        dst_extent.height,
        dst_format,
        vk::ImageUsageFlags::TRANSFER_DST,
        current_access,
    );

    graph.add_pass("clear_swapchain", QueueType::Graphics, |builder| {
        builder.write_as(swapchain, AccessKind::TransferWrite);
        builder.finish_as(swapchain, AccessKind::Present);
        Box::new(move |ctx| {
            let color = vk::ClearColorValue {
                float32: [0.015, 0.018, 0.022, 1.0],
            };
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            unsafe {
                ctx.device.cmd_clear_color_image(
                    ctx.command_buffer,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &color,
                    std::slice::from_ref(&range),
                );
            }
        })
    });
    Ok(())
}

fn parse_exit_after_frames() -> Option<u64> {
    std::env::var("REVOLUMETRIC_EXIT_AFTER_FRAMES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&frames| frames > 0)
}

struct RevolumetricApp {
    world: World,
    schedule: Schedule,
    renderer: Option<RenderDevice>,
    gpu_profiler: Option<GpuProfiler>,
    primary_ray_pass: Option<PrimaryRayPass>,
    lighting_pass: Option<LightingPass>,
    postprocess_pass: Option<PostprocessPass>,
    vpt_pass: Option<VptPass>,
    restir_di_pass: Option<RestirDiPass>,
    ucvh: Option<Ucvh>,
    ucvh_gpu: Option<UcvhGpuResources>,
    ucvh_uploaded: bool,
    scene_ubo: Option<SceneUniformBuffer>,
    lighting_settings: LightingSettings,
    restir_di_settings: RestirDiSettings,
    vpt_sample_index: u32,
    last_vpt_camera_key: Option<[u32; 13]>,
    vpt_accumulation_needs_init: bool,
    window_descriptor: WindowDescriptor,
    window: Option<Window>,
    window_id: Option<WindowId>,
    initialized: bool,
    last_cursor_pos: Option<(f64, f64)>,
    last_frame_time: Option<std::time::Instant>,
    rendered_frames: u64,
    exit_after_frames: Option<u64>,
}

impl RevolumetricApp {
    fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Time::default());

        let mut schedule = Schedule::new();
        schedule.add_stage(Stage::Startup);
        schedule.add_stage(Stage::PreUpdate);
        schedule.add_stage(Stage::Update);
        schedule.add_stage(Stage::PostUpdate);
        schedule.add_stage(Stage::ExtractRender);
        schedule.add_stage(Stage::PrepareRender);
        schedule.add_stage(Stage::ExecuteRender);

        schedule.add_system(Stage::Startup, systems::bootstrap_scene);

        Self {
            world,
            schedule,
            renderer: None,
            gpu_profiler: None,
            primary_ray_pass: None,
            lighting_pass: None,
            postprocess_pass: None,
            vpt_pass: None,
            restir_di_pass: None,
            ucvh: None,
            ucvh_gpu: None,
            ucvh_uploaded: false,
            scene_ubo: None,
            lighting_settings: LightingSettings::default(),
            restir_di_settings: RestirDiSettings::default(),
            vpt_sample_index: 0,
            last_vpt_camera_key: None,
            vpt_accumulation_needs_init: true,
            window_descriptor: WindowDescriptor::default(),
            window: None,
            window_id: None,
            initialized: false,
            last_cursor_pos: None,
            last_frame_time: None,
            rendered_frames: 0,
            exit_after_frames: parse_exit_after_frames(),
        }
    }

    fn restir_di_vpt_enabled(&self) -> bool {
        self.lighting_settings.render_mode == RenderMode::Vpt && self.restir_di_settings.enabled
    }

    fn resize_render_passes(&mut self, width: u32, height: u32) -> Result<()> {
        // Extract device (Clone) and allocator (raw ptr) to avoid borrow conflicts
        // with pass fields. Safe because allocator lives in self.renderer and isn't
        // moved or dropped during this method.
        let (device, allocator) = match self.renderer.as_ref() {
            Some(r) => (
                r.device().clone(),
                r.allocator() as *const crate::render::allocator::GpuAllocator,
            ),
            None => return Ok(()),
        };
        let allocator = unsafe { &*allocator };

        if let Some(primary) = &mut self.primary_ray_pass {
            primary
                .resize_images(&device, allocator, width, height)
                .context("failed to resize primary ray images")?;
        }
        if let (Some(lighting), Some(primary)) = (&mut self.lighting_pass, &self.primary_ray_pass) {
            lighting
                .resize_images(&device, allocator, width, height, primary)
                .context("failed to resize lighting images")?;
        }
        if self.lighting_settings.render_mode != RenderMode::Vpt
            && let (Some(postprocess), Some(lighting), Some(scene_ubo)) = (
                &mut self.postprocess_pass,
                &self.lighting_pass,
                &self.scene_ubo,
            )
        {
            postprocess
                .resize_images(
                    &device,
                    allocator,
                    width,
                    height,
                    &lighting.output_image,
                    scene_ubo,
                )
                .context("failed to resize postprocess images")?;
        }
        if let (Some(vpt), Some(scene_ubo), Some(ucvh_gpu)) =
            (&mut self.vpt_pass, &self.scene_ubo, &self.ucvh_gpu)
        {
            vpt.resize_images(&device, allocator, width, height, scene_ubo, ucvh_gpu)
                .context("failed to resize VPT images")?;
            self.vpt_sample_index = 0;
            self.last_vpt_camera_key = None;
            self.vpt_accumulation_needs_init = true;
        }
        if self.restir_di_vpt_enabled()
            && let Some(restir_di) = &mut self.restir_di_pass
        {
            restir_di
                .resize_buffers(&device, allocator, width, height)
                .context("failed to resize ReSTIR-DI buffers")?;
        }
        if self.lighting_settings.render_mode == RenderMode::Vpt
            && let (Some(postprocess), Some(vpt), Some(scene_ubo)) =
                (&mut self.postprocess_pass, &self.vpt_pass, &self.scene_ubo)
        {
            postprocess
                .resize_images(
                    &device,
                    allocator,
                    width,
                    height,
                    &vpt.output_image,
                    scene_ubo,
                )
                .context("failed to resize VPT postprocess images")?;
        }

        Ok(())
    }

    fn update_camera(&mut self, dt: f32) {
        // Clone InputState (it's Copy) to avoid borrow conflicts
        let input = match self.world.resource::<InputState>() {
            Some(input) => *input,
            None => return,
        };

        if let Some(rig) = self.world.resource_mut::<CameraRig>() {
            update_fly_camera(rig, input, dt);
        }
    }

    fn tick_frame(&mut self) -> Result<()> {
        // Real delta time
        let now = std::time::Instant::now();
        let dt = match self.last_frame_time {
            Some(last) => now.duration_since(last).as_secs_f32().min(0.1),
            None => 0.0,
        };
        self.last_frame_time = Some(now);

        if let Some(time) = self.world.resource_mut::<Time>() {
            time.advance(dt);
        }

        self.schedule.run_stage(Stage::PreUpdate, &mut self.world)?;
        self.schedule.run_stage(Stage::Update, &mut self.world)?;
        self.schedule
            .run_stage(Stage::PostUpdate, &mut self.world)?;
        self.update_camera(dt);
        self.schedule
            .run_stage(Stage::ExtractRender, &mut self.world)?;
        self.schedule
            .run_stage(Stage::PrepareRender, &mut self.world)?;

        let restir_di_enabled = self.restir_di_vpt_enabled();
        if let Some(renderer) = self.renderer.as_mut() {
            let frame = renderer.begin_frame()?;
            if frame.should_render {
                if let Some(profiler) = &mut self.gpu_profiler {
                    profiler.begin_frame(
                        renderer.device(),
                        frame.command_buffer,
                        frame.frame_slot,
                        frame.frame_index,
                    );
                }

                // Upload UCVH data to GPU (first frame only)
                if !self.ucvh_uploaded {
                    if let (Some(ucvh), Some(gpu)) = (&self.ucvh, &self.ucvh_gpu) {
                        match gpu.upload_all(renderer.device(), frame.command_buffer, ucvh) {
                            Ok(()) => {
                                self.ucvh_uploaded = true;
                                tracing::info!("uploaded UCVH data to GPU");
                            }
                            Err(error) => {
                                tracing::error!(%error, "failed to upload UCVH data to GPU");
                            }
                        }
                    }
                }
                let ucvh_ready = self.ucvh_uploaded;

                let mut graph = RenderGraph::new();
                let profiler = self.gpu_profiler.as_ref();
                let mut vpt_accumulation_written = false;

                if ucvh_ready {
                    let (cam_pos, cam_forward, cam_up, fov_y) = {
                        let rig = self.world.resource::<CameraRig>();
                        match rig {
                            Some(rig) => (
                                rig.camera.position,
                                rig.camera.forward,
                                rig.camera.up,
                                rig.camera.fov_y_radians,
                            ),
                            None => (
                                glam::Vec3::new(64.0, 80.0, -40.0),
                                glam::Vec3::Z,
                                glam::Vec3::Y,
                                std::f32::consts::FRAC_PI_4,
                            ),
                        }
                    };

                    let pixel_to_ray = compute_pixel_to_ray(
                        cam_pos,
                        cam_forward,
                        cam_up,
                        fov_y,
                        frame.swapchain_extent.width,
                        frame.swapchain_extent.height,
                    );
                    let camera_key = [
                        cam_pos.x.to_bits(),
                        cam_pos.y.to_bits(),
                        cam_pos.z.to_bits(),
                        cam_forward.x.to_bits(),
                        cam_forward.y.to_bits(),
                        cam_forward.z.to_bits(),
                        cam_up.x.to_bits(),
                        cam_up.y.to_bits(),
                        cam_up.z.to_bits(),
                        fov_y.to_bits(),
                        frame.swapchain_extent.width,
                        frame.swapchain_extent.height,
                        self.lighting_settings.vpt_max_bounces,
                    ];
                    if self.lighting_settings.render_mode == RenderMode::Vpt {
                        if self.last_vpt_camera_key == Some(camera_key) {
                            self.vpt_sample_index = self.vpt_sample_index.saturating_add(1);
                        } else {
                            self.vpt_sample_index = 0;
                            self.last_vpt_camera_key = Some(camera_key);
                        }
                    } else {
                        self.vpt_sample_index = 0;
                        self.last_vpt_camera_key = None;
                    }
                    let scene_vpt_sample_index = if self.vpt_accumulation_needs_init {
                        0
                    } else {
                        self.vpt_sample_index
                    };

                    // Read DirectionalLight from World
                    let (sun_dir, sun_intensity) = {
                        let light = self.world.resource::<DirectionalLight>();
                        match light {
                            Some(l) => (l.direction, l.intensity),
                            None => (
                                glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
                                glam::Vec3::new(2.0, 1.5, 1.25),
                            ),
                        }
                    };

                    let scene_data = build_scene_uniforms(SceneUniformInputs {
                        pixel_to_ray,
                        resolution: [frame.swapchain_extent.width, frame.swapchain_extent.height],
                        sun_direction: sun_dir,
                        sun_intensity,
                        sky_color: [0.4, 0.5, 0.7],
                        ground_color: [0.15, 0.1, 0.08],
                        time: self
                            .world
                            .resource::<Time>()
                            .map_or(0.0, |t| t.elapsed_seconds),
                        lighting_settings: self.lighting_settings,
                        vpt_sample_index: scene_vpt_sample_index,
                    });

                    if let Some(ubo) = &self.scene_ubo {
                        ubo.update(frame.frame_slot, &scene_data);
                    }

                    let use_vpt = self.lighting_settings.render_mode == RenderMode::Vpt;
                    if use_vpt {
                        if let (Some(vpt), Some(postprocess)) =
                            (&self.vpt_pass, &self.postprocess_pass)
                        {
                            if restir_di_enabled && let Some(restir_di) = &self.restir_di_pass {
                                restir_di.update_uniforms(
                                    frame.frame_slot,
                                    self.restir_di_settings,
                                    frame.frame_index,
                                );

                                let (uniform_buffer, uniform_size, uniform_usage) =
                                    restir_di.uniform_buffer(frame.frame_slot);
                                let (direct_light_buffer, direct_light_size, direct_light_usage) =
                                    restir_di.direct_light_buffer();
                                let (initial_buffer, initial_size, initial_usage) =
                                    restir_di.initial_buffer();
                                let (temporal_buffer, temporal_size, temporal_usage) =
                                    restir_di.temporal_buffer();
                                let (spatial_buffer, spatial_size, spatial_usage) =
                                    restir_di.spatial_buffer();
                                let (history_buffer, history_size, history_usage) =
                                    restir_di.history_buffer();

                                let initial_resource = graph.import_buffer_with_access(
                                    initial_buffer.handle,
                                    initial_size,
                                    initial_usage,
                                    AccessKind::Undefined,
                                );
                                let temporal_resource = graph.import_buffer_with_access(
                                    temporal_buffer.handle,
                                    temporal_size,
                                    temporal_usage,
                                    AccessKind::Undefined,
                                );
                                let spatial_resource = graph.import_buffer_with_access(
                                    spatial_buffer.handle,
                                    spatial_size,
                                    spatial_usage,
                                    AccessKind::Undefined,
                                );
                                let history_resource = graph.import_buffer_with_access(
                                    history_buffer.handle,
                                    history_size,
                                    history_usage,
                                    AccessKind::Undefined,
                                );
                                let uniform_resource = graph.import_buffer_with_access(
                                    uniform_buffer.handle,
                                    uniform_size,
                                    uniform_usage,
                                    AccessKind::ComputeShaderRead,
                                );
                                let direct_light_resource = graph.import_buffer_with_access(
                                    direct_light_buffer.handle,
                                    direct_light_size,
                                    direct_light_usage,
                                    AccessKind::ComputeShaderRead,
                                );

                                let slot = frame.frame_slot;
                                let initial_writes = graph.add_pass(
                                    "restir_di_initial",
                                    QueueType::Compute,
                                    |builder| {
                                        builder.read_as(
                                            uniform_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            direct_light_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.write_as(
                                            initial_resource,
                                            AccessKind::ComputeShaderWrite,
                                        );
                                        Box::new(move |ctx| {
                                            restir_di.record_initial(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                            );
                                        })
                                    },
                                );
                                let initial_dep = initial_writes[0];
                                let temporal_writes = graph.add_pass(
                                    "restir_di_temporal",
                                    QueueType::Compute,
                                    |builder| {
                                        builder.read_as(
                                            uniform_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(initial_dep, AccessKind::ComputeShaderRead);
                                        builder.read_as(
                                            history_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.write_as(
                                            temporal_resource,
                                            AccessKind::ComputeShaderWrite,
                                        );
                                        Box::new(move |ctx| {
                                            restir_di.record_temporal(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                            );
                                        })
                                    },
                                );
                                let temporal_dep = temporal_writes[0];
                                graph.add_pass(
                                    "restir_di_spatial",
                                    QueueType::Compute,
                                    |builder| {
                                        builder.read_as(
                                            uniform_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder
                                            .read_as(temporal_dep, AccessKind::ComputeShaderRead);
                                        builder.write_as(
                                            spatial_resource,
                                            AccessKind::ComputeShaderWrite,
                                        );
                                        Box::new(move |ctx| {
                                            restir_di.record_spatial(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                            );
                                        })
                                    },
                                );
                            }

                            postprocess.update_input_image(
                                renderer.device(),
                                &vpt.output_image,
                                frame.frame_slot,
                            );
                            let vpt_output = vpt.output_image.handle;
                            let vpt_initial_access = if self.vpt_accumulation_needs_init {
                                AccessKind::Undefined
                            } else {
                                AccessKind::ComputeShaderRead
                            };
                            let vpt_write_access =
                                if self.vpt_accumulation_needs_init || self.vpt_sample_index == 0 {
                                    AccessKind::ComputeShaderWrite
                                } else {
                                    AccessKind::ComputeShaderReadWrite
                                };
                            let vpt_resource = graph.import_image_with_access(
                                vpt_output,
                                vpt.output_image.extent.width,
                                vpt.output_image.extent.height,
                                vk::Format::R16G16B16A16_SFLOAT,
                                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                                vpt_initial_access,
                            );
                            vpt_accumulation_written = true;
                            let vpt_writes = graph.add_pass("vpt", QueueType::Compute, |builder| {
                                builder.write_as(vpt_resource, vpt_write_access);
                                let slot = frame.frame_slot;
                                Box::new(move |ctx| {
                                    if let Some(profiler) = profiler {
                                        profiler.begin_scope(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                            GpuProfileScope::Vpt,
                                        );
                                    }
                                    vpt.record(ctx.device, ctx.command_buffer, slot);
                                    if let Some(profiler) = profiler {
                                        profiler.end_scope(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                            GpuProfileScope::Vpt,
                                        );
                                    }
                                })
                            });

                            let postprocess_output = postprocess.output_image.handle;
                            let postprocess_extent = postprocess.output_image.extent;
                            let vpt_dep = vpt_writes[0];
                            let slot = frame.frame_slot;
                            let postprocess_writes =
                                graph.add_pass("postprocess", QueueType::Compute, |builder| {
                                    builder.read_as(vpt_dep, AccessKind::ComputeShaderRead);
                                    builder.create_image(
                                        frame.swapchain_extent.width,
                                        frame.swapchain_extent.height,
                                        vk::Format::R8G8B8A8_UNORM,
                                        vk::ImageUsageFlags::STORAGE
                                            | vk::ImageUsageFlags::TRANSFER_SRC,
                                    );
                                    Box::new(move |ctx| {
                                        if let Some(profiler) = profiler {
                                            profiler.begin_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::Postprocess,
                                            );
                                        }
                                        postprocess.record(ctx.device, ctx.command_buffer, slot);
                                        if let Some(profiler) = profiler {
                                            profiler.end_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::Postprocess,
                                            );
                                        }
                                    })
                                });

                            let src_image = postprocess_output;
                            let src_extent = postprocess_extent;
                            let dst_image = frame.swapchain_image;
                            let dst_extent = frame.swapchain_extent;
                            let dep_handle = postprocess_writes[0];
                            graph.bind_image(dep_handle, src_image);
                            let swapchain_dep = graph.import_image_with_access(
                                dst_image,
                                dst_extent.width,
                                dst_extent.height,
                                frame.swapchain_format,
                                vk::ImageUsageFlags::TRANSFER_DST,
                                swapchain_access_from_layout(frame.swapchain_image_layout)?,
                            );
                            graph.add_pass("blit_to_swapchain", QueueType::Graphics, |builder| {
                                builder.read_as(dep_handle, AccessKind::TransferRead);
                                builder.write_as(swapchain_dep, AccessKind::TransferWrite);
                                builder.finish_as(swapchain_dep, AccessKind::Present);
                                Box::new(move |ctx| {
                                    if let Some(profiler) = profiler {
                                        profiler.begin_scope(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                            GpuProfileScope::BlitToSwapchain,
                                        );
                                    }
                                    blit_to_swapchain::record_blit_core(
                                        ctx.device,
                                        ctx.command_buffer,
                                        src_image,
                                        src_extent,
                                        dst_image,
                                        dst_extent,
                                    );
                                    if let Some(profiler) = profiler {
                                        profiler.end_scope(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                            GpuProfileScope::BlitToSwapchain,
                                        );
                                    }
                                })
                            });
                        } else {
                            self.vpt_sample_index = 0;
                            self.last_vpt_camera_key = None;
                            tracing::warn!(
                                vpt_ready = self.vpt_pass.is_some(),
                                postprocess_ready = self.postprocess_pass.is_some(),
                                "skipping VPT frame until required passes are initialized"
                            );
                        }
                    } else {
                        if let Some(pass) = &self.primary_ray_pass {
                            let primary_ray_writes =
                                graph.add_pass("primary_ray", QueueType::Compute, |builder| {
                                    let _gbp = builder.create_image(
                                        frame.swapchain_extent.width,
                                        frame.swapchain_extent.height,
                                        vk::Format::R32G32B32A32_SFLOAT,
                                        vk::ImageUsageFlags::STORAGE,
                                    );
                                    let _gb0 = builder.create_image(
                                        frame.swapchain_extent.width,
                                        frame.swapchain_extent.height,
                                        vk::Format::R8G8B8A8_UNORM,
                                        vk::ImageUsageFlags::STORAGE
                                            | vk::ImageUsageFlags::TRANSFER_SRC,
                                    );
                                    let _gb1 = builder.create_image(
                                        frame.swapchain_extent.width,
                                        frame.swapchain_extent.height,
                                        vk::Format::R8G8B8A8_UINT,
                                        vk::ImageUsageFlags::STORAGE,
                                    );
                                    let slot = frame.frame_slot;
                                    Box::new(move |ctx| {
                                        if let Some(profiler) = profiler {
                                            profiler.begin_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::PrimaryRay,
                                            );
                                        }
                                        pass.record(ctx.device, ctx.command_buffer, slot);
                                        if let Some(profiler) = profiler {
                                            profiler.end_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::PrimaryRay,
                                            );
                                        }
                                    })
                                });
                            let primary_images = [
                                pass.gbuffer_pos.handle,
                                pass.gbuffer0.handle,
                                pass.gbuffer1.handle,
                            ];
                            for (&handle, &image) in
                                primary_ray_writes.iter().zip(primary_images.iter())
                            {
                                graph.bind_image(handle, image);
                            }

                            if let Some(lighting) = &self.lighting_pass {
                                if let Some(postprocess) = &self.postprocess_pass {
                                    postprocess.update_input_image(
                                        renderer.device(),
                                        &lighting.output_image,
                                        frame.frame_slot,
                                    );
                                }
                                let lighting_output = lighting.output_image.handle;
                                let lighting_extent = lighting.output_image.extent;
                                let dep0 = primary_ray_writes[0];
                                let dep1 = primary_ray_writes[1];
                                let dep2 = primary_ray_writes[2];
                                let slot = frame.frame_slot;

                                let lighting_writes =
                                    graph.add_pass("lighting", QueueType::Compute, |builder| {
                                        builder.read(dep0);
                                        builder.read(dep1);
                                        builder.read(dep2);
                                        let _out = builder.create_image(
                                            frame.swapchain_extent.width,
                                            frame.swapchain_extent.height,
                                            vk::Format::R16G16B16A16_SFLOAT,
                                            vk::ImageUsageFlags::STORAGE
                                                | vk::ImageUsageFlags::TRANSFER_SRC,
                                        );
                                        Box::new(move |ctx| {
                                            if let Some(profiler) = profiler {
                                                profiler.begin_scope(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                    GpuProfileScope::Lighting,
                                                );
                                            }
                                            lighting.record(ctx.device, ctx.command_buffer, slot);
                                            if let Some(profiler) = profiler {
                                                profiler.end_scope(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                    GpuProfileScope::Lighting,
                                                );
                                            }
                                        })
                                    });

                                if let Some(postprocess) = &self.postprocess_pass {
                                    let postprocess_output = postprocess.output_image.handle;
                                    let postprocess_extent = postprocess.output_image.extent;
                                    let lighting_dep = lighting_writes[0];
                                    graph.bind_image(lighting_dep, lighting_output);
                                    let postprocess_writes = graph.add_pass(
                                        "postprocess",
                                        QueueType::Compute,
                                        |builder| {
                                            builder.read_as(
                                                lighting_dep,
                                                AccessKind::ComputeShaderRead,
                                            );
                                            builder.create_image(
                                                frame.swapchain_extent.width,
                                                frame.swapchain_extent.height,
                                                vk::Format::R8G8B8A8_UNORM,
                                                vk::ImageUsageFlags::STORAGE
                                                    | vk::ImageUsageFlags::TRANSFER_SRC,
                                            );
                                            Box::new(move |ctx| {
                                                if let Some(profiler) = profiler {
                                                    profiler.begin_scope(
                                                        ctx.device,
                                                        ctx.command_buffer,
                                                        slot,
                                                        GpuProfileScope::Postprocess,
                                                    );
                                                }
                                                postprocess.record(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                );
                                                if let Some(profiler) = profiler {
                                                    profiler.end_scope(
                                                        ctx.device,
                                                        ctx.command_buffer,
                                                        slot,
                                                        GpuProfileScope::Postprocess,
                                                    );
                                                }
                                            })
                                        },
                                    );

                                    let src_image = postprocess_output;
                                    let src_extent = postprocess_extent;
                                    let dst_image = frame.swapchain_image;
                                    let dst_extent = frame.swapchain_extent;
                                    let dep_handle = postprocess_writes[0];
                                    graph.bind_image(dep_handle, src_image);
                                    let swapchain_dep = graph.import_image_with_access(
                                        dst_image,
                                        dst_extent.width,
                                        dst_extent.height,
                                        frame.swapchain_format,
                                        vk::ImageUsageFlags::TRANSFER_DST,
                                        swapchain_access_from_layout(frame.swapchain_image_layout)?,
                                    );
                                    graph.add_pass(
                                        "blit_to_swapchain",
                                        QueueType::Graphics,
                                        |builder| {
                                            builder.read_as(dep_handle, AccessKind::TransferRead);
                                            builder
                                                .write_as(swapchain_dep, AccessKind::TransferWrite);
                                            builder.finish_as(swapchain_dep, AccessKind::Present);
                                            Box::new(move |ctx| {
                                                if let Some(profiler) = profiler {
                                                    profiler.begin_scope(
                                                        ctx.device,
                                                        ctx.command_buffer,
                                                        slot,
                                                        GpuProfileScope::BlitToSwapchain,
                                                    );
                                                }
                                                blit_to_swapchain::record_blit_core(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    src_image,
                                                    src_extent,
                                                    dst_image,
                                                    dst_extent,
                                                );
                                                if let Some(profiler) = profiler {
                                                    profiler.end_scope(
                                                        ctx.device,
                                                        ctx.command_buffer,
                                                        slot,
                                                        GpuProfileScope::BlitToSwapchain,
                                                    );
                                                }
                                            })
                                        },
                                    );
                                } else {
                                    let src_image = lighting_output;
                                    let src_extent = lighting_extent;
                                    let dst_image = frame.swapchain_image;
                                    let dst_extent = frame.swapchain_extent;
                                    let dep_handle = lighting_writes[0];
                                    graph.bind_image(dep_handle, src_image);
                                    let swapchain_dep = graph.import_image_with_access(
                                        dst_image,
                                        dst_extent.width,
                                        dst_extent.height,
                                        frame.swapchain_format,
                                        vk::ImageUsageFlags::TRANSFER_DST,
                                        swapchain_access_from_layout(frame.swapchain_image_layout)?,
                                    );
                                    graph.add_pass(
                                        "blit_to_swapchain",
                                        QueueType::Graphics,
                                        |builder| {
                                            builder.read_as(dep_handle, AccessKind::TransferRead);
                                            builder
                                                .write_as(swapchain_dep, AccessKind::TransferWrite);
                                            builder.finish_as(swapchain_dep, AccessKind::Present);
                                            Box::new(move |ctx| {
                                                if let Some(profiler) = profiler {
                                                    profiler.begin_scope(
                                                        ctx.device,
                                                        ctx.command_buffer,
                                                        slot,
                                                        GpuProfileScope::BlitToSwapchain,
                                                    );
                                                }
                                                blit_to_swapchain::record_blit_core(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    src_image,
                                                    src_extent,
                                                    dst_image,
                                                    dst_extent,
                                                );
                                                if let Some(profiler) = profiler {
                                                    profiler.end_scope(
                                                        ctx.device,
                                                        ctx.command_buffer,
                                                        slot,
                                                        GpuProfileScope::BlitToSwapchain,
                                                    );
                                                }
                                            })
                                        },
                                    );
                                }
                            } else {
                                // Fallback: blit raw G-buffer if lighting pass not ready
                                let src_image = pass.gbuffer0.handle;
                                let src_extent = pass.gbuffer0.extent;
                                let dst_image = frame.swapchain_image;
                                let dst_extent = frame.swapchain_extent;
                                let dep_handle = primary_ray_writes[1];
                                let swapchain_dep = graph.import_image_with_access(
                                    dst_image,
                                    dst_extent.width,
                                    dst_extent.height,
                                    frame.swapchain_format,
                                    vk::ImageUsageFlags::TRANSFER_DST,
                                    swapchain_access_from_layout(frame.swapchain_image_layout)?,
                                );
                                let slot = frame.frame_slot;
                                graph.add_pass(
                                    "blit_to_swapchain",
                                    QueueType::Graphics,
                                    |builder| {
                                        builder.read_as(dep_handle, AccessKind::TransferRead);
                                        builder.write_as(swapchain_dep, AccessKind::TransferWrite);
                                        builder.finish_as(swapchain_dep, AccessKind::Present);
                                        Box::new(move |ctx| {
                                            if let Some(profiler) = profiler {
                                                profiler.begin_scope(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                    GpuProfileScope::BlitToSwapchain,
                                                );
                                            }
                                            blit_to_swapchain::record_blit_core(
                                                ctx.device,
                                                ctx.command_buffer,
                                                src_image,
                                                src_extent,
                                                dst_image,
                                                dst_extent,
                                            );
                                            if let Some(profiler) = profiler {
                                                profiler.end_scope(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                    GpuProfileScope::BlitToSwapchain,
                                                );
                                            }
                                        })
                                    },
                                );
                            }
                        } else {
                            tracing::warn!(
                                "skipping VCT frame until primary ray pass is initialized"
                            );
                        }
                    }
                } else {
                    tracing::warn!("skipping UCVH render passes until GPU upload succeeds");
                }

                if !graph.has_final_access(AccessKind::Present) {
                    tracing::warn!(
                        "render graph produced no presentable output; clearing swapchain fallback"
                    );
                    add_swapchain_clear_present_pass(
                        &mut graph,
                        frame.swapchain_image,
                        frame.swapchain_extent,
                        frame.swapchain_format,
                        frame.swapchain_image_layout,
                    )?;
                }
                graph.compile()?;
                graph.execute(renderer.device(), frame.command_buffer, frame.frame_index);
                if vpt_accumulation_written {
                    self.vpt_accumulation_needs_init = false;
                }
                renderer.end_frame(frame)?;
            }
        }

        if let Some(input) = self.world.resource_mut::<InputState>() {
            input.clear_per_frame();
        }

        self.schedule
            .run_stage(Stage::ExecuteRender, &mut self.world)?;
        Ok(())
    }
}

impl Drop for RevolumetricApp {
    fn drop(&mut self) {
        // Destroy GPU passes before the renderer (which owns the device/allocator).
        if let Some(renderer) = &self.renderer {
            unsafe { renderer.device().device_wait_idle().ok() };
            if let Some(profiler) = self.gpu_profiler.take() {
                profiler.destroy(renderer.device());
            }
            if let Some(pass) = self.postprocess_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(pass) = self.vpt_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(pass) = self.restir_di_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(pass) = self.lighting_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(pass) = self.primary_ray_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(gpu) = self.ucvh_gpu.take() {
                gpu.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(ubo) = self.scene_ubo.take() {
                ubo.destroy(renderer.device(), renderer.allocator());
            }
        }
    }
}

impl ApplicationHandler for RevolumetricApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = match event_loop.create_window(self.window_descriptor.attributes()) {
            Ok(window) => window,
            Err(error) => {
                tracing::error!(%error, "failed to create main window");
                event_loop.exit();
                return;
            }
        };

        let renderer = match RenderDevice::new(&window) {
            Ok(renderer) => renderer,
            Err(error) => {
                tracing::error!(%error, "failed to initialize Vulkan bootstrap");
                event_loop.exit();
                return;
            }
        };

        tracing::info!(
            renderer = %renderer.backend_name(),
            physical_device = %renderer.physical_device_name(),
            graphics_queue_family = renderer.graphics_queue_family_index(),
            present_queue_family = renderer.present_queue_family_index(),
            swapchain_format = ?renderer.swapchain_format(),
            swapchain_extent = ?renderer.swapchain_extent(),
            swapchain_images = renderer.swapchain_image_count(),
            surface = ?renderer.surface(),
            "initialized renderer bootstrap"
        );

        let window_id = window.id();
        let lighting_settings_result = LightingSettings::from_env_report();
        for warning in &lighting_settings_result.warnings {
            tracing::warn!(
                variable = warning.variable,
                value = %warning.value,
                expected = warning.expected,
                "invalid lighting setting override; using default value"
            );
        }
        self.lighting_settings = lighting_settings_result.settings;
        let restir_di_settings_result = RestirDiSettings::from_env();
        for warning in &restir_di_settings_result.warnings {
            tracing::warn!(
                variable = warning.variable,
                value = %warning.value,
                expected = warning.expected,
                "invalid ReSTIR-DI setting override; using default value"
            );
        }
        self.restir_di_settings = restir_di_settings_result.settings;

        self.gpu_profiler = match GpuProfiler::new(
            renderer.device(),
            renderer
                .physical_device_properties()
                .limits
                .timestamp_period,
            renderer.graphics_queue_timestamp_valid_bits(),
            renderer.frame_slot_count(),
            GpuProfilerConfig::from_env(),
        ) {
            Ok(profiler) => profiler,
            Err(error) => {
                tracing::warn!(%error, "failed to initialize GPU profiler; continuing without profiling");
                None
            }
        };
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.window_id = Some(window_id);

        // Create Scene UBO
        if self.scene_ubo.is_none() {
            let renderer = self.renderer.as_ref().unwrap();
            match SceneUniformBuffer::new(
                renderer.device(),
                renderer.allocator(),
                renderer.swapchain_image_count(),
            ) {
                Ok(ubo) => {
                    tracing::info!(
                        frame_count = renderer.swapchain_image_count(),
                        "created scene UBO"
                    );
                    self.scene_ubo = Some(ubo);
                }
                Err(e) => tracing::error!(%e, "failed to create scene UBO"),
            }
        }

        // Generate UCVH sponza demo scene
        if self.ucvh.is_none() {
            let config = UcvhConfig::new(glam::UVec3::splat(128));
            let mut ucvh = Ucvh::new(config);
            let brick_count = generator::generate_sponza_scene(&mut ucvh);
            ucvh.rebuild_hierarchy();
            tracing::info!(
                bricks = brick_count,
                total_voxels = ucvh.pool.allocated_count() as u64 * 512,
                "generated sponza demo scene"
            );

            let renderer = self.renderer.as_ref().unwrap();
            match UcvhGpuResources::new(renderer.device(), renderer.allocator(), &ucvh) {
                Ok(gpu) => {
                    tracing::info!("created UCVH GPU resources");
                    self.ucvh_gpu = Some(gpu);
                }
                Err(e) => tracing::error!(%e, "failed to create UCVH GPU resources"),
            }
            self.ucvh = Some(ucvh);
        }

        // Initialize primary ray pass (requires UCVH GPU resources + Scene UBO)
        if self.primary_ray_pass.is_none() {
            if let (Some(ucvh_gpu), Some(scene_ubo_ref)) = (&self.ucvh_gpu, &self.scene_ubo) {
                let renderer = self.renderer.as_ref().unwrap();
                let extent = renderer.swapchain_extent();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/primary_ray.spv"));
                if spirv.is_empty() {
                    tracing::warn!("primary_ray.spv is empty 鈥?slangc may not be installed");
                } else {
                    match PrimaryRayPass::new(
                        renderer.device(),
                        renderer.allocator(),
                        extent.width,
                        extent.height,
                        spirv,
                        ucvh_gpu,
                        scene_ubo_ref,
                    ) {
                        Ok(pass) => {
                            tracing::info!(
                                width = extent.width,
                                height = extent.height,
                                "initialized primary ray pass"
                            );
                            self.primary_ray_pass = Some(pass);
                        }
                        Err(error) => {
                            tracing::error!(%error, "failed to create primary ray pass");
                        }
                    }
                }
            }
        }

        // Initialize lighting pass (requires primary ray pass + scene UBO)
        if self.lighting_pass.is_none() {
            if let (Some(primary), Some(ucvh_gpu), Some(scene_ubo_ref)) =
                (&self.primary_ray_pass, &self.ucvh_gpu, &self.scene_ubo)
            {
                let renderer = self.renderer.as_ref().unwrap();
                let extent = renderer.swapchain_extent();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/lighting.spv"));
                if spirv.is_empty() {
                    tracing::warn!("lighting.spv is empty 鈥?slangc may not be installed");
                } else {
                    match LightingPass::new(
                        renderer.device(),
                        renderer.allocator(),
                        extent.width,
                        extent.height,
                        spirv,
                        primary,
                        ucvh_gpu,
                        scene_ubo_ref,
                    ) {
                        Ok(pass) => {
                            tracing::info!(
                                width = extent.width,
                                height = extent.height,
                                "initialized lighting pass"
                            );
                            self.lighting_pass = Some(pass);
                        }
                        Err(error) => {
                            tracing::error!(%error, "failed to create lighting pass");
                        }
                    }
                }
            }
        }

        // Initialize postprocess pass (requires lighting HDR output + scene UBO)
        if self.postprocess_pass.is_none() {
            if let (Some(lighting), Some(scene_ubo_ref)) = (&self.lighting_pass, &self.scene_ubo) {
                let renderer = self.renderer.as_ref().unwrap();
                let extent = renderer.swapchain_extent();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/postprocess.spv"));
                if spirv.is_empty() {
                    tracing::warn!("postprocess.spv is empty 閳?slangc may not be installed");
                } else {
                    match PostprocessPass::new(
                        renderer.device(),
                        renderer.allocator(),
                        extent.width,
                        extent.height,
                        spirv,
                        &lighting.output_image,
                        scene_ubo_ref,
                    ) {
                        Ok(pass) => {
                            tracing::info!(
                                width = extent.width,
                                height = extent.height,
                                "initialized postprocess pass"
                            );
                            self.postprocess_pass = Some(pass);
                        }
                        Err(error) => {
                            tracing::error!(%error, "failed to create postprocess pass");
                        }
                    }
                }
            }
        }

        // Initialize VPT reference pass (requires UCVH GPU resources + Scene UBO)
        if self.vpt_pass.is_none() {
            if let (Some(ucvh_gpu), Some(scene_ubo_ref)) = (&self.ucvh_gpu, &self.scene_ubo) {
                let renderer = self.renderer.as_ref().unwrap();
                let extent = renderer.swapchain_extent();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt.spv"));
                if spirv.is_empty() {
                    tracing::warn!("vpt.spv is empty 閳?slangc may not be installed");
                } else {
                    match VptPass::new(
                        renderer.device(),
                        renderer.allocator(),
                        extent.width,
                        extent.height,
                        spirv,
                        ucvh_gpu,
                        scene_ubo_ref,
                    ) {
                        Ok(pass) => {
                            tracing::info!(
                                width = extent.width,
                                height = extent.height,
                                "initialized VPT reference pass"
                            );
                            self.vpt_pass = Some(pass);
                            self.vpt_accumulation_needs_init = true;
                        }
                        Err(error) => {
                            tracing::error!(%error, "failed to create VPT pass");
                        }
                    }
                }
            }
        }

        if self.restir_di_pass.is_none()
            && self.restir_di_vpt_enabled()
            && let (Some(ucvh), Some(scene_ubo_ref)) = (&self.ucvh, &self.scene_ubo)
        {
            let renderer = self.renderer.as_ref().unwrap();
            let extent = renderer.swapchain_extent();
            let initial_spirv =
                include_bytes!(concat!(env!("OUT_DIR"), "/shaders/restir_di_initial.spv"));
            let temporal_spirv =
                include_bytes!(concat!(env!("OUT_DIR"), "/shaders/restir_di_temporal.spv"));
            let spatial_spirv =
                include_bytes!(concat!(env!("OUT_DIR"), "/shaders/restir_di_spatial.spv"));
            if initial_spirv.is_empty() || temporal_spirv.is_empty() || spatial_spirv.is_empty() {
                tracing::warn!("ReSTIR-DI shaders are empty; slangc may not be installed");
            } else {
                let (sun_direction, sun_intensity) = self
                    .world
                    .resource::<DirectionalLight>()
                    .map_or(([0.5, 1.0, 0.25], 2.0), |light| {
                        (
                            light.direction.to_array(),
                            light.intensity.max_element().max(0.0),
                        )
                    });
                let direct_lights =
                    build_direct_lights_from_ucvh(ucvh, sun_direction, sun_intensity, 4096);
                match RestirDiPass::new(
                    renderer.device(),
                    renderer.allocator(),
                    RestirDiPassCreateInfo {
                        width: extent.width,
                        height: extent.height,
                        frame_count: scene_ubo_ref.frame_count(),
                        initial_spirv,
                        temporal_spirv,
                        spatial_spirv,
                        direct_lights: &direct_lights,
                    },
                ) {
                    Ok(pass) => {
                        tracing::info!(
                            width = extent.width,
                            height = extent.height,
                            direct_lights = direct_lights.len(),
                            "initialized ReSTIR-DI VPT pass skeleton"
                        );
                        self.restir_di_pass = Some(pass);
                    }
                    Err(error) => {
                        tracing::error!(%error, "failed to create ReSTIR-DI pass");
                    }
                }
            }
        }

        if self.postprocess_pass.is_none()
            && self.lighting_settings.render_mode == RenderMode::Vpt
            && let (Some(vpt), Some(scene_ubo_ref)) = (&self.vpt_pass, &self.scene_ubo)
        {
            let renderer = self.renderer.as_ref().unwrap();
            let extent = renderer.swapchain_extent();
            let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/postprocess.spv"));
            if spirv.is_empty() {
                tracing::warn!("postprocess.spv is empty; slangc may not be installed");
            } else {
                match PostprocessPass::new(
                    renderer.device(),
                    renderer.allocator(),
                    extent.width,
                    extent.height,
                    spirv,
                    &vpt.output_image,
                    scene_ubo_ref,
                ) {
                    Ok(pass) => {
                        tracing::info!(
                            width = extent.width,
                            height = extent.height,
                            "initialized postprocess pass from VPT output"
                        );
                        self.postprocess_pass = Some(pass);
                    }
                    Err(error) => {
                        tracing::error!(%error, "failed to create VPT postprocess pass");
                    }
                }
            }
        }

        if !self.initialized {
            if let Err(error) = self.schedule.run_stage(Stage::Startup, &mut self.world) {
                tracing::error!(%error, "startup stage failed");
                event_loop.exit();
                return;
            }
            self.initialized = true;
        }

        tracing::info!(?window_id, "window created");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                // Skip rendering when minimized (zero-size window)
                if let Some(window) = &self.window {
                    let size = window.inner_size();
                    if size.width == 0 || size.height == 0 {
                        return;
                    }
                }
                if let Err(error) = self.tick_frame() {
                    tracing::error!(%error, "frame execution failed");
                    event_loop.exit();
                    return;
                }
                self.rendered_frames += 1;
                if let Some(limit) = self.exit_after_frames
                    && self.rendered_frames >= limit
                {
                    tracing::info!(
                        rendered_frames = self.rendered_frames,
                        "exit-after-frames limit reached"
                    );
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    return; // minimized 鈥?skip resize
                }
                if let Some(renderer) = self.renderer.as_mut() {
                    if let Err(error) = renderer.handle_resize(size.width, size.height) {
                        tracing::error!(%error, "failed to recreate swapchain after resize");
                        event_loop.exit();
                        return;
                    }
                }
                if let Err(error) = self.resize_render_passes(size.width, size.height) {
                    tracing::error!(%error, "failed to resize render passes");
                    event_loop.exit();
                    return;
                }
                tracing::debug!(width = size.width, height = size.height, "window resized");
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.repeat {
                    return; // ignore key repeat
                }
                let pressed = event.state == winit::event::ElementState::Pressed;
                let value = if pressed { 1.0_f32 } else { -1.0 };

                if let PhysicalKey::Code(key) = event.physical_key {
                    if let Some(input) = self.world.resource_mut::<InputState>() {
                        match key {
                            KeyCode::KeyW => input.move_forward += value,
                            KeyCode::KeyS => input.move_forward -= value,
                            KeyCode::KeyD => input.move_right += value,
                            KeyCode::KeyA => input.move_right -= value,
                            KeyCode::Space => input.move_up += value,
                            KeyCode::ShiftLeft => input.move_up -= value,
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Right {
                    let pressed = state == winit::event::ElementState::Pressed;
                    if let Some(input) = self.world.resource_mut::<InputState>() {
                        input.right_mouse_held = pressed;
                    }
                    // Grab/release cursor for FPS camera
                    if let Some(window) = &self.window {
                        if pressed {
                            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                            window.set_cursor_visible(false);
                        } else {
                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                            window.set_cursor_visible(true);
                            self.last_cursor_pos = None;
                        }
                    }
                    if !pressed {
                        self.last_cursor_pos = None;
                    }
                }
            }
            WindowEvent::CursorMoved { .. } => {
                // Mouse deltas handled via DeviceEvent::MouseMotion for reliable FPS camera
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 120.0,
                };
                if let Some(input) = self.world.resource_mut::<InputState>() {
                    input.scroll_delta += scroll;
                }
            }
            WindowEvent::Focused(false) => {
                if let Some(input) = self.world.resource_mut::<InputState>() {
                    input.reset_axes();
                    input.right_mouse_held = false;
                }
                self.last_cursor_pos = None;
                if let Some(window) = &self.window {
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if let Some(input) = self.world.resource_mut::<InputState>() {
                if input.right_mouse_held {
                    input.mouse_dx += delta.0 as f32;
                    input.mouse_dy += delta.1 as f32;
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn init_tracing() {
    let _ = fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .try_init();
}
