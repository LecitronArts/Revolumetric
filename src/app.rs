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
use crate::render::area_restir::{AreaRestirDebugView, AreaRestirSettings};
use crate::render::camera::compute_pixel_to_ray;
use crate::render::device::RenderDevice;
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler, GpuProfilerConfig};
use crate::render::graph::RenderGraph;
use crate::render::passes::area_restir::{AreaRestirPass, AreaRestirPassCreateInfo};
use crate::render::passes::blit_to_swapchain;
use crate::render::passes::postprocess::PostprocessPass;
use crate::render::passes::restir_di::{RestirDiPass, RestirDiPassCreateInfo};
use crate::render::passes::vpt::VptPass;
use crate::render::passes::vpt_surface::VptSurfacePass;
use crate::render::passes::vpt_temporal::{
    VptTemporalPass, VptTemporalPassCreateInfo, VptTemporalPassResizeInfo,
};
use crate::render::resource::{AccessKind, QueueType};
use crate::render::restir_di::{RestirDiSettings, build_direct_lights_from_ucvh};
use crate::render::scene_ubo::{
    LightingSettings, SceneUniformBuffer, SceneUniformInputs, VptDebugView, build_scene_uniforms,
};
use crate::render::vpt_history::{
    GpuVptHistoryUniforms, VPT_HISTORY_FLAG_CAMERA_CUT, VPT_HISTORY_FLAG_RESIZE,
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

fn area_restir_debug_to_vpt_debug_view(debug_view: AreaRestirDebugView) -> Option<VptDebugView> {
    match debug_view {
        AreaRestirDebugView::Off => None,
        AreaRestirDebugView::Subpixel => Some(VptDebugView::AreaSubpixel),
        AreaRestirDebugView::Lens => Some(VptDebugView::AreaLens),
        AreaRestirDebugView::Weight => Some(VptDebugView::AreaWeight),
        AreaRestirDebugView::HistoryValid => Some(VptDebugView::AreaHistoryValid),
        AreaRestirDebugView::Rejection => Some(VptDebugView::AreaRejection),
        AreaRestirDebugView::Jacobian => Some(VptDebugView::AreaJacobian),
    }
}

fn area_restir_effective_settings(
    settings: AreaRestirSettings,
    history_initialized: bool,
) -> AreaRestirSettings {
    let mut settings = settings;
    if !history_initialized {
        settings.temporal_enabled = false;
    }
    settings.spatial_enabled = settings.temporal_enabled && settings.spatial_enabled;
    settings
}

fn restir_di_effective_settings(
    settings: RestirDiSettings,
    history_initialized: bool,
) -> RestirDiSettings {
    let mut settings = settings;
    if !history_initialized {
        settings.temporal_enabled = false;
    }
    settings.spatial_enabled = settings.temporal_enabled && settings.spatial_enabled;
    settings
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

fn compute_view_proj(
    camera_pos: glam::Vec3,
    camera_forward: glam::Vec3,
    camera_up: glam::Vec3,
    fov_y: f32,
    width: u32,
    height: u32,
) -> glam::Mat4 {
    let forward = camera_forward.normalize();
    let right = camera_up.cross(forward).normalize();
    let up = forward.cross(right);
    let view = glam::Mat4::from_cols(
        glam::Vec4::new(right.x, up.x, -forward.x, 0.0),
        glam::Vec4::new(right.y, up.y, -forward.y, 0.0),
        glam::Vec4::new(right.z, up.z, -forward.z, 0.0),
        glam::Vec4::new(
            -right.dot(camera_pos),
            -up.dot(camera_pos),
            forward.dot(camera_pos),
            1.0,
        ),
    );
    let projection =
        glam::Mat4::perspective_rh(fov_y, width as f32 / height as f32, 0.01, 10_000.0);
    projection * view
}

struct RevolumetricApp {
    world: World,
    schedule: Schedule,
    renderer: Option<RenderDevice>,
    gpu_profiler: Option<GpuProfiler>,
    postprocess_pass: Option<PostprocessPass>,
    vpt_surface_pass: Option<VptSurfacePass>,
    vpt_pass: Option<VptPass>,
    vpt_temporal_pass: Option<VptTemporalPass>,
    area_restir_pass: Option<AreaRestirPass>,
    restir_di_pass: Option<RestirDiPass>,
    ucvh: Option<Ucvh>,
    ucvh_gpu: Option<UcvhGpuResources>,
    ucvh_uploaded: bool,
    scene_ubo: Option<SceneUniformBuffer>,
    lighting_settings: LightingSettings,
    area_restir_settings: AreaRestirSettings,
    restir_di_settings: RestirDiSettings,
    vpt_sample_index: u32,
    last_vpt_camera_key: Option<[u32; 15]>,
    vpt_accumulation_needs_init: bool,
    vpt_temporal_history_initialized: bool,
    postprocess_output_initialized: bool,
    area_restir_history_initialized: bool,
    restir_di_history_initialized: bool,
    previous_vpt_view_proj: Option<glam::Mat4>,
    previous_vpt_resolution: Option<[u32; 2]>,
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
            postprocess_pass: None,
            vpt_surface_pass: None,
            vpt_pass: None,
            vpt_temporal_pass: None,
            area_restir_pass: None,
            restir_di_pass: None,
            ucvh: None,
            ucvh_gpu: None,
            ucvh_uploaded: false,
            scene_ubo: None,
            lighting_settings: LightingSettings::default(),
            area_restir_settings: AreaRestirSettings::default(),
            restir_di_settings: RestirDiSettings::default(),
            vpt_sample_index: 0,
            last_vpt_camera_key: None,
            vpt_accumulation_needs_init: true,
            vpt_temporal_history_initialized: false,
            postprocess_output_initialized: false,
            area_restir_history_initialized: false,
            restir_di_history_initialized: false,
            previous_vpt_view_proj: None,
            previous_vpt_resolution: None,
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
        self.restir_di_settings.enabled
    }

    fn area_restir_vpt_enabled(&self) -> bool {
        self.area_restir_settings.enabled
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
        let restir_di_enabled = self.restir_di_vpt_enabled();
        let area_restir_enabled = self.area_restir_vpt_enabled();

        if let (Some(vpt), Some(scene_ubo), Some(ucvh_gpu)) =
            (&mut self.vpt_pass, &self.scene_ubo, &self.ucvh_gpu)
        {
            vpt.resize_images(&device, allocator, width, height, scene_ubo, ucvh_gpu)
                .context("failed to resize VPT images")?;
            self.vpt_sample_index = 0;
            self.last_vpt_camera_key = None;
            self.vpt_accumulation_needs_init = true;
            self.previous_vpt_view_proj = None;
            self.previous_vpt_resolution = None;
        }
        if let (Some(vpt_surface), Some(scene_ubo), Some(ucvh_gpu)) =
            (&mut self.vpt_surface_pass, &self.scene_ubo, &self.ucvh_gpu)
        {
            vpt_surface
                .resize_images(&device, allocator, width, height, scene_ubo, ucvh_gpu)
                .context("failed to resize VPT surface images")?;
            if restir_di_enabled && let Some(restir_di) = &self.restir_di_pass {
                restir_di.update_surface_descriptors(&device, vpt_surface);
            }
            if area_restir_enabled && let Some(area_restir) = &self.area_restir_pass {
                area_restir.update_surface_descriptors(&device, vpt_surface);
            }
        }
        if restir_di_enabled && let Some(restir_di) = &mut self.restir_di_pass {
            restir_di
                .resize_buffers(&device, allocator, width, height)
                .context("failed to resize ReSTIR-DI buffers")?;
            self.restir_di_history_initialized = false;
        }
        if area_restir_enabled && let Some(area_restir) = &mut self.area_restir_pass {
            area_restir
                .resize_buffers(&device, allocator, width, height)
                .context("failed to resize Area ReSTIR buffers")?;
            if let Some(scene_ubo) = &self.scene_ubo {
                area_restir.update_scene_descriptors(&device, scene_ubo);
            }
            if let Some(ucvh_gpu) = &self.ucvh_gpu {
                area_restir.update_ucvh_descriptors(&device, ucvh_gpu);
            }
            if let (Some(vpt), Some(scene_ubo)) = (&self.vpt_pass, &self.scene_ubo) {
                for slot in 0..scene_ubo.frame_count() {
                    let (area_uniform_buffer, _, _) = area_restir.uniform_buffer(slot);
                    let (area_selected_current_buffer, _, _) =
                        area_restir.selected_current_buffer(slot);
                    vpt.update_area_restir_descriptors(
                        &device,
                        slot,
                        area_uniform_buffer,
                        area_selected_current_buffer,
                    );
                }
            }
            self.area_restir_history_initialized = false;
        }
        if let (Some(vpt_temporal), Some(vpt), Some(vpt_surface), Some(scene_ubo)) = (
            &mut self.vpt_temporal_pass,
            &self.vpt_pass,
            &self.vpt_surface_pass,
            &self.scene_ubo,
        ) {
            vpt_temporal
                .resize_images(
                    &device,
                    allocator,
                    VptTemporalPassResizeInfo {
                        width,
                        height,
                        scene_ubo,
                        vpt,
                        vpt_surface,
                    },
                )
                .context("failed to resize VPT temporal images")?;
            self.vpt_temporal_history_initialized = false;
        }
        if let (Some(postprocess), Some(vpt_temporal), Some(scene_ubo)) = (
            &mut self.postprocess_pass,
            &self.vpt_temporal_pass,
            &self.scene_ubo,
        ) {
            postprocess
                .resize_images(
                    &device,
                    allocator,
                    width,
                    height,
                    &vpt_temporal.accumulated_radiance,
                    scene_ubo,
                )
                .context("failed to resize VPT postprocess images")?;
            self.postprocess_output_initialized = false;
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
        let area_restir_enabled = self.area_restir_vpt_enabled();
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
                let mut restir_di_selected_written = false;
                let mut area_restir_selected_written = false;
                let mut current_vpt_view_proj: Option<glam::Mat4> = None;

                if ucvh_ready {
                    let (cam_pos, cam_forward, cam_up, fov_y, aperture_radius, focal_distance) = {
                        let rig = self.world.resource::<CameraRig>();
                        match rig {
                            Some(rig) => (
                                rig.camera.position,
                                rig.camera.forward,
                                rig.camera.up,
                                rig.camera.fov_y_radians,
                                rig.camera.aperture_radius,
                                rig.camera.focal_distance,
                            ),
                            None => (
                                glam::Vec3::new(64.0, 80.0, -40.0),
                                glam::Vec3::Z,
                                glam::Vec3::Y,
                                std::f32::consts::FRAC_PI_4,
                                0.0,
                                128.0,
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
                        aperture_radius.to_bits(),
                        focal_distance.to_bits(),
                        frame.swapchain_extent.width,
                        frame.swapchain_extent.height,
                        self.lighting_settings.vpt_max_bounces,
                    ];
                    if self.last_vpt_camera_key == Some(camera_key) {
                        self.vpt_sample_index = self.vpt_sample_index.saturating_add(1);
                    } else {
                        self.vpt_sample_index = 0;
                        self.last_vpt_camera_key = Some(camera_key);
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
                        camera_right: cam_up.cross(cam_forward).normalize(),
                        camera_up: cam_up.normalize(),
                        camera_forward: cam_forward.normalize(),
                        aperture_radius,
                        focal_distance,
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

                    let current_view_proj = compute_view_proj(
                        cam_pos,
                        cam_forward,
                        cam_up,
                        fov_y,
                        frame.swapchain_extent.width,
                        frame.swapchain_extent.height,
                    );
                    current_vpt_view_proj = Some(current_view_proj);
                    let previous_view_proj =
                        self.previous_vpt_view_proj.unwrap_or(current_view_proj);
                    let previous_resolution = self
                        .previous_vpt_resolution
                        .unwrap_or([frame.swapchain_extent.width, frame.swapchain_extent.height]);
                    let history_flags = if self.previous_vpt_view_proj.is_none() {
                        VPT_HISTORY_FLAG_CAMERA_CUT
                    } else if self.previous_vpt_resolution.is_none()
                        || previous_resolution
                            != [frame.swapchain_extent.width, frame.swapchain_extent.height]
                    {
                        VPT_HISTORY_FLAG_RESIZE
                    } else {
                        0
                    };
                    let history_uniforms = GpuVptHistoryUniforms {
                        current_view_proj: current_view_proj.transpose().to_cols_array_2d(),
                        previous_view_proj: previous_view_proj.transpose().to_cols_array_2d(),
                        current_resolution: [
                            frame.swapchain_extent.width,
                            frame.swapchain_extent.height,
                        ],
                        previous_resolution,
                        current_jitter: [0.0, 0.0],
                        previous_jitter: [0.0, 0.0],
                        frame_index: frame.frame_index as u32,
                        history_reset_generation: 0,
                        flags: history_flags,
                        _pad0: 0,
                    };
                    if let Some(vpt_surface) = &self.vpt_surface_pass {
                        vpt_surface.update_history_uniforms(frame.frame_slot, &history_uniforms);
                    }

                    if let (Some(vpt_surface), Some(vpt), Some(vpt_temporal), Some(postprocess)) = (
                        &self.vpt_surface_pass,
                        &self.vpt_pass,
                        &self.vpt_temporal_pass,
                        &self.postprocess_pass,
                    ) {
                        let surface_position_resource = graph.import_image_with_access(
                            vpt_surface.surface_position_depth.handle,
                            vpt_surface.surface_position_depth.extent.width,
                            vpt_surface.surface_position_depth.extent.height,
                            vk::Format::R32G32B32A32_SFLOAT,
                            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            AccessKind::Undefined,
                        );
                        let surface_normal_resource = graph.import_image_with_access(
                            vpt_surface.surface_normal_roughness.handle,
                            vpt_surface.surface_normal_roughness.extent.width,
                            vpt_surface.surface_normal_roughness.extent.height,
                            vk::Format::R32G32B32A32_SFLOAT,
                            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            AccessKind::Undefined,
                        );
                        let surface_albedo_resource = graph.import_image_with_access(
                            vpt_surface.surface_albedo_material.handle,
                            vpt_surface.surface_albedo_material.extent.width,
                            vpt_surface.surface_albedo_material.extent.height,
                            vk::Format::R32G32B32A32_SFLOAT,
                            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            AccessKind::Undefined,
                        );
                        let motion_history_resource = graph.import_image_with_access(
                            vpt_surface.motion_history.handle,
                            vpt_surface.motion_history.extent.width,
                            vpt_surface.motion_history.extent.height,
                            vk::Format::R32G32B32A32_SFLOAT,
                            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            AccessKind::Undefined,
                        );
                        let previous_surface_access = if self.vpt_temporal_history_initialized {
                            AccessKind::TransferWrite
                        } else {
                            AccessKind::Undefined
                        };
                        let previous_surface_position_resource = graph.import_image_with_access(
                            vpt_surface.previous_surface_position_depth.handle,
                            vpt_surface.previous_surface_position_depth.extent.width,
                            vpt_surface.previous_surface_position_depth.extent.height,
                            vk::Format::R32G32B32A32_SFLOAT,
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                            previous_surface_access,
                        );
                        let previous_surface_normal_resource = graph.import_image_with_access(
                            vpt_surface.previous_surface_normal_roughness.handle,
                            vpt_surface.previous_surface_normal_roughness.extent.width,
                            vpt_surface.previous_surface_normal_roughness.extent.height,
                            vk::Format::R32G32B32A32_SFLOAT,
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                            previous_surface_access,
                        );
                        let previous_surface_albedo_resource = graph.import_image_with_access(
                            vpt_surface.previous_surface_albedo_material.handle,
                            vpt_surface.previous_surface_albedo_material.extent.width,
                            vpt_surface.previous_surface_albedo_material.extent.height,
                            vk::Format::R32G32B32A32_SFLOAT,
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                            previous_surface_access,
                        );
                        let slot = frame.frame_slot;
                        let surface_writes =
                            graph.add_pass("vpt_surface", QueueType::Compute, |builder| {
                                builder.write_as(
                                    surface_position_resource,
                                    AccessKind::ComputeShaderWrite,
                                );
                                builder.write_as(
                                    surface_normal_resource,
                                    AccessKind::ComputeShaderWrite,
                                );
                                builder.write_as(
                                    surface_albedo_resource,
                                    AccessKind::ComputeShaderWrite,
                                );
                                builder.write_as(
                                    motion_history_resource,
                                    AccessKind::ComputeShaderWrite,
                                );
                                Box::new(move |ctx| {
                                    if let Some(profiler) = profiler {
                                        profiler.begin_scope(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                            GpuProfileScope::VptSurface,
                                        );
                                    }
                                    vpt_surface.record(ctx.device, ctx.command_buffer, slot);
                                    if let Some(profiler) = profiler {
                                        profiler.end_scope(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                            GpuProfileScope::VptSurface,
                                        );
                                    }
                                })
                            });
                        let surface_images = [
                            vpt_surface.surface_position_depth.handle,
                            vpt_surface.surface_normal_roughness.handle,
                            vpt_surface.surface_albedo_material.handle,
                            vpt_surface.motion_history.handle,
                        ];
                        for (&resource, &image) in surface_writes.iter().zip(surface_images.iter())
                        {
                            graph.bind_image(resource, image);
                        }

                        let mut vpt_restir_reads = None;
                        if restir_di_enabled && let Some(restir_di) = &self.restir_di_pass {
                            let restir_di_settings = restir_di_effective_settings(
                                self.restir_di_settings,
                                self.restir_di_history_initialized,
                            );
                            restir_di.update_uniforms(
                                frame.frame_slot,
                                restir_di_settings,
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
                            let (
                                selected_current_buffer,
                                selected_current_size,
                                selected_current_usage,
                            ) = restir_di.selected_current_buffer(frame.frame_slot);
                            let (
                                selected_history_buffer,
                                selected_history_size,
                                selected_history_usage,
                            ) = restir_di.selected_history_buffer(frame.frame_slot);
                            restir_di.update_frame_descriptors(
                                renderer.device(),
                                frame.frame_slot,
                                selected_history_buffer,
                                selected_current_buffer,
                                restir_di_settings.spatial_enabled,
                            );

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
                            let selected_current_resource = graph.import_buffer_with_access(
                                selected_current_buffer.handle,
                                selected_current_size,
                                selected_current_usage,
                                AccessKind::Undefined,
                            );
                            let selected_history_resource = graph.import_buffer_with_access(
                                selected_history_buffer.handle,
                                selected_history_size,
                                selected_history_usage,
                                if self.restir_di_history_initialized {
                                    AccessKind::ComputeShaderWrite
                                } else {
                                    AccessKind::Undefined
                                },
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
                                    builder
                                        .read_as(uniform_resource, AccessKind::ComputeShaderRead);
                                    builder
                                        .read_as(surface_writes[0], AccessKind::ComputeShaderRead);
                                    builder
                                        .read_as(surface_writes[1], AccessKind::ComputeShaderRead);
                                    builder
                                        .read_as(surface_writes[2], AccessKind::ComputeShaderRead);
                                    builder.read_as(
                                        direct_light_resource,
                                        AccessKind::ComputeShaderRead,
                                    );
                                    builder
                                        .write_as(initial_resource, AccessKind::ComputeShaderWrite);
                                    Box::new(move |ctx| {
                                        if let Some(profiler) = profiler {
                                            profiler.begin_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::RestirDiInitial,
                                            );
                                        }
                                        restir_di.record_initial(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                        );
                                        if let Some(profiler) = profiler {
                                            profiler.end_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::RestirDiInitial,
                                            );
                                        }
                                    })
                                },
                            );
                            let initial_dep = initial_writes[0];
                            let temporal_writes = graph.add_pass(
                                "restir_di_temporal",
                                QueueType::Compute,
                                |builder| {
                                    builder
                                        .read_as(uniform_resource, AccessKind::ComputeShaderRead);
                                    builder
                                        .read_as(surface_writes[0], AccessKind::ComputeShaderRead);
                                    builder
                                        .read_as(surface_writes[1], AccessKind::ComputeShaderRead);
                                    builder
                                        .read_as(surface_writes[2], AccessKind::ComputeShaderRead);
                                    builder
                                        .read_as(surface_writes[3], AccessKind::ComputeShaderRead);
                                    builder.read_as(
                                        previous_surface_position_resource,
                                        AccessKind::ComputeShaderRead,
                                    );
                                    builder.read_as(
                                        previous_surface_normal_resource,
                                        AccessKind::ComputeShaderRead,
                                    );
                                    builder.read_as(
                                        previous_surface_albedo_resource,
                                        AccessKind::ComputeShaderRead,
                                    );
                                    builder.read_as(initial_dep, AccessKind::ComputeShaderRead);
                                    builder.read_as(
                                        selected_history_resource,
                                        AccessKind::ComputeShaderRead,
                                    );
                                    let temporal_output_resource =
                                        if restir_di_settings.spatial_enabled {
                                            temporal_resource
                                        } else {
                                            selected_current_resource
                                        };
                                    builder.write_as(
                                        temporal_output_resource,
                                        AccessKind::ComputeShaderWrite,
                                    );
                                    Box::new(move |ctx| {
                                        if let Some(profiler) = profiler {
                                            profiler.begin_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::RestirDiTemporal,
                                            );
                                        }
                                        restir_di.record_temporal(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                        );
                                        if let Some(profiler) = profiler {
                                            profiler.end_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::RestirDiTemporal,
                                            );
                                        }
                                    })
                                },
                            );
                            let temporal_dep = temporal_writes[0];
                            let selected_current_dep = if restir_di_settings.spatial_enabled {
                                let spatial_writes = graph.add_pass(
                                    "restir_di_spatial",
                                    QueueType::Compute,
                                    |builder| {
                                        builder.read_as(
                                            uniform_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            surface_writes[0],
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            surface_writes[1],
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            surface_writes[2],
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder
                                            .read_as(temporal_dep, AccessKind::ComputeShaderRead);
                                        builder.write_as(
                                            selected_current_resource,
                                            AccessKind::ComputeShaderWrite,
                                        );
                                        Box::new(move |ctx| {
                                            if let Some(profiler) = profiler {
                                                profiler.begin_scope(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                    GpuProfileScope::RestirDiSpatial,
                                                );
                                            }
                                            restir_di.record_spatial(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                            );
                                            if let Some(profiler) = profiler {
                                                profiler.end_scope(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                    GpuProfileScope::RestirDiSpatial,
                                                );
                                            }
                                        })
                                    },
                                );
                                spatial_writes[0]
                            } else {
                                temporal_dep
                            };
                            restir_di_selected_written = true;
                            vpt.update_restir_di_descriptors(
                                renderer.device(),
                                frame.frame_slot,
                                uniform_buffer,
                                selected_current_buffer,
                            );
                            vpt_restir_reads = Some((uniform_resource, selected_current_dep));
                        }

                        let mut vpt_area_restir_reads = None;
                        if area_restir_enabled && let Some(area_restir) = &self.area_restir_pass {
                            let area_restir_settings = area_restir_effective_settings(
                                self.area_restir_settings,
                                self.area_restir_history_initialized,
                            );
                            let area_temporal_active = area_restir_settings.temporal_enabled;
                            let area_spatial_active =
                                area_temporal_active && area_restir_settings.spatial_enabled;
                            area_restir.update_uniforms(
                                frame.frame_slot,
                                area_restir_settings,
                                frame.frame_index,
                            );

                            let (area_uniform_buffer, area_uniform_size, area_uniform_usage) =
                                area_restir.uniform_buffer(frame.frame_slot);
                            let (area_initial_buffer, area_initial_size, area_initial_usage) =
                                area_restir.initial_buffer();
                            let (area_temporal_buffer, area_temporal_size, area_temporal_usage) =
                                area_restir.temporal_buffer();
                            let (
                                area_selected_current_buffer,
                                area_selected_current_size,
                                area_selected_current_usage,
                            ) = area_restir.selected_current_buffer(frame.frame_slot);
                            let (
                                area_selected_history_buffer,
                                area_selected_history_size,
                                area_selected_history_usage,
                            ) = area_restir.selected_history_buffer(frame.frame_slot);
                            area_restir.update_frame_descriptors(
                                renderer.device(),
                                frame.frame_slot,
                                area_selected_history_buffer,
                                area_selected_current_buffer,
                                area_temporal_active,
                                area_spatial_active,
                            );

                            let area_uniform_resource = graph.import_buffer_with_access(
                                area_uniform_buffer.handle,
                                area_uniform_size,
                                area_uniform_usage,
                                AccessKind::ComputeShaderRead,
                            );
                            let area_initial_resource = graph.import_buffer_with_access(
                                area_initial_buffer.handle,
                                area_initial_size,
                                area_initial_usage,
                                AccessKind::Undefined,
                            );
                            let area_temporal_resource = graph.import_buffer_with_access(
                                area_temporal_buffer.handle,
                                area_temporal_size,
                                area_temporal_usage,
                                AccessKind::Undefined,
                            );
                            let area_selected_current_resource = graph.import_buffer_with_access(
                                area_selected_current_buffer.handle,
                                area_selected_current_size,
                                area_selected_current_usage,
                                AccessKind::Undefined,
                            );
                            let area_selected_history_resource = graph.import_buffer_with_access(
                                area_selected_history_buffer.handle,
                                area_selected_history_size,
                                area_selected_history_usage,
                                if self.area_restir_history_initialized {
                                    AccessKind::ComputeShaderWrite
                                } else {
                                    AccessKind::Undefined
                                },
                            );
                            let area_debug_resource = graph.import_image_with_access(
                                area_restir.debug_image.handle,
                                area_restir.debug_image.extent.width,
                                area_restir.debug_image.extent.height,
                                vk::Format::R16G16B16A16_SFLOAT,
                                vk::ImageUsageFlags::STORAGE
                                    | vk::ImageUsageFlags::TRANSFER_SRC
                                    | vk::ImageUsageFlags::TRANSFER_DST,
                                AccessKind::Undefined,
                            );

                            let slot = frame.frame_slot;
                            let area_initial_writes = graph.add_pass(
                                "area_restir_initial",
                                QueueType::Compute,
                                |builder| {
                                    builder.read_as(
                                        area_uniform_resource,
                                        AccessKind::ComputeShaderRead,
                                    );
                                    builder
                                        .read_as(surface_writes[0], AccessKind::ComputeShaderRead);
                                    builder
                                        .read_as(surface_writes[1], AccessKind::ComputeShaderRead);
                                    builder
                                        .read_as(surface_writes[2], AccessKind::ComputeShaderRead);
                                    let area_initial_output_resource = if area_temporal_active {
                                        area_initial_resource
                                    } else {
                                        area_selected_current_resource
                                    };
                                    builder.write_as(
                                        area_initial_output_resource,
                                        AccessKind::ComputeShaderWrite,
                                    );
                                    builder.write_as(
                                        area_debug_resource,
                                        AccessKind::ComputeShaderWrite,
                                    );
                                    Box::new(move |ctx| {
                                        if let Some(profiler) = profiler {
                                            profiler.begin_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::AreaRestirInitial,
                                            );
                                        }
                                        area_restir.record_initial(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                        );
                                        if let Some(profiler) = profiler {
                                            profiler.end_scope(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                                GpuProfileScope::AreaRestirInitial,
                                            );
                                        }
                                    })
                                },
                            );
                            let area_initial_dep = area_initial_writes[0];
                            let area_debug_dep = area_initial_writes[1];
                            let area_temporal_dep = if area_temporal_active {
                                let area_temporal_writes = graph.add_pass(
                                    "area_restir_temporal",
                                    QueueType::Compute,
                                    |builder| {
                                        builder.read_as(
                                            area_uniform_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            area_initial_dep,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            area_selected_history_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            surface_writes[0],
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            surface_writes[1],
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            surface_writes[2],
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            surface_writes[3],
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            previous_surface_position_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            previous_surface_normal_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        builder.read_as(
                                            previous_surface_albedo_resource,
                                            AccessKind::ComputeShaderRead,
                                        );
                                        let area_temporal_output_resource = if area_spatial_active {
                                            area_temporal_resource
                                        } else {
                                            area_selected_current_resource
                                        };
                                        builder.write_as(
                                            area_temporal_output_resource,
                                            AccessKind::ComputeShaderWrite,
                                        );
                                        Box::new(move |ctx| {
                                            if let Some(profiler) = profiler {
                                                profiler.begin_scope(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                    GpuProfileScope::AreaRestirTemporal,
                                                );
                                            }
                                            area_restir.record_temporal(
                                                ctx.device,
                                                ctx.command_buffer,
                                                slot,
                                            );
                                            if let Some(profiler) = profiler {
                                                profiler.end_scope(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                    GpuProfileScope::AreaRestirTemporal,
                                                );
                                            }
                                        })
                                    },
                                );
                                area_temporal_writes[0]
                            } else {
                                area_initial_dep
                            };
                            let (area_selected_reservoir_resource, area_final_debug_dep) =
                                if area_spatial_active {
                                    let area_spatial_writes = graph.add_pass(
                                        "area_restir_spatial",
                                        QueueType::Compute,
                                        |builder| {
                                            builder.read_as(
                                                area_uniform_resource,
                                                AccessKind::ComputeShaderRead,
                                            );
                                            builder.read_as(
                                                area_temporal_dep,
                                                AccessKind::ComputeShaderRead,
                                            );
                                            builder.read_as(
                                                surface_writes[0],
                                                AccessKind::ComputeShaderRead,
                                            );
                                            builder.read_as(
                                                surface_writes[1],
                                                AccessKind::ComputeShaderRead,
                                            );
                                            builder.read_as(
                                                surface_writes[2],
                                                AccessKind::ComputeShaderRead,
                                            );
                                            builder.write_as(
                                                area_selected_current_resource,
                                                AccessKind::ComputeShaderWrite,
                                            );
                                            builder.write_as(
                                                area_debug_dep,
                                                AccessKind::ComputeShaderWrite,
                                            );
                                            Box::new(move |ctx| {
                                                if let Some(profiler) = profiler {
                                                    profiler.begin_scope(
                                                        ctx.device,
                                                        ctx.command_buffer,
                                                        slot,
                                                        GpuProfileScope::AreaRestirSpatial,
                                                    );
                                                }
                                                area_restir.record_spatial(
                                                    ctx.device,
                                                    ctx.command_buffer,
                                                    slot,
                                                );
                                                if let Some(profiler) = profiler {
                                                    profiler.end_scope(
                                                        ctx.device,
                                                        ctx.command_buffer,
                                                        slot,
                                                        GpuProfileScope::AreaRestirSpatial,
                                                    );
                                                }
                                            })
                                        },
                                    );
                                    (area_spatial_writes[0], area_spatial_writes[1])
                                } else {
                                    if area_temporal_active {
                                        (area_temporal_dep, area_debug_dep)
                                    } else {
                                        (area_initial_dep, area_debug_dep)
                                    }
                                };
                            let _ = area_final_debug_dep;
                            area_restir_selected_written = true;
                            vpt.update_area_restir_descriptors(
                                renderer.device(),
                                frame.frame_slot,
                                area_uniform_buffer,
                                area_selected_current_buffer,
                            );
                            vpt_area_restir_reads =
                                Some((area_uniform_resource, area_selected_reservoir_resource));
                        }

                        postprocess.update_input_image(
                            renderer.device(),
                            &vpt_temporal.accumulated_radiance,
                            frame.frame_slot,
                        );
                        let noisy_initial_access = if self.vpt_accumulation_needs_init {
                            AccessKind::Undefined
                        } else {
                            AccessKind::ComputeShaderRead
                        };
                        let noisy_radiance_resource = graph.import_image_with_access(
                            vpt.noisy_radiance_image.handle,
                            vpt.noisy_radiance_image.extent.width,
                            vpt.noisy_radiance_image.extent.height,
                            vk::Format::R16G16B16A16_SFLOAT,
                            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            noisy_initial_access,
                        );
                        let noisy_moments_resource = graph.import_image_with_access(
                            vpt.noisy_moments_image.handle,
                            vpt.noisy_moments_image.extent.width,
                            vpt.noisy_moments_image.extent.height,
                            vk::Format::R16G16B16A16_SFLOAT,
                            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            noisy_initial_access,
                        );
                        vpt_accumulation_written = true;
                        let vpt_writes = graph.add_pass("vpt", QueueType::Compute, |builder| {
                            builder
                                .write_as(noisy_radiance_resource, AccessKind::ComputeShaderWrite);
                            builder
                                .write_as(noisy_moments_resource, AccessKind::ComputeShaderWrite);
                            if let Some((restir_uniform_resource, restir_reservoir_resource)) =
                                vpt_restir_reads
                            {
                                builder.read_as(
                                    restir_uniform_resource,
                                    AccessKind::ComputeShaderRead,
                                );
                                builder.read_as(
                                    restir_reservoir_resource,
                                    AccessKind::ComputeShaderRead,
                                );
                            }
                            if let Some((area_uniform_resource, area_selected_reservoir_resource)) =
                                vpt_area_restir_reads
                            {
                                builder
                                    .read_as(area_uniform_resource, AccessKind::ComputeShaderRead);
                                builder.read_as(
                                    area_selected_reservoir_resource,
                                    AccessKind::ComputeShaderRead,
                                );
                            }
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

                        let noisy_radiance_dep = vpt_writes[0];
                        let noisy_moments_dep = vpt_writes[1];
                        let temporal_radiance_initial_access =
                            if self.vpt_temporal_history_initialized {
                                AccessKind::ComputeShaderRead
                            } else {
                                AccessKind::Undefined
                            };
                        let temporal_moments_initial_access =
                            if self.vpt_temporal_history_initialized {
                                AccessKind::TransferRead
                            } else {
                                AccessKind::Undefined
                            };
                        let previous_temporal_access = if self.vpt_temporal_history_initialized {
                            AccessKind::TransferWrite
                        } else {
                            AccessKind::Undefined
                        };
                        let temporal_radiance_resource = graph.import_image_with_access(
                            vpt_temporal.accumulated_radiance.handle,
                            vpt_temporal.accumulated_radiance.extent.width,
                            vpt_temporal.accumulated_radiance.extent.height,
                            vk::Format::R16G16B16A16_SFLOAT,
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                            temporal_radiance_initial_access,
                        );
                        let temporal_moments_resource = graph.import_image_with_access(
                            vpt_temporal.accumulated_moments_history.handle,
                            vpt_temporal.accumulated_moments_history.extent.width,
                            vpt_temporal.accumulated_moments_history.extent.height,
                            vk::Format::R16G16B16A16_SFLOAT,
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                            temporal_moments_initial_access,
                        );
                        let previous_temporal_radiance_resource = graph.import_image_with_access(
                            vpt_temporal.previous_accumulated_radiance.handle,
                            vpt_temporal.previous_accumulated_radiance.extent.width,
                            vpt_temporal.previous_accumulated_radiance.extent.height,
                            vk::Format::R16G16B16A16_SFLOAT,
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                            previous_temporal_access,
                        );
                        let previous_temporal_moments_resource = graph.import_image_with_access(
                            vpt_temporal.previous_accumulated_moments_history.handle,
                            vpt_temporal
                                .previous_accumulated_moments_history
                                .extent
                                .width,
                            vpt_temporal
                                .previous_accumulated_moments_history
                                .extent
                                .height,
                            vk::Format::R16G16B16A16_SFLOAT,
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                            previous_temporal_access,
                        );
                        let slot = frame.frame_slot;
                        let temporal_writes =
                            graph.add_pass("vpt_temporal", QueueType::Compute, |builder| {
                                builder.read_as(noisy_radiance_dep, AccessKind::ComputeShaderRead);
                                builder.read_as(noisy_moments_dep, AccessKind::ComputeShaderRead);
                                builder.read_as(surface_writes[0], AccessKind::ComputeShaderRead);
                                builder.read_as(surface_writes[1], AccessKind::ComputeShaderRead);
                                builder.read_as(surface_writes[2], AccessKind::ComputeShaderRead);
                                builder.read_as(surface_writes[3], AccessKind::ComputeShaderRead);
                                builder.read_as(
                                    previous_surface_position_resource,
                                    AccessKind::ComputeShaderRead,
                                );
                                builder.read_as(
                                    previous_surface_normal_resource,
                                    AccessKind::ComputeShaderRead,
                                );
                                builder.read_as(
                                    previous_surface_albedo_resource,
                                    AccessKind::ComputeShaderRead,
                                );
                                builder.read_as(
                                    previous_temporal_radiance_resource,
                                    AccessKind::ComputeShaderRead,
                                );
                                builder.read_as(
                                    previous_temporal_moments_resource,
                                    AccessKind::ComputeShaderRead,
                                );
                                builder.write_as(
                                    temporal_radiance_resource,
                                    AccessKind::ComputeShaderWrite,
                                );
                                builder.write_as(
                                    temporal_moments_resource,
                                    AccessKind::ComputeShaderWrite,
                                );
                                Box::new(move |ctx| {
                                    if let Some(profiler) = profiler {
                                        profiler.begin_scope(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                            GpuProfileScope::VptTemporal,
                                        );
                                    }
                                    vpt_temporal.record(ctx.device, ctx.command_buffer, slot);
                                    if let Some(profiler) = profiler {
                                        profiler.end_scope(
                                            ctx.device,
                                            ctx.command_buffer,
                                            slot,
                                            GpuProfileScope::VptTemporal,
                                        );
                                    }
                                })
                            });
                        let temporal_radiance_dep = temporal_writes[0];
                        let temporal_moments_dep = temporal_writes[1];
                        graph.add_pass("vpt_history_update", QueueType::Transfer, |builder| {
                            builder.read_as(temporal_radiance_dep, AccessKind::TransferRead);
                            builder.read_as(temporal_moments_dep, AccessKind::TransferRead);
                            builder.write_as(
                                previous_temporal_radiance_resource,
                                AccessKind::TransferWrite,
                            );
                            builder.write_as(
                                previous_temporal_moments_resource,
                                AccessKind::TransferWrite,
                            );
                            builder.read_as(surface_writes[0], AccessKind::TransferRead);
                            builder.read_as(surface_writes[1], AccessKind::TransferRead);
                            builder.read_as(surface_writes[2], AccessKind::TransferRead);
                            builder.write_as(
                                previous_surface_position_resource,
                                AccessKind::TransferWrite,
                            );
                            builder.write_as(
                                previous_surface_normal_resource,
                                AccessKind::TransferWrite,
                            );
                            builder.write_as(
                                previous_surface_albedo_resource,
                                AccessKind::TransferWrite,
                            );
                            Box::new(move |ctx| {
                                vpt_temporal.record_history_update(ctx.device, ctx.command_buffer);
                                vpt_surface.record_history_update(ctx.device, ctx.command_buffer);
                            })
                        });

                        let postprocess_initial_access = if self.postprocess_output_initialized {
                            AccessKind::TransferRead
                        } else {
                            AccessKind::Undefined
                        };
                        let postprocess_output = postprocess.output_image.handle;
                        let postprocess_extent = postprocess.output_image.extent;
                        let postprocess_output_resource = graph.import_image_with_access(
                            postprocess_output,
                            postprocess_extent.width,
                            postprocess_extent.height,
                            vk::Format::R8G8B8A8_UNORM,
                            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            postprocess_initial_access,
                        );
                        let postprocess_writes =
                            graph.add_pass("postprocess", QueueType::Compute, |builder| {
                                builder
                                    .read_as(temporal_radiance_dep, AccessKind::ComputeShaderRead);
                                builder.write_as(
                                    postprocess_output_resource,
                                    AccessKind::ComputeShaderWrite,
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
                            vpt_temporal_ready = self.vpt_temporal_pass.is_some(),
                            postprocess_ready = self.postprocess_pass.is_some(),
                            "skipping VPT frame until required passes are initialized"
                        );
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
                if let Some(current_view_proj) = current_vpt_view_proj {
                    self.previous_vpt_view_proj = Some(current_view_proj);
                    self.previous_vpt_resolution =
                        Some([frame.swapchain_extent.width, frame.swapchain_extent.height]);
                }
                if vpt_accumulation_written {
                    self.vpt_accumulation_needs_init = false;
                    self.vpt_temporal_history_initialized = true;
                    self.postprocess_output_initialized = true;
                }
                if restir_di_selected_written {
                    self.restir_di_history_initialized = true;
                }
                if area_restir_selected_written {
                    self.area_restir_history_initialized = true;
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
            if let Some(pass) = self.vpt_temporal_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(pass) = self.area_restir_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(pass) = self.vpt_surface_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(pass) = self.vpt_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(pass) = self.restir_di_pass.take() {
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
        let area_restir_settings_result = AreaRestirSettings::from_env();
        for warning in &area_restir_settings_result.warnings {
            tracing::warn!(
                variable = warning.variable,
                value = %warning.value,
                expected = warning.expected,
                "invalid Area ReSTIR setting override; using default value"
            );
        }
        self.area_restir_settings = area_restir_settings_result.settings;
        if let Some(vpt_debug_view) =
            area_restir_debug_to_vpt_debug_view(self.area_restir_settings.debug_view)
        {
            self.lighting_settings.vpt_debug_view = vpt_debug_view;
        }

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

        // Initialize VPT surface pass (requires UCVH GPU resources + Scene UBO)
        if self.vpt_surface_pass.is_none() {
            if let (Some(ucvh_gpu), Some(scene_ubo_ref)) = (&self.ucvh_gpu, &self.scene_ubo) {
                let renderer = self.renderer.as_ref().unwrap();
                let extent = renderer.swapchain_extent();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt_surface.spv"));
                if spirv.is_empty() {
                    tracing::warn!("vpt_surface.spv is empty; slangc may not be installed");
                } else {
                    match VptSurfacePass::new(
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
                                "initialized VPT surface pass"
                            );
                            self.vpt_surface_pass = Some(pass);
                        }
                        Err(error) => {
                            tracing::error!(%error, "failed to create VPT surface pass");
                        }
                    }
                }
            }
        }

        // Initialize VPT pass (requires UCVH GPU resources + Scene UBO)
        if self.vpt_pass.is_none() {
            if let (Some(ucvh_gpu), Some(scene_ubo_ref)) = (&self.ucvh_gpu, &self.scene_ubo) {
                let renderer = self.renderer.as_ref().unwrap();
                let extent = renderer.swapchain_extent();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt.spv"));
                if spirv.is_empty() {
                    tracing::warn!("vpt.spv is empty 闁?slangc may not be installed");
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
                                "initialized VPT pass"
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

        if self.area_restir_pass.is_none()
            && self.area_restir_vpt_enabled()
            && let (Some(scene_ubo_ref), Some(ucvh_gpu)) = (&self.scene_ubo, &self.ucvh_gpu)
        {
            let renderer = self.renderer.as_ref().unwrap();
            let extent = renderer.swapchain_extent();
            let initial_spirv =
                include_bytes!(concat!(env!("OUT_DIR"), "/shaders/area_restir_initial.spv"));
            let temporal_spirv = include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/area_restir_temporal.spv"
            ));
            let spatial_spirv =
                include_bytes!(concat!(env!("OUT_DIR"), "/shaders/area_restir_spatial.spv"));
            if initial_spirv.is_empty() || temporal_spirv.is_empty() || spatial_spirv.is_empty() {
                tracing::warn!("Area ReSTIR shaders are empty; slangc may not be installed");
            } else {
                match AreaRestirPass::new(
                    renderer.device(),
                    renderer.allocator(),
                    AreaRestirPassCreateInfo {
                        width: extent.width,
                        height: extent.height,
                        frame_count: scene_ubo_ref.frame_count(),
                        initial_spirv,
                        temporal_spirv,
                        spatial_spirv,
                        scene_ubo: scene_ubo_ref,
                        ucvh_gpu,
                    },
                ) {
                    Ok(pass) => {
                        tracing::info!(
                            width = extent.width,
                            height = extent.height,
                            "initialized Area ReSTIR VPT sample-area pass"
                        );
                        self.area_restir_pass = Some(pass);
                        if let (Some(area_restir), Some(vpt)) =
                            (&self.area_restir_pass, &self.vpt_pass)
                        {
                            for slot in 0..scene_ubo_ref.frame_count() {
                                let (area_uniform_buffer, _, _) = area_restir.uniform_buffer(slot);
                                let (area_selected_current_buffer, _, _) =
                                    area_restir.selected_current_buffer(slot);
                                vpt.update_area_restir_descriptors(
                                    renderer.device(),
                                    slot,
                                    area_uniform_buffer,
                                    area_selected_current_buffer,
                                );
                            }
                        }
                        self.area_restir_history_initialized = false;
                    }
                    Err(error) => {
                        tracing::error!(%error, "failed to create Area ReSTIR pass");
                    }
                }
            }
        }

        if self.vpt_temporal_pass.is_none()
            && let (Some(vpt), Some(vpt_surface), Some(scene_ubo_ref)) =
                (&self.vpt_pass, &self.vpt_surface_pass, &self.scene_ubo)
        {
            let renderer = self.renderer.as_ref().unwrap();
            let extent = renderer.swapchain_extent();
            let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt_temporal.spv"));
            if spirv.is_empty() {
                tracing::warn!("vpt_temporal.spv is empty; slangc may not be installed");
            } else {
                match VptTemporalPass::new(
                    renderer.device(),
                    renderer.allocator(),
                    VptTemporalPassCreateInfo {
                        width: extent.width,
                        height: extent.height,
                        spirv_bytes: spirv,
                        scene_ubo: scene_ubo_ref,
                        vpt,
                        vpt_surface,
                    },
                ) {
                    Ok(pass) => {
                        tracing::info!(
                            width = extent.width,
                            height = extent.height,
                            "initialized VPT temporal pass"
                        );
                        self.vpt_temporal_pass = Some(pass);
                        self.vpt_temporal_history_initialized = false;
                    }
                    Err(error) => {
                        tracing::error!(%error, "failed to create VPT temporal pass");
                    }
                }
            }
        }

        if self.postprocess_pass.is_none()
            && let (Some(vpt_temporal), Some(scene_ubo_ref)) =
                (&self.vpt_temporal_pass, &self.scene_ubo)
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
                    &vpt_temporal.accumulated_radiance,
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
                    return; // minimized 閳?skip resize
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_projection_round_trips_pixel_to_ray_center_coordinates() {
        let camera_pos = glam::Vec3::new(64.0, 32.0, -40.0);
        let camera_forward = glam::Vec3::new(0.0, -0.03, 0.99955).normalize();
        let camera_up = glam::Vec3::Y;
        let width = 800;
        let height = 600;
        let pixel = glam::Vec2::new(337.0, 214.0);

        let pixel_to_ray = compute_pixel_to_ray(
            camera_pos,
            camera_forward,
            camera_up,
            std::f32::consts::FRAC_PI_4,
            width,
            height,
        );
        let ray_basis = glam::Mat3::from_cols(
            pixel_to_ray.col(0).truncate(),
            pixel_to_ray.col(1).truncate(),
            pixel_to_ray.col(2).truncate(),
        );
        let world_position =
            camera_pos + (ray_basis * glam::Vec3::new(pixel.x, pixel.y, 1.0)).normalize() * 100.0;

        let clip = compute_view_proj(
            camera_pos,
            camera_forward,
            camera_up,
            std::f32::consts::FRAC_PI_4,
            width,
            height,
        ) * world_position.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let reprojected = glam::Vec2::new(
            (ndc.x * 0.5 + 0.5) * width as f32,
            (0.5 - ndc.y * 0.5) * height as f32,
        );

        let expected = pixel + 0.5;
        assert!(
            (reprojected - expected).length() < 1.0e-3,
            "expected {expected}, got {reprojected}"
        );
    }
}
