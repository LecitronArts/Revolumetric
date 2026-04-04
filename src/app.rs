use anyhow::Result;
use ash::vk;
use tracing_subscriber::{fmt, EnvFilter};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::ecs::schedule::{Schedule, Stage};
use crate::ecs::world::World;
use crate::platform::time::Time;
use crate::platform::window::WindowDescriptor;
use crate::render::device::RenderDevice;
use crate::render::graph::RenderGraph;
use crate::render::passes::blit_to_swapchain;
use crate::render::camera::{compute_pixel_to_ray, PrimaryRayPushConstants};
use crate::render::passes::primary_ray::PrimaryRayPass;
use crate::render::resource::QueueType;
use crate::scene::systems;
use crate::voxel::ucvh::{Ucvh, UcvhConfig};
use crate::voxel::generator;
use crate::voxel::gpu_upload::UcvhGpuResources;

pub fn run() -> Result<()> {
    init_tracing();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = RevolumetricApp::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct RevolumetricApp {
    world: World,
    schedule: Schedule,
    renderer: Option<RenderDevice>,
    primary_ray_pass: Option<PrimaryRayPass>,
    ucvh: Option<Ucvh>,
    ucvh_gpu: Option<UcvhGpuResources>,
    ucvh_uploaded: bool,
    window_descriptor: WindowDescriptor,
    window: Option<Window>,
    window_id: Option<WindowId>,
    initialized: bool,
    start_time: std::time::Instant,
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
            primary_ray_pass: None,
            ucvh: None,
            ucvh_gpu: None,
            ucvh_uploaded: false,
            window_descriptor: WindowDescriptor::default(),
            window: None,
            window_id: None,
            initialized: false,
            start_time: std::time::Instant::now(),
        }
    }

    fn tick_frame(&mut self) -> Result<()> {
        self.schedule.run_stage(Stage::PreUpdate, &mut self.world)?;
        self.schedule.run_stage(Stage::Update, &mut self.world)?;
        self.schedule.run_stage(Stage::PostUpdate, &mut self.world)?;
        self.schedule.run_stage(Stage::ExtractRender, &mut self.world)?;
        self.schedule.run_stage(Stage::PrepareRender, &mut self.world)?;

        if let Some(renderer) = self.renderer.as_mut() {
            let frame = renderer.begin_frame()?;
            if frame.should_render {
                // Upload UCVH data to GPU (first frame only)
                if !self.ucvh_uploaded {
                    if let (Some(ucvh), Some(gpu)) = (&self.ucvh, &self.ucvh_gpu) {
                        gpu.upload_all(renderer.device(), frame.command_buffer, ucvh);
                        self.ucvh_uploaded = true;
                        tracing::info!("uploaded UCVH data to GPU");
                    }
                }

                let mut graph = RenderGraph::new();

                if let Some(pass) = &self.primary_ray_pass {
                    // Hardcoded camera (Phase 9 adds FPS controller)
                    let camera_pos = glam::Vec3::new(64.0, 80.0, -40.0);
                    let camera_target = glam::Vec3::new(64.0, 64.0, 64.0);
                    let camera_forward = (camera_target - camera_pos).normalize();
                    let camera_up = glam::Vec3::Y;
                    let fov_y = std::f32::consts::FRAC_PI_4; // 45° FOV

                    let pixel_to_ray = compute_pixel_to_ray(
                        camera_pos, camera_forward, camera_up, fov_y,
                        frame.swapchain_extent.width, frame.swapchain_extent.height,
                    );

                    let pc = PrimaryRayPushConstants {
                        // Slang float4x4 uses RowMajor in SPIR-V: driver reads memory as rows
                        // and transposes into columns. Pre-transpose so columns arrive correctly.
                        pixel_to_ray: pixel_to_ray.transpose().to_cols_array_2d(),
                        resolution: [frame.swapchain_extent.width, frame.swapchain_extent.height],
                        _pad: [0; 2],
                    };

                    let output_extent = pass.output_image.extent;
                    let output_img = pass.output_image.handle;

                    let primary_ray_writes = graph.add_pass(
                        "primary_ray",
                        QueueType::Compute,
                        |builder| {
                            let _output = builder.create_image(
                                frame.swapchain_extent.width,
                                frame.swapchain_extent.height,
                                vk::Format::R8G8B8A8_UNORM,
                                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            );
                            Box::new(move |ctx| {
                                pass.record(ctx.device, ctx.command_buffer, &pc);
                            })
                        },
                    );

                    let src_image = output_img;
                    let src_extent = output_extent;
                    let dst_image = frame.swapchain_image;
                    let dst_extent = frame.swapchain_extent;
                    let dep_handle = primary_ray_writes[0];
                    graph.add_pass(
                        "blit_to_swapchain",
                        QueueType::Graphics,
                        |builder| {
                            builder.read(dep_handle);
                            Box::new(move |ctx| {
                                blit_to_swapchain::record_blit(
                                    ctx.device, ctx.command_buffer,
                                    src_image, src_extent, dst_image, dst_extent,
                                );
                            })
                        },
                    );
                }

                graph.compile();
                graph.execute(renderer.device(), frame.command_buffer, frame.frame_index);
                renderer.end_frame(frame)?;
            }
        }

        self.schedule.run_stage(Stage::ExecuteRender, &mut self.world)?;
        Ok(())
    }
}

impl Drop for RevolumetricApp {
    fn drop(&mut self) {
        // Destroy GPU passes before the renderer (which owns the device/allocator).
        if let Some(renderer) = &self.renderer {
            unsafe { renderer.device().device_wait_idle().ok() };
            if let Some(pass) = self.primary_ray_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(gpu) = self.ucvh_gpu.take() {
                gpu.destroy(renderer.device(), renderer.allocator());
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
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.window_id = Some(window_id);

        // Generate UCVH demo scene
        if self.ucvh.is_none() {
            let config = UcvhConfig::new(glam::UVec3::splat(128));
            let mut ucvh = Ucvh::new(config);
            let brick_count = generator::generate_demo_scene(&mut ucvh);
            ucvh.rebuild_hierarchy();
            tracing::info!(
                bricks = brick_count,
                total_voxels = ucvh.pool.allocated_count() as u64 * 512,
                "generated demo sphere scene"
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

        // Initialize primary ray pass (requires UCVH GPU resources)
        if self.primary_ray_pass.is_none() {
            if let Some(ucvh_gpu) = &self.ucvh_gpu {
                let renderer = self.renderer.as_ref().unwrap();
                let extent = renderer.swapchain_extent();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/primary_ray.spv"));
                if spirv.is_empty() {
                    tracing::warn!("primary_ray.spv is empty — slangc may not be installed");
                } else {
                    match PrimaryRayPass::new(
                        renderer.device(),
                        renderer.allocator(),
                        extent.width,
                        extent.height,
                        spirv,
                        ucvh_gpu,
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
                if let Err(error) = self.tick_frame() {
                    tracing::error!(%error, "frame execution failed");
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    if let Err(error) = renderer.handle_resize(size.width, size.height) {
                        tracing::error!(%error, "failed to recreate swapchain after resize");
                        event_loop.exit();
                        return;
                    }
                }
                tracing::debug!(width = size.width, height = size.height, "window resized");
            }
            _ => {}
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
