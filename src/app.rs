use anyhow::Result;
use ash::vk;
use tracing_subscriber::{fmt, EnvFilter};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::platform::input::InputState;
use crate::scene::components::CameraRig;

use crate::ecs::schedule::{Schedule, Stage};
use crate::ecs::world::World;
use crate::platform::time::Time;
use crate::platform::window::WindowDescriptor;
use crate::render::device::RenderDevice;
use crate::render::graph::RenderGraph;
use crate::render::passes::blit_to_swapchain;
use crate::render::camera::compute_pixel_to_ray;
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::scene::light::DirectionalLight;
use crate::render::passes::lighting::LightingPass;
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
    lighting_pass: Option<LightingPass>,
    ucvh: Option<Ucvh>,
    ucvh_gpu: Option<UcvhGpuResources>,
    ucvh_uploaded: bool,
    scene_ubo: Option<SceneUniformBuffer>,
    window_descriptor: WindowDescriptor,
    window: Option<Window>,
    window_id: Option<WindowId>,
    initialized: bool,
    last_cursor_pos: Option<(f64, f64)>,
    last_frame_time: Option<std::time::Instant>,
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
            lighting_pass: None,
            ucvh: None,
            ucvh_gpu: None,
            ucvh_uploaded: false,
            scene_ubo: None,
            window_descriptor: WindowDescriptor::default(),
            window: None,
            window_id: None,
            initialized: false,
            last_cursor_pos: None,
            last_frame_time: None,
        }
    }

    fn update_camera(&mut self, dt: f32) {
        // Clone InputState (it's Copy) to avoid borrow conflicts
        let input = match self.world.resource::<InputState>() {
            Some(input) => *input,
            None => return,
        };

        let rig = match self.world.resource_mut::<CameraRig>() {
            Some(rig) => rig,
            None => return,
        };
        let ctrl = &mut rig.controller;
        let cam = &mut rig.camera;

        // Scroll → adjust speed
        if input.scroll_delta != 0.0 {
            ctrl.move_speed *= ctrl.scroll_multiplier.powf(input.scroll_delta);
            ctrl.move_speed = ctrl.move_speed.clamp(ctrl.min_speed, ctrl.max_speed);
        }

        // Mouse → yaw/pitch
        let sens_rad = ctrl.mouse_sensitivity * std::f32::consts::PI / 180.0;
        ctrl.yaw += input.mouse_dx * sens_rad;
        ctrl.pitch -= input.mouse_dy * sens_rad;
        ctrl.pitch = ctrl.pitch.clamp(-1.553, 1.553); // ±89°

        // Recompute forward from yaw/pitch
        cam.forward = glam::Vec3::new(
            ctrl.pitch.cos() * ctrl.yaw.sin(),
            ctrl.pitch.sin(),
            ctrl.pitch.cos() * ctrl.yaw.cos(),
        );

        // Horizontal movement
        let hz_forward = glam::Vec3::new(ctrl.yaw.sin(), 0.0, ctrl.yaw.cos());
        let hz_right = glam::Vec3::Y.cross(hz_forward);

        let mut velocity = hz_forward * input.move_forward
            + hz_right * input.move_right
            + glam::Vec3::Y * input.move_up;
        if velocity.length_squared() > 0.0 {
            velocity = velocity.normalize();
        }

        cam.position += velocity * ctrl.move_speed * dt;
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
        self.schedule.run_stage(Stage::PostUpdate, &mut self.world)?;
        self.update_camera(dt);
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
                        cam_pos, cam_forward, cam_up, fov_y,
                        frame.swapchain_extent.width, frame.swapchain_extent.height,
                    );

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

                    // Fill and upload Scene UBO
                    let scene_data = GpuSceneUniforms {
                        pixel_to_ray: pixel_to_ray.transpose().to_cols_array_2d(),
                        resolution: [frame.swapchain_extent.width, frame.swapchain_extent.height],
                        _pad0: [0; 2],
                        sun_direction: sun_dir.to_array(),
                        _pad1: 0.0,
                        sun_intensity: sun_intensity.to_array(),
                        _pad2: 0.0,
                        sky_color: [0.4, 0.5, 0.7],
                        _pad3: 0.0,
                        ground_color: [0.15, 0.1, 0.08],
                        time: self.world.resource::<Time>().map_or(0.0, |t| t.elapsed_seconds),
                        rc_c0_grid: [16, 16, 16],
                        rc_c0_offset: 0,
                        rc_enabled: 0,  // disabled until RC passes are wired up
                        _pad4: [0; 3],
                    };

                    if let Some(ubo) = &self.scene_ubo {
                        ubo.update(frame.frame_slot, &scene_data);
                    }

                    let primary_ray_writes = graph.add_pass(
                        "primary_ray",
                        QueueType::Compute,
                        |builder| {
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
                                vk::ImageUsageFlags::STORAGE,
                            );
                            let _gb1 = builder.create_image(
                                frame.swapchain_extent.width,
                                frame.swapchain_extent.height,
                                vk::Format::R8G8B8A8_UINT,
                                vk::ImageUsageFlags::STORAGE,
                            );
                            let slot = frame.frame_slot;
                            Box::new(move |ctx| {
                                pass.record(ctx.device, ctx.command_buffer, slot);
                            })
                        },
                    );

                    // Lighting pass
                    if let Some(lighting) = &self.lighting_pass {
                        let gbuf_images = [
                            pass.gbuffer_pos.handle,
                            pass.gbuffer0.handle,
                            pass.gbuffer1.handle,
                        ];
                        let lighting_output = lighting.output_image.handle;
                        let lighting_extent = lighting.output_image.extent;
                        let dep0 = primary_ray_writes[0];
                        let dep1 = primary_ray_writes[1];
                        let dep2 = primary_ray_writes[2];
                        let slot = frame.frame_slot;

                        let lighting_writes = graph.add_pass(
                            "lighting",
                            QueueType::Compute,
                            |builder| {
                                builder.read(dep0);
                                builder.read(dep1);
                                builder.read(dep2);
                                let _out = builder.create_image(
                                    frame.swapchain_extent.width,
                                    frame.swapchain_extent.height,
                                    vk::Format::R8G8B8A8_UNORM,
                                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                                );
                                Box::new(move |ctx| {
                                    lighting.record(ctx.device, ctx.command_buffer, slot, gbuf_images);
                                })
                            },
                        );

                        // Blit lighting output to swapchain
                        let src_image = lighting_output;
                        let src_extent = lighting_extent;
                        let dst_image = frame.swapchain_image;
                        let dst_extent = frame.swapchain_extent;
                        let dep_handle = lighting_writes[0];
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
                    } else {
                        // Fallback: blit raw G-buffer if lighting pass not ready
                        let src_image = pass.gbuffer0.handle;
                        let src_extent = pass.gbuffer0.extent;
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
                }

                graph.compile();
                graph.execute(renderer.device(), frame.command_buffer, frame.frame_index);
                renderer.end_frame(frame)?;
            }
        }

        if let Some(input) = self.world.resource_mut::<InputState>() {
            input.clear_per_frame();
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
                    tracing::info!(frame_count = renderer.swapchain_image_count(), "created scene UBO");
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
                    tracing::warn!("primary_ray.spv is empty — slangc may not be installed");
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
                    tracing::warn!("lighting.spv is empty — slangc may not be installed");
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
                    if !pressed {
                        self.last_cursor_pos = None; // prevent jump on re-press
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(input) = self.world.resource_mut::<InputState>() {
                    if input.right_mouse_held {
                        if let Some((last_x, last_y)) = self.last_cursor_pos {
                            input.mouse_dx += (position.x - last_x) as f32;
                            input.mouse_dy += (position.y - last_y) as f32;
                        }
                        self.last_cursor_pos = Some((position.x, position.y));
                    }
                }
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
                }
                self.last_cursor_pos = None;
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
