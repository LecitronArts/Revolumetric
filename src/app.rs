use anyhow::Result;
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
use crate::render::capture::RenderCapture;
use crate::render::device::RenderDevice;
use crate::render::gpu_profiler::{GpuProfiler, GpuProfilerConfig};
use crate::render::restir_di::RestirDiSettings;
use crate::render::scene_ubo::{LightingSettings, SceneUniformBuffer, VptDebugView};
use crate::render::vpt_pipeline::{VptCameraFrame, VptFrameInputs, VptRuntimePipeline};
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
    capture: Option<RenderCapture>,
    vpt_pipeline: VptRuntimePipeline,
    ucvh: Option<Ucvh>,
    ucvh_gpu: Option<UcvhGpuResources>,
    ucvh_uploaded: bool,
    scene_ubo: Option<SceneUniformBuffer>,
    lighting_settings: LightingSettings,
    area_restir_settings: AreaRestirSettings,
    restir_di_settings: RestirDiSettings,
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
            capture: None,
            vpt_pipeline: VptRuntimePipeline::new(),
            ucvh: None,
            ucvh_gpu: None,
            ucvh_uploaded: false,
            scene_ubo: None,
            lighting_settings: LightingSettings::default(),
            area_restir_settings: AreaRestirSettings::default(),
            restir_di_settings: RestirDiSettings::default(),
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
        let renderer = match self.renderer.as_ref() {
            Some(renderer) => renderer,
            None => return Ok(()),
        };
        let (scene_ubo, ucvh_gpu) = match (&self.scene_ubo, &self.ucvh_gpu) {
            (Some(scene_ubo), Some(ucvh_gpu)) => (scene_ubo, ucvh_gpu),
            _ => return Ok(()),
        };

        self.vpt_pipeline.resize(
            renderer,
            scene_ubo,
            ucvh_gpu,
            width,
            height,
            self.restir_di_vpt_enabled(),
            self.area_restir_vpt_enabled(),
        )
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

    fn current_vpt_camera_frame(&self) -> VptCameraFrame {
        match self.world.resource::<CameraRig>() {
            Some(rig) => VptCameraFrame {
                position: rig.camera.position,
                forward: rig.camera.forward,
                up: rig.camera.up,
                fov_y_radians: rig.camera.fov_y_radians,
                aperture_radius: rig.camera.aperture_radius,
                focal_distance: rig.camera.focal_distance,
            },
            None => VptCameraFrame {
                position: glam::Vec3::new(64.0, 80.0, -40.0),
                forward: glam::Vec3::Z,
                up: glam::Vec3::Y,
                fov_y_radians: std::f32::consts::FRAC_PI_4,
                aperture_radius: 0.0,
                focal_distance: 128.0,
            },
        }
    }

    fn current_sun_light(&self) -> (glam::Vec3, glam::Vec3) {
        match self.world.resource::<DirectionalLight>() {
            Some(light) => (light.direction, light.intensity),
            None => (
                glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
                glam::Vec3::new(2.0, 1.5, 1.25),
            ),
        }
    }

    fn current_elapsed_seconds(&self) -> f32 {
        self.world
            .resource::<Time>()
            .map_or(0.0, |time| time.elapsed_seconds)
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
        let camera = self.current_vpt_camera_frame();
        let (sun_direction, sun_intensity) = self.current_sun_light();
        let elapsed_seconds = self.current_elapsed_seconds();
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

                if let Some(scene_ubo) = &self.scene_ubo {
                    let record_result = self.vpt_pipeline.record_and_execute_frame(
                        renderer,
                        &frame,
                        VptFrameInputs {
                            scene_ubo,
                            camera,
                            sun_direction,
                            sun_intensity,
                            elapsed_seconds,
                            lighting_settings: self.lighting_settings,
                            restir_di_settings: self.restir_di_settings,
                            area_restir_settings: self.area_restir_settings,
                            restir_di_enabled,
                            area_restir_enabled,
                            ucvh_ready: self.ucvh_uploaded,
                            capture: self.capture.as_mut(),
                            profiler: self.gpu_profiler.as_ref(),
                        },
                    )?;
                    let submitted_fence = record_result.submitted_fence;
                    let mut pending_capture = record_result.pending_capture;
                    renderer.end_frame(frame)?;
                    if let Some(metadata) = pending_capture.take() {
                        renderer.wait_for_fence(submitted_fence)?;
                        if let Some(capture) = &self.capture {
                            capture.write_rgba8_capture(&metadata)?;
                            tracing::info!(
                                frame_index = metadata.frame_index,
                                ppm = %metadata.ppm_path.display(),
                                json = %metadata.json_path.display(),
                                "wrote postprocess capture"
                            );
                        }
                    }
                } else {
                    tracing::warn!("skipping render frame until scene UBO is initialized");
                    renderer.end_frame(frame)?;
                }
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
            if let Some(capture) = self.capture.take() {
                capture.destroy(renderer.device(), renderer.allocator());
            }
            let vpt_pipeline = std::mem::take(&mut self.vpt_pipeline);
            vpt_pipeline.destroy(renderer.device(), renderer.allocator());
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
        self.capture = match RenderCapture::from_env() {
            Ok(capture) => {
                if let Some(capture) = &capture {
                    tracing::info!(
                        target_frame = ?capture.config().target_frame,
                        output_dir = %capture.config().output_dir.display(),
                        prefix = %capture.config().prefix,
                        "enabled postprocess capture"
                    );
                }
                capture
            }
            Err(error) => {
                tracing::warn!(%error, "invalid postprocess capture configuration; capture disabled");
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

        let restir_di_enabled = self.restir_di_vpt_enabled();
        let area_restir_enabled = self.area_restir_vpt_enabled();
        if let (Some(renderer), Some(scene_ubo)) = (self.renderer.as_ref(), self.scene_ubo.as_ref())
        {
            self.vpt_pipeline.ensure_passes(
                renderer,
                scene_ubo,
                self.ucvh.as_ref(),
                self.ucvh_gpu.as_ref(),
                restir_di_enabled,
                area_restir_enabled,
            );
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
                    return; // minimized, skip resize
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
    use crate::render::camera::{compute_pixel_to_ray, compute_view_proj};
    use crate::scene::camera::Camera;
    use crate::scene::components::CameraRig;
    use crate::scene::light::DirectionalLight;

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

    #[test]
    fn current_vpt_camera_frame_uses_rig_when_present() {
        let mut app = RevolumetricApp::new();
        app.world.insert_resource(CameraRig {
            camera: Camera {
                position: glam::Vec3::new(1.0, 2.0, 3.0),
                forward: glam::Vec3::new(0.25, 0.5, 0.75).normalize(),
                up: glam::Vec3::Y,
                fov_y_radians: 1.1,
                aperture_radius: 0.25,
                focal_distance: 42.0,
            },
            ..CameraRig::default()
        });

        let camera = app.current_vpt_camera_frame();
        assert_eq!(camera.position, glam::Vec3::new(1.0, 2.0, 3.0));
        assert!((camera.forward - glam::Vec3::new(0.25, 0.5, 0.75).normalize()).length() < 1e-6);
        assert_eq!(camera.up, glam::Vec3::Y);
        assert!((camera.fov_y_radians - 1.1).abs() < 1e-6);
        assert!((camera.aperture_radius - 0.25).abs() < 1e-6);
        assert!((camera.focal_distance - 42.0).abs() < 1e-6);
    }

    #[test]
    fn current_vpt_camera_frame_falls_back_to_expected_defaults() {
        let app = RevolumetricApp::new();
        let camera = app.current_vpt_camera_frame();

        assert_eq!(camera.position, glam::Vec3::new(64.0, 80.0, -40.0));
        assert_eq!(camera.forward, glam::Vec3::Z);
        assert_eq!(camera.up, glam::Vec3::Y);
        assert!((camera.fov_y_radians - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert_eq!(camera.aperture_radius, 0.0);
        assert_eq!(camera.focal_distance, 128.0);
    }

    #[test]
    fn current_sun_light_uses_world_light_when_present() {
        let mut app = RevolumetricApp::new();
        app.world.insert_resource(DirectionalLight {
            direction: glam::Vec3::new(-1.0, 0.5, 0.25).normalize(),
            intensity: glam::Vec3::new(4.0, 3.0, 2.0),
        });

        let (direction, intensity) = app.current_sun_light();
        assert!((direction - glam::Vec3::new(-1.0, 0.5, 0.25).normalize()).length() < 1e-6);
        assert_eq!(intensity, glam::Vec3::new(4.0, 3.0, 2.0));
    }

    #[test]
    fn current_sun_light_falls_back_to_expected_defaults() {
        let app = RevolumetricApp::new();
        let (direction, intensity) = app.current_sun_light();

        assert!((direction - glam::Vec3::new(0.5, 1.0, 0.25).normalize()).length() < 1e-6);
        assert_eq!(intensity, glam::Vec3::new(2.0, 1.5, 1.25));
    }

    #[test]
    fn current_elapsed_seconds_defaults_to_zero_without_time_resource() {
        let app = RevolumetricApp::new();
        assert_eq!(app.current_elapsed_seconds(), 0.0);
    }

    #[test]
    fn app_delegates_vpt_pass_ownership_to_runtime_pipeline() {
        let source = crate::render::source_checks::read_source("src/app.rs");
        let app_struct = source
            .split("struct RevolumetricApp")
            .nth(1)
            .expect("RevolumetricApp struct should exist")
            .split("impl RevolumetricApp")
            .next()
            .expect("RevolumetricApp struct should end before impl");

        assert!(app_struct.contains("vpt_pipeline: VptRuntimePipeline"));
        assert!(!app_struct.contains("vpt_pass: Option<VptPass>"));
        assert!(!app_struct.contains("vpt_surface_pass: Option<VptSurfacePass>"));
        assert!(!app_struct.contains("postprocess_pass: Option<PostprocessPass>"));
        assert!(source.contains("self.vpt_pipeline.record_and_execute_frame("));
        let add_pass_call = ["graph", ".", "add_pass"].concat();
        for pass_name in [
            "vpt".to_string(),
            ["vpt", "_temporal"].concat(),
            ["vpt", "_atrous"].concat(),
            ["post", "process"].concat(),
            ["capture", "_post", "process"].concat(),
            ["blit", "_to_", "swapchain"].concat(),
        ] {
            let forbidden = format!("{add_pass_call}(\"{pass_name}\"");
            assert!(
                !source.contains(&forbidden),
                "app.rs must not own VPT frame graph pass {forbidden}"
            );
        }
    }
}
