use anyhow::Result;
use tracing_subscriber::{EnvFilter, fmt};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, Touch, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

#[cfg(target_os = "android")]
use winit::platform::android::EventLoopBuilderExtAndroid;
#[cfg(target_os = "android")]
use winit::platform::android::activity::AndroidApp;

#[cfg(not(target_os = "android"))]
use crate::editor::fonts::{configure_editor_fonts, configure_editor_style};
#[cfg(not(target_os = "android"))]
use crate::editor::ui::{EditorUi, EditorUiFrameState};
use crate::platform::input::InputState;
use crate::scene::camera::{Camera, CameraPathConfig, apply_camera_path, update_fly_camera};
use crate::scene::components::CameraRig;

use crate::ecs::schedule::{Schedule, Stage};
use crate::ecs::world::World;
use crate::platform::time::Time;
use crate::platform::window::WindowDescriptor;
use crate::render::area_restir::{AreaRestirDebugView, AreaRestirSettings};
use crate::render::capture::CaptureCameraPathMetadata;
#[cfg(not(target_os = "android"))]
use crate::render::egui_renderer::EguiFrame;
use crate::render::restir_di::RestirDiSettings;
use crate::render::rt_settings::RtSettings;
use crate::render::runtime::{RenderFrameInput, RenderRuntime, RuntimeSettings};
use crate::render::scene_ubo::{LightingSettings, VptDebugView};
use crate::render::vpt_pipeline::VptCameraFrame;
use crate::scene::light::DirectionalLight;
use crate::scene::systems;
use crate::voxel::generator;
use crate::voxel::ucvh::Ucvh;
use crate::voxel::vox_loader::VoxTargetBounds;
use std::sync::mpsc::{self, Receiver, TryRecvError};

pub fn run() -> Result<()> {
    init_tracing();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = RevolumetricApp::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(target_os = "android")]
pub fn run_android(app: AndroidApp) -> Result<()> {
    init_tracing();

    let mut builder = EventLoop::builder();
    builder.with_android_app(app);
    let event_loop = builder.build()?;
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

const ACTIVE_REDRAW_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);
const DEFAULT_SCENE_LOADING_REDRAW_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(100);

struct RevolumetricApp {
    world: World,
    schedule: Schedule,
    render_runtime: Option<RenderRuntime>,
    ucvh: Option<Ucvh>,
    lighting_settings: LightingSettings,
    rt_settings: RtSettings,
    area_restir_settings: AreaRestirSettings,
    restir_di_settings: RestirDiSettings,
    window_descriptor: WindowDescriptor,
    window: Option<Window>,
    window_id: Option<WindowId>,
    #[cfg(not(target_os = "android"))]
    egui_ctx: Option<egui::Context>,
    #[cfg(not(target_os = "android"))]
    egui_state: Option<egui_winit::State>,
    #[cfg(not(target_os = "android"))]
    editor_ui: Option<EditorUi>,
    #[cfg(not(target_os = "android"))]
    pending_egui_textures_delta: egui::TexturesDelta,
    initialized: bool,
    touch_look: TouchLookState,
    last_frame_time: Option<std::time::Instant>,
    rendered_frames: u64,
    exit_after_frames: Option<u64>,
    camera_path: Option<CameraPathConfig>,
    default_scene_target_bounds: Option<VoxTargetBounds>,
    default_scene_load: Option<Receiver<DefaultSceneLoadMessage>>,
    next_redraw_at: std::time::Instant,
}

struct DefaultSceneLoadMessage {
    ucvh: Ucvh,
    scene_result: generator::DefaultSceneLoadResult,
    elapsed: std::time::Duration,
}

#[derive(Debug, Default)]
struct TouchLookState {
    active_touch_id: Option<u64>,
    last_position: Option<(f64, f64)>,
}

impl TouchLookState {
    fn clear(&mut self) {
        self.active_touch_id = None;
        self.last_position = None;
    }

    fn handle_touch(&mut self, input: &mut InputState, touch: Touch) {
        match touch.phase {
            TouchPhase::Started => {
                if self.active_touch_id.is_none() {
                    self.active_touch_id = Some(touch.id);
                    self.last_position = Some((touch.location.x, touch.location.y));
                }
            }
            TouchPhase::Moved => {
                if self.active_touch_id == Some(touch.id) {
                    if let Some((last_x, last_y)) = self.last_position {
                        input.mouse_dx += (touch.location.x - last_x) as f32;
                        input.mouse_dy += (touch.location.y - last_y) as f32;
                    }
                    self.last_position = Some((touch.location.x, touch.location.y));
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                if self.active_touch_id == Some(touch.id) {
                    self.clear();
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct UiInputCapture {
    consumed: bool,
    wants_keyboard_input: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CameraInputEventKind {
    Keyboard,
    Pointer,
}

fn camera_should_receive_input(kind: CameraInputEventKind, capture: UiInputCapture) -> bool {
    if capture.consumed {
        return false;
    }
    match kind {
        CameraInputEventKind::Keyboard => !capture.wants_keyboard_input,
        CameraInputEventKind::Pointer => true,
    }
}

fn should_request_redraw_for_size(size: winit::dpi::PhysicalSize<u32>) -> bool {
    size.width > 0 && size.height > 0
}

fn clear_input_for_inactive_window(
    input: Option<&mut InputState>,
    touch_look: &mut TouchLookState,
    window: Option<&Window>,
) {
    if let Some(input) = input {
        input.clear_for_focus_loss();
    }
    touch_look.clear();
    if let Some(window) = window {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
    }
}

fn camera_for_scene_bounds(bounds: Option<VoxTargetBounds>, world_size: glam::UVec3) -> Camera {
    let bounds = bounds.unwrap_or_else(|| VoxTargetBounds {
        min: glam::UVec3::ZERO,
        max_exclusive: glam::UVec3::new(world_size.x.max(1), 1, world_size.z.max(1)),
    });
    let min = bounds.min.as_vec3();
    let max = bounds.max_exclusive.as_vec3().max(min + glam::Vec3::ONE);
    let extent = max - min;
    let center = (min + max) * 0.5;
    let horizontal_span = extent.x.max(extent.z);
    let entry_depth = (extent.z * 0.25)
        .clamp(48.0, 640.0)
        .min((extent.z * 0.45).max(1.0));
    let lookahead = (horizontal_span * 0.18)
        .clamp(160.0, 512.0)
        .min((extent.z - entry_depth).max(1.0));
    let position_z = min.z + entry_depth;
    let target = glam::Vec3::new(
        center.x,
        min.y + (extent.y * 0.35).clamp(24.0, 128.0),
        (position_z + lookahead).min(max.z - 1.0),
    );
    let position = glam::Vec3::new(
        center.x,
        target.y + (extent.y * 0.22).clamp(24.0, 160.0),
        position_z,
    );
    let to_target = target - position;
    let focal_distance = to_target.length().max(1.0);
    let forward = to_target.normalize_or_zero();

    Camera {
        position,
        forward: if forward.length_squared() > 0.0 {
            forward
        } else {
            glam::Vec3::Z
        },
        up: glam::Vec3::Y,
        fov_y_radians: std::f32::consts::FRAC_PI_4,
        aperture_radius: 0.0,
        focal_distance,
    }
}

fn align_fly_controller_to_camera(rig: &mut CameraRig) {
    let forward = rig.camera.forward.normalize_or_zero();
    if forward.length_squared() == 0.0 {
        return;
    }
    rig.controller.pitch = forward.y.clamp(-1.0, 1.0).asin();
    rig.controller.yaw = forward.x.atan2(forward.z);
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
            render_runtime: None,
            ucvh: None,
            lighting_settings: LightingSettings::default(),
            rt_settings: RtSettings::default(),
            area_restir_settings: AreaRestirSettings::default(),
            restir_di_settings: RestirDiSettings::default(),
            window_descriptor: WindowDescriptor::default(),
            window: None,
            window_id: None,
            #[cfg(not(target_os = "android"))]
            egui_ctx: None,
            #[cfg(not(target_os = "android"))]
            egui_state: None,
            #[cfg(not(target_os = "android"))]
            editor_ui: None,
            #[cfg(not(target_os = "android"))]
            pending_egui_textures_delta: egui::TexturesDelta::default(),
            initialized: false,
            touch_look: TouchLookState::default(),
            last_frame_time: None,
            rendered_frames: 0,
            exit_after_frames: parse_exit_after_frames(),
            camera_path: CameraPathConfig::from_env(),
            default_scene_target_bounds: None,
            default_scene_load: None,
            next_redraw_at: std::time::Instant::now(),
        }
    }

    fn restir_di_vpt_enabled(&self) -> bool {
        self.restir_di_settings.enabled
    }

    fn area_restir_vpt_enabled(&self) -> bool {
        self.area_restir_settings.enabled
    }

    fn runtime_settings(&self) -> RuntimeSettings {
        RuntimeSettings {
            lighting: self.lighting_settings,
            rt: self.rt_settings,
            restir_di: self.restir_di_settings,
            area_restir: self.area_restir_settings,
        }
    }

    fn resize_render_runtime(&mut self, width: u32, height: u32) -> Result<()> {
        let settings = self.runtime_settings();
        let restir_di_enabled = self.restir_di_vpt_enabled();
        let area_restir_enabled = self.area_restir_vpt_enabled();
        if let Some(runtime) = self.render_runtime.as_mut() {
            runtime.resize(
                width,
                height,
                self.ucvh.as_ref(),
                settings,
                restir_di_enabled,
                area_restir_enabled,
            )?;
        }
        Ok(())
    }

    fn redraw_interval(&self) -> std::time::Duration {
        if self.default_scene_load.is_some() {
            DEFAULT_SCENE_LOADING_REDRAW_INTERVAL
        } else {
            ACTIVE_REDRAW_INTERVAL
        }
    }

    fn start_default_scene_load(&mut self) {
        if self.ucvh.is_some() || self.default_scene_load.is_some() {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        self.default_scene_load = Some(receiver);
        std::thread::Builder::new()
            .name("default-scene-loader".to_string())
            .spawn(move || {
                let start = std::time::Instant::now();
                let mut ucvh = Ucvh::new(generator::default_scene_ucvh_config());
                let scene_result = generator::generate_default_scene(&mut ucvh);
                ucvh.rebuild_hierarchy();
                let elapsed = start.elapsed();
                let _ = sender.send(DefaultSceneLoadMessage {
                    ucvh,
                    scene_result,
                    elapsed,
                });
            })
            .expect("default scene loader thread should spawn");
        tracing::info!("started default scene load on background thread");
    }

    fn poll_default_scene_load(&mut self) {
        let Some(result) = self
            .default_scene_load
            .as_ref()
            .map(|receiver| receiver.try_recv())
        else {
            return;
        };
        match result {
            Ok(message) => {
                self.default_scene_load = None;
                self.apply_default_scene_load(message);
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.default_scene_load = None;
                tracing::error!("default scene loader stopped before sending a scene");
            }
        }
    }

    fn apply_default_scene_load(&mut self, message: DefaultSceneLoadMessage) {
        let scene_target_bounds = message
            .scene_result
            .teardown_zip_stats
            .as_ref()
            .and_then(|stats| stats.target_bounds)
            .or_else(|| {
                message
                    .scene_result
                    .vox_stats
                    .as_ref()
                    .and_then(|stats| stats.target_bounds)
            });
        self.log_default_scene_load(&message.scene_result, &message.ucvh, message.elapsed);
        self.default_scene_target_bounds = scene_target_bounds;
        self.ucvh = Some(message.ucvh);
        if self.camera_path.is_none() {
            let world_size = self
                .ucvh
                .as_ref()
                .map(|ucvh| ucvh.config.world_size)
                .unwrap_or(generator::DEFAULT_SCENE_WORLD_SIZE);
            if let Some(rig) = self.world.resource_mut::<CameraRig>() {
                rig.camera = camera_for_scene_bounds(self.default_scene_target_bounds, world_size);
                align_fly_controller_to_camera(rig);
            }
        }
    }

    fn log_default_scene_load(
        &self,
        scene_result: &generator::DefaultSceneLoadResult,
        ucvh: &Ucvh,
        elapsed: std::time::Duration,
    ) {
        match scene_result.kind {
            generator::DefaultSceneKind::TeardownZip => {
                let default_stats = Default::default();
                let stats = scene_result
                    .teardown_zip_stats
                    .as_ref()
                    .unwrap_or(&default_stats);
                tracing::info!(
                    path = %generator::default_zip_map_path().display(),
                    bricks = scene_result.brick_count,
                    input_voxels = stats.input_voxels,
                    written_voxels = stats.written_voxels,
                    unique_written_voxels = stats.unique_written_voxels,
                    total_voxels = ucvh.pool.allocated_count() as u64 * 512,
                    elapsed_ms = elapsed.as_millis(),
                    "loaded default teardown zip scene"
                );
            }
            generator::DefaultSceneKind::VoxFile => {
                let default_stats = Default::default();
                let stats = scene_result.vox_stats.as_ref().unwrap_or(&default_stats);
                tracing::info!(
                    path = generator::DEFAULT_VOX_MAP_PATH,
                    bricks = scene_result.brick_count,
                    input_voxels = stats.input_voxels,
                    written_voxels = stats.written_voxels,
                    unique_written_voxels = stats.unique_written_voxels,
                    total_voxels = ucvh.pool.allocated_count() as u64 * 512,
                    elapsed_ms = elapsed.as_millis(),
                    "loaded default vox scene"
                );
            }
            generator::DefaultSceneKind::CheckerboardFallback => {
                tracing::warn!(
                    path = generator::DEFAULT_VOX_MAP_PATH,
                    error = ?scene_result.vox_error,
                    bricks = scene_result.brick_count,
                    total_voxels = ucvh.pool.allocated_count() as u64 * 512,
                    elapsed_ms = elapsed.as_millis(),
                    "default vox scene unavailable; generated checkerboard fallback"
                );
            }
        }
    }

    fn update_camera(&mut self, dt: f32) {
        if let Some(camera_path) = self.camera_path {
            if let Some(rig) = self.world.resource_mut::<CameraRig>() {
                apply_camera_path(rig, camera_path, self.rendered_frames);
            }
            return;
        }

        // Clone InputState (it's Copy) to avoid borrow conflicts
        let input = match self.world.resource::<InputState>() {
            Some(input) => *input,
            None => return,
        };

        if let Some(rig) = self.world.resource_mut::<CameraRig>() {
            update_fly_camera(rig, input, dt);
        }
    }

    fn capture_camera_path_metadata(&self) -> CaptureCameraPathMetadata {
        self.camera_path
            .map_or_else(CaptureCameraPathMetadata::default, |path| {
                CaptureCameraPathMetadata {
                    path: path.path_name().to_owned(),
                    center: path.center_csv(),
                    radius: path.radius,
                    height: path.height,
                    period_frames: path.period_frames,
                }
            })
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
            None => {
                let light = DirectionalLight::default();
                (light.direction, light.intensity)
            }
        }
    }

    fn current_elapsed_seconds(&self) -> f32 {
        self.world
            .resource::<Time>()
            .map_or(0.0, |time| time.elapsed_seconds)
    }

    #[cfg(not(target_os = "android"))]
    fn initialize_editor_ui(&mut self, window: &Window) {
        let egui_ctx = egui::Context::default();
        let font_report = configure_editor_fonts(std::path::Path::new("assets").join("fonts"));
        for warning in &font_report.warnings {
            tracing::warn!(
                font = warning.font_name,
                searched_paths = ?warning.searched_paths,
                "optional editor font asset missing; using egui fallback"
            );
        }
        egui_ctx.set_fonts(font_report.fonts);
        egui_ctx.set_style(configure_editor_style());
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );

        self.egui_ctx = Some(egui_ctx);
        self.egui_state = Some(egui_state);
        self.editor_ui = Some(EditorUi::new());
    }

    #[cfg(not(target_os = "android"))]
    fn process_egui_window_event(&mut self, event: &WindowEvent) -> UiInputCapture {
        let Some(window) = self.window.as_ref() else {
            return UiInputCapture::default();
        };
        let Some(egui_state) = self.egui_state.as_mut() else {
            return UiInputCapture::default();
        };
        let response = egui_state.on_window_event(window, event);
        let Some(egui_ctx) = self.egui_ctx.as_ref() else {
            return UiInputCapture {
                consumed: response.consumed,
                ..UiInputCapture::default()
            };
        };
        UiInputCapture {
            consumed: response.consumed,
            wants_keyboard_input: egui_ctx.wants_keyboard_input(),
        }
    }

    #[cfg(target_os = "android")]
    fn process_egui_window_event(&mut self, _event: &WindowEvent) -> UiInputCapture {
        UiInputCapture::default()
    }

    #[cfg(not(target_os = "android"))]
    fn build_egui_frame(
        &mut self,
        camera: VptCameraFrame,
        viewport_extent: [u32; 2],
    ) -> Option<EguiFrame> {
        let window = self.window.as_ref()?;
        let egui_ctx = self.egui_ctx.as_ref()?;
        let egui_state = self.egui_state.as_mut()?;
        let runtime_status = self.render_runtime.as_ref().map(RenderRuntime::status);
        let editor_ui = self.editor_ui.as_mut()?;
        let raw_input = egui_state.take_egui_input(window);
        let full_output = egui_ctx.run(raw_input, |ctx| {
            editor_ui.show(
                ctx,
                EditorUiFrameState {
                    lighting: &mut self.lighting_settings,
                    rt: &mut self.rt_settings,
                    restir_di: &mut self.restir_di_settings,
                    area_restir: &mut self.area_restir_settings,
                    runtime_status,
                    camera,
                    viewport_extent,
                    rendered_frames: self.rendered_frames,
                },
            );
        });
        egui_state.handle_platform_output(window, full_output.platform_output);
        let font_texture_ready = self
            .render_runtime
            .as_ref()
            .is_some_and(RenderRuntime::egui_font_texture_ready);
        let mut textures_delta = full_output.textures_delta;
        if font_texture_ready {
            self.pending_egui_textures_delta.clear();
        } else {
            self.pending_egui_textures_delta.append(textures_delta);
            textures_delta = self.pending_egui_textures_delta.clone();
        }
        let clipped_primitives =
            egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        Some(EguiFrame {
            clipped_primitives,
            textures_delta,
            pixels_per_point: full_output.pixels_per_point,
        })
    }

    #[cfg(target_os = "android")]
    fn build_egui_frame(
        &mut self,
        _camera: VptCameraFrame,
        _viewport_extent: [u32; 2],
    ) -> Option<()> {
        None
    }

    fn set_camera_look(&mut self, pressed: bool) {
        if let Some(input) = self.world.resource_mut::<InputState>() {
            input.right_mouse_held = pressed;
        }
        if let Some(window) = &self.window {
            if pressed {
                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                window.set_cursor_visible(false);
            } else {
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            }
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
        let camera = self.current_vpt_camera_frame();
        let camera_path = self.capture_camera_path_metadata();
        let viewport_extent = self
            .window
            .as_ref()
            .map(|window| {
                let size = window.inner_size();
                [size.width, size.height]
            })
            .unwrap_or([0, 0]);
        #[cfg(not(target_os = "android"))]
        let egui_frame = self.build_egui_frame(camera, viewport_extent);
        let (sun_direction, sun_intensity) = self.current_sun_light();
        let elapsed_seconds = self.current_elapsed_seconds();
        let settings = self.runtime_settings();
        if let Some(runtime) = self.render_runtime.as_mut() {
            runtime.render_frame(RenderFrameInput {
                camera,
                camera_path,
                sun_direction,
                sun_intensity,
                elapsed_seconds,
                settings,
                restir_di_enabled,
                area_restir_enabled,
                ucvh: self.ucvh.as_mut(),
                #[cfg(not(target_os = "android"))]
                egui_frame,
            })?;
        }
        if let Some(input) = self.world.resource_mut::<InputState>() {
            input.clear_per_frame();
        }

        self.schedule
            .run_stage(Stage::ExecuteRender, &mut self.world)?;
        Ok(())
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
        let rt_settings_result = RtSettings::from_env();
        for warning in &rt_settings_result.warnings {
            tracing::warn!(
                variable = warning.variable,
                value = %warning.value,
                expected = warning.expected,
                "invalid RT setting override; using default value"
            );
        }
        self.rt_settings = rt_settings_result.settings;
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

        self.start_default_scene_load();

        let render_runtime =
            match RenderRuntime::new(&window, self.runtime_settings(), self.ucvh.as_ref()) {
                Ok(runtime) => runtime,
                Err(error) => {
                    tracing::error!(%error, "failed to initialize render runtime");
                    event_loop.exit();
                    return;
                }
            };
        self.render_runtime = Some(render_runtime);
        #[cfg(not(target_os = "android"))]
        self.initialize_editor_ui(&window);
        self.window = Some(window);
        self.window_id = Some(window_id);

        if !self.initialized {
            if let Err(error) = self.schedule.run_stage(Stage::Startup, &mut self.world) {
                tracing::error!(%error, "startup stage failed");
                event_loop.exit();
                return;
            }
            self.initialized = true;
            if self.camera_path.is_none() {
                let world_size = self
                    .ucvh
                    .as_ref()
                    .map(|ucvh| ucvh.config.world_size)
                    .unwrap_or(generator::DEFAULT_SCENE_WORLD_SIZE);
                if let Some(rig) = self.world.resource_mut::<CameraRig>() {
                    rig.camera =
                        camera_for_scene_bounds(self.default_scene_target_bounds, world_size);
                    align_fly_controller_to_camera(rig);
                }
            }
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
        let ui_capture = self.process_egui_window_event(&event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                // Skip rendering when minimized (zero-size window)
                if let Some(window) = &self.window {
                    if !should_request_redraw_for_size(window.inner_size()) {
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
                    clear_input_for_inactive_window(
                        self.world.resource_mut::<InputState>(),
                        &mut self.touch_look,
                        self.window.as_ref(),
                    );
                    return; // minimized, skip resize
                }
                if let Err(error) = self.resize_render_runtime(size.width, size.height) {
                    tracing::error!(%error, "failed to resize render runtime");
                    event_loop.exit();
                    return;
                }
                tracing::debug!(width = size.width, height = size.height, "window resized");
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if !camera_should_receive_input(CameraInputEventKind::Keyboard, ui_capture) {
                    return;
                }
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
                if !camera_should_receive_input(CameraInputEventKind::Pointer, ui_capture) {
                    if button == winit::event::MouseButton::Right
                        && state == winit::event::ElementState::Released
                    {
                        self.set_camera_look(false);
                    }
                    return;
                }
                if button == winit::event::MouseButton::Right {
                    let pressed = state == winit::event::ElementState::Pressed;
                    self.set_camera_look(pressed);
                }
            }
            WindowEvent::Touch(touch) => {
                if !camera_should_receive_input(CameraInputEventKind::Pointer, ui_capture) {
                    return;
                }
                let (world, touch_look) = (&mut self.world, &mut self.touch_look);
                if let Some(input) = world.resource_mut::<InputState>() {
                    touch_look.handle_touch(input, touch);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !camera_should_receive_input(CameraInputEventKind::Pointer, ui_capture) {
                    return;
                }
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 120.0,
                };
                if let Some(input) = self.world.resource_mut::<InputState>() {
                    input.scroll_delta += scroll;
                }
            }
            WindowEvent::Focused(false) => {
                clear_input_for_inactive_window(
                    self.world.resource_mut::<InputState>(),
                    &mut self.touch_look,
                    self.window.as_ref(),
                );
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
        if let DeviceEvent::MouseMotion { delta } = event
            && let Some(input) = self.world.resource_mut::<InputState>()
            && input.right_mouse_held
        {
            input.mouse_dx += delta.0 as f32;
            input.mouse_dy += delta.1 as f32;
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.poll_default_scene_load();
        let now = std::time::Instant::now();
        let redraw_interval = self.redraw_interval();
        if let Some(window) = &self.window
            && should_request_redraw_for_size(window.inner_size())
            && now >= self.next_redraw_at
        {
            window.request_redraw();
            self.next_redraw_at = now + redraw_interval;
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_redraw_at));
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
    use crate::scene::camera::{Camera, CameraPathConfig, CameraPathKind};
    use crate::scene::components::CameraRig;
    use crate::scene::light::DirectionalLight;
    use crate::voxel::vox_loader::VoxTargetBounds;
    use winit::event::{DeviceId, Touch, TouchPhase};

    fn touch_event(id: u64, phase: TouchPhase, x: f64, y: f64) -> Touch {
        Touch {
            device_id: DeviceId::dummy(),
            phase,
            location: winit::dpi::PhysicalPosition::new(x, y),
            force: None,
            id,
        }
    }

    #[test]
    fn camera_input_gate_blocks_consumed_ui_events() {
        let capture = UiInputCapture {
            consumed: true,
            wants_keyboard_input: false,
        };

        assert!(!camera_should_receive_input(
            CameraInputEventKind::Pointer,
            capture
        ));
        assert!(!camera_should_receive_input(
            CameraInputEventKind::Keyboard,
            capture
        ));
    }

    #[test]
    fn camera_input_gate_allows_unconsumed_pointer_events_for_viewport_control() {
        assert!(camera_should_receive_input(
            CameraInputEventKind::Pointer,
            UiInputCapture::default()
        ));
        assert!(camera_should_receive_input(
            CameraInputEventKind::Keyboard,
            UiInputCapture::default()
        ));
    }

    #[test]
    fn camera_input_gate_blocks_keyboard_focus_independently() {
        assert!(!camera_should_receive_input(
            CameraInputEventKind::Keyboard,
            UiInputCapture {
                wants_keyboard_input: true,
                ..UiInputCapture::default()
            }
        ));
    }

    #[test]
    fn redraw_requests_are_suppressed_for_minimized_windows() {
        assert!(should_request_redraw_for_size(
            winit::dpi::PhysicalSize::new(1, 1)
        ));
        assert!(!should_request_redraw_for_size(
            winit::dpi::PhysicalSize::new(0, 1)
        ));
        assert!(!should_request_redraw_for_size(
            winit::dpi::PhysicalSize::new(1, 0)
        ));
        assert!(!should_request_redraw_for_size(
            winit::dpi::PhysicalSize::new(0, 0)
        ));
    }

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
    fn automatic_camera_path_overrides_manual_fly_input() {
        let mut app = RevolumetricApp::new();
        app.camera_path = Some(CameraPathConfig {
            kind: CameraPathKind::Orbit,
            center: glam::Vec3::new(64.0, 32.0, 64.0),
            radius: 32.0,
            height: 48.0,
            period_frames: 4,
        });
        app.rendered_frames = 1;
        app.world.insert_resource(CameraRig::default());
        app.world.insert_resource(InputState {
            move_forward: 1.0,
            move_right: 1.0,
            move_up: 1.0,
            ..InputState::default()
        });

        app.update_camera(1.0);

        let rig = app.world.resource::<CameraRig>().expect("rig should exist");
        assert!((rig.camera.position - glam::Vec3::new(64.0, 48.0, 96.0)).length() < 1e-4);
        assert!(
            (rig.camera.forward
                - (app.camera_path.unwrap().center - rig.camera.position).normalize())
            .length()
                < 1e-5
        );
    }

    #[test]
    fn camera_for_default_scene_bounds_starts_near_the_loaded_map() {
        let bounds = VoxTargetBounds {
            min: glam::UVec3::new(4, 4, 4),
            max_exclusive: glam::UVec3::new(1020, 128, 524),
        };

        let camera = camera_for_scene_bounds(Some(bounds), glam::UVec3::new(1024, 512, 1024));
        let center = glam::Vec3::new(512.0, 66.0, 264.0);

        assert!((camera.position.x - center.x).abs() < 1.0);
        assert!(camera.position.y > bounds.min.y as f32);
        assert!(camera.position.z > bounds.min.z as f32);
        assert!(camera.position.z < center.z);
        assert!(camera.forward.z > 0.6);
        assert!(camera.forward.y < 0.0);
        assert!(camera.focal_distance <= 800.0);
    }

    #[test]
    fn camera_for_scene_bounds_uses_world_center_when_vox_bounds_are_missing() {
        let camera = camera_for_scene_bounds(None, glam::UVec3::new(1024, 512, 1024));

        assert!((camera.position.x - 512.0).abs() < 1.0);
        assert!(camera.position.y > 0.0);
        assert!(camera.position.z > 0.0);
        assert!(camera.position.z <= 256.0);
        assert!(camera.forward.z > 0.0);
        assert!(camera.forward.y < 0.0);
    }

    #[test]
    fn camera_for_vintessa_sized_bounds_starts_inside_loaded_map_not_front_cutaway() {
        let bounds = VoxTargetBounds {
            min: glam::UVec3::new(4, 4, 4),
            max_exclusive: glam::UVec3::new(3850, 525, 2551),
        };

        let camera = camera_for_scene_bounds(Some(bounds), glam::UVec3::new(4096, 768, 3072));

        assert!(
            camera.focal_distance <= 800.0,
            "initial camera should start near the playable map, not at a whole-map overview distance"
        );
        assert!(
            camera.position.z > bounds.min.z as f32 + 256.0,
            "initial camera must start beyond the imported terrain front edge instead of looking at its cutaway"
        );
        assert!(camera.position.z < 900.0);
        assert!(camera.forward.z > 0.6);
        assert!(camera.forward.y < 0.0);
    }

    #[test]
    fn app_resumed_does_not_block_window_creation_on_default_scene_load() {
        let source = crate::render::source_checks::read_source("src/app.rs");
        let resumed = source
            .split("fn resumed(&mut self")
            .nth(1)
            .expect("resumed should exist")
            .split("fn window_event")
            .next()
            .expect("resumed should end before window_event");

        assert!(
            !resumed.contains("generate_default_scene("),
            "window creation must not synchronously import Vintessa on the UI thread"
        );
        assert!(
            resumed.contains("start_default_scene_load"),
            "resumed should kick off an asynchronous default scene load"
        );
    }

    #[test]
    fn app_throttles_redraws_instead_of_busy_polling_during_startup() {
        let source = crate::render::source_checks::read_source("src/app.rs");
        let run = source
            .split("pub fn run()")
            .nth(1)
            .expect("run should exist")
            .split("#[cfg(target_os = \"android\")]")
            .next()
            .expect("desktop run should end before android run");
        let about_to_wait = source
            .split("fn about_to_wait")
            .nth(1)
            .expect("about_to_wait should exist")
            .split("fn init_tracing")
            .next()
            .expect("about_to_wait should end before init_tracing");
        let process_egui = source
            .split("fn process_egui_window_event")
            .nth(1)
            .expect("process_egui_window_event should exist")
            .split("#[cfg(target_os = \"android\")]")
            .next()
            .expect("desktop process_egui_window_event should end before android variant");

        assert!(
            !run.contains("ControlFlow::Poll"),
            "startup must not busy-poll while Vintessa is building on a background thread"
        );
        assert!(
            about_to_wait.contains("set_control_flow(ControlFlow::WaitUntil"),
            "about_to_wait should cap redraw cadence instead of requesting unbounded frames"
        );
        assert!(
            about_to_wait.contains("now >= self.next_redraw_at")
                && about_to_wait.contains("self.next_redraw_at = now + redraw_interval"),
            "about_to_wait should request redraws only when the next paced frame is due"
        );
        assert!(
            !process_egui.contains("request_redraw()"),
            "egui repaint events must not bypass the paced redraw scheduler during startup"
        );
    }

    #[test]
    fn app_uses_slower_empty_scene_redraws_while_default_scene_is_loading() {
        let source = crate::render::source_checks::read_source("src/app.rs");
        let about_to_wait = source
            .split("fn about_to_wait")
            .nth(1)
            .expect("about_to_wait should exist")
            .split("fn init_tracing")
            .next()
            .expect("about_to_wait should end before init_tracing");

        assert!(
            source.contains("DEFAULT_SCENE_LOADING_REDRAW_INTERVAL"),
            "app should define a slower redraw interval while the background Vintessa loader owns CPU"
        );
        assert!(
            about_to_wait.contains("self.redraw_interval()"),
            "about_to_wait should choose a startup-aware redraw interval instead of always using 16ms"
        );
    }

    #[test]
    fn app_buffers_egui_texture_delta_until_renderer_font_texture_is_ready() {
        let source = crate::render::source_checks::read_source("src/app.rs");
        let app_struct = source
            .split("struct RevolumetricApp")
            .nth(1)
            .expect("app struct should exist")
            .split("struct DefaultSceneLoadMessage")
            .next()
            .expect("app struct should end before default scene message");
        let build_egui_frame = source
            .split("fn build_egui_frame")
            .nth(1)
            .expect("build_egui_frame should exist")
            .split("#[cfg(target_os = \"android\")]")
            .next()
            .expect("desktop build_egui_frame should end before android variant");

        assert!(
            app_struct.contains("pending_egui_textures_delta"),
            "app should retain egui texture uploads while render passes are not ready"
        );
        assert!(
            build_egui_frame.contains("egui_font_texture_ready"),
            "app should keep replaying retained font texture deltas until the renderer confirms upload"
        );
    }

    #[test]
    fn startup_wait_paths_do_not_emit_warning_spam() {
        let vpt = crate::render::source_checks::read_source("src/render/vpt_pipeline.rs");
        let rt = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");

        for message in [
            "skipping UCVH render passes until GPU upload succeeds",
            "render graph produced no presentable output; clearing swapchain fallback",
            "rendering RT fallback output until UCVH data is ready",
            "skipping RT graph until required passes are initialized",
            "skipping RT surface pass creation without UCVH GPU descriptors",
            "skipping RT surface trace without built TLAS and AABB buffer",
            "skipping RT ReSTIR-DI pass creation without CPU UCVH scene",
            "skipping RT ReSTIR-GI pass creation without UCVH GPU descriptors",
        ] {
            let forbidden = format!("tracing::warn!(\"{message}");
            assert!(
                !vpt.contains(&forbidden) && !rt.contains(&forbidden),
                "transient startup wait message should not warn every paced frame: {message}"
            );
        }
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
        let default_light = DirectionalLight::default();

        assert!((direction - default_light.direction).length() < 1e-6);
        assert_eq!(intensity, default_light.intensity);
    }

    #[test]
    fn current_elapsed_seconds_defaults_to_zero_without_time_resource() {
        let app = RevolumetricApp::new();
        assert_eq!(app.current_elapsed_seconds(), 0.0);
    }

    #[test]
    fn app_passes_rt_settings_to_editor_ui_frame_state() {
        let source = crate::render::source_checks::read_source("src/app.rs");
        let build_egui_frame = source
            .split("fn build_egui_frame")
            .nth(1)
            .expect("build_egui_frame should exist")
            .split("#[cfg(target_os = \"android\")]")
            .next()
            .expect("desktop build_egui_frame should end before android variant");

        assert!(build_egui_frame.contains("rt: &mut self.rt_settings"));
    }

    #[test]
    fn app_passes_runtime_status_to_editor_ui_frame_state() {
        let source = crate::render::source_checks::read_source("src/app.rs");
        let build_egui_frame = source
            .split("fn build_egui_frame")
            .nth(1)
            .expect("build_egui_frame should exist")
            .split("#[cfg(target_os = \"android\")]")
            .next()
            .expect("desktop build_egui_frame should end before android variant");

        assert!(build_egui_frame.contains(
            "let runtime_status = self.render_runtime.as_ref().map(RenderRuntime::status);"
        ));
        assert!(build_egui_frame.contains("runtime_status,"));
    }

    #[test]
    fn touch_look_tracks_one_finger_drag_and_ignores_secondary_touch() {
        let mut touch_look = TouchLookState::default();
        let mut input = InputState::default();

        touch_look.handle_touch(&mut input, touch_event(7, TouchPhase::Started, 10.0, 20.0));
        touch_look.handle_touch(&mut input, touch_event(7, TouchPhase::Moved, 14.0, 17.0));
        touch_look.handle_touch(
            &mut input,
            touch_event(8, TouchPhase::Started, 100.0, 100.0),
        );
        touch_look.handle_touch(&mut input, touch_event(8, TouchPhase::Moved, 150.0, 120.0));

        assert_eq!(input.mouse_dx, 4.0);
        assert_eq!(input.mouse_dy, -3.0);
        assert_eq!(touch_look.active_touch_id, Some(7));

        touch_look.handle_touch(&mut input, touch_event(7, TouchPhase::Ended, 14.0, 17.0));

        assert!(touch_look.active_touch_id.is_none());
        assert!(touch_look.last_position.is_none());
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

        assert!(app_struct.contains("render_runtime: Option<RenderRuntime>"));
        assert!(!app_struct.contains("vpt_pipeline: VptRuntimePipeline"));
        assert!(!app_struct.contains("vpt_pass: Option<VptPass>"));
        assert!(!app_struct.contains("vpt_surface_pass: Option<VptSurfacePass>"));
        assert!(!app_struct.contains("postprocess_pass: Option<PostprocessPass>"));
        let record_call = [
            "self",
            ".",
            "vpt_pipeline",
            ".",
            "record_and_execute_frame",
            "(",
        ]
        .concat();
        assert!(!source.contains(&record_call));
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
