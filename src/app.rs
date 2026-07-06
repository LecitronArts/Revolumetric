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
use crate::scene::camera::update_fly_camera;
use crate::scene::components::CameraRig;

use crate::ecs::schedule::{Schedule, Stage};
use crate::ecs::world::World;
use crate::platform::time::Time;
use crate::platform::window::WindowDescriptor;
use crate::render::area_restir::{AreaRestirDebugView, AreaRestirSettings};
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
use crate::voxel::ucvh::{Ucvh, UcvhConfig};

pub fn run() -> Result<()> {
    init_tracing();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

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
    initialized: bool,
    touch_look: TouchLookState,
    last_frame_time: Option<std::time::Instant>,
    rendered_frames: u64,
    exit_after_frames: Option<u64>,
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
            initialized: false,
            touch_look: TouchLookState::default(),
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
        if response.repaint && should_request_redraw_for_size(window.inner_size()) {
            window.request_redraw();
        }
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
        let clipped_primitives =
            egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        Some(EguiFrame {
            clipped_primitives,
            textures_delta: full_output.textures_delta,
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
            self.ucvh = Some(ucvh);
        }

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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window
            && should_request_redraw_for_size(window.inner_size())
        {
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
