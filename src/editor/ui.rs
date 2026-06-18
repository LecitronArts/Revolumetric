//! Editor shell state and setting controls.

use crate::render::area_restir::{AreaRestirDebugView, AreaRestirSettings};
use crate::render::restir_di::{RestirDiDebugView, RestirDiSettings};
use crate::render::scene_ubo::{
    LightingDebugView, LightingSettings, VptDebugView, VptDenoiserMode,
};
use crate::render::vpt_pipeline::VptCameraFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorPanel {
    Scene,
    Render,
    Restir,
    Debug,
}

#[derive(Debug)]
pub struct EditorUi {
    pub visible: bool,
    pub selected_panel: EditorPanel,
    pub show_advanced: bool,
    pub console_lines: Vec<String>,
}

pub struct EditorUiFrameState<'a> {
    pub lighting: &'a mut LightingSettings,
    pub restir_di: &'a mut RestirDiSettings,
    pub area_restir: &'a mut AreaRestirSettings,
    pub camera: VptCameraFrame,
    pub viewport_extent: [u32; 2],
    pub rendered_frames: u64,
}

impl Default for EditorUi {
    fn default() -> Self {
        Self {
            visible: true,
            selected_panel: EditorPanel::Render,
            show_advanced: false,
            console_lines: vec!["editor shell initialized".to_owned()],
        }
    }
}

impl EditorUi {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_console_line(&mut self, line: impl Into<String>) {
        self.console_lines.push(line.into());
        const MAX_CONSOLE_LINES: usize = 128;
        let overflow = self.console_lines.len().saturating_sub(MAX_CONSOLE_LINES);
        if overflow > 0 {
            self.console_lines.drain(0..overflow);
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, frame: EditorUiFrameState<'_>) {
        if !self.visible {
            return;
        }

        let EditorUiFrameState {
            lighting,
            restir_di,
            area_restir,
            camera,
            viewport_extent,
            rendered_frames,
        } = frame;

        self.show_top_bar(ctx, lighting, viewport_extent, rendered_frames);
        self.show_left_rail(ctx, camera);
        self.show_inspector(ctx, lighting, restir_di, area_restir);
        self.show_console(ctx, lighting, restir_di, area_restir);
        self.show_viewport_overlay(ctx, camera, lighting);
    }

    fn show_top_bar(
        &mut self,
        ctx: &egui::Context,
        lighting: &mut LightingSettings,
        viewport_extent: [u32; 2],
        rendered_frames: u64,
    ) {
        egui::TopBottomPanel::top("editor_top_bar")
            .exact_height(36.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.strong("REVOLUMETRIC");
                    ui.separator();
                    ui.label("VPT Editor");
                    ui.separator();
                    ui.label(format!("{} x {}", viewport_extent[0], viewport_extent[1]));
                    ui.label(format!("frame {rendered_frames}"));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.checkbox(&mut self.show_advanced, "Advanced");
                        egui::ComboBox::from_id_salt("top_denoiser_mode")
                            .selected_text(denoiser_label(lighting.denoiser_mode))
                            .show_ui(ui, |ui| {
                                denoiser_combo(ui, &mut lighting.denoiser_mode);
                            });
                    });
                });
            });
    }

    fn show_left_rail(&mut self, ctx: &egui::Context, camera: VptCameraFrame) {
        egui::SidePanel::left("editor_left_rail")
            .resizable(false)
            .exact_width(220.0)
            .show(ctx, |ui| {
                ui.heading("Scene");
                ui.add_space(4.0);
                selectable_panel(
                    ui,
                    &mut self.selected_panel,
                    EditorPanel::Scene,
                    "Scene Graph",
                );
                selectable_panel(
                    ui,
                    &mut self.selected_panel,
                    EditorPanel::Render,
                    "Renderer",
                );
                selectable_panel(
                    ui,
                    &mut self.selected_panel,
                    EditorPanel::Restir,
                    "Sampling",
                );
                selectable_panel(ui, &mut self.selected_panel, EditorPanel::Debug, "Debug");

                ui.separator();
                ui.label(egui::RichText::new("Camera").monospace().strong());
                ui.monospace(format_vec3("pos", camera.position));
                ui.monospace(format_vec3("dir", camera.forward));
                ui.monospace(format!(
                    "fov {:>5.1} deg",
                    camera.fov_y_radians.to_degrees()
                ));
                ui.monospace(format!("aperture {:>5.3}", camera.aperture_radius));
                ui.monospace(format!("focus {:>6.2}", camera.focal_distance));
            });
    }

    fn show_inspector(
        &mut self,
        ctx: &egui::Context,
        lighting: &mut LightingSettings,
        restir_di: &mut RestirDiSettings,
        area_restir: &mut AreaRestirSettings,
    ) {
        egui::SidePanel::right("editor_inspector")
            .default_width(340.0)
            .width_range(300.0..=440.0)
            .show(ctx, |ui| {
                ui.heading("Inspector");
                ui.add_space(4.0);
                match self.selected_panel {
                    EditorPanel::Scene => show_scene_panel(ui, lighting),
                    EditorPanel::Render => show_render_panel(ui, lighting),
                    EditorPanel::Restir => {
                        show_restir_panel(ui, lighting, restir_di, area_restir, self.show_advanced);
                    }
                    EditorPanel::Debug => show_debug_panel(ui, lighting, restir_di, area_restir),
                }
            });
    }

    fn show_console(
        &self,
        ctx: &egui::Context,
        lighting: &LightingSettings,
        restir_di: &RestirDiSettings,
        area_restir: &AreaRestirSettings,
    ) {
        egui::TopBottomPanel::bottom("editor_console")
            .resizable(true)
            .default_height(112.0)
            .height_range(72.0..=220.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Console");
                    ui.separator();
                    ui.monospace(format!("denoiser={}", lighting.denoiser_mode_name()));
                    ui.monospace(format!("restir_di={}", restir_di.enabled));
                    ui.monospace(format!("area_restir={}", area_restir.enabled));
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.console_lines {
                            ui.monospace(line);
                        }
                    });
            });
    }

    fn show_viewport_overlay(
        &self,
        ctx: &egui::Context,
        camera: VptCameraFrame,
        lighting: &LightingSettings,
    ) {
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let painter = ui.painter_at(rect);
                let overlay_rect = egui::Rect::from_min_size(
                    rect.left_top() + egui::vec2(18.0, 18.0),
                    egui::vec2(310.0, 96.0),
                );
                painter.rect_filled(
                    overlay_rect,
                    egui::Rounding::same(8.0),
                    egui::Color32::from_black_alpha(170),
                );
                painter.rect_stroke(
                    overlay_rect,
                    egui::Rounding::same(8.0),
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 78, 84)),
                );
                ui.allocate_new_ui(
                    egui::UiBuilder::new().max_rect(overlay_rect.shrink(12.0)),
                    |ui| {
                        ui.label(egui::RichText::new("Viewport").strong());
                        ui.monospace(format!(
                            "debug {}",
                            vpt_debug_label(lighting.vpt_debug_view)
                        ));
                        ui.monospace(format!(
                            "camera {:.1}, {:.1}, {:.1}",
                            camera.position.x, camera.position.y, camera.position.z
                        ));
                        ui.label("RMB drag to look, WASD/Space/Shift to move");
                    },
                );
            });
    }
}

pub fn clamp_vpt_max_bounces(value: u32) -> u32 {
    value.clamp(1, 8)
}

pub fn clamp_denoiser_atrous_iterations(value: u32) -> u32 {
    value.clamp(0, 5)
}

pub fn set_restir_di_enabled(settings: &mut RestirDiSettings, enabled: bool) {
    settings.enabled = enabled;
}

pub fn set_restir_di_debug_view(
    lighting: &mut LightingSettings,
    restir_di: &mut RestirDiSettings,
    area_restir: &mut AreaRestirSettings,
    debug_view: RestirDiDebugView,
) {
    restir_di.debug_view = debug_view;
    match restir_di_debug_to_vpt_debug_view(debug_view) {
        Some(vpt_debug_view) => {
            lighting.vpt_debug_view = vpt_debug_view;
            area_restir.debug_view = AreaRestirDebugView::Off;
        }
        None if debug_view != RestirDiDebugView::Off => {
            lighting.vpt_debug_view = VptDebugView::Final;
            area_restir.debug_view = AreaRestirDebugView::Off;
        }
        None if lighting.vpt_debug_view == VptDebugView::ReservoirWeight => {
            lighting.vpt_debug_view = VptDebugView::Final;
        }
        None => {}
    }
}

pub fn set_area_restir_enabled(settings: &mut AreaRestirSettings, enabled: bool) {
    settings.enabled = enabled;
}

pub fn restir_di_debug_to_vpt_debug_view(debug_view: RestirDiDebugView) -> Option<VptDebugView> {
    match debug_view {
        RestirDiDebugView::ReservoirWeight => Some(VptDebugView::ReservoirWeight),
        RestirDiDebugView::Off
        | RestirDiDebugView::LightId
        | RestirDiDebugView::Visibility
        | RestirDiDebugView::TemporalValid
        | RestirDiDebugView::SpatialNeighbors => None,
    }
}

pub fn vpt_debug_to_restir_di_debug_view(debug_view: VptDebugView) -> Option<RestirDiDebugView> {
    match debug_view {
        VptDebugView::ReservoirWeight => Some(RestirDiDebugView::ReservoirWeight),
        _ => None,
    }
}

pub fn set_area_restir_debug_view(
    lighting: &mut LightingSettings,
    restir_di: &mut RestirDiSettings,
    area_restir: &mut AreaRestirSettings,
    debug_view: AreaRestirDebugView,
) {
    area_restir.debug_view = debug_view;
    match area_restir_debug_to_vpt_debug_view(debug_view) {
        Some(vpt_debug_view) => {
            lighting.vpt_debug_view = vpt_debug_view;
            restir_di.debug_view = RestirDiDebugView::Off;
        }
        None if is_area_vpt_debug_view(lighting.vpt_debug_view) => {
            lighting.vpt_debug_view = VptDebugView::Final;
        }
        None => {}
    }
}

pub fn set_vpt_debug_view(
    lighting: &mut LightingSettings,
    restir_di: &mut RestirDiSettings,
    area_restir: &mut AreaRestirSettings,
    debug_view: VptDebugView,
) {
    lighting.vpt_debug_view = debug_view;

    if let Some(restir_debug) = vpt_debug_to_restir_di_debug_view(debug_view) {
        restir_di.debug_view = restir_debug;
    } else {
        restir_di.debug_view = RestirDiDebugView::Off;
    }

    if let Some(area_debug) = vpt_debug_to_area_restir_debug_view(debug_view) {
        area_restir.debug_view = area_debug;
    } else {
        area_restir.debug_view = AreaRestirDebugView::Off;
    }
}

pub fn area_restir_debug_to_vpt_debug_view(
    debug_view: AreaRestirDebugView,
) -> Option<VptDebugView> {
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

pub fn vpt_debug_to_area_restir_debug_view(
    debug_view: VptDebugView,
) -> Option<AreaRestirDebugView> {
    match debug_view {
        VptDebugView::AreaSubpixel => Some(AreaRestirDebugView::Subpixel),
        VptDebugView::AreaLens => Some(AreaRestirDebugView::Lens),
        VptDebugView::AreaWeight => Some(AreaRestirDebugView::Weight),
        VptDebugView::AreaHistoryValid => Some(AreaRestirDebugView::HistoryValid),
        VptDebugView::AreaRejection => Some(AreaRestirDebugView::Rejection),
        VptDebugView::AreaJacobian => Some(AreaRestirDebugView::Jacobian),
        _ => None,
    }
}

pub fn is_area_vpt_debug_view(debug_view: VptDebugView) -> bool {
    matches!(
        debug_view,
        VptDebugView::AreaSubpixel
            | VptDebugView::AreaLens
            | VptDebugView::AreaWeight
            | VptDebugView::AreaHistoryValid
            | VptDebugView::AreaRejection
            | VptDebugView::AreaJacobian
    )
}

fn selectable_panel(
    ui: &mut egui::Ui,
    selected: &mut EditorPanel,
    panel: EditorPanel,
    label: &str,
) {
    if ui.selectable_label(*selected == panel, label).clicked() {
        *selected = panel;
    }
}

fn show_scene_panel(ui: &mut egui::Ui, lighting: &mut LightingSettings) {
    ui.label("Lighting");
    ui.checkbox(&mut lighting.shadows_enabled, "Sun shadows");
    ui.checkbox(&mut lighting.skip_backface_shadows, "Skip backface shadows");
    ui.add(
        egui::Slider::new(&mut lighting.sun_angular_radius, 0.0001..=0.25)
            .text("sun angular radius")
            .logarithmic(false),
    );
    if !lighting.sun_angular_radius.is_finite() || lighting.sun_angular_radius <= 0.0 {
        lighting.sun_angular_radius = LightingSettings::default().sun_angular_radius;
    }
}

fn show_render_panel(ui: &mut egui::Ui, lighting: &mut LightingSettings) {
    ui.label("Path Tracing");
    ui.add(egui::Slider::new(&mut lighting.vpt_max_bounces, 1..=8).text("max bounces"));
    lighting.vpt_max_bounces = clamp_vpt_max_bounces(lighting.vpt_max_bounces);
    ui.add(
        egui::Slider::new(&mut lighting.exposure, 0.0..=8.0)
            .text("exposure")
            .logarithmic(false),
    );
    if !lighting.exposure.is_finite() || lighting.exposure < 0.0 {
        lighting.exposure = LightingSettings::default().exposure;
    }

    ui.separator();
    ui.label("Denoiser");
    egui::ComboBox::from_id_salt("inspector_denoiser_mode")
        .selected_text(denoiser_label(lighting.denoiser_mode))
        .show_ui(ui, |ui| {
            denoiser_combo(ui, &mut lighting.denoiser_mode);
        });
    ui.add(
        egui::Slider::new(&mut lighting.denoiser_atrous_iterations, 0..=5)
            .text("atrous iterations"),
    );
    lighting.denoiser_atrous_iterations =
        clamp_denoiser_atrous_iterations(lighting.denoiser_atrous_iterations);
}

fn show_restir_panel(
    ui: &mut egui::Ui,
    lighting: &mut LightingSettings,
    restir_di: &mut RestirDiSettings,
    area_restir: &mut AreaRestirSettings,
    show_advanced: bool,
) {
    ui.label("ReSTIR-DI");
    let mut restir_enabled = restir_di.enabled;
    if ui
        .checkbox(&mut restir_enabled, "Enable ReSTIR-DI")
        .changed()
    {
        set_restir_di_enabled(restir_di, restir_enabled);
    }
    ui.checkbox(&mut restir_di.temporal_enabled, "Temporal reuse");
    ui.checkbox(&mut restir_di.spatial_enabled, "Spatial reuse");
    ui.add(
        egui::Slider::new(&mut restir_di.initial_candidate_count, 1..=16)
            .text("initial candidates"),
    );
    ui.add(egui::Slider::new(&mut restir_di.spatial_sample_count, 0..=8).text("spatial samples"));
    ui.add(egui::Slider::new(&mut restir_di.history_length, 1..=64).text("history length"));
    if show_advanced {
        let mut restir_debug = restir_di.debug_view;
        egui::ComboBox::from_id_salt("restir_di_debug")
            .selected_text(restir_di_debug_label(restir_debug))
            .show_ui(ui, |ui| {
                restir_di_debug_combo(ui, &mut restir_debug);
            });
        if restir_debug != restir_di.debug_view {
            set_restir_di_debug_view(lighting, restir_di, area_restir, restir_debug);
        }
    }

    ui.separator();
    ui.label("Area ReSTIR");
    let mut area_enabled = area_restir.enabled;
    if ui
        .checkbox(&mut area_enabled, "Enable Area ReSTIR")
        .changed()
    {
        set_area_restir_enabled(area_restir, area_enabled);
    }
    ui.checkbox(&mut area_restir.temporal_enabled, "Temporal reuse");
    ui.checkbox(&mut area_restir.spatial_enabled, "Spatial reuse");
    ui.checkbox(&mut area_restir.subpixel_enabled, "Subpixel samples");
    ui.checkbox(&mut area_restir.lens_enabled, "Lens samples");
    ui.add(
        egui::Slider::new(&mut area_restir.initial_candidate_count, 1..=16)
            .text("initial candidates"),
    );
    ui.add(
        egui::Slider::new(&mut area_restir.spatial_sample_count, 0..=16).text("spatial samples"),
    );
    ui.add(egui::Slider::new(&mut area_restir.history_length, 1..=64).text("history length"));
    if show_advanced {
        ui.add(
            egui::Slider::new(&mut area_restir.normal_threshold, 0.0..=1.0)
                .text("normal threshold"),
        );
        ui.add(
            egui::Slider::new(&mut area_restir.depth_threshold, 0.0..=0.25).text("depth threshold"),
        );
        ui.add(
            egui::Slider::new(&mut area_restir.spatial_radius, 0.0..=96.0).text("spatial radius"),
        );
    }

    let mut area_debug = area_restir.debug_view;
    egui::ComboBox::from_id_salt("area_restir_debug")
        .selected_text(area_restir_debug_label(area_debug))
        .show_ui(ui, |ui| area_restir_debug_combo(ui, &mut area_debug));
    if area_debug != area_restir.debug_view {
        set_area_restir_debug_view(lighting, restir_di, area_restir, area_debug);
    }
}

fn show_debug_panel(
    ui: &mut egui::Ui,
    lighting: &mut LightingSettings,
    restir_di: &mut RestirDiSettings,
    area: &mut AreaRestirSettings,
) {
    ui.label("Lighting Debug");
    egui::ComboBox::from_id_salt("lighting_debug_view")
        .selected_text(lighting_debug_label(lighting.debug_view))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut lighting.debug_view, LightingDebugView::Final, "Final");
            ui.selectable_value(
                &mut lighting.debug_view,
                LightingDebugView::DirectDiffuse,
                "Direct Diffuse",
            );
            ui.selectable_value(
                &mut lighting.debug_view,
                LightingDebugView::Normal,
                "Normal",
            );
        });

    ui.separator();
    ui.label("VPT Debug");
    let mut vpt_debug = lighting.vpt_debug_view;
    egui::ComboBox::from_id_salt("vpt_debug_view")
        .selected_text(vpt_debug_label(vpt_debug))
        .show_ui(ui, |ui| {
            for (view, label) in VPT_DEBUG_OPTIONS {
                ui.selectable_value(&mut vpt_debug, *view, *label);
            }
        });
    if vpt_debug != lighting.vpt_debug_view {
        set_vpt_debug_view(lighting, restir_di, area, vpt_debug);
    }

    ui.separator();
    ui.label("ReSTIR-DI Debug");
    let mut restir_debug = restir_di.debug_view;
    egui::ComboBox::from_id_salt("debug_restir_di_view")
        .selected_text(restir_di_debug_label(restir_debug))
        .show_ui(ui, |ui| {
            restir_di_debug_combo(ui, &mut restir_debug);
        });
    if restir_debug != restir_di.debug_view {
        set_restir_di_debug_view(lighting, restir_di, area, restir_debug);
    }

    ui.separator();
    let mut area_debug = area.debug_view;
    egui::ComboBox::from_id_salt("debug_area_restir_bridge")
        .selected_text(area_restir_debug_label(area_debug))
        .show_ui(ui, |ui| area_restir_debug_combo(ui, &mut area_debug));
    if area_debug != area.debug_view {
        set_area_restir_debug_view(lighting, restir_di, area, area_debug);
    }
}

fn denoiser_combo(ui: &mut egui::Ui, mode: &mut VptDenoiserMode) {
    ui.selectable_value(mode, VptDenoiserMode::Off, "Off");
    ui.selectable_value(mode, VptDenoiserMode::Svgf, "SVGF");
    ui.selectable_value(mode, VptDenoiserMode::Relax, "NRD RELAX");
    ui.selectable_value(mode, VptDenoiserMode::Reblur, "NRD REBLUR");
}

fn restir_di_debug_combo(ui: &mut egui::Ui, debug_view: &mut RestirDiDebugView) {
    ui.selectable_value(debug_view, RestirDiDebugView::Off, "Off");
    ui.selectable_value(
        debug_view,
        RestirDiDebugView::ReservoirWeight,
        "Reservoir Weight",
    );
    ui.selectable_value(debug_view, RestirDiDebugView::LightId, "Light ID");
    ui.selectable_value(debug_view, RestirDiDebugView::Visibility, "Visibility");
    ui.selectable_value(
        debug_view,
        RestirDiDebugView::TemporalValid,
        "Temporal Valid",
    );
    ui.selectable_value(
        debug_view,
        RestirDiDebugView::SpatialNeighbors,
        "Spatial Neighbors",
    );
}

fn area_restir_debug_combo(ui: &mut egui::Ui, debug_view: &mut AreaRestirDebugView) {
    ui.selectable_value(debug_view, AreaRestirDebugView::Off, "Off");
    ui.selectable_value(debug_view, AreaRestirDebugView::Subpixel, "Subpixel");
    ui.selectable_value(debug_view, AreaRestirDebugView::Lens, "Lens");
    ui.selectable_value(debug_view, AreaRestirDebugView::Weight, "Weight");
    ui.selectable_value(
        debug_view,
        AreaRestirDebugView::HistoryValid,
        "History Valid",
    );
    ui.selectable_value(debug_view, AreaRestirDebugView::Rejection, "Rejection");
    ui.selectable_value(debug_view, AreaRestirDebugView::Jacobian, "Jacobian");
}

fn denoiser_label(mode: VptDenoiserMode) -> &'static str {
    match mode {
        VptDenoiserMode::Off => "Off",
        VptDenoiserMode::Svgf => "SVGF",
        VptDenoiserMode::Relax => "NRD RELAX",
        VptDenoiserMode::Reblur => "NRD REBLUR",
    }
}

fn lighting_debug_label(debug_view: LightingDebugView) -> &'static str {
    match debug_view {
        LightingDebugView::Final => "Final",
        LightingDebugView::DirectDiffuse => "Direct Diffuse",
        LightingDebugView::Normal => "Normal",
    }
}

fn restir_di_debug_label(debug_view: RestirDiDebugView) -> &'static str {
    match debug_view {
        RestirDiDebugView::Off => "Off",
        RestirDiDebugView::ReservoirWeight => "Reservoir Weight",
        RestirDiDebugView::LightId => "Light ID",
        RestirDiDebugView::Visibility => "Visibility",
        RestirDiDebugView::TemporalValid => "Temporal Valid",
        RestirDiDebugView::SpatialNeighbors => "Spatial Neighbors",
    }
}

fn area_restir_debug_label(debug_view: AreaRestirDebugView) -> &'static str {
    match debug_view {
        AreaRestirDebugView::Off => "Off",
        AreaRestirDebugView::Subpixel => "Subpixel",
        AreaRestirDebugView::Lens => "Lens",
        AreaRestirDebugView::Weight => "Weight",
        AreaRestirDebugView::HistoryValid => "History Valid",
        AreaRestirDebugView::Rejection => "Rejection",
        AreaRestirDebugView::Jacobian => "Jacobian",
    }
}

fn vpt_debug_label(debug_view: VptDebugView) -> &'static str {
    VPT_DEBUG_OPTIONS
        .iter()
        .find_map(|(view, label)| (*view == debug_view).then_some(*label))
        .unwrap_or("Unknown")
}

fn format_vec3(label: &str, value: glam::Vec3) -> String {
    format!("{label} {:>7.2} {:>7.2} {:>7.2}", value.x, value.y, value.z)
}

const VPT_DEBUG_OPTIONS: &[(VptDebugView, &str)] = &[
    (VptDebugView::Final, "Final"),
    (VptDebugView::Raw, "Raw"),
    (VptDebugView::Temporal, "Temporal"),
    (VptDebugView::Variance, "Variance"),
    (VptDebugView::HistoryValid, "History Valid"),
    (VptDebugView::Motion, "Motion"),
    (VptDebugView::Normal, "Normal"),
    (VptDebugView::Depth, "Depth"),
    (VptDebugView::ReservoirWeight, "Reservoir Weight"),
    (VptDebugView::Direct, "Direct"),
    (VptDebugView::Indirect, "Indirect"),
    (VptDebugView::AreaSubpixel, "Area Subpixel"),
    (VptDebugView::AreaLens, "Area Lens"),
    (VptDebugView::AreaWeight, "Area Weight"),
    (VptDebugView::AreaHistoryValid, "Area History Valid"),
    (VptDebugView::AreaRejection, "Area Rejection"),
    (VptDebugView::AreaJacobian, "Area Jacobian"),
    (VptDebugView::VoxelBrick, "Voxel Brick"),
    (VptDebugView::VoxelLocal, "Voxel Local"),
    (VptDebugView::VoxelHit, "Voxel Hit"),
    (VptDebugView::NrdNormalRoughness, "NRD Normal/Roughness"),
    (VptDebugView::NrdViewZ, "NRD ViewZ"),
    (VptDebugView::NrdMotion, "NRD Motion"),
    (VptDebugView::NrdMotionZ, "NRD Motion Z"),
    (VptDebugView::NrdValidation, "NRD Validation"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::area_restir::{AreaRestirDebugView, AreaRestirSettings};
    use crate::render::restir_di::RestirDiSettings;
    use crate::render::scene_ubo::{LightingSettings, VptDebugView};

    #[test]
    fn editor_vpt_bounce_control_clamps_to_runtime_range() {
        assert_eq!(clamp_vpt_max_bounces(0), 1);
        assert_eq!(clamp_vpt_max_bounces(4), 4);
        assert_eq!(clamp_vpt_max_bounces(128), 8);
    }

    #[test]
    fn editor_denoiser_iteration_control_clamps_to_runtime_range() {
        assert_eq!(clamp_denoiser_atrous_iterations(0), 0);
        assert_eq!(clamp_denoiser_atrous_iterations(3), 3);
        assert_eq!(clamp_denoiser_atrous_iterations(99), 5);
    }

    #[test]
    fn area_restir_debug_bridge_updates_area_and_vpt_debug_views() {
        let mut lighting = LightingSettings::default();
        let mut restir_di = RestirDiSettings::default();
        let mut area = AreaRestirSettings::default();

        set_area_restir_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area,
            AreaRestirDebugView::Weight,
        );

        assert_eq!(area.debug_view, AreaRestirDebugView::Weight);
        assert_eq!(lighting.vpt_debug_view, VptDebugView::AreaWeight);
        assert_eq!(restir_di.debug_view, RestirDiDebugView::Off);
    }

    #[test]
    fn area_restir_debug_off_restores_final_only_from_area_debug_views() {
        let mut lighting = LightingSettings {
            vpt_debug_view: VptDebugView::AreaLens,
            ..LightingSettings::default()
        };
        let mut area = AreaRestirSettings {
            debug_view: AreaRestirDebugView::Lens,
            ..AreaRestirSettings::default()
        };
        let mut restir_di = RestirDiSettings::default();

        set_area_restir_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area,
            AreaRestirDebugView::Off,
        );

        assert_eq!(area.debug_view, AreaRestirDebugView::Off);
        assert_eq!(lighting.vpt_debug_view, VptDebugView::Final);

        lighting.vpt_debug_view = VptDebugView::Normal;
        set_area_restir_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area,
            AreaRestirDebugView::Off,
        );

        assert_eq!(lighting.vpt_debug_view, VptDebugView::Normal);
    }

    #[test]
    fn restir_toggle_controls_mutate_existing_settings_without_shadow_state() {
        let mut restir_di = RestirDiSettings::default();
        let mut area_restir = AreaRestirSettings::default();

        set_restir_di_enabled(&mut restir_di, true);
        set_area_restir_enabled(&mut area_restir, true);

        assert!(restir_di.enabled);
        assert!(area_restir.enabled);

        set_restir_di_enabled(&mut restir_di, false);
        set_area_restir_enabled(&mut area_restir, false);

        assert!(!restir_di.enabled);
        assert!(!area_restir.enabled);
    }

    #[test]
    fn viewport_central_panel_keeps_world_visible_behind_editor_overlay() {
        let source = crate::render::source_checks::read_source("src/editor/ui.rs");
        let viewport_overlay = source
            .split("fn show_viewport_overlay")
            .nth(1)
            .expect("viewport overlay function should exist")
            .split("pub fn clamp_vpt_max_bounces")
            .next()
            .expect("viewport overlay function should end before helpers");
        let compact = crate::render::source_checks::compact(viewport_overlay);

        assert!(
            compact.contains("egui::CentralPanel::default().frame(egui::Frame::none()).show("),
            "viewport CentralPanel must not paint an opaque background over the rendered world"
        );
    }

    #[test]
    fn sun_angular_radius_control_keeps_finite_disk_positive() {
        let source = crate::render::source_checks::read_source("src/editor/ui.rs");
        let scene_panel = source
            .split("fn show_scene_panel")
            .nth(1)
            .expect("scene panel function should exist")
            .split("fn show_render_panel")
            .next()
            .expect("scene panel should end before render panel");

        assert!(
            scene_panel
                .contains("egui::Slider::new(&mut lighting.sun_angular_radius, 0.0001..=0.25)"),
            "sun angular radius UI must not allow zero because the VPT sun estimator now treats this as a finite disk, not a delta directional light"
        );
        assert!(
            scene_panel.contains("lighting.sun_angular_radius <= 0.0"),
            "sun angular radius UI must sanitize non-positive values back to the default finite disk"
        );
    }

    #[test]
    fn debug_panel_exposes_restir_di_debug_controls_without_advanced_gate() {
        let source = crate::render::source_checks::read_source("src/editor/ui.rs");
        let debug_panel = source
            .split("fn show_debug_panel")
            .nth(1)
            .expect("debug panel function should exist")
            .split("fn denoiser_combo")
            .next()
            .expect("debug panel should end before combo helpers");

        assert!(
            debug_panel.contains("restir_di: &mut RestirDiSettings"),
            "Debug panel must receive the live ReSTIR-DI settings, not only lighting and Area ReSTIR"
        );
        assert!(
            debug_panel.contains("debug_restir_di_view"),
            "Debug panel must expose a stable ReSTIR-DI debug combo id"
        );
        assert!(
            debug_panel
                .contains("set_restir_di_debug_view(lighting, restir_di, area, restir_debug)"),
            "Debug panel must synchronize the live ReSTIR-DI debug view with the VPT debug view"
        );
    }

    #[test]
    fn restir_di_debug_bridge_updates_restir_and_vpt_debug_views() {
        let mut lighting = LightingSettings::default();
        let mut restir_di = RestirDiSettings::default();
        let mut area_restir = AreaRestirSettings::default();

        set_restir_di_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            RestirDiDebugView::ReservoirWeight,
        );

        assert_eq!(restir_di.debug_view, RestirDiDebugView::ReservoirWeight);
        assert_eq!(lighting.vpt_debug_view, VptDebugView::ReservoirWeight);
        assert_eq!(area_restir.debug_view, AreaRestirDebugView::Off);
    }

    #[test]
    fn restir_di_debug_off_restores_final_only_from_restir_debug_view() {
        let mut lighting = LightingSettings {
            vpt_debug_view: VptDebugView::ReservoirWeight,
            ..LightingSettings::default()
        };
        let mut restir_di = RestirDiSettings {
            debug_view: RestirDiDebugView::ReservoirWeight,
            ..RestirDiSettings::default()
        };
        let mut area_restir = AreaRestirSettings::default();

        set_restir_di_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            RestirDiDebugView::Off,
        );

        assert_eq!(restir_di.debug_view, RestirDiDebugView::Off);
        assert_eq!(lighting.vpt_debug_view, VptDebugView::Final);

        lighting.vpt_debug_view = VptDebugView::Direct;
        set_restir_di_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            RestirDiDebugView::Off,
        );

        assert_eq!(lighting.vpt_debug_view, VptDebugView::Direct);
    }

    #[test]
    fn restir_di_non_vpt_debug_views_fall_back_to_final_and_clear_area_restir_debug_view() {
        for restir_debug in [
            RestirDiDebugView::LightId,
            RestirDiDebugView::Visibility,
            RestirDiDebugView::TemporalValid,
            RestirDiDebugView::SpatialNeighbors,
        ] {
            let mut lighting = LightingSettings::default();
            let mut restir_di = RestirDiSettings::default();
            let mut area_restir = AreaRestirSettings::default();

            set_area_restir_debug_view(
                &mut lighting,
                &mut restir_di,
                &mut area_restir,
                AreaRestirDebugView::Lens,
            );
            set_restir_di_debug_view(
                &mut lighting,
                &mut restir_di,
                &mut area_restir,
                restir_debug,
            );

            assert_eq!(restir_di.debug_view, restir_debug);
            assert_eq!(area_restir.debug_view, AreaRestirDebugView::Off);
            assert_eq!(lighting.vpt_debug_view, VptDebugView::Final);

            set_restir_di_debug_view(
                &mut lighting,
                &mut restir_di,
                &mut area_restir,
                RestirDiDebugView::ReservoirWeight,
            );
            set_restir_di_debug_view(
                &mut lighting,
                &mut restir_di,
                &mut area_restir,
                restir_debug,
            );

            assert_eq!(restir_di.debug_view, restir_debug);
            assert_eq!(area_restir.debug_view, AreaRestirDebugView::Off);
            assert_eq!(lighting.vpt_debug_view, VptDebugView::Final);
        }
    }

    #[test]
    fn selecting_bridged_vpt_debug_view_clears_other_restir_debug_view() {
        let mut lighting = LightingSettings::default();
        let mut restir_di = RestirDiSettings::default();
        let mut area_restir = AreaRestirSettings::default();

        set_area_restir_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            AreaRestirDebugView::Lens,
        );
        set_vpt_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            VptDebugView::ReservoirWeight,
        );

        assert_eq!(lighting.vpt_debug_view, VptDebugView::ReservoirWeight);
        assert_eq!(restir_di.debug_view, RestirDiDebugView::ReservoirWeight);
        assert_eq!(area_restir.debug_view, AreaRestirDebugView::Off);

        set_vpt_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            VptDebugView::AreaJacobian,
        );

        assert_eq!(lighting.vpt_debug_view, VptDebugView::AreaJacobian);
        assert_eq!(restir_di.debug_view, RestirDiDebugView::Off);
        assert_eq!(area_restir.debug_view, AreaRestirDebugView::Jacobian);
    }

    #[test]
    fn selecting_area_restir_debug_view_clears_restir_di_debug_view() {
        let mut lighting = LightingSettings::default();
        let mut restir_di = RestirDiSettings::default();
        let mut area_restir = AreaRestirSettings::default();

        set_restir_di_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            RestirDiDebugView::ReservoirWeight,
        );
        set_area_restir_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            AreaRestirDebugView::Weight,
        );

        assert_eq!(lighting.vpt_debug_view, VptDebugView::AreaWeight);
        assert_eq!(area_restir.debug_view, AreaRestirDebugView::Weight);
        assert_eq!(restir_di.debug_view, RestirDiDebugView::Off);
    }

    #[test]
    fn selecting_restir_di_debug_view_clears_area_restir_debug_view() {
        let mut lighting = LightingSettings::default();
        let mut restir_di = RestirDiSettings::default();
        let mut area_restir = AreaRestirSettings::default();

        set_area_restir_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            AreaRestirDebugView::Jacobian,
        );
        set_restir_di_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            RestirDiDebugView::ReservoirWeight,
        );

        assert_eq!(lighting.vpt_debug_view, VptDebugView::ReservoirWeight);
        assert_eq!(restir_di.debug_view, RestirDiDebugView::ReservoirWeight);
        assert_eq!(area_restir.debug_view, AreaRestirDebugView::Off);
    }

    #[test]
    fn selecting_plain_vpt_debug_view_clears_restir_debug_views() {
        let mut lighting = LightingSettings::default();
        let mut restir_di = RestirDiSettings::default();
        let mut area_restir = AreaRestirSettings::default();

        set_restir_di_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            RestirDiDebugView::ReservoirWeight,
        );
        set_vpt_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            VptDebugView::Final,
        );

        assert_eq!(lighting.vpt_debug_view, VptDebugView::Final);
        assert_eq!(restir_di.debug_view, RestirDiDebugView::Off);
        assert_eq!(area_restir.debug_view, AreaRestirDebugView::Off);

        set_area_restir_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            AreaRestirDebugView::Lens,
        );
        set_vpt_debug_view(
            &mut lighting,
            &mut restir_di,
            &mut area_restir,
            VptDebugView::Direct,
        );

        assert_eq!(lighting.vpt_debug_view, VptDebugView::Direct);
        assert_eq!(restir_di.debug_view, RestirDiDebugView::Off);
        assert_eq!(area_restir.debug_view, AreaRestirDebugView::Off);
    }
}
