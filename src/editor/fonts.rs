//! Font and style configuration for the desktop editor UI.

use egui::{Color32, FontData, FontDefinitions, FontFamily, FontId, Stroke, TextStyle, Vec2};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const INTER_FONT_NAME: &str = "Inter";
const NOTO_SANS_SC_FONT_NAME: &str = "Noto Sans SC";
const JETBRAINS_MONO_FONT_NAME: &str = "JetBrains Mono";

struct EditorFontSpec {
    display_name: &'static str,
    asset_names: &'static [&'static str],
}

const INTER_FONT: EditorFontSpec = EditorFontSpec {
    display_name: INTER_FONT_NAME,
    asset_names: &["Inter-Regular.ttf"],
};

const NOTO_SANS_SC_FONT: EditorFontSpec = EditorFontSpec {
    display_name: NOTO_SANS_SC_FONT_NAME,
    asset_names: &["NotoSansSC-Regular.otf", "NotoSansSC-VF.ttf"],
};

const JETBRAINS_MONO_FONT: EditorFontSpec = EditorFontSpec {
    display_name: JETBRAINS_MONO_FONT_NAME,
    asset_names: &["JetBrainsMono-Regular.ttf"],
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorFontWarning {
    pub font_name: &'static str,
    pub searched_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct EditorFontReport {
    pub fonts: FontDefinitions,
    pub warnings: Vec<EditorFontWarning>,
}

pub fn configure_editor_fonts(font_dir: impl AsRef<Path>) -> EditorFontReport {
    let font_dir = font_dir.as_ref();
    let mut fonts = FontDefinitions::default();
    let mut warnings = Vec::new();

    let loaded_noto = load_font(font_dir, &NOTO_SANS_SC_FONT, &mut fonts, &mut warnings);
    let loaded_inter = load_font(font_dir, &INTER_FONT, &mut fonts, &mut warnings);
    let loaded_jetbrains = load_font(font_dir, &JETBRAINS_MONO_FONT, &mut fonts, &mut warnings);

    if loaded_noto {
        prepend_font_family(&mut fonts, FontFamily::Proportional, NOTO_SANS_SC_FONT_NAME);
        prepend_font_family(&mut fonts, FontFamily::Monospace, NOTO_SANS_SC_FONT_NAME);
    }
    if loaded_inter {
        prepend_font_family(&mut fonts, FontFamily::Proportional, INTER_FONT_NAME);
    }
    if loaded_jetbrains {
        prepend_font_family(&mut fonts, FontFamily::Monospace, JETBRAINS_MONO_FONT_NAME);
    }

    EditorFontReport { fonts, warnings }
}

pub fn configure_editor_style() -> egui::Style {
    let mut style = egui::Style {
        visuals: egui::Visuals::dark(),
        ..egui::Style::default()
    };

    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(18.0, FontFamily::Proportional),
    );
    style
        .text_styles
        .insert(TextStyle::Body, FontId::new(13.0, FontFamily::Proportional));
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(12.5, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Small,
        FontId::new(11.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Monospace,
        FontId::new(12.5, FontFamily::Monospace),
    );

    style.spacing.item_spacing = Vec2::new(6.0, 5.0);
    style.spacing.button_padding = Vec2::new(7.0, 4.0);
    style.spacing.slider_width = 180.0;
    style.spacing.indent = 14.0;
    style.spacing.interact_size = Vec2::new(18.0, 18.0);

    let graphite_panel = Color32::from_rgb(18, 20, 22);
    let graphite_window = Color32::from_rgb(24, 27, 30);
    let graphite_edge = Color32::from_rgb(64, 70, 76);
    let cyan = Color32::from_rgb(76, 190, 204);
    let amber = Color32::from_rgb(220, 155, 60);

    style.visuals.panel_fill = graphite_panel;
    style.visuals.window_fill = graphite_window;
    style.visuals.extreme_bg_color = Color32::from_rgb(10, 12, 14);
    style.visuals.faint_bg_color = Color32::from_rgb(32, 36, 40);
    style.visuals.code_bg_color = Color32::from_rgb(16, 18, 20);
    style.visuals.selection.bg_fill = cyan;
    style.visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(220, 250, 255));
    style.visuals.hyperlink_color = cyan;
    style.visuals.warn_fg_color = amber;
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, graphite_edge);
    style.visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(30, 34, 38);
    style.visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(42, 48, 52);
    style.visuals.widgets.active.weak_bg_fill = Color32::from_rgb(48, 58, 60);

    style
}

fn load_font(
    font_dir: &Path,
    spec: &EditorFontSpec,
    fonts: &mut FontDefinitions,
    warnings: &mut Vec<EditorFontWarning>,
) -> bool {
    let searched_paths = spec
        .asset_names
        .iter()
        .map(|asset_name| font_dir.join(asset_name))
        .collect::<Vec<_>>();

    for path in &searched_paths {
        match std::fs::read(path) {
            Ok(bytes) => {
                fonts.font_data.insert(
                    spec.display_name.to_owned(),
                    Arc::new(FontData::from_owned(bytes)),
                );
                return true;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::warn!(
                    font = spec.display_name,
                    path = %path.display(),
                    %error,
                    "failed to read optional editor font asset"
                );
            }
        }
    }

    warnings.push(EditorFontWarning {
        font_name: spec.display_name,
        searched_paths,
    });
    false
}

fn prepend_font_family(fonts: &mut FontDefinitions, family: FontFamily, font_name: &str) {
    let family_fonts = fonts.families.entry(family).or_default();
    family_fonts.retain(|existing| existing != font_name);
    family_fonts.insert(0, font_name.to_owned());
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Color32, FontFamily, TextStyle};
    use std::fs;

    fn unique_font_dir(test_name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("revolumetric_{test_name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp font dir should be created");
        dir
    }

    #[test]
    fn missing_editor_font_assets_report_warnings_but_keep_default_fonts() {
        let dir = unique_font_dir("missing_editor_font_assets");

        let report = configure_editor_fonts(&dir);

        assert_eq!(report.warnings.len(), 3);
        assert!(
            !report.fonts.font_data.is_empty(),
            "egui default fonts must remain available when optional editor fonts are missing"
        );
        assert!(
            report
                .fonts
                .families
                .contains_key(&FontFamily::Proportional)
        );
        assert!(report.fonts.families.contains_key(&FontFamily::Monospace));
    }

    #[test]
    fn available_editor_font_assets_are_prioritized_by_family() {
        let dir = unique_font_dir("available_editor_font_assets");
        fs::write(dir.join("Inter-Regular.ttf"), b"inter").expect("write Inter test asset");
        fs::write(dir.join("NotoSansSC-Regular.otf"), b"noto").expect("write Noto test asset");
        fs::write(dir.join("JetBrainsMono-Regular.ttf"), b"mono").expect("write mono test asset");

        let report = configure_editor_fonts(&dir);
        let proportional = report
            .fonts
            .families
            .get(&FontFamily::Proportional)
            .expect("proportional family should exist");
        let monospace = report
            .fonts
            .families
            .get(&FontFamily::Monospace)
            .expect("monospace family should exist");

        assert!(report.warnings.is_empty());
        assert_eq!(&proportional[..2], ["Inter", "Noto Sans SC"]);
        assert_eq!(&monospace[..2], ["JetBrains Mono", "Noto Sans SC"]);
        assert!(report.fonts.font_data.contains_key("Inter"));
        assert!(report.fonts.font_data.contains_key("Noto Sans SC"));
        assert!(report.fonts.font_data.contains_key("JetBrains Mono"));
    }

    #[test]
    fn editor_style_uses_compact_graphite_theme_without_purple_defaults() {
        let style = configure_editor_style();

        assert!(style.visuals.dark_mode);
        assert_eq!(
            style.text_styles[&TextStyle::Monospace].family,
            FontFamily::Monospace
        );
        assert!(style.spacing.item_spacing.x <= 8.0);
        assert!(style.spacing.slider_width >= 160.0);
        assert_ne!(
            style.visuals.selection.bg_fill,
            Color32::from_rgb(128, 92, 255),
            "editor accent must not use a generic purple default"
        );
    }
}
