use winit::dpi::LogicalSize;
use winit::window::WindowAttributes;

#[derive(Debug, Clone)]
pub struct WindowDescriptor {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

impl Default for WindowDescriptor {
    fn default() -> Self {
        Self {
            title: "Revolumetric".to_string(),
            width: 1280,
            height: 720,
        }
    }
}

impl WindowDescriptor {
    pub fn attributes(&self) -> WindowAttributes {
        WindowAttributes::default()
            .with_title(self.title.clone())
            .with_inner_size(LogicalSize::new(self.width as f64, self.height as f64))
    }
}
