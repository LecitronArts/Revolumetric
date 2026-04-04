#[derive(Debug, Default, Clone, Copy)]
pub struct Time {
    pub delta_seconds: f32,
    pub elapsed_seconds: f32,
}

impl Time {
    pub fn advance(&mut self, delta_seconds: f32) {
        self.delta_seconds = delta_seconds;
        self.elapsed_seconds += delta_seconds;
    }
}
