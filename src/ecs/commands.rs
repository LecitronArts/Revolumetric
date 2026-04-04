#[derive(Default)]
pub struct Commands {
    queued_spawns: usize,
}

impl Commands {
    pub fn queue_spawn(&mut self) {
        self.queued_spawns += 1;
    }

    pub fn queued_spawns(&self) -> usize {
        self.queued_spawns
    }
}
