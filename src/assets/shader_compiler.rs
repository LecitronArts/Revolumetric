use anyhow::Result;

#[derive(Debug, Default, Clone)]
pub struct ShaderCompiler;

impl ShaderCompiler {
    pub fn new() -> Self {
        Self
    }

    pub fn compile_all(&self) -> Result<()> {
        Ok(())
    }
}
