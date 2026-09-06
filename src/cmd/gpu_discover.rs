//! `llmman gpu-discover` — a hidden (undocumented, absent from `--help`)
//! standalone entry point for the same accelerator probe `llmman serve`
//! and `llama_release`/`container` already run internally (see
//! `crate::hostgpu`), so a user or an issue report can run the probe on
//! its own and see exactly what it found, without needing to start a
//! whole server first. Mirrors `ollama gpu-discover`
//! (`cmd/cmd.go`'s hidden `gpu-discover` command) in spirit — llmman's own
//! probe has no `--lib-dir` equivalent to take (dynamic-loaded vendor
//! libraries are found by the OS's normal search path, not a directory
//! llmman itself searches), so this takes no flags at all.

use clap::Args;

#[derive(Args, Debug)]
pub struct GpuDiscoverArgs {}

pub fn run(_args: &GpuDiscoverArgs) -> anyhow::Result<()> {
    let (gpu, vram) = crate::hostgpu::detect_with_vram();
    let kind = match gpu {
        crate::hostgpu::HostGpu::None => "none".to_string(),
        crate::hostgpu::HostGpu::Cuda { major } => format!("cuda (driver major version {major})"),
        crate::hostgpu::HostGpu::Rocm => "rocm".to_string(),
        crate::hostgpu::HostGpu::Vulkan => "vulkan".to_string(),
        crate::hostgpu::HostGpu::Metal => "metal".to_string(),
    };
    println!("accelerator: {kind}");
    if vram > 0 {
        println!("vram: {}", crate::fmt::human_size(vram));
    } else {
        println!("vram: unknown");
    }
    Ok(())
}
