use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Context;
use clap::Args;

use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// Tag for the resulting image (e.g. registry.example.com/mymodel:latest)
    #[arg(short, long, value_name = "REFERENCE")]
    pub tag: String,

    /// Directory whose files will be packaged as OCI layers
    #[arg(value_name = "CONTEXT_DIR", default_value = ".")]
    pub context_dir: PathBuf,

    /// Key=value labels to embed in the image config
    #[arg(short, long, value_name = "KEY=VALUE")]
    pub label: Vec<String>,
}

pub fn run(args: &BuildArgs) -> anyhow::Result<()> {
    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;

    let labels: HashMap<String, String> = args
        .label
        .iter()
        .map(|kv| {
            let mut parts = kv.splitn(2, '=');
            let k = parts.next().unwrap_or("").to_string();
            let v = parts.next().unwrap_or("").to_string();
            (k, v)
        })
        .collect();

    let context_dir = args
        .context_dir
        .canonicalize()
        .with_context(|| format!("context dir: {}", args.context_dir.display()))?;

    let desc = store.build(&context_dir, &args.tag, &labels)?;
    println!("Built {} ({})", args.tag, desc.digest);
    Ok(())
}
