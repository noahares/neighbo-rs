use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;

use clap::Parser;

fn normalized_ratio(s: &str) -> Result<f64> {
    let ratio: f64 = s
        .parse()
        .with_context(|| format!("`{s}` isn't a valid ratio"))?;
    if (0.0..=1.0).contains(&ratio) {
        Ok(ratio)
    } else {
        Err(anyhow!(format!("Ratio is not in range {}-{}", 0.0, 1.0)))
    }
}

#[derive(Parser)]
#[command(author, version, about)]
pub struct Args {
    /// Path to the reference tree file
    #[arg(short = 't', long)]
    pub reference_tree: Option<PathBuf>,
    /// Path to the reference tree distribution
    #[arg(short = 'r', long)]
    pub reference_distribution: PathBuf,
    /// Path(s) to the tree distributions
    #[arg(short = 'd', long, value_delimiter = ' ', num_args = 1..)]
    pub distribution_paths: Vec<PathBuf>,
    /// Consensus cutoff
    #[arg(short = 'c', long, default_value_t = 0.5, value_parser = normalized_ratio)]
    pub consensus_cutoff: f64,
    /// Output path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}
