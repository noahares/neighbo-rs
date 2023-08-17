use anyhow::{anyhow, Context, Result};
use serde::{de, Deserialize, Serialize};
use std::{path::PathBuf, str::FromStr};

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
pub struct Args {
    /// Path to the JSON config file
    #[arg(
        long,
        conflicts_with = "reference_tree",
        conflicts_with = "reference_distribution",
        conflicts_with = "distribution_paths"
    )]
    pub config: Option<PathBuf>,
    /// Path to the reference tree file
    #[arg(short = 't', long)]
    pub reference_tree: Option<PathBuf>,
    /// Path to the reference tree distribution
    #[arg(short = 'r', long, requires = "distribution_paths")]
    pub reference_distribution: Option<PathBuf>,
    /// Path(s) to the tree distributions
    #[arg(short = 'd', long, value_delimiter = ' ', num_args = 1..)]
    pub distribution_paths: Option<Vec<PathBuf>>,
    /// Consensus cutoff
    #[arg(short = 'c', long, default_value_t = 0.5, value_parser = normalized_ratio)]
    pub consensus_cutoff: f64,
    /// Output path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Serialize, Clone, Debug)]
pub enum Moltype {
    AA,
    Dna,
}

impl FromStr for Moltype {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "protein" => Ok(Self::AA),
            "DNA" => Ok(Self::Dna),
            _ => Err(anyhow!(format!("Unknown Moltype: {}", s))),
        }
    }
}

impl<'de> Deserialize<'de> for Moltype {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        FromStr::from_str(&s).map_err(de::Error::custom)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Data {
    pub datasets: Vec<DataSet>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DataSet {
    pub sequence_file: PathBuf,
    pub reference_tree: Option<PathBuf>,
    pub moltype: Moltype,
    pub seed: u32,
    pub num_trees: usize,
    pub perturbation: f64,
    pub ratio: f64,
    pub reference_tool: Tool,
    pub tools: Vec<Tool>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub distribution_path: PathBuf,
    pub time: f64,
}
