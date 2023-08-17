use anyhow::{anyhow, Context, Result};
use serde::{de, Deserialize, Serialize};
use std::{fmt::Display, io::Write, path::PathBuf, str::FromStr};

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

impl Args {
    pub fn get_output(&self) -> Result<Box<dyn Write>> {
        match self.output {
            Some(ref path) => Ok(std::fs::File::create(path)
                .map(|f| Box::new(f) as Box<dyn Write>)?),
            None => Ok(Box::new(std::io::stdout())),
        }
    }
}

#[derive(Serialize, Clone, Debug, Copy)]
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

impl Display for Moltype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Moltype::AA => write!(f, "protein"),
            Moltype::Dna => write!(f, "DNA"),
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

#[derive(Clone, Debug)]
pub struct Metrics {
    pub sequence_file: PathBuf,
    pub reference_tree: Option<PathBuf>,
    pub moltype: Moltype,
    pub seed: u32,
    pub num_trees: usize,
    pub perturbation: f64,
    pub ratio: f64,
    pub reference_tool: String,
    pub tool: String,
    pub reference_metrics: Option<crate::metrics::ReferenceTreeMetrics>,
    pub distance_metrics: crate::metrics::DistanceMetrics,
}

impl Metrics {
    pub fn get_csv_header() -> String {
        format!(
            "sequence_file,\
                     reference_tree,\
                     moltype,\
                     seed,\
                     num_trees,\
                     perturbation,\
                     ratio,\
                     reference_tool,\
                     tool,\
                     {},\
                     {}",
            crate::metrics::ReferenceTreeMetrics::get_csv_header(),
            crate::metrics::DistanceMetrics::get_csv_header()
        )
    }
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            self.sequence_file.to_str().unwrap(),
            self.reference_tree
                .clone()
                .unwrap_or(PathBuf::default())
                .to_str()
                .unwrap(),
            self.moltype,
            self.seed,
            self.num_trees,
            self.perturbation,
            self.ratio,
            self.reference_tool,
            self.tool,
            match self.reference_metrics.clone() {
                Some(m) => m.to_csv_row(),
                None => String::from(""),
            },
            self.distance_metrics.to_csv_row()
        )
    }
}
