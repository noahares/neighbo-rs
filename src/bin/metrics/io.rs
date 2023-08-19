use anyhow::{anyhow, Context, Result};
use serde::{de, Deserialize, Serialize};
use std::{
    collections::HashMap, fmt::Display, io::Write, path::PathBuf, str::FromStr,
};

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
    #[serde(rename = "protein")]
    AA,
    #[serde(rename = "DNA")]
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
    pub datasets: HashMap<String, DataSet>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DataSet {
    pub sequence_file: PathBuf,
    pub reference_tree: Option<PathBuf>,
    pub moltype: Moltype,
    pub seed: u32,
    pub num_trees: usize,
    pub reference_tool: Tool,
    pub tools: Vec<Tool>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub distribution_path: PathBuf,
    pub time: f64,
    pub perturbation: Option<f64>,
    pub ratio: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Metadata {
    pub sequence_file: PathBuf,
    pub reference_tree: Option<PathBuf>,
    pub moltype: Moltype,
    pub seed: u32,
    pub num_trees: usize,
    pub perturbation: Option<f64>,
    pub ratio: Option<f64>,
    pub reference_tool: String,
    pub tool: String,
}

impl From<(&DataSet, &Tool)> for Metadata {
    fn from((d, t): (&DataSet, &Tool)) -> Self {
        Self {
            sequence_file: d.sequence_file.clone(),
            reference_tree: d.reference_tree.clone(),
            moltype: d.moltype,
            seed: d.seed,
            num_trees: d.num_trees,
            perturbation: t.perturbation,
            ratio: t.ratio,
            reference_tool: d.reference_tool.name.clone(),
            tool: t.name.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metrics {
    sequence_file: PathBuf,
    reference_tree: Option<PathBuf>,
    moltype: Moltype,
    seed: u32,
    num_trees: usize,
    perturbation: Option<f64>,
    ratio: Option<f64>,
    reference_tool: String,
    tool: String,
    #[serde(rename = "reference_missed_splits_ratio")]
    missed_splits_ratio: Option<f64>,
    min: Option<f64>,
    mean: Option<f64>,
    max: Option<f64>,
    simple_distance: f64,
    hellinger_distance: f64,
    asdsf: f64,
    consensus_distance: f64,
    #[serde(rename = "pcc")]
    pearson_correlation_coefficient: f64,
    #[serde(rename = "ubr")]
    unique_bipartition_ratio: f64,
    #[serde(rename = "ubr_ref")]
    unique_bipartition_ratio_per_chain_ref: f64,
    #[serde(rename = "ubr_tool")]
    unique_bipartition_ratio_per_chain_tool: f64,
    num_biparts: usize,
}

impl
    From<(
        Metadata,
        Option<crate::metrics::ReferenceTreeMetrics>,
        crate::metrics::DistanceMetrics,
    )> for Metrics
{
    fn from(
        (meta, rtm, dm): (
            Metadata,
            Option<crate::metrics::ReferenceTreeMetrics>,
            crate::metrics::DistanceMetrics,
        ),
    ) -> Self {
        let (missed_splits_ratio, min, mean, max) = match rtm {
            Some(rtm) => (
                Some(rtm.missed_splits_ratio),
                Some(rtm.rf_distance_stats.min),
                Some(rtm.rf_distance_stats.mean),
                Some(rtm.rf_distance_stats.max),
            ),
            None => (None, None, None, None),
        };
        Self {
            sequence_file: meta.sequence_file,
            reference_tree: meta.reference_tree,
            moltype: meta.moltype,
            seed: meta.seed,
            num_trees: meta.num_trees,
            perturbation: meta.perturbation,
            ratio: meta.ratio,
            reference_tool: meta.reference_tool,
            tool: meta.tool,
            missed_splits_ratio,
            min,
            mean,
            max,
            simple_distance: dm.simple_distance,
            hellinger_distance: dm.hellinger_distance,
            asdsf: dm.asdsf,
            consensus_distance: dm.consensus_distance,
            pearson_correlation_coefficient: dm
                .pearson_correlation_coefficient,
            unique_bipartition_ratio: dm.unique_bipartition_ratio,
            unique_bipartition_ratio_per_chain_ref: dm
                .unique_bipartition_ratio_per_chain
                .0,
            unique_bipartition_ratio_per_chain_tool: dm
                .unique_bipartition_ratio_per_chain
                .1,
            num_biparts: dm.num_bipartitions,
        }
    }
}
