use anyhow::{anyhow, Context, Result};
use clap_verbosity_flag::Verbosity;
use serde::{Deserialize, Serialize};
use std::{io::Write, path::PathBuf};

use clap::Parser;

use crate::datastructures::{Metadata, Moltype};

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
    #[command(flatten)]
    pub verbosity: Verbosity,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metrics {
    sequence_file: Option<PathBuf>,
    reference_tree: Option<PathBuf>,
    moltype: Option<Moltype>,
    seed: Option<u32>,
    num_trees: Option<usize>,
    difficulty: Option<f64>,
    perturbation: Option<f64>,
    ratio: Option<f64>,
    strategy: Option<String>,
    percentile: Option<f64>,
    reference_tool: String,
    tool: String,
    reference_missed_splits_ratio: Option<f64>,
    reference_min: Option<f64>,
    reference_mean: Option<f64>,
    reference_max: Option<f64>,
    tool_missed_splits_ratio: Option<f64>,
    tool_min: Option<f64>,
    tool_mean: Option<f64>,
    tool_max: Option<f64>,
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
        Option<crate::metrics::ReferenceTreeMetrics>,
        crate::metrics::DistanceMetrics,
    )> for Metrics
{
    fn from(
        (meta, rtm, ttm, dm): (
            Metadata,
            Option<crate::metrics::ReferenceTreeMetrics>,
            Option<crate::metrics::ReferenceTreeMetrics>,
            crate::metrics::DistanceMetrics,
        ),
    ) -> Self {
        let (missed_splits_ratio_r, min_r, mean_r, max_r) = match rtm {
            Some(rtm) => (
                Some(rtm.missed_splits_ratio),
                Some(rtm.rf_distance_stats.min),
                Some(rtm.rf_distance_stats.mean),
                Some(rtm.rf_distance_stats.max),
            ),
            None => (None, None, None, None),
        };
        let (missed_splits_ratio_t, min_t, mean_t, max_t) = match ttm {
            Some(ttm) => (
                Some(ttm.missed_splits_ratio),
                Some(ttm.rf_distance_stats.min),
                Some(ttm.rf_distance_stats.mean),
                Some(ttm.rf_distance_stats.max),
            ),
            None => (None, None, None, None),
        };
        Self {
            sequence_file: meta.sequence_file,
            reference_tree: meta.reference_tree,
            moltype: meta.moltype,
            seed: meta.seed,
            num_trees: meta.num_trees,
            difficulty: meta.difficulty,
            perturbation: meta.perturbation,
            ratio: meta.ratio,
            strategy: meta.strategy,
            percentile: meta.percentile,
            reference_tool: meta.reference_tool,
            tool: meta.tool,
            reference_missed_splits_ratio: missed_splits_ratio_r,
            reference_min: min_r,
            reference_mean: mean_r,
            reference_max: max_r,
            tool_missed_splits_ratio: missed_splits_ratio_t,
            tool_min: min_t,
            tool_mean: mean_t,
            tool_max: max_t,
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
