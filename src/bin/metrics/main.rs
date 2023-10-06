use anyhow::{anyhow, Context, Result};
use clap::Parser;
use itertools::Itertools;
use log::{debug, info, warn};
use rayon::prelude::*;

use crate::datastructures::{Data, Metadata, Tool};

#[macro_use]
extern crate assert_float_eq;

mod datastructures;
mod io;
mod metrics;
mod parser;

fn main() -> Result<()> {
    let args = io::Args::parse();
    env_logger::Builder::new()
        .filter_level(args.verbosity.log_level_filter())
        .init();
    let metrics: Vec<io::Metrics> =
        if let Some(config_path) = args.config.as_ref() {
            debug!(
                "Running evaluation from JSON file: {}",
                config_path.display()
            );
            let config_str = std::fs::read_to_string(config_path)?;
            let config: Data = serde_json::from_str(&config_str)?;
            config
                .datasets
                .into_par_iter()
                .filter_map(|(_, d)| {
                    info!(
                        "Processing dataset {}",
                        d.sequence_file.clone().display()
                    );
                    let metadata = d
                        .tools
                        .iter()
                        .map(|t| Metadata::from((&d, t)))
                        .collect_vec();
                    match metrics::evaluate_dataset(
                        d.reference_tree.clone(),
                        &d.reference_tool,
                        &d.tools,
                        args.consensus_cutoff,
                        metadata,
                    ) {
                        Ok(r) => Some(r),
                        Err(..) => {
                            warn!(
                                "Failed to process dataset {}",
                                d.sequence_file.clone().display()
                            );
                            None
                        }
                    }
                })
                .flatten()
                .collect::<Vec<io::Metrics>>()
        } else {
            debug!("Running single evaluation with options from command line");
            let reference_tool = Tool::try_from(
                args.reference_distribution
                    .clone()
                    .context("No reference distribution path")?,
            )?;
            match args.distribution_paths.as_ref() {
                Some(paths) => {
                    let tools: Vec<Tool> = paths
                        .iter()
                        .map(|p| Tool::try_from(p.clone()))
                        .collect::<Result<Vec<Tool>>>()?;
                    let metadata = tools
                        .iter()
                        .map(|t| {
                            Metadata::from_only_tools(
                                &reference_tool.name,
                                &t.name,
                            )
                        })
                        .collect_vec();
                    metrics::evaluate_dataset(
                        args.reference_tree.clone(),
                        &reference_tool,
                        &tools,
                        args.consensus_cutoff,
                        metadata,
                    )
                }
                None => Err(anyhow!("No reference distribution path")),
            }?
        };
    let mut wtr = csv::Writer::from_writer(args.get_output()?);
    for m in &metrics {
        wtr.serialize(m)?;
    }
    wtr.flush()?;
    Ok(())
}
