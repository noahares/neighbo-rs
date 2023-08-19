use anyhow::{anyhow, Result};
use bitvec::vec::BitVec;
use clap::Parser;
use itertools::Itertools;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[macro_use]
extern crate assert_float_eq;

mod io;
mod metrics;
mod parser;

fn main() -> Result<()> {
    let args = io::Args::parse();
    if let Some(config_path) = args.config.as_ref() {
        let config_str = std::fs::read_to_string(config_path)?;
        let config: io::Data = serde_json::from_str(&config_str)?;
        let metrics: Vec<io::Metrics> = config
            .datasets
            .values()
            .map(|d| {
                if let Ok((reference_metrics, distance_metrics)) =
                    evaulate_dataset(
                        d.reference_tree.clone(),
                        d.reference_tool.clone(),
                        &d.tools,
                        args.consensus_cutoff,
                    )
                {
                    Ok(distance_metrics
                        .iter()
                        .zip(d.tools.iter())
                        .map(|(m, t)| {
                            io::Metrics::from((
                                (d, t).into(),
                                reference_metrics.clone(),
                                m.clone(),
                            ))
                        })
                        .collect_vec())
                } else {
                    Err(anyhow!(format!(
                        "Error processing the following dataframe:\n{:?}",
                        d
                    )))
                }
            })
            .collect::<Result<Vec<Vec<io::Metrics>>>>()?
            .into_iter()
            .flatten()
            .collect_vec();
        let mut wtr = csv::Writer::from_writer(args.get_output()?);
        for m in &metrics {
            wtr.serialize(m)?;
        }
        wtr.flush()?;
    } else {
        todo!()
    }
    Ok(())
}

fn evaulate_dataset(
    reference_path: Option<PathBuf>,
    reference_tool: io::Tool,
    other_tools: &[io::Tool],
    cutoff: f64,
) -> Result<(
    Option<metrics::ReferenceTreeMetrics>,
    Vec<metrics::DistanceMetrics>,
)> {
    let (mapping, reference_tree_bipartitions) = if let Some(reference_path) =
        reference_path
    {
        let reference_file = File::open(reference_path)?;
        let reference_string = {
            let mut reference_string: String = String::from("");
            BufReader::new(reference_file).read_line(&mut reference_string)?;
            parser::NewickParser::preprocess_input(reference_string)?
        };
        let reference_mapping =
            parser::NewickParser::get_taxa_mapping(&reference_string);
        let reference_bipartitions =
            parser::NewickParser::new(&reference_string, &reference_mapping)
                .parse();
        (Some(reference_mapping), Some(reference_bipartitions))
    } else {
        (None, None)
    };
    let reference_distribution_file =
        File::open(reference_tool.distribution_path)?;
    let lines: Vec<String> = BufReader::new(reference_distribution_file)
        .lines()
        .map_while(|l| parser::NewickParser::preprocess_input(l.ok()?).ok())
        .collect::<Vec<String>>();
    let mapping = mapping
        .unwrap_or_else(|| parser::NewickParser::get_taxa_mapping(&lines[0]));
    let reference_distribution_bipartitions: Vec<Vec<BitVec>> = lines
        .iter()
        .map(|l| -> Result<Vec<BitVec>> {
            Ok(parser::NewickParser::new(l, &mapping).parse())
        })
        .collect::<Result<Vec<Vec<BitVec>>>>()?;

    let reference_metrics =
        if let Some(bipartitions) = reference_tree_bipartitions {
            let reference_metrics =
                metrics::compare_distribution_against_reference_tree(
                    &reference_distribution_bipartitions,
                    &bipartitions,
                )?;
            Some(reference_metrics)
        } else {
            None
        };

    let reference_distribution_bipartitions =
        vec![reference_distribution_bipartitions
            .into_iter()
            .flatten()
            .collect_vec()];

    let bipartitions_per_chain = other_tools
        .iter()
        .map(|t| {
            let file = File::open(t.distribution_path.clone())
                .expect("Failed to open file");
            let lines: Vec<String> = BufReader::new(file)
                .lines()
                .map_while(|l| {
                    parser::NewickParser::preprocess_input(l.ok()?).ok()
                })
                .collect::<Vec<String>>();
            lines
                .iter()
                .flat_map(|l| parser::NewickParser::new(l, &mapping).parse())
                .collect_vec()
        })
        .collect_vec();
    let bipartitions_per_chain = reference_distribution_bipartitions
        .into_iter()
        .chain(bipartitions_per_chain.into_iter())
        .collect_vec();
    let metrics_data = metrics::MetricsData::new(&bipartitions_per_chain)?;
    let metrics: Vec<metrics::DistanceMetrics> = (1..bipartitions_per_chain
        .len())
        .map(|i| metrics_data.distance_metrics(0, i, cutoff))
        .collect::<Result<Vec<metrics::DistanceMetrics>>>()?;
    Ok((reference_metrics, metrics))
}
