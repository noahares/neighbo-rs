use anyhow::Result;
use bitvec::vec::BitVec;
use clap::Parser;
use itertools::Itertools;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[macro_use]
extern crate assert_float_eq;

mod io;
mod metrics;
mod parser;

fn main() -> Result<()> {
    let args = io::Args::parse();
    let file_paths = args.distribution_paths;
    let (mapping, reference_tree_bipartitions) = if let Some(reference_path) =
        args.reference_tree
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
    let reference_distribution_file = File::open(args.reference_distribution)?;
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

    if let Some(bipartitions) = reference_tree_bipartitions {
        let reference_metrics =
            metrics::compare_distribution_against_reference_tree(
                &reference_distribution_bipartitions,
                &bipartitions,
            )?;
        dbg!(reference_metrics);
    }

    let reference_distribution_bipartitions =
        vec![reference_distribution_bipartitions
            .into_iter()
            .flatten()
            .collect_vec()];

    let bipartitions_per_chain = file_paths
        .iter()
        .map(|f| {
            let file = File::open(f).expect("Failed to open file");
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
        .map(|i| metrics_data.distance_metrics(0, i, args.consensus_cutoff))
        .collect::<Result<Vec<metrics::DistanceMetrics>>>()?;
    for m in &metrics {
        dbg!(m);
    }
    Ok(())
}
