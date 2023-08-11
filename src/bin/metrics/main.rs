use anyhow::Result;
use bitvec::vec::BitVec;
use counter::Counter;
use itertools::Itertools;
use ndarray::{Array2, Axis};
use ndarray_stats::CorrelationExt;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

mod parser;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let file_paths = &args[1..=2];
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
            let mapping = parser::NewickParser::get_taxa_mapping(&lines[0]);
            lines
                .iter()
                .flat_map(|l| parser::NewickParser::new(l, &mapping).parse())
                .collect_vec()
        })
        .collect_vec();
    let counters: Vec<Counter<_>> = bipartitions_per_chain
        .iter()
        .map(|b| b.iter().collect::<Counter<_>>())
        .collect();
    let all_biparts: HashSet<BitVec> = bipartitions_per_chain
        .iter()
        .flat_map(|b| b.clone())
        .collect();
    // TODO: is it correct to directly normalize and not divide by num_trees per chain? <noahares>
    let bipartition_freqs_per_chain = Array2::from_shape_vec(
        (2, all_biparts.len()),
        all_biparts
            .iter()
            .map(|b| {
                (counters[0][&b] as f64)
                    / (bipartitions_per_chain[0].len() as f64)
            })
            .chain(all_biparts.iter().map(|b| {
                (counters[1][&b] as f64)
                    / (bipartitions_per_chain[1].len() as f64)
            }))
            .collect_vec(),
    )?;
    let sum_of_squared_freqs =
        bipartition_freqs_per_chain.fold(0f64, |acc, v| acc + v * v);
    let simple_distance = 0.5_f64
        * bipartition_freqs_per_chain
            .map_axis(Axis(0), |col| (col[0] - col[1]).abs())
            .sum();
    let hellinger_distance = (0.5_f64
        * bipartition_freqs_per_chain
            .map_axis(Axis(0), |col| (col[0].sqrt() - col[1].sqrt()).powi(2))
            .sum())
    .sqrt();
    let asdsf = (bipartition_freqs_per_chain
        .map_axis(Axis(0), |col| (col[0] - col[1]).powi(2))
        .sum()
        / sum_of_squared_freqs)
        .sqrt();
    let pcc = bipartition_freqs_per_chain.pearson_correlation()?;

    let bipartition_sets: Vec<HashSet<BitVec>> = bipartitions_per_chain
        .into_iter()
        .map(HashSet::from_iter)
        .collect_vec();
    let unique_bipartitions_ratio = (bipartition_sets[0]
        .symmetric_difference(&bipartition_sets[1])
        .count() as f64)
        / (all_biparts.len() as f64);
    let unique_bipartitions_per_chain = [
        (bipartition_sets[0].difference(&bipartition_sets[1]).count() as f64)
            / (bipartition_sets[0].len() as f64),
        (bipartition_sets[1].difference(&bipartition_sets[0]).count() as f64)
            / (bipartition_sets[1].len() as f64),
    ];
    dbg!(simple_distance);
    dbg!(hellinger_distance);
    dbg!(asdsf);
    dbg!(pcc[(0, 1)]);
    dbg!(unique_bipartitions_ratio);
    dbg!(unique_bipartitions_per_chain);
    dbg!(all_biparts.len());
    Ok(())
}
