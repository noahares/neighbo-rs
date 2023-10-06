use std::{
    collections::HashSet,
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use anyhow::{Context, Result};
use bitvec::vec::BitVec;
use counter::Counter;
use itertools::{izip, Either, Itertools};
use logging_timer::{time, timer, Level};
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use serde::Serialize;

use crate::{
    datastructures::{Metadata, Tool},
    io, parser,
};

#[derive(Debug, Clone)]
pub struct MetricsData {
    bipartitions_per_chain: Vec<Vec<BitVec>>,
    bipartition_freqs_per_chain: Vec<Vec<f64>>,
}

impl MetricsData {
    pub fn new(bipartitions_per_chain: &[Vec<BitVec>]) -> Result<Self> {
        let counters: Vec<Counter<_>> = bipartitions_per_chain
            .iter()
            .map(|b| b.iter().collect::<Counter<_>>())
            .collect();
        let all_bipartitions: HashSet<BitVec> = bipartitions_per_chain
            .iter()
            .flat_map(|b| b.clone())
            .collect();
        let bipartition_freqs_per_chain: Vec<Vec<f64>> =
            bipartitions_per_chain
                .iter()
                .zip_eq(counters.iter())
                .map(|(biparts, counter)| {
                    all_bipartitions
                        .iter()
                        .map(|b| (counter[&b] as f64) / (biparts.len() as f64))
                        .collect()
                })
                .collect_vec();
        debug_assert!(bipartition_freqs_per_chain.iter().all(|freqs| (freqs
            .iter()
            .sum::<f64>()
            - 1.0)
            .abs()
            <= 1e-7));
        Ok(Self {
            bipartitions_per_chain: bipartitions_per_chain.to_vec(),
            bipartition_freqs_per_chain,
        })
    }

    #[time("info")]
    pub fn distance_metrics(
        &self,
        chain_index_a: usize,
        chain_index_b: usize,
        cutoff: f64,
    ) -> Result<DistanceMetrics> {
        let (
            unique_bipartition_ratio,
            num_bipartitions,
            unique_bipartition_ratio_per_chain,
        ) = self.unique_bipartition_stats(chain_index_a, chain_index_b)?;
        Ok(DistanceMetrics {
            simple_distance: self
                .simple_distance(chain_index_a, chain_index_b)?,
            hellinger_distance: self
                .hellinger_distance(chain_index_a, chain_index_b)?,
            asdsf: self.asdsf(chain_index_a, chain_index_b)?,
            consensus_distance: self.consensus_rf(
                chain_index_a,
                chain_index_b,
                cutoff,
            )?,
            pearson_correlation_coefficient: self
                .pearson_correlation_coefficient(
                    chain_index_a,
                    chain_index_b,
                )?,
            unique_bipartition_ratio,
            unique_bipartition_ratio_per_chain,
            num_bipartitions,
        })
    }

    pub fn simple_distance(
        &self,
        chain_index_a: usize,
        chain_index_b: usize,
    ) -> Result<f64> {
        Ok(0.5f64
            * self.bipartition_freqs_per_chain[chain_index_a]
                .iter()
                .zip_eq(self.bipartition_freqs_per_chain[chain_index_b].iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum::<f64>())
    }

    pub fn hellinger_distance(
        &self,
        chain_index_a: usize,
        chain_index_b: usize,
    ) -> Result<f64> {
        Ok((0.5f64
            * self.bipartition_freqs_per_chain[chain_index_a]
                .iter()
                .zip_eq(self.bipartition_freqs_per_chain[chain_index_b].iter())
                .map(|(&a, &b)| (a.sqrt() - b.sqrt()).powi(2))
                .sum::<f64>())
        .sqrt())
    }

    pub fn asdsf(
        &self,
        chain_index_a: usize,
        chain_index_b: usize,
    ) -> Result<f64> {
        let mut squared_sum = 0.0f64;
        Ok((self.bipartition_freqs_per_chain[chain_index_a]
            .iter()
            .zip_eq(self.bipartition_freqs_per_chain[chain_index_b].iter())
            .map(|(&a, &b)| {
                squared_sum += a * a + b * b;
                (a - b).powi(2)
            })
            .sum::<f64>()
            / squared_sum)
            .sqrt())
    }

    pub fn pearson_correlation_coefficient(
        &self,
        chain_index_a: usize,
        chain_index_b: usize,
    ) -> Result<f64> {
        let (relevant_biparts_a, relevant_biparts_b): (Vec<f64>, Vec<f64>) =
            self.bipartition_freqs_per_chain[chain_index_a]
                .iter()
                .zip_eq(self.bipartition_freqs_per_chain[chain_index_b].iter())
                .filter(|(&a, &b)| -> bool { a > 0.0 || b > 0.0 })
                .unzip();
        debug_assert_eq!(relevant_biparts_a.len(), relevant_biparts_b.len());
        Ok(Array2::from_shape_vec(
            (2, relevant_biparts_a.len()),
            relevant_biparts_a
                .into_iter()
                .chain(relevant_biparts_b.into_iter())
                .collect_vec(),
        )?
        .pearson_correlation()?[(0, 1)]
            .to_owned())
    }

    pub fn consensus_rf(
        &self,
        chain_index_a: usize,
        chain_index_b: usize,
        cutoff: f64,
    ) -> Result<f64> {
        let unique_count_a = self.bipartitions_per_chain[chain_index_a]
            .iter()
            .unique()
            .count();
        let unique_count_b = self.bipartitions_per_chain[chain_index_b]
            .iter()
            .unique()
            .count();
        let (consensus_bipartitions_a, remainder_a) = consensus_bipartitions(
            &self.bipartitions_per_chain[chain_index_a],
            cutoff,
        )?;
        let (consensus_bipartitions_b, remainder_b) = consensus_bipartitions(
            &self.bipartitions_per_chain[chain_index_b],
            cutoff,
        )?;
        let consensus_set_a: HashSet<&BitVec> =
            HashSet::from_iter(consensus_bipartitions_a.iter());
        let consensus_set_b: HashSet<&BitVec> =
            HashSet::from_iter(consensus_bipartitions_b.iter());
        let symmetric_difference = Iterator::count(
            consensus_set_a.symmetric_difference(&consensus_set_b),
        ) + remainder_a.len()
            + remainder_b.len();
        Ok(symmetric_difference as f64
            / (unique_count_a + unique_count_b) as f64)
    }

    pub fn unique_bipartition_stats(
        &self,
        chain_index_a: usize,
        chain_index_b: usize,
    ) -> Result<(f64, usize, (f64, f64))> {
        let bipartition_sets: Vec<HashSet<&BitVec>> = [
            self.bipartitions_per_chain[chain_index_a].iter(),
            self.bipartitions_per_chain[chain_index_b].iter(),
        ]
        .into_iter()
        .map(HashSet::from_iter)
        .collect_vec();
        let num_biparts =
            bipartition_sets[0].union(&bipartition_sets[1]).count();
        let unique_bipartitions_ratio = (bipartition_sets[0]
            .symmetric_difference(&bipartition_sets[1])
            .count() as f64)
            / (num_biparts as f64);
        let unique_bipartitions_per_chain = (
            (bipartition_sets[0].difference(&bipartition_sets[1]).count()
                as f64)
                / (bipartition_sets[0].len() as f64),
            (bipartition_sets[1].difference(&bipartition_sets[0]).count()
                as f64)
                / (bipartition_sets[1].len() as f64),
        );

        Ok((
            unique_bipartitions_ratio,
            num_biparts,
            unique_bipartitions_per_chain,
        ))
    }
}

#[derive(Default, Debug, Clone, Serialize)]
pub struct DistanceMetrics {
    pub simple_distance: f64,
    pub hellinger_distance: f64,
    pub asdsf: f64,
    pub consensus_distance: f64,
    pub pearson_correlation_coefficient: f64,
    pub unique_bipartition_ratio: f64,
    pub unique_bipartition_ratio_per_chain: (f64, f64),
    pub num_bipartitions: usize,
}

impl Display for DistanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\nSimple Distance: {}\
               \nHellinger Distance: {}\
               \n ASDSF: {}\
               \nPCC: {}\
               \nUBR: ({}, {})\
               \nPUBR: {}\
               \nNumber of Bipartitions: {}",
            self.simple_distance,
            self.hellinger_distance,
            self.asdsf,
            self.pearson_correlation_coefficient,
            self.unique_bipartition_ratio,
            self.unique_bipartition_ratio_per_chain.0,
            self.unique_bipartition_ratio_per_chain.1,
            self.num_bipartitions
        )
    }
}

pub fn rf_distance(
    bipartitions_a: &[BitVec],
    bipartitions_b: &[BitVec],
) -> Result<f64> {
    let set_a: HashSet<&BitVec> = HashSet::from_iter(bipartitions_a.iter());
    let set_b: HashSet<&BitVec> = HashSet::from_iter(bipartitions_b.iter());
    let symmetric_difference =
        Iterator::count(set_a.symmetric_difference(&set_b));
    Ok(symmetric_difference as f64 / (set_a.len() + set_b.len()) as f64)
}

pub fn missed_splits_ratio(
    distribution_bipartitions: &[BitVec],
    reference_bipartitions: &[BitVec],
) -> Result<f64> {
    let distribution_set: HashSet<&BitVec> =
        HashSet::from_iter(distribution_bipartitions.iter());
    let reference_set: HashSet<&BitVec> =
        HashSet::from_iter(reference_bipartitions.iter());
    let missed_splits =
        Iterator::count(reference_set.difference(&distribution_set));
    Ok(missed_splits as f64 / reference_set.len() as f64)
}

pub fn consensus_bipartitions(
    bipartitions: &[BitVec],
    cutoff: f64,
) -> Result<(Vec<BitVec>, Vec<BitVec>)> {
    // NOTE: this is kinda ugly to get the number of trees <noahares>
    let num_trees = bipartitions.len() / (bipartitions[0].len() - 3);
    Ok(bipartitions.iter().counts().into_iter().partition_map(
        |(b, c)| match c as f64 / num_trees as f64 >= cutoff {
            true => Either::Left(b.clone()),
            false => Either::Right(b.clone()),
        },
    ))
}

#[derive(Debug, Default, Clone)]
pub struct RFDistanceStats {
    pub min: f64,
    pub mean: f64,
    pub max: f64,
}

#[derive(Debug, Default, Clone)]
pub struct ReferenceTreeMetrics {
    pub missed_splits_ratio: f64,
    pub rf_distance_stats: RFDistanceStats,
}

#[time("debug")]
pub fn compare_distribution_against_reference_tree(
    distribution_bipartitions: &[Vec<BitVec>],
    reference_tree_bipartitions: &[BitVec],
) -> Result<ReferenceTreeMetrics> {
    let reference_tree_set: HashSet<BitVec> = reference_tree_bipartitions
        .iter()
        .map(BitVec::to_owned)
        .collect();
    let rf_distances = distribution_bipartitions
        .iter()
        .map(|b| {
            let biparts: HashSet<BitVec> =
                b.iter().map(BitVec::to_owned).collect();
            let symmetric_difference = Iterator::count(
                reference_tree_set.symmetric_difference(&biparts),
            );
            symmetric_difference as f64
                / (reference_tree_set.len() + biparts.len()) as f64
        })
        .collect_vec();
    let distribution_set = distribution_bipartitions
        .iter()
        .flat_map(|t| t.iter().map(BitVec::to_owned))
        .unique()
        .collect_vec();
    Ok(ReferenceTreeMetrics {
        missed_splits_ratio: missed_splits_ratio(
            &distribution_set,
            reference_tree_bipartitions,
        )?,
        rf_distance_stats: RFDistanceStats {
            min: *rf_distances
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .context("Empty list of RF-distances.")?,
            mean: rf_distances.iter().sum::<f64>() / rf_distances.len() as f64,
            max: *rf_distances
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .context("Empty list of RF-distances.")?,
        },
    })
}

#[time("debug")]
pub fn evaluate_dataset(
    reference_path: Option<PathBuf>,
    reference_tool: &Tool,
    other_tools: &[Tool],
    cutoff: f64,
    metadata: Vec<Metadata>,
) -> Result<Vec<io::Metrics>> {
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
        File::open(reference_tool.distribution_path.clone())?;
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

    let reference_metrics = if let Some(bipartitions) =
        &reference_tree_bipartitions
    {
        let reference_metrics = compare_distribution_against_reference_tree(
            &reference_distribution_bipartitions,
            bipartitions,
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

    let bipartitions_per_chain = {
        let _tmr = timer!(Level::Info; "Parse distributions", "Parsed {} distributions", other_tools.len());
        other_tools
            .iter()
            .map(|t| {
                let file = File::open(t.distribution_path.clone())
                    .expect("Failed to open file");
                BufReader::new(file)
                    .lines()
                    .map_while(|l| {
                        parser::NewickParser::preprocess_input(l.ok()?).ok()
                    })
                    .map(|l| parser::NewickParser::new(&l, &mapping).parse())
                    .collect_vec()
            })
            .collect_vec()
    };

    let tool_reference_metrics = if let Some(bipartitions) =
        reference_tree_bipartitions
    {
        bipartitions_per_chain
            .iter()
            .map(|b| {
                compare_distribution_against_reference_tree(b, &bipartitions)
                    .ok()
            })
            .collect()
    } else {
        vec![None; bipartitions_per_chain.len() - 1]
    };

    let bipartitions_per_chain = reference_distribution_bipartitions
        .into_iter()
        .chain(
            bipartitions_per_chain
                .into_iter()
                .map(|c| c.into_iter().flatten().collect()),
        )
        .collect_vec();
    let metrics_data = MetricsData::new(&bipartitions_per_chain)?;
    izip!(
        (1..bipartitions_per_chain.len())
            .map(|i| metrics_data.distance_metrics(0, i, cutoff)),
        metadata.iter(),
        tool_reference_metrics.into_iter()
    )
    .map(|(m, meta, tm)| -> Result<io::Metrics> {
        Ok(io::Metrics::from((
            meta.clone(),
            reference_metrics.clone(),
            tm,
            m?,
        )))
    })
    .collect::<Result<Vec<io::Metrics>>>()
}

#[cfg(test)]
mod tests {
    use bitvec::prelude::*;

    use crate::metrics::{missed_splits_ratio, rf_distance, MetricsData};

    #[test]
    fn test_rf_distance() {
        let bipartitions_a = vec![
            bitvec![0, 0, 1],
            bitvec![0, 1, 0],
            bitvec![0, 1, 1],
            bitvec![1, 0, 0],
        ];
        let bipartitions_b = vec![
            bitvec![0, 0, 1],
            bitvec![0, 1, 0],
            bitvec![1, 1, 0],
            bitvec![1, 1, 1],
        ];
        assert_float_absolute_eq!(
            rf_distance(&bipartitions_a, &bipartitions_b).unwrap(),
            0.5f64
        );
    }

    #[test]
    fn test_missed_split_ratio() {
        let bipartitions_a = vec![
            bitvec![0, 0, 1],
            bitvec![0, 1, 0],
            bitvec![0, 1, 1],
            bitvec![1, 0, 0],
            bitvec![1, 1, 0],
        ];
        let bipartitions_b = vec![
            bitvec![0, 0, 1],
            bitvec![0, 1, 0],
            bitvec![1, 1, 1],
            bitvec![1, 0, 1],
        ];
        assert_float_absolute_eq!(
            missed_splits_ratio(&bipartitions_a, &bipartitions_b).unwrap(),
            0.5f64
        );
    }

    #[test]
    fn test_metrics_datastructures() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 0],
                bitvec![1, 1, 1],
            ],
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 1],
                bitvec![0, 1, 1],
            ],
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 1],
                bitvec![1, 0, 0],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_eq!(
            metrics_data
                .bipartition_freqs_per_chain
                .into_iter()
                .map(|b| {
                    let mut to_sort = b.clone();
                    to_sort.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    to_sort
                })
                .collect::<Vec<Vec<f64>>>(),
            vec![
                vec![0.0, 0.0, 0.2, 0.4, 0.4],
                vec![0.0, 0.0, 0.25, 0.25, 0.5],
                vec![0.0, 0.0, 0.25, 0.25, 0.5]
            ]
        );
    }

    #[test]
    fn test_simple_distance() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 0],
                bitvec![1, 1, 1],
            ],
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 1],
                bitvec![0, 1, 1],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_float_absolute_eq!(
            metrics_data.simple_distance(0, 1).unwrap(),
            0.5f64
        );
    }

    #[test]
    fn test_max_simple_distance() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 0],
                bitvec![1, 1, 1],
            ],
            vec![
                bitvec![0, 1, 1],
                bitvec![0, 1, 1],
                bitvec![1, 1, 0],
                bitvec![1, 0, 1],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_float_absolute_eq!(
            metrics_data.simple_distance(0, 1).unwrap(),
            1.0f64
        );
    }

    #[test]
    fn test_hellinger_distance() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 0],
                bitvec![1, 1, 1],
            ],
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 1],
                bitvec![0, 1, 1],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_float_absolute_eq!(
            metrics_data.hellinger_distance(0, 1).unwrap(),
            0.60625446f64
        );
    }

    #[test]
    fn test_max_hellinger_distance() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 0],
                bitvec![1, 1, 1],
            ],
            vec![
                bitvec![0, 1, 1],
                bitvec![0, 1, 1],
                bitvec![1, 1, 0],
                bitvec![1, 0, 1],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_float_absolute_eq!(
            metrics_data.hellinger_distance(0, 1).unwrap(),
            1.0f64
        );
    }

    #[test]
    fn test_asdsf() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 0],
                bitvec![1, 1, 1],
            ],
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 1],
                bitvec![0, 1, 1],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_float_absolute_eq!(
            metrics_data.asdsf(0, 1).unwrap(),
            0.675116518f64
        );
    }

    #[test]
    fn test_max_asdsf() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 0],
                bitvec![1, 1, 1],
            ],
            vec![
                bitvec![0, 1, 1],
                bitvec![0, 1, 1],
                bitvec![1, 1, 0],
                bitvec![1, 0, 1],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_float_absolute_eq!(metrics_data.asdsf(0, 1).unwrap(), 1.0);
    }

    #[test]
    fn test_max_pcc() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 0],
                bitvec![1, 1, 1],
            ],
            vec![
                bitvec![0, 0, 1],
                bitvec![0, 0, 1],
                bitvec![0, 1, 0],
                bitvec![0, 1, 0],
                bitvec![1, 1, 1],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_float_absolute_eq!(
            metrics_data.pearson_correlation_coefficient(0, 1).unwrap(),
            1.0
        );
    }

    #[test]
    fn test_max_consensus_distance() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 1, 1],
                bitvec![0, 0, 1, 1],
                bitvec![0, 1, 1, 1],
            ],
            vec![
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 1, 0],
                bitvec![0, 0, 1, 0],
                bitvec![0, 0, 1, 0],
                bitvec![0, 1, 1, 1],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_float_absolute_eq!(
            metrics_data.consensus_rf(0, 1, 0.5).unwrap(),
            1.0
        );
    }

    #[test]
    fn test_min_consensus_distance() {
        let bipartitions = vec![
            vec![
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 1, 0],
                bitvec![0, 0, 1, 0],
                bitvec![0, 0, 1, 0],
            ],
            vec![
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 0, 1],
                bitvec![0, 0, 1, 0],
                bitvec![0, 0, 1, 0],
                bitvec![0, 0, 1, 0],
            ],
        ];
        let metrics_data = MetricsData::new(&bipartitions).unwrap();
        assert_float_absolute_eq!(
            metrics_data.consensus_rf(0, 1, 0.5).unwrap(),
            0.0
        );
    }
}
