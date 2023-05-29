use anyhow::Result;
use ordered_float::NotNan;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::datastructures::{DistanceMatrix, PhyloTree};

pub fn weighted_min_pos(
    active_index_weight_pairs: Vec<((usize, usize), f64)>,
    rng: &mut impl rand::Rng,
) -> Option<(usize, usize)> {
    // because q matrix values are nagative, their absolute value can be used as a weight to prefer
    // small values
    let cumulative_weights: Vec<((usize, usize), f64)> =
        active_index_weight_pairs
            .iter()
            .scan(((0, 0), 0.0), |(_, acc), &w| {
                *acc += w.1.abs();
                Some((w.0, *acc))
            })
            .collect();
    assert!(cumulative_weights.iter().all(|v| v.1 > 0.0));
    let sum_of_weights: f64 = cumulative_weights.last().copied().unwrap().1;
    let selection_threshold = rng.gen_range(0.0..=sum_of_weights);
    cumulative_weights
        .iter()
        .find(|&cw| cw.1 >= selection_threshold)
        .copied()
        .map(|selected_pos| selected_pos.0)
}

pub fn min_from_sample(
    active_index_weight_pairs: Vec<((usize, usize), f64)>,
    rng: &mut impl rand::Rng,
    percentile: f64,
) -> Option<(usize, usize)> {
    let num_samples =
        (active_index_weight_pairs.len() as f64 * percentile) as usize;
    let samples = rand::seq::index::sample(
        rng,
        active_index_weight_pairs.len(),
        num_samples,
    );
    samples
        .iter()
        .min_by_key(|v| NotNan::new(active_index_weight_pairs[*v].1).unwrap())
        .map(|selected_index| active_index_weight_pairs[selected_index].0)
}

pub fn random_min_by_threshold(
    active_index_weight_pairs: Vec<((usize, usize), f64)>,
    rng: &mut impl rand::Rng,
    percentile: f64,
) -> Option<(usize, usize)> {
    let mut active_index_weight_pairs_cloned: Vec<((usize, usize), f64)> =
        active_index_weight_pairs.to_owned();
    let threshold =
        (active_index_weight_pairs.len() as f64 * percentile) as usize;
    active_index_weight_pairs_cloned
        .select_nth_unstable_by_key(threshold, |v| NotNan::new(v.1).unwrap());
    let selected_index = rng.gen_range(0..=threshold);
    Some(active_index_weight_pairs_cloned[selected_index].0)
}

pub fn deterministic_min(
    active_index_weight_pairs: Vec<((usize, usize), f64)>,
) -> Option<(usize, usize)> {
    active_index_weight_pairs
        .iter()
        .min_by_key(|&(_, v)| NotNan::new(*v).unwrap())
        .map(|min| min.0)
}

#[derive(Copy, Clone, clap::ValueEnum)]
pub enum RandomizationStrategy {
    WeightedSelection,
    ThresholdBasedRandomization,
    RandomSampling,
    Deterministic,
}

pub fn nj(
    mut distance_matrix: DistanceMatrix,
    strategy: RandomizationStrategy,
    seed: u64,
    percentile: f64,
) -> Result<PhyloTree> {
    let n = distance_matrix.num_taxa();
    let mut active: Vec<usize> = (0..n).collect();
    let mut trees: Vec<Option<PhyloTree>> = distance_matrix
        .labels()
        .map(|name| Some(PhyloTree::new_leaf(name)))
        .collect();
    while active.len() > 2 {
        let sum_d = |i: usize| -> f64 {
            active.iter().map(|&k| distance_matrix.get(i, k)).sum()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let active_index_weight_pairs =
            distance_matrix.q_values_with_active_indices(&active);
        let (i, j) = match strategy {
            RandomizationStrategy::WeightedSelection => {
                weighted_min_pos(active_index_weight_pairs, &mut rng).unwrap()
            }
            RandomizationStrategy::ThresholdBasedRandomization => {
                random_min_by_threshold(
                    active_index_weight_pairs,
                    &mut rng,
                    percentile,
                )
                .unwrap()
            }
            RandomizationStrategy::RandomSampling => min_from_sample(
                active_index_weight_pairs,
                &mut rng,
                percentile,
            )
            .unwrap(),
            RandomizationStrategy::Deterministic => {
                deterministic_min(active_index_weight_pairs).unwrap()
            }
        };
        assert!(i < j);
        // let (i, j) = (i.min(j), i.max(j));
        let d_i = distance_matrix.get(i, j) / 2.
            + (sum_d(i) - sum_d(j)) / (2. * (active.len() - 2) as f64);
        let d_j = distance_matrix.get(i, j) - d_i;

        active.remove(active.iter().position(|&x| x == j).unwrap());
        active.iter().filter(|&&k| k != i).for_each(|&k| {
            let d_k = (distance_matrix.get(i, k) + distance_matrix.get(j, k)
                - distance_matrix.get(i, j))
                / 2.;
            distance_matrix.set(i, k, d_k);
            // distance_matrix.set(k, i, d_k);
        });

        trees[i] = Some(PhyloTree::join(
            "",
            (trees[i].take().unwrap(), d_i),
            (trees[j].take().unwrap(), d_j),
        ));
    }

    // finalize remaining 2 nodes
    if let [i, j] = active[..] {
        let d = distance_matrix.get(i, j) / 2.;
        trees[i] = Some(PhyloTree::join(
            "",
            (trees[i].take().unwrap(), d),
            (trees[j].take().unwrap(), d),
        ))
    }
    Ok(trees[0].take().unwrap())
}
