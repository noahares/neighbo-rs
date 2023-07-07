use anyhow::Result;
use itertools::Itertools;
use ordered_float::NotNan;
use rand::{seq::IteratorRandom, seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::datastructures::{DistanceMatrix, PhyloTree};

pub fn weighted_min_pos<F>(
    active: &[usize],
    q: F,
    rng: &mut impl rand::Rng,
) -> (usize, usize)
where
    F: Fn(&(&usize, &usize)) -> NotNan<f64>,
{
    let active_indices = active
        .iter()
        .cartesian_product(active.iter())
        .filter(|&(&i, &j)| i < j)
        .collect_vec();
    let (&i, &j) =
        active_indices.choose_weighted(rng, |p| q(p).abs()).unwrap();
    (i, j)
}

pub fn min_from_sample<F>(
    active: &[usize],
    q: F,
    rng: &mut impl rand::Rng,
    percentile: f64,
) -> (usize, usize)
where
    F: Fn(&(&usize, &usize)) -> NotNan<f64>,
{
    let num_entries = (active.len() * active.len() - 1) / 2;
    let num_samples = (num_entries as f64 * percentile).max(1.0) as usize;
    let (&i, &j) = active
        .iter()
        .cartesian_product(active.iter())
        .filter(|&(&i, &j)| i < j)
        .choose_multiple(rng, num_samples)
        .into_iter()
        .min_by_key(q)
        .unwrap();
    (i, j)
}

pub fn random_min_by_threshold<F>(
    active: &[usize],
    q: F,
    rng: &mut impl rand::Rng,
    percentile: f64,
) -> (usize, usize)
where
    F: Fn(&(&usize, &usize)) -> NotNan<f64>,
{
    let num_entries = (active.len() * active.len() - 1) / 2;
    let threshold = (num_entries as f64 * percentile) as usize;
    let mut active_indices = active
        .iter()
        .cartesian_product(active.iter())
        .filter(|&(&i, &j)| i < j)
        .collect_vec();
    active_indices.select_nth_unstable_by_key(threshold, q);
    let selected_index = rng.gen_range(0..=threshold);
    let (&i, &j) = active_indices[selected_index];
    (i, j)
}

pub fn deterministic_min<F>(active: &[usize], q: F) -> (usize, usize)
where
    F: Fn(&(&usize, &usize)) -> NotNan<f64>,
{
    let (&i, &j) = active
        .iter()
        .cartesian_product(active.iter())
        .filter(|&(&i, &j)| i < j)
        .min_by_key(q)
        .unwrap();
    (i, j)
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
        let sum_d: std::collections::HashMap<usize, f64> = active
            .iter()
            .map(|i: &usize| -> (usize, f64) {
                (*i, active.iter().map(|&k| distance_matrix.get(*i, k)).sum())
            })
            .collect();

        let q = |&(&i, &j): &(&usize, &usize)| -> NotNan<f64> {
            NotNan::new(
                (active.len() - 2) as f64 * distance_matrix.get(i, j)
                    - sum_d.get(&i).unwrap()
                    - sum_d.get(&j).unwrap(),
            )
            .unwrap()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (i, j) = match strategy {
            RandomizationStrategy::WeightedSelection => {
                weighted_min_pos(&active, q, &mut rng)
            }
            RandomizationStrategy::ThresholdBasedRandomization => {
                random_min_by_threshold(&active, q, &mut rng, percentile)
            }
            RandomizationStrategy::RandomSampling => {
                min_from_sample(&active, q, &mut rng, percentile)
            }
            RandomizationStrategy::Deterministic => {
                deterministic_min(&active, q)
            }
        };
        assert!(i < j);
        let d_i = distance_matrix.get(i, j) / 2.
            + (sum_d.get(&i).unwrap() - sum_d.get(&j).unwrap())
                / (2. * (active.len() - 2) as f64);
        let d_j = distance_matrix.get(i, j) - d_i;

        active.remove(active.iter().position(|&x| x == j).unwrap());
        active.iter().filter(|&&k| k != i).for_each(|&k| {
            let d_k = (distance_matrix.get(i, k) + distance_matrix.get(j, k)
                - distance_matrix.get(i, j))
                / 2.;
            distance_matrix.set(i, k, d_k);
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

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::datastructures::DistanceMatrix;

    use super::nj;

    #[test]
    fn test_nj_step() {
        let raw_matrix = r"4
                           taxon0 0.0 17.0 21.0 27.0
                           taxon1 17.0 0.0 12.0 18.0
                           taxon2 21.0 12.0 0.0 14.0
                           taxon3 27.0 18.0 14.0 0.0";
        let matrix = DistanceMatrix::from_str(raw_matrix).unwrap();
        let result =
            nj(matrix, super::RandomizationStrategy::Deterministic, 0, 0.0)
                .unwrap();
        // println!("{}", result);
        assert_eq!(
            result.to_string(),
            "(((taxon0:13,taxon1:4):4,taxon2:4):5,taxon3:5);"
        );
    }
}
