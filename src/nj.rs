use ::logging_timer::time;
use anyhow::Result;
use itertools::Itertools;
use ordered_float::NotNan;
use rand::{seq::IteratorRandom, seq::SliceRandom};

use crate::{
    datastructures::{DistanceMatrix, PhyloTree},
    distance_distribution::DistanceMatrixSamples,
};

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
    let (&i, &j) = active_indices
        .choose_weighted(rng, |p| (-q(p)).exp())
        .unwrap();
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

#[inline]
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

#[derive(Copy, Clone, clap::ValueEnum, PartialEq)]
pub enum RandomizationStrategy {
    WeightedSelection,
    ThresholdBasedRandomization,
    RandomSampling,
    Deterministic,
}

#[time("info")]
pub fn resampling_nj(
    mut distance_matrix_samples: DistanceMatrixSamples,
    rng: &mut impl rand::Rng,
    ratio: f64,
) -> Result<PhyloTree> {
    let n = distance_matrix_samples.num_taxa();
    let mut active: Vec<usize> = (0..n).collect();
    let mut trees: Vec<Option<PhyloTree>> = distance_matrix_samples
        .labels()
        .map(|name| Some(PhyloTree::new_leaf(name)))
        .collect();
    let mut distance_matrix = distance_matrix_samples.ml_distances()?;
    while active.len() > 3 {
        let sum_d = {
            let mut sum_d = vec![0.0; n];
            active.iter().for_each(|i: &usize| {
                sum_d[*i] = active
                    .iter()
                    .filter(|&&k| k != *i)
                    .map(|&k| distance_matrix.get(*i, k))
                    .sum()
            });
            sum_d
        };

        let q = |&(&i, &j): &(&usize, &usize)| -> NotNan<f64> {
            debug_assert!(i < j);
            unsafe {
                NotNan::new_unchecked(
                    (active.len() - 2) as f64 * distance_matrix.get_lt(i, j)
                        - sum_d[i]
                        - sum_d[j],
                )
            }
        };
        let (i, j) = deterministic_min(&active, q);
        debug_assert!(i < j);
        let (d_i, d_j) = {
            let mut d_i = (distance_matrix.get_lt(i, j) / 2.
                + (sum_d[i] - sum_d[j]) / (2. * (active.len() - 2) as f64))
                .max(0.0);
            let mut d_j = distance_matrix.get_lt(i, j) - d_i;
            if d_j < 0.0 {
                d_i = (d_i + d_j).max(0.0);
                d_j = 0.0;
            }
            (d_i, d_j)
        };
        debug_assert!(d_i >= 0.0 && d_j >= 0.0);

        active.remove(active.iter().position(|&x| x == j).unwrap());
        distance_matrix_samples.set(
            i,
            j,
            crate::distance_distribution::DistanceDistributionSamples::Fixed(
                distance_matrix.get_lt(i, j),
            ),
        );
        active.iter().filter(|&&k| k != i).for_each(|&k| {
            distance_matrix_samples.update(i, j, k, rng);
        });

        trees[i] = Some(PhyloTree::join(vec![
            (trees[i].take().unwrap(), d_i),
            (trees[j].take().unwrap(), d_j),
        ]));

        distance_matrix = distance_matrix_samples.sample(
            rng,
            ratio,
            distance_matrix.distances(),
        )?
    }

    // finalize remaining 3 nodes
    if let [i, j, k] = active[..] {
        let (d_i, d_j, d_k) = {
            let mut d_i = ((distance_matrix.get_lt(i, j)
                + distance_matrix.get_lt(i, k)
                - distance_matrix.get_lt(j, k))
                / 2.)
                .max(0.0);
            let mut d_j = distance_matrix.get_lt(i, j) - d_i;
            if d_j < 0.0 {
                d_i = (d_i + d_j).max(0.0);
                d_j = 0.0;
            }
            let mut d_k = distance_matrix.get_lt(i, k) - d_i;
            if d_k < 0.0 {
                d_i = (d_i + d_k).max(0.0);
                d_k = 0.0;
            }
            (d_i, d_j, d_k)
        };
        debug_assert!(d_i >= 0.0 && d_j >= 0.0 && d_k >= 0.0);
        trees[i] = Some(PhyloTree::join(vec![
            (trees[i].take().unwrap(), d_i),
            (trees[j].take().unwrap(), d_j),
            (trees[k].take().unwrap(), d_k),
        ]))
    }
    Ok(trees[0].take().unwrap())
}

#[time("info")]
pub fn nj(
    mut distance_matrix: DistanceMatrix,
    strategy: RandomizationStrategy,
    rng: &mut impl rand::Rng,
    percentile: f64,
) -> Result<PhyloTree> {
    let n = distance_matrix.num_taxa();
    let mut active: Vec<usize> = (0..n).collect();
    let mut trees: Vec<Option<PhyloTree>> = distance_matrix
        .labels()
        .map(|name| Some(PhyloTree::new_leaf(name)))
        .collect();
    while active.len() > 3 {
        let sum_d = {
            let mut sum_d = vec![0.0; n];
            active.iter().for_each(|i: &usize| {
                sum_d[*i] = active
                    .iter()
                    .filter(|&&k| k != *i)
                    .map(|&k| distance_matrix.get(*i, k))
                    .sum()
            });
            sum_d
        };

        let q = |&(&i, &j): &(&usize, &usize)| -> NotNan<f64> {
            debug_assert!(i < j);
            unsafe {
                NotNan::new_unchecked(
                    (active.len() - 2) as f64 * distance_matrix.get_lt(i, j)
                        - sum_d[i]
                        - sum_d[j],
                )
            }
        };
        let (i, j) = match strategy {
            RandomizationStrategy::WeightedSelection => {
                weighted_min_pos(&active, q, rng)
            }
            RandomizationStrategy::ThresholdBasedRandomization => {
                random_min_by_threshold(&active, q, rng, percentile)
            }
            RandomizationStrategy::RandomSampling => {
                min_from_sample(&active, q, rng, percentile)
            }
            RandomizationStrategy::Deterministic => {
                deterministic_min(&active, q)
            }
        };
        debug_assert!(i < j);
        let (d_i, d_j) = {
            let mut d_i = (distance_matrix.get_lt(i, j) / 2.
                + (sum_d[i] - sum_d[j]) / (2. * (active.len() - 2) as f64))
                .max(0.0);
            let mut d_j = distance_matrix.get_lt(i, j) - d_i;
            if d_j < 0.0 {
                d_i = (d_i + d_j).max(0.0);
                d_j = 0.0;
            }
            (d_i, d_j)
        };
        debug_assert!(d_i >= 0.0 && d_j >= 0.0);

        active.remove(active.iter().position(|&x| x == j).unwrap());
        active.iter().filter(|&&k| k != i).for_each(|&k| {
            distance_matrix.update(i, j, k);
        });

        trees[i] = Some(PhyloTree::join(vec![
            (trees[i].take().unwrap(), d_i),
            (trees[j].take().unwrap(), d_j),
        ]));
    }

    // finalize remaining 3 nodes
    if let [i, j, k] = active[..] {
        let (d_i, d_j, d_k) = {
            let mut d_i = ((distance_matrix.get_lt(i, j)
                + distance_matrix.get_lt(i, k)
                - distance_matrix.get_lt(j, k))
                / 2.)
                .max(0.0);
            let mut d_j = distance_matrix.get_lt(i, j) - d_i;
            if d_j < 0.0 {
                d_i = (d_i + d_j).max(0.0);
                d_j = 0.0;
            }
            let mut d_k = distance_matrix.get_lt(i, k) - d_i;
            if d_k < 0.0 {
                d_i = (d_i + d_k).max(0.0);
                d_k = 0.0;
            }
            (d_i, d_j, d_k)
        };
        debug_assert!(d_i >= 0.0 && d_j >= 0.0 && d_k >= 0.0);
        trees[i] = Some(PhyloTree::join(vec![
            (trees[i].take().unwrap(), d_i),
            (trees[j].take().unwrap(), d_j),
            (trees[k].take().unwrap(), d_k),
        ]))
    }
    Ok(trees[0].take().unwrap())
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::datastructures::DistanceMatrix;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoroshiro128PlusPlus;

    use super::nj;

    #[test]
    fn test_nj_step() {
        let raw_matrix = r"4
                           taxon0 0.0 17.0 21.0 27.0
                           taxon1 17.0 0.0 12.0 18.0
                           taxon2 21.0 12.0 0.0 14.0
                           taxon3 27.0 18.0 14.0 0.0";
        let matrix = DistanceMatrix::from_str(raw_matrix).unwrap();
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
        let result = nj(
            matrix,
            super::RandomizationStrategy::Deterministic,
            &mut rng,
            0.0,
        )
        .unwrap();
        assert_eq!(
            result.to_string(),
            "((taxon0:13,taxon1:4):4,taxon2:4,taxon3:10);"
        );
    }

    #[test]
    fn test_nj_step_2() {
        let raw_matrix = r"4
                           taxon0 0.0 5.0 9.0 9.0 8.0
                           taxon1 5.0 0.0 10.0 10.0 9.0
                           taxon2 9.0 10 0.0 8.0 7.0
                           taxon3 9.0 10.0 8.0 0.0 3.0
                           taxon4 8.0 9.0 7.0 3.0 0.0";
        let matrix = DistanceMatrix::from_str(raw_matrix).unwrap();
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
        let result = nj(
            matrix,
            super::RandomizationStrategy::Deterministic,
            &mut rng,
            0.0,
        )
        .unwrap();
        assert_eq!(
            result.to_string(),
            "(((taxon0:2,taxon1:3):3,taxon2:4):2,taxon3:2,taxon4:1);"
        );
    }
}
