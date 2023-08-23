use anyhow::Result;
use core::{fmt, str::FromStr};
use itertools::Itertools;
use logging_timer::time;
use std::path::Path;

#[derive(Clone)]
pub struct DistanceMatrix {
    labels: Vec<String>,
    distances: Vec<f64>,
}

impl DistanceMatrix {
    fn new(labels: Vec<String>, distances: Vec<f64>) -> Self {
        Self { labels, distances }
    }

    #[inline]
    fn index_from_row_and_col(i: usize, j: usize, n: usize) -> usize {
        let (i, j) = (i.min(j), i.max(j));
        (i * n) - (i * (i + 3) / 2) + (j - 1)
    }

    // optimised version if i < j is guaranteed
    #[inline]
    fn index_from_row_and_col_lt(i: usize, j: usize, n: usize) -> usize {
        (i * n) - (i * (i + 3) / 2) + (j - 1)
    }

    pub fn labels(&self) -> std::slice::Iter<String> {
        self.labels.iter()
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert_ne!(i, j);
        self.distances[Self::index_from_row_and_col(i, j, self.num_taxa())]
    }

    // optimised version if i < j is guaranteed
    #[inline]
    pub fn get_lt(&self, i: usize, j: usize) -> f64 {
        self.distances[Self::index_from_row_and_col_lt(i, j, self.num_taxa())]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        debug_assert_ne!(i, j);
        self.distances
            [Self::index_from_row_and_col(i, j, self.labels.len())] = v;
    }
    pub fn from_file(p: &Path) -> Result<Self> {
        std::fs::read_to_string(p)?.parse()
    }

    pub fn num_taxa(&self) -> usize {
        self.labels.len()
    }

    #[time("info")]
    #[inline]
    pub fn perturb(
        &mut self,
        rng: &mut impl rand::Rng,
        distribution: &impl rand::distributions::Distribution<f64>,
        ratio: f64,
        common_noise: bool,
    ) {
        let single_noise = if common_noise {
            distribution.sample(rng).abs()
        } else {
            0.0
        };
        self.distances.iter_mut().for_each(|i| {
            if *i > 0.0 && (ratio == 1.0 || rng.gen_bool(ratio)) {
                *i *= if common_noise {
                    single_noise
                } else {
                    distribution.sample(rng).abs()
                }
            }
        });
    }

    #[inline(always)]
    pub fn update(&mut self, i: usize, j: usize, k: usize) {
        let d_k = (self.get(i, k) + self.get(j, k) - self.get_lt(i, j)) / 2.;
        self.set(i, k, d_k);
    }
}

impl FromStr for DistanceMatrix {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let mut lines = s.lines().filter(|&line| !line.is_empty());
        let n = lines.next().expect("Expected n").parse()?;

        let mut d =
            DistanceMatrix::new(Vec::with_capacity(n), Vec::with_capacity(n));

        for (i, line) in lines.enumerate() {
            let mut it = line.split_ascii_whitespace();
            d.labels.push(it.next().unwrap().to_string());
            d.distances.extend(
                it.enumerate()
                    .filter(|(j, _)| i < *j)
                    .map(|(_, chars)| chars.parse::<f64>().unwrap()),
            );
        }

        Ok(d)
    }
}

pub struct PhyloTree {
    name: String,
    children: Vec<(PhyloTree, Option<f64>)>,
}

impl PhyloTree {
    fn new(name: &str, children: Vec<(Self, Option<f64>)>) -> Self {
        Self {
            name: name.to_string(),
            children,
        }
    }

    pub fn new_leaf(name: &str) -> Self {
        Self::new(name, vec![])
    }

    pub fn join(name: &str, nodes: Vec<(Self, f64)>) -> Self {
        Self::new(
            name,
            nodes.into_iter().map(|(t, d)| (t, Some(d))).collect_vec(),
        )
    }

    fn to_string_impl(&self) -> String {
        if self.children.is_empty() {
            self.name.to_string()
        } else {
            "(".to_string()
                + &self
                    .children
                    .iter()
                    .map(|child| {
                        let (c, d) = child;
                        if let Some(d) = d {
                            Self::to_string_impl(c) + ":" + &d.to_string()
                        } else {
                            Self::to_string_impl(c)
                        }
                    })
                    .collect::<Vec<String>>()
                    .join(",")
                + ")"
                + &self.name
        }
    }
}

impl fmt::Display for PhyloTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{};", Self::to_string_impl(self))
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoroshiro128PlusPlus;

    use super::DistanceMatrix;

    #[test]
    fn test_upper_triangle_distance_matrix() {
        let raw_matrix = r"4
                           taxon0 0.0 1.0 2.0 3.0
                           taxon1 1.0 0.0 1.0 2.0
                           taxon2 2.0 1.0 0.0 1.0
                           taxon3 3.0 2.0 1.0 0.0";
        let matrix = DistanceMatrix::from_str(raw_matrix).unwrap();
        assert_eq!(matrix.distances, &[1.0, 2.0, 3.0, 1.0, 2.0, 1.0]);
        assert_eq!(matrix.get(0, 1), 1.0);
        assert_eq!(matrix.get(1, 3), 2.0);
        assert_eq!(matrix.get(3, 0), 3.0);
        assert_eq!(matrix.get(2, 3), 1.0);
    }

    #[test]
    fn test_no_perturbation() {
        let raw_matrix = r"4
                           taxon0 0.0 1.0 2.0 3.0
                           taxon1 1.0 0.0 1.0 2.0
                           taxon2 2.0 1.0 0.0 1.0
                           taxon3 3.0 2.0 1.0 0.0";
        let matrix = DistanceMatrix::from_str(raw_matrix).unwrap();
        let mut matrix_2 = matrix.clone();
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
        let distribution = rand_distr::Normal::new(1.0, 0.0).unwrap();
        matrix_2.perturb(&mut rng, &distribution, 1.0, false);
        assert_eq!(matrix.distances, matrix_2.distances);
    }
}
