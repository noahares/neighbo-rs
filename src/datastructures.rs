use anyhow::Result;
use core::{fmt, str::FromStr};
use itertools::Itertools;
use rand_distr::num_traits::ToPrimitive;
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

    fn index_from_row_and_col(i: usize, j: usize, n: usize) -> usize {
        let (i, j) = (i.min(j), i.max(j));
        (i * n) - (i * (i + 3) / 2) + (j - 1)
    }

    pub fn labels(&self) -> std::slice::Iter<String> {
        self.labels.iter()
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i == j {
            0.0
        } else {
            self.distances[Self::index_from_row_and_col(i, j, self.num_taxa())]
        }
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        assert_ne!(i, j);
        self.distances
            [Self::index_from_row_and_col(i, j, self.labels.len())] = v;
    }
    pub fn from_file(p: &Path) -> Result<Self> {
        std::fs::read_to_string(p)?.parse()
    }

    pub fn num_taxa(&self) -> usize {
        self.labels.len()
    }

    pub fn perturb(
        &mut self,
        rng: &mut impl rand::Rng,
        distribution: &impl rand::distributions::Distribution<f64>,
        ratio: f64,
    ) {
        self.distances.iter_mut().for_each(|i| {
            if *i > 0.0 && (ratio == 1.0 || rng.gen_bool(ratio)) {
                *i *= distribution.sample(rng).abs()
            }
        });
    }

    pub fn q_values_with_active_indices(
        &self,
        active: &Vec<usize>,
    ) -> Vec<((usize, usize), f64)> {
        let sum_d =
            |i: usize| -> f64 { active.iter().map(|&k| self.get(i, k)).sum() };
        let q = |(i, j): (usize, usize)| -> f64 {
            (active.len() - 2) as f64 * self.get(i, j) - sum_d(i) - sum_d(j)
        };
        active
            .iter()
            .cartesian_product(active.iter())
            .filter(|&(&i, &j)| i < j)
            .map(|(&i, &j)| ((i, j), q((i, j)).to_f64().unwrap()))
            .collect()
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

    pub fn join(
        name: &str,
        (l, d_l): (Self, f64),
        (r, d_r): (Self, f64),
    ) -> Self {
        Self::new(name, vec![(l, Some(d_l)), (r, Some(d_r))])
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

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

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
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let distribution = rand_distr::Normal::new(1.0, 0.0).unwrap();
        matrix_2.perturb(&mut rng, &distribution, 1.0);
        assert_eq!(matrix.distances, matrix_2.distances);
    }
}
