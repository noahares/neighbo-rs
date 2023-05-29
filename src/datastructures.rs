use core::{str::FromStr, fmt};
use std::path::Path;
use anyhow::Result;
use itertools::Itertools;
use rand_distr::num_traits::ToPrimitive;

#[derive(Clone)]
pub struct DistanceMatrix {
    labels: Vec<String>,
    distances: Vec<f64>,
}

impl DistanceMatrix {
    fn new(labels: Vec<String>, distances: Vec<f64>) -> Self {
        Self { labels, distances }
    }

    fn index_from_row_and_col(i: usize, j: usize) -> usize {
        let (i, j) = (i.min(j), i.max(j));
        (i * (i + 1) / 2) + j - i
    }

    pub fn labels(&self) -> std::slice::Iter<String> {
        self.labels.iter()
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.distances[Self::index_from_row_and_col(i, j)]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.distances[Self::index_from_row_and_col(i, j)] = v;
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
        let mut lines = s.lines();
        let n = lines.next().expect("Expected n").parse()?;

        let mut d =
            DistanceMatrix::new(Vec::with_capacity(n), Vec::with_capacity(n));

        for (i, line) in lines.enumerate() {
            let mut it = line.split_ascii_whitespace();
            d.labels.push(it.next().unwrap().to_string());
            d.distances.extend(
                it.enumerate()
                    .filter(|(j, _)| i <= *j)
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

    pub fn join(name: &str, (l, d_l): (Self, f64), (r, d_r): (Self, f64)) -> Self {
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
