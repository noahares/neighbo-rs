// based on https://docs.rs/phylogeny/latest/src/phylogeny/lib.rs.html#289-353
use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use ordered_float::NotNan;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use core::fmt;
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

mod nj;
mod parser;

#[derive(Clone)]
struct DistanceMatrix {
    labels: Vec<String>,
    distances: Vec<f64>,
}

impl DistanceMatrix {
    fn new(labels: Vec<String>, distances: Vec<f64>) -> Self {
        Self { labels, distances }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.distances[i * self.labels.len() + j]
    }

    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.distances[i * self.labels.len() + j] = v;
    }
    fn from_file(p: &Path) -> Result<Self> {
        std::fs::read_to_string(p)?.parse()
    }

    fn num_taxa(&self) -> usize {
        self.labels.len()
    }

    fn perturb(&mut self, rng: &mut impl rand::Rng, distribution: &impl rand::distributions::Distribution<f64>) {
        self.distances.iter_mut().for_each(|i| {
            if *i > 0.0 { *i *= distribution.sample(rng).abs() }
        });
    }
}

impl FromStr for DistanceMatrix {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let mut lines = s.lines();
        let n = lines.next().expect("Expected n").parse()?;

        let mut d =
            DistanceMatrix::new(Vec::with_capacity(n), Vec::with_capacity(n));

        for line in lines {
            let mut it = line.split_ascii_whitespace();
            d.labels.push(it.next().unwrap().to_string());
            d.distances
                .extend(it.map(|chars| chars.parse::<f64>().unwrap()));
        }

        Ok(d)
    }
}

struct PhyloTree {
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

    fn new_leaf(name: &str) -> Self {
        Self::new(name, vec![])
    }

    fn join(name: &str, (l, d_l): (Self, f64), (r, d_r): (Self, f64)) -> Self {
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

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to the sequence file
    #[arg(short = 'i', long)]
    pub sequence_file: PathBuf,
    /// Number of trees to generate
    #[arg(short = 't', long, default_value_t = 1)]
    pub num_trees: usize,
    /// stddev of noise as a multiplicative factor
    #[arg(short, long, default_value_t = 0.0)]
    pub noise: f64,
    /// Seed for the noise
    #[arg(short, long, default_value_t = 0)]
    pub seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let distance_matrix = DistanceMatrix::from_file(&args.sequence_file)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let distribution = rand_distr::Normal::new(1.0, args.noise)?;
    let trees: Vec<PhyloTree> = (0..args.num_trees).map(|_| -> Result<PhyloTree> {
        let mut distance_matrix = distance_matrix.clone();
        distance_matrix.perturb(&mut rng, &distribution);
        nj(distance_matrix)
    })
    .filter_map(Result::ok)
    .collect_vec();
    for tree in &trees {
        println!("{}", tree);
    }
    Ok(())
}

fn nj(mut distance_matrix: DistanceMatrix) -> Result<PhyloTree> {
    let n = distance_matrix.num_taxa();
    let mut active: Vec<usize> = (0..n).collect();
    let mut trees: Vec<Option<PhyloTree>> = distance_matrix
        .labels
        .iter()
        .map(|name| Some(PhyloTree::new_leaf(name)))
        .collect();
    while active.len() > 2 {
        let sum_d = |i: usize| -> f64 {
            active.iter().map(|&k| distance_matrix.get(i, k)).sum()
        };
        let q = |&(&i, &j): &(&usize, &usize)| -> NotNan<f64> {
            NotNan::new(
                (active.len() - 2) as f64 * distance_matrix.get(i, j)
                    - sum_d(i)
                    - sum_d(j),
            )
            .unwrap()
        };
        let (&i, &j) = active
            .iter()
            .cartesian_product(active.iter())
            .filter(|&(&i, &j)| i != j)
            .min_by_key(q)
            .unwrap();
        let (i, j) = (i.min(j), i.max(j));
        let d_i = distance_matrix.get(i, j) / 2.
            + (sum_d(i) - sum_d(j)) / (2. * (active.len() - 2) as f64);
        let d_j = distance_matrix.get(i, j) - d_i;

        active.remove(active.iter().position(|&x| x == j).unwrap());
        active.iter().filter(|&&k| k != i).for_each(|&k| {
            let d_k = (distance_matrix.get(i, k) + distance_matrix.get(j, k)
                - distance_matrix.get(i, j))
                / 2.;
            distance_matrix.set(i, k, d_k);
            distance_matrix.set(k, i, d_k);
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
