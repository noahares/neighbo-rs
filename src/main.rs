// based on https://docs.rs/phylogeny/latest/src/phylogeny/lib.rs.html#289-353
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use core::fmt;
use itertools::Itertools;
use log::warn;
use ordered_float::NotNan;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::num_traits::ToPrimitive;
use std::{
    io::Write,
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

    fn perturb(
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

    fn q_values_with_active_indices(
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
            .filter(|&(&i, &j)| i != j)
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

fn normalized_ratio(s: &str) -> Result<f64> {
    let ratio: f64 = s
        .parse()
        .with_context(|| format!("`{s}` isn't a valid ratio"))?;
    if (0.0..=1.0).contains(&ratio) {
        Ok(ratio)
    } else {
        Err(anyhow!(format!("Ratio is not in range {}-{}", 0.0, 0.1)))
    }
}

fn weighted_min_pos(
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

fn min_from_sample(
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

fn random_min_by_threshold(
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

fn deterministic_min(
    active_index_weight_pairs: Vec<((usize, usize), f64)>,
) -> Option<(usize, usize)> {
    active_index_weight_pairs
        .iter()
        .min_by_key(|&(_, v)| NotNan::new(*v).unwrap())
        .map(|min| min.0)
}

#[derive(Copy, Clone, clap::ValueEnum)]
enum RandomizationStrategy {
    WeightedSelection,
    ThresholdBasedRandomization,
    RandomSampling,
    Deterministic,
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
    #[arg(short, long, default_value_t = 0.01, value_parser = normalized_ratio)]
    pub noise: f64,
    /// ratio of matrix entries that get perturbed in each iteration
    #[arg(short = 'r', long, default_value_t = 1.0, value_parser = normalized_ratio)]
    pub noise_ratio: f64,
    /// Strategy for randomizing the neighbor joining steps
    #[arg(short = 'm', long, default_value_t = RandomizationStrategy::Deterministic, value_enum)]
    pub strategy: RandomizationStrategy,
    /// Percentile for threshold and sample strategies
    #[arg(short, long, default_value_t = 0.1)]
    pub percentile: f64,
    /// Seed for the noise
    #[arg(short, long, default_value_t = 0)]
    pub seed: u64,
    /// Output path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

impl Args {
    fn get_output(&self) -> Result<Box<dyn Write>> {
        match self.output {
            Some(ref path) => Ok(std::fs::File::create(path)
                .map(|f| Box::new(f) as Box<dyn Write>)?),
            None => Ok(Box::new(std::io::stdout())),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if (args.noise == 0.0 || args.noise_ratio == 0.0) && args.num_trees > 1 {
        warn!("Generating {} identical trees! Consider increasing `--noise` or `--noise_ratio`", args.num_trees)
    }
    let distance_matrix = DistanceMatrix::from_file(&args.sequence_file)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let distribution = rand_distr::Normal::new(1.0, args.noise)?;
    let trees: Vec<PhyloTree> = (0..args.num_trees)
        .map(|_| -> Result<PhyloTree> {
            let mut distance_matrix = distance_matrix.clone();
            distance_matrix.perturb(&mut rng, &distribution, args.noise_ratio);
            nj(distance_matrix, args.strategy, args.seed, args.percentile)
        })
        .filter_map(Result::ok)
        .collect_vec();
    let mut output = args.get_output()?;
    for tree in &trees {
        writeln!(output, "{}", tree)?;
    }
    Ok(())
}

fn nj(
    mut distance_matrix: DistanceMatrix,
    strategy: RandomizationStrategy,
    seed: u64,
    percentile: f64,
) -> Result<PhyloTree> {
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
        // let q = |&(&i, &j): &(&usize, &usize)| -> NotNan<f64> {
        //     NotNan::new(
        //         (active.len() - 2) as f64 * distance_matrix.get(i, j)
        //             - sum_d(i)
        //             - sum_d(j),
        //     )
        //     .unwrap()
        // };
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
                // let (&i, &j) = active
                //     .iter()
                //     .cartesian_product(active.iter())
                //     .filter(|&(&i, &j)| i != j)
                //     .min_by_key(q)
                //     .unwrap();
                // (i, j)
            }
        };
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
