use anyhow::{bail, Context, Result};
use bitvec::prelude::*;
use itertools::{izip, Itertools};
use log::{debug, info};
use logging_timer::time;
use plotpy::{Curve, Plot};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;
use rayon::prelude::*;
use regex::Regex;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    str::FromStr,
};

use nalgebra::DMatrix;

use crate::datastructures::DistanceMatrix;

pub trait SubstitutionModel {
    fn get_rates(&self) -> &[f64];
    fn get_frequencies(&self) -> &[f64];
    fn get_rate_matrix_dimension(&self) -> usize;
    fn get_characters(&self) -> &[char];
}

#[derive(PartialEq, Debug)]
pub enum Moltype {
    Dna {
        rates: [f64; 6],
        frequencies: [f64; 4],
        characters: [char; 4],
    },
    Protein {
        rates: Box<[f64; 190]>,
        frequencies: [f64; 20],
        characters: [char; 20],
    },
}

impl SubstitutionModel for Moltype {
    fn get_rates(&self) -> &[f64] {
        match self {
            Moltype::Dna { rates, .. } => rates,
            Moltype::Protein { rates, .. } => rates.as_ref(),
        }
    }

    fn get_frequencies(&self) -> &[f64] {
        match self {
            Moltype::Dna { frequencies, .. } => frequencies,
            Moltype::Protein { frequencies, .. } => frequencies,
        }
    }

    fn get_rate_matrix_dimension(&self) -> usize {
        match self {
            Moltype::Dna { .. } => 4,
            Moltype::Protein { .. } => 20,
        }
    }

    fn get_characters(&self) -> &[char] {
        match self {
            Moltype::Dna { characters, .. } => characters,
            Moltype::Protein { characters, .. } => characters,
        }
    }
}

impl FromStr for Moltype {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let re = Regex::new(r"\{([^}]*)\}.*\{([^}]*)\}")?;
        if let Some(captures) = re.captures(s) {
            let rates_str =
                captures.get(1).context("No rates found")?.as_str();
            let frequencies_str =
                captures.get(2).context("No frequencies found")?.as_str();

            let rates: Vec<f64> = rates_str
                .split('/')
                .map(|val| {
                    val.parse()
                        .with_context(|| format!("cannot parse {}", val))
                })
                .collect::<Result<Vec<f64>>>()?;

            let frequencies: Vec<f64> = frequencies_str
                .split('/')
                .map(|val| {
                    val.parse()
                        .with_context(|| format!("cannot parse {}", val))
                })
                .collect::<Result<Vec<f64>>>()?;

            if rates.len() == 6 && frequencies.len() == 4 {
                let mut rates_array = [0.0; 6];
                rates_array.copy_from_slice(&rates);

                let mut frequencies_array = [0.0; 4];
                frequencies_array.copy_from_slice(&frequencies);

                Ok(Moltype::Dna {
                    rates: rates_array,
                    frequencies: frequencies_array,
                    characters: ['A', 'C', 'G', 'T'],
                })
            } else if rates.len() == 190 && frequencies.len() == 20 {
                let mut rates_array = [0.0; 190];
                rates_array.copy_from_slice(&rates);

                let mut frequencies_array = [0.0; 20];
                frequencies_array.copy_from_slice(&frequencies);

                let characters = [
                    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
                    'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                ];

                Ok(Moltype::Protein {
                    rates: Box::new(rates_array),
                    frequencies: frequencies_array,
                    characters,
                })
            } else {
                bail!(
                    "Rates and frequencies could not be parsed from string {}",
                    s
                )
            }
        } else {
            info!("No model parameters found, assuming LG");
            let characters = [
                'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
            ];
            Ok(Moltype::Protein {
                rates: Box::new([
                    0.425093, 0.276818, 0.395144, 2.489084, 0.969894,
                    1.038545, 2.066040, 0.358858, 0.149830, 0.395337,
                    0.536518, 1.124035, 0.253701, 1.177651, 4.727182,
                    2.139501, 0.180717, 0.218959, 2.547870, 0.751878,
                    0.123954, 0.534551, 2.807908, 0.363970, 0.390192,
                    2.426601, 0.126991, 0.301848, 6.326067, 0.484133,
                    0.052722, 0.332533, 0.858151, 0.578987, 0.593607,
                    0.314440, 0.170887, 5.076149, 0.528768, 1.695752,
                    0.541712, 1.437645, 4.509238, 0.191503, 0.068427,
                    2.145078, 0.371004, 0.089525, 0.161787, 4.008358,
                    2.000679, 0.045376, 0.612025, 0.083688, 0.062556,
                    0.523386, 5.243870, 0.844926, 0.927114, 0.010690,
                    0.015076, 0.282959, 0.025548, 0.017416, 0.394456,
                    1.240275, 0.425860, 0.029890, 0.135107, 0.037967,
                    0.084808, 0.003499, 0.569265, 0.640543, 0.320627,
                    0.594007, 0.013266, 0.893680, 1.105251, 0.075382,
                    2.784478, 1.143480, 0.670128, 1.165532, 1.959291,
                    4.128591, 0.267959, 4.813505, 0.072854, 0.582457,
                    3.234294, 1.672569, 0.035855, 0.624294, 1.223828,
                    1.080136, 0.236199, 0.257336, 0.210332, 0.348847,
                    0.423881, 0.044265, 0.069673, 1.807177, 0.173735,
                    0.018811, 0.419409, 0.611973, 0.604545, 0.077852,
                    0.120037, 0.245034, 0.311484, 0.008705, 0.044261,
                    0.296636, 0.139538, 0.089586, 0.196961, 1.739990,
                    0.129836, 0.268491, 0.054679, 0.076701, 0.108882,
                    0.366317, 0.697264, 0.442472, 0.682139, 0.508851,
                    0.990012, 0.584262, 0.597054, 5.306834, 0.119013,
                    4.145067, 0.159069, 4.273607, 1.112727, 0.078281,
                    0.064105, 1.033739, 0.111660, 0.232523, 10.649107,
                    0.137500, 6.312358, 2.592692, 0.249060, 0.182287,
                    0.302936, 0.619632, 0.299648, 1.702745, 0.656604,
                    0.023918, 0.390322, 0.748683, 1.136863, 0.049906,
                    0.131932, 0.185202, 1.798853, 0.099849, 0.346960,
                    2.020366, 0.696175, 0.481306, 1.898718, 0.094464,
                    0.361819, 0.165001, 2.457121, 7.803902, 0.654683,
                    1.338132, 0.571468, 0.095131, 0.089613, 0.296501,
                    6.472279, 0.248862, 0.400547, 0.098369, 0.140825,
                    0.245841, 2.188158, 3.151815, 0.189510, 0.249313,
                ]),
                frequencies: [
                    0.079066, 0.055941, 0.041977, 0.053052, 0.012937,
                    0.040767, 0.071586, 0.057337, 0.022355, 0.062157,
                    0.099081, 0.064600, 0.022951, 0.042302, 0.044040,
                    0.061197, 0.053287, 0.012066, 0.034155, 0.069147,
                ],
                characters,
            })
        }
    }
}

impl Moltype {
    pub fn to_matrix(&self) -> DMatrix<f64> {
        let dim = self.get_rate_matrix_dimension();
        let mut matrix = DMatrix::zeros(dim, dim);

        let rates = self.get_rates();
        for i in 0..dim - 1 {
            for j in (i + 1)..dim {
                let rate = rates
                    [DistanceMatrix::index_from_row_and_col_lt(i, j, dim)];
                matrix[(i, j)] = rate;
                matrix[(j, i)] = rate;
            }
        }
        let mut factor = 0.0;
        for i in 0..dim {
            matrix[(i, i)] = -matrix.row(i).sum();
            factor += self.get_frequencies()[i] * matrix[(i, i)];
        }
        info!("Factor = {}", factor);

        matrix / -factor
    }
}

#[derive(Debug)]
pub struct MsaData {
    labels: Vec<String>,
    msa: Vec<Vec<u8>>,
    u: DMatrix<f64>,
    d: DMatrix<f64>,
    moltype: Moltype,
    pub average_pairwise_distance: f64,
}

impl MsaData {
    pub fn new(
        sequence_path: &PathBuf,
        model_path: &Option<PathBuf>,
    ) -> Result<Self> {
        // let phylip_file = File::open(sequence_path)?;
        let (labels, sequences): (Vec<String>, Vec<String>) =
            Self::parse_phylip_file(sequence_path)?;
        let mut model_string = String::default();
        match model_path {
            Some(p) => {
                let model_file = File::open(p)?;
                BufReader::new(model_file).read_line(&mut model_string)?;
            }
            None => (),
        }
        let moltype: Moltype = model_string.parse()?;
        let rate_matrix = moltype.to_matrix();
        let (u, d) = decomposed_rate_matrix(&rate_matrix);
        let msa = normalize_msa(&sequences, &moltype);
        let average_pairwise_distance =
            Self::average_pairwise_distance(&msa, &moltype);
        info!("Average distance: {}", average_pairwise_distance);
        Ok(Self {
            labels,
            msa,
            u,
            d,
            moltype,
            average_pairwise_distance,
        })
    }

    pub fn parse_phylip_file(
        sequence_path: &PathBuf,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let phylip_file = File::open(sequence_path)?;
        let (labels, sequences): (Vec<String>, Vec<String>) = {
            let lines: Vec<String> = BufReader::new(phylip_file)
                .lines()
                .map_ok(|l| l.trim().to_string())
                .filter_ok(|l| !l.is_empty())
                .filter_map(Result::ok)
                .collect();
            if !lines[0].starts_with(['>', ';']) {
                bail!("Not a valid FASTA file")
            }
            let mut labels: Vec<String> = Vec::new();
            let mut sequences: Vec<String> = Vec::new();
            let mut current_sequence = String::new();
            for line in lines {
                if line.starts_with(';') {
                    continue;
                }
                if let Some(label) = line.strip_prefix('>') {
                    labels.push(label.to_string());
                    if labels.len() > 1 {
                        sequences.push(current_sequence);
                        current_sequence = String::new();
                    }
                } else {
                    current_sequence.extend(
                        line.chars()
                            .filter(|c| c.is_alphabetic() || *c == '-'),
                    );
                }
            }
            sequences.push(current_sequence);
            (labels, sequences)
        };
        assert_eq!(labels.len(), sequences.len());
        assert!(sequences.iter().map(|s| s.len()).all_equal());
        Ok((labels, sequences))
    }

    pub fn label_sequence_map(&self) -> HashMap<&String, &Vec<u8>> {
        self.labels
            .iter()
            .zip_eq(self.msa.iter())
            .map(|(l, s)| (l, s))
            .collect::<HashMap<_, _>>()
    }

    pub fn get_char_map(&self) -> HashMap<u8, BitVec> {
        let num_chars = self.moltype.get_rate_matrix_dimension() + 1;
        (0..num_chars as u8)
            .zip((0..num_chars).map(|i| {
                let mut bv = bitvec![0; num_chars - 1];
                if i == num_chars - 1 {
                    bv.fill(true);
                } else {
                    bv.set(i, true);
                }
                bv
            }))
            .collect::<HashMap<_, _>>()
    }

    pub fn sequence_length(&self) -> usize {
        self.msa[0].len()
    }

    fn average_pairwise_distance(msa: &[Vec<u8>], moltype: &Moltype) -> f64 {
        let num_alignments = msa.len();
        let gap = moltype.get_rate_matrix_dimension() as u8;
        let indices: Vec<(usize, usize)> = (0..num_alignments - 1)
            .cartesian_product(1..num_alignments)
            .filter(|&(i, j)| i < j)
            .collect();
        indices
            .iter()
            .map(|(i, j)| {
                let (distance, gaps) =
                    msa[*i].iter().zip_eq(msa[*j].iter()).fold(
                        (0, 0),
                        |(dist, n_gaps): (usize, usize), (&a, &b)| {
                            (
                                (dist
                                    + (a != gap && b != gap && a != b)
                                        as usize),
                                (n_gaps + (a == gap || b == gap) as usize),
                            )
                        },
                    );
                let valid_chars = msa[*i].len() - gaps;
                if valid_chars == 0 {
                    0.0
                } else {
                    distance as f64 / valid_chars as f64
                }
            })
            .sum::<f64>()
            / indices.len() as f64
    }

    #[time("debug")]
    pub fn sample_distance(
        &self,
        sequences: (usize, usize),
        distance_prior_distribution: &impl rand::distributions::Distribution<f64>,
        rng: &mut impl rand::Rng,
        x_0: f64,
        n_samples: usize,
        burnin: usize,
    ) -> Vec<DistributionSample> {
        let mut samples = Vec::with_capacity(n_samples);
        let mut last_likelihood = std::f64::NEG_INFINITY;
        samples.push(DistributionSample {
            branch_length: x_0,
            likelihood: last_likelihood,
        });
        let mut total_num_samples = 0;

        while samples.len() < n_samples + burnin {
            total_num_samples += 1;
            // divide proposed branch length by 2 because we introduce a virtual root in the middle
            let proposed_sample = distance_prior_distribution.sample(rng);

            let new_likelihood = branch_likelihood(
                &self.msa[sequences.0],
                &self.msa[sequences.1],
                self.moltype.get_frequencies(),
                &p_t(&self.u, &self.d, proposed_sample),
            );

            if rng.gen::<f64>().ln() < new_likelihood - last_likelihood {
                last_likelihood = new_likelihood;
                samples.push(DistributionSample {
                    branch_length: proposed_sample,
                    likelihood: new_likelihood,
                });
            }
        }
        debug!(
            "Total samples: {}, burnin: {}, taken: {}",
            total_num_samples, burnin, n_samples
        );
        samples = samples[burnin..].to_vec();
        samples
            .sort_by(|a, b| b.likelihood.partial_cmp(&a.likelihood).unwrap());
        samples
    }
}

pub trait Sample {
    fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: rand::Rng + ?Sized;
    fn ml_element(&self) -> f64;
}

#[derive(Clone)]
pub enum DistanceDistributionSamples {
    Approximation(f64, rand_distr::Normal<f64>),
    RealSamples(Vec<f64>),
    Fixed(f64),
}

impl Sample for DistanceDistributionSamples {
    fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: rand::Rng + ?Sized,
    {
        match self {
            DistanceDistributionSamples::Approximation(
                _ml_element,
                normal,
            ) => normal.sample(rng),
            DistanceDistributionSamples::RealSamples(samples) => {
                *samples.choose(rng).unwrap()
            }
            DistanceDistributionSamples::Fixed(sample) => *sample,
        }
    }

    fn ml_element(&self) -> f64 {
        match self {
            DistanceDistributionSamples::Approximation(
                ml_element,
                _normal,
            ) => *ml_element,
            DistanceDistributionSamples::RealSamples(samples) => {
                *samples.first().unwrap()
            }
            DistanceDistributionSamples::Fixed(sample) => *sample,
        }
    }
}

#[derive(Clone)]
pub struct DistributionSample {
    branch_length: f64,
    likelihood: f64,
}

#[derive(Clone)]
pub struct DistanceMatrixSamples {
    labels: Vec<String>,
    samples: Vec<DistanceDistributionSamples>,
    sample_size: usize,
}

impl DistanceMatrixSamples {
    pub fn new(
        labels: &[String],
        samples: Vec<Vec<DistributionSample>>,
        approximate_with_normal_distribution: bool,
        stddev_scale: f64,
    ) -> Result<Self> {
        let sample_size = samples[0].len();
        debug_assert!(samples.iter().all(|s| s
            .windows(2)
            .all(|w| w[1].likelihood <= w[0].likelihood)));
        if !approximate_with_normal_distribution {
            Ok(Self {
                labels: labels.to_vec(),
                samples: samples
                    .into_iter()
                    .map(|s| {
                        DistanceDistributionSamples::RealSamples(
                            s.into_iter()
                                .map(|v| v.branch_length)
                                .collect_vec(),
                        )
                    })
                    .collect_vec(),
                sample_size,
            })
        } else {
            let means: Vec<f64> = samples
                .iter()
                .map(|s| {
                    if !s.is_empty() {
                        Ok(s.iter()
                            .map(|sample| sample.branch_length)
                            .sum::<f64>()
                            / s.len() as f64)
                    } else {
                        bail!("No samples available")
                    }
                })
                .collect::<Result<Vec<f64>>>()?;
            let stddevs: Vec<f64> = samples
                .iter()
                .zip_eq(means.iter())
                .map(|(s, m)| {
                    (s.iter()
                        .map(|v| (m - v.branch_length).powi(2))
                        .sum::<f64>()
                        / s.len() as f64)
                        .sqrt()
                })
                .collect();
            let distributions: Vec<DistanceDistributionSamples> =
                izip!(samples.iter(), means.iter(), stddevs.iter())
                    .map(|(s, mean, stddev)| {
                        Ok(DistanceDistributionSamples::Approximation(
                            s.first().unwrap().branch_length,
                            rand_distr::Normal::new(
                                *mean,
                                *stddev * stddev_scale,
                            )?,
                        ))
                    })
                    .collect::<Result<Vec<DistanceDistributionSamples>>>()?;
            Ok(Self {
                labels: labels.to_vec(),
                samples: distributions,
                sample_size,
            })
        }
    }

    pub fn labels(&self) -> std::slice::Iter<String> {
        self.labels.iter()
    }

    pub fn sample<'a, I>(
        &self,
        rng: &mut impl rand::Rng,
        ratio: f64,
        old_distances: I,
    ) -> DistanceMatrix
    where
        I: Iterator<Item = &'a f64>,
    {
        let distances: Vec<f64> = self
            .samples
            .iter()
            .zip(old_distances)
            .map(|(s, d)| if ratio > rng.gen() { s.sample(rng) } else { *d })
            .collect::<Vec<f64>>();
        debug!("{:?}", distances);
        DistanceMatrix::new(self.labels.clone(), distances)
    }

    pub fn sample_ml_fallback(
        &self,
        rng: &mut impl rand::Rng,
        ratio: f64,
    ) -> DistanceMatrix {
        let distances: Vec<f64> = self
            .samples
            .iter()
            .zip(self.samples.iter().map(|s| s.ml_element()))
            .map(|(s, d)| if ratio > rng.gen() { s.sample(rng) } else { d })
            .collect::<Vec<f64>>();
        debug!("{:?}", distances);
        DistanceMatrix::new(self.labels.clone(), distances)
    }

    pub fn sample_entry(
        &self,
        row: usize,
        col: usize,
        rng: &mut impl rand::Rng,
    ) -> f64 {
        let (i, j) = (row.min(col), row.max(col));
        let index =
            DistanceMatrix::index_from_row_and_col_lt(i, j, self.num_taxa());
        self.samples[index].sample(rng)
    }

    pub fn get(&self, i: usize, j: usize) -> &DistanceDistributionSamples {
        debug_assert_ne!(i, j);
        &self.samples
            [DistanceMatrix::index_from_row_and_col(i, j, self.num_taxa())]
    }

    pub fn set(
        &mut self,
        row: usize,
        col: usize,
        value: DistanceDistributionSamples,
    ) {
        let (i, j) = (row.min(col), row.max(col));
        let index =
            DistanceMatrix::index_from_row_and_col_lt(i, j, self.num_taxa());
        self.samples[index] = value
    }

    pub fn update(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        rng: &mut impl rand::Rng,
    ) {
        let d_k = (0..self.sample_size)
            .map(|_| {
                (self.sample_entry(i, k, rng) + self.sample_entry(j, k, rng)
                    - self.sample_entry(i, j, rng))
                    / 2.
            })
            .collect_vec();
        self.set(i, k, DistanceDistributionSamples::RealSamples(d_k));
    }

    pub fn ml_distance_matrix(&self) -> DistanceMatrix {
        let distances: Vec<f64> = self
            .samples
            .iter()
            .map(|s| s.ml_element())
            .collect::<Vec<f64>>();
        debug!("{:?}", distances);
        DistanceMatrix::new(self.labels.clone(), distances)
    }

    pub fn num_taxa(&self) -> usize {
        self.labels.len()
    }
}

fn plot_distance_distribution(
    samples: &Vec<Vec<DistributionSample>>,
    path: &PathBuf,
) -> Result<()> {
    let mut plot = Plot::new();
    for samples in samples {
        let mut curve = Curve::new();
        curve.set_line_width(1.0);
        curve.points_begin();
        for s in samples.iter() {
            curve.points_add(s.branch_length, s.likelihood);
        }
        curve.points_end();
        plot.add(&curve)
            .grid_and_labels("branch length", "log likelihood");
    }
    plot.save(path).unwrap();
    Ok(())
}

#[time("info")]
pub fn generate_distance_matrix_samples<D>(
    msa_data: &MsaData,
    distance_prior_distribution: &D,
    seed: u64,
    n_samples: usize,
    burnin: usize,
    stddev_scale: f64,
    plot_path: &Option<PathBuf>,
) -> Result<DistanceMatrixSamples>
where
    D: rand::distributions::Distribution<f64> + std::marker::Sync,
{
    let dim = msa_data.labels.len();
    let indices: Vec<(usize, usize)> = (0..dim - 1)
        .cartesian_product(1..dim)
        .filter(|&(i, j)| i < j)
        .collect();
    // TODO: use batches for less rng objects <noahares>
    let samples: Vec<Vec<DistributionSample>> = indices
        .into_par_iter()
        .map(|(i, j)| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            rng.set_stream((i + j) as u64);
            let x_0 = distance_prior_distribution.sample(&mut rng);
            msa_data.sample_distance(
                (i, j),
                distance_prior_distribution,
                &mut rng,
                x_0,
                n_samples,
                burnin,
            )
        })
        .collect();
    debug_assert_eq!(samples.len(), (dim * dim - dim) / 2);
    if let Some(distance_distribution_output_path) = plot_path {
        plot_distance_distribution(
            &samples,
            distance_distribution_output_path,
        )?;
    }
    DistanceMatrixSamples::new(&msa_data.labels, samples, true, stddev_scale)
}

fn normalize_msa(msa: &[String], moltype: &Moltype) -> Vec<Vec<u8>> {
    let mapping: HashMap<char, u8> = match moltype {
        Moltype::Dna { .. } => ['A', 'C', 'G', 'T', '-']
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c, i as u8))
            .collect(),
        Moltype::Protein { .. } => [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
            'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-',
        ]
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i as u8))
        .collect(),
    };
    msa.iter()
        .map(|sequence| sequence.chars().map(|c| mapping[&c]).collect())
        .collect()
}

fn branch_likelihood(
    sequence_a: &[u8],
    sequence_b: &[u8],
    priors: &[f64],
    p_t: &DMatrix<f64>,
) -> f64 {
    let dim = priors.len() as u8;
    sequence_a
        .iter()
        .zip_eq(sequence_b.iter())
        .map(|(&a, &b)| {
            // if a or b is a gap, the site should not contribute to the likelihood
            if a == dim || b == dim {
                0.0
            } else {
                // simplified likelihood because only 2 taxa in the "tree" and LG is time
                // reversable
                ((priors[a as usize] + priors[b as usize])
                    * p_t[(a as usize, b as usize)])
                    .ln()
            }
        })
        // log likelihood -> sum
        .sum()
}

fn decomposed_rate_matrix(
    rate_matrix: &DMatrix<f64>,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let decomposed_rate_matrix = rate_matrix.clone().symmetric_eigen();
    let dim = decomposed_rate_matrix.eigenvalues.len();
    let d: DMatrix<f64> = {
        let mut d = DMatrix::zeros(dim, dim);
        d.set_diagonal(&decomposed_rate_matrix.eigenvalues);
        d
    };
    let u: DMatrix<f64> = DMatrix::from_column_slice(
        dim,
        dim,
        decomposed_rate_matrix.eigenvectors.as_slice(),
    );
    (u, d)
}

fn p_t(
    u: &DMatrix<f64>,
    d: &DMatrix<f64>,
    branch_length: f64,
) -> DMatrix<f64> {
    u * DMatrix::from_diagonal(
        &d.map_diagonal(|val| (val * branch_length).exp()),
    ) * u.transpose()
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use nalgebra::DMatrix;

    use crate::distance_distribution::{branch_likelihood, normalize_msa};

    use super::{decomposed_rate_matrix, p_t, Moltype};

    #[test]
    fn test_model_parser() {
        let dna_input = "GTR{5.002025/5.265654/3.291279/1.648017/8.361876/1.000000}+FU{0.294729/0.253416/0.175738/0.276117}, noname = 1-705";
        let dna_model: Moltype = dna_input.parse().unwrap();
        assert_eq!(
            dna_model,
            Moltype::Dna {
                rates: [5.002025, 5.265654, 3.291279, 1.648017, 8.361876, 1.0],
                frequencies: [0.294729, 0.253416, 0.175738, 0.276117],
                characters: ['A', 'C', 'G', 'T'],
            }
        )
    }

    #[test]
    fn test_to_matrix() {
        let dna_input = "GTR{5.002025/5.265654/3.291279/1.648017/8.361876/1.000000}+FU{0.294729/0.253416/0.175738/0.276117}, noname = 1-705";
        let dna_model: Moltype = dna_input.parse().unwrap();
        let matrix = dna_model.to_matrix();
        matrix
            .as_slice()
            .iter()
            .zip_eq(
                [
                    -1.068900145278174,
                    0.3943271488255261,
                    0.41510994617614405,
                    0.2594630502765038,
                    0.3943271488255261,
                    -1.183442070629914,
                    0.1299189517897246,
                    0.6591959700146631,
                    0.41510994617614405,
                    0.1299189517897246,
                    -0.6238624001625842,
                    0.07883350219671556,
                    0.2594630502765038,
                    0.6591959700146631,
                    0.07883350219671556,
                    -0.9974925224878824,
                ]
                .iter(),
            )
            .for_each(|(a, b)| assert_float_absolute_eq!(a, b));
        assert_float_absolute_eq!(matrix.sum(), 0.0);
    }

    #[test]
    fn test_decompose_matrix() {
        let rate_matrix = DMatrix::from_column_slice(
            4,
            4,
            &[
                -1.068900145278174,
                0.3943271488255261,
                0.41510994617614405,
                0.2594630502765038,
                0.3943271488255261,
                -1.183442070629914,
                0.1299189517897246,
                0.6591959700146631,
                0.41510994617614405,
                0.1299189517897246,
                -0.6238624001625842,
                0.07883350219671556,
                0.2594630502765038,
                0.6591959700146631,
                0.07883350219671556,
                -0.9974925224878824,
            ],
        );
        let (u, d) = decomposed_rate_matrix(&rate_matrix);
        // test u is unitary
        assert!((u.clone() * u.clone().transpose()).is_identity(1e-7));
        // test d is diagonal
        assert!((0..4)
            .cartesian_product(0..4)
            .filter(|(a, b)| a != b)
            .all(|(a, b)| d[(a, b)] == 0.0));
        let reconstructed_matrix = u.clone() * d * u.transpose();
        rate_matrix
            .as_slice()
            .iter()
            .zip_eq(reconstructed_matrix.as_slice().iter())
            .for_each(|(a, b)| assert_float_absolute_eq!(a, b));
    }

    #[test]
    fn test_normalize_msa() {
        let msa = vec![
            String::from("ATACGA"),
            "AAATGA".into(),
            "TCACGA".into(),
            "T-AC-A".into(),
        ];
        assert_eq!(
            normalize_msa(
                &msa,
                &Moltype::Dna {
                    rates: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    frequencies: [0.0, 0.0, 0.0, 0.0],
                    characters: ['A', 'C', 'G', 'T'],
                }
            ),
            vec![
                vec![0, 3, 0, 1, 2, 0],
                vec![0, 0, 0, 3, 2, 0],
                vec![3, 1, 0, 1, 2, 0],
                vec![3, 4, 0, 1, 4, 0],
            ]
        )
    }

    #[test]
    fn test_branch_likelihood() {
        let rate_matrix = DMatrix::from_column_slice(
            4,
            4,
            &[
                -1.068900145278174,
                0.3943271488255261,
                0.41510994617614405,
                0.2594630502765038,
                0.3943271488255261,
                -1.183442070629914,
                0.1299189517897246,
                0.6591959700146631,
                0.41510994617614405,
                0.1299189517897246,
                -0.6238624001625842,
                0.07883350219671556,
                0.2594630502765038,
                0.6591959700146631,
                0.07883350219671556,
                -0.9974925224878824,
            ],
        );
        let (u, d) = decomposed_rate_matrix(&rate_matrix);
        let p_t = p_t(&u, &d, 0.0);
        assert!(p_t.is_identity(1e-7));
        let priors = [0.294729, 0.253416, 0.175738, 0.276117];
        let sequence_a = vec![1, 3, 2, 0];
        let sequence_b = vec![1, 3, 2, 0];
        assert_float_absolute_eq!(
            branch_likelihood(&sequence_a, &sequence_b, &priors, &p_t),
            priors.iter().map(|p| (2.0 * p).ln()).sum::<f64>()
        );
    }
}
