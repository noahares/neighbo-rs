use anyhow::{bail, Context, Result};
use itertools::{Either, Itertools};
use rand::seq::SliceRandom;
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
}

#[derive(PartialEq, Debug)]
pub enum Moltype {
    Dna {
        rates: [f64; 6],
        frequencies: [f64; 4],
    },
    Protein {
        rates: [f64; 190],
        frequencies: [f64; 20],
    },
}

impl SubstitutionModel for Moltype {
    fn get_rates(&self) -> &[f64] {
        match self {
            Moltype::Dna { rates, .. } => rates,
            Moltype::Protein { rates, .. } => rates,
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
                })
            } else if rates.len() == 190 && frequencies.len() == 20 {
                let mut rates_array = [0.0; 190];
                rates_array.copy_from_slice(&rates);

                let mut frequencies_array = [0.0; 20];
                frequencies_array.copy_from_slice(&frequencies);

                Ok(Moltype::Protein {
                    rates: rates_array,
                    frequencies: frequencies_array,
                })
            } else {
                bail!(
                    "Rates and frequencies could not be parsed from string {}",
                    s
                )
            }
        } else {
            bail!(
                "Rates and frequencies could not be parsed from string {}",
                s
            )
        }
    }
}

impl Moltype {
    #[inline]
    fn index_from_row_and_col_lt(i: usize, j: usize, n: usize) -> usize {
        (i * n) - (i * (i + 3) / 2) + (j - 1)
    }

    pub fn to_matrix(&self) -> DMatrix<f64> {
        let dim = self.get_rate_matrix_dimension();
        let mut matrix = DMatrix::zeros(dim, dim);

        let rates = self.get_rates();
        for i in 0..dim - 1 {
            for j in (i + 1)..dim {
                let rate = rates[Self::index_from_row_and_col_lt(i, j, dim)];
                matrix[(i, j)] = rate;
                matrix[(j, i)] = rate;
            }
        }

        for i in 0..dim {
            matrix[(i, i)] = -matrix.row(i).sum();
        }

        matrix
    }
}

pub struct MsaData {
    labels: Vec<String>,
    msa: Vec<Vec<usize>>,
    u: DMatrix<f64>,
    d: DMatrix<f64>,
    priors: Vec<f64>,
}

impl MsaData {
    pub fn new(sequence_path: &PathBuf, model_path: &PathBuf) -> Result<Self> {
        let phylip_file = File::open(sequence_path)?;
        let (labels, sequences): (Vec<String>, Vec<String>) =
            BufReader::new(phylip_file)
                .lines()
                .filter_ok(|l| !l.is_empty())
                .filter_map(Result::ok)
                .partition_map(|l| {
                    if l.starts_with('>') {
                        Either::Left(l.trim()[1..].to_string())
                    } else {
                        Either::Right(l.trim().to_string())
                    }
                });
        let model_file = File::open(model_path)?;
        let mut model_string = String::default();
        BufReader::new(model_file).read_line(&mut model_string)?;
        let moltype: Moltype = model_string.parse()?;
        let rate_matrix = moltype.to_matrix();
        let (u, d) = decomposed_rate_matrix(&rate_matrix);
        Ok(Self {
            labels,
            msa: normalize_msa(&sequences, &moltype),
            u,
            d,
            priors: moltype.get_frequencies().to_vec(),
        })
    }

    pub fn sample_distance(
        &self,
        sequences: (usize, usize),
        distance_prior_distribution: &impl rand::distributions::Distribution<f64>,
        rng: &mut impl rand::Rng,
        x_0: f64,
        n_samples: usize,
        burnin: usize,
    ) -> Vec<f64> {
        let mut samples = Vec::with_capacity(n_samples);
        samples.push(x_0);
        let mut last_likelihood = 1e-7;

        while samples.len() < n_samples + burnin {
            // divide proposed branch length by 2 because we introduce a virtual root in the middle
            let proposed_sample =
                distance_prior_distribution.sample(rng) / 2_f64;

            let new_likelihood = branch_likelihood(
                &self.msa[sequences.0],
                &self.msa[sequences.1],
                &self.priors,
                &p_t(&self.u, &self.d, proposed_sample),
            );

            if rng.gen::<f64>() < new_likelihood / last_likelihood {
                last_likelihood = new_likelihood;
                samples.push(proposed_sample);
            }
        }

        samples[burnin..].to_vec()
    }
}

pub struct DistanceMatrixSamples {
    labels: Vec<String>,
    samples: Vec<Vec<f64>>,
}

impl DistanceMatrixSamples {
    pub fn sample(&self, rng: &mut impl rand::Rng) -> Result<DistanceMatrix> {
        let distances: Vec<f64> = self
            .samples
            .iter()
            .map(|s| s.choose(rng).context("No samples available").cloned())
            .collect::<Result<Vec<f64>>>()?;
        Ok(DistanceMatrix::new(self.labels.clone(), distances))
    }
}

pub fn generate_distance_matrix_samples(
    msa_data: &MsaData,
    distance_prior_distribution: &impl rand::distributions::Distribution<f64>,
    rng: &mut impl rand::Rng,
    x_0: f64,
    n_samples: usize,
    burnin: usize,
) -> DistanceMatrixSamples {
    let dim = msa_data.labels.len();
    let samples = (0..dim - 1)
        .cartesian_product(1..dim)
        .filter(|&(i, j)| i < j)
        .map(|(i, j)| {
            msa_data.sample_distance(
                (i, j),
                distance_prior_distribution,
                rng,
                x_0,
                n_samples,
                burnin,
            )
        })
        .collect();
    DistanceMatrixSamples {
        labels: msa_data.labels.clone(),
        samples,
    }
}

fn normalize_msa(msa: &[String], moltype: &Moltype) -> Vec<Vec<usize>> {
    let mapping: HashMap<char, usize> = match moltype {
        Moltype::Dna { .. } => ['A', 'C', 'G', 'T', '-']
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect(),
        Moltype::Protein { .. } => [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
            'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-',
        ]
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect(),
    };
    msa.iter()
        .map(|sequence| sequence.chars().map(|c| mapping[&c]).collect())
        .collect()
}

fn branch_likelihood(
    sequence_a: &[usize],
    sequence_b: &[usize],
    priors: &[f64],
    p_t: &DMatrix<f64>,
) -> f64 {
    let dim = priors.len();
    sequence_a
        .iter()
        .zip(sequence_b.iter())
        .map(|(&a, &b)| {
            // if a or b is a gap, the site should not contribute to the likelihood
            if a == dim || b == dim {
                1.0
            } else {
                priors
                    .iter()
                    .enumerate()
                    .map(|(i, prior)| *prior * p_t[(i, a)] * p_t[(i, b)])
                    .sum::<f64>()
            }
        })
        .product()
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
    u * d.map_diagonal(|val| val.powf(branch_length)) * u.transpose()
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use nalgebra::DMatrix;

    use crate::distance_distribution::normalize_msa;

    use super::{decomposed_rate_matrix, Moltype};

    #[test]
    fn test_model_parser() {
        let dna_input = "GTR{5.002025/5.265654/3.291279/1.648017/8.361876/1.000000}+FU{0.294729/0.253416/0.175738/0.276117}, noname = 1-705";
        let dna_model: Moltype = dna_input.parse().unwrap();
        assert_eq!(
            dna_model,
            Moltype::Dna {
                rates: [5.002025, 5.265654, 3.291279, 1.648017, 8.361876, 1.0],
                frequencies: [0.294729, 0.253416, 0.175738, 0.276117]
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
                    -13.558957999999999,
                    5.002025,
                    5.265654,
                    3.291279,
                    5.002025,
                    -15.011918000000001,
                    1.648017,
                    8.361876,
                    5.265654,
                    1.648017,
                    -7.913671,
                    1.0,
                    3.291279,
                    8.361876,
                    1.0,
                    -12.653155,
                ]
                .iter(),
            )
            .for_each(|(a, b)| assert_float_absolute_eq!(a, b));
    }

    #[test]
    fn test_decompose_matrix() {
        let rate_matrix = DMatrix::from_column_slice(
            4,
            4,
            &[
                0.294729, 5.002025, 5.265654, 3.291279, 5.002025, 0.253416,
                1.648017, 8.361876, 5.265654, 1.648017, 0.175738, 1.000000,
                3.291279, 8.361876, 1.000000, 0.276117,
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
                    frequencies: [0.0, 0.0, 0.0, 0.0]
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
}
