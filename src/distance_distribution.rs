use anyhow::{bail, Context, Result};
use regex::Regex;
use std::str::FromStr;

use nalgebra::DMatrix;

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
        let (dim, rates, frequencies) = match self {
            Moltype::Dna { rates, frequencies } => {
                (4, rates.as_slice(), frequencies.as_slice())
            }
            Moltype::Protein { rates, frequencies } => {
                (20, rates.as_slice(), frequencies.as_slice())
            }
        };
        let mut matrix = DMatrix::zeros(dim, dim);

        for i in 0..dim {
            matrix[(i, i)] = frequencies[i];
        }

        for i in 0..dim - 1 {
            for j in (i + 1)..dim {
                let rate = rates[Self::index_from_row_and_col_lt(i, j, dim)];
                matrix[(i, j)] = rate;
                matrix[(j, i)] = rate;
            }
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::Moltype;

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
        assert_eq!(
            matrix.data.as_slice(),
            [
                0.294729, 5.002025, 5.265654, 3.291279, 5.002025, 0.253416,
                1.648017, 8.361876, 5.265654, 1.648017, 0.175738, 1.000000,
                3.291279, 8.361876, 1.000000, 0.276117
            ]
        )
    }
}
