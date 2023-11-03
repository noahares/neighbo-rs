use crate::nj::RandomizationStrategy;
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use clap_verbosity_flag::Verbosity;
use std::{io::Write, path::PathBuf};

fn normalized_ratio(s: &str) -> Result<f64> {
    let ratio: f64 = s
        .parse()
        .with_context(|| format!("`{s}` isn't a valid ratio"))?;
    if (0.0..=1.0).contains(&ratio) {
        Ok(ratio)
    } else {
        Err(anyhow!(format!("Ratio is not in range {}-{}", 0.0, 1.0)))
    }
}

#[derive(Parser)]
#[command(author, version, about)]
pub struct Args {
    /// Path to the sequence file
    #[arg(short = 'i', long)]
    pub sequence_file: PathBuf,
    /// Path to the model file (output of raxml-ng)
    #[arg(short = 'a', long)]
    pub model_file: Option<PathBuf>,
    /// Number of trees to generate
    #[arg(short = 't', long, default_value_t = 1)]
    pub num_trees: usize,
    /// Number of distance samples to generate
    #[arg(long, default_value_t = 100)]
    pub num_samples: usize,
    /// Shape parameter for distance priors
    #[arg(long = "shape", default_value_t = 1.0)]
    pub distance_prior_shape: f64,
    /// plot distance distribution
    #[arg(long = "plot")]
    pub plot_distance_distribution: Option<PathBuf>,
    /// Number of distance samples to discard as burnin
    #[arg(short, long, default_value_t = 10)]
    pub burnin: usize,
    /// stddev of noise as a multiplicative factor
    #[arg(short, long, default_value_t = 0.2, value_parser = normalized_ratio)]
    pub noise: f64,
    /// use same noise for all distances
    #[arg(long, default_value_t = false)]
    pub single_noise: bool,
    /// ratio of matrix entries that get perturbed in each iteration
    #[arg(short = 'r', long, default_value_t = 1.0, value_parser = normalized_ratio)]
    pub noise_ratio: f64,
    /// use ML distances with percentage noise
    #[arg(long = "percentage-noise", default_value_t = false)]
    pub ml_with_percentage: bool,
    /// Strategy for randomizing the neighbor joining steps
    #[arg(short = 'm', long, default_value_t = RandomizationStrategy::Deterministic, value_enum)]
    pub strategy: RandomizationStrategy,
    /// Use resampling of distances during NJ steps
    #[arg(long, default_value_t = false)]
    pub nj_resampling: bool,
    /// Percentile for threshold and sample strategies
    #[arg(short, long, default_value_t = 0.1)]
    pub percentile: f64,
    /// Seed for the noise
    #[arg(short, long, default_value_t = 0)]
    pub seed: u64,
    /// parsimony filtering factor
    #[arg(long, default_value_t = 1)]
    pub parsimony: usize,
    /// Output path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[command(flatten)]
    pub verbosity: Verbosity,
}

impl Args {
    pub fn get_output(&self) -> Result<Box<dyn Write>> {
        match self.output {
            Some(ref path) => Ok(std::fs::File::create(path)
                .map(|f| Box::new(f) as Box<dyn Write>)?),
            None => Ok(Box::new(std::io::stdout())),
        }
    }
}
