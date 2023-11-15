use anyhow::Result;
use clap::Parser;
use clap_verbosity_flag::Verbosity;
use log::info;
use neighbo_rs::distance_distribution::generate_distance_matrix_samples;
use neighbo_rs::distance_distribution::MsaData;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args = Args::parse();
    env_logger::Builder::new()
        .filter_level(args.verbosity.log_level_filter())
        .init();
    let msa_data = MsaData::new(&args.sequence_file, &args.model_file, true)?;
    info!("Found MSA. Precomputing in distance distributions");
    let scale = msa_data.average_pairwise_distance / args.distance_prior_shape;
    let distribution =
        rand_distr::Gamma::new(args.distance_prior_shape, scale)?;
    let sample_matrix = generate_distance_matrix_samples(
        &msa_data,
        &distribution,
        args.seed,
        args.num_samples,
        args.burnin,
        None,
        &args.plot_distance_distribution,
    )?;
    let output = std::fs::File::create(args.output)?;
    serde_json::to_writer(output, &sample_matrix)?;
    Ok(())
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
    /// Seed for the noise
    #[arg(short, long, default_value_t = 0)]
    pub seed: u64,
    /// Output path
    #[arg(short, long)]
    pub output: PathBuf,
    #[command(flatten)]
    pub verbosity: Verbosity,
}
