// based on https://docs.rs/phylogeny/latest/src/phylogeny/lib.rs.html#289-353
use anyhow::Result;
use clap::Parser;
use distance_distribution::{generate_distance_matrix_samples, MsaData};
use log::warn;
use rand_distr::Distribution;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[macro_use]
extern crate assert_float_eq;

mod datastructures;
mod distance_distribution;
mod io;
mod nj;

fn main() -> Result<()> {
    let args = io::Args::parse();
    if (args.noise == 0.0 || args.noise_ratio == 0.0)
        && args.num_trees > 1
        && args.strategy != nj::RandomizationStrategy::Deterministic
    {
        warn!("Generating {} identical trees! Consider increasing `--noise` or `--noise_ratio`", args.num_trees)
    }
    env_logger::Builder::new()
        .filter_level(args.verbosity.log_level_filter())
        .init();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);
    let scale = 1.0 / args.distance_prior_rate;
    let distribution =
        rand_distr::Gamma::new(args.distance_prior_shape, scale)?;
    let msa_data = MsaData::new(&args.sequence_file, &args.model_file)?;
    let x_0 = distribution.sample(&mut rng);
    let sample_matrix = generate_distance_matrix_samples(
        &msa_data,
        &distribution,
        args.seed,
        x_0,
        args.num_samples,
        args.burnin,
    );
    let trees: Vec<datastructures::PhyloTree> = (0..args.num_trees)
        .map(|_| -> Result<datastructures::PhyloTree> {
            let distance_matrix = sample_matrix.sample(&mut rng)?;
            nj::nj(distance_matrix, args.strategy, &mut rng, args.percentile)
        })
        .collect::<Result<Vec<datastructures::PhyloTree>>>()?;
    let mut output = args.get_output()?;
    for tree in &trees {
        writeln!(output, "{}", tree)?;
    }
    Ok(())
}
