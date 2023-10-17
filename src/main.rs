// based on https://docs.rs/phylogeny/latest/src/phylogeny/lib.rs.html#289-353
use anyhow::{bail, Result};
use clap::Parser;
use distance_distribution::{generate_distance_matrix_samples, MsaData};
use itertools::Itertools;
use log::{debug, info, warn};
use logging_timer::{timer, Level};
use rand_distr::Distribution;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::io::Write;

use neighbo_rs::datastructures;
use neighbo_rs::distance_distribution;
use neighbo_rs::io;
use neighbo_rs::nj;
use neighbo_rs::parsimony;

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
    let trees: Vec<datastructures::PhyloTree> = if let Ok(distance_matrix) =
        datastructures::DistanceMatrix::from_file(&args.sequence_file)
    {
        info!("Found distance matrix. Running in randomized-noise mode");
        let distribution = rand_distr::Normal::new(0.0, args.noise)?;
        (0..args.num_trees)
            .map(|i| -> Result<datastructures::PhyloTree> {
                let distance_matrix = if i > 0 {
                    distance_matrix.perturb(
                        &mut rng,
                        &distribution,
                        args.noise_ratio,
                        args.single_noise,
                    )
                } else {
                    distance_matrix.clone()
                };
                nj::nj(
                    distance_matrix,
                    args.strategy,
                    &mut rng,
                    args.percentile,
                )
            })
            .collect::<Result<Vec<datastructures::PhyloTree>>>()?
    } else if let Ok(msa_data) =
        MsaData::new(&args.sequence_file, &args.model_file)
    {
        info!("Found MSA. Running in distance distribution mode");
        let scale =
            msa_data.average_pairwise_distance / args.distance_prior_shape;
        let distribution =
            rand_distr::Gamma::new(args.distance_prior_shape, scale)?;
        let x_0 = distribution.sample(&mut rng);
        let sample_matrix = generate_distance_matrix_samples(
            &msa_data,
            &distribution,
            args.seed,
            x_0,
            args.num_samples,
            args.burnin,
        );
        if let Some(distance_distribution_output_path) =
            args.plot_distance_distribution.clone()
        {
            sample_matrix.plot_distance_distribution(
                distance_distribution_output_path,
            )?;
        }
        let noise_distribution = rand_distr::Normal::new(0.0, args.noise)?;
        (0..args.num_trees)
            .map(|i| -> Result<datastructures::PhyloTree> {
                let distance_matrix = if i == 0 || args.noise_ratio == 0.0 {
                    sample_matrix.ml_distances()
                } else if args.ml_with_percentage {
                    Ok(sample_matrix.ml_distances()?.perturb(
                        &mut rng,
                        &noise_distribution,
                        args.noise_ratio,
                        args.single_noise,
                    ))
                } else {
                    sample_matrix.sample(
                        &mut rng,
                        args.noise_ratio,
                        args.noise,
                    )
                }?;
                nj::nj(
                    distance_matrix,
                    args.strategy,
                    &mut rng,
                    args.percentile,
                )
            })
            .collect::<Result<Vec<datastructures::PhyloTree>>>()?
    } else {
        bail!("Input was neither a distance matrix nor a MSA!")
    };
    let mut output = args.get_output()?;
    for tree in &trees {
        writeln!(output, "{}", tree)?;
    }
    Ok(())
}
