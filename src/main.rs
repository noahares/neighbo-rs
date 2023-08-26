// based on https://docs.rs/phylogeny/latest/src/phylogeny/lib.rs.html#289-353
use anyhow::Result;
use clap::Parser;
use log::warn;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

mod datastructures;
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
    let distance_matrix =
        datastructures::DistanceMatrix::from_file(&args.sequence_file)?;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);
    let distribution = rand_distr::Normal::new(0.0, args.noise)?;
    let trees: Vec<datastructures::PhyloTree> = (0..args.num_trees)
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
            nj::nj(distance_matrix, args.strategy, &mut rng, args.percentile)
        })
        .collect::<Result<Vec<datastructures::PhyloTree>>>()?;
    let mut output = args.get_output()?;
    for tree in &trees {
        writeln!(output, "{}", tree)?;
    }
    Ok(())
}
