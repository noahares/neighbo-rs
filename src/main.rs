// based on https://docs.rs/phylogeny/latest/src/phylogeny/lib.rs.html#289-353
use anyhow::Result;
use itertools::Itertools;
use log::warn;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use clap::Parser;

mod datastructures;
mod nj;
mod io;

fn main() -> Result<()> {
    let args = io::Args::parse();
    if (args.noise == 0.0 || args.noise_ratio == 0.0) && args.num_trees > 1 {
        warn!("Generating {} identical trees! Consider increasing `--noise` or `--noise_ratio`", args.num_trees)
    }
    let distance_matrix = datastructures::DistanceMatrix::from_file(&args.sequence_file)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let distribution = rand_distr::Normal::new(1.0, args.noise)?;
    let trees: Vec<datastructures::PhyloTree> = (0..args.num_trees)
        .map(|_| -> Result<datastructures::PhyloTree> {
            let mut distance_matrix = distance_matrix.clone();
            distance_matrix.perturb(&mut rng, &distribution, args.noise_ratio);
            nj::nj(distance_matrix, args.strategy, args.seed, args.percentile)
        })
        .filter_map(Result::ok)
        .collect_vec();
    let mut output = args.get_output()?;
    for tree in &trees {
        writeln!(output, "{}", tree)?;
    }
    Ok(())
}
