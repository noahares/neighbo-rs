// based on https://docs.rs/phylogeny/latest/src/phylogeny/lib.rs.html#289-353
use anyhow::{bail, Result};
use clap::Parser;
use distance_distribution::{generate_distance_matrix_samples, MsaData};
use itertools::Itertools;
use log::{debug, info, warn};
use logging_timer::finish;
use logging_timer::{timer, Level};
use neighbo_rs::distance_distribution::DistanceMatrixSamples;
use rand_chacha::ChaCha8Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
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
    let _total_tmr = timer!(Level::Info; "Total Runtime");
    if let Ok(distance_matrix) =
        datastructures::DistanceMatrix::from_file(&args.sequence_file)
    {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);
        info!("Found distance matrix. Running in randomized-noise mode");
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
                nj::nj(
                    distance_matrix,
                    args.strategy,
                    &mut rng,
                    args.percentile,
                )
            })
            .collect::<Result<Vec<datastructures::PhyloTree>>>()?;
        let mut output = args.get_output()?;
        for tree in &trees {
            writeln!(output, "{}", tree)?;
        }
    } else if let Ok(msa_data) = MsaData::new(
        &args.sequence_file,
        &args.model_file,
        args.precomputed_distances.is_none(),
    ) {
        info!("Found MSA. Running in distance distribution mode");
        let sample_matrix = if let Some(ref precomputed_distances_path) =
            args.precomputed_distances
        {
            let sample_str =
                std::fs::read_to_string(precomputed_distances_path)?;
            let mut sample_matrix: DistanceMatrixSamples =
                serde_json::from_str(&sample_str)?;
            sample_matrix.approximate_from_samples(args.noise)?;
            sample_matrix
        } else {
            let scale =
                msa_data.average_pairwise_distance / args.distance_prior_shape;
            let distribution =
                rand_distr::Gamma::new(args.distance_prior_shape, scale)?;
            generate_distance_matrix_samples(
                &msa_data,
                &distribution,
                args.seed,
                args.num_samples,
                args.burnin,
                Some(args.noise),
                &args.plot_distance_distribution,
            )?
        };
        let noise_distribution = rand_distr::Normal::new(0.0, args.noise)?;
        let trees = (0..args.num_trees * args.parsimony)
            .into_par_iter()
            .map(|i| -> Result<datastructures::PhyloTree> {
                let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
                rng.set_stream(i as u64);
                if args.nj_resampling {
                    nj::resampling_nj(
                        sample_matrix.clone(),
                        &mut rng,
                        args.noise_ratio,
                    )
                } else {
                    let distance_matrix = if i == 0 || args.noise_ratio == 0.0
                    {
                        sample_matrix.ml_distance_matrix()
                    } else if args.ml_with_percentage {
                        sample_matrix.ml_distance_matrix().perturb(
                            &mut rng,
                            &noise_distribution,
                            args.noise_ratio,
                            args.single_noise,
                        )
                    } else {
                        sample_matrix
                            .sample_ml_fallback(&mut rng, args.noise_ratio)
                    };
                    nj::nj(
                        distance_matrix,
                        args.strategy,
                        &mut rng,
                        args.percentile,
                    )
                }
            })
            .collect::<Result<Vec<datastructures::PhyloTree>>>()?;

        let mut output = args.get_output()?;
        if args.parsimony > 1 {
            let char_map = msa_data.get_char_map();
            let name_sequence_map = msa_data.label_sequence_map();
            // let sequence_length = msa_data.sequence_length();
            let parsimony_scores = {
                let _tmr = timer!(Level::Info; "Compute all Parsimony scores");
                trees
                    .par_iter()
                    .map(|t| {
                        parsimony::parsimony_score_sequential(
                            t,
                            &char_map,
                            &name_sequence_map,
                            // sequence_length,
                        )
                    })
                    .collect::<Result<Vec<usize>>>()?
            };
            debug!("Parsimony Scores: \n{:?}", parsimony_scores);
            let score_tree_pairs: Vec<(usize, usize)> = parsimony_scores
                .into_iter()
                .zip_eq(0..trees.len())
                .sorted_by(|a, b| a.0.cmp(&b.0))
                .collect();
            let trees_to_keep: Vec<usize> = parsimony::filter_trees_by_score(
                &score_tree_pairs,
                args.num_trees,
            );
            if let Some(path) = args.output {
                let parsimony_path =
                    path.with_extension("parsimony_filtered.nwk");
                let mut parsimony_file =
                    std::fs::File::create(parsimony_path)?;
                for i in &trees_to_keep {
                    writeln!(parsimony_file, "{}", trees[*i])?;
                }
                let all_samples_path =
                    path.with_extension("all_samples_with_scores.nwkx");
                let mut all_samples_file =
                    std::fs::File::create(all_samples_path)?;
                for (s, i) in &score_tree_pairs {
                    writeln!(all_samples_file, "{} {}", trees[*i], s)?;
                }
            }
        }
        for tree in &trees[..args.num_trees] {
            writeln!(output, "{}", tree)?;
        }
    } else {
        bail!("Input was neither a distance matrix nor a MSA!")
    };
    finish!(_total_tmr);
    Ok(())
}
