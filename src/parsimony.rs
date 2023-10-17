use logging_timer::time;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    ops::{BitAnd, BitOr},
};

use crate::datastructures::PhyloTree;
use anyhow::{Context, Result};
use bitvec::prelude::*;

#[time("debug")]
pub fn parsimony_score(
    tree: &PhyloTree,
    char_map: &HashMap<u8, BitVec>,
    name_sequence_map: &HashMap<&String, &Vec<u8>>,
    sequence_length: usize,
) -> Result<usize> {
    Ok((0..sequence_length)
        .into_par_iter()
        .map(|i| -> Result<usize> {
            let (first_bitvec, first_score) =
                parsimony_recursion(tree, i, char_map, name_sequence_map)?;
            if let Some(third_root_child) = tree.third_child() {
                let (second_bitvec, second_score) = parsimony_recursion(
                    third_root_child,
                    i,
                    char_map,
                    name_sequence_map,
                )?;
                let intersection = first_bitvec.bitand(second_bitvec);
                Ok(first_score + second_score + intersection.not_any() as usize)
            } else {
                Ok(first_score)
            }
        }).collect::<Result<Vec<usize>>>()?
        .iter()
        .sum())
}

fn parsimony_recursion(
    node: &PhyloTree,
    site: usize,
    char_map: &HashMap<u8, BitVec>,
    name_sequence_map: &HashMap<&String, &Vec<u8>>,
) -> Result<(BitVec, usize)> {
    match node.name() {
        Some(name) => {
            let sequence = name_sequence_map.get(name).with_context(|| {
                format!("Unexpected sequence name {}", name)
            })?;
            Ok((
                char_map
                    .get(&(sequence[site]))
                    .with_context(|| {
                        format!(
                            "Unexpected character at site {}, {}",
                            site, sequence[site]
                        )
                    })
                    .cloned()?,
                0,
            ))
        }
        None => {
            let (left_bitvec, left_score) = parsimony_recursion(
                node.first_child(),
                site,
                char_map,
                name_sequence_map,
            )?;
            let (right_bitvec, right_score) = parsimony_recursion(
                node.second_child(),
                site,
                char_map,
                name_sequence_map,
            )?;
            let intersection = left_bitvec.clone().bitand(&right_bitvec);
            if intersection.not_any() {
                Ok((
                    left_bitvec.bitor(right_bitvec),
                    left_score + right_score + 1,
                ))
            } else {
                Ok((intersection, left_score + right_score))
            }
        }
    }
}

pub fn filter_trees_by_score(
    score_tree_pairs: &[(usize, usize)],
    num_trees_to_keep: usize,
) -> Vec<usize> {
    score_tree_pairs[..num_trees_to_keep]
        .iter()
        .map(|(_, i)| *i)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bitvec::prelude::*;
    use itertools::Itertools;

    use crate::{datastructures::PhyloTree, parsimony::parsimony_score};

    #[test]
    fn test_parsimony_score() {
        let num_chars = 5;
        let msa: Vec<Vec<u8>> = vec![
            vec![0, 0, 2, 2],
            vec![0, 0, 0, 4],
            vec![3, 3, 0, 3],
            vec![0, 2, 0, 2],
        ];
        let char_map = (0..num_chars as u8)
            .zip((0..num_chars).map(|i| {
                let mut bv = bitvec![0; num_chars - 1];
                if i == num_chars - 1 {
                    bv.fill(true);
                } else {
                    bv.set(i, true);
                }
                bv
            }))
            .collect::<HashMap<_, _>>();
        let labels = [
            "seq_1".to_string(),
            "seq_2".into(),
            "seq_3".into(),
            "seq_4".into(),
        ];
        let name_sequence_map = labels
            .iter()
            .zip_eq(msa.iter())
            .map(|(l, s)| (l, s))
            .collect::<HashMap<_, _>>();
        let sequence_length = 4;
        let mut trees: Vec<Option<PhyloTree>> = labels
            .iter()
            .map(|name| Some(PhyloTree::new_leaf(name)))
            .collect();
        trees[0] = Some(PhyloTree::join(vec![
            (trees[0].take().unwrap(), 1.0),
            (trees[1].take().unwrap(), 1.0),
        ]));
        let tree = PhyloTree::join(vec![
            (trees[0].take().unwrap(), 1.0),
            (trees[2].take().unwrap(), 1.0),
            (trees[3].take().unwrap(), 1.0),
        ]);
        assert_eq!(
            parsimony_score(
                &tree,
                &char_map,
                &name_sequence_map,
                sequence_length
            )
            .unwrap(),
            5
        );
    }
}
