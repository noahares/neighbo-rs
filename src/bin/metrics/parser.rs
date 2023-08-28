use anyhow::{anyhow, bail, Result};
use bitvec::prelude::*;
use itertools::Itertools;
use logging_timer::time;
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    ops::BitOr,
};

pub struct NewickParser<'a, 'b> {
    tokenizer: Tokenizer<'a>,
    mapping: &'b HashMap<String, BitVec>,
    bipartitions: HashSet<BitVec>,
    num_taxa: usize,
}

impl<'a, 'b> NewickParser<'a, 'b> {
    pub fn new(input: &'a str, mapping: &'b HashMap<String, BitVec>) -> Self {
        let tokenizer = Tokenizer::new(input);
        NewickParser {
            tokenizer,
            mapping,
            bipartitions: HashSet::with_capacity(mapping.len()),
            num_taxa: mapping.len(),
        }
    }

    #[time("info")]
    pub fn get_taxa_mapping(input: &str) -> HashMap<String, BitVec> {
        let taxa_regex = Regex::new(r"[A-Za-z0-9_|]+").unwrap();
        let taxa: Vec<String> = taxa_regex
            .find_iter(input)
            .map(|m| m.as_str().into())
            .sorted()
            .collect_vec();
        let n_taxa = taxa.len();
        taxa.into_iter()
            .zip((0..n_taxa).map(|i| {
                let mut bv = bitvec![0; n_taxa];
                bv.set(i, true);
                bv
            }))
            .collect::<HashMap<_, _>>()
    }

    pub fn preprocess_input(input: String) -> Result<String> {
        if input.is_empty() {
            Err(anyhow!("Empty line"))
        } else {
            Ok(Regex::new(r":[0-9.+eE-]+")?
                .replace_all(
                    &input.trim().trim_end_matches(';').replace(")1", ")"),
                    "",
                )
                .to_string())
        }
    }

    pub fn parse(mut self) -> Vec<BitVec> {
        self.parse_tree().unwrap_or_else(|e| panic!("{}", e));
        debug_assert_eq!(
            self.num_taxa - 3,
            self.bipartitions
                .clone()
                .into_iter()
                .filter(Self::is_inner_bipartition)
                .count()
        );
        debug_assert!(self.bipartitions.iter().all(|b| b[0]));
        self.bipartitions
            .into_iter()
            .filter(Self::is_inner_bipartition)
            .collect_vec()
    }

    fn parse_tree(&mut self) -> Result<BitVec> {
        if let Some(token) = self.tokenizer.next() {
            match token {
                Token::OpenParen => {
                    let left_child = self.parse_tree()?;
                    self.tokenizer.expect_token(Token::Comma);
                    let right_child = self.parse_tree()?;
                    if self.tokenizer.next() == Some(Token::Comma) {
                        let third_child = self.parse_tree()?;
                        let extra_combined_bitset_1 =
                            third_child.clone().bitor(&left_child);
                        let extra_combined_bitset_2 =
                            third_child.bitor(&right_child);
                        self.insert_bipartition_normalized(
                            extra_combined_bitset_1,
                        );
                        self.insert_bipartition_normalized(
                            extra_combined_bitset_2,
                        );
                    }
                    let combined_bitset = left_child.bitor(right_child);
                    self.insert_bipartition_normalized(
                        combined_bitset.clone(),
                    );
                    Ok(combined_bitset)
                }
                Token::Taxon(name) => Ok(self.mapping[&name].clone()),
                _ => bail!(format!("Unexpected token: {:?}", token)),
            }
        } else {
            bail!("Unexpected end of string")
        }
    }

    fn insert_bipartition_normalized(&mut self, bipartition: BitVec) {
        if bipartition[0] {
            self.bipartitions.insert(bipartition);
        } else {
            self.bipartitions.insert(!bipartition);
        }
    }

    fn is_inner_bipartition(bipartition: &BitVec) -> bool {
        let n_ones = bipartition.count_ones();
        let n_zeros = bipartition.count_zeros();
        n_ones > 1 && n_zeros > 1
    }
}

#[derive(PartialEq, Debug, Clone)]
enum Token {
    OpenParen,
    CloseParen,
    Comma,
    Taxon(String),
}

struct Tokenizer<'a> {
    input: &'a str,
    token_regex: Regex,
}

impl<'a> Tokenizer<'a> {
    fn new(input: &'a str) -> Self {
        let token_regex = Regex::new(r"[(),]|([A-Za-z0-9_|]+)").unwrap();
        Self { input, token_regex }
    }

    fn expect_token(&mut self, expected_token: Token) {
        if self.next() != Some(expected_token.clone()) {
            panic!("Expected token: {:?}", expected_token);
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(captures) = self.token_regex.captures(self.input) {
            self.input = &self.input[captures[0].len()..];

            if let Some(name) = captures.get(1) {
                return Some(Token::Taxon(name.as_str().to_string()));
            }

            match captures[0].as_ref() {
                "(" => Some(Token::OpenParen),
                ")" => Some(Token::CloseParen),
                "," => Some(Token::Comma),
                _ => None,
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use bitvec::prelude::*;

    use super::NewickParser;

    #[test]
    fn test_preprocessing() {
        let input = String::from("((z,b:1.223)1,c:1223.332,d:132.323);");
        let result = NewickParser::preprocess_input(input).unwrap();
        assert_eq!(result, String::from("((z,b),c,d)"));
    }

    #[test]
    fn test_simple_tree_trichotomy() {
        let input = "((z,b),c,d)";
        let mapping = NewickParser::get_taxa_mapping(input);
        let result = NewickParser::new(input, &mapping).parse();
        assert_eq!(result, vec![bitvec![1, 0, 0, 1],]);
    }

    #[test]
    fn test_pure_trichotomy() {
        let input = "(a,c,d)";
        let mapping = NewickParser::get_taxa_mapping(input);
        let result = NewickParser::new(input, &mapping).parse();
        assert!(result.is_empty());
    }

    #[test]
    fn test_minimal_binary() {
        let input = "((a,b), c)";
        let mapping = NewickParser::get_taxa_mapping(input);
        let result = NewickParser::new(input, &mapping).parse();
        assert!(result.is_empty());
    }

    #[test]
    fn test_simple_tree_binary() {
        let input = "((z,b),((c,d), a))";
        let mapping = NewickParser::get_taxa_mapping(input);
        let result = NewickParser::new(input, &mapping).parse();
        assert!(vec![bitvec![1, 0, 1, 1, 0], bitvec![1, 1, 0, 0, 1],]
            .iter()
            .all(|b| result.contains(b)));
    }
}
