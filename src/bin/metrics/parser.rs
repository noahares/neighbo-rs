use anyhow::Result;
use bitvec::prelude::*;
use itertools::Itertools;
use regex::Regex;
use std::{collections::HashMap, ops::BitOr};

pub struct NewickParser<'a, 'b> {
    tokenizer: Tokenizer<'a>,
    mapping: &'b HashMap<String, BitVec>,
    bipartitions: Vec<BitVec>,
}

impl<'a, 'b> NewickParser<'a, 'b> {
    pub fn new(input: &'a str, mapping: &'b HashMap<String, BitVec>) -> Self {
        let tokenizer = Tokenizer::new(input);
        NewickParser {
            tokenizer,
            mapping,
            bipartitions: Vec::with_capacity(mapping.len()),
        }
    }

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
        Ok(Regex::new(r":[0-9.+eE-]+")?
            .replace_all(
                &input.trim().trim_end_matches(';').replace(")1", ")"),
                "",
            )
            .to_string())
    }

    pub fn parse(&mut self) -> Vec<BitVec> {
        self.parse_tree();
        debug_assert!(
            (0..=1).contains(&(self.mapping.len() - self.bipartitions.len()))
        );
        self.bipartitions.clone()
    }

    fn parse_tree(&mut self) -> BitVec {
        if let Some(token) = self.tokenizer.next() {
            match token {
                Token::OpenParen => {
                    let left_child = self.parse_tree();
                    self.tokenizer.expect_token(Token::Comma);
                    let right_child = self.parse_tree();
                    if self.tokenizer.next() == Some(Token::Comma) {
                        let third_child = self.parse_tree();
                        let extra_combined_bitset_1 =
                            left_child.clone().bitor(&third_child);
                        let extra_combined_bitset_2 =
                            right_child.clone().bitor(&third_child);
                        self.bipartitions
                            .push(extra_combined_bitset_1.clone());
                        self.bipartitions
                            .push(extra_combined_bitset_2.clone());
                    }
                    let combined_bitset = left_child.bitor(right_child);
                    self.bipartitions.push(combined_bitset.clone());
                    combined_bitset
                }
                Token::Taxon(name) => {
                    self.mapping[&name].clone()
                    // do not need to add trivial partitions, as long as we know the number of taxa
                    // self.bipartitions.push(bitset.clone());
                    // bitset.to_owned()
                }
                _ => panic!("Unexpected token: {:?}", token),
            }
        } else {
            panic!("Unexpected end of string")
        }
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
