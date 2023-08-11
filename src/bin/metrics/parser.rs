use bitvec::prelude::*;
use itertools::Itertools;
use regex::Regex;
use std::{collections::HashMap, ops::BitOr};

pub struct NewickParser {
    tokenizer: Tokenizer,
    mapping: HashMap<String, BitVec>,
    bipartitions: Vec<BitVec>,
}

impl NewickParser {
    pub fn new(input: &str, mapping: &HashMap<String, BitVec>) -> Self {
        let tokenizer = Tokenizer::new(input);
        NewickParser {
            tokenizer,
            mapping: mapping.clone(),
            bipartitions: Vec::new(),
        }
    }

    pub fn get_taxa_mapping(input: &str) -> HashMap<String, BitVec> {
        let taxa_regex = Regex::new(r"[A-Za-z0-9_|]+").unwrap();
        let mut taxa: Vec<String> = taxa_regex
            .find_iter(input)
            .map(|m| m.as_str().into())
            .collect_vec();
        taxa.sort();
        let n_taxa = taxa.len();
        taxa.into_iter()
            .zip((0..n_taxa).map(|i| {
                let mut bv = bitvec![0; n_taxa];
                bv.set(i, true);
                bv
            }))
            .collect::<HashMap<_, _>>()
    }

    pub fn parse(&mut self) -> Vec<BitVec> {
        self.parse_tree();
        debug_assert!(
            (0..=1).contains(&(self.mapping.len() - self.bipartitions.len()))
        );
        self.bipartitions.clone()
    }

    fn parse_tree(&mut self) -> BitVec {
        let token = self.tokenizer.get_next_token();
        match token {
            Token::OpenParen => {
                let left_child = self.parse_tree();
                self.tokenizer.expect_token(Token::Comma);
                let right_child = self.parse_tree();
                if self.tokenizer.get_next_token() == Token::Comma {
                    let third_child = self.parse_tree();
                    let extra_combined_bitset_1 =
                        left_child.clone().bitor(&third_child);
                    let extra_combined_bitset_2 =
                        right_child.clone().bitor(&third_child);
                    self.bipartitions.push(extra_combined_bitset_1.clone());
                    self.bipartitions.push(extra_combined_bitset_2.clone());
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
    }
}

#[derive(PartialEq, Debug, Clone)]
enum Token {
    OpenParen,
    CloseParen,
    Comma,
    Taxon(String),
}

struct Tokenizer {
    tokens: Vec<Token>,
    current_pos: usize,
}

impl Tokenizer {
    fn new(input: &str) -> Self {
        let clean_string = input.trim().replace(';', "").to_string();
        let branch_length_regex = Regex::new(r":[0-9.+eE-]+").unwrap();
        let inner_node_name_regex = Regex::new(r"(\))(1)").unwrap();
        let token_regex = Regex::new(r"[(),]|([A-Za-z0-9_|]+)").unwrap();
        let clean_string = branch_length_regex.replace_all(&clean_string, "");
        let clean_string = inner_node_name_regex
            .replace_all(&clean_string, "$1")
            .to_string();
        let mut s = clean_string.as_str();
        let mut tokens = Vec::new();
        while let Some(capture) = token_regex.captures(s) {
            s = &s[capture[0].len()..];
            if let Some(name) = capture.get(1) {
                tokens.push(Token::Taxon(name.as_str().to_string()))
            } else {
                match capture[0].as_ref() {
                    "(" => tokens.push(Token::OpenParen),
                    ")" => tokens.push(Token::CloseParen),
                    "," => tokens.push(Token::Comma),
                    _ => (),
                }
            }
        }
        // dbg!(&tokens);
        Self {
            tokens,
            current_pos: 0,
        }
    }

    fn get_next_token(&mut self) -> Token {
        let token = self.tokens[self.current_pos].clone();
        self.current_pos += 1;
        token
    }

    fn expect_token(&mut self, expected: Token) {
        if self.get_next_token() == expected {
            return;
        }
        panic!("Unexpected token");
    }
}
