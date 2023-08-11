use itertools::Itertools;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

mod parser;

fn main() {
    let args: Vec<String> = env::args().collect();
    let file_path = &args[1];
    let file = File::open(file_path).expect("Failed to open file");
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .map_while(|l| parser::NewickParser::preprocess_input(l.ok()?).ok())
        .collect::<Vec<String>>();
    let mapping = parser::NewickParser::get_taxa_mapping(&lines[0]);
    let all_biparts = lines
        .iter()
        .map(|l| parser::NewickParser::new(l, &mapping).parse())
        .collect_vec();
    dbg!(all_biparts.len());
}
