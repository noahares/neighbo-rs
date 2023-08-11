use anyhow::Result;
use itertools::Itertools;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
mod parser;

fn main() {
    let args: Vec<String> = env::args().collect();
    // let input = "((z,b:1.223)1,c:1223.332,d:132.323);";

    let file_path = &args[1];
    let file = File::open(file_path).expect("Failed to open file");
    let reader = BufReader::new(file);
    let lines = reader.lines().map_while(Result::ok).collect_vec();
    let mapping = parser::NewickParser::get_taxa_mapping(&lines[0]);
    let all_biparts = lines
        .iter()
        .map(|l| {
            let mut newick_parser = parser::NewickParser::new(l, &mapping);
            newick_parser.parse()
        })
        .collect_vec();
    dbg!(all_biparts.len());
}
