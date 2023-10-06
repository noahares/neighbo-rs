use anyhow::{anyhow, Context, Result};
use serde::{de, Deserialize, Serialize};
use std::{collections::HashMap, fmt::Display, path::PathBuf, str::FromStr};

#[derive(Serialize, Clone, Debug, Copy)]
pub enum Moltype {
    #[serde(rename = "protein")]
    AA,
    #[serde(rename = "DNA")]
    Dna,
}

impl FromStr for Moltype {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "protein" => Ok(Self::AA),
            "DNA" => Ok(Self::Dna),
            _ => Err(anyhow!(format!("Unknown Moltype: {}", s))),
        }
    }
}

impl Display for Moltype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Moltype::AA => write!(f, "protein"),
            Moltype::Dna => write!(f, "DNA"),
        }
    }
}

impl<'de> Deserialize<'de> for Moltype {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        FromStr::from_str(&s).map_err(de::Error::custom)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Data {
    pub datasets: HashMap<String, DataSet>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DataSet {
    pub sequence_file: PathBuf,
    pub reference_tree: Option<PathBuf>,
    pub moltype: Moltype,
    pub seed: u32,
    pub num_trees: usize,
    pub difficulty: Option<f64>,
    pub reference_tool: Tool,
    pub tools: Vec<Tool>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub distribution_path: PathBuf,
    pub time: f64,
    pub perturbation: Option<f64>,
    pub ratio: Option<f64>,
    pub strategy: Option<String>,
    pub percentile: Option<f64>,
}

impl TryFrom<PathBuf> for Tool {
    type Error = anyhow::Error;
    fn try_from(path: PathBuf) -> Result<Self> {
        Ok(Self {
            name: path
                .file_name()
                .context(format!(
                    "Filename could not be identified from {}",
                    path.display()
                ))?
                .to_str()
                .unwrap()
                .to_string(),
            distribution_path: path,
            time: 0.0,
            perturbation: None,
            ratio: None,
            strategy: None,
            percentile: None,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Metadata {
    pub sequence_file: Option<PathBuf>,
    pub reference_tree: Option<PathBuf>,
    pub moltype: Option<Moltype>,
    pub seed: Option<u32>,
    pub num_trees: Option<usize>,
    pub difficulty: Option<f64>,
    pub perturbation: Option<f64>,
    pub ratio: Option<f64>,
    pub strategy: Option<String>,
    pub percentile: Option<f64>,
    pub reference_tool: String,
    pub tool: String,
}

impl From<(&DataSet, &Tool)> for Metadata {
    fn from((d, t): (&DataSet, &Tool)) -> Self {
        Self {
            sequence_file: Some(d.sequence_file.clone()),
            reference_tree: d.reference_tree.clone(),
            moltype: Some(d.moltype),
            seed: Some(d.seed),
            num_trees: Some(d.num_trees),
            difficulty: d.difficulty,
            perturbation: t.perturbation,
            ratio: t.ratio,
            strategy: t.strategy.clone(),
            percentile: t.percentile,
            reference_tool: d.reference_tool.name.clone(),
            tool: t.name.clone(),
        }
    }
}

impl Metadata {
    pub fn from_only_tools(reference_tool: &str, tool: &str) -> Self {
        Self {
            reference_tool: reference_tool.to_string(),
            tool: tool.to_string(),
            ..Default::default()
        }
    }
}
