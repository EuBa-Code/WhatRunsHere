//! The model catalog.
//!
//! Curated rather than scraped. Scraping `HuggingFace` produces tens of thousands
//! of entries — re-uploads, abandoned merges, single-file experiments with a
//! parameter count of two — and no ranking recovers a useful list from that. A
//! catalog whose entries are all real is worth more than one that is merely
//! large.
//!
//! Each entry carries the exact architecture and, where they exist, the *real
//! sizes of real files*. A computed size is accurate to about a percent; a
//! measured one is exact, and [`ModelEntry::build`] is how the rest of the
//! system gets at it.

use crate::arch::Architecture;
use crate::quality::Benchmarks;
use serde::{Deserialize, Serialize};

/// A published quantized build, with the size its file actually is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgufBuild {
    /// Quantization name, matching [`crate::quant::WEIGHT_QUANTS`].
    pub quant: String,
    /// `HuggingFace` repository the file lives in.
    pub repo: String,
    /// Filename within that repository.
    pub file: String,
    /// Exact size in bytes.
    pub bytes: u64,
}

impl GgufBuild {
    /// A command that downloads this build.
    pub fn download_command(&self) -> String {
        format!(
            "huggingface-cli download {} {} --local-dir .",
            self.repo, self.file
        )
    }
}

/// One model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// `HuggingFace` identifier the architecture was read from.
    pub id: String,
    /// Name to show a person.
    pub display_name: String,
    /// Family, for grouping.
    pub family: String,
    /// Exact tensor shapes.
    pub architecture: Architecture,
    /// Published evaluations, where any exist.
    #[serde(default)]
    pub benchmarks: Benchmarks,
    /// Published builds with measured sizes, largest first.
    #[serde(default)]
    pub builds: Vec<GgufBuild>,
    /// Licence identifier.
    #[serde(default)]
    pub license: Option<String>,
    /// Release date, as an ISO calendar date.
    #[serde(default)]
    pub released: Option<String>,
}

impl ModelEntry {
    /// The measured build for a quantization, if one was published.
    pub fn build(&self, quant: &str) -> Option<&GgufBuild> {
        self.builds
            .iter()
            .find(|build| build.quant.eq_ignore_ascii_case(quant))
    }

    /// Whether any measured build exists.
    pub fn has_measurements(&self) -> bool {
        !self.builds.is_empty()
    }

    /// Total stored parameters.
    pub fn total_params(&self) -> u64 {
        self.architecture.total_params()
    }

    /// Parameters read per generated token.
    pub fn active_params(&self) -> u64 {
        self.architecture.active_params()
    }

    /// A short size label such as `32B` or `30B-A3B`.
    ///
    /// Sparse models get both numbers, because quoting only the total makes a
    /// 30B model that reads 3B per token sound ten times heavier than it runs.
    pub fn size_label(&self) -> String {
        let billions = |n: u64| n as f64 / 1e9;
        let total = billions(self.total_params());
        let format_one = |value: f64| {
            if value < 1.0 {
                format!("{:.0}M", value * 1000.0)
            } else if value < 10.0 {
                format!("{value:.1}B")
            } else {
                format!("{value:.0}B")
            }
        };
        if self.architecture.is_sparse() {
            format!(
                "{}-A{}",
                format_one(total),
                format_one(billions(self.active_params()))
            )
        } else {
            format_one(total)
        }
    }
}

/// The catalog as published.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    /// Schema version, so an old binary can refuse a newer file rather than
    /// misread it.
    pub version: u32,
    /// When the catalog was generated.
    pub generated: String,
    /// The models.
    pub models: Vec<ModelEntry>,
}

/// The schema version this build understands.
pub const SUPPORTED_VERSION: u32 = 1;

impl Catalog {
    /// Exact lookup by identifier, ignoring case.
    pub fn find(&self, id: &str) -> Option<&ModelEntry> {
        self.models
            .iter()
            .find(|model| model.id.eq_ignore_ascii_case(id))
    }

    /// Models whose identifier, display name or family contains `query`.
    ///
    /// Ordered by how early the match starts, so typing "qwen3-32" reaches
    /// Qwen3 32B before Qwen3 32B's coder sibling.
    pub fn search(&self, query: &str) -> Vec<&ModelEntry> {
        let needle = query.to_ascii_lowercase();
        let mut hits: Vec<(usize, &ModelEntry)> = self
            .models
            .iter()
            .filter_map(|model| {
                let haystacks = [&model.id, &model.display_name, &model.family];
                haystacks
                    .iter()
                    .filter_map(|text| text.to_ascii_lowercase().find(&needle))
                    .min()
                    .map(|position| (position, model))
            })
            .collect();
        hits.sort_by_key(|(position, model)| (*position, model.total_params()));
        hits.into_iter().map(|(_, model)| model).collect()
    }

    /// Whether this build can read the catalog.
    pub fn is_supported(&self) -> bool {
        self.version <= SUPPORTED_VERSION
    }

    /// Measured builds across the whole catalog.
    pub fn measured_builds(&self) -> usize {
        self.models.iter().map(|m| m.builds.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{AttentionKind, FfnKind, LayerLayout, LayerSpec, MoeSpec};

    fn dense(id: &str, name: &str, family: &str, layers: u32, hidden: u32) -> ModelEntry {
        ModelEntry {
            id: id.to_owned(),
            display_name: name.to_owned(),
            family: family.to_owned(),
            architecture: Architecture {
                hidden_size: hidden,
                heads: 32,
                intermediate_size: hidden * 4,
                vocab_size: 151_936,
                layers: LayerLayout::uniform(
                    layers,
                    LayerSpec::global(AttentionKind::Grouped {
                        kv_heads: 8,
                        head_dim: 128,
                    }),
                ),
                ffn: FfnKind::Gated,
                tied_embeddings: false,
                moe: None,
                norms_per_layer: 2,
                softcapped_attention: false,
                native_expert_quant: None,
                max_context: 32_768,
            },
            benchmarks: Benchmarks::default(),
            builds: Vec::new(),
            license: None,
            released: None,
        }
    }

    fn catalog() -> Catalog {
        Catalog {
            version: 1,
            generated: "2026-09-10".to_owned(),
            models: vec![
                dense("Qwen/Qwen3-32B", "Qwen3 32B", "Qwen3", 64, 5120),
                dense(
                    "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "Qwen2.5-Coder 32B",
                    "Qwen2.5-Coder",
                    64,
                    5120,
                ),
                dense("Qwen/Qwen3-8B", "Qwen3 8B", "Qwen3", 36, 4096),
            ],
        }
    }

    #[test]
    fn lookup_is_exact_and_case_insensitive() {
        let catalog = catalog();
        assert!(catalog.find("qwen/qwen3-32b").is_some());
        assert!(catalog.find("Qwen/Qwen3-32B").is_some());
        assert!(catalog.find("qwen3").is_none(), "find is not a search");
    }

    #[test]
    fn search_prefers_the_earlier_match() {
        let catalog = catalog();
        let hits = catalog.search("Qwen3");
        assert!(hits.len() >= 2);
        // "Qwen3" appears at position 5 in "Qwen/Qwen3-32B" and only inside the
        // family for the coder model, so the plain Qwen3 models come first.
        assert_eq!(hits[0].family, "Qwen3");
    }

    #[test]
    fn search_ties_break_towards_the_smaller_model() {
        let catalog = catalog();
        let hits = catalog.search("qwen3");
        let qwen3: Vec<_> = hits.iter().filter(|m| m.family == "Qwen3").collect();
        assert!(
            qwen3[0].total_params() < qwen3[1].total_params(),
            "an equally good match should offer the cheaper model first"
        );
    }

    #[test]
    fn a_measured_build_is_found_however_it_is_spelled() {
        let mut model = dense("test/model", "Test", "Test", 4, 512);
        model.builds.push(GgufBuild {
            quant: "Q4_K_M".to_owned(),
            repo: "someone/test-GGUF".to_owned(),
            file: "test-Q4_K_M.gguf".to_owned(),
            bytes: 1_234_567,
        });
        assert!(model.has_measurements());
        assert_eq!(model.build("q4_k_m").map(|b| b.bytes), Some(1_234_567));
        assert!(model.build("Q8_0").is_none());
        assert!(model
            .build("Q4_K_M")
            .expect("present")
            .download_command()
            .contains("someone/test-GGUF"));
    }

    #[test]
    fn sparse_models_are_labelled_with_both_sizes() {
        let mut model = dense("test/moe", "Test MoE", "Test", 48, 2048);
        model.architecture.intermediate_size = 6144;
        model.architecture.moe = Some(MoeSpec {
            experts: 128,
            experts_per_token: 8,
            expert_intermediate: 768,
            shared_experts: 0,
            shared_intermediate: 0,
            dense_layers: 0,
        });
        let label = model.size_label();
        assert!(
            label.contains('A'),
            "a sparse model needs both figures: {label}"
        );
        assert!(model.active_params() < model.total_params());
    }

    #[test]
    fn a_newer_schema_is_refused_rather_than_misread() {
        let mut catalog = catalog();
        assert!(catalog.is_supported());
        catalog.version = SUPPORTED_VERSION + 1;
        assert!(!catalog.is_supported());
    }
}
