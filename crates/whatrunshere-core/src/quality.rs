//! How good the model is, and how much of that survives quantization.
//!
//! The usual approach scores quality from parameter count and a hand-assigned
//! "family reputation". That was defensible when models scaled uniformly. It is
//! not any more: a current 4B model beats a two-year-old 13B on most things a
//! person would actually ask it, and no amount of reputation weighting recovers
//! that from a size number.
//!
//! So quality here is anchored to published evaluations, weighted per use case,
//! and reported with the fraction of that weight actually backed by data. When
//! nothing is known the module says so, as [`QualityBasis::Inferred`], instead
//! of dressing a size prior up as a measurement.
//!
//! On top of that sits the second question, which almost nothing answers: what
//! does dropping to Q3 actually cost? Degradation is not uniform. The same
//! quantization that a 70B shrugs off can visibly damage a 3B, because smaller
//! models have less redundancy to spend. [`degradation_curve`] makes that
//! trade-off explicit.

use crate::arch::Architecture;
use crate::quant::{KvQuant, WeightQuant, WEIGHT_QUANTS};
use serde::{Deserialize, Serialize};

/// A published evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// MMLU-Pro: broad knowledge and reasoning, harder than the original MMLU.
    MmluPro,
    /// GPQA Diamond: graduate-level science questions.
    GpqaDiamond,
    /// `LiveCodeBench`: competitive programming on problems published after the
    /// models' training cutoffs.
    LiveCodeBench,
    /// MATH: competition mathematics.
    Math,
    /// `IFEval`: whether the model does what it was told.
    IfEval,
    /// `LMArena` Elo: aggregate human preference.
    ArenaElo,
    /// Long-context retrieval accuracy.
    LongContextRetrieval,
    /// Multilingual understanding.
    Multilingual,
}

/// Published scores for one model, all optional.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Benchmarks {
    /// MMLU-Pro accuracy, 0 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmlu_pro: Option<f64>,
    /// GPQA Diamond accuracy, 0 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpqa_diamond: Option<f64>,
    /// `LiveCodeBench` pass rate, 0 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub livecodebench: Option<f64>,
    /// MATH accuracy, 0 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub math: Option<f64>,
    /// `IFEval` strict accuracy, 0 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ifeval: Option<f64>,
    /// `LMArena` Elo rating, typically 1000 to 1450.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arena_elo: Option<f64>,
    /// Long-context retrieval accuracy, 0 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long_context: Option<f64>,
    /// Multilingual accuracy, 0 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multilingual: Option<f64>,
}

/// Elo below which a model scores zero on the normalised scale.
const ELO_FLOOR: f64 = 1000.0;
/// Elo span mapped onto the full 0 to 100 scale.
const ELO_SPAN: f64 = 500.0;

impl Benchmarks {
    /// One metric, normalised onto the 0 to 100 scale.
    ///
    /// Elo is rescaled onto the same range so it can be averaged with accuracy
    /// scores without dominating them.
    pub fn normalised(&self, metric: Metric) -> Option<f64> {
        let raw = match metric {
            Metric::MmluPro => self.mmlu_pro,
            Metric::GpqaDiamond => self.gpqa_diamond,
            Metric::LiveCodeBench => self.livecodebench,
            Metric::Math => self.math,
            Metric::IfEval => self.ifeval,
            Metric::LongContextRetrieval => self.long_context,
            Metric::Multilingual => self.multilingual,
            Metric::ArenaElo => {
                return self
                    .arena_elo
                    .map(|elo| ((elo - ELO_FLOOR) / ELO_SPAN * 100.0).clamp(0.0, 100.0))
            }
        };
        raw.map(|value| value.clamp(0.0, 100.0))
    }

    /// Whether any evaluation at all is recorded.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// What the model is going to be used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UseCase {
    /// No particular specialisation.
    General,
    /// Writing and modifying code.
    Coding,
    /// Multi-step reasoning and analysis.
    Reasoning,
    /// Mathematics.
    Math,
    /// Conversation, where following instructions matters more than depth.
    Chat,
    /// Agent loops, where instruction-following and tool discipline dominate.
    Agentic,
    /// Working over long documents.
    LongContext,
    /// Working across languages.
    Multilingual,
}

impl UseCase {
    /// Which evaluations matter for this use case, and how much.
    ///
    /// Weights sum to one, so the coverage figure below reads as a fraction.
    pub const fn weights(self) -> &'static [(Metric, f64)] {
        match self {
            Self::General => &[
                (Metric::MmluPro, 0.35),
                (Metric::ArenaElo, 0.25),
                (Metric::IfEval, 0.20),
                (Metric::GpqaDiamond, 0.20),
            ],
            Self::Coding => &[
                (Metric::LiveCodeBench, 0.55),
                (Metric::IfEval, 0.20),
                (Metric::MmluPro, 0.15),
                (Metric::ArenaElo, 0.10),
            ],
            Self::Reasoning => &[
                (Metric::GpqaDiamond, 0.40),
                (Metric::MmluPro, 0.30),
                (Metric::Math, 0.20),
                (Metric::ArenaElo, 0.10),
            ],
            Self::Math => &[
                (Metric::Math, 0.60),
                (Metric::GpqaDiamond, 0.20),
                (Metric::MmluPro, 0.20),
            ],
            Self::Chat => &[
                (Metric::ArenaElo, 0.45),
                (Metric::IfEval, 0.30),
                (Metric::MmluPro, 0.25),
            ],
            Self::Agentic => &[
                (Metric::IfEval, 0.40),
                (Metric::LiveCodeBench, 0.25),
                (Metric::MmluPro, 0.20),
                (Metric::LongContextRetrieval, 0.15),
            ],
            Self::LongContext => &[
                (Metric::LongContextRetrieval, 0.55),
                (Metric::MmluPro, 0.25),
                (Metric::IfEval, 0.20),
            ],
            Self::Multilingual => &[
                (Metric::Multilingual, 0.55),
                (Metric::MmluPro, 0.25),
                (Metric::ArenaElo, 0.20),
            ],
        }
    }

    /// A label for display.
    pub const fn label(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Coding => "Coding",
            Self::Reasoning => "Reasoning",
            Self::Math => "Mathematics",
            Self::Chat => "Chat",
            Self::Agentic => "Agentic",
            Self::LongContext => "Long context",
            Self::Multilingual => "Multilingual",
        }
    }

    /// Every use case, for enumeration in a UI.
    pub const ALL: &'static [Self] = &[
        Self::General,
        Self::Coding,
        Self::Reasoning,
        Self::Math,
        Self::Chat,
        Self::Agentic,
        Self::LongContext,
        Self::Multilingual,
    ];
}

/// What stands behind a quality score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityBasis {
    /// Most of the use case's weight is backed by published evaluations.
    Measured,
    /// Some evaluations exist, but a substantial share of the weight does not.
    Partial,
    /// Nothing published. The score is a size prior and should be read as one.
    Inferred,
}

impl QualityBasis {
    fn from_coverage(coverage: f64) -> Self {
        if coverage >= 0.7 {
            Self::Measured
        } else if coverage > 0.0 {
            Self::Partial
        } else {
            Self::Inferred
        }
    }

    /// A short label for display.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Measured => "from published evaluations",
            Self::Partial => "partly evaluated",
            Self::Inferred => "inferred from size",
        }
    }
}

/// A quality verdict for one model in one configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Quality at full precision, 0 to 100.
    pub full_precision: f64,
    /// Quality after weight and cache quantization, 0 to 100.
    pub quantized: f64,
    /// Points lost to quantization.
    pub degradation: f64,
    /// Fraction of the use case's weight backed by real evaluations.
    pub coverage: f64,
    /// What the score rests on.
    pub basis: QualityBasis,
}

/// Model size at which quantization sensitivity is taken as neutral.
const SENSITIVITY_PIVOT_B: f64 = 30.0;
/// How sharply sensitivity rises as models get smaller.
const SENSITIVITY_EXPONENT: f64 = 0.35;

/// How much more a model of this size suffers from quantization than the
/// reference size does.
///
/// Small models have less redundancy to give up, so the same format costs them
/// more. The exponent is fitted to published perplexity and benchmark deltas
/// across the range from 1B to 70B, and the result is clamped so that neither a
/// sub-billion model nor a frontier one leaves the plausible range.
pub fn quantization_sensitivity(total_params: u64) -> f64 {
    let billions = (total_params as f64 / 1e9).max(0.05);
    (SENSITIVITY_PIVOT_B / billions)
        .powf(SENSITIVITY_EXPONENT)
        .clamp(0.6, 4.0)
}

/// A size-only prior, used when nothing has been published.
///
/// Calibrated against MMLU-Pro results for well-known models: it lands close
/// for an 8B and a 70B of the era it was fitted to, and will read low for a
/// current small model. That bias is the reason the result is labelled
/// [`QualityBasis::Inferred`] rather than presented as a measurement.
fn size_prior(total_params: u64, active_params: u64) -> f64 {
    // Sparse models behave between their active and total size; the geometric
    // mean of the two tracks published results better than either alone.
    let total_b = (total_params as f64 / 1e9).max(0.05);
    let active_b = (active_params as f64 / 1e9).max(0.05);
    let effective = (total_b * active_b).sqrt();
    (20.0 + 18.0 * effective.log10()).clamp(5.0, 92.0)
}

/// Score a model for a use case, before and after quantization.
pub fn assess(
    benchmarks: &Benchmarks,
    use_case: UseCase,
    total_params: u64,
    active_params: u64,
    weight_quant: &WeightQuant,
    kv_quant: KvQuant,
) -> QualityAssessment {
    let mut weighted = 0.0;
    let mut covered = 0.0;
    for &(metric, weight) in use_case.weights() {
        if let Some(value) = benchmarks.normalised(metric) {
            weighted += value * weight;
            covered += weight;
        }
    }

    let coverage = covered.clamp(0.0, 1.0);
    let prior = size_prior(total_params, active_params);
    // Blend measured evidence with the prior in proportion to how much of the
    // use case it actually covers, so a single reported metric does not stand
    // in for the whole picture.
    let full_precision = if covered > 0.0 {
        weighted / covered * coverage + prior * (1.0 - coverage)
    } else {
        prior
    };

    let sensitivity = quantization_sensitivity(total_params);
    let degradation = (weight_quant.quality_cost + kv_quant.quality_cost()) * sensitivity;
    let quantized = (full_precision - degradation).max(0.0);

    QualityAssessment {
        full_precision,
        quantized,
        degradation,
        coverage,
        basis: QualityBasis::from_coverage(coverage),
    }
}

/// One rung of the quantization ladder, with what it costs and what it saves.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DegradationPoint {
    /// The quantization scheme.
    pub quant: WeightQuant,
    /// Quality retained, 0 to 100.
    pub quality: f64,
    /// Points lost against full precision.
    pub lost: f64,
    /// Size of the weights at this rung.
    pub bytes: u64,
    /// Fraction of the unquantized size this rung occupies.
    pub relative_size: f64,
}

/// Walk the quantization ladder for one model, showing the trade at each rung.
///
/// This is the answer to "should I drop to Q3 to make it fit": not a yes or no,
/// but the bytes saved against the quality given up, for *this* model, since
/// the same format costs a 3B far more than it costs a 70B.
///
/// Note that the quality column is not monotone, and should not be forced to
/// be. An I-quant beating the K-quant a rung above it, at a smaller size, is
/// precisely the reason to choose one.
pub fn degradation_curve(
    benchmarks: &Benchmarks,
    use_case: UseCase,
    arch: &Architecture,
    kv_quant: KvQuant,
) -> Vec<DegradationPoint> {
    let total_params = arch.total_params();
    let active_params = arch.active_params();
    let reference = WEIGHT_QUANTS
        .first()
        .map_or(1, |scheme| scheme.weight_bytes(arch))
        .max(1);
    WEIGHT_QUANTS
        .iter()
        .map(|quant| {
            let bytes = quant.weight_bytes(arch);
            let assessment = assess(
                benchmarks,
                use_case,
                total_params,
                active_params,
                quant,
                kv_quant,
            );
            DegradationPoint {
                quant: *quant,
                quality: assessment.quantized,
                lost: assessment.degradation,
                bytes,
                relative_size: bytes as f64 / reference as f64,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    // These assert exact zeroes and ones the code produces by construction.
    #![allow(clippy::float_cmp)]

    use super::*;
    use crate::arch::{AttentionKind, FfnKind, LayerLayout, LayerSpec};
    use crate::quant::weight_quant;

    fn qwen3_32b() -> Architecture {
        Architecture {
            hidden_size: 5120,
            heads: 64,
            intermediate_size: 25600,
            vocab_size: 151_936,
            layers: LayerLayout::uniform(
                64,
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
            max_context: 40_960,
        }
    }

    fn q4_k_m() -> WeightQuant {
        *weight_quant("Q4_K_M").expect("scheme exists")
    }

    fn f16() -> WeightQuant {
        *weight_quant("F16").expect("scheme exists")
    }

    /// Roughly Qwen3-32B's published profile.
    fn strong_generalist() -> Benchmarks {
        Benchmarks {
            mmlu_pro: Some(65.0),
            gpqa_diamond: Some(54.0),
            livecodebench: Some(30.0),
            math: Some(72.0),
            ifeval: Some(85.0),
            arena_elo: Some(1250.0),
            ..Benchmarks::default()
        }
    }

    /// A model that codes well but is otherwise unremarkable.
    fn coding_specialist() -> Benchmarks {
        Benchmarks {
            mmlu_pro: Some(48.0),
            livecodebench: Some(52.0),
            ifeval: Some(78.0),
            arena_elo: Some(1180.0),
            ..Benchmarks::default()
        }
    }

    #[test]
    fn elo_is_rescaled_onto_the_same_range_as_accuracy_scores() {
        let benchmarks = Benchmarks {
            arena_elo: Some(1250.0),
            ..Benchmarks::default()
        };
        let value = benchmarks
            .normalised(Metric::ArenaElo)
            .expect("elo is present");
        assert!((value - 50.0).abs() < 0.001, "got {value}");

        let floor = Benchmarks {
            arena_elo: Some(900.0),
            ..Benchmarks::default()
        };
        assert_eq!(floor.normalised(Metric::ArenaElo), Some(0.0));
    }

    #[test]
    fn the_use_case_decides_which_model_wins() {
        let generalist = assess(
            &strong_generalist(),
            UseCase::Coding,
            32_000_000_000,
            32_000_000_000,
            &f16(),
            KvQuant::F16,
        );
        let specialist = assess(
            &coding_specialist(),
            UseCase::Coding,
            14_000_000_000,
            14_000_000_000,
            &f16(),
            KvQuant::F16,
        );
        assert!(
            specialist.full_precision > generalist.full_precision,
            "for coding, the smaller specialist should win: {:.1} vs {:.1}",
            specialist.full_precision,
            generalist.full_precision
        );

        let generalist_reasoning = assess(
            &strong_generalist(),
            UseCase::Reasoning,
            32_000_000_000,
            32_000_000_000,
            &f16(),
            KvQuant::F16,
        );
        let specialist_reasoning = assess(
            &coding_specialist(),
            UseCase::Reasoning,
            14_000_000_000,
            14_000_000_000,
            &f16(),
            KvQuant::F16,
        );
        assert!(
            generalist_reasoning.full_precision > specialist_reasoning.full_precision,
            "for reasoning the ordering must reverse"
        );
    }

    #[test]
    fn weights_within_each_use_case_sum_to_one() {
        for use_case in UseCase::ALL {
            let total: f64 = use_case.weights().iter().map(|(_, w)| w).sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "{} weights sum to {total}",
                use_case.label()
            );
        }
    }

    #[test]
    fn an_unevaluated_model_is_labelled_as_inferred() {
        let assessment = assess(
            &Benchmarks::default(),
            UseCase::General,
            8_000_000_000,
            8_000_000_000,
            &f16(),
            KvQuant::F16,
        );
        assert_eq!(assessment.basis, QualityBasis::Inferred);
        assert_eq!(assessment.coverage, 0.0);
        assert!(assessment.full_precision > 0.0, "a prior is still a number");
    }

    #[test]
    fn partial_evidence_is_labelled_as_partial() {
        let sparse = Benchmarks {
            mmlu_pro: Some(60.0),
            ..Benchmarks::default()
        };
        // MMLU-Pro carries 0.35 of the General weighting: real, but not enough.
        let assessment = assess(
            &sparse,
            UseCase::General,
            8_000_000_000,
            8_000_000_000,
            &f16(),
            KvQuant::F16,
        );
        assert_eq!(assessment.basis, QualityBasis::Partial);
        assert!((assessment.coverage - 0.35).abs() < 1e-9);
    }

    #[test]
    fn small_models_lose_more_to_the_same_quantization() {
        let small = assess(
            &Benchmarks::default(),
            UseCase::General,
            3_000_000_000,
            3_000_000_000,
            &q4_k_m(),
            KvQuant::F16,
        );
        let large = assess(
            &Benchmarks::default(),
            UseCase::General,
            70_000_000_000,
            70_000_000_000,
            &q4_k_m(),
            KvQuant::F16,
        );
        assert!(
            small.degradation > large.degradation * 1.5,
            "3B lost {:.2} points, 70B lost {:.2}",
            small.degradation,
            large.degradation
        );
    }

    #[test]
    fn full_precision_costs_nothing_and_is_the_ceiling() {
        let assessment = assess(
            &strong_generalist(),
            UseCase::General,
            32_000_000_000,
            32_000_000_000,
            &f16(),
            KvQuant::F16,
        );
        assert_eq!(assessment.degradation, 0.0);
        assert!((assessment.quantized - assessment.full_precision).abs() < 1e-9);
    }

    #[test]
    fn the_ladder_trades_size_for_quality() {
        let arch = qwen3_32b();
        let curve = degradation_curve(&strong_generalist(), UseCase::General, &arch, KvQuant::F16);
        assert_eq!(curve.len(), WEIGHT_QUANTS.len());
        for pair in curve.windows(2) {
            assert!(
                pair[1].relative_size < pair[0].relative_size,
                "{} is not smaller than {}",
                pair[1].quant.id,
                pair[0].quant.id
            );
        }

        let top = curve.first().expect("ladder is not empty");
        let bottom = curve.last().expect("ladder is not empty");
        assert_eq!(top.lost, 0.0, "the top of the ladder is lossless");
        assert_eq!(top.relative_size, 1.0);
        assert!(
            bottom.quality < top.quality - 20.0,
            "the bottom rung should be visibly worse: {:.1} against {:.1}",
            bottom.quality,
            top.quality
        );
        assert!(
            bottom.relative_size < 0.2,
            "the bottom rung should be a fraction of the size, got {:.2}",
            bottom.relative_size
        );
    }

    #[test]
    fn a_quantized_cache_costs_less_quality_than_quantized_weights() {
        let params = 8_000_000_000;
        let weights_only = assess(
            &Benchmarks::default(),
            UseCase::General,
            params,
            params,
            &q4_k_m(),
            KvQuant::F16,
        );
        let cache_only = assess(
            &Benchmarks::default(),
            UseCase::General,
            params,
            params,
            &f16(),
            KvQuant::Q8_0,
        );
        assert!(
            cache_only.degradation < weights_only.degradation,
            "an 8-bit cache should be far cheaper than 4-bit weights"
        );
    }

    #[test]
    fn sparse_models_are_priced_between_their_active_and_total_size() {
        // Mixtral: 46.7B stored, 12.9B active. The prior should sit between the
        // two, not at either extreme.
        let sparse = size_prior(46_700_000_000, 12_900_000_000);
        let as_active = size_prior(12_900_000_000, 12_900_000_000);
        let as_total = size_prior(46_700_000_000, 46_700_000_000);
        assert!(
            as_active < sparse && sparse < as_total,
            "{as_active:.1} < {sparse:.1} < {as_total:.1}"
        );
    }
}
