//! Quantization schemes for weights and for the attention cache.
//!
//! The usual shortcut is to multiply a parameter count by one bits-per-weight
//! number. That is wrong in a way that matters: llama.cpp's K-quants are mixed
//! precision, keeping the output projection at `Q6_K` while the body goes to
//! `Q4_K`. For a 7B model with a 32k vocabulary the error is a rounding
//! difference. For a 1B model with a 256k vocabulary (Gemma 3), the output
//! projection is a quarter of the file, and a flat bits-per-weight estimate is
//! off by enough to change the answer.
//!
//! So schemes here carry three numbers, one per tensor class. They remain a
//! fallback: when the catalog knows the real size of a real GGUF file, that
//! measurement wins. See [`crate::memory`].

use crate::arch::Architecture;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Which quantization lineage a scheme belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantFamily {
    /// Unquantized floating point.
    Float,
    /// The original round-to-nearest block formats: `Q4_0`, `Q5_0`, `Q8_0`.
    Legacy,
    /// K-quants: mixed-precision blocks of 256 with per-sub-block scales.
    KQuant,
    /// I-quants: importance-matrix formats, smaller but slower to decode.
    IQuant,
    /// Microscaling formats such as `MXFP4`.
    Microscaling,
}

impl QuantFamily {
    /// Relative cost of decoding this format during inference.
    ///
    /// I-quants trade decode speed for size: they need a lookup table per
    /// block, so a memory-bound model that only counts bytes will overestimate
    /// their throughput. This factor corrects for that.
    pub const fn decode_efficiency(self) -> f64 {
        match self {
            Self::Float | Self::Legacy => 1.0,
            Self::KQuant | Self::Microscaling => 0.97,
            Self::IQuant => 0.88,
        }
    }
}

/// A weight quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightQuant {
    /// Canonical name, as it appears in GGUF filenames.
    pub id: &'static str,
    /// Lineage, which determines decode cost.
    pub family: QuantFamily,
    /// Effective bits per weight across attention and feed-forward tensors.
    ///
    /// Back-computed from published llama.cpp file sizes after removing the
    /// embedding and output tensors, so it describes the body alone.
    pub body_bpw: f64,
    /// Bits per weight for the token embedding table.
    pub embedding_bpw: f64,
    /// Bits per weight for the output projection, which llama.cpp keeps at
    /// higher precision than the body for every K-quant.
    pub lm_head_bpw: f64,
    /// Quality lost relative to bf16, in normalised points on a 0 to 100 scale.
    ///
    /// Calibrated from published perplexity and KL-divergence measurements.
    /// Zero means indistinguishable from the unquantized model.
    pub quality_cost: f64,
}

/// A scheme travels as its name and is resolved against the registry on the way
/// back, so a client cannot invent a format with arbitrary properties, and the
/// numbers in [`WEIGHT_QUANTS`] stay the single source of truth.
impl Serialize for WeightQuant {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.id)
    }
}

impl<'de> Deserialize<'de> for WeightQuant {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let id = String::deserialize(deserializer)?;
        weight_quant(&id)
            .copied()
            .ok_or_else(|| serde::de::Error::custom(format!("unknown quantization `{id}`")))
    }
}

/// Bytes needed to store `params` weights at `bpw` bits each.
fn bytes_at(params: u64, bpw: f64) -> u64 {
    ((params as f64) * bpw / 8.0).round() as u64
}

impl WeightQuant {
    /// Bytes of storage for this model's weights.
    ///
    /// A tied model stores one tensor that serves as both embedding table and
    /// output projection, and llama.cpp quantizes it at the *output* precision
    /// rather than the embedding's. Verified against published GGUF builds:
    /// assuming otherwise underestimates a tied 4B model by up to 8%, and the
    /// error grows as the format gets more aggressive.
    pub fn weight_bytes(&self, arch: &Architecture) -> u64 {
        let params = arch.params();
        let vocabulary = if arch.tied_embeddings {
            bytes_at(params.embedding, self.lm_head_bpw)
        } else {
            bytes_at(params.embedding, self.embedding_bpw)
                + bytes_at(params.lm_head, self.lm_head_bpw)
        };
        bytes_at(params.body() - params.expert_ffn, self.body_bpw)
            + bytes_at(params.expert_ffn, self.expert_bpw(arch))
            + vocabulary
    }

    /// Bits per weight the experts are actually stored at.
    ///
    /// Normally the body figure. For a model trained quantized, the format its
    /// experts were trained in, because no published build converts them away
    /// from it. See [`Architecture::native_expert_quant`].
    fn expert_bpw(&self, arch: &Architecture) -> f64 {
        arch.native_expert_quant
            .as_deref()
            .and_then(weight_quant)
            .map_or(self.body_bpw, |native| {
                // A build coarser than the native format does quantize them.
                self.body_bpw.min(native.body_bpw)
            })
    }

    /// Bytes read from memory to generate one token.
    ///
    /// The output projection is read in full every token, at its own precision,
    /// whether or not it shares storage with the embedding table.
    pub fn decode_traffic_bytes(&self, arch: &Architecture) -> u64 {
        let traffic = arch.decode_traffic();
        bytes_at(traffic.body() - traffic.expert_ffn, self.body_bpw)
            + bytes_at(traffic.expert_ffn, self.expert_bpw(arch))
            + bytes_at(traffic.lm_head, self.lm_head_bpw)
    }

    /// Average bits per weight across the whole file, for this architecture.
    ///
    /// Unlike the constants quoted in quantization tables, this is specific to
    /// the model: a large vocabulary pulls it up.
    pub fn effective_bpw(&self, arch: &Architecture) -> f64 {
        let total = arch.total_params();
        if total == 0 {
            return 0.0;
        }
        (self.weight_bytes(arch) as f64) * 8.0 / (total as f64)
    }
}

/// Every weight scheme `WhatLLM` understands, ordered from highest fidelity to
/// most compressed. The order is the ladder the fit solver walks down.
pub static WEIGHT_QUANTS: &[WeightQuant] = &[
    WeightQuant {
        id: "F16",
        family: QuantFamily::Float,
        body_bpw: 16.0,
        embedding_bpw: 16.0,
        lm_head_bpw: 16.0,
        quality_cost: 0.0,
    },
    WeightQuant {
        id: "Q8_0",
        family: QuantFamily::Legacy,
        body_bpw: 8.5,
        embedding_bpw: 8.5,
        lm_head_bpw: 8.5,
        quality_cost: 0.1,
    },
    WeightQuant {
        id: "Q6_K",
        family: QuantFamily::KQuant,
        body_bpw: 6.56,
        embedding_bpw: 6.5625,
        lm_head_bpw: 8.5,
        quality_cost: 0.4,
    },
    WeightQuant {
        id: "Q5_K_M",
        family: QuantFamily::KQuant,
        body_bpw: 5.67,
        embedding_bpw: 5.5,
        lm_head_bpw: 6.5625,
        quality_cost: 1.1,
    },
    WeightQuant {
        id: "Q5_K_S",
        family: QuantFamily::KQuant,
        body_bpw: 5.52,
        embedding_bpw: 5.5,
        lm_head_bpw: 6.5625,
        quality_cost: 1.5,
    },
    WeightQuant {
        id: "Q4_K_M",
        family: QuantFamily::KQuant,
        body_bpw: 4.82,
        embedding_bpw: 4.5,
        lm_head_bpw: 6.5625,
        quality_cost: 3.0,
    },
    WeightQuant {
        id: "Q4_K_S",
        family: QuantFamily::KQuant,
        body_bpw: 4.55,
        embedding_bpw: 4.5,
        lm_head_bpw: 6.5625,
        quality_cost: 4.0,
    },
    WeightQuant {
        id: "MXFP4",
        family: QuantFamily::Microscaling,
        body_bpw: 4.25,
        embedding_bpw: 4.25,
        lm_head_bpw: 8.5,
        quality_cost: 3.2,
    },
    WeightQuant {
        id: "IQ4_XS",
        family: QuantFamily::IQuant,
        body_bpw: 4.25,
        embedding_bpw: 4.25,
        lm_head_bpw: 6.5625,
        quality_cost: 4.2,
    },
    // Solved from published builds of a 32B and an 8B simultaneously: 4.24 and
    // 4.50, the figures the usual tables imply, overshoot both by 1.6-2.5%.
    WeightQuant {
        id: "Q3_K_L",
        family: QuantFamily::KQuant,
        body_bpw: 4.19,
        embedding_bpw: 3.4375,
        lm_head_bpw: 6.5625,
        quality_cost: 7.0,
    },
    WeightQuant {
        id: "Q3_K_M",
        family: QuantFamily::KQuant,
        body_bpw: 3.88,
        embedding_bpw: 3.4375,
        lm_head_bpw: 6.5625,
        quality_cost: 9.0,
    },
    WeightQuant {
        id: "Q3_K_S",
        family: QuantFamily::KQuant,
        body_bpw: 3.46,
        embedding_bpw: 3.4375,
        lm_head_bpw: 5.5,
        quality_cost: 13.0,
    },
    // Slightly smaller than Q3_K_S and distinctly better: that inversion is the
    // whole reason I-quants exist, and it is paid for in decode speed rather
    // than in size. See `QuantFamily::decode_efficiency`.
    WeightQuant {
        id: "IQ3_S",
        family: QuantFamily::IQuant,
        body_bpw: 3.44,
        embedding_bpw: 3.4375,
        lm_head_bpw: 5.5,
        quality_cost: 11.0,
    },
    WeightQuant {
        id: "IQ3_XXS",
        family: QuantFamily::IQuant,
        body_bpw: 3.06,
        embedding_bpw: 3.0625,
        lm_head_bpw: 5.5,
        quality_cost: 15.0,
    },
    WeightQuant {
        id: "Q2_K",
        family: QuantFamily::KQuant,
        // Back-computed from published builds of a 32B and a 4B, which agree on
        // roughly 2.97 despite very different feed-forward ratios. The 3.35
        // quoted in llama.cpp's own table is a whole-file average for a model
        // with a 32k vocabulary, not a body figure.
        body_bpw: 2.97,
        embedding_bpw: 2.625,
        lm_head_bpw: 5.5,
        quality_cost: 22.0,
    },
    WeightQuant {
        id: "IQ2_XS",
        family: QuantFamily::IQuant,
        body_bpw: 2.31,
        embedding_bpw: 2.3125,
        lm_head_bpw: 5.5,
        quality_cost: 30.0,
    },
    WeightQuant {
        id: "IQ2_XXS",
        family: QuantFamily::IQuant,
        body_bpw: 2.06,
        embedding_bpw: 2.0625,
        lm_head_bpw: 5.5,
        quality_cost: 38.0,
    },
];

/// Look up a weight scheme by name, ignoring case and separators.
///
/// Accepts the spellings that appear in the wild: `q4_k_m`, `Q4-K-M`, `Q4KM`.
pub fn weight_quant(id: &str) -> Option<&'static WeightQuant> {
    let wanted = normalise(id);
    WEIGHT_QUANTS
        .iter()
        .find(|scheme| normalise(scheme.id) == wanted)
}

/// Lowercase and strip separators so lookups are forgiving.
fn normalise(id: &str) -> String {
    id.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Quantization of the attention cache.
///
/// A separate axis from weight quantization, and an underused one: dropping the
/// cache to `Q8_0` costs almost no quality and nearly halves the memory that
/// long contexts demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvQuant {
    /// 32-bit float. Rarely worth it.
    F32,
    /// 16-bit float. The default in llama.cpp and vLLM.
    #[default]
    F16,
    /// 8-bit blocks of 32 with an fp16 scale: 8.5 bits per element.
    Q8_0,
    /// 5-bit blocks of 32 with scale and minimum: 6 bits per element.
    Q5_1,
    /// 5-bit blocks of 32 with an fp16 scale: 5.5 bits per element.
    Q5_0,
    /// 4-bit blocks of 32 with scale and minimum: 5 bits per element.
    Q4_1,
    /// 4-bit blocks of 32 with an fp16 scale: 4.5 bits per element.
    Q4_0,
}

impl KvQuant {
    /// Bytes per cached scalar, including block scale overhead.
    pub const fn bytes_per_elem(self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::F16 => 2.0,
            // 32 int8 values plus one fp16 scale = 34 bytes per 32 elements.
            Self::Q8_0 => 1.0625,
            // 16 packed bytes plus fp16 scale and minimum = 20 per 32.
            Self::Q5_1 => 0.75,
            // 20 packed bytes plus an fp16 scale = 22 per 32.
            Self::Q5_0 => 0.6875,
            Self::Q4_1 => 0.625,
            // 16 packed bytes plus an fp16 scale = 18 per 32.
            Self::Q4_0 => 0.5625,
        }
    }

    /// Canonical name as passed to llama.cpp's `--cache-type-k`.
    pub const fn id(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q5_1 => "q5_1",
            Self::Q5_0 => "q5_0",
            Self::Q4_1 => "q4_1",
            Self::Q4_0 => "q4_0",
        }
    }

    /// Quality lost relative to an fp16 cache, in the same normalised points
    /// used by [`WeightQuant::quality_cost`].
    ///
    /// Cache quantization is far more forgiving than weight quantization, which
    /// is the point: `Q8_0` is close to free and halves long-context memory.
    pub const fn quality_cost(self) -> f64 {
        match self {
            Self::F32 | Self::F16 => 0.0,
            Self::Q8_0 => 0.2,
            Self::Q5_1 => 1.4,
            Self::Q5_0 => 1.8,
            Self::Q4_1 => 3.0,
            Self::Q4_0 => 4.0,
        }
    }

    /// The options a fit solver may try, from highest fidelity down.
    pub const LADDER: &'static [Self] = &[Self::F16, Self::Q8_0, Self::Q5_1, Self::Q4_0];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{AttentionKind, FfnKind, LayerLayout, LayerSpec};

    fn llama_3_1_8b() -> Architecture {
        Architecture {
            hidden_size: 4096,
            heads: 32,
            intermediate_size: 14336,
            vocab_size: 128_256,
            layers: LayerLayout::uniform(
                32,
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
            max_context: 131_072,
        }
    }

    /// Gemma 3 1B: 1B parameters, 262k vocabulary, tied embeddings. The case
    /// that breaks flat bits-per-weight estimates.
    fn gemma_3_1b() -> Architecture {
        let local = LayerSpec::windowed(
            AttentionKind::Grouped {
                kv_heads: 1,
                head_dim: 256,
            },
            512,
        );
        let global = LayerSpec::global(AttentionKind::Grouped {
            kv_heads: 1,
            head_dim: 256,
        });
        Architecture {
            hidden_size: 1152,
            heads: 4,
            intermediate_size: 6912,
            vocab_size: 262_144,
            layers: LayerLayout::cycling(26, vec![local, local, local, local, local, global]),
            ffn: FfnKind::Gated,
            tied_embeddings: true,
            moe: None,
            norms_per_layer: 4,
            softcapped_attention: false,
            native_expert_quant: None,
            max_context: 32_768,
        }
    }

    #[test]
    fn lookup_tolerates_the_spellings_people_actually_type() {
        for spelling in ["Q4_K_M", "q4_k_m", "Q4-K-M", "q4km"] {
            assert_eq!(
                weight_quant(spelling).map(|q| q.id),
                Some("Q4_K_M"),
                "failed on {spelling}"
            );
        }
        assert!(weight_quant("Q9_K_XL").is_none());
    }

    #[test]
    fn q4_k_m_reproduces_the_published_file_size_for_llama_8b() {
        let arch = llama_3_1_8b();
        let quant = weight_quant("Q4_K_M").expect("scheme exists");
        let gb = quant.weight_bytes(&arch) as f64 / 1e9;
        // Published GGUF builds of Llama 3.1 8B Q4_K_M are 4.92 GB.
        assert!(
            (gb - 4.92).abs() < 0.15,
            "estimated {gb:.2} GB against a published 4.92 GB"
        );
    }

    #[test]
    fn q8_0_reproduces_the_published_file_size_for_llama_8b() {
        let arch = llama_3_1_8b();
        let quant = weight_quant("Q8_0").expect("scheme exists");
        let gb = quant.weight_bytes(&arch) as f64 / 1e9;
        // Published Q8_0 builds are 8.54 GB.
        assert!(
            (gb - 8.54).abs() < 0.2,
            "estimated {gb:.2} GB against a published 8.54 GB"
        );
    }

    #[test]
    fn a_large_vocabulary_pulls_effective_bits_per_weight_above_the_nominal() {
        let quant = weight_quant("Q4_K_M").expect("scheme exists");
        let big_model = quant.effective_bpw(&llama_3_1_8b());
        let small_model = quant.effective_bpw(&gemma_3_1b());
        // Gemma 3 1B holds a quarter of its parameters in the embedding table,
        // so the whole-file average sits well above the body's 4.82.
        assert!(
            small_model > big_model,
            "Gemma 1B effective {small_model:.2} bpw, Llama 8B {big_model:.2} bpw"
        );
        assert!(
            small_model > 4.5,
            "expected the vocabulary to dominate, got {small_model:.2} bpw"
        );
    }

    #[test]
    fn a_tied_vocabulary_tensor_is_stored_at_the_output_precision() {
        let arch = gemma_3_1b();
        let quant = weight_quant("Q4_K_M").expect("scheme exists");
        let params = arch.params();
        let body = bytes_at(params.body(), quant.body_bpw);

        assert_eq!(
            quant.weight_bytes(&arch),
            body + bytes_at(params.embedding, quant.lm_head_bpw)
        );
        // Charging the shared tensor at the embedding precision is the mistake
        // this test exists to prevent: it underestimates a tied 4B by ~8%.
        assert!(body + bytes_at(params.embedding, quant.embedding_bpw) < quant.weight_bytes(&arch));
    }

    #[test]
    fn the_ladder_descends_in_size() {
        // The solver walks this list to find the best format that fits, so the
        // order has to be strictly decreasing in bytes.
        let arch = llama_3_1_8b();
        let mut previous = u64::MAX;
        for quant in WEIGHT_QUANTS {
            let bytes = quant.weight_bytes(&arch);
            assert!(
                bytes < previous,
                "{} is not smaller than the scheme above it",
                quant.id
            );
            previous = bytes;
        }
    }

    #[test]
    fn quality_cost_rises_down_each_family() {
        // Not across the whole ladder: an I-quant beating the K-quant just
        // above it on quality, at a smaller size, is the point of the format.
        for family in [QuantFamily::KQuant, QuantFamily::IQuant] {
            let mut previous = -1.0;
            for quant in WEIGHT_QUANTS.iter().filter(|q| q.family == family) {
                assert!(
                    quant.quality_cost > previous,
                    "{} does not cost more than the scheme above it in its family",
                    quant.id
                );
                previous = quant.quality_cost;
            }
        }
        let top = WEIGHT_QUANTS.first().expect("ladder is not empty");
        let bottom = WEIGHT_QUANTS.last().expect("ladder is not empty");
        assert!(bottom.quality_cost > top.quality_cost + 30.0);
    }

    #[test]
    fn an_eight_bit_cache_costs_a_little_over_half_an_fp16_one() {
        let ratio = KvQuant::Q8_0.bytes_per_elem() / KvQuant::F16.bytes_per_elem();
        assert!((ratio - 0.531).abs() < 0.01, "ratio was {ratio:.3}");
        assert!(KvQuant::Q8_0.quality_cost() < KvQuant::Q4_0.quality_cost());
    }
}
