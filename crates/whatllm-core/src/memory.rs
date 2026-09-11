//! What a loaded model actually occupies.
//!
//! Weights are the part everyone estimates, and on a long context they are
//! frequently not the largest part. A full accounting has five terms:
//!
//! 1. **Weights**: measured from a real file when the catalog knows one,
//!    computed per tensor class otherwise.
//! 2. **Attention cache**: grows with context, and by wildly different slopes
//!    depending on the architecture. See [`crate::arch`].
//! 3. **Activations**: the graph working set. Dominated, when flash attention
//!    is off, by the materialised score matrix, which is quadratic in context
//!    and is the single most common cause of an out-of-memory surprise.
//! 4. **Runtime overhead**: the CUDA context and the framework's own fixed
//!    allocations, several hundred megabytes before a single weight is read.
//! 5. **Headroom**: allocator slack. A pool filled to the last byte does not
//!    load, whatever the arithmetic says.

use crate::arch::Architecture;
use crate::quant::{KvQuant, WeightQuant};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Fixed allocations and graph-shape constants for one inference runtime.
///
/// The constants are calibrated against the compute-buffer sizes these runtimes
/// report on load, and are exposed rather than buried so a measurement can
/// correct them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeProfile {
    /// Runtime name, as shown to the user.
    pub id: &'static str,
    /// Allocation made before any weight is read: the accelerator context,
    /// kernel images, and the framework's fixed structures.
    pub fixed_overhead_bytes: u64,
    /// How many hidden-sized tensors of the graph are live at peak.
    ///
    /// Calibrated against reported compute-buffer sizes; a runtime that reuses
    /// buffers aggressively sits lower.
    pub live_graph_tensors: f64,
    /// Whether the runtime materialises the attention score matrix.
    ///
    /// With flash attention the scores are never written to memory, which turns
    /// a term quadratic in context into a negligible one. Leaving it off is the
    /// difference between a model loading and not loading at long context.
    pub flash_attention: bool,
    /// Fraction of the memory pool to leave unallocated for fragmentation.
    pub headroom_fraction: f64,
    /// Floor on that headroom, whatever the fraction works out to.
    pub min_headroom_bytes: u64,
}

impl RuntimeProfile {
    /// llama.cpp and its descendants (Ollama, LM Studio) with flash attention
    /// enabled, which is the modern default.
    pub const LLAMA_CPP: Self = Self {
        id: "llama.cpp",
        fixed_overhead_bytes: 400 * MIB,
        live_graph_tensors: 6.0,
        flash_attention: true,
        headroom_fraction: 0.03,
        min_headroom_bytes: 256 * MIB,
    };

    /// llama.cpp without flash attention, which is still what you get on some
    /// backends and with some quantized cache combinations.
    pub const LLAMA_CPP_NO_FLASH: Self = Self {
        id: "llama.cpp (no flash attention)",
        flash_attention: false,
        ..Self::LLAMA_CPP
    };

    /// vLLM, which preallocates aggressively and keeps CUDA graphs resident.
    pub const VLLM: Self = Self {
        id: "vLLM",
        fixed_overhead_bytes: 1200 * MIB,
        live_graph_tensors: 8.0,
        flash_attention: true,
        headroom_fraction: 0.05,
        min_headroom_bytes: 512 * MIB,
    };

    /// MLX on Apple Silicon, where there is no separate device context to pay
    /// for and allocation is unified.
    pub const MLX: Self = Self {
        id: "MLX",
        fixed_overhead_bytes: 200 * MIB,
        live_graph_tensors: 5.0,
        flash_attention: true,
        headroom_fraction: 0.04,
        min_headroom_bytes: 384 * MIB,
    };

    /// Every runtime `WhatLLM` models, for enumeration in a UI.
    pub const ALL: &'static [Self] = &[
        Self::LLAMA_CPP,
        Self::LLAMA_CPP_NO_FLASH,
        Self::VLLM,
        Self::MLX,
    ];
}

impl Default for RuntimeProfile {
    fn default() -> Self {
        Self::LLAMA_CPP
    }
}

/// Look up a runtime profile by name.
pub fn runtime_profile(id: &str) -> Option<&'static RuntimeProfile> {
    RuntimeProfile::ALL.iter().find(|profile| profile.id == id)
}

/// Like [`WeightQuant`], a profile travels as its name and is resolved against
/// the table on the way back.
impl Serialize for RuntimeProfile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.id)
    }
}

impl<'de> Deserialize<'de> for RuntimeProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let id = String::deserialize(deserializer)?;
        runtime_profile(&id)
            .copied()
            .ok_or_else(|| serde::de::Error::custom(format!("unknown runtime `{id}`")))
    }
}

/// One mebibyte, for writing the constants above legibly.
const MIB: u64 = 1024 * 1024;

/// How a model is being asked to run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoadConfig {
    /// Context length in tokens, per sequence.
    pub context: u32,
    /// Sequences served concurrently. Each one needs its own attention cache,
    /// which is what makes multi-user serving expensive.
    pub parallel: u32,
    /// Prompt-processing micro-batch. Sets the width of the graph working set.
    pub ubatch: u32,
    /// Weight quantization.
    pub weight_quant: WeightQuant,
    /// Attention cache quantization.
    pub kv_quant: KvQuant,
    /// Measured file size in bytes, when one is known. Always preferred over
    /// the computed estimate.
    pub measured_weight_bytes: Option<u64>,
    /// Runtime constants.
    pub runtime: RuntimeProfile,
}

impl LoadConfig {
    /// A single-user configuration at the given context, on llama.cpp defaults.
    pub fn single_user(context: u32, weight_quant: WeightQuant) -> Self {
        Self {
            context,
            parallel: 1,
            ubatch: 512,
            weight_quant,
            kv_quant: KvQuant::F16,
            measured_weight_bytes: None,
            runtime: RuntimeProfile::LLAMA_CPP,
        }
    }

    /// The same configuration with a measured weight size substituted in.
    #[must_use]
    pub fn with_measured_weights(mut self, bytes: u64) -> Self {
        self.measured_weight_bytes = Some(bytes);
        self
    }

    /// The same configuration at a different context length.
    #[must_use]
    pub fn at_context(mut self, context: u32) -> Self {
        self.context = context;
        self
    }

    /// The same configuration serving `parallel` concurrent sequences.
    #[must_use]
    pub fn serving(mut self, parallel: u32) -> Self {
        self.parallel = parallel.max(1);
        self
    }

    /// The same configuration with a quantized attention cache.
    #[must_use]
    pub fn with_kv_quant(mut self, kv: KvQuant) -> Self {
        self.kv_quant = kv;
        self
    }
}

/// Where the graph working set goes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationBreakdown {
    /// Logits over the vocabulary. Small, except for very large vocabularies.
    pub logits: u64,
    /// Hidden-state and feed-forward intermediates held live in the graph.
    pub graph: u64,
    /// The attention score matrix.
    ///
    /// Zero under flash attention. Otherwise `ubatch * context * heads * 4`,
    /// which reaches gigabytes at long context and catches people out.
    pub attention_scores: u64,
}

impl ActivationBreakdown {
    /// Total activation memory.
    pub const fn total(&self) -> u64 {
        self.logits + self.graph + self.attention_scores
    }
}

/// A full accounting of one model's footprint under one configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlan {
    /// Model weights.
    pub weights: u64,
    /// True when `weights` came from a real file rather than an estimate.
    pub weights_measured: bool,
    /// Attention cache across every concurrent sequence.
    pub kv_cache: u64,
    /// Graph working set.
    pub activations: ActivationBreakdown,
    /// Runtime fixed overhead.
    pub runtime_overhead: u64,
    /// Allocator slack deliberately left free.
    pub headroom: u64,
}

impl MemoryPlan {
    /// Everything that must be resident, excluding headroom.
    pub const fn resident(&self) -> u64 {
        self.weights + self.kv_cache + self.activations.total() + self.runtime_overhead
    }

    /// What the pool must supply for this configuration to load and stay up.
    pub const fn required(&self) -> u64 {
        self.resident() + self.headroom
    }

    /// Whether this configuration fits in a pool of `available` bytes.
    pub const fn fits_in(&self, available: u64) -> bool {
        self.required() <= available
    }

    /// Fraction of a pool of `available` bytes this configuration consumes.
    ///
    /// Returns `f64::INFINITY` for an empty pool, which sorts and compares the
    /// way a caller wants without a special case.
    pub fn utilisation(&self, available: u64) -> f64 {
        if available == 0 {
            return f64::INFINITY;
        }
        self.required() as f64 / available as f64
    }

    /// Bytes by which this configuration exceeds `available`, if it does.
    pub const fn overflow(&self, available: u64) -> u64 {
        self.required().saturating_sub(available)
    }
}

/// Bytes of attention cache for one sequence at the given context.
///
/// Two parts, priced differently. The key/value cache of the attention layers
/// is stored in the format the runtime was asked for, and that is where a
/// compressed cache saves its memory. The recurrent state of a hybrid model's
/// linear-attention layers is rewritten every token and kept at full
/// precision whatever the cache format: llama.cpp holds it as f32, and
/// nothing quantizes it.
pub fn kv_cache_bytes(arch: &Architecture, context: u32, kv_quant: KvQuant) -> u64 {
    let state = arch.constant_state_elems();
    let cached = arch.cache_elems(context).saturating_sub(state);
    ((cached as f64) * kv_quant.bytes_per_elem()).round() as u64 + state * 4
}

/// Bytes of attention cache added by one more token, at the given context.
///
/// Falls in steps rather than staying constant, because windowed layers stop
/// contributing once saturated.
pub fn marginal_kv_bytes(arch: &Architecture, context: u32, kv_quant: KvQuant) -> u64 {
    let elems = arch.marginal_cache_elems(context);
    ((elems as f64) * kv_quant.bytes_per_elem()).round() as u64
}

/// The width of the feed-forward intermediate that is live during a forward
/// pass: the routed experts' width for a sparse model, the dense width
/// otherwise.
fn live_intermediate(arch: &Architecture) -> u64 {
    match &arch.moe {
        Some(moe) => u64::from(moe.experts_per_token) * u64::from(moe.expert_intermediate),
        None => u64::from(arch.intermediate_size),
    }
}

/// Compute the full footprint of `arch` under `cfg`.
pub fn plan(arch: &Architecture, cfg: &LoadConfig) -> MemoryPlan {
    let weights = cfg
        .measured_weight_bytes
        .unwrap_or_else(|| cfg.weight_quant.weight_bytes(arch));

    let parallel = u64::from(cfg.parallel.max(1));
    let kv_cache = kv_cache_bytes(arch, cfg.context, cfg.kv_quant) * parallel;

    let ubatch = u64::from(cfg.ubatch.max(1));
    let hidden = u64::from(arch.hidden_size);
    let heads = u64::from(arch.heads);

    // Logits are produced for one position per sequence during decode.
    let logits = u64::from(arch.vocab_size) * parallel * 4;

    // The live working set: residual-stream tensors plus the feed-forward
    // intermediate, at fp32 accumulation width.
    let graph = ((ubatch * (2 * hidden + live_intermediate(arch)) * 4) as f64
        * cfg.runtime.live_graph_tensors
        / 4.0)
        .round() as u64;

    // Without flash attention the full score matrix is written out: one fp32
    // value per (batch position, cached position, head).
    // A soft-capped model cannot use flash attention whatever the runtime is
    // configured to do, because the kernels have nowhere to apply the cap.
    let flash = cfg.runtime.flash_attention && !arch.softcapped_attention;
    let attention_scores = if flash {
        0
    } else {
        ubatch * u64::from(cfg.context) * heads * 4
    };

    let activations = ActivationBreakdown {
        logits,
        graph,
        attention_scores,
    };

    let resident = weights + kv_cache + activations.total() + cfg.runtime.fixed_overhead_bytes;
    let headroom = (((resident as f64) * cfg.runtime.headroom_fraction).round() as u64)
        .max(cfg.runtime.min_headroom_bytes);

    MemoryPlan {
        weights,
        weights_measured: cfg.measured_weight_bytes.is_some(),
        kv_cache,
        activations,
        runtime_overhead: cfg.runtime.fixed_overhead_bytes,
        headroom,
    }
}

/// The longest context that fits in `available` bytes, or `None` if the model
/// does not fit at even a single token.
///
/// This is what makes a context slider honest: instead of asking the user to
/// find the ceiling by trial and error, say where it is.
pub fn max_context_for(arch: &Architecture, cfg: &LoadConfig, available: u64) -> Option<u32> {
    let ceiling = arch.max_context;
    if !plan(arch, &cfg.at_context(1)).fits_in(available) {
        return None;
    }
    if plan(arch, &cfg.at_context(ceiling)).fits_in(available) {
        return Some(ceiling);
    }

    // Footprint is monotonic in context, so bisect.
    let (mut lo, mut hi) = (1u32, ceiling);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if plan(arch, &cfg.at_context(mid)).fits_in(available) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// The most concurrent sequences that fit in `available` bytes at the
/// configured context.
pub fn max_parallel_for(arch: &Architecture, cfg: &LoadConfig, available: u64) -> u32 {
    let mut best = 0;
    // Sequences are cheap relative to weights, so a linear walk with an early
    // exit is both fast enough and easier to trust than another bisection.
    for n in 1..=1024 {
        if plan(arch, &cfg.serving(n)).fits_in(available) {
            best = n;
        } else {
            break;
        }
    }
    best
}

/// A point on the memory-versus-context curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    /// Context length at this sample.
    pub context: u32,
    /// The footprint there.
    pub plan: MemoryPlan,
}

/// Sample the footprint across context lengths, for plotting.
///
/// Samples on a power-of-two ladder because that is how the curve is read: the
/// interesting question is which doubling breaks the budget.
pub fn context_curve(arch: &Architecture, cfg: &LoadConfig) -> Vec<CurvePoint> {
    let mut points = Vec::new();
    let mut context = 1024u32;
    while context < arch.max_context {
        points.push(CurvePoint {
            context,
            plan: plan(arch, &cfg.at_context(context)),
        });
        context = context.saturating_mul(2);
    }
    points.push(CurvePoint {
        context: arch.max_context,
        plan: plan(arch, &cfg.at_context(arch.max_context)),
    });
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{AttentionKind, FfnKind, LayerLayout, LayerSpec};
    use crate::quant::weight_quant;

    const GIB: u64 = 1024 * 1024 * 1024;

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

    fn llama_3_3_70b() -> Architecture {
        Architecture {
            hidden_size: 8192,
            heads: 64,
            intermediate_size: 28672,
            vocab_size: 128_256,
            layers: LayerLayout::uniform(
                80,
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

    fn q4_k_m() -> WeightQuant {
        *weight_quant("Q4_K_M").expect("scheme exists")
    }

    #[test]
    fn recurrent_state_is_priced_at_full_precision_whatever_the_cache_format() {
        // A stack of nothing but recurrent layers has no per-token cache at
        // all: what it holds is the same at any context and in any format.
        let mut arch = llama_3_1_8b();
        let state = 32 * 128 * 128;
        arch.layers = LayerLayout::uniform(
            32,
            LayerSpec::global(AttentionKind::Recurrent {
                state_elems: state,
                params: 40_000_000,
            }),
        );
        let f16 = kv_cache_bytes(&arch, 65_536, KvQuant::F16);
        let q4 = kv_cache_bytes(&arch, 65_536, KvQuant::Q4_0);
        assert_eq!(f16, 32 * state * 4);
        assert_eq!(q4, f16, "a compressed cache format cannot touch the state");
        assert_eq!(kv_cache_bytes(&arch, 1, KvQuant::F16), f16);
        assert_eq!(marginal_kv_bytes(&arch, 65_536, KvQuant::F16), 0);
    }

    #[test]
    fn kv_cache_matches_the_hand_calculation() {
        let arch = llama_3_1_8b();
        // 32 layers x 2 x 8 kv heads x 128 dims x 2 bytes = 128 KiB per token.
        let per_token = kv_cache_bytes(&arch, 1, KvQuant::F16);
        assert_eq!(per_token, 32 * 2 * 8 * 128 * 2);
        assert_eq!(per_token, 131_072);

        // At 32k that is 4 GiB, which is most of the weight footprint again.
        let at_32k = kv_cache_bytes(&arch, 32_768, KvQuant::F16);
        assert_eq!(at_32k, 4 * GIB);
    }

    #[test]
    fn a_quantized_cache_buys_back_most_of_that() {
        let arch = llama_3_1_8b();
        let f16 = kv_cache_bytes(&arch, 32_768, KvQuant::F16);
        let q8 = kv_cache_bytes(&arch, 32_768, KvQuant::Q8_0);
        let saved = (f16 - q8) as f64 / 1e9;
        assert!(saved > 1.9, "expected ~2 GB saved, got {saved:.2} GB");
    }

    #[test]
    fn the_cache_can_outweigh_the_weights_at_long_context() {
        let arch = llama_3_1_8b();
        let cfg = LoadConfig::single_user(131_072, q4_k_m());
        let plan = plan(&arch, &cfg);
        assert!(
            plan.kv_cache > plan.weights,
            "cache {} vs weights {} at 128k",
            plan.kv_cache,
            plan.weights
        );
    }

    #[test]
    fn disabling_flash_attention_adds_a_term_quadratic_in_context() {
        let arch = llama_3_1_8b();
        let base = LoadConfig::single_user(32_768, q4_k_m());
        let with_flash = plan(&arch, &base);
        let without = plan(
            &arch,
            &LoadConfig {
                runtime: RuntimeProfile::LLAMA_CPP_NO_FLASH,
                ..base
            },
        );
        assert_eq!(with_flash.activations.attention_scores, 0);
        // 512 x 32768 x 32 heads x 4 bytes = 2 GiB of score matrix.
        assert_eq!(without.activations.attention_scores, 2 * GIB);
        assert!(without.required() > with_flash.required() + GIB);
    }

    #[test]
    fn a_measured_file_size_overrides_the_estimate() {
        let arch = llama_3_1_8b();
        let measured = 4_920_000_000;
        let cfg = LoadConfig::single_user(8192, q4_k_m()).with_measured_weights(measured);
        let plan = plan(&arch, &cfg);
        assert_eq!(plan.weights, measured);
        assert!(plan.weights_measured);
    }

    #[test]
    fn the_context_ceiling_is_found_exactly() {
        let arch = llama_3_1_8b();
        let cfg = LoadConfig::single_user(8192, q4_k_m());
        let available = 12 * GIB;
        let ceiling = max_context_for(&arch, &cfg, available).expect("8B fits in 12 GiB");

        assert!(plan(&arch, &cfg.at_context(ceiling)).fits_in(available));
        assert!(!plan(&arch, &cfg.at_context(ceiling + 1)).fits_in(available));
    }

    #[test]
    fn a_quantized_cache_raises_that_ceiling() {
        let arch = llama_3_1_8b();
        let available = 12 * GIB;
        let base = LoadConfig::single_user(8192, q4_k_m());
        let at_f16 = max_context_for(&arch, &base, available).expect("fits");
        let at_q8 =
            max_context_for(&arch, &base.with_kv_quant(KvQuant::Q8_0), available).expect("fits");
        assert!(
            at_q8 > at_f16,
            "q8_0 cache gave {at_q8} tokens, f16 gave {at_f16}"
        );
    }

    #[test]
    fn a_model_too_large_for_the_pool_has_no_context_ceiling() {
        let arch = llama_3_3_70b();
        let cfg = LoadConfig::single_user(8192, q4_k_m());
        assert!(max_context_for(&arch, &cfg, 8 * GIB).is_none());
    }

    #[test]
    fn concurrency_is_bounded_by_cache_not_weights() {
        let arch = llama_3_1_8b();
        let cfg = LoadConfig::single_user(8192, q4_k_m());
        let on_24gb = max_parallel_for(&arch, &cfg, 24 * GIB);
        let on_12gb = max_parallel_for(&arch, &cfg, 12 * GIB);
        assert!(on_24gb > on_12gb, "{on_24gb} vs {on_12gb}");
        assert!(on_12gb >= 1, "a single user must fit in 12 GiB");
    }

    #[test]
    fn the_curve_rises_monotonically_and_ends_at_the_ceiling() {
        let arch = llama_3_1_8b();
        let cfg = LoadConfig::single_user(4096, q4_k_m());
        let curve = context_curve(&arch, &cfg);
        assert!(curve.len() > 4);
        for pair in curve.windows(2) {
            assert!(
                pair[1].plan.required() >= pair[0].plan.required(),
                "footprint fell from {} to {}",
                pair[0].context,
                pair[1].context
            );
        }
        assert_eq!(
            curve.last().expect("non-empty").context,
            arch.max_context,
            "the curve must reach the model's own ceiling"
        );
    }

    #[test]
    fn utilisation_and_overflow_agree_with_each_other() {
        let arch = llama_3_3_70b();
        let cfg = LoadConfig::single_user(8192, q4_k_m());
        let plan = plan(&arch, &cfg);
        let pool = 24 * GIB;
        assert!(plan.utilisation(pool) > 1.0);
        assert_eq!(plan.overflow(pool), plan.required() - pool);
        assert_eq!(plan.overflow(plan.required() * 2), 0);
    }
}
