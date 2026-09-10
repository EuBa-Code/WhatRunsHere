//! How fast the thing will actually go.
//!
//! Two regimes, modelled separately because they are bound by different
//! resources:
//!
//! - **Decode** is memory-bound. Every generated token reads the active weights
//!   once *and the whole attention cache once*. That second term is the one
//!   usually left out, and it is why throughput sags as a conversation grows:
//!   at 32k context on an 8B model the cache is comparable in size to the
//!   weights, so the same hardware runs at roughly half the speed it showed on
//!   an empty prompt. A model that ignores it reports the empty-prompt number
//!   and quietly overpromises.
//!
//! - **Prefill** is compute-bound. It costs roughly `2 * active_params` FLOPs
//!   per prompt token plus an attention term quadratic in prompt length, and
//!   memory bandwidth says nothing useful about it.
//!
//! Both are driven by a [`Calibration`], which prefers a measurement taken on
//! the actual machine to a number copied off a datasheet, and always records
//! which of the two it used.

use crate::arch::Architecture;
use crate::memory::kv_cache_bytes;
use crate::quant::{KvQuant, WeightQuant};
use serde::{Deserialize, Serialize};

/// Fraction of a streaming benchmark's bandwidth that an inference kernel
/// achieves. Inference reads are less regular than a copy loop.
const MEASURED_KERNEL_EFFICIENCY: f64 = 0.85;

/// Fraction of a datasheet's peak bandwidth an inference kernel achieves.
///
/// Measured, not assumed: fitting the decode model to 136 community
/// measurements across five machines gives 72% on average, ranging from 38% on
/// a 2016 Pascal card to 86% on an M4 Pro. See the throughput validation test.
const VENDOR_BANDWIDTH_EFFICIENCY: f64 = 0.72;

/// Fraction of streaming bandwidth a scattered expert gather achieves.
///
/// A dense model reads its weights as one long sequential stream. A sparse one
/// reads eight experts out of a hundred and twenty-eight, scattered across the
/// allocation, and a gather does not reach streaming speed. Counting only the
/// active parameters therefore overstates sparse throughput.
///
/// The figure comes from holding sparse models out of the fit and comparing
/// them against the line the dense ones defined on the same machines: they
/// reach 79% of the speed their active parameters predict.
const SPARSE_GATHER_EFFICIENCY: f64 = 0.79;

/// Fraction of a datasheet's peak FLOPs a prefill kernel achieves.
const VENDOR_COMPUTE_EFFICIENCY: f64 = 0.65;

/// Where a performance number came from.
///
/// Kept alongside every estimate because "measured on this machine" and
/// "derived from a spec sheet" are both reported in tokens per second, and the
/// difference between them is the difference between a fact and a guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// Observed by running this exact model on this exact machine.
    MeasuredHere,
    /// Derived from an on-device hardware probe: the silicon was measured, the
    /// model's behaviour on it was not.
    Calibrated,
    /// Contributed by someone whose hardware matches this machine.
    MeasuredElsewhere,
    /// Derived from published specifications.
    VendorSpec,
    /// A per-backend constant, because nothing better was available.
    Fallback,
}

impl Confidence {
    /// A short label for display.
    pub const fn label(self) -> &'static str {
        match self {
            Self::MeasuredHere => "measured here",
            Self::Calibrated => "calibrated",
            Self::MeasuredElsewhere => "measured on matching hardware",
            Self::VendorSpec => "from specifications",
            Self::Fallback => "rough estimate",
        }
    }

    /// Whether a real measurement stands behind this number.
    pub const fn is_measured(self) -> bool {
        matches!(self, Self::MeasuredHere | Self::MeasuredElsewhere)
    }
}

/// What one compute device can sustain.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeviceThroughput {
    /// Achievable memory bandwidth in bytes per second.
    pub bandwidth_bytes_per_s: f64,
    /// Achievable half-precision throughput in FLOPs per second, when known.
    ///
    /// `None` rather than zero when unknown: zero would read as
    /// "immeasurably slow" where the truth is "not established".
    pub compute_flops: Option<f64>,
}

impl DeviceThroughput {
    /// A device described by a datasheet, discounted to what kernels reach.
    pub fn from_vendor_spec(peak_bandwidth_gbps: f64, peak_tflops_fp16: Option<f64>) -> Self {
        Self {
            bandwidth_bytes_per_s: peak_bandwidth_gbps * 1e9 * VENDOR_BANDWIDTH_EFFICIENCY,
            compute_flops: peak_tflops_fp16.map(|t| t * 1e12 * VENDOR_COMPUTE_EFFICIENCY),
        }
    }

    /// A device measured by the on-device probe, discounted to what an
    /// inference kernel reaches relative to a streaming benchmark.
    pub fn from_probe(measured_bandwidth_bytes_per_s: f64, measured_flops: Option<f64>) -> Self {
        Self {
            bandwidth_bytes_per_s: measured_bandwidth_bytes_per_s * MEASURED_KERNEL_EFFICIENCY,
            compute_flops: measured_flops,
        }
    }
}

/// The machine's measured or assumed capabilities, and where they came from.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// The accelerator, if there is one.
    pub accelerator: Option<DeviceThroughput>,
    /// System memory, which bounds anything running on the CPU.
    pub host: DeviceThroughput,
    /// Fixed per-token cost: kernel launches, sampling, framework dispatch.
    pub overhead_ms_per_token: f64,
    /// How the numbers above were obtained.
    pub source: Confidence,
}

impl Calibration {
    /// The bandwidth that applies to whichever device holds a given share of
    /// the model.
    const fn device(&self, on_accelerator: bool) -> Option<DeviceThroughput> {
        if on_accelerator {
            self.accelerator
        } else {
            Some(self.host)
        }
    }
}

/// How a model's bytes are divided between accelerator and host.
///
/// A partial offload is not a small penalty. The devices work in sequence, one
/// layer after another, so the times add: put a tenth of a model in system RAM
/// and that tenth can cost more time than the nine tenths on the GPU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrafficSplit {
    /// Weight bytes read per token from accelerator memory.
    pub accelerator_weights: u64,
    /// Weight bytes read per token from system memory.
    pub host_weights: u64,
    /// Cache bytes read per token from accelerator memory.
    pub accelerator_cache: u64,
    /// Cache bytes read per token from system memory.
    pub host_cache: u64,
}

impl TrafficSplit {
    /// Everything on the accelerator.
    pub const fn all_accelerator(weights: u64, cache: u64) -> Self {
        Self {
            accelerator_weights: weights,
            host_weights: 0,
            accelerator_cache: cache,
            host_cache: 0,
        }
    }

    /// Everything in system memory.
    pub const fn all_host(weights: u64, cache: u64) -> Self {
        Self {
            accelerator_weights: 0,
            host_weights: weights,
            accelerator_cache: 0,
            host_cache: cache,
        }
    }

    /// A layer-wise split with `gpu_layers` of `total_layers` offloaded, the
    /// cache following its layers.
    pub fn by_layers(weights: u64, cache: u64, gpu_layers: u32, total_layers: u32) -> Self {
        if total_layers == 0 {
            return Self::all_host(weights, cache);
        }
        let fraction = f64::from(gpu_layers.min(total_layers)) / f64::from(total_layers);
        let gpu_weights = ((weights as f64) * fraction).round() as u64;
        let gpu_cache = ((cache as f64) * fraction).round() as u64;
        Self {
            accelerator_weights: gpu_weights,
            host_weights: weights.saturating_sub(gpu_weights),
            accelerator_cache: gpu_cache,
            host_cache: cache.saturating_sub(gpu_cache),
        }
    }

    /// Total bytes read per token, wherever they live.
    pub const fn total(&self) -> u64 {
        self.accelerator_weights + self.host_weights + self.accelerator_cache + self.host_cache
    }
}

/// Every input behind an estimate, so the number can be checked rather than
/// trusted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EstimateBasis {
    /// Weight bytes read per generated token.
    pub weight_traffic_bytes: u64,
    /// Cache bytes read per generated token at the evaluated context.
    pub cache_traffic_bytes: u64,
    /// Context length the estimate was made at.
    pub context: u32,
    /// Effective bandwidth applied to accelerator-resident bytes.
    pub accelerator_bandwidth_bytes_per_s: Option<f64>,
    /// Effective bandwidth applied to host-resident bytes.
    pub host_bandwidth_bytes_per_s: f64,
    /// Decode-cost multiplier for the weight format.
    pub format_decode_efficiency: f64,
    /// Fixed per-token cost included in the result.
    pub overhead_ms_per_token: f64,
}

/// An estimated or measured generation speed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Throughput {
    /// Generated tokens per second at the evaluated context.
    pub decode_tps: f64,
    /// Prompt tokens processed per second, when compute throughput is known.
    pub prefill_tps: Option<f64>,
    /// How trustworthy the numbers are.
    pub confidence: Confidence,
    /// The inputs that produced them.
    pub basis: EstimateBasis,
}

impl Throughput {
    /// Time to first token for a prompt of `prompt_tokens`, in milliseconds.
    ///
    /// `None` when compute throughput is unknown, which is deliberately
    /// different from zero.
    pub fn ttft_ms(&self, prompt_tokens: u32) -> Option<f64> {
        self.prefill_tps
            .map(|tps| f64::from(prompt_tokens) / tps * 1000.0)
    }

    /// Wall-clock seconds to answer a prompt of `prompt_tokens` with
    /// `output_tokens` of output.
    pub fn response_seconds(&self, prompt_tokens: u32, output_tokens: u32) -> Option<f64> {
        let prefill = f64::from(prompt_tokens) / self.prefill_tps?;
        Some(prefill + f64::from(output_tokens) / self.decode_tps)
    }
}

/// Bytes of weights read per generated token, before any placement.
pub fn weight_traffic_bytes(arch: &Architecture, quant: &WeightQuant) -> u64 {
    quant.decode_traffic_bytes(arch)
}

/// How much of streaming bandwidth this architecture's read pattern achieves.
///
/// One for a dense model, which reads sequentially. Below one for a sparse one,
/// in proportion to how much of its per-token traffic is gathered expert
/// weights rather than streamed ones.
pub fn read_pattern_efficiency(arch: &Architecture) -> f64 {
    if !arch.is_sparse() {
        return 1.0;
    }
    let traffic = arch.decode_traffic();
    let total = traffic.total() as f64;
    if total <= 0.0 {
        return 1.0;
    }
    let gathered = traffic.expert_ffn as f64 / total;
    // Only the gathered share pays the penalty; the attention and output
    // tensors are still read in sequence.
    1.0 / (1.0 + gathered * (1.0 / SPARSE_GATHER_EFFICIENCY - 1.0))
}

/// Estimate decode and prefill speed.
///
/// `split` says where the bytes live; the two devices are charged separately
/// and their times add, because that is how a partial offload actually runs.
pub fn throughput(
    arch: &Architecture,
    quant: &WeightQuant,
    calibration: &Calibration,
    split: &TrafficSplit,
    context: u32,
) -> Throughput {
    let format_efficiency = quant.family.decode_efficiency() * read_pattern_efficiency(arch);

    let accelerator_bw = calibration
        .accelerator
        .map(|d| d.bandwidth_bytes_per_s * format_efficiency);
    let host_bw = calibration.host.bandwidth_bytes_per_s * format_efficiency;

    let mut seconds = calibration.overhead_ms_per_token / 1000.0;

    let accelerator_bytes = split.accelerator_weights + split.accelerator_cache;
    if accelerator_bytes > 0 {
        // Without an accelerator the bytes cannot be there; charge them to the
        // host rather than silently dropping them.
        let bandwidth = accelerator_bw.unwrap_or(host_bw);
        seconds += accelerator_bytes as f64 / bandwidth;
    }
    let host_bytes = split.host_weights + split.host_cache;
    if host_bytes > 0 {
        seconds += host_bytes as f64 / host_bw;
    }

    let decode_tps = if seconds > 0.0 { 1.0 / seconds } else { 0.0 };

    // Prefill runs wherever the weights are; when they are split, the slower
    // device sets the pace for its share, so charge the same way.
    let compute = calibration
        .device(split.accelerator_weights >= split.host_weights)
        .and_then(|d| d.compute_flops);
    let prefill_tps = compute.map(|flops| flops / prefill_flops_per_token(arch, context));

    Throughput {
        decode_tps,
        prefill_tps,
        confidence: calibration.source,
        basis: EstimateBasis {
            weight_traffic_bytes: split.accelerator_weights + split.host_weights,
            cache_traffic_bytes: split.accelerator_cache + split.host_cache,
            context,
            accelerator_bandwidth_bytes_per_s: accelerator_bw,
            host_bandwidth_bytes_per_s: host_bw,
            format_decode_efficiency: format_efficiency,
            overhead_ms_per_token: calibration.overhead_ms_per_token,
        },
    }
}

/// FLOPs to process one prompt token in a prompt of length `context`.
///
/// The matrix multiplications cost `2 * active_params` each. Attention adds a
/// term that grows with prompt length: negligible at 2k, more than a third of
/// the work at 32k.
pub fn prefill_flops_per_token(arch: &Architecture, context: u32) -> f64 {
    let matmul = 2.0 * arch.active_params() as f64;

    // Averaged over a causal pass, each position attends to half the prompt.
    let context = f64::from(context.max(1));
    let per_layer_head_dims: u64 = arch
        .layers
        .iter()
        .map(|layer| match layer.attention {
            crate::arch::AttentionKind::Grouped { head_dim, .. } => {
                u64::from(arch.heads) * u64::from(head_dim)
            }
            crate::arch::AttentionKind::Latent {
                qk_nope_head_dim,
                qk_rope_head_dim,
                ..
            } => u64::from(arch.heads) * u64::from(qk_nope_head_dim + qk_rope_head_dim),
            crate::arch::AttentionKind::Recurrent { .. } => 0,
        })
        .sum();
    // Scores and the value-weighted sum: two passes over the attended span.
    let attention = 2.0 * context * per_layer_head_dims as f64;

    matmul + attention
}

/// A point on the throughput-versus-context curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeedPoint {
    /// Context length at this sample.
    pub context: u32,
    /// Generated tokens per second there.
    pub decode_tps: f64,
}

/// Sample decode speed across context lengths.
///
/// The shape of this curve is the honest answer to "how fast is it": a single
/// number taken at an empty prompt describes a situation nobody is in.
pub fn decode_curve(
    arch: &Architecture,
    quant: &WeightQuant,
    calibration: &Calibration,
    kv_quant: KvQuant,
    on_accelerator: bool,
    max_context: u32,
) -> Vec<SpeedPoint> {
    let weights = weight_traffic_bytes(arch, quant);
    let ceiling = max_context.min(arch.max_context).max(1024);
    let mut points = Vec::new();
    let mut context = 1024u32;
    loop {
        let sample = context.min(ceiling);
        let cache = kv_cache_bytes(arch, sample, kv_quant);
        let split = if on_accelerator {
            TrafficSplit::all_accelerator(weights, cache)
        } else {
            TrafficSplit::all_host(weights, cache)
        };
        points.push(SpeedPoint {
            context: sample,
            decode_tps: throughput(arch, quant, calibration, &split, sample).decode_tps,
        });
        if sample >= ceiling {
            break;
        }
        context = context.saturating_mul(2);
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{AttentionKind, FfnKind, LayerLayout, LayerSpec};
    use crate::quant::weight_quant;

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

    fn q4_k_m() -> WeightQuant {
        *weight_quant("Q4_K_M").expect("scheme exists")
    }

    /// An RTX 4090: 1008 GB/s and 165 TFLOPs of fp16 with fp32 accumulate.
    fn rtx_4090() -> Calibration {
        Calibration {
            accelerator: Some(DeviceThroughput::from_vendor_spec(1008.0, Some(165.0))),
            host: DeviceThroughput::from_vendor_spec(80.0, None),
            overhead_ms_per_token: 0.2,
            source: Confidence::VendorSpec,
        }
    }

    #[test]
    fn an_8b_on_a_4090_lands_where_llama_cpp_lands() {
        let arch = llama_3_1_8b();
        let quant = q4_k_m();
        let weights = weight_traffic_bytes(&arch, &quant);
        let split = TrafficSplit::all_accelerator(weights, 0);
        let tps = throughput(&arch, &quant, &rtx_4090(), &split, 0).decode_tps;
        // Published llama.cpp figures for this pairing sit around 130-160 tok/s.
        assert!(
            (110.0..185.0).contains(&tps),
            "estimated {tps:.0} tok/s, expected roughly 130-160"
        );
    }

    #[test]
    fn throughput_decays_as_the_conversation_grows() {
        let arch = llama_3_1_8b();
        let quant = q4_k_m();
        let curve = decode_curve(&arch, &quant, &rtx_4090(), KvQuant::F16, true, 131_072);
        assert!(curve.len() > 5);
        for pair in curve.windows(2) {
            assert!(
                pair[1].decode_tps < pair[0].decode_tps,
                "speed did not fall from {} to {} tokens of context",
                pair[0].context,
                pair[1].context
            );
        }
        let first = curve.first().expect("non-empty").decode_tps;
        let last = curve.last().expect("non-empty").decode_tps;
        // At 128k the cache outweighs the weights, so speed must be well under
        // half the empty-prompt figure.
        assert!(
            last < first * 0.5,
            "128k ran at {last:.0} tok/s against {first:.0} on an empty prompt"
        );
    }

    #[test]
    fn a_quantized_cache_recovers_long_context_speed() {
        let arch = llama_3_1_8b();
        let quant = q4_k_m();
        let calibration = rtx_4090();
        let weights = weight_traffic_bytes(&arch, &quant);
        let at = |kv| {
            let cache = kv_cache_bytes(&arch, 65_536, kv);
            throughput(
                &arch,
                &quant,
                &calibration,
                &TrafficSplit::all_accelerator(weights, cache),
                65_536,
            )
            .decode_tps
        };
        assert!(at(KvQuant::Q8_0) > at(KvQuant::F16) * 1.2);
    }

    #[test]
    fn spilling_a_tenth_of_the_model_to_ram_costs_far_more_than_a_tenth() {
        let arch = llama_3_1_8b();
        let quant = q4_k_m();
        let calibration = rtx_4090();
        let weights = weight_traffic_bytes(&arch, &quant);

        let full = throughput(
            &arch,
            &quant,
            &calibration,
            &TrafficSplit::all_accelerator(weights, 0),
            0,
        )
        .decode_tps;
        let spilled = throughput(
            &arch,
            &quant,
            &calibration,
            &TrafficSplit::by_layers(weights, 0, 29, 32),
            0,
        )
        .decode_tps;

        let retained = spilled / full;
        assert!(
            retained < 0.5,
            "keeping 90% on the GPU retained {:.0}% of the speed; the point is \
             that it does not",
            retained * 100.0
        );
    }

    #[test]
    fn prefill_attention_becomes_significant_at_long_prompts() {
        let arch = llama_3_1_8b();
        let short = prefill_flops_per_token(&arch, 2_048);
        let long = prefill_flops_per_token(&arch, 32_768);
        assert!(long > short * 1.25, "attention barely registered");
    }

    #[test]
    fn time_to_first_token_is_absent_rather_than_zero_without_compute_data() {
        let arch = llama_3_1_8b();
        let quant = q4_k_m();
        let calibration = Calibration {
            accelerator: Some(DeviceThroughput::from_vendor_spec(1008.0, None)),
            ..rtx_4090()
        };
        let weights = weight_traffic_bytes(&arch, &quant);
        let result = throughput(
            &arch,
            &quant,
            &calibration,
            &TrafficSplit::all_accelerator(weights, 0),
            0,
        );
        assert_eq!(result.prefill_tps, None);
        assert_eq!(result.ttft_ms(4096), None);
    }

    #[test]
    fn a_measured_probe_outranks_a_datasheet() {
        assert!(Confidence::MeasuredHere < Confidence::Calibrated);
        assert!(Confidence::Calibrated < Confidence::VendorSpec);
        assert!(Confidence::VendorSpec < Confidence::Fallback);
        assert!(Confidence::MeasuredHere.is_measured());
        assert!(!Confidence::Calibrated.is_measured());
    }

    #[test]
    fn the_basis_reports_every_input_that_produced_the_number() {
        let arch = llama_3_1_8b();
        let quant = q4_k_m();
        let weights = weight_traffic_bytes(&arch, &quant);
        let cache = kv_cache_bytes(&arch, 8_192, KvQuant::F16);
        let result = throughput(
            &arch,
            &quant,
            &rtx_4090(),
            &TrafficSplit::all_accelerator(weights, cache),
            8_192,
        );
        assert_eq!(result.basis.weight_traffic_bytes, weights);
        assert_eq!(result.basis.cache_traffic_bytes, cache);
        assert_eq!(result.basis.context, 8_192);
        assert!(result.basis.accelerator_bandwidth_bytes_per_s.is_some());
    }
}
