//! Validate the throughput model against measurements from real machines.
//!
//! The memory model was checked against real files. This checks the other half:
//! that the bytes-per-token figure the decode model computes actually predicts
//! the tokens per second people observe.
//!
//! The dataset does not record memory bandwidth — only the hardware name and
//! the speed somebody got — so rather than assume a bandwidth and check the
//! speed, this fits the model to the observations.
//!
//! The decode model says the time to produce one token is
//!
//! ```text
//! seconds_per_token = bytes_read_per_token / bandwidth + fixed_overhead
//! ```
//!
//! which is a straight line in `bytes` against `1 / tokens_per_second`. Fitting
//! that line across the forty different models one machine ran recovers both
//! unknowns at once: the slope is the machine's achieved bandwidth, the
//! intercept is its per-token overhead. Neither is assumed, and no datasheet is
//! needed.
//!
//! That makes the fit quality the real test. If bytes-per-token were computed
//! wrongly — mishandled mixture-of-experts sparsity, a forgotten output
//! projection, the wrong bits per weight — the points would not fall on a line,
//! because the error would vary with whatever was mishandled. A high
//! coefficient of determination across dozens of different models on one
//! machine is hard to get by accident.
//!
//! Sparse models are held out of the fit and compared against the line the
//! dense ones define, which is how the mixture-of-experts penalty is measured
//! rather than guessed.
//!
//! Data: llmfit's community benchmark submissions, MIT licensed.
//! <https://github.com/AlexsJones/llmfit>

use serde::Deserialize;
use std::collections::BTreeMap;
use whatllm_core::arch::Architecture;
use whatllm_core::memory::{self, kv_cache_bytes, LoadConfig, RuntimeProfile};
use whatllm_core::perf::weight_traffic_bytes;
use whatllm_core::quant::{weight_quant, KvQuant};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Fits with fewer points than this are reported but not used to judge the
/// model: a handful of measurements on a card where most models are borderline
/// says little either way.
const TRUSTWORTHY: usize = 10;

#[derive(Debug, Deserialize)]
struct Dataset {
    architectures: BTreeMap<String, Architecture>,
    machines: Vec<Machine>,
}

#[derive(Debug, Deserialize)]
struct Machine {
    hardware: Hardware,
    measurements: Vec<Measurement>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Hardware {
    #[serde(default, rename = "hardwareName")]
    name: String,
    #[serde(default)]
    vram_gb: f64,
    #[serde(default)]
    ram_gb: f64,
    #[serde(default)]
    unified_memory: bool,
    #[serde(default)]
    cpu: String,
}

impl Hardware {
    /// Identifies the machine, so separate submissions from the same one can be
    /// pooled. More points make a better fit, and the fit quality itself says
    /// whether pooling them was fair.
    fn fingerprint(&self) -> String {
        format!(
            "{}|{}|{:.1}|{:.1}",
            self.name, self.cpu, self.vram_gb, self.ram_gb
        )
    }

    /// Bytes a model can be held in.
    fn pool_bytes(&self) -> u64 {
        let gib = if self.unified_memory {
            self.ram_gb
        } else {
            self.vram_gb
        };
        (gib * GIB) as u64
    }
}

#[derive(Debug, Deserialize)]
struct Measurement {
    repo: String,
    name: String,
    quant: String,
    tokens_per_second: f64,
    #[serde(default)]
    output_tokens: Option<f64>,
}

/// Published memory bandwidth, in GB/s, for the parts in this dataset.
///
/// Used only to sanity-check the absolute level the fit arrives at. The fit
/// itself needs none of it, and the shipping tool measures rather than
/// consulting a table like this one.
fn datasheet_gbps(name: &str) -> Option<f64> {
    let name = name.to_ascii_lowercase();
    let table = [
        ("rx 7900 xt", 800.0),
        ("gtx 1050 ti", 112.0),
        ("rtx 2080", 448.0),
        ("rtx 5070", 672.0),
        ("m4 pro", 273.0),
        ("gb10", 273.0),
        ("rtx 4060 laptop", 272.0),
    ];
    table
        .iter()
        .find(|(needle, _)| name.contains(needle))
        .map(|(_, gbps)| *gbps)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match values.len() {
        0 => 0.0,
        n if n % 2 == 1 => values[n / 2],
        n => f64::midpoint(values[n / 2 - 1], values[n / 2]),
    }
}

/// One measurement, reduced to the two numbers the model relates.
struct Point {
    label: String,
    bytes_per_token: f64,
    seconds_per_token: f64,
    sparse: bool,
    /// Ran without flash attention because the architecture rules it out.
    softcapped: bool,
    fits: bool,
}

/// A straight-line fit of seconds-per-token against bytes-per-token.
struct Fit {
    /// Achieved bandwidth, from the slope.
    bytes_per_s: f64,
    /// Fixed per-token cost, from the intercept.
    overhead_ms: f64,
    /// How much of the variation the line accounts for.
    r_squared: f64,
    /// Points the line was fitted through.
    points: usize,
}

/// Ordinary least squares through `(bytes, seconds)`.
fn fit_line(points: &[&Point]) -> Option<Fit> {
    if points.len() < 4 {
        return None;
    }
    let n = points.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for point in points {
        let (x, y) = (point.bytes_per_token, point.seconds_per_token);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denominator = n * sxx - sx * sx;
    if denominator.abs() < f64::EPSILON {
        return None;
    }
    let slope = (n * sxy - sx * sy) / denominator;
    if slope <= 0.0 {
        return None;
    }
    let intercept = (sy - slope * sx) / n;

    let mean = sy / n;
    let mut residual = 0.0;
    let mut total = 0.0;
    for point in points {
        let predicted = slope * point.bytes_per_token + intercept;
        residual += (point.seconds_per_token - predicted).powi(2);
        total += (point.seconds_per_token - mean).powi(2);
    }

    Some(Fit {
        bytes_per_s: 1.0 / slope,
        // A negative intercept is a fitting artefact, not negative time.
        overhead_ms: (intercept * 1000.0).max(0.0),
        r_squared: if total > 0.0 {
            1.0 - residual / total
        } else {
            0.0
        },
        points: points.len(),
    })
}

fn evaluate(dataset: &Dataset, machine: &Machine) -> Vec<Point> {
    let pool = machine.hardware.pool_bytes();
    let mut points = Vec::new();
    for measurement in &machine.measurements {
        let Some(arch) = dataset.architectures.get(&measurement.repo) else {
            continue;
        };
        let Some(quant) = weight_quant(&measurement.quant) else {
            continue;
        };
        if measurement.tokens_per_second <= 0.0 {
            continue;
        }

        let weights = weight_traffic_bytes(arch, quant);
        // The prompt length was not recorded and the generation is short, so
        // charge the cache at the average position over the run.
        let context = measurement.output_tokens.unwrap_or(256.0) as u32 / 2;
        let cache = kv_cache_bytes(arch, context, KvQuant::F16);

        // A model that did not fit was partially offloaded to system RAM, which
        // is a different placement at a different speed; those measurements say
        // nothing about this machine's bandwidth. Deciding that on weights
        // alone is too generous — a 9B on an 8 GB card fits by that measure and
        // spills in reality — so ask the full memory model, and leave a margin
        // on top because a configuration at the very edge spills too.
        let footprint = memory::plan(
            arch,
            &LoadConfig {
                context: context.max(2048),
                parallel: 1,
                ubatch: 512,
                weight_quant: *quant,
                kv_quant: KvQuant::F16,
                measured_weight_bytes: None,
                runtime: RuntimeProfile::LLAMA_CPP,
            },
        );
        points.push(Point {
            label: format!("{} {}", measurement.name, measurement.quant),
            bytes_per_token: (weights + cache) as f64,
            seconds_per_token: 1.0 / measurement.tokens_per_second,
            sparse: arch.is_sparse(),
            softcapped: arch.softcapped_attention,
            fits: footprint.required() < (pool as f64 * 0.92) as u64,
        });
    }
    points
}

#[test]
fn the_decode_model_predicts_real_measurements() {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../validation/measurements.json"
    ))
    .expect("validation dataset is committed alongside the source");
    let dataset: Dataset = serde_json::from_str(&raw).expect("dataset parses");

    // Pool submissions from the same machine before fitting.
    let mut grouped: BTreeMap<String, (&Hardware, Vec<Point>)> = BTreeMap::new();
    for machine in &dataset.machines {
        grouped
            .entry(machine.hardware.fingerprint())
            .or_insert_with(|| (&machine.hardware, Vec::new()))
            .1
            .extend(evaluate(&dataset, machine));
    }

    println!("\n  Fitting the decode model to measurements from real machines\n");
    println!(
        "  {:<26} {:>3} {:>11} {:>7} {:>9} {:>7} {:>9}",
        "MACHINE", "N", "FITTED BW", "R^2", "OVERHEAD", "SHEET", "ACHIEVED"
    );
    println!("  {}", "-".repeat(82));

    let mut fitted = 0;
    let mut trusted = 0;
    let mut worst_r_squared = f64::MAX;
    let mut achieved_fractions = Vec::new();
    let mut overheads = Vec::new();
    let mut sparse_ratios: Vec<f64> = Vec::new();
    let mut softcapped_ratios: Vec<f64> = Vec::new();
    let mut dense_points = 0;

    for (hardware, points) in grouped.values() {
        // Fit through the plain case only. Sparse models and soft-capped ones
        // take different kernel paths, and are compared against the line
        // afterwards rather than allowed to bend it.
        let dense: Vec<&Point> = points
            .iter()
            .filter(|p| p.fits && !p.sparse && !p.softcapped)
            .collect();
        let Some(fit) = fit_line(&dense) else {
            continue;
        };
        if fit.points < 5 {
            continue;
        }

        let sheet = datasheet_gbps(&hardware.name);
        let achieved = sheet.map(|s| fit.bytes_per_s / 1e9 / s);
        println!(
            "  {:<26} {:>3} {:>8.0} GB/s {:>7.3} {:>6.2} ms {:>7} {:>9}",
            hardware.name,
            fit.points,
            fit.bytes_per_s / 1e9,
            fit.r_squared,
            fit.overhead_ms,
            sheet.map_or_else(|| "-".to_owned(), |s| format!("{s:.0}")),
            achieved.map_or_else(|| "-".to_owned(), |a| format!("{:.0}%", a * 100.0)),
        );

        // A poor fit is a finding, not a nuisance. Print what it was fitted
        // through so the reason can be seen rather than tuned away.
        if fit.r_squared < 0.85 {
            println!("      poor fit — every dense point it was built from:");
            let mut listed: Vec<&&Point> = dense.iter().collect();
            listed.sort_by(|a, b| {
                a.bytes_per_token
                    .partial_cmp(&b.bytes_per_token)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for point in listed {
                let implied = point.bytes_per_token / point.seconds_per_token / 1e9;
                println!(
                    "        {:>6.2} GB/token  {:>7.1} tok/s  implies {:>6.0} GB/s   {}",
                    point.bytes_per_token / 1e9,
                    1.0 / point.seconds_per_token,
                    implied,
                    point.label
                );
            }
        }

        // Sparse models against the line the dense ones defined. A ratio below
        // one means the machine ran them slower than their active-parameter
        // traffic predicts.
        for point in points
            .iter()
            .filter(|p| p.fits && (p.sparse || p.softcapped))
        {
            let predicted = point.bytes_per_token / fit.bytes_per_s + fit.overhead_ms / 1000.0;
            let ratio = predicted / point.seconds_per_token;
            let kind = if point.softcapped { "capped" } else { "sparse" };
            println!("      {kind} {:>5.0}%   {}", ratio * 100.0, point.label);
            if fit.points >= TRUSTWORTHY {
                if point.softcapped {
                    softcapped_ratios.push(ratio);
                } else {
                    sparse_ratios.push(ratio);
                }
            }
        }

        fitted += 1;
        dense_points += fit.points;
        if fit.points >= TRUSTWORTHY {
            trusted += 1;
            worst_r_squared = worst_r_squared.min(fit.r_squared);
            overheads.push(fit.overhead_ms);
            if let Some(fraction) = achieved {
                achieved_fractions.push(fraction);
            }
        }
    }

    println!(
        "\n  {dense_points} dense measurements across {fitted} machines; of the {trusted} with \
         {TRUSTWORTHY} or more, the worst R^2 is {worst_r_squared:.3}"
    );
    if !achieved_fractions.is_empty() {
        let mean = achieved_fractions.iter().sum::<f64>() / achieved_fractions.len() as f64;
        println!(
            "  Achieved bandwidth: {:.0}% of datasheet on average ({:.0}%-{:.0}%)",
            mean * 100.0,
            achieved_fractions.iter().copied().fold(f64::MAX, f64::min) * 100.0,
            achieved_fractions.iter().copied().fold(f64::MIN, f64::max) * 100.0
        );
    }
    if !overheads.is_empty() {
        println!(
            "  Per-token overhead: {:.2} ms median",
            median(&mut overheads.clone())
        );
    }
    if !sparse_ratios.is_empty() {
        println!(
            "  Sparse models reach {:.0}% of the speed their active parameters predict \
             ({} measurements)",
            median(&mut sparse_ratios.clone()) * 100.0,
            sparse_ratios.len()
        );
    }
    if !softcapped_ratios.is_empty() {
        println!(
            "  Soft-capped models, which cannot use flash attention, reach {:.0}%              ({} measurements)",
            median(&mut softcapped_ratios.clone()) * 100.0,
            softcapped_ratios.len()
        );
    }
    println!();

    assert!(
        trusted >= 4,
        "only {trusted} machines had {TRUSTWORTHY} or more usable dense measurements"
    );
    assert!(
        worst_r_squared > 0.85,
        "the worst trustworthy fit explained only {worst_r_squared:.3} of the variation, \
         which would mean bytes-per-token is not predicting speed"
    );
    for fraction in &achieved_fractions {
        assert!(
            (0.30..=1.00).contains(fraction),
            "a fitted bandwidth of {:.0}% of datasheet is not believable",
            fraction * 100.0
        );
    }
}
