//! Hold the weight-size model to the real files it claims to predict.
//!
//! The catalog carries the exact byte count of every published build it knows
//! about. That makes this check free and offline: for each of them, compute the
//! size from the architecture and the quantization scheme, and compare against
//! what the file actually weighs.
//!
//! It is the counterpart to the throughput validation, and it exists because
//! the accuracy claim in the README should not be a measurement somebody took
//! once. Every rebuild of the catalog runs this, so a scraper change that
//! quietly mis-reads an architecture, or a new model family whose shape is not
//! described correctly, fails here rather than shipping.
//!
//! It would have caught the two size bugs this project has already had: the
//! tied-embedding tensor stored at the output precision, and the bits-per-weight
//! figures taken from whole-file averages.

use std::collections::BTreeMap;
use whatllm_core::model::Catalog;
use whatllm_core::quant::weight_quant;

/// Mean absolute error the model is expected to hold to.
const MEAN_ERROR_LIMIT: f64 = 0.025;

/// Bits per weight at or above which the formula is expected to be tight.
const RELIABLE_BPW: f64 = 4.0;
/// Worst single build allowed among those formats.
const WORST_ERROR_LIMIT: f64 = 0.12;
/// Worst single build allowed among the two- and three-bit formats.
///
/// Deliberately looser, and the reason is worth stating rather than hiding in a
/// tolerance. llama.cpp does not take a model to two bits uniformly: below about
/// four, it holds more and more tensors back at higher precision, and how many
/// depends on the feed-forward ratio. A mixture-of-experts model, where the
/// experts dominate, drifts furthest — the formula reads a third under on
/// gpt-oss at `Q2_K`.
///
/// Nobody is served by pretending otherwise, and nobody is much harmed by it
/// either: these are the formats where the catalog almost always has a measured
/// file size, which the model prefers over its own arithmetic.
const WORST_LOW_BIT_ERROR_LIMIT: f64 = 0.35;

fn load_catalog() -> Catalog {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../catalog/catalog.json"
    ))
    .expect("the catalog is committed alongside the source");
    serde_json::from_str(&raw).expect("catalog parses")
}

#[test]
fn computed_sizes_match_the_files_they_predict() {
    struct Sample {
        label: String,
        error: f64,
        bits_per_weight: f64,
    }

    let catalog = load_catalog();

    let mut samples = Vec::new();
    let mut by_quant: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut by_model: BTreeMap<&str, Vec<f64>> = BTreeMap::new();

    for model in &catalog.models {
        for build in &model.builds {
            let Some(quant) = weight_quant(&build.quant) else {
                continue;
            };
            if build.bytes == 0 {
                continue;
            }
            let computed = quant.weight_bytes(&model.architecture) as f64;
            let error = (computed - build.bytes as f64) / build.bytes as f64;
            by_quant.entry(quant.id).or_default().push(error);
            by_model
                .entry(model.display_name.as_str())
                .or_default()
                .push(error);
            samples.push(Sample {
                label: format!("{} {}", model.display_name, build.quant),
                error,
                bits_per_weight: quant.body_bpw,
            });
        }
    }

    assert!(
        samples.len() > 200,
        "only {} builds to check; the catalog is too thin to prove anything",
        samples.len()
    );

    let mean = samples.iter().map(|s| s.error.abs()).sum::<f64>() / samples.len() as f64;

    println!(
        "\n  {} measured builds across {} models\n",
        samples.len(),
        catalog.models.len()
    );
    println!("  {:<10} {:>6} {:>9} {:>9}", "FORMAT", "N", "MEAN", "WORST");
    println!("  {}", "-".repeat(38));
    for (quant, errors) in &by_quant {
        let quant_mean = errors.iter().map(|e| e.abs()).sum::<f64>() / errors.len() as f64;
        let worst = errors.iter().copied().fold(0.0f64, |a, b| a.max(b.abs()));
        println!(
            "  {quant:<10} {:>6} {:>8.2}% {:>8.2}%",
            errors.len(),
            quant_mean * 100.0,
            worst * 100.0
        );
    }

    // A model wrong on every build has an architecture this project describes
    // incorrectly, and belongs in `catalog/sources.json` as excluded until it
    // can be described. A model wrong on one build has one bad file. Separating
    // the two is what turns this failure into an action.
    let mut systematic: Vec<(&str, f64, usize)> = by_model
        .iter()
        .filter_map(|(name, errors)| {
            let mut sorted: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[sorted.len() / 2];
            (median > 0.10 && errors.len() >= 3).then_some((*name, median, errors.len()))
        })
        .collect();
    systematic.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if !systematic.is_empty() {
        println!("\n  Wrong on most of their builds - the architecture, not the file:");
        for (name, median, count) in &systematic {
            println!("    {:>6.1}%  {name} ({count} builds)", median * 100.0);
        }
    }

    samples.sort_by(|a, b| {
        b.error
            .abs()
            .partial_cmp(&a.error.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("\n  Furthest out:");
    for sample in samples.iter().take(8) {
        println!("    {:>+7.1}%  {}", sample.error * 100.0, sample.label);
    }
    println!("\n  Mean absolute error: {:.2}%\n", mean * 100.0);

    assert!(
        systematic.is_empty(),
        "{} models are wrong on most of their builds ({}). Their architectures are not described correctly; exclude them in catalog/sources.json with the reason, or fix the derivation.",
        systematic.len(),
        systematic
            .iter()
            .map(|(name, _, _)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(
        mean <= MEAN_ERROR_LIMIT,
        "mean absolute error is {:.2}%, above the {:.1}% this model is held to",
        mean * 100.0,
        MEAN_ERROR_LIMIT * 100.0
    );
    for sample in &samples {
        let limit = if sample.bits_per_weight >= RELIABLE_BPW {
            WORST_ERROR_LIMIT
        } else {
            WORST_LOW_BIT_ERROR_LIMIT
        };
        assert!(
            sample.error.abs() <= limit,
            "{} is {:.1}% out at {:.1} bits per weight, past the {:.0}% allowed there",
            sample.label,
            sample.error * 100.0,
            sample.bits_per_weight,
            limit * 100.0
        );
    }
}

#[test]
fn every_catalog_architecture_describes_a_model_that_could_exist() {
    let catalog = load_catalog();
    assert!(catalog.is_supported());
    for model in &catalog.models {
        model
            .architecture
            .validate()
            .unwrap_or_else(|error| panic!("{}: {error}", model.id));
        assert!(
            model.total_params() > 0,
            "{} has no parameters at all",
            model.id
        );
        assert!(
            model.active_params() <= model.total_params(),
            "{} reads more per token than it stores",
            model.id
        );
    }
}

#[test]
fn a_measured_build_is_never_absurd_against_its_own_parameter_count() {
    // A separate net from the mean: catches a build whose bytes were parsed
    // from the wrong file entirely, which an averaged error can hide.
    let catalog = load_catalog();
    for model in &catalog.models {
        let params = model.total_params() as f64;
        for build in &model.builds {
            let bits_per_weight = build.bytes as f64 * 8.0 / params;
            assert!(
                (1.0..=17.5).contains(&bits_per_weight),
                "{} {} works out at {bits_per_weight:.1} bits per weight, which is \
                 not a quantization of a {:.1}B model",
                model.display_name,
                build.quant,
                params / 1e9
            );
        }
    }
}
