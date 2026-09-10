//! The shape the window is written against.
//!
//! `ui/src/engine.ts` declares, by hand, the types these commands return. A
//! hand-written declaration is a promise about a shape nothing else checks:
//! rename a field in Rust and the TypeScript keeps compiling, keeps type
//! checking, and starts reading `undefined`. The window then shows a blank
//! where a figure should be, on someone else's machine, with nothing in any
//! log to say why.
//!
//! So the field names are asserted here, against real serialised output from
//! the machine the tests run on. Renaming one fails this test, and the failure
//! names the file that has to change with it.

use serde_json::Value;
use whatllm_app::api::{self, CostQuery, Engine, Preference, RuntimeName, Sizing, UseCase};

fn sizing() -> Sizing {
    Sizing {
        context: 8192,
        parallel: 1,
        use_case: UseCase::General,
        preference: Preference::Balanced,
        runtime: RuntimeName::LlamaCpp,
    }
}

/// Assert that `value` is an object carrying every one of `keys`.
///
/// Reports all of the missing ones at once: a rename usually moves a family of
/// fields, and finding them one test run at a time is needless.
fn has_keys(value: &Value, keys: &[&str], what: &str) {
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("{what} did not serialise as an object: {value}"));
    let missing: Vec<&str> = keys
        .iter()
        .copied()
        .filter(|key| !object.contains_key(*key))
        .collect();
    assert!(
        missing.is_empty(),
        "{what} is missing {missing:?}. ui/src/engine.ts declares these; \
         update both together. Present: {:?}",
        object.keys().collect::<Vec<_>>()
    );
}

#[test]
fn the_machine_command_carries_the_fields_the_window_reads() {
    let engine = Engine::new();
    let json = serde_json::to_value(api::machine_of(&engine)).expect("Machine serialises");

    has_keys(
        &json,
        &[
            "detection",
            "calibration",
            "measurement",
            "pools",
            "catalog_source",
            "catalog_size",
            "catalog_generated",
        ],
        "Machine",
    );
    has_keys(&json["detection"], &["system", "notes"], "Detection");
    has_keys(
        &json["detection"]["system"],
        &["cpu", "memory", "accelerators", "os"],
        "SystemProfile",
    );
    has_keys(
        &json["detection"]["system"]["memory"],
        &["total_bytes", "available_bytes"],
        "HostMemory",
    );
    has_keys(
        &json["calibration"],
        &["host", "accelerator", "overhead_ms_per_token", "source"],
        "Calibration",
    );
    has_keys(
        &json["calibration"]["host"],
        &["bandwidth_bytes_per_s"],
        "DeviceThroughput",
    );

    // Every machine has at least one pool: the memory it is made of.
    let pools = json["pools"].as_array().expect("pools is an array");
    assert!(!pools.is_empty(), "a machine always has somewhere to load");
    has_keys(
        &pools[0],
        &["label", "kind", "usable_bytes", "devices"],
        "MemoryPool",
    );
}

#[test]
fn a_ranked_model_carries_the_fields_the_list_reads() {
    let engine = Engine::new();
    let ranked = api::rank_of(&engine, sizing());
    assert!(
        !ranked.is_empty(),
        "56 models and a machine with memory: something must be placeable"
    );

    let json = serde_json::to_value(&ranked[0]).expect("RankedModel serialises");
    has_keys(
        &json,
        &[
            "id",
            "name",
            "family",
            "parameters",
            "active_parameters",
            "sparse",
            "max_trained_context",
            "fit",
        ],
        "RankedModel",
    );
    has_keys(
        &json["fit"],
        &[
            "quant",
            "bits_per_weight",
            "kv_quant",
            "run_mode",
            "pool",
            "memory",
            "utilisation",
            "verdict",
            "decode_tps",
            "prefill_tps",
            "confidence",
            "quality",
            "quality_full_precision",
            "degradation",
            "quality_coverage",
            "max_context",
            "score",
            "notes",
        ],
        "FitView",
    );
    has_keys(
        &json["fit"]["memory"],
        &[
            "weights",
            "weights_measured",
            "kv_cache",
            "activations",
            "attention_scores",
            "runtime_overhead",
            "headroom",
            "resident",
            "required",
        ],
        "Footprint",
    );
    // `run_mode` is an internally tagged enum, so the window switches on this.
    assert!(
        json["fit"]["run_mode"]["mode"].is_string(),
        "run_mode must carry its `mode` tag"
    );
}

#[test]
fn the_ranking_is_ordered_and_the_footprint_adds_up() {
    let engine = Engine::new();
    let ranked = api::rank_of(&engine, sizing());

    for pair in ranked.windows(2) {
        assert!(
            pair[0].fit.score >= pair[1].fit.score,
            "{} scored {} above {} at {}",
            pair[1].name,
            pair[1].fit.score,
            pair[0].name,
            pair[0].fit.score
        );
    }

    // The window draws the memory column from the parts and labels it with the
    // total. If those disagreed the column would not fill its own bar.
    for model in &ranked {
        let m = &model.fit.memory;
        assert_eq!(
            m.resident,
            m.weights + m.kv_cache + m.activations + m.runtime_overhead,
            "{}'s parts do not sum to its resident total",
            model.name
        );
        assert_eq!(
            m.required,
            m.resident + m.headroom,
            "{}'s headroom is not counted in what it requires",
            model.name
        );
    }
}

#[test]
fn a_plan_carries_a_curve_the_window_can_draw() {
    let engine = Engine::new();
    let ranked = api::rank_of(&engine, sizing());
    let top = ranked.first().expect("something is placeable");

    let plan = api::plan_of(&engine, &top.id, sizing()).expect("the top model plans");
    let json = serde_json::to_value(&plan).expect("Plan serialises");
    has_keys(
        &json,
        &[
            "id",
            "name",
            "family",
            "license",
            "released",
            "fit",
            "builds",
            "curve",
            "pool_bytes",
        ],
        "Plan",
    );

    assert!(!plan.curve.is_empty(), "a plan always has a curve");
    has_keys(
        &json["curve"][0],
        &["context", "required", "decode_tps"],
        "CurveSample",
    );

    // Memory rises with context and speed falls, which is the whole shape the
    // chart exists to show. A curve that did neither would plot as a flat line
    // and quietly say nothing.
    let first = plan.curve.first().expect("at least one point");
    let last = plan.curve.last().expect("at least one point");
    if plan.curve.len() > 1 {
        assert!(
            last.required >= first.required,
            "footprint fell as context grew: {} to {}",
            first.required,
            last.required
        );
        assert!(
            last.decode_tps <= first.decode_tps,
            "speed rose as context grew: {} to {}",
            first.decode_tps,
            last.decode_tps
        );
    }

    // Exactly one build is marked chosen, and it is the one the fit names.
    let chosen: Vec<&str> = plan
        .builds
        .iter()
        .filter(|b| b.chosen)
        .map(|b| b.quant.as_str())
        .collect();
    assert!(
        chosen.len() <= 1,
        "more than one build marked as chosen: {chosen:?}"
    );
    if let Some(name) = chosen.first() {
        assert!(
            name.eq_ignore_ascii_case(&plan.fit.quant),
            "the chosen build {name} is not the fit's {}",
            plan.fit.quant
        );
    }
}

#[test]
fn a_cost_comparison_serialises_with_its_verdict_tag() {
    let engine = Engine::new();
    let ranked = api::rank_of(&engine, sizing());
    let top = ranked.first().expect("something is placeable");

    let comparison = api::cost_of(
        &engine,
        &top.id,
        sizing(),
        CostQuery {
            requests: 3000,
            input: 8000,
            output: 1200,
            price_per_kwh: 0.25,
            watts: 450.0,
            hardware_cost: 0.0,
            api_input: 0.30,
            api_output: 1.20,
        },
    )
    .expect("a placeable model can be costed");

    let json = serde_json::to_value(comparison).expect("CostComparison serialises");
    let object = json.as_object().expect("an object");
    assert!(
        object.contains_key("verdict"),
        "the verdict tag the window switches on is missing: {:?}",
        object.keys().collect::<Vec<_>>()
    );
}

#[test]
fn an_unknown_model_is_absent_rather_than_a_failure() {
    // The window can hold a selection across a catalog update that dropped the
    // model. Nothing is returned, and nothing panics.
    let engine = Engine::new();
    assert!(api::plan_of(&engine, "nobody/does-not-exist", sizing()).is_none());
    assert!(api::cost_of(
        &engine,
        "nobody/does-not-exist",
        sizing(),
        CostQuery {
            requests: 1,
            input: 1,
            output: 1,
            price_per_kwh: 0.0,
            watts: 0.0,
            hardware_cost: 0.0,
            api_input: 0.0,
            api_output: 0.0,
        },
    )
    .is_none());
}
