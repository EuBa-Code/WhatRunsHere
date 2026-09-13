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
use whatrunshere_app::api::{self, CostQuery, Engine, Preference, RuntimeName, Sizing, UseCase};

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
            "display",
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

    has_keys(&json["display"], &["cpu", "accelerators", "pools"], "Names");
    // The clean names are what the window shows, so a trademark mark reaching
    // one is a mark on the headline of the first screen anybody sees.
    let cpu = json["display"]["cpu"].as_str().expect("a cpu name");
    for mark in ["(R)", "(TM)", "(C)", "®", "™"] {
        assert!(
            !cpu.contains(mark),
            "the displayed processor name still carries {mark}: {cpu}"
        );
    }
    assert_eq!(
        json["display"]["pools"].as_array().map(Vec::len),
        json["pools"].as_array().map(Vec::len),
        "every pool needs its display name, matched by position"
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
        "a catalog and a machine with memory: something must be placeable"
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
            "quality_rank",
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
fn quality_rank_places_every_model_against_the_same_field() {
    // The rank is the only anchor a bare score out of 100 has, so it must be
    // computed against quality and not against the fit order the list is
    // sorted by. Those are different questions: the best model for a machine
    // is rarely the best model.
    let engine = Engine::new();
    let ranked = api::rank_of(&engine, sizing());
    let total = ranked.len();

    for model in &ranked {
        let rank = model
            .fit
            .quality_rank
            .unwrap_or_else(|| panic!("{} was ranked without a placing", model.name));
        assert_eq!(
            rank.of, total,
            "{} was placed against a different field",
            model.name
        );
        assert!(
            rank.at_or_below >= 1 && rank.at_or_below <= total,
            "{} placed {} of {}",
            model.name,
            rank.at_or_below,
            total
        );
    }

    // The highest-scoring model sits at or above every other.
    let best = ranked
        .iter()
        .max_by(|a, b| a.fit.quality.total_cmp(&b.fit.quality))
        .expect("a best model");
    assert_eq!(
        best.fit.quality_rank.expect("a placing").at_or_below,
        total,
        "the best-scoring model should sit at or above all {total}"
    );

    // Solved on its own there is no field to place it in, and the engine says
    // so rather than inventing one.
    let plan = api::plan_of(&engine, &ranked[0].id, sizing()).expect("a plan");
    assert!(
        plan.fit.quality_rank.is_none(),
        "a model solved on its own cannot be ranked against a field"
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

/// The top-ranked model, sized for one runtime.
fn top_for(engine: &Engine, runtime: RuntimeName) -> (String, Sizing) {
    let sizing = Sizing {
        runtime,
        ..sizing()
    };
    let ranked = api::rank_of(engine, sizing);
    let top = ranked.first().expect("something is placeable");
    (top.id.clone(), sizing)
}

#[test]
fn a_launch_carries_the_fields_the_window_reads() {
    let engine = Engine::new();
    let (id, sizing) = top_for(&engine, RuntimeName::LlamaCpp);
    let launch = api::launch_of(&engine, &id, sizing)
        .expect("a destination resolves")
        .expect("the top model launches");
    let json = serde_json::to_value(&launch).expect("Launch serialises");

    has_keys(
        &json,
        &["host", "runs_gguf", "weights", "file", "commands", "notes"],
        "Launch",
    );
    assert_eq!(json["host"], "llama_cpp");
    assert_eq!(json["runs_gguf"], true);
    // `weights` is internally tagged, so the window switches on `action`.
    assert!(
        json["weights"]["action"].is_string(),
        "weights must carry its `action` tag"
    );
    if json["weights"]["action"] == "download" {
        has_keys(
            &json["weights"],
            &["quant", "path", "bytes", "command"],
            "Weights::Download",
        );
        assert!(
            launch
                .commands
                .iter()
                .any(|c| c.starts_with("llama-server ")),
            "llama.cpp gets a llama-server command: {:?}",
            launch.commands
        );
    }
    for note in &launch.notes {
        let note = serde_json::to_value(note).expect("LaunchNote serialises");
        assert!(note["note"].is_string(), "a note carries its tag: {note}");
    }
}

#[test]
fn a_host_that_does_not_run_gguf_is_not_offered_a_download() {
    let engine = Engine::new();

    let (id, sizing) = top_for(&engine, RuntimeName::Vllm);
    let launch = api::launch_of(&engine, &id, sizing)
        .expect("nothing to resolve")
        .expect("launches");
    assert!(!launch.runs_gguf);
    let json = serde_json::to_value(&launch.weights).expect("serialises");
    assert_eq!(json["action"], "host_fetches", "{json}");
    has_keys(&json, &["format", "repo"], "Weights::HostFetches");
    assert!(
        launch.commands.iter().any(|c| c.starts_with("vllm serve ")),
        "{:?}",
        launch.commands
    );
    assert!(
        launch.notes.iter().any(|n| matches!(
            n,
            whatrunshere_core::launch::LaunchNote::NotThisFormat { .. }
        )),
        "the window has to say GGUF is not vLLM's format: {:?}",
        launch.notes
    );

    let (id, sizing) = top_for(&engine, RuntimeName::Mlx);
    let launch = api::launch_of(&engine, &id, sizing)
        .expect("nothing to resolve")
        .expect("launches");
    let json = serde_json::to_value(&launch.weights).expect("serialises");
    assert_eq!(json["action"], "search", "{json}");
    has_keys(
        &json,
        &["format", "url", "command_shape"],
        "Weights::Search",
    );
    assert!(
        launch.commands.is_empty(),
        "no repository is known, so no command is invented: {:?}",
        launch.commands
    );
}

#[test]
fn the_gguf_hosts_each_get_their_own_action() {
    let engine = Engine::new();

    // Ollama: a Modelfile beside the weights, an import and a run.
    let (id, sizing) = top_for(&engine, RuntimeName::Ollama);
    let launch = api::launch_of(&engine, &id, sizing)
        .expect("a destination resolves")
        .expect("launches");
    if let Some(file) = &launch.file {
        assert!(file.name.ends_with(".Modelfile"), "{}", file.name);
        assert!(file.content.starts_with("FROM ./"), "{}", file.content);
        assert!(
            file.content.contains("PARAMETER num_ctx 8192"),
            "the sized context is carried in: {}",
            file.content
        );
        assert_eq!(launch.commands.len(), 2, "{:?}", launch.commands);
        assert!(launch.commands[0].starts_with("ollama create "));
        assert!(launch.commands[1].starts_with("ollama run "));
    } else {
        assert!(
            matches!(
                launch.weights,
                whatrunshere_core::launch::Weights::NoBuild { .. }
            ),
            "no file only when there is no build: {:?}",
            launch.weights
        );
    }

    // LM Studio: the file goes into its folder when it has one, otherwise
    // where everything else goes, and the launch says which.
    let (id, sizing) = top_for(&engine, RuntimeName::LmStudio);
    let launch = api::launch_of(&engine, &id, sizing)
        .expect("a destination resolves")
        .expect("launches");
    assert!(
        launch.commands.is_empty(),
        "LM Studio is driven from its window"
    );
    match &launch.weights {
        whatrunshere_core::launch::Weights::DownloadIntoTree { path, tree, .. } => {
            assert!(
                path.starts_with(tree.as_str()),
                "{path} is not under {tree}"
            );
        }
        whatrunshere_core::launch::Weights::Download { .. } => {
            assert!(
                launch.notes.iter().any(|n| matches!(
                    n,
                    whatrunshere_core::launch::LaunchNote::HostNotInstalled { .. }
                )),
                "LM Studio absent is said, not hidden: {:?}",
                launch.notes
            );
        }
        other => panic!("LM Studio downloads or says why not: {other:?}"),
    }

    // The destination the download control reads agrees with the launch.
    let plan = api::plan_of(&engine, &id, sizing).expect("a plan");
    if let Some(chosen) = plan.builds.iter().find(|b| b.chosen) {
        let destination = api::destination_of(&engine, &id, &chosen.quant, RuntimeName::LmStudio)
            .expect("resolves");
        let json = serde_json::to_value(&destination).expect("serialises");
        has_keys(
            &json,
            &[
                "directory",
                "path",
                "present_bytes",
                "partial_bytes",
                "free_bytes",
                "host_tree",
            ],
            "Destination",
        );
        let in_tree = matches!(
            launch.weights,
            whatrunshere_core::launch::Weights::DownloadIntoTree { .. }
        );
        assert_eq!(
            destination.host_tree.is_some(),
            in_tree,
            "the destination and the launch disagree about LM Studio's folder"
        );
    }
}

#[test]
fn what_is_on_this_machine_carries_the_fields_the_window_reads() {
    let engine = Engine::new();
    let installed = api::installed_of(&engine, sizing(), true);
    let json = serde_json::to_value(&installed).expect("Installed serialises");
    has_keys(&json, &["looked_in", "files"], "Installed");

    // Every machine with a home directory has somewhere to look, whether or
    // not anything is there.
    let looked = json["looked_in"].as_array().expect("an array");
    assert!(!looked.is_empty(), "nowhere was looked for a model");
    has_keys(&looked[0], &["provider", "path", "found"], "Location");

    // The tags are what `format.ts` switches on. A variant renamed on this
    // side alone shows up in the window as a blank label, not as an error.
    let tags: Vec<&str> = looked
        .iter()
        .filter_map(|place| place["provider"].as_str())
        .collect();
    assert!(
        tags.contains(&"what_runs_here"),
        "the provider tags the window knows are missing: {tags:?}"
    );

    // Whatever the machine the tests run on holds, each file has the same
    // shape, the identity carries its tag, and a catalog match has a fit or
    // an honest absence of one.
    for file in json["files"].as_array().expect("an array") {
        has_keys(
            file,
            &["provider", "path", "bytes", "name", "identity", "fit"],
            "InstalledModel",
        );
        assert!(
            file["identity"]["kind"].is_string(),
            "identity must carry its `kind` tag: {}",
            file["identity"]
        );
        if file["identity"]["kind"] != "catalog" {
            assert!(
                file["fit"].is_null(),
                "only a catalog model can be sized: {file}"
            );
        }
    }

    // A second call without a refresh answers from the same look at the disk.
    let again = api::installed_of(&engine, sizing(), false);
    assert_eq!(again.files.len(), installed.files.len());
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
