//! Handing the model to the program that will run it.
//!
//! The catalog carries GGUF files and nothing else. Three of the hosts a model
//! can be sized for run a GGUF: llama.cpp, LM Studio and Ollama, the last two
//! being llama.cpp underneath. Two do not. vLLM runs the original safetensors
//! from the model's own repository, and MLX runs its own format, published by
//! the community. A download button that fetched the same file whichever host
//! was chosen would hand vLLM a file it does not run, and nobody would find
//! out until they tried, which is the failure this project exists to prevent
//! everywhere else.
//!
//! So the action changes with the host, and this module says what it is: what
//! to do about the weights, the command that starts the model with the
//! context, the layer split and the cache format the solver settled on, and
//! what is worth knowing before pressing anything. The caller supplies the two
//! facts this crate cannot establish, where the file goes and what machine
//! this is. Nothing here reads a disk or an environment.
//!
//! The commands are content, and get the treatment everything else here gets:
//! they reproduce the configuration that was sized. A command that starts the
//! model at a different context, with a different cache format or with half
//! the layers describes a footprint other than the one on screen, and one
//! that does not run at all is worse than none. Paths are quoted for the
//! platform they will be typed on, because a download directory can carry a
//! space on every one of them.

use crate::fit::RunMode;
use crate::memory;
use crate::model::{GgufBuild, ModelEntry};
use crate::quant::{weight_quant, KvQuant};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

/// The program that will host the model.
///
/// Not the same thing as a [`crate::memory::RuntimeProfile`]. LM Studio and
/// Ollama are llama.cpp underneath and share its memory profile, but each
/// wants the file in a different place and starts it a different way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Host {
    /// llama.cpp itself, driven from the command line.
    LlamaCpp,
    /// LM Studio, which lists whatever sits in its own model folder.
    LmStudio,
    /// Ollama, which imports a file into its own store through a Modelfile.
    Ollama,
    /// vLLM, which runs safetensors from the model's original repository.
    Vllm,
    /// MLX, which runs its own format on Apple Silicon.
    Mlx,
}

impl Host {
    /// The name a person knows it by.
    pub const fn label(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            Self::LmStudio => "LM Studio",
            Self::Ollama => "Ollama",
            Self::Vllm => "vLLM",
            Self::Mlx => "MLX",
        }
    }

    /// Whether the host runs the GGUF the catalog carries.
    pub const fn runs_gguf(self) -> bool {
        matches!(self, Self::LlamaCpp | Self::LmStudio | Self::Ollama)
    }
}

/// The operating system the commands will be typed into.
///
/// It decides how a path is quoted, and whether a host runs here at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Platform {
    /// Windows, where both shells accept a double-quoted path.
    Windows,
    /// macOS.
    MacOs,
    /// Linux.
    Linux,
    /// Anything else, treated as a POSIX shell.
    Other,
}

/// Where the caller has decided the weights would be written.
///
/// Resolved outside this crate because it takes a settings file and a look at
/// the disk: whether LM Studio's folder exists is a fact about the machine.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    /// The directory the file lands in.
    pub directory: &'a str,
    /// The file's full path once it has landed.
    pub path: &'a str,
    /// The host's own model folder, when `directory` sits inside it.
    pub host_tree: Option<&'a str>,
    /// Where the host's folder was looked for and not found.
    pub host_tree_missing: Option<&'a str>,
}

/// Everything a launch depends on.
#[derive(Debug, Clone, Copy)]
pub struct LaunchRequest<'a> {
    /// The program that will run the model.
    pub host: Host,
    /// The operating system the commands are for.
    pub platform: Platform,
    /// Whether this machine is Apple Silicon, the one place MLX runs.
    pub apple_silicon: bool,
    /// The model.
    pub model: &'a ModelEntry,
    /// The published build at the chosen weight format, when there is one.
    pub build: Option<&'a GgufBuild>,
    /// The weight format the solver chose, by name.
    pub quant: &'a str,
    /// The cache format the solver chose.
    pub kv_quant: KvQuant,
    /// Where the solver placed the model.
    pub run_mode: RunMode,
    /// Whether the sizing assumed flash attention.
    pub flash_attention: bool,
    /// Context length in tokens, per sequence.
    pub context: u32,
    /// Concurrent sequences.
    pub parallel: u32,
    /// Where the weights would go.
    pub target: Target<'a>,
}

/// What to do to get the model running here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Launch {
    /// The program the commands are for.
    pub host: Host,
    /// Whether that program runs the GGUF the catalog carries. When it does
    /// not, the builds in the catalog are not what it needs.
    pub runs_gguf: bool,
    /// What to do about the weights.
    pub weights: Weights,
    /// A file to write before the commands run, when the host wants one.
    pub file: Option<LaunchFile>,
    /// What to type, in order. Empty for a host driven from its own window.
    pub commands: Vec<String>,
    /// Worth knowing before pressing anything.
    pub notes: Vec<LaunchNote>,
}

/// What happens to the weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Weights {
    /// Fetch the build to the chosen directory. The commands run it from there.
    Download {
        /// Which build.
        quant: String,
        /// Where it lands.
        path: String,
        /// How many bytes that is.
        bytes: u64,
        /// A command that fetches it, for a front end that does not.
        command: String,
    },
    /// Fetch the build into the host's own model folder, where the host lists
    /// it without being told.
    DownloadIntoTree {
        /// Which build.
        quant: String,
        /// Where it lands.
        path: String,
        /// How many bytes that is.
        bytes: u64,
        /// A command that fetches it, for a front end that does not.
        command: String,
        /// The folder the host reads.
        tree: String,
    },
    /// Nothing to fetch from here: the host fetches its own format itself,
    /// from the model's original repository.
    HostFetches {
        /// The format it fetches.
        format: String,
        /// The repository it fetches from.
        repo: String,
    },
    /// Nothing to fetch from here, and the catalog cannot name the repository:
    /// the conversion has to be searched for.
    Search {
        /// The format wanted.
        format: String,
        /// Where to look.
        url: String,
        /// What the command looks like once the repository is known. Not a
        /// command: the repository in it is for the reader to fill in.
        command_shape: String,
    },
    /// The host needs a GGUF and none is published at the chosen format.
    NoBuild {
        /// The format the solver chose.
        quant: String,
    },
}

/// A file the host reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchFile {
    /// Its name.
    pub name: String,
    /// The directory it belongs in, which is the one the weights are in.
    pub directory: String,
    /// What it says.
    pub content: String,
}

/// One setting a host reads from its environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvVar {
    /// The variable.
    pub name: String,
    /// Its value.
    pub value: String,
}

/// Something worth knowing before the model is started.
///
/// Data rather than prose, like [`crate::fit::FitNote`]: each front end puts
/// it into words of its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "note", rename_all = "snake_case")]
pub enum LaunchNote {
    /// The catalog's GGUF is not this host's format. The sizing on screen
    /// describes the same model in a different container.
    NotThisFormat {
        /// What the host runs instead.
        host_format: String,
    },
    /// The host fetches the original weights at full precision, which is
    /// this many bytes rather than the build that was sized.
    FullPrecisionWeights {
        /// Bytes of weights at 16 bits each.
        bytes: u64,
    },
    /// The host runs only somewhere this machine is not.
    HostNeeds {
        /// What it needs.
        platform: String,
    },
    /// The host's folder was not found, so it is probably not installed.
    HostNotInstalled {
        /// Where it was looked for.
        looked_in: String,
    },
    /// Once the file has landed, the host lists the model under this name.
    AppearsInHost {
        /// The name in the host's list.
        name: String,
    },
    /// The host copies the file into its own store, so the disk holds two.
    SecondCopy {
        /// Bytes of the second copy.
        bytes: u64,
    },
    /// The host reads these settings from its environment, and the sizing
    /// assumed them.
    HostEnvironment {
        /// What to set before starting the host.
        variables: Vec<EnvVar>,
    },
    /// The cache format that was sized is not one the host offers.
    CacheFormatUnavailable {
        /// What was sized.
        sized: String,
        /// What the host offers.
        offers: Vec<String>,
    },
    /// Without flash attention the value half of the cache cannot be
    /// compressed, so it stays at 16 bits: about this much more than sized.
    VCacheUncompressed {
        /// Bytes above the sized cache.
        extra_bytes: u64,
    },
    /// The host's context length is set in its own window, and its default is
    /// not what was sized.
    SetContextInHost {
        /// The context that was sized.
        tokens: u32,
    },
}

/// Say how to get the model running on this host.
pub fn launch(req: &LaunchRequest<'_>) -> Launch {
    match req.host {
        Host::LlamaCpp => llama_cpp(req),
        Host::LmStudio => lm_studio(req),
        Host::Ollama => ollama(req),
        Host::Vllm => vllm(req),
        Host::Mlx => mlx(req),
    }
}

/// llama.cpp: fetch the file, then one command that reproduces the sizing.
fn llama_cpp(req: &LaunchRequest<'_>) -> Launch {
    let build = match published(req) {
        Ok(build) => build,
        Err(none) => return without_build(req.host, none),
    };

    let mut command = format!(
        "llama-server -m {} -c {}",
        quoted(req.target.path, req.platform),
        // One cache shared across every slot: llama-server splits `-c` between
        // `-np` slots, so each sequence gets what was sized only when the
        // total is the product.
        u64::from(req.context) * u64::from(req.parallel.max(1)),
    );
    if req.parallel > 1 {
        let _ = write!(command, " -np {}", req.parallel);
    }
    let _ = write!(command, " -ngl {}", gpu_layers(req.run_mode));
    command.push_str(if req.flash_attention {
        " -fa on"
    } else {
        " -fa off"
    });

    let mut notes = Vec::new();
    if req.kv_quant != KvQuant::F16 {
        let _ = write!(command, " -ctk {}", req.kv_quant.id());
        if req.flash_attention {
            let _ = write!(command, " -ctv {}", req.kv_quant.id());
        } else {
            // llama.cpp refuses a quantized value cache without flash
            // attention, so the command leaves it at f16 and says what that
            // costs rather than failing to start.
            notes.push(LaunchNote::VCacheUncompressed {
                extra_bytes: v_cache_penalty(req),
            });
        }
    }

    Launch {
        host: req.host,
        runs_gguf: true,
        weights: download(req, build),
        file: None,
        commands: vec![command],
        notes,
    }
}

/// LM Studio: fetch the file into its own folder, and there is nothing to
/// type. It is driven from its window.
fn lm_studio(req: &LaunchRequest<'_>) -> Launch {
    let build = match published(req) {
        Ok(build) => build,
        Err(none) => return without_build(req.host, none),
    };

    let mut notes = Vec::new();
    let weights = if let Some(tree) = req.target.host_tree {
        notes.push(LaunchNote::AppearsInHost {
            name: build.repo.clone(),
        });
        let fetch = fetch(req, build);
        Weights::DownloadIntoTree {
            quant: fetch.quant,
            path: fetch.path,
            bytes: fetch.bytes,
            command: fetch.command,
            tree: tree.to_owned(),
        }
    } else {
        if let Some(looked_in) = req.target.host_tree_missing {
            notes.push(LaunchNote::HostNotInstalled {
                looked_in: looked_in.to_owned(),
            });
        }
        download(req, build)
    };
    notes.push(LaunchNote::SetContextInHost {
        tokens: req.context,
    });

    Launch {
        host: req.host,
        runs_gguf: true,
        weights,
        file: None,
        commands: Vec::new(),
        notes,
    }
}

/// Ollama: fetch the file, write a Modelfile beside it, import, run.
fn ollama(req: &LaunchRequest<'_>) -> Launch {
    /// The cache formats Ollama's server accepts.
    const OLLAMA_CACHE_TYPES: [&str; 3] = ["f16", "q8_0", "q4_0"];

    let build = match published(req) {
        Ok(build) => build,
        Err(none) => return without_build(req.host, none),
    };

    let name = ollama_name(&req.model.id, req.quant);
    let stem = build
        .file
        .strip_suffix(".gguf")
        .or_else(|| build.file.strip_suffix(".GGUF"))
        .unwrap_or(&build.file);
    let file = LaunchFile {
        name: format!("{stem}.Modelfile"),
        directory: req.target.directory.to_owned(),
        // Relative to the Modelfile, which Ollama resolves against the file's
        // own directory. The filename is a plain one out of the catalog, so it
        // needs no quoting; the directory, which can carry a space, is quoted
        // once, on the command line.
        content: format!("FROM ./{}\nPARAMETER num_ctx {}\n", build.file, req.context),
    };
    let modelfile = joined(req.target.directory, &file.name, req.platform);

    let commands = vec![
        format!(
            "ollama create {name} -f {}",
            quoted(&modelfile, req.platform)
        ),
        format!("ollama run {name}"),
    ];

    let mut notes = vec![LaunchNote::SecondCopy { bytes: build.bytes }];
    let mut variables = Vec::new();
    if req.flash_attention {
        variables.push(EnvVar {
            name: "OLLAMA_FLASH_ATTENTION".to_owned(),
            value: "1".to_owned(),
        });
    }
    if req.kv_quant != KvQuant::F16 {
        let sized = req.kv_quant.id();
        if !OLLAMA_CACHE_TYPES.contains(&sized) {
            notes.push(LaunchNote::CacheFormatUnavailable {
                sized: sized.to_owned(),
                offers: OLLAMA_CACHE_TYPES.iter().map(|s| (*s).to_owned()).collect(),
            });
        } else if req.flash_attention {
            variables.push(EnvVar {
                name: "OLLAMA_KV_CACHE_TYPE".to_owned(),
                value: sized.to_owned(),
            });
        } else {
            notes.push(LaunchNote::VCacheUncompressed {
                extra_bytes: v_cache_penalty(req),
            });
        }
    }
    if req.parallel > 1 {
        variables.push(EnvVar {
            name: "OLLAMA_NUM_PARALLEL".to_owned(),
            value: req.parallel.to_string(),
        });
    }
    if !variables.is_empty() {
        notes.push(LaunchNote::HostEnvironment { variables });
    }

    Launch {
        host: req.host,
        runs_gguf: true,
        weights: download(req, build),
        file: Some(file),
        commands,
        notes,
    }
}

/// vLLM: nothing to fetch from here. It pulls the original repository itself.
fn vllm(req: &LaunchRequest<'_>) -> Launch {
    let mut command = format!(
        "vllm serve {} --max-model-len {}",
        req.model.id, req.context
    );
    if req.parallel > 1 {
        let _ = write!(command, " --max-num-seqs {}", req.parallel);
    }

    let mut notes = vec![LaunchNote::NotThisFormat {
        host_format: "safetensors".to_owned(),
    }];
    if let Some(f16) = weight_quant("F16") {
        notes.push(LaunchNote::FullPrecisionWeights {
            bytes: f16.weight_bytes(&req.model.architecture),
        });
    }
    if req.platform != Platform::Linux {
        notes.push(LaunchNote::HostNeeds {
            platform: "Linux with an NVIDIA or AMD GPU (on Windows, inside WSL)".to_owned(),
        });
    }

    Launch {
        host: req.host,
        runs_gguf: false,
        weights: Weights::HostFetches {
            format: "safetensors".to_owned(),
            repo: req.model.id.clone(),
        },
        file: None,
        commands: vec![command],
        notes,
    }
}

/// MLX: nothing to fetch from here, and no repository to name. The catalog
/// does not record which conversion exists, and guessing one from the model
/// id is exactly the approximation this project refuses.
fn mlx(req: &LaunchRequest<'_>) -> Launch {
    let name = req.model.id.rsplit('/').next().unwrap_or(&req.model.id);
    let mut notes = vec![LaunchNote::NotThisFormat {
        host_format: "its own, published under mlx-community".to_owned(),
    }];
    if !req.apple_silicon {
        notes.push(LaunchNote::HostNeeds {
            platform: "Apple Silicon".to_owned(),
        });
    }

    Launch {
        host: req.host,
        runs_gguf: false,
        weights: Weights::Search {
            format: "its own format".to_owned(),
            url: format!(
                "https://huggingface.co/models?library=mlx&search={}",
                encoded(name)
            ),
            command_shape: "mlx_lm.server --model mlx-community/<the conversion you chose>"
                .to_owned(),
        },
        file: None,
        commands: Vec::new(),
        notes,
    }
}

/// The build a GGUF host needs, or the answer when none is published.
fn published<'a>(req: &LaunchRequest<'a>) -> Result<&'a GgufBuild, Weights> {
    req.build.ok_or_else(|| Weights::NoBuild {
        quant: req.quant.to_owned(),
    })
}

/// A GGUF host with nothing to fetch has nothing to run either.
fn without_build(host: Host, weights: Weights) -> Launch {
    Launch {
        host,
        runs_gguf: true,
        weights,
        file: None,
        commands: Vec::new(),
        notes: Vec::new(),
    }
}

/// What every GGUF host's download says, before it is said which kind.
struct Fetch {
    quant: String,
    path: String,
    bytes: u64,
    command: String,
}

fn fetch(req: &LaunchRequest<'_>, build: &GgufBuild) -> Fetch {
    Fetch {
        quant: build.quant.clone(),
        path: req.target.path.to_owned(),
        bytes: build.bytes,
        command: format!(
            "huggingface-cli download {} {} --local-dir {}",
            build.repo,
            build.file,
            quoted(req.target.directory, req.platform)
        ),
    }
}

/// The ordinary case: the file goes where the caller said.
fn download(req: &LaunchRequest<'_>, build: &GgufBuild) -> Weights {
    let fetch = fetch(req, build);
    Weights::Download {
        quant: fetch.quant,
        path: fetch.path,
        bytes: fetch.bytes,
        command: fetch.command,
    }
}

/// What `-ngl` should say to reproduce the placement.
///
/// 99 rather than the layer count for a full offload: llama.cpp counts the
/// output layer as one more, and anything past the total means all of them.
fn gpu_layers(run_mode: RunMode) -> u32 {
    match run_mode {
        RunMode::Accelerated { .. } | RunMode::Unified => 99,
        RunMode::Offloaded { gpu_layers, .. } => gpu_layers,
        RunMode::Cpu => 0,
    }
}

/// How much larger the cache is when only its key half can be compressed.
///
/// Half the cache is values, so half the saving the sized format promised is
/// not delivered. An approximation for latent-attention models, whose cache
/// is not split evenly, and reported as one.
fn v_cache_penalty(req: &LaunchRequest<'_>) -> u64 {
    let arch = &req.model.architecture;
    let at_f16 = memory::kv_cache_bytes(arch, req.context, KvQuant::F16);
    let sized = memory::kv_cache_bytes(arch, req.context, req.kv_quant);
    at_f16.saturating_sub(sized) / 2 * u64::from(req.parallel.max(1))
}

/// A name Ollama accepts: lowercase, with the format as the tag.
///
/// `Qwen/Qwen3-30B-A3B` at `Q4_K_M` becomes `qwen3-30b-a3b:q4_k_m`.
fn ollama_name(id: &str, quant: &str) -> String {
    fn slug(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for c in text.chars() {
            let c = c.to_ascii_lowercase();
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                out.push(c);
            } else if !out.ends_with('-') {
                out.push('-');
            }
        }
        out.trim_matches('-').to_owned()
    }
    let name = id.rsplit('/').next().unwrap_or(id);
    format!("{}:{}", slug(name), slug(quant))
}

/// A path as an argument, for the shell it will be typed into.
///
/// Double quotes on Windows, which both of its shells accept and which a
/// Windows path cannot contain. Single quotes elsewhere, which nothing inside
/// them can expand; an apostrophe in the path is the one thing that has to be
/// escaped.
fn quoted(path: &str, platform: Platform) -> String {
    match platform {
        Platform::Windows => format!("\"{path}\""),
        Platform::MacOs | Platform::Linux | Platform::Other => {
            format!("'{}'", path.replace('\'', r"'\''"))
        }
    }
}

/// A file inside a directory, with the platform's separator.
fn joined(directory: &str, name: &str, platform: Platform) -> String {
    let separator = match platform {
        Platform::Windows => '\\',
        Platform::MacOs | Platform::Linux | Platform::Other => '/',
    };
    if directory.ends_with(['/', '\\']) {
        format!("{directory}{name}")
    } else {
        format!("{directory}{separator}{name}")
    }
}

/// A search term as a query string value.
fn encoded(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{Architecture, AttentionKind, FfnKind, LayerLayout, LayerSpec};
    use crate::quality::Benchmarks;

    fn model() -> ModelEntry {
        ModelEntry {
            id: "Test/Test-Model-8B".to_owned(),
            display_name: "Test Model 8B".to_owned(),
            family: "Test".to_owned(),
            architecture: Architecture {
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
            },
            benchmarks: Benchmarks::default(),
            builds: vec![GgufBuild {
                quant: "Q4_K_M".to_owned(),
                repo: "someone/Test-Model-8B-GGUF".to_owned(),
                file: "Test-Model-8B-Q4_K_M.gguf".to_owned(),
                bytes: 4_920_000_000,
            }],
            license: None,
            released: None,
        }
    }

    /// A request for a GGUF host on a Linux box with one card, the build
    /// published and the file going to a directory with a space in it.
    fn request(model: &ModelEntry, host: Host) -> LaunchRequest<'_> {
        LaunchRequest {
            host,
            platform: Platform::Linux,
            apple_silicon: false,
            model,
            build: model.builds.first(),
            quant: "Q4_K_M",
            kv_quant: KvQuant::Q8_0,
            run_mode: RunMode::Accelerated { devices: 1 },
            flash_attention: true,
            context: 8192,
            parallel: 1,
            target: Target {
                directory: "/home/me/My Models",
                path: "/home/me/My Models/Test-Model-8B-Q4_K_M.gguf",
                host_tree: None,
                host_tree_missing: None,
            },
        }
    }

    #[test]
    fn llama_cpp_reproduces_the_sizing_on_the_command_line() {
        let model = model();
        let launch = launch(&request(&model, Host::LlamaCpp));

        assert!(launch.runs_gguf);
        assert!(matches!(launch.weights, Weights::Download { .. }));
        assert_eq!(launch.commands.len(), 1);
        let command = &launch.commands[0];
        assert!(
            command.starts_with("llama-server -m '/home/me/My Models/Test-Model-8B-Q4_K_M.gguf'"),
            "the path is quoted, because it has a space in it: {command}"
        );
        for flag in ["-c 8192", "-ngl 99", "-fa on", "-ctk q8_0", "-ctv q8_0"] {
            assert!(command.contains(flag), "{flag} missing from {command}");
        }
        assert!(!command.contains("-np"), "one sequence needs no slots");
        assert!(launch.notes.is_empty(), "{:?}", launch.notes);
    }

    #[test]
    fn a_partial_offload_and_concurrency_are_carried_into_the_command() {
        let model = model();
        let mut req = request(&model, Host::LlamaCpp);
        req.run_mode = RunMode::Offloaded {
            gpu_layers: 20,
            total_layers: 32,
        };
        req.parallel = 4;
        let command = &launch(&req).commands[0];

        assert!(command.contains("-ngl 20"), "{command}");
        // Four sequences of 8k each share one cache, so the cache is 32k.
        assert!(command.contains("-c 32768 -np 4"), "{command}");
    }

    #[test]
    fn without_flash_attention_the_value_cache_is_left_alone_and_said_so() {
        let model = model();
        let mut req = request(&model, Host::LlamaCpp);
        req.flash_attention = false;
        let launch = launch(&req);
        let command = &launch.commands[0];

        assert!(command.contains("-fa off"), "{command}");
        assert!(command.contains("-ctk q8_0"), "{command}");
        assert!(
            !command.contains("-ctv"),
            "llama.cpp refuses a quantized V cache without flash attention: {command}"
        );
        assert!(
            launch.notes.iter().any(
                |n| matches!(n, LaunchNote::VCacheUncompressed { extra_bytes } if *extra_bytes > 0)
            ),
            "{:?}",
            launch.notes
        );
    }

    #[test]
    fn cpu_placement_keeps_every_layer_off_the_card() {
        let model = model();
        let mut req = request(&model, Host::LlamaCpp);
        req.run_mode = RunMode::Cpu;
        req.kv_quant = KvQuant::F16;
        let command = &launch(&req).commands[0];
        assert!(command.contains("-ngl 0"), "{command}");
        assert!(
            !command.contains("-ctk"),
            "an f16 cache is the default: {command}"
        );
    }

    #[test]
    fn windows_paths_are_double_quoted() {
        let model = model();
        let mut req = request(&model, Host::LlamaCpp);
        req.platform = Platform::Windows;
        req.target = Target {
            directory: "D:\\WhatRunsHere Models",
            path: "D:\\WhatRunsHere Models\\Test-Model-8B-Q4_K_M.gguf",
            host_tree: None,
            host_tree_missing: None,
        };
        let launch = launch(&req);
        assert!(
            launch.commands[0]
                .contains("-m \"D:\\WhatRunsHere Models\\Test-Model-8B-Q4_K_M.gguf\""),
            "{}",
            launch.commands[0]
        );
        let Weights::Download { command, .. } = &launch.weights else {
            panic!("a download")
        };
        assert!(
            command.ends_with("--local-dir \"D:\\WhatRunsHere Models\""),
            "{command}"
        );
    }

    #[test]
    fn an_apostrophe_survives_posix_quoting() {
        assert_eq!(
            quoted("/home/o'brien/models", Platform::MacOs),
            "'/home/o'\\''brien/models'"
        );
    }

    #[test]
    fn lm_studio_downloads_into_its_own_folder_when_it_has_one() {
        let model = model();
        let mut req = request(&model, Host::LmStudio);
        req.target = Target {
            directory: "/home/me/.lmstudio/models/someone/Test-Model-8B-GGUF",
            path: "/home/me/.lmstudio/models/someone/Test-Model-8B-GGUF/Test-Model-8B-Q4_K_M.gguf",
            host_tree: Some("/home/me/.lmstudio/models"),
            host_tree_missing: None,
        };
        let launch = launch(&req);

        assert!(
            matches!(&launch.weights, Weights::DownloadIntoTree { tree, .. } if tree == "/home/me/.lmstudio/models"),
            "{:?}",
            launch.weights
        );
        assert!(
            launch.commands.is_empty(),
            "LM Studio is driven from its window"
        );
        assert!(launch.notes.iter().any(
            |n| matches!(n, LaunchNote::AppearsInHost { name } if name == "someone/Test-Model-8B-GGUF")
        ));
        assert!(launch
            .notes
            .iter()
            .any(|n| matches!(n, LaunchNote::SetContextInHost { tokens: 8192 })));
    }

    #[test]
    fn lm_studio_absent_is_said_rather_than_failed() {
        let model = model();
        let mut req = request(&model, Host::LmStudio);
        req.target.host_tree_missing = Some("/home/me/.lmstudio/models");
        let launch = launch(&req);

        assert!(
            matches!(launch.weights, Weights::Download { .. }),
            "falls back to the chosen directory: {:?}",
            launch.weights
        );
        assert!(launch.notes.iter().any(
            |n| matches!(n, LaunchNote::HostNotInstalled { looked_in } if looked_in == "/home/me/.lmstudio/models")
        ));
    }

    #[test]
    fn ollama_gets_a_modelfile_beside_the_weights_and_an_import() {
        let model = model();
        let mut req = request(&model, Host::Ollama);
        req.parallel = 2;
        let launch = launch(&req);

        let file = launch.file.as_ref().expect("a Modelfile");
        assert_eq!(file.name, "Test-Model-8B-Q4_K_M.Modelfile");
        assert_eq!(file.directory, "/home/me/My Models");
        assert_eq!(
            file.content,
            "FROM ./Test-Model-8B-Q4_K_M.gguf\nPARAMETER num_ctx 8192\n"
        );

        assert_eq!(
            launch.commands,
            vec![
                "ollama create test-model-8b:q4_k_m -f '/home/me/My Models/Test-Model-8B-Q4_K_M.Modelfile'",
                "ollama run test-model-8b:q4_k_m",
            ]
        );

        assert!(launch.notes.iter().any(|n| matches!(
            n,
            LaunchNote::SecondCopy {
                bytes: 4_920_000_000
            }
        )));
        let Some(LaunchNote::HostEnvironment { variables }) = launch
            .notes
            .iter()
            .find(|n| matches!(n, LaunchNote::HostEnvironment { .. }))
        else {
            panic!("the settings the sizing assumed: {:?}", launch.notes)
        };
        let pairs: Vec<(&str, &str)> = variables
            .iter()
            .map(|v| (v.name.as_str(), v.value.as_str()))
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("OLLAMA_FLASH_ATTENTION", "1"),
                ("OLLAMA_KV_CACHE_TYPE", "q8_0"),
                ("OLLAMA_NUM_PARALLEL", "2"),
            ]
        );
    }

    #[test]
    fn a_cache_format_ollama_does_not_offer_is_named_rather_than_mapped() {
        let model = model();
        let mut req = request(&model, Host::Ollama);
        req.kv_quant = KvQuant::Q5_1;
        let launch = launch(&req);

        assert!(launch.notes.iter().any(
            |n| matches!(n, LaunchNote::CacheFormatUnavailable { sized, offers } if sized == "q5_1" && offers.len() == 3)
        ));
        let vars: Vec<&EnvVar> = launch
            .notes
            .iter()
            .filter_map(|n| match n {
                LaunchNote::HostEnvironment { variables } => Some(variables),
                _ => None,
            })
            .flatten()
            .collect();
        assert!(
            !vars.iter().any(|v| v.name == "OLLAMA_KV_CACHE_TYPE"),
            "no cache type is invented: {vars:?}"
        );
    }

    #[test]
    fn vllm_fetches_the_original_repository_itself() {
        let model = model();
        let mut req = request(&model, Host::Vllm);
        req.parallel = 8;
        let launch = launch(&req);

        assert!(!launch.runs_gguf);
        assert!(
            matches!(&launch.weights, Weights::HostFetches { repo, format } if repo == "Test/Test-Model-8B" && format == "safetensors"),
            "{:?}",
            launch.weights
        );
        assert_eq!(
            launch.commands,
            vec!["vllm serve Test/Test-Model-8B --max-model-len 8192 --max-num-seqs 8"]
        );
        assert!(launch
            .notes
            .iter()
            .any(|n| matches!(n, LaunchNote::NotThisFormat { host_format } if host_format == "safetensors")));
        // The build sized was a 4-bit one; the weights vLLM pulls are 16-bit.
        assert!(launch.notes.iter().any(
            |n| matches!(n, LaunchNote::FullPrecisionWeights { bytes } if *bytes > 3 * 4_920_000_000)
        ));
        assert!(
            !launch
                .notes
                .iter()
                .any(|n| matches!(n, LaunchNote::HostNeeds { .. })),
            "Linux is where vLLM lives"
        );

        req.platform = Platform::Windows;
        assert!(super::launch(&req)
            .notes
            .iter()
            .any(|n| matches!(n, LaunchNote::HostNeeds { .. })));
    }

    #[test]
    fn mlx_is_searched_for_rather_than_guessed() {
        let model = model();
        let mut req = request(&model, Host::Mlx);
        let launch = launch(&req);

        assert!(!launch.runs_gguf);
        assert!(launch.commands.is_empty(), "no repository, no command");
        let Weights::Search {
            url, command_shape, ..
        } = &launch.weights
        else {
            panic!("{:?}", launch.weights)
        };
        assert_eq!(
            url,
            "https://huggingface.co/models?library=mlx&search=Test-Model-8B"
        );
        assert!(command_shape.contains("mlx-community/<"));
        assert!(launch.notes.iter().any(
            |n| matches!(n, LaunchNote::HostNeeds { platform } if platform == "Apple Silicon")
        ));

        req.apple_silicon = true;
        assert!(!super::launch(&req)
            .notes
            .iter()
            .any(|n| matches!(n, LaunchNote::HostNeeds { .. })));
    }

    #[test]
    fn a_gguf_host_without_a_published_build_has_nothing_to_run() {
        let model = model();
        let mut req = request(&model, Host::Ollama);
        req.build = None;
        req.quant = "Q6_K";
        let launch = launch(&req);
        assert!(matches!(&launch.weights, Weights::NoBuild { quant } if quant == "Q6_K"));
        assert!(launch.commands.is_empty());
        assert!(launch.file.is_none());
    }

    #[test]
    fn ollama_names_are_lowercase_and_tagged_with_the_format() {
        assert_eq!(
            ollama_name("Qwen/Qwen3-30B-A3B", "Q4_K_M"),
            "qwen3-30b-a3b:q4_k_m"
        );
        assert_eq!(
            ollama_name("google/gemma-3-27b-it", "IQ4_XS"),
            "gemma-3-27b-it:iq4_xs"
        );
        assert_eq!(
            ollama_name("odd name/With Spaces!", "Q8_0"),
            "with-spaces:q8_0"
        );
    }

    #[test]
    fn a_search_term_is_encoded_for_a_query_string() {
        assert_eq!(encoded("Test-Model-8B"), "Test-Model-8B");
        assert_eq!(encoded("a b&c"), "a%20b%26c");
    }

    #[test]
    fn a_directory_with_a_trailing_separator_is_not_given_a_second() {
        assert_eq!(
            joined("D:\\", "x.Modelfile", Platform::Windows),
            "D:\\x.Modelfile"
        );
        assert_eq!(
            joined("D:\\Models", "x", Platform::Windows),
            "D:\\Models\\x"
        );
        assert_eq!(joined("/models/", "x", Platform::Linux), "/models/x");
    }
}
