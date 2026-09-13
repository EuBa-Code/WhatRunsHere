//! What is already on this machine.
//!
//! Every local runtime keeps its model files somewhere of its own, and a
//! person who has used two of them has weights in two places and no list.
//! This crate reads the places and identifies what it finds against the
//! catalog, so the answer to "which models do I have" comes with the answer
//! to "and how would each of them run here", which the engine already knows
//! how to give for anything it can identify.
//!
//! Five places, each documented by its owner:
//!
//! - the directory `WhatRunsHere` downloads into;
//! - LM Studio's folder, `~/.lmstudio/models/<publisher>/<repo>/`;
//! - Ollama's store, a manifest per tag beside a content-addressed blob store,
//!   where the blob is the GGUF byte for byte;
//! - llama.cpp's cache, where `-hf` downloads land;
//! - the `HuggingFace` hub cache.
//!
//! Identification is by exact size. The catalog records the byte count of
//! every published build, and a multi-gigabyte length is as good as a
//! fingerprint; the filename only breaks a tie, because a name is the one
//! thing anybody can change. A file the catalog does not know is described
//! from its own header (name, architecture, format) and declared not sizeable,
//! rather than sized from a guess.
//!
//! Nothing here reaches the network, and nothing talks to a running program.
//! A directory that is not there is the common case, and is reported as where
//! it was looked for rather than as an error.

#![forbid(unsafe_code)]

pub mod gguf;
mod ollama;

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use whatrunshere_core::model::Catalog;

/// Who keeps the file.
///
/// Ordered as listed, which is the order the places are read in and the order
/// a person would expect them in: this application's own folder first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    /// This application's own download directory.
    WhatRunsHere,
    /// LM Studio's model folder.
    LmStudio,
    /// Ollama's blob store.
    Ollama,
    /// llama.cpp's download cache.
    LlamaCpp,
    /// The `HuggingFace` hub cache, shared by every tool that uses their library.
    HuggingFace,
}

impl Provider {
    /// The name a person knows it by.
    pub const fn label(self) -> &'static str {
        match self {
            Self::WhatRunsHere => "WhatRunsHere",
            Self::LmStudio => "LM Studio",
            Self::Ollama => "Ollama",
            Self::LlamaCpp => "llama.cpp",
            Self::HuggingFace => "HuggingFace cache",
        }
    }
}

/// Where a provider's files were looked for, and whether the place exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Location {
    /// Whose place it is.
    pub provider: Provider,
    /// The directory.
    pub path: String,
    /// Whether it is there. Absent is the common case and means the program
    /// is probably not installed.
    pub found: bool,
}

/// What a file turned out to be.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Identity {
    /// A published build the catalog knows, matched by exact size.
    Catalog {
        /// Catalog id.
        id: String,
        /// Name to show a person.
        display_name: String,
        /// Which build.
        quant: String,
    },
    /// A GGUF the catalog does not know, described from its own header.
    Header(gguf::Header),
    /// A file that could not be read as a GGUF at all.
    Unknown {
        /// Why not.
        reason: String,
    },
}

/// One model file on this machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledFile {
    /// Who keeps it.
    pub provider: Provider,
    /// Where it is.
    pub path: String,
    /// How many bytes.
    pub bytes: u64,
    /// What the provider calls it: the filename, or for Ollama `name:tag`.
    pub name: String,
    /// What it is.
    pub identity: Identity,
}

/// Everything found, and everywhere looked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scan {
    /// Where each provider's files were looked for.
    pub looked_in: Vec<Location>,
    /// Every model file found, catalog models first, largest first within.
    pub files: Vec<InstalledFile>,
}

/// The directories to read, one per provider, resolved from the environment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Roots {
    /// The download directory, chosen or default.
    pub whatrunshere: Option<PathBuf>,
    /// `~/.lmstudio/models`.
    pub lm_studio: Option<PathBuf>,
    /// `~/.ollama/models`, or `OLLAMA_MODELS`.
    pub ollama: Option<PathBuf>,
    /// The platform cache directory plus `llama.cpp`, or `LLAMA_CACHE`.
    pub llama_cpp: Option<PathBuf>,
    /// `~/.cache/huggingface/hub`, or `HF_HUB_CACHE`, or `HF_HOME/hub`.
    pub hugging_face: Option<PathBuf>,
}

impl Roots {
    /// Where each provider keeps its files on this machine.
    ///
    /// Each path is the one the program's own documentation gives, with the
    /// environment variable the program honours taken first when it is set.
    pub fn from_environment() -> Self {
        let home = dirs::home_dir();
        let env = |name: &str| std::env::var_os(name).map(PathBuf::from);
        Self {
            whatrunshere: whatrunshere_state::weights::download_dir().ok(),
            lm_studio: whatrunshere_state::weights::lm_studio_models_dir(),
            ollama: env("OLLAMA_MODELS")
                .or_else(|| home.as_ref().map(|h| h.join(".ollama").join("models"))),
            // llama.cpp's `fs_get_cache_directory`: the platform cache
            // directory, then `llama.cpp/`.
            llama_cpp: env("LLAMA_CACHE")
                .or_else(|| dirs::cache_dir().map(|c| c.join("llama.cpp"))),
            hugging_face: env("HF_HUB_CACHE")
                .or_else(|| env("HF_HOME").map(|h| h.join("hub")))
                .or_else(|| {
                    home.as_ref()
                        .map(|h| h.join(".cache").join("huggingface").join("hub"))
                }),
        }
    }
}

/// Read every place on this machine and identify what is there.
pub fn scan(catalog: &Catalog) -> Scan {
    scan_roots(&Roots::from_environment(), catalog)
}

/// Read the given places. What [`scan`] does, with the places chosen by the
/// caller, which is how the tests point it at directories of their own.
pub fn scan_roots(roots: &Roots, catalog: &Catalog) -> Scan {
    /// How deep a provider's tree can go before this stops following it.
    /// The deepest documented layout is the hub cache's
    /// `models--x--y/snapshots/<rev>/<subdir>/<file>`.
    const DEPTH: usize = 6;

    let mut looked_in = Vec::new();
    let mut files = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    let places = [
        (Provider::WhatRunsHere, &roots.whatrunshere),
        (Provider::LmStudio, &roots.lm_studio),
        (Provider::Ollama, &roots.ollama),
        (Provider::LlamaCpp, &roots.llama_cpp),
        (Provider::HuggingFace, &roots.hugging_face),
    ];
    for (provider, root) in places {
        let Some(root) = root else { continue };
        let found = root.is_dir();
        looked_in.push(Location {
            provider,
            path: root.display().to_string(),
            found,
        });
        if !found {
            continue;
        }
        if provider == Provider::Ollama {
            for model in ollama::models(root) {
                if let Some(file) = describe(provider, &model.blob, model.name, catalog, &mut seen)
                {
                    files.push(file);
                }
            }
            continue;
        }
        for path in gguf_files(root, DEPTH) {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if let Some(file) = describe(provider, &path, name, catalog, &mut seen) {
                files.push(file);
            }
        }
    }

    // Catalog models first, so what the engine can size leads; then largest
    // first, which is the order a disk is read in.
    files.sort_by(|a, b| {
        let rank = |f: &InstalledFile| match f.identity {
            Identity::Catalog { .. } => 0,
            Identity::Header(_) => 1,
            Identity::Unknown { .. } => 2,
        };
        rank(a)
            .cmp(&rank(b))
            .then_with(|| b.bytes.cmp(&a.bytes))
            .then_with(|| a.path.cmp(&b.path))
    });

    Scan { looked_in, files }
}

/// Say what one file is. `None` when it has been seen already under another
/// path, which the hub cache's snapshot links make routine.
fn describe(
    provider: Provider,
    path: &Path,
    name: String,
    catalog: &Catalog,
    seen: &mut HashSet<PathBuf>,
) -> Option<InstalledFile> {
    let Ok(meta) = std::fs::metadata(path) else {
        return None;
    };
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if !seen.insert(canonical) {
        return None;
    }
    let bytes = meta.len();

    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let identity = match catalog.identify(bytes, &file_name) {
        Some((model, build)) => Identity::Catalog {
            id: model.id.clone(),
            display_name: model.display_name.clone(),
            quant: build.quant.clone(),
        },
        None => match gguf::read_header(path) {
            Ok(header) => Identity::Header(header),
            Err(reason) => Identity::Unknown { reason },
        },
    };

    Some(InstalledFile {
        provider,
        path: path.display().to_string(),
        bytes,
        name,
        identity,
    })
}

/// Every `.gguf` under `root`, to `depth` levels, hidden directories skipped.
///
/// A `.part` beside a finished file is a transfer in progress and is not a
/// model yet; the hub cache's `.locks` and `.no_exist` are bookkeeping.
fn gguf_files(root: &Path, depth: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, level)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                if level < depth && !name.starts_with('.') {
                    stack.push((path, level + 1));
                }
            } else if name.to_ascii_lowercase().ends_with(".gguf") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use whatrunshere_core::arch::{Architecture, AttentionKind, FfnKind, LayerLayout, LayerSpec};
    use whatrunshere_core::model::{GgufBuild, ModelEntry};
    use whatrunshere_core::quality::Benchmarks;

    fn catalog() -> Catalog {
        Catalog {
            version: 1,
            generated: "today".to_owned(),
            models: vec![ModelEntry {
                id: "Test/Test-8B".to_owned(),
                display_name: "Test 8B".to_owned(),
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
                    repo: "someone/Test-8B-GGUF".to_owned(),
                    file: "Test-8B-Q4_K_M.gguf".to_owned(),
                    bytes: 4_444,
                }],
                license: None,
                released: None,
            }],
        }
    }

    fn sandbox() -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let root = std::env::temp_dir().join(format!("whatrunshere-providers-{unique}"));
        std::fs::create_dir_all(&root).expect("sandbox");
        root
    }

    /// A GGUF that says what it is and nothing more, at the size asked for.
    fn gguf(name: &str, architecture: &str, size: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes());
        for (key, value) in [
            ("general.name", name),
            ("general.architecture", architecture),
        ] {
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(value.len() as u64).to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        }
        out.resize(size, 0);
        out
    }

    #[test]
    fn every_place_is_read_and_each_file_is_said_to_be_what_it_is() {
        let root = sandbox();
        let catalog = catalog();

        // WhatRunsHere's directory: the catalog build, renamed, and a transfer in
        // progress beside it.
        let whatrunshere = root.join("WhatRunsHere Models");
        std::fs::create_dir_all(&whatrunshere).expect("dir");
        std::fs::write(whatrunshere.join("moved.gguf"), vec![0u8; 4_444]).expect("file");
        std::fs::write(
            whatrunshere.join("Test-8B-Q4_K_M.gguf.part"),
            vec![0u8; 100],
        )
        .expect("part");

        // LM Studio's tree: a GGUF the catalog does not know.
        let lm = root
            .join(".lmstudio")
            .join("models")
            .join("pub")
            .join("repo");
        std::fs::create_dir_all(&lm).expect("dir");
        std::fs::write(lm.join("other-Q8_0.gguf"), gguf("Other 3B", "llama", 3_000)).expect("file");

        // Ollama's store: the catalog build as a blob.
        let ollama = root.join(".ollama").join("models");
        let manifests = ollama
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("test");
        std::fs::create_dir_all(&manifests).expect("dir");
        std::fs::create_dir_all(ollama.join("blobs")).expect("dir");
        std::fs::write(ollama.join("blobs").join("sha256-0123"), vec![0u8; 4_444]).expect("blob");
        std::fs::write(
            manifests.join("8b"),
            r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:0123"}]}"#,
        )
        .expect("manifest");

        // The hub cache: something that is not a GGUF at all despite its name.
        let hub = root
            .join("hub")
            .join("models--x--y")
            .join("snapshots")
            .join("abc");
        std::fs::create_dir_all(&hub).expect("dir");
        std::fs::write(hub.join("not-really.gguf"), b"hello").expect("file");

        let roots = Roots {
            whatrunshere: Some(whatrunshere.clone()),
            lm_studio: Some(root.join(".lmstudio").join("models")),
            ollama: Some(ollama),
            llama_cpp: Some(root.join("no-such-cache")),
            hugging_face: Some(root.join("hub")),
        };
        let scan = scan_roots(&roots, &catalog);

        // Everywhere was looked, and the absent one says so.
        assert_eq!(scan.looked_in.len(), 5);
        let cache = scan
            .looked_in
            .iter()
            .find(|l| l.provider == Provider::LlamaCpp)
            .expect("looked for llama.cpp's cache");
        assert!(!cache.found);

        // Catalog models first, then described, then unreadable. The two
        // catalog files are the same size, so their order between themselves
        // is by path and not worth pinning.
        let names: Vec<(Provider, &str)> = scan
            .files
            .iter()
            .map(|f| (f.provider, f.name.as_str()))
            .collect();
        let mut leading = names[..2].to_vec();
        leading.sort();
        assert_eq!(
            leading,
            vec![
                (Provider::WhatRunsHere, "moved.gguf"),
                (Provider::Ollama, "test:8b")
            ]
        );
        assert_eq!(
            &names[2..],
            &[
                (Provider::LmStudio, "other-Q8_0.gguf"),
                (Provider::HuggingFace, "not-really.gguf"),
            ]
        );

        for file in &scan.files[..2] {
            assert!(
                matches!(
                    &file.identity,
                    Identity::Catalog { id, quant, .. } if id == "Test/Test-8B" && quant == "Q4_K_M"
                ),
                "{file:?}"
            );
        }
        assert!(matches!(
            &scan.files[2].identity,
            Identity::Header(h) if h.name.as_deref() == Some("Other 3B") && h.architecture.as_deref() == Some("llama")
        ));
        assert!(matches!(&scan.files[3].identity, Identity::Unknown { .. }));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_file_reachable_twice_is_listed_once() {
        let root = sandbox();
        let catalog = catalog();
        let dir = root.join("models");
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join("a.gguf"), vec![0u8; 4_444]).expect("file");

        // The same directory offered as two providers' roots.
        let roots = Roots {
            whatrunshere: Some(dir.clone()),
            lm_studio: Some(dir),
            ..Roots::default()
        };
        let scan = scan_roots(&roots, &catalog);
        assert_eq!(scan.files.len(), 1, "{:?}", scan.files);
        assert_eq!(scan.files[0].provider, Provider::WhatRunsHere);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn nothing_configured_is_nothing_looked_for() {
        let scan = scan_roots(&Roots::default(), &catalog());
        assert!(scan.looked_in.is_empty());
        assert!(scan.files.is_empty());
    }
}
