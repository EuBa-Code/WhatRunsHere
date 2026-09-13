//! Ollama's store.
//!
//! Ollama does not keep named files. It keeps a content-addressed blob store
//! and, beside it, a manifest per model tag that lists which blobs make the
//! model up. The weights are one of those blobs, and the blob is the GGUF
//! file itself, byte for byte, which is why its size can be matched against
//! the catalog like any other file's.
//!
//! ```text
//! <root>/manifests/registry.ollama.ai/library/qwen3/30b-a3b   (JSON)
//! <root>/blobs/sha256-<hex>                                    (the GGUF)
//! ```
//!
//! The manifest's path is the model's name: registry, namespace, name and
//! tag. `library` is the namespace Ollama's own models sit in and is dropped
//! from the name shown, because `ollama list` drops it too.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The layer type the weights are stored under.
const MODEL_LAYER: &str = "application/vnd.ollama.image.model";

/// One model in the store: its name as `ollama run` wants it, and its blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    /// `name:tag`, with the namespace in front when it is not `library`.
    pub name: String,
    /// The weights blob.
    pub blob: PathBuf,
}

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    layers: Vec<Layer>,
}

#[derive(Deserialize)]
struct Layer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
}

/// Every model whose manifest and weights blob are both present under `root`.
///
/// A manifest without its blob is a model Ollama has removed or not finished
/// pulling, and is left out rather than listed with a size of nothing.
pub fn models(root: &Path) -> Vec<Model> {
    let manifests = root.join("manifests");
    let blobs = root.join("blobs");
    let mut found = Vec::new();
    // registry / namespace / name / tag
    for registry in directories(&manifests) {
        for namespace in directories(&registry) {
            for name in directories(&namespace) {
                for tag in files(&name) {
                    if let Some(model) = read(&namespace, &name, &tag, &blobs) {
                        found.push(model);
                    }
                }
            }
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

fn read(namespace: &Path, name: &Path, tag: &Path, blobs: &Path) -> Option<Model> {
    let text = std::fs::read_to_string(tag).ok()?;
    let manifest: Manifest = serde_json::from_str(&text).ok()?;
    let digest = manifest
        .layers
        .iter()
        .find(|layer| layer.media_type == MODEL_LAYER)
        .map(|layer| layer.digest.as_str())?;
    // `sha256:<hex>` in the manifest, `sha256-<hex>` on disk.
    let (algorithm, hex) = digest.split_once(':')?;
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) || !algorithm.chars().all(char::is_alphanumeric)
    {
        return None;
    }
    let blob = blobs.join(format!("{algorithm}-{hex}"));
    if !blob.is_file() {
        return None;
    }

    let namespace = namespace.file_name()?.to_string_lossy();
    let name = name.file_name()?.to_string_lossy();
    let tag = tag.file_name()?.to_string_lossy();
    let shown = if namespace == "library" {
        format!("{name}:{tag}")
    } else {
        format!("{namespace}/{name}:{tag}")
    };
    Some(Model { name: shown, blob })
}

fn directories(path: &Path) -> Vec<PathBuf> {
    entries(path, true)
}

fn files(path: &Path) -> Vec<PathBuf> {
    entries(path, false)
}

fn entries(path: &Path, want_dirs: bool) -> Vec<PathBuf> {
    let Ok(read) = std::fs::read_dir(path) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = read
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| if want_dirs { p.is_dir() } else { p.is_file() })
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let root = std::env::temp_dir().join(format!("whatrunshere-ollama-{unique}"));
        std::fs::create_dir_all(root.join("blobs")).expect("temp store");
        root
    }

    fn manifest(root: &Path, namespace: &str, name: &str, tag: &str, digest: &str) {
        let dir = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join(namespace)
            .join(name);
        std::fs::create_dir_all(&dir).expect("manifest dir");
        std::fs::write(
            dir.join(tag),
            format!(
                r#"{{"schemaVersion":2,"layers":[
                    {{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:aaaa","size":10}},
                    {{"mediaType":"{MODEL_LAYER}","digest":"{digest}","size":1234}}]}}"#
            ),
        )
        .expect("manifest");
    }

    #[test]
    fn a_model_is_its_manifest_path_and_its_weights_blob() {
        let root = store();
        std::fs::write(root.join("blobs").join("sha256-ab12"), [0u8; 1234]).expect("blob");
        manifest(&root, "library", "qwen3", "30b-a3b", "sha256:ab12");
        manifest(&root, "someone", "custom", "latest", "sha256:ab12");
        // A manifest whose blob is not there: not a model.
        manifest(&root, "library", "gone", "latest", "sha256:ffff");
        // A digest that is not hex: not followed onto the disk.
        manifest(
            &root,
            "library",
            "hostile",
            "latest",
            "sha256:../../etc/passwd",
        );

        let found = models(&root);
        let names: Vec<&str> = found.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["qwen3:30b-a3b", "someone/custom:latest"]);
        assert!(found[0].blob.ends_with("sha256-ab12"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_empty_or_missing_store_is_no_models() {
        assert!(models(Path::new("/nowhere/that/exists")).is_empty());
        let root = store();
        assert!(models(&root).is_empty());
        let _ = std::fs::remove_dir_all(root);
    }
}
