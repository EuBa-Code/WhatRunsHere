//! Where a model's weights go on this disk.
//!
//! Shared by the window, which writes the file, and the command line, which
//! prints the commands that read it. Two answers to "where is the file" would
//! be one too many: a command naming a directory the window never wrote to is
//! a command that does not run.
//!
//! The answer depends on who will read the file. llama.cpp and Ollama read
//! from wherever it is, so it goes to the chosen directory, or to the
//! platform's download folder until one is chosen. LM Studio reads only its
//! own folder, so the file goes straight there, under the publisher and
//! repository it came from, and appears in LM Studio's list with nothing to
//! move. Downloading to the chosen directory and offering to copy would double
//! a twenty-gigabyte file for no reason.

use std::path::PathBuf;
use whatllm_core::launch::Host;
use whatllm_core::model::GgufBuild;

/// Where models are kept when nothing about the host says otherwise.
///
/// A chosen directory wins. Otherwise the platform's download directory, in a
/// subdirectory of its own, because a nineteen-gigabyte file belongs somewhere
/// a person can find without being told where to look, and a catalog of them
/// should not bury everything else that lands there.
///
/// The choice matters more than most settings: a model is tens of gigabytes
/// and the system disk is frequently the small fast one, so anybody with a
/// second drive will want to say so.
///
/// # Errors
/// When nothing has been chosen and the system reports neither a download
/// directory nor a home directory.
pub fn download_dir() -> Result<PathBuf, String> {
    if let Some(chosen) = crate::settings::load().download_dir {
        return Ok(chosen);
    }
    let base = dirs::download_dir()
        .or_else(dirs::home_dir)
        .ok_or_else(|| "no download or home directory on this system".to_owned())?;
    Ok(base.join("WhatLLM Models"))
}

/// Where LM Studio keeps its models, whether or not it is installed.
///
/// `~/.lmstudio/models` on every platform, from LM Studio's own documentation.
/// Whether the directory exists is for the caller to check: its absence is
/// the common case and means LM Studio is not installed, which is not an
/// error.
///
/// `None` only when there is no home directory to look under.
pub fn lm_studio_models_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".lmstudio").join("models"))
}

/// Where one build goes for one host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    /// The directory the file lands in.
    pub directory: PathBuf,
    /// The host's own model folder, when `directory` sits inside it.
    pub host_tree: Option<PathBuf>,
    /// Where the host's folder was looked for and not found.
    pub host_tree_missing: Option<PathBuf>,
}

/// Decide where a build goes for the program that will read it.
///
/// When LM Studio's folder exists the file goes into it, under the publisher
/// and repository the catalog recorded. When it does not, LM Studio is most
/// likely not installed: the file goes where every other host's would, and
/// the caller is told where the folder was looked for so it can say so.
///
/// # Errors
/// When the repository name is not a plain `publisher/repository`, or there
/// is no directory to fall back to.
pub fn target(host: Host, build: &GgufBuild) -> Result<Target, String> {
    if host == Host::LmStudio {
        if let Some(tree) = lm_studio_models_dir() {
            if tree.is_dir() {
                let (publisher, repository) = safe_repo(&build.repo)?;
                return Ok(Target {
                    directory: tree.join(publisher).join(repository),
                    host_tree: Some(tree),
                    host_tree_missing: None,
                });
            }
            return Ok(Target {
                directory: download_dir()?,
                host_tree: None,
                host_tree_missing: Some(tree),
            });
        }
    }
    Ok(Target {
        directory: download_dir()?,
        host_tree: None,
        host_tree_missing: None,
    })
}

/// A repository name from the catalog, checked before it becomes two
/// directories under LM Studio's folder.
///
/// The catalog is assembled from metadata written by strangers. A name
/// carrying `..` or a second separator would place the write somewhere nobody
/// asked for, so anything but `publisher/repository`, each side a plain name,
/// is refused rather than repaired: a sanitised version of a hostile name is
/// still a name nobody chose.
///
/// # Errors
/// When the name is anything else.
pub fn safe_repo(repo: &str) -> Result<(&str, &str), String> {
    let refused = || {
        format!(
            "the catalog gives this build the repository {repo:?}, which is not a plain \
             publisher/repository name. It has not been written anywhere."
        )
    };
    let (publisher, repository) = repo.split_once('/').ok_or_else(refused)?;
    if plain_segment(publisher) && plain_segment(repository) {
        Ok((publisher, repository))
    } else {
        Err(refused())
    }
}

/// Whether a name can only ever refer to a file or directory inside the one
/// it is joined to: no separator, no parent reference, nothing hidden.
pub fn plain_segment(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && !name.starts_with('.')
        && !name.contains("..")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repository_name_becomes_two_directories_only_when_it_is_plain() {
        assert_eq!(
            safe_repo("bartowski/Qwen_Qwen3-30B-A3B-GGUF"),
            Ok(("bartowski", "Qwen_Qwen3-30B-A3B-GGUF"))
        );
        for hostile in [
            "../escape/repo",
            "bartowski/../../etc",
            "bartowski/sub/repo",
            "/repo",
            "bartowski/",
            "norepo",
            ".hidden/repo",
            "bartowski/repo\\x",
        ] {
            assert!(
                safe_repo(hostile).is_err(),
                "{hostile} should have been refused"
            );
        }
    }

    #[test]
    fn only_lm_studio_has_a_folder_of_its_own() {
        let build = GgufBuild {
            quant: "Q4_K_M".to_owned(),
            repo: "someone/test-GGUF".to_owned(),
            file: "test-Q4_K_M.gguf".to_owned(),
            bytes: 1,
        };
        for host in [Host::LlamaCpp, Host::Ollama, Host::Vllm, Host::Mlx] {
            if let Ok(target) = target(host, &build) {
                assert!(target.host_tree.is_none(), "{host:?} reads from anywhere");
                assert!(target.host_tree_missing.is_none());
            }
        }
        // LM Studio's answer depends on the machine the test runs on, but it
        // is always one of the two, never both and never neither.
        if let Ok(target) = target(Host::LmStudio, &build) {
            assert_ne!(
                target.host_tree.is_some(),
                target.host_tree_missing.is_some(),
                "either the folder was found or it was looked for: {target:?}"
            );
            if let Some(tree) = &target.host_tree {
                assert!(target.directory.starts_with(tree));
                assert!(target.directory.ends_with("someone/test-GGUF"));
            }
        }
    }
}
