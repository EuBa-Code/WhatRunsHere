//! Fetching a model's weights.
//!
//! This is the one place in `WhatLLM` that touches the network, and it does so
//! only when somebody presses the button. The window itself still cannot: its
//! content security policy admits the bundled assets and the IPC channel and
//! nothing else, so a request can only originate here, in the Rust process,
//! against an address the catalog recorded.
//!
//! Three things are done carefully because a model file is large enough that
//! getting them wrong costs an afternoon rather than a moment.
//!
//! **A partial file is never mistaken for a whole one.** Bytes land in a
//! `.part` beside the destination and are renamed into place only after the
//! length is checked. A download interrupted by a closed lid, a dropped
//! connection or a cancelled task leaves a `.part`, which the next attempt
//! continues from with a range request rather than starting again.
//!
//! **The length is checked against the catalog, not against the server.** The
//! catalog records the exact byte count of every published build, measured
//! when it was built — see `size_validation`. A truncated transfer that ends
//! cleanly is otherwise indistinguishable from a complete one, and a 19 GB
//! file that is wrong in its last megabyte fails at load time with a message
//! about tensors.
//!
//! **A filename out of the catalog is not trusted as a path.** The catalog is
//! assembled from metadata written by strangers, and a name carrying `..` or a
//! separator would place the write somewhere nobody asked for. Anything but a
//! plain `.gguf` filename is refused.

use futures_util::StreamExt;
use serde::Serialize;
use std::collections::HashMap;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};

/// How often progress is reported. Fast enough to look continuous, rare enough
/// that a 19 GB transfer does not spend its time serialising JSON.
const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_millis(120);

/// The event a running transfer reports itself on.
pub const PROGRESS_EVENT: &str = "download:progress";

/// Where a transfer has got to.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Progress {
    /// Connecting, or waiting for the server to answer.
    Starting {
        /// Which build, as `<model id>@<quant>`.
        key: String,
    },
    /// Bytes are arriving.
    Running {
        /// Which build.
        key: String,
        /// Bytes on disk so far, resumed bytes included.
        received: u64,
        /// What the catalog says the finished file weighs.
        total: u64,
        /// Recent transfer rate, in bytes per second.
        bytes_per_s: f64,
    },
    /// Stopped, and resumable from where it stopped.
    Paused {
        /// Which build.
        key: String,
        /// Bytes on disk.
        received: u64,
        /// What the catalog says the finished file weighs.
        total: u64,
    },
    /// On disk, at the recorded length.
    Done {
        /// Which build.
        key: String,
        /// Where it was written.
        path: String,
        /// How many bytes that is.
        bytes: u64,
    },
    /// Stopped and the partial file removed.
    Cancelled {
        /// Which build.
        key: String,
    },
    /// Something went wrong, said plainly.
    Failed {
        /// Which build.
        key: String,
        /// What happened, in words a person can act on.
        reason: String,
    },
}

impl Progress {
    fn key(&self) -> &str {
        match self {
            Self::Starting { key }
            | Self::Running { key, .. }
            | Self::Paused { key, .. }
            | Self::Done { key, .. }
            | Self::Cancelled { key }
            | Self::Failed { key, .. } => key,
        }
    }
}

/// Why a transfer stopped before it finished.
enum Stopped {
    /// Asked to pause, and the partial file is kept.
    Paused,
    /// Asked to cancel, and the partial file is removed.
    Cancelled,
}

/// What a running transfer can be told.
#[derive(Default)]
struct Signals {
    pause: AtomicBool,
    cancel: AtomicBool,
}

/// Every transfer this session has started, by build key.
#[derive(Default)]
pub struct Downloads {
    running: Mutex<HashMap<String, Arc<Signals>>>,
    /// The last thing each transfer said, so a window that was on another view
    /// when it finished is not left showing a progress bar that stopped
    /// moving. Events are fire-and-forget; this is what makes them reliable.
    latest: Mutex<HashMap<String, Progress>>,
}

impl Downloads {
    /// Everything each transfer last reported.
    ///
    /// # Panics
    /// If a panic elsewhere poisoned the table, which would mean the state of
    /// every running transfer is unknown.
    pub fn snapshot(&self) -> Vec<Progress> {
        self.latest
            .lock()
            .expect("the download table was poisoned by a panic elsewhere")
            .values()
            .cloned()
            .collect()
    }

    fn remember(&self, progress: &Progress) {
        self.latest
            .lock()
            .expect("the download table was poisoned by a panic elsewhere")
            .insert(progress.key().to_owned(), progress.clone());
    }

    fn signals(&self, key: &str) -> Option<Arc<Signals>> {
        self.running
            .lock()
            .expect("the download table was poisoned by a panic elsewhere")
            .get(key)
            .cloned()
    }

    /// Register a transfer, unless one is already running for this build.
    fn claim(&self, key: &str) -> Option<Arc<Signals>> {
        let mut running = self
            .running
            .lock()
            .expect("the download table was poisoned by a panic elsewhere");
        if running.contains_key(key) {
            return None;
        }
        let signals = Arc::new(Signals::default());
        running.insert(key.to_owned(), Arc::clone(&signals));
        Some(signals)
    }

    fn release(&self, key: &str) {
        self.running
            .lock()
            .expect("the download table was poisoned by a panic elsewhere")
            .remove(key);
    }
}

/// A filename from the catalog, checked before it is used as one.
///
/// The catalog is built from repository metadata written by strangers. A name
/// carrying a separator or a parent reference would write outside the
/// directory it was given, so anything that is not a plain GGUF filename is
/// refused rather than repaired — a sanitised version of a hostile name is
/// still a name nobody chose.
fn safe_filename(name: &str) -> Result<&str, String> {
    let plain = !name.is_empty()
        && name.len() <= 255
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && !name.starts_with('.')
        && !name.contains("..")
        && name.to_ascii_lowercase().ends_with(".gguf");

    if plain {
        Ok(name)
    } else {
        Err(format!(
            "the catalog gives this build the filename {name:?}, which is not a plain \
             .gguf name. It has not been written anywhere."
        ))
    }
}

/// The address a build is fetched from.
///
/// `resolve` follows `HuggingFace`'s pointer to the storage the file actually
/// lives on, which is what any client has to do for a file kept in LFS.
fn source_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}?download=true")
}

/// Where models are kept.
///
/// The platform's download directory, because a nineteen-gigabyte file belongs
/// somewhere a person can find without being told where to look. A subdirectory
/// keeps a catalog of them from burying everything else that lands there.
///
/// # Errors
/// When the system reports neither a download directory nor a home directory.
pub fn destination_dir() -> Result<PathBuf, String> {
    let base = dirs::download_dir()
        .or_else(dirs::home_dir)
        .ok_or_else(|| "no download or home directory on this system".to_owned())?;
    Ok(base.join("WhatLLM Models"))
}

/// Begin, or continue, fetching one build.
pub async fn run(
    app: AppHandle,
    downloads: Arc<Downloads>,
    key: String,
    repo: String,
    file: String,
    expected_bytes: u64,
) {
    let Some(signals) = downloads.claim(&key) else {
        // Already running. The window is showing its progress already.
        return;
    };

    let outcome = transfer(
        &app,
        &downloads,
        &signals,
        &key,
        &repo,
        &file,
        expected_bytes,
    )
    .await;
    downloads.release(&key);

    let final_state = match outcome {
        Ok(path) => Progress::Done {
            key: key.clone(),
            path: path.display().to_string(),
            bytes: expected_bytes,
        },
        Err(TransferError::Stopped(Stopped::Paused, received)) => Progress::Paused {
            key: key.clone(),
            received,
            total: expected_bytes,
        },
        Err(TransferError::Stopped(Stopped::Cancelled, _)) => {
            Progress::Cancelled { key: key.clone() }
        }
        Err(TransferError::Failed(reason)) => Progress::Failed {
            key: key.clone(),
            reason,
        },
    };
    report(&app, &downloads, &final_state);
}

enum TransferError {
    Stopped(Stopped, u64),
    Failed(String),
}

impl From<String> for TransferError {
    fn from(reason: String) -> Self {
        Self::Failed(reason)
    }
}

fn report(app: &AppHandle, downloads: &Downloads, progress: &Progress) {
    downloads.remember(progress);
    // A window that has gone away is not a failure worth reporting to itself.
    let _ = app.emit(PROGRESS_EVENT, progress);
}

#[allow(clippy::too_many_lines)]
async fn transfer(
    app: &AppHandle,
    downloads: &Downloads,
    signals: &Signals,
    key: &str,
    repo: &str,
    file: &str,
    expected_bytes: u64,
) -> Result<PathBuf, TransferError> {
    report(
        app,
        downloads,
        &Progress::Starting {
            key: key.to_owned(),
        },
    );

    let name = safe_filename(file)?;
    let dir = destination_dir()?;
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("could not create {}: {e}", dir.display()))?;

    let target = dir.join(name);
    let partial = dir.join(format!("{name}.part"));

    // A file already there at the right length is the answer; nothing needs
    // fetching. At the wrong length it is something else with the same name,
    // and overwriting it silently would be worse than saying so.
    if let Ok(meta) = tokio::fs::metadata(&target).await {
        if meta.len() == expected_bytes {
            return Ok(target);
        }
        return Err(TransferError::Failed(format!(
            "{} already exists and is {} bytes, not the {expected_bytes} this build \
             should weigh. Move or delete it and try again.",
            target.display(),
            meta.len()
        )));
    }

    // Whatever a previous attempt left behind, continued from. A `.part` at
    // or past the full length is not a partial anything, and neither is one
    // that is no longer there; both start again.
    let resume_from = tokio::fs::metadata(&partial)
        .await
        .map_or(0, |meta| meta.len());
    let resume_from = if resume_from < expected_bytes {
        resume_from
    } else {
        0
    };

    let client = reqwest::Client::builder()
        // A stalled connection should fail rather than hold a progress bar at
        // the same number for an hour.
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(concat!("whatllm/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("could not start an HTTP client: {e}"))?;

    let mut request = client.get(source_url(repo, file));
    if resume_from > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={resume_from}-"));
    }

    let response = request
        .send()
        .await
        .map_err(|e| format!("could not reach huggingface.co: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        return Err(TransferError::Failed(match status.as_u16() {
            401 | 403 => format!(
                "huggingface.co refused the request for {repo} ({status}). This model's \
                 repository asks you to accept its licence before downloading; opening \
                 it in a browser and accepting will let this through."
            ),
            404 => format!(
                "huggingface.co has no file {file} in {repo} ({status}). The catalog \
                 entry is out of date, or the publisher moved it."
            ),
            _ => format!("huggingface.co answered {status} for {repo}"),
        }));
    }

    // A server that ignored the range header starts again from zero, and
    // appending to the partial file would interleave two copies.
    let appending = resume_from > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
    let mut received = if appending { resume_from } else { 0 };

    let mut handle = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(!appending)
        .open(&partial)
        .await
        .map_err(|e| format!("could not open {}: {e}", partial.display()))?;
    if appending {
        handle
            .seek(SeekFrom::Start(resume_from))
            .await
            .map_err(|e| format!("could not continue {}: {e}", partial.display()))?;
    }

    let mut stream = response.bytes_stream();
    let mut last_report = std::time::Instant::now();
    let mut window_start = last_report;
    let mut window_bytes = 0u64;

    while let Some(chunk) = stream.next().await {
        if signals.cancel.load(Ordering::Relaxed) {
            drop(handle);
            let _ = tokio::fs::remove_file(&partial).await;
            return Err(TransferError::Stopped(Stopped::Cancelled, received));
        }
        if signals.pause.load(Ordering::Relaxed) {
            // Flushed before returning, so the bytes already accepted are on
            // disk and the next attempt resumes from them rather than from the
            // last place the operating system happened to write.
            let _ = handle.flush().await;
            return Err(TransferError::Stopped(Stopped::Paused, received));
        }

        let chunk = chunk.map_err(|e| format!("the transfer was interrupted: {e}"))?;
        handle
            .write_all(&chunk)
            .await
            .map_err(|e| format!("could not write to {}: {e}", partial.display()))?;
        received += chunk.len() as u64;
        window_bytes += chunk.len() as u64;

        if last_report.elapsed() >= PROGRESS_INTERVAL {
            let seconds = window_start.elapsed().as_secs_f64();
            report(
                app,
                downloads,
                &Progress::Running {
                    key: key.to_owned(),
                    received,
                    total: expected_bytes,
                    bytes_per_s: if seconds > 0.0 {
                        window_bytes as f64 / seconds
                    } else {
                        0.0
                    },
                },
            );
            last_report = std::time::Instant::now();
            window_start = last_report;
            window_bytes = 0;
        }
    }

    handle
        .flush()
        .await
        .map_err(|e| format!("could not finish writing {}: {e}", partial.display()))?;
    drop(handle);

    // The check the catalog makes possible. A transfer that ends cleanly but
    // short is otherwise indistinguishable from one that worked, and the
    // failure surfaces much later as a complaint about tensors.
    let written = tokio::fs::metadata(&partial)
        .await
        .map_err(|e| format!("could not measure {}: {e}", partial.display()))?
        .len();
    if written != expected_bytes {
        let _ = tokio::fs::remove_file(&partial).await;
        return Err(TransferError::Failed(format!(
            "the transfer finished at {written} bytes where the catalog records \
             {expected_bytes}. The partial file has been removed rather than left to \
             look complete."
        )));
    }

    tokio::fs::rename(&partial, &target)
        .await
        .map_err(|e| format!("could not move the finished file into place: {e}"))?;
    Ok(target)
}

/// Ask a running transfer to stop, keeping what it has.
pub fn pause(downloads: &Downloads, key: &str) {
    if let Some(signals) = downloads.signals(key) {
        signals.pause.store(true, Ordering::Relaxed);
    }
}

/// Ask a running transfer to stop and discard what it has.
pub fn cancel(downloads: &Downloads, key: &str) {
    if let Some(signals) = downloads.signals(key) {
        signals.cancel.store(true, Ordering::Relaxed);
    }
}

/// Show a finished file where it landed.
///
/// The command per platform is fixed here and the path is one this process
/// wrote, so nothing a catalog could contain reaches a shell.
///
/// # Errors
/// When the file is gone, or no file manager could be started.
pub fn reveal(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("{} is not there any more", path.display()));
    }
    #[cfg(windows)]
    let result = std::process::Command::new("explorer")
        .arg("/select,")
        .arg(path)
        .spawn();
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open")
        .arg("-R")
        .arg(path)
        .spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let result = std::process::Command::new("xdg-open")
        .arg(path.parent().unwrap_or(path))
        .spawn();

    result
        .map(|_| ())
        .map_err(|e| format!("could not open a file manager: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_filename_from_the_catalog_is_not_trusted_as_a_path() {
        // The names the catalog actually carries.
        assert!(safe_filename("Qwen_Qwen3-30B-A3B-Q4_K_M.gguf").is_ok());
        assert!(safe_filename("gemma-3-27b-it-Q5_K_M.gguf").is_ok());

        // Anything that would write outside the directory it was given.
        for hostile in [
            "../../../etc/passwd.gguf",
            "..\\..\\windows\\system32\\evil.gguf",
            "/absolute/path.gguf",
            "sub/dir/model.gguf",
            "..gguf",
            ".hidden.gguf",
        ] {
            assert!(
                safe_filename(hostile).is_err(),
                "{hostile} should have been refused"
            );
        }

        // Anything that is not a model file.
        for wrong in ["model.exe", "model.gguf.exe", "README.md", ""] {
            assert!(safe_filename(wrong).is_err(), "{wrong} is not a build");
        }
    }

    #[test]
    fn the_source_address_is_the_repository_the_catalog_recorded() {
        assert_eq!(
            source_url(
                "bartowski/Qwen_Qwen3-30B-A3B-GGUF",
                "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
            ),
            "https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/resolve/main/\
             Qwen_Qwen3-30B-A3B-Q4_K_M.gguf?download=true"
        );
    }

    #[test]
    fn a_second_request_for_a_running_transfer_is_refused() {
        let downloads = Downloads::default();
        assert!(downloads.claim("a@Q4_K_M").is_some());
        assert!(
            downloads.claim("a@Q4_K_M").is_none(),
            "one build should not be fetched twice at once"
        );
        // A different build is unaffected.
        assert!(downloads.claim("a@Q5_K_M").is_some());

        downloads.release("a@Q4_K_M");
        assert!(
            downloads.claim("a@Q4_K_M").is_some(),
            "released, so claimable"
        );
    }

    #[test]
    fn the_last_word_of_each_transfer_is_kept_for_a_window_that_was_elsewhere() {
        let downloads = Downloads::default();
        downloads.remember(&Progress::Running {
            key: "a".into(),
            received: 10,
            total: 100,
            bytes_per_s: 5.0,
        });
        downloads.remember(&Progress::Done {
            key: "a".into(),
            path: "/tmp/a.gguf".into(),
            bytes: 100,
        });
        downloads.remember(&Progress::Running {
            key: "b".into(),
            received: 1,
            total: 2,
            bytes_per_s: 1.0,
        });

        let snapshot = downloads.snapshot();
        assert_eq!(snapshot.len(), 2, "one entry per build, not per event");
        assert!(
            snapshot
                .iter()
                .any(|p| matches!(p, Progress::Done { key, .. } if key == "a")),
            "the latest word about a build replaces the earlier one"
        );
    }
}
