//! Where the catalog and the calibration come from.
//!
//! Shared rather than duplicated per front end, and deliberately so: the
//! calibration cache is one file that `whatllm probe` writes and the desktop
//! application reads. Two copies of this that drifted apart would leave a
//! machine measured by one and unmeasured by the other, with nothing to say
//! why.
//!
//! The catalog is looked for on disk before the copy compiled into the binary.
//! That ordering is deliberate: a tool whose model list only updates when you
//! reinstall it is out of date the week after it ships, and the models people
//! ask about are the ones released since.
//!
//! The calibration is cached per machine, keyed on a fingerprint, so plugging a
//! disk into a different computer cannot silently reuse the wrong measurement.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use whatllm_core::hardware::SystemProfile;
use whatllm_core::model::Catalog;

/// The catalog compiled into this binary, used when no newer one is on disk.
const EMBEDDED_CATALOG: &str = include_str!("../../../catalog/catalog.json");

/// Where the catalog came from, so a report can say so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogSource {
    /// Read from a file.
    File(PathBuf),
    /// The copy built into this binary.
    Embedded,
}

impl std::fmt::Display for CatalogSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(path) => write!(f, "{}", path.display()),
            Self::Embedded => write!(f, "built in"),
        }
    }
}

/// This user's `WhatLLM` directory, if a home directory can be found.
pub fn state_dir() -> Option<PathBuf> {
    let home = std::env::var_os("WHATLLM_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))?;
    Some(if std::env::var_os("WHATLLM_HOME").is_some() {
        home
    } else {
        home.join(".whatllm")
    })
}

/// Load the catalog, preferring the freshest source available.
///
/// # Errors
/// Fails only when an explicitly requested file cannot be read or parsed. A
/// missing optional source falls through to the next one.
pub fn load_catalog(explicit: Option<&Path>) -> Result<(Catalog, CatalogSource)> {
    if let Some(path) = explicit {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading catalog from {}", path.display()))?;
        let catalog: Catalog = serde_json::from_str(&text)
            .with_context(|| format!("parsing catalog at {}", path.display()))?;
        return Ok((catalog, CatalogSource::File(path.to_path_buf())));
    }

    let mut catalog: Catalog =
        serde_json::from_str(EMBEDDED_CATALOG).context("parsing the built-in catalog")?;

    // A catalog the user has is merged over the one that shipped, not
    // substituted for it. Replacing would mean that adding one model by hand
    // silently removes every model the binary already knew, and that a
    // half-finished update leaves the tool knowing less than it did.
    if let Some(path) = state_dir().map(|dir| dir.join("catalog.json")) {
        if let Ok(text) = std::fs::read_to_string(&path) {
            // A newer schema than this binary understands, or a damaged file:
            // carry on with what shipped rather than fail.
            if let Ok(user) = serde_json::from_str::<Catalog>(&text) {
                if user.is_supported() && merge_into(&mut catalog, user) > 0 {
                    return Ok((catalog, CatalogSource::File(path)));
                }
            }
        }
    }

    Ok((catalog, CatalogSource::Embedded))
}

/// Fold `user` into `base`, preferring the user's entry where the ids collide.
///
/// Returns how many entries the user's file contributed, which is zero when it
/// says nothing the binary did not already know.
fn merge_into(base: &mut Catalog, user: Catalog) -> usize {
    let mut contributed = 0;
    for entry in user.models {
        if let Some(existing) = base.models.iter_mut().find(|m| m.id == entry.id) {
            if *existing != entry {
                *existing = entry;
                contributed += 1;
            }
        } else {
            base.models.push(entry);
            contributed += 1;
        }
    }
    base.models.sort_by(|a, b| a.id.cmp(&b.id));
    contributed
}

/// A measurement of this machine, and enough to know it is still this machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedCalibration {
    /// Identifies the machine the measurement was taken on.
    pub fingerprint: String,
    /// Sustained host read bandwidth, in bytes per second.
    pub host_bytes_per_s: f64,
    /// Single-threaded host read bandwidth.
    pub single_thread_bytes_per_s: f64,
    /// Seconds since the Unix epoch.
    pub measured_at: u64,
}

/// Identify a machine well enough to notice when it is a different one.
pub fn fingerprint(system: &SystemProfile) -> String {
    let accelerators: Vec<&str> = system
        .accelerators
        .iter()
        .map(|a| a.name.as_str())
        .collect();
    format!(
        "{}|{}c|{}|{}",
        system.cpu.brand,
        system.cpu.logical_cores,
        system.memory.total_bytes,
        accelerators.join(",")
    )
}

/// Seconds since the Unix epoch, or zero if the clock is before it.
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Read the cached calibration, if it belongs to this machine.
pub fn load_calibration(system: &SystemProfile) -> Option<CachedCalibration> {
    let path = state_dir()?.join("calibration.json");
    let text = std::fs::read_to_string(path).ok()?;
    let cached: CachedCalibration = serde_json::from_str(&text).ok()?;
    (cached.fingerprint == fingerprint(system)).then_some(cached)
}

/// Write a calibration for this machine.
///
/// # Errors
/// Fails when the state directory cannot be created or written.
pub fn save_calibration(
    system: &SystemProfile,
    host_bytes_per_s: f64,
    single_thread_bytes_per_s: f64,
) -> Result<PathBuf> {
    let dir = state_dir().context("no home directory to store the calibration in")?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join("calibration.json");
    let cached = CachedCalibration {
        fingerprint: fingerprint(system),
        host_bytes_per_s,
        single_thread_bytes_per_s,
        measured_at: now(),
    };
    std::fs::write(&path, serde_json::to_string_pretty(&cached)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Describe how long ago something happened, in words.
pub fn age(seconds_since_epoch: u64) -> String {
    let elapsed = now().saturating_sub(seconds_since_epoch);
    match elapsed {
        0..=90 => "just now".to_owned(),
        91..=5399 => format!("{} minutes ago", elapsed / 60),
        5400..=169_199 => format!("{} hours ago", elapsed / 3600),
        _ => format!("{} days ago", elapsed / 86400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_catalog_parses_and_is_not_empty() {
        let (catalog, source) = load_catalog(None).expect("a catalog is always available");
        assert!(catalog.is_supported());
        assert!(
            !catalog.models.is_empty(),
            "the catalog shipped with no models"
        );
        // The state directory may hold a newer one; either source is valid.
        assert!(matches!(
            source,
            CatalogSource::Embedded | CatalogSource::File(_)
        ));
    }

    #[test]
    fn a_user_catalog_adds_to_the_built_in_one_rather_than_replacing_it() {
        let (mut base, _) = load_catalog(None).expect("a catalog is always available");
        let shipped = base.models.len();
        assert!(
            shipped > 1,
            "this test needs a catalog with something in it"
        );

        let mut newcomer = base.models[0].clone();
        newcomer.id = "someone/brand-new-model".to_owned();
        let user = Catalog {
            version: base.version,
            generated: "2026-09-10".to_owned(),
            models: vec![newcomer],
        };

        let added = merge_into(&mut base, user);
        assert_eq!(added, 1);
        assert_eq!(
            base.models.len(),
            shipped + 1,
            "adding one model must not remove the others"
        );
        assert!(base.find("someone/brand-new-model").is_some());
    }

    #[test]
    fn a_user_entry_wins_where_the_ids_collide() {
        let (mut base, _) = load_catalog(None).expect("a catalog is always available");
        let mut revised = base.models[0].clone();
        revised.display_name = "Renamed by the user".to_owned();
        let id = revised.id.clone();

        let version = base.version;
        let added = merge_into(
            &mut base,
            Catalog {
                version,
                generated: "2026-09-10".to_owned(),
                models: vec![revised],
            },
        );
        assert_eq!(added, 1);
        assert_eq!(
            base.find(&id).map(|m| m.display_name.as_str()),
            Some("Renamed by the user")
        );
    }

    #[test]
    fn an_explicitly_named_missing_catalog_is_an_error_not_a_fallback() {
        let missing = Path::new("does-not-exist-catalog.json");
        assert!(
            load_catalog(Some(missing)).is_err(),
            "asking for a specific file and silently getting another is worse \
             than failing"
        );
    }

    #[test]
    fn a_fingerprint_changes_with_the_machine() {
        let detection = whatllm_hw::detect();
        let mine = fingerprint(&detection.system);
        assert!(!mine.is_empty());

        let mut other = detection.system.clone();
        other.memory.total_bytes += 1;
        assert_ne!(mine, fingerprint(&other), "more RAM is a different machine");
    }

    #[test]
    fn ages_are_described_in_the_largest_useful_unit() {
        let now = now();
        assert_eq!(age(now), "just now");
        assert!(age(now.saturating_sub(600)).contains("minutes"));
        assert!(age(now.saturating_sub(7200)).contains("hours"));
        assert!(age(now.saturating_sub(400_000)).contains("days"));
    }
}
