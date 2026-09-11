//! Choices a person made, kept between runs.
//!
//! Separate from the calibration next door, which is a measurement rather than
//! a preference: a calibration is invalidated by changing machine, and a
//! preference is not. They are also written at different moments, and mixing
//! them would mean a probe could lose a setting by racing it.
//!
//! Everything here is optional and everything has a default, so a missing or
//! damaged file costs nothing: it is read as "nothing has been chosen yet",
//! which is the truth on a fresh installation anyway.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The file, inside the state directory.
const FILE: &str = "settings.json";

/// What a person has chosen.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// Where downloaded weights are written.
    ///
    /// `None` means nothing has been chosen and the platform's download
    /// directory is used. A model file is tens of gigabytes, so this is the
    /// setting that matters most on a machine with more than one disk: the
    /// system drive is frequently the small fast one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download_dir: Option<PathBuf>,
}

/// Read what has been chosen.
///
/// Never fails. A file that is missing, unreadable or no longer parses is the
/// same answer as a fresh installation, and refusing to start over a settings
/// file would be a poor trade.
pub fn load() -> Settings {
    let Some(path) = crate::state_dir().map(|dir| dir.join(FILE)) else {
        return Settings::default();
    };
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Write what has been chosen.
///
/// # Errors
/// When there is no home directory, or the file cannot be written.
pub fn save(settings: &Settings) -> Result<PathBuf, String> {
    let dir =
        crate::state_dir().ok_or_else(|| "no home directory to store settings in".to_owned())?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let path = dir.join(FILE);
    let text = serde_json::to_string_pretty(settings)
        .map_err(|e| format!("the settings did not serialise: {e}"))?;
    std::fs::write(&path, text).map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Both tests point `WHATLLM_HOME` somewhere of their own, and an
    /// environment variable belongs to the process rather than to a thread.
    /// Without this they pass alone and fail together, which is the worst
    /// shape a test can have.
    static ENV: Mutex<()> = Mutex::new(());

    #[test]
    fn a_missing_or_damaged_file_reads_as_a_fresh_installation() {
        let _guard = ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Pointed at a directory with no settings in it.
        let empty = tempdir();
        std::env::set_var("WHATLLM_HOME", &empty);
        assert_eq!(load(), Settings::default());

        // And at one holding something that is not settings at all.
        std::fs::write(empty.join(FILE), "this is not json").expect("write");
        assert_eq!(
            load(),
            Settings::default(),
            "a damaged settings file must not be an error, only an absence"
        );
        std::env::remove_var("WHATLLM_HOME");
    }

    #[test]
    fn a_chosen_directory_survives_the_round_trip() {
        let _guard = ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let home = tempdir();
        std::env::set_var("WHATLLM_HOME", &home);

        let chosen = PathBuf::from("D:\\Models");
        save(&Settings {
            download_dir: Some(chosen.clone()),
        })
        .expect("settings save");
        assert_eq!(load().download_dir, Some(chosen));

        std::env::remove_var("WHATLLM_HOME");
    }

    /// A directory of our own under the system temporary directory.
    fn tempdir() -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!("whatllm-settings-{unique}"));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }
}
