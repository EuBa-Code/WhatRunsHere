//! Generates the Tauri context: the window configuration, the capability set
//! and, on Windows, the resource file carrying the application icon.
//!
//! One thing about `tauri.conf.json` is worth knowing before editing it, and
//! cannot be written down there because its schema rejects comment keys.
//!
//! `beforeDevCommand` and `beforeBuildCommand` run in the **parent** of the
//! directory holding that file, here `crates/`. Tauri's standard layout puts
//! `src-tauri/` inside the front-end project, so that parent is the front-end
//! root; this layout keeps the crate beside its siblings instead, and the two
//! commands are therefore one level shallower than they look: `../ui`, not
//! `../../ui`. `frontendDist` in the same block is resolved against the file
//! itself and does need `../../ui/dist`, so the two paths differ on purpose.

fn main() {
    tauri_build::build();
}
