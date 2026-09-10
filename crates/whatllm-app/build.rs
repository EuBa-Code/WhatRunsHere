//! Generates the Tauri context: the window configuration, the capability set
//! and, on Windows, the resource file carrying the application icon.

fn main() {
    tauri_build::build();
}
