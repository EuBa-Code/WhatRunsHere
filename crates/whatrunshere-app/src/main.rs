//! The `WhatRunsHere` desktop application.
//!
//! A window over the same engine the command line drives, and nothing more.
//! Every answer this application gives is computed by `whatrunshere-core` from the
//! machine `whatrunshere-hw` detected and the bandwidth `whatrunshere-probe` measured;
//! this crate adds no modelling of its own, because a second implementation of
//! any of it would be a second thing to keep correct.
//!
//! Nothing here reaches the network except [`download`], and that only when
//! somebody presses the button. The webview cannot reach it at all: the content
//! security policy in `tauri.conf.json` permits the bundled assets and the IPC
//! channel and nothing else, so a request can only originate in this process,
//! from the one module that makes them. The boundary is enforced rather than
//! asserted, which is the only kind of promise worth making about it.

#![forbid(unsafe_code)]
// The desktop build must not open a console window behind the application.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use whatrunshere_app::api;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(api::Engine::new())
        .invoke_handler(tauri::generate_handler![
            api::machine,
            api::catalog,
            api::rank,
            api::plan,
            api::cost,
            api::measure,
            api::destination,
            api::download_build,
            api::pause_download,
            api::cancel_download,
            api::downloads,
            api::reveal,
            api::save_image,
            api::volumes,
            api::set_download_dir,
            api::launch,
            api::save_launch_file,
            api::installed,
        ])
        .run(tauri::generate_context!())
        .expect("the application window could not be created");
}
