//! The `WhatLLM` desktop application.
//!
//! A window over the same engine the command line drives, and nothing more.
//! Every answer this application gives is computed by `whatllm-core` from the
//! machine `whatllm-hw` detected and the bandwidth `whatllm-probe` measured;
//! this crate adds no modelling of its own, because a second implementation of
//! any of it would be a second thing to keep correct.
//!
//! It makes no network calls. Neither does the webview: the content security
//! policy in `tauri.conf.json` permits only the bundled assets and the IPC
//! channel, so the promise the engine makes is enforced rather than asserted.

#![forbid(unsafe_code)]
// The desktop build must not open a console window behind the application.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use whatllm_app::api;

fn main() {
    tauri::Builder::default()
        .manage(api::Engine::new())
        .invoke_handler(tauri::generate_handler![
            api::machine,
            api::catalog,
            api::rank,
            api::plan,
            api::cost,
            api::measure,
        ])
        .run(tauri::generate_context!())
        .expect("the application window could not be created");
}
