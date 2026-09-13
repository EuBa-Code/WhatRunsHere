//! The `WhatRunsHere` desktop application, as a library.
//!
//! The window is a binary; this is the same crate's command surface, exposed
//! so it can be exercised without one. `tests/contract.rs` is the reason: the
//! TypeScript in `ui/src/engine.ts` describes these shapes by hand, and a
//! hand-written description of a shape nothing checks is a promise that breaks
//! silently.

#![forbid(unsafe_code)]

pub mod api;
pub mod download;
