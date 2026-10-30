//! Speech recognition on the ggml library of the pinned llama.cpp release.
//! `model` reads the Canary GGUF description, checks its tensor contract
//! and uploads the weights onto a `mediagen::backend::Backend`.

pub mod model;
