//! Speech recognition on the ggml library of the pinned llama.cpp release.
//! `model` reads the Canary GGUF description, checks its tensor contract
//! and uploads the weights onto a `mediagen::backend::Backend`. `wav`
//! and `mel` turn PCM16 WAV input into the log-mel features the encoder
//! reads.

pub mod mel;
pub mod model;
pub mod wav;
