//! Speech recognition on the ggml library of the pinned llama.cpp release.
//! `model` reads the Canary GGUF description, checks its tensor contract
//! and uploads the weights onto a `mediagen::backend::Backend`. `wav`
//! and `mel` turn PCM16 WAV input into the log-mel features the encoder
//! reads. `encoder` builds the FastConformer encoder as a ggml graph on
//! that backend, with the host-side arithmetic it consumes: subsampled
//! frame count, relative-position table and the folded BatchNorm.

pub mod encoder;
pub mod mel;
pub mod model;
pub mod wav;
