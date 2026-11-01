//! Speech recognition on the ggml library of the pinned llama.cpp release.
//! `model` reads the Canary GGUF description, checks its tensor contract
//! and uploads the weights onto a `mediagen::backend::Backend`. `wav`
//! and `mel` turn PCM16 WAV input into the log-mel features the encoder
//! reads. `encoder` builds the FastConformer encoder as a ggml graph on
//! that backend, with the host-side arithmetic it consumes: subsampled
//! frame count, relative-position table and the folded BatchNorm.
//! `prompt` builds the canary2 decoder prompt from the hyperparameters
//! and the vocabulary; `decoder` runs the Transformer decoder over the
//! encoder output with a KV cache; `Vocab::detokenize` turns decoder
//! output back into text. `server` is the worker process's HTTP side:
//! it parses and bounds a transcription request, runs that sequence
//! one request at a time and answers in the OpenAI shapes.

pub mod decoder;
pub mod encoder;
pub mod mel;
pub mod model;
pub mod prompt;
pub mod server;
pub mod wav;
