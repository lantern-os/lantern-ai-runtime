//! `lantern-ai-runtime` — LanternOS's local inference engine
//! ([RFC-0034](https://github.com/lantern-os/lantern-rfcs/blob/main/rfcs/0034-local-inference-and-a-file-assistant.md)).
//!
//! A Llama-family transformer (RMSNorm, rotary embeddings, SwiGLU, grouped-query attention) over int8 weights, after
//! Andrej Karpathy's llama2.c `runq.c` (MIT), in `no_std` Rust with no allocation: the inference service gives it the
//! model file and a workspace, both in memory the loader mapped.
//!
//! - [`model`]: the `LMODEL01` file and its weights, viewed in place.
//! - [`tokenizer`]: llama2.c's byte-pair tokenizer.
//! - [`engine`]: the forward pass, sampling and generation.
//! - `convert` (feature `std`): a llama2.c f32 checkpoint to an `LMODEL01` file, for `lantern-mkmodel`.
//!
//! Its output for a model, a prompt and a seed is deterministic. `lantern-reference` (feature `std`) checks it against an
//! independent f32 forward pass over the original checkpoint, which int8 tracks for a long prefix and then drifts from.
//! The rotary base is llama2.c's, 10000.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

pub mod engine;
pub mod model;
pub mod tokenizer;

#[cfg(any(test, feature = "std"))]
pub mod convert;
#[cfg(any(test, feature = "std"))]
pub mod reference;

#[cfg(test)]
mod tests;
