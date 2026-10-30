//! Exporting a trained [`Nnue`](super::Nnue) or [`NnueV2`](super::NnueV2) for the base-only
//! Haskell engine.
//!
//! The pieces, in the order they are useful to the port:
//! - [`weights`]: the trained tensors on the host, already in the layout the Haskell code
//!   consumes (every matrix one row per output).
//! - [`scalar`]: a plain-loop evaluator over those weights that mirrors the Haskell forward
//!   pass line by line: board tokens, folded first-layer rows, the 21-row accumulator, the
//!   dense layers, and a softmax over the legal moves only. It is the executable
//!   specification the generated Haskell is checked against, and the place where the
//!   quantisation error of an export is measured without a GHC in the loop.
//! - [`quant`]: int16 quantisation and the base64 stream the weights ship as.
//! - [`haskell`] and [`codegen`]: the generated Haskell modules; [`book`]: the opening book;
//!   [`vecgen`]: the `FixedVec` vector module.

pub mod book;
pub mod codegen;
pub mod haskell;
pub mod quant;
pub mod scalar;
pub mod vecgen;
pub mod weights;

pub use haskell::{Deviations, ExportOptions, deviations, error_positions, write_all};
pub use quant::Int8Arrays;
pub use scalar::Evaluation;
pub use vecgen::VecSpec;
pub use weights::{Array, Dense, Matrix, Weights};
