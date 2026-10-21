//! Policy/value networks and everything around them: position encoding, the models, training
//! on self-play data, an agent that plays with them, and runtime backend selection.
//!
//! All tensors use burn's [`Dispatch`](burn::Dispatch) backend, whose device enum picks the
//! concrete backend (ndarray, wgpu, CUDA, MLIR CPU) at runtime among those compiled in, and
//! which doubles as its own autodiff backend for training. One model type serves everywhere.
//!
//! Every model implements [`Model`]: it encodes raw [`BitState`](crate::game::BitState)s into
//! its own input tensors and predicts win/draw/loss logits plus move logits ([`Prediction`]).
//! [`AnyModel`] is the concrete enum over the model kinds, described by a [`ModelSpec`] that
//! is saved next to the weights.

pub mod agent;
pub mod any_model;
pub mod backend;
pub mod board_transformer;
pub mod cell_transformer;
pub mod encode;
pub mod heads;
pub mod linear;
pub mod model;
pub mod pipeline;
pub mod train;

pub use agent::NetAgent;
pub use any_model::{AnyModel, ModelSpec};
pub use backend::BackendKind;
pub use board_transformer::{BoardTransformer, BoardTransformerConfig};
pub use cell_transformer::{CellTransformer, CellTransformerConfig};
pub use encode::{FEATURES, encode_batch, encode_into};
pub use linear::{LinearNet, LinearNetConfig};
pub use model::{HostPrediction, MOVES, Model, Prediction, Verdict, expected_value};
pub use pipeline::{GenerationReport, LoopConfig, ReplayBuffer, run_loop, train_model};
pub use train::{Dataset, Sample, TrainOptions, TrainReport, generate, generate_parallel, train};

/// Tests that touch a device take this lock: initialising or driving several CubeCL backends
/// (wgpu, cuda, MLIR CPU) from parallel test threads in one process deadlocks.
#[cfg(test)]
pub(crate) fn device_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
