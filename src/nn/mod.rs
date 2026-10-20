//! The value network and everything around it: feature encoding, the model, training on
//! self-play data, an agent that plays with it, and runtime backend selection.
//!
//! All tensors use burn's [`Dispatch`](burn::Dispatch) backend, whose device enum picks the
//! concrete backend (ndarray, wgpu, CUDA, MLIR CPU) at runtime among those compiled in, and
//! which doubles as its own autodiff backend for training. One model type serves everywhere.

pub mod agent;
pub mod backend;
pub mod encode;
pub mod model;
pub mod train;

pub use agent::ValueNetAgent;
pub use backend::BackendKind;
pub use encode::{FEATURES, encode_batch, encode_into};
pub use model::{ValueNet, ValueNetConfig};
pub use train::{Dataset, Sample, TrainOptions, TrainReport, generate, train};

/// Tests that touch a device take this lock: initialising or driving several CubeCL backends
/// (wgpu, cuda, MLIR CPU) from parallel test threads in one process deadlocks.
#[cfg(test)]
pub(crate) fn device_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
