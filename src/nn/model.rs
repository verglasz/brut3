//! [`ValueNet`]: a single-layer perceptron from position features to a value in `[-1, 1]`.

use std::path::Path;

use burn::module::{Initializer, Param};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError};
use burn::tensor::activation::tanh;

use super::encode::FEATURES;

/// Weights are stored as named MessagePack; the recorder appends its own `.mpk` extension.
pub type Recorder = NamedMpkFileRecorder<FullPrecisionSettings>;

#[derive(Config, Debug)]
pub struct ValueNetConfig {
    pub features: usize,
}

impl Default for ValueNetConfig {
    fn default() -> Self {
        Self::new(FEATURES)
    }
}

impl ValueNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ValueNet<B> {
        // Same scheme as burn's Linear: Kaiming uniform with gain 1/sqrt(3), i.e. U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        let init = Initializer::KaimingUniform {
            gain: 1.0 / 3f64.sqrt(),
            fan_out_only: false,
        };
        ValueNet {
            weight: init.init_with([1, self.features], Some(self.features), Some(1), device),
            bias: Param::from_tensor(Tensor::zeros([1], device)),
        }
    }
}

/// `tanh(x . w + b)`: one linear layer, one output.
///
/// The dot product is written as an elementwise multiply and a sum over features rather than
/// `burn::nn::Linear` or a `matmul`. In burn 0.21, on the CubeCL backends (cuda, wgpu, MLIR
/// CPU), any elementwise op fused onto a matmul whose output has a single column (bias add,
/// tanh, or the fused `linear` op) returns wrong values for every row but the first in the
/// batch. The bare matmul is correct, and so is this formulation; see `nn::model::tests`.
#[derive(Module, Debug)]
pub struct ValueNet<B: Backend> {
    /// `[1, FEATURES]`
    weight: Param<Tensor<B, 2>>,
    /// `[1]`
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> ValueNet<B> {
    /// `[batch, FEATURES] -> [batch, 1]`, values in `[-1, 1]` for the side to move.
    pub fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let bias = self.bias.val().unsqueeze::<2>(); // [1, 1], broadcast over the batch
        let products = features * self.weight.val(); // [batch, FEATURES] * [1, FEATURES]
        tanh(products.sum_dim(1) + bias)
    }

    /// Runs the network on `features` (a multiple of [`FEATURES`] long) and reads the values
    /// back to the host.
    pub fn values(&self, features: &[f32], device: &B::Device) -> Vec<f32> {
        assert_eq!(features.len() % FEATURES, 0);
        let n = features.len() / FEATURES;
        if n == 0 {
            return Vec::new();
        }
        let x =
            Tensor::<B, 2>::from_data(TensorData::new(features.to_vec(), [n, FEATURES]), device);
        self.forward(x)
            .into_data()
            .to_vec::<f32>()
            .expect("value tensor is f32")
    }

    /// Host-side copy of the parameters: `(weight[FEATURES], bias)`.
    pub fn parameters(&self) -> (Vec<f32>, f32) {
        let w = self
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 weights");
        let b = self
            .bias
            .val()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 bias")[0];
        (w, b)
    }

    /// Saves to `path` plus the recorder's `.mpk` extension.
    pub fn save(self, path: &Path) -> Result<(), RecorderError> {
        self.save_file(path.to_path_buf(), &Recorder::new())
    }

    /// Loads weights saved by [`ValueNet::save`] onto `device`.
    pub fn load(path: &Path, device: &B::Device) -> Result<Self, RecorderError> {
        ValueNetConfig::default().init::<B>(device).load_file(
            path.to_path_buf(),
            &Recorder::new(),
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::Dispatch;

    use super::*;
    use crate::nn::BackendKind;

    /// Forward pass against a host-side computation, on every compiled backend. Guards the
    /// single-output-column bug that `burn::nn::Linear` has on CubeCL backends.
    #[test]
    fn forward_matches_host_on_every_backend() {
        let _device = crate::nn::device_lock();
        let n = 64;
        let features: Vec<f32> = (0..n * FEATURES)
            .map(|i| ((i * 7) % 5 == 0) as u8 as f32)
            .collect();
        for kind in BackendKind::compiled() {
            let device = kind.device();
            let net = ValueNetConfig::default().init::<Dispatch>(&device);
            let (w, b) = net.parameters();
            assert_eq!(w.len(), FEATURES);
            let want: Vec<f32> = features
                .chunks(FEATURES)
                .map(|x| (x.iter().zip(&w).map(|(a, b)| a * b).sum::<f32>() + b).tanh())
                .collect();
            let got = net.values(&features, &device);
            assert_eq!(got.len(), n);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-4, "{kind}: row {i}: got {g}, want {w}");
            }
            assert!(net.values(&[], &device).is_empty());
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let net = ValueNetConfig::default().init::<Dispatch>(&device);
        let features: Vec<f32> = (0..FEATURES).map(|i| (i % 3) as f32 * 0.5).collect();
        let before = net.values(&features, &device);
        let dir = std::env::temp_dir().join(format!("uttt-valuenet-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("net");
        net.save(&path).unwrap();
        let loaded = ValueNet::<Dispatch>::load(&path, &device).unwrap();
        assert_eq!(loaded.values(&features, &device), before);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
