//! [`LinearNet`]: one linear layer per head over the flat position features. The simplest
//! possible policy/value model; the baseline everything else is measured against.

use burn::module::{Initializer, Param};
use burn::prelude::*;

use super::encode::{FEATURES, encode_batch};
use super::model::{MOVES, Model, Prediction, Verdict};
use crate::game::BitState;

#[derive(Config, Debug)]
pub struct LinearNetConfig {
    pub features: usize,
}

impl Default for LinearNetConfig {
    fn default() -> Self {
        Self::new(FEATURES)
    }
}

impl LinearNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearNet<B> {
        // Same scheme as burn's Linear: Kaiming uniform with gain 1/sqrt(3).
        let init = Initializer::KaimingUniform {
            gain: 1.0 / 3f64.sqrt(),
            fan_out_only: false,
        };
        let head = |out: usize| {
            (
                init.init_with([self.features, out], Some(self.features), Some(out), device),
                Param::from_tensor(Tensor::zeros([out], device)),
            )
        };
        let (wdl_weight, wdl_bias) = head(Verdict::ALL.len());
        let (policy_weight, policy_bias) = head(MOVES);
        LinearNet {
            wdl_weight,
            wdl_bias,
            policy_weight,
            policy_bias,
        }
    }
}

/// `wdl = x W1 + b1`, `policy = x W2 + b2` over the [`FEATURES`] flat features.
///
/// Written as plain matmuls rather than `burn::nn::Linear`: in burn 0.21 the fused `linear` op
/// returns wrong values on the CubeCL backends for a single output column, and the tests in
/// this file check every compiled backend against a host computation to catch anything similar.
#[derive(Module, Debug)]
pub struct LinearNet<B: Backend> {
    /// `[FEATURES, 3]`
    wdl_weight: Param<Tensor<B, 2>>,
    /// `[3]`
    wdl_bias: Param<Tensor<B, 1>>,
    /// `[FEATURES, MOVES]`
    policy_weight: Param<Tensor<B, 2>>,
    /// `[MOVES]`
    policy_bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> LinearNet<B> {
    /// Host-side copies of the parameters: `(wdl_weight, wdl_bias, policy_weight, policy_bias)`,
    /// weights row-major `[FEATURES, out]`.
    pub fn parameters(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let host = |t: Tensor<B, 2>| t.into_data().to_vec::<f32>().expect("f32");
        let host1 = |t: Tensor<B, 1>| t.into_data().to_vec::<f32>().expect("f32");
        (
            host(self.wdl_weight.val()),
            host1(self.wdl_bias.val()),
            host(self.policy_weight.val()),
            host1(self.policy_bias.val()),
        )
    }
}

impl<B: Backend> Model<B> for LinearNet<B> {
    /// `[batch, FEATURES]`
    type Input = Tensor<B, 2>;

    fn encode(states: &[BitState], device: &B::Device) -> Tensor<B, 2> {
        let data = encode_batch(states);
        Tensor::from_data(TensorData::new(data, [states.len(), FEATURES]), device)
    }

    fn predict(&self, features: Tensor<B, 2>) -> Prediction<B> {
        Prediction {
            wdl: features.clone().matmul(self.wdl_weight.val()) + self.wdl_bias.val().unsqueeze(),
            policy: features.matmul(self.policy_weight.val()) + self.policy_bias.val().unsqueeze(),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::Dispatch;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::game::GameState;
    use crate::nn::BackendKind;
    use crate::nn::model::predict_host;

    /// Host-side `softmax(x W + b)` per row, with `mask` (if given) zeroing illegal columns.
    fn host_head(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        out: usize,
        mask: Option<&[bool]>,
    ) -> Vec<Vec<f32>> {
        x.chunks(FEATURES)
            .enumerate()
            .map(|(row, x)| {
                let mut logits = b.to_vec();
                for (i, xi) in x.iter().enumerate() {
                    for (c, l) in logits.iter_mut().enumerate() {
                        *l += xi * w[i * out + c];
                    }
                }
                if let Some(mask) = mask {
                    for (c, l) in logits.iter_mut().enumerate() {
                        if !mask[row * out + c] {
                            *l = f32::NEG_INFINITY;
                        }
                    }
                }
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
                let sum: f32 = exp.iter().sum();
                exp.iter().map(|e| e / sum).collect()
            })
            .collect()
    }

    fn random_states(n: usize, seed: u64) -> Vec<BitState> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut states = Vec::new();
        while states.len() < n {
            let mut s = BitState::initial();
            while s.outcome().is_none() && states.len() < n {
                states.push(s);
                let legal = s.legal_moves();
                let k = rng.random_range(0..crate::game::MoveSet::len(&legal));
                s.apply(crate::game::MoveSet::iter(&legal).nth(k).unwrap());
            }
        }
        states
    }

    /// Both heads against a host computation, on every compiled backend.
    #[test]
    fn predictions_match_host_on_every_backend() {
        let _device = crate::nn::device_lock();
        let states = random_states(64, 1);
        let x = encode_batch(&states);
        let mut legal = vec![false; states.len() * MOVES];
        for (i, s) in states.iter().enumerate() {
            for mv in s.legal_moves() {
                legal[i * MOVES + mv.index()] = true;
            }
        }
        for kind in BackendKind::for_tests() {
            let device = kind.device();
            let net = LinearNetConfig::default().init::<Dispatch>(&device);
            let (w1, b1, w2, b2) = net.parameters();
            let want_wdl = host_head(&x, &w1, &b1, 3, None);
            let want_policy = host_head(&x, &w2, &b2, MOVES, Some(&legal));
            let got = predict_host(&net, &states, &device);
            assert_eq!(got.wdl.len(), states.len());
            for (i, (g, w)) in got.wdl.iter().zip(&want_wdl).enumerate() {
                for c in 0..3 {
                    assert!(
                        (g[c] - w[c]).abs() < 1e-4,
                        "{kind}: wdl row {i}: got {g:?}, want {w:?}"
                    );
                }
            }
            for (i, (g, w)) in got.policy.iter().zip(&want_policy).enumerate() {
                assert_eq!(g.len(), states[i].legal_moves_vec().len());
                let sum: f32 = g.iter().map(|(_, p)| p).sum();
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "{kind}: policy row {i} sums to {sum}"
                );
                for (mv, p) in g {
                    assert!(
                        (p - w[mv.index()]).abs() < 1e-4,
                        "{kind}: policy row {i} move {mv}: got {p}, want {}",
                        w[mv.index()]
                    );
                }
            }
            assert!(predict_host(&net, &[], &device).wdl.is_empty());
        }
    }
}
