//! A plain-loop evaluator over [`Weights`], written the way the Haskell evaluator is.
//!
//! Nothing here is fast: the folded rows are recomputed on every call instead of cached.
//! The point is to be the readable, executable definition of the forward pass that the
//! generated Haskell mirrors, checked against burn's [`Nnue::predict`](crate::nn::Nnue)
//! on real positions.

use super::weights::{Dense, Layer, Matrix, Weights};
use crate::game::{BitState, GameState, Move};
use crate::nn::model::MOVES;
use crate::nn::nnue::{BOARDS, GATES, IDS, IDS_PER_BOARD, TOKENS, nnue_ids};

/// Everything the evaluator computes for one position, for comparison with a port.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// The input ids, as [`nnue_ids`] writes them.
    pub ids: [i32; IDS],
    /// The accumulator after the ReLU.
    pub acc: Vec<f32>,
    /// The last hidden layer.
    pub hidden: Vec<f32>,
    /// Win/draw/loss logits and their softmax.
    pub wdl_logits: [f32; 3],
    pub wdl: [f32; 3],
    /// Every legal move with its logit and its probability (softmax over the legal moves).
    pub policy: Vec<(Move, f32, f32)>,
}

impl Evaluation {
    /// `p_win - p_loss`.
    pub fn value(&self) -> f32 {
        self.wdl[0] - self.wdl[2]
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `W x`, one row per output.
fn product(w: &Matrix, x: &[f32]) -> Vec<f32> {
    (0..w.rows).map(|i| dot(w.row(i), x)).collect()
}

/// `W x + b`.
fn linear(layer: &Dense, x: &[f32]) -> Vec<f32> {
    let mut y = product(&layer.weight, x);
    for (y, b) in y.iter_mut().zip(&layer.bias) {
        *y += b;
    }
    y
}

fn relus(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| relu(v)).collect()
}

/// `x + D (g * u)`, the gated layers' sum.
fn gated(x: &[f32], down: &Matrix, g: &[f32], u: &[f32]) -> Vec<f32> {
    let inner: Vec<f32> = g.iter().zip(u).map(|(g, u)| g * u).collect();
    x.iter()
        .zip(product(down, &inner))
        .map(|(x, d)| x + d)
        .collect()
}

impl Layer {
    /// The layer's output for input `x`.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let add =
            |x: &[f32], y: Vec<f32>| -> Vec<f32> { x.iter().zip(y).map(|(x, y)| x + y).collect() };
        match self {
            Layer::Dense(d) => relus(&linear(d, x)),
            Layer::Linear(d) => linear(d, x),
            Layer::Residual(d) => relus(&add(x, linear(d, x))),
            Layer::ReluResidual(d) => relus(&add(x, linear(d, &relus(x)))),
            Layer::SelfGated { up, down } => gated(x, down, &relus(x), &linear(up, x)),
            Layer::ReGlu { gate, up, down } => {
                gated(x, down, &relus(&linear(gate, x)), &linear(up, x))
            }
        }
    }
}

impl Weights {
    /// A board token: the sum of its content row and its nine cell (or padding) rows.
    pub fn token(&self, ids: &[i32]) -> Vec<f32> {
        let mut token = vec![0.0; self.d_embed];
        for &id in ids {
            for (t, e) in token.iter_mut().zip(self.content.row(id as usize)) {
                *t += e;
            }
        }
        token
    }

    /// The folded first-layer row of a token at board `k` (`BOARDS` for the forced token):
    /// `t · W_k`, what an exported table stores per content.
    pub fn folded_row(&self, k: usize, token: &[f32]) -> Vec<f32> {
        let w = if k == BOARDS {
            &self.forced
        } else {
            &self.mix[k]
        };
        (0..w.rows).map(|i| dot(w.row(i), token)).collect()
    }

    pub fn evaluate(&self, state: &BitState) -> Evaluation {
        let mut ids = [0i32; IDS];
        nnue_ids(state, &mut ids);

        // Accumulator: ten folded rows, the feature rows and the bias, then the ReLU.
        let mut acc = self.mix_bias.clone();
        for k in 0..TOKENS {
            let token = self.token(&ids[k * IDS_PER_BOARD..(k + 1) * IDS_PER_BOARD]);
            for (a, r) in acc.iter_mut().zip(self.folded_row(k, &token)) {
                *a += r;
            }
        }
        for &id in &ids[TOKENS * IDS_PER_BOARD..] {
            for (a, f) in acc.iter_mut().zip(self.features.row(id as usize)) {
                *a += f;
            }
        }
        for a in &mut acc {
            *a = relu(*a);
        }

        let mut h = acc.clone();
        for layer in &self.trunk {
            h = layer.apply(&h);
        }

        // The head gates: the value's mix, then the local and send tables' (all 1 without).
        let gates = match &self.gates {
            Some(g) => linear(g, &h).into_iter().map(sigmoid).collect(),
            None => vec![1.0; GATES],
        };
        let wdl_logits: [f32; 3] = std::array::from_fn(|i| {
            let value = dot(self.value.weight.row(i), &h) + self.value.bias[i];
            let skip = dot(self.skip.row(i), &acc);
            match self.gates {
                Some(_) => (1.0 - gates[0]) * value + gates[0] * skip,
                None => value + skip,
            }
        });
        let wdl = softmax3(wdl_logits);

        // Policy over the legal moves only: the global head, the board's own cell logit and
        // the send logit of the board the move sends the opponent to, each through its gate.
        let (local_gate, send_gate) = (&gates[1..1 + BOARDS], &gates[1 + BOARDS..]);
        let content = |b: usize| ids[b * IDS_PER_BOARD] as usize;
        let mut policy: Vec<(Move, f32, f32)> = state
            .legal_moves_vec()
            .into_iter()
            .map(|mv| {
                let (b, sq) = (mv.board.index(), mv.square.index());
                let logit = dot(self.policy.weight.row(mv.index()), &h)
                    + self.policy.bias[mv.index()]
                    + local_gate[b] * self.local.row(content(b))[sq]
                    + send_gate[sq] * self.send[content(sq)];
                (mv, logit, 0.0)
            })
            .collect();
        let max = policy.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        let total: f32 = policy.iter().map(|p| (p.1 - max).exp()).sum();
        for p in &mut policy {
            p.2 = (p.1 - max).exp() / total;
        }
        debug_assert!(policy.len() <= MOVES);

        Evaluation {
            ids,
            acc,
            hidden: h,
            wdl_logits,
            wdl,
            policy,
        }
    }
}

fn softmax3(z: [f32; 3]) -> [f32; 3] {
    let max = z[0].max(z[1]).max(z[2]);
    let e = z.map(|x| (x - max).exp());
    let total = e[0] + e[1] + e[2];
    e.map(|x| x / total)
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::game::test_positions::{o_must_block, x_wins_in_one};
    use crate::nn::BackendKind;
    use crate::nn::model::predict_host;
    use crate::nn::nnue::NnueConfig;
    use crate::nn::nnue_v2::NnueV2Config;

    /// Positions along random games, one every few plies.
    fn random_positions(seed: u64, n: usize) -> Vec<BitState> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut out = Vec::new();
        while out.len() < n {
            let mut state = BitState::initial();
            let mut ply = 0;
            while !state.is_terminal() && out.len() < n {
                if ply % 5 == 3 {
                    out.push(state.clone());
                }
                let legal = state.legal_moves_vec();
                state.apply(legal[rng.random_range(0..legal.len())].into());
                ply += 1;
            }
        }
        out
    }

    /// Adds noise to every parameter, so that the zero-initialised gates and ReZero scales
    /// matter.
    struct Jitter;

    impl burn::module::ModuleMapper for Jitter {
        fn map_float<const D: usize>(
            &mut self,
            param: burn::module::Param<burn::prelude::Tensor<D>>,
        ) -> burn::module::Param<burn::prelude::Tensor<D>> {
            param.map(|t| {
                let noise = burn::prelude::Tensor::random(
                    t.shape(),
                    burn::tensor::Distribution::Normal(0.0, 0.1),
                    &t.device(),
                );
                t + noise
            })
        }
    }

    /// The scalar evaluator reproduces burn's forward pass on real positions, with the same
    /// parameters but the `folded` ReZero scalars.
    fn assert_matches<M>(
        model: &M,
        weights: &Weights,
        folded: usize,
        device: &burn::prelude::Device,
    ) where
        M: crate::nn::model::Predictor + burn::module::Module,
    {
        assert_eq!(
            weights.parameters() + folded,
            count_parameters(model),
            "{}",
            weights.spec
        );
        let mut states = vec![BitState::initial(), x_wins_in_one(), o_must_block()];
        states.extend(random_positions(7, 40));
        let expected = predict_host(model, &states, device);
        for (i, state) in states.iter().enumerate() {
            let eval = weights.evaluate(state);
            for (a, b) in eval.wdl.iter().zip(&expected.wdl[i]) {
                assert!((a - b).abs() < 2e-4, "wdl of position {i}: {a} vs {b}");
            }
            assert_eq!(eval.policy.len(), expected.policy[i].len());
            for ((mv, _, p), (emv, ep)) in eval.policy.iter().zip(&expected.policy[i]) {
                assert_eq!(mv, emv, "move order of position {i}");
                assert!(
                    (p - ep).abs() < 2e-4,
                    "policy of {mv} in position {i}: {p} vs {ep}"
                );
            }
        }
    }

    #[test]
    fn matches_the_model() {
        use burn::module::Module;

        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let nnue = NnueConfig::new()
            .with_d_embed(16)
            .with_d_acc(32)
            .with_d_hidden(16)
            .with_layers(3);
        for config in [nnue.clone(), nnue.with_reglu(Some(24))] {
            let model = config.init(&device).map(&mut Jitter);
            assert_matches(&model, &Weights::from_model(&model), 0, &device);
        }
        let v2 = NnueV2Config::new()
            .with_d_embed(16)
            .with_d_acc(32)
            .with_d_hidden(16)
            .with_layers(3)
            .with_rezero(true);
        for config in [v2.clone(), v2.with_reglu(Some(24))] {
            let model = config.init(&device).map(&mut Jitter);
            // Two ReZero scalars: the first layer's and the third's.
            assert_matches(&model, &Weights::from_v2(&model), 2, &device);
        }
    }

    fn count_parameters(model: &impl burn::module::Module) -> usize {
        model.num_params()
    }
}
