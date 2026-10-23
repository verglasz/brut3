//! A plain-loop evaluator over [`Weights`], written the way the Haskell evaluator is.
//!
//! Nothing here is fast: the folded rows are recomputed on every call instead of cached.
//! The point is to be the readable, executable definition of the forward pass that the
//! generated Haskell mirrors, checked against burn's [`Nnue::predict`](crate::nn::Nnue)
//! on real positions.

use super::weights::{Dense, Weights};
use crate::game::{BitState, GameState, Move};
use crate::nn::model::MOVES;
use crate::nn::nnue::{BOARDS, IDS, IDS_PER_BOARD, TOKENS, nnue_ids};

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

/// `relu(W x + b)` for a layer with one row per output.
fn dense(layer: &Dense, x: &[f32]) -> Vec<f32> {
    (0..layer.weight.rows)
        .map(|i| relu(dot(layer.weight.row(i), x) + layer.bias[i]))
        .collect()
}

/// `relu(x + W x + b)`.
fn residual(layer: &Dense, x: &[f32]) -> Vec<f32> {
    (0..layer.weight.rows)
        .map(|i| relu(x[i] + dot(layer.weight.row(i), x) + layer.bias[i]))
        .collect()
}

impl Weights {
    /// A board token: the sum of its content row and its nine cell (or padding) rows.
    pub fn token(&self, ids: &[i32]) -> Vec<f32> {
        let mut token = vec![0.0; self.config.d_embed];
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

        let mut h = dense(&self.hidden[0], &acc);
        for layer in &self.hidden[1..] {
            h = residual(layer, &h);
        }

        let wdl_logits: [f32; 3] = std::array::from_fn(|i| {
            dot(self.value.weight.row(i), &h) + self.value.bias[i] + dot(self.skip.row(i), &acc)
        });
        let wdl = softmax3(wdl_logits);

        // Policy over the legal moves only: the global head, the board's own cell logit and
        // the send logit of the board the move sends the opponent to.
        let content = |b: usize| ids[b * IDS_PER_BOARD] as usize;
        let mut policy: Vec<(Move, f32, f32)> = state
            .legal_moves_vec()
            .into_iter()
            .map(|mv| {
                let (b, sq) = (mv.board.index(), mv.square.index());
                let logit = dot(self.policy.weight.row(mv.index()), &h)
                    + self.policy.bias[mv.index()]
                    + self.local.row(content(b))[sq]
                    + self.send[content(sq)];
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
    use super::*;
    use crate::game::test_positions::{o_must_block, x_wins_in_one};
    use crate::nn::BackendKind;
    use crate::nn::model::predict_host;
    use crate::nn::nnue::NnueConfig;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

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
                state.apply(legal[rng.random_range(0..legal.len())]);
                ply += 1;
            }
        }
        out
    }

    /// The scalar evaluator reproduces burn's forward pass on real positions.
    #[test]
    fn matches_the_model() {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let model = NnueConfig::new()
            .with_d_embed(16)
            .with_d_acc(32)
            .with_d_hidden(16)
            .with_layers(3)
            .init(&device);
        let weights = Weights::from_model(&model);
        assert_eq!(weights.parameters(), count_parameters(&model));

        let mut states = vec![BitState::initial(), x_wins_in_one(), o_must_block()];
        states.extend(random_positions(7, 40));
        let expected = predict_host(&model, &states, &device);
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

    fn count_parameters(model: &crate::nn::Nnue) -> usize {
        use burn::module::Module;
        model.num_params()
    }
}
