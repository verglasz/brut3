//! What every network has in common: the [`Model`] trait, its [`Prediction`], and the
//! win/draw/loss [`Verdict`].

use burn::prelude::*;
use burn::tensor::activation::log_softmax;

use crate::game::{BitState, GameState, Move, MoveSet, Outcome, Player};

/// Number of distinct moves, one per square of the big board (`Move::index`).
pub const MOVES: usize = Move::COUNT;

/// Width of one row of raw network outputs: the three verdict logits, then the move logits.
pub const LOGITS: usize = 3 + MOVES;

/// Logit added to illegal moves before the softmax: effectively zero probability.
const ILLEGAL_LOGIT: f32 = -1e4;

/// How a game ends from the side to move's point of view. Also the class index of the
/// network's three win/draw/loss logits.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    Win = 0,
    Draw = 1,
    Loss = 2,
}

impl Verdict {
    pub const ALL: [Verdict; 3] = [Verdict::Win, Verdict::Draw, Verdict::Loss];

    pub fn from_outcome(outcome: Outcome, player: Player) -> Self {
        match outcome {
            Outcome::Win(p) if p == player => Verdict::Win,
            Outcome::Win(_) => Verdict::Loss,
            Outcome::Draw => Verdict::Draw,
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(index: usize) -> Self {
        Self::ALL[index]
    }

    /// The verdict as a win/draw/loss distribution.
    pub const fn one_hot(self) -> [f32; 3] {
        let mut p = [0.0; 3];
        p[self as usize] = 1.0;
        p
    }

    /// `1` for a win, `0` for a draw, `-1` for a loss.
    pub const fn value(self) -> f32 {
        match self {
            Verdict::Win => 1.0,
            Verdict::Draw => 0.0,
            Verdict::Loss => -1.0,
        }
    }
}

/// Expected value of a win/draw/loss distribution: `p_win - p_loss`.
pub fn expected_value(probabilities: &[f32; 3]) -> f32 {
    probabilities[Verdict::Win.index()] - probabilities[Verdict::Loss.index()]
}

/// The most probable verdict of a win/draw/loss distribution.
pub fn most_likely(probabilities: &[f32; 3]) -> Verdict {
    let mut best = 0;
    for c in 1..3 {
        if probabilities[c] > probabilities[best] {
            best = c;
        }
    }
    Verdict::from_index(best)
}

/// A network's raw outputs for a batch of positions.
pub struct Prediction {
    /// `[batch, 3]` win/draw/loss logits, side to move's view.
    pub wdl: Tensor<2>,
    /// `[batch, MOVES]` move logits, indexed by `Move::index`.
    pub policy: Tensor<2>,
}

impl Prediction {
    /// Log-probabilities over moves with illegal ones (where `legal` is false) suppressed,
    /// for the training loss. Inference reads the raw logits back instead
    /// ([`predict_logits`]).
    pub fn policy_log_probabilities(&self, legal: Tensor<2, Bool>) -> Tensor<2> {
        log_softmax(self.masked_policy(legal), 1)
    }

    fn masked_policy(&self, legal: Tensor<2, Bool>) -> Tensor<2> {
        self.policy
            .clone()
            .mask_fill(legal.bool_not(), ILLEGAL_LOGIT)
    }
}

/// A policy/value network over positions.
///
/// Each model encodes positions into its own input tensors, so models with different views
/// of the board (flat features, board tokens, cell tokens) share one training pipeline.
pub trait Model: Module {
    type Input: Clone + Send + std::fmt::Debug + 'static;

    /// Encodes positions into the model's input, from each position's side to move's view.
    fn encode(states: &[BitState], device: &Device) -> Self::Input;

    fn predict(&self, input: Self::Input) -> Prediction;
}

/// `[batch, MOVES]` mask of legal moves.
pub fn legal_mask(states: &[BitState], device: &Device) -> Tensor<2, Bool> {
    let mut mask = vec![false; states.len() * MOVES];
    for (i, state) in states.iter().enumerate() {
        for mv in state.legal_moves() {
            mask[i * MOVES + mv.index()] = true;
        }
    }
    Tensor::from_data(TensorData::new(mask, [states.len(), MOVES]), device)
}

/// A batch of predictions read back to the host.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HostPrediction {
    /// Win/draw/loss probabilities per position.
    pub wdl: Vec<[f32; 3]>,
    /// Probability of each legal move per position (illegal moves are absent).
    pub policy: Vec<Vec<(Move, f32)>>,
}

impl HostPrediction {
    /// `p_win - p_loss` per position.
    pub fn values(&self) -> Vec<f32> {
        self.wdl.iter().map(expected_value).collect()
    }

    /// Most likely legal move per position.
    pub fn best_moves(&self) -> Vec<Move> {
        self.policy
            .iter()
            .map(|p| {
                p.iter()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(mv, _)| *mv)
                    .expect("at least one legal move")
            })
            .collect()
    }
}

/// Runs `model` on `states` and reads the raw logits back in one transfer: `states.len()`
/// rows of [`LOGITS`] floats, each the three verdict logits followed by the move logits
/// indexed by `Move::index`. Masking and softmax are left to the host, where they are a few
/// dozen flops per position: see [`host_wdl`] and [`host_policy_into`].
pub fn predict_logits<M: Model>(model: &M, states: &[BitState], device: &Device) -> Vec<f32> {
    if states.is_empty() {
        return Vec::new();
    }
    logits_to_host(predict_logits_tensor(model, states, device))
}

/// Smallest batch the device is asked for; see [`padded_len`].
const MIN_BATCH: usize = 32;

/// The batch size the device sees for `n` positions: `n` rounded up to a multiple of an
/// eighth of its power-of-two octave (so 8 sizes per octave, under 12.5% padding), at least
/// [`MIN_BATCH`]. Kernels and autotune results are compiled per shape, and self-play batches
/// shrink round by round as games finish, so unpadded batches made the CUDA runtime compile
/// kernels for every new size: two thirds of all CPU time in a profile. With a few dozen
/// sizes in all, the compiled kernels serve every round after the first few.
fn padded_len(n: usize) -> usize {
    let n = n.max(MIN_BATCH);
    let step = (n.next_power_of_two() / 16).max(1);
    n.div_ceil(step) * step
}

/// The `[n, LOGITS]` tensor behind [`predict_logits`], still on the device: queue it with
/// this, do other work, then read it with [`logits_to_host`]. `states` must not be empty.
/// The batch is padded to [`padded_len`] positions on the way and the padding sliced off.
pub fn predict_logits_tensor<M: Model>(
    model: &M,
    states: &[BitState],
    device: &Device,
) -> Tensor<2> {
    let n = states.len();
    let padded = padded_len(n);
    let prediction = if padded == n {
        model.predict(M::encode(states, device))
    } else {
        let mut full = Vec::with_capacity(padded);
        full.extend_from_slice(states);
        full.resize(padded, states[0]);
        model.predict(M::encode(&full, device))
    };
    let logits = Tensor::cat(vec![prediction.wdl, prediction.policy], 1);
    if padded == n {
        logits
    } else {
        logits.slice([0..n])
    }
}

/// Reads a [`predict_logits_tensor`] result back, row-major.
pub fn logits_to_host(logits: Tensor<2>) -> Vec<f32> {
    logits
        .into_data()
        .try_into_vec::<f32>()
        .expect("f32 logits")
}

/// Win/draw/loss probabilities from one row of logits.
pub fn host_wdl(row: &[f32]) -> [f32; 3] {
    let max = row[0].max(row[1]).max(row[2]);
    let e = [
        (row[0] - max).exp(),
        (row[1] - max).exp(),
        (row[2] - max).exp(),
    ];
    let total = e[0] + e[1] + e[2];
    [e[0] / total, e[1] / total, e[2] / total]
}

/// Softmax of one row's move logits restricted to `legal`, written to `out[mv.index()]` for
/// each legal move; other entries are left untouched. With no legal moves nothing is written.
pub fn host_policy_into(row: &[f32], legal: &impl MoveSet, out: &mut [f32; MOVES]) {
    let logits = &row[3..3 + MOVES];
    let mut max = f32::NEG_INFINITY;
    for mv in legal.iter() {
        max = max.max(logits[mv.index()]);
    }
    let mut total = 0.0f32;
    for mv in legal.iter() {
        let e = (logits[mv.index()] - max).exp();
        out[mv.index()] = e;
        total += e;
    }
    if total > 0.0 {
        for mv in legal.iter() {
            out[mv.index()] /= total;
        }
    }
}

/// Runs `model` on `states` and reads the probabilities back.
pub fn predict_host<M: Model>(model: &M, states: &[BitState], device: &Device) -> HostPrediction {
    let logits = predict_logits(model, states, device);
    let mut dense = [0.0f32; MOVES];
    let mut prediction = HostPrediction::default();
    for (state, row) in states.iter().zip(logits.chunks_exact(LOGITS)) {
        prediction.wdl.push(host_wdl(row));
        let legal = state.legal_moves();
        host_policy_into(row, &legal, &mut dense);
        prediction
            .policy
            .push(legal.iter().map(|mv| (mv, dense[mv.index()])).collect());
    }
    prediction
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::test_positions::x_wins_in_one;

    #[test]
    fn padded_len_rounds_up_to_eighths_of_an_octave() {
        assert_eq!(padded_len(1), MIN_BATCH);
        assert_eq!(padded_len(MIN_BATCH), MIN_BATCH);
        assert_eq!(padded_len(33), 36);
        assert_eq!(padded_len(575), 576);
        assert_eq!(padded_len(577), 640);
        assert_eq!(padded_len(1024), 1024);
        assert_eq!(padded_len(1025), 1152);
        for n in 1..5000 {
            let p = padded_len(n);
            assert!(
                p >= n && (p - n) as f32 <= 0.125 * p as f32 || p == MIN_BATCH,
                "{n} -> {p}"
            );
        }
    }

    #[test]
    fn verdict_round_trips() {
        for v in Verdict::ALL {
            assert_eq!(Verdict::from_index(v.index()), v);
        }
        assert_eq!(expected_value(&[0.6, 0.3, 0.1]), 0.5);
        assert_eq!(
            Verdict::from_outcome(Outcome::Win(Player::X), Player::X),
            Verdict::Win
        );
        assert_eq!(
            Verdict::from_outcome(Outcome::Win(Player::X), Player::O),
            Verdict::Loss
        );
        assert_eq!(
            Verdict::from_outcome(Outcome::Draw, Player::O),
            Verdict::Draw
        );
        assert_eq!(Verdict::Loss.value(), -1.0);
    }

    #[test]
    fn legal_mask_marks_exactly_the_legal_moves() {
        let _device = crate::nn::device_lock();
        let device = crate::nn::BackendKind::preferred().device();
        let states = [BitState::initial(), x_wins_in_one()];
        let mask: Vec<bool> = legal_mask(&states, &device)
            .int()
            .into_data()
            .try_into_vec::<i32>()
            .unwrap()
            .into_iter()
            .map(|v| v != 0)
            .collect();
        assert_eq!(mask[..MOVES].iter().filter(|&&b| b).count(), 81);
        assert_eq!(mask[MOVES..].iter().filter(|&&b| b).count(), 7);
        for mv in x_wins_in_one().legal_moves() {
            assert!(mask[MOVES + mv.index()]);
        }
    }
}
