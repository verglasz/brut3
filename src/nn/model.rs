//! What every network has in common: the [`Model`] trait, its [`Prediction`], and the
//! win/draw/loss [`Verdict`].

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};

use crate::game::{BitState, GameState, Move, MoveSet, Outcome, Player};

/// Number of distinct moves, one per square of the big board (`Move::index`).
pub const MOVES: usize = 81;

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

/// A network's raw outputs for a batch of positions.
pub struct Prediction<B: Backend> {
    /// `[batch, 3]` win/draw/loss logits, side to move's view.
    pub wdl: Tensor<B, 2>,
    /// `[batch, MOVES]` move logits, indexed by `Move::index`.
    pub policy: Tensor<B, 2>,
}

impl<B: Backend> Prediction<B> {
    pub fn wdl_probabilities(&self) -> Tensor<B, 2> {
        softmax(self.wdl.clone(), 1)
    }

    /// Log-probabilities over moves with illegal ones (where `legal` is false) suppressed.
    pub fn policy_log_probabilities(&self, legal: Tensor<B, 2, Bool>) -> Tensor<B, 2> {
        log_softmax(self.masked_policy(legal), 1)
    }

    /// Probabilities over moves with illegal ones suppressed.
    pub fn policy_probabilities(&self, legal: Tensor<B, 2, Bool>) -> Tensor<B, 2> {
        softmax(self.masked_policy(legal), 1)
    }

    fn masked_policy(&self, legal: Tensor<B, 2, Bool>) -> Tensor<B, 2> {
        self.policy
            .clone()
            .mask_fill(legal.bool_not(), ILLEGAL_LOGIT)
    }
}

/// A policy/value network over positions.
///
/// Each model encodes positions into its own input tensors, so models with different views
/// of the board (flat features, board tokens, cell tokens) share one training pipeline.
pub trait Model<B: Backend>: Module<B> {
    type Input: Clone + Send + std::fmt::Debug + 'static;

    /// Encodes positions into the model's input, from each position's side to move's view.
    fn encode(states: &[BitState], device: &B::Device) -> Self::Input;

    fn predict(&self, input: Self::Input) -> Prediction<B>;
}

/// `[batch, MOVES]` mask of legal moves.
pub fn legal_mask<B: Backend>(states: &[BitState], device: &B::Device) -> Tensor<B, 2, Bool> {
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

/// Runs `model` on `states` and reads the probabilities back.
pub fn predict_host<B: Backend, M: Model<B>>(
    model: &M,
    states: &[BitState],
    device: &B::Device,
) -> HostPrediction {
    if states.is_empty() {
        return HostPrediction::default();
    }
    let prediction = model.predict(M::encode(states, device));
    let wdl = prediction
        .wdl_probabilities()
        .into_data()
        .to_vec::<f32>()
        .expect("f32 probabilities");
    let policy = prediction
        .policy_probabilities(legal_mask(states, device))
        .into_data()
        .to_vec::<f32>()
        .expect("f32 probabilities");
    HostPrediction {
        wdl: wdl.chunks(3).map(|p| [p[0], p[1], p[2]]).collect(),
        policy: states
            .iter()
            .zip(policy.chunks(MOVES))
            .map(|(state, row)| {
                state
                    .legal_moves()
                    .iter()
                    .map(|mv| (mv, row[mv.index()]))
                    .collect()
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::test_positions::x_wins_in_one;

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
        let mask: Vec<bool> = legal_mask::<burn::Dispatch>(&states, &device)
            .int()
            .into_data()
            .to_vec::<i32>()
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
