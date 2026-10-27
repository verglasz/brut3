//! [`NetAgent`]: plays and evaluates with a trained policy/value network.

use std::path::Path;

use burn::prelude::Device;
use burn::prelude::Tensor;

use super::any_model::AnyModel;
use super::backend::BackendKind;
use super::model::{
    HostPrediction,
    LOGITS,
    expected_value,
    host_policy_into,
    host_wdl,
    logits_to_host,
};
use crate::agent::{Evaluation, Evaluations, UtttAgent};
use crate::game::{BitState, GameState, Move};

pub struct NetAgent {
    model: AnyModel,
    device: Device,
    backend: BackendKind,
    /// A forward pass queued by `begin_batch` and not yet read back: the logits tensor and how
    /// many positions it covers.
    in_flight: Option<(Tensor<2>, usize)>,
}

impl NetAgent {
    pub fn new(model: AnyModel, backend: BackendKind) -> Self {
        Self {
            model,
            device: backend.device(),
            backend,
            in_flight: None,
        }
    }

    /// Loads a model saved by [`AnyModel::save`] onto `backend`'s default device.
    pub fn load(path: &Path, backend: BackendKind) -> Result<Self, String> {
        let device = backend.device();
        let model = AnyModel::load(path, &device)?;
        Ok(Self {
            model,
            device,
            backend,
            in_flight: None,
        })
    }

    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    pub fn model(&self) -> &AnyModel {
        &self.model
    }

    /// Network outputs for `states`, one forward pass.
    pub fn predict(&self, states: &[BitState]) -> HostPrediction {
        self.model.predict_host(states, &self.device)
    }

    /// Appends the evaluations of `states` from their rows of `logits`: exact values for
    /// terminal states, the network's value and legal-masked policy otherwise, with masking
    /// and softmax done here on the host straight into the batch's prior rows.
    fn fill(states: &[BitState], logits: &[f32], out: &mut Evaluations) {
        debug_assert_eq!(logits.len(), states.len() * LOGITS);
        for (state, row) in states.iter().zip(logits.chunks_exact(LOGITS)) {
            match state.outcome() {
                Some(outcome) => out.push_value(outcome.value_for(state.to_move())),
                None => {
                    let priors = out.push_row(expected_value(&host_wdl(row)));
                    host_policy_into(row, &state.legal_moves(), priors);
                }
            }
        }
    }

    /// Exact value for terminal states, the network's otherwise.
    fn evaluation(state: &BitState, wdl: &[f32; 3], policy: Vec<(Move, f32)>) -> Evaluation {
        match state.outcome() {
            Some(outcome) => Evaluation::value_only(outcome.value_for(state.to_move())),
            None => Evaluation {
                value: expected_value(wdl),
                priors: policy,
            },
        }
    }
}

impl UtttAgent for NetAgent {
    type State = BitState;

    /// The most likely move under the policy head. No search, no lookahead.
    fn choose_move(&mut self, state: &BitState) -> Move {
        assert!(
            state.outcome().is_none(),
            "choose_move called on a terminal state"
        );
        self.predict(std::slice::from_ref(state)).best_moves()[0]
    }

    fn choose_move_with_policy(&mut self, state: &BitState) -> (Move, Vec<(Move, f32)>) {
        assert!(
            state.outcome().is_none(),
            "choose_move called on a terminal state"
        );
        let mut prediction = self.predict(std::slice::from_ref(state));
        let mv = prediction.best_moves()[0];
        (mv, prediction.policy.swap_remove(0))
    }

    fn evaluate(&mut self, state: &BitState) -> Evaluation {
        let mut p = self.predict(std::slice::from_ref(state));
        Self::evaluation(state, &p.wdl[0], p.policy.swap_remove(0))
    }

    /// One forward pass and one readback for the whole batch.
    fn evaluate_batch(&mut self, states: &[BitState], out: &mut Evaluations) {
        let logits = self.model.predict_logits(states, &self.device);
        Self::fill(states, &logits, out);
    }

    /// Queues the forward pass; the device runs it while the caller works.
    fn begin_batch(&mut self, states: &[BitState]) {
        assert!(self.in_flight.is_none(), "finish the previous batch first");
        if !states.is_empty() {
            let logits = self.model.predict_logits_tensor(states, &self.device);
            self.in_flight = Some((logits, states.len()));
        }
    }

    /// Reads the queued forward pass back.
    fn finish_batch(&mut self, states: &[BitState], out: &mut Evaluations) {
        match self.in_flight.take() {
            Some((logits, n)) => {
                assert_eq!(
                    n,
                    states.len(),
                    "finish_batch with the states of begin_batch"
                );
                Self::fill(states, &logits_to_host(logits), out);
            }
            None => {
                assert!(states.is_empty(), "finish_batch without begin_batch");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::test_positions::{X_WINNING_MOVE, x_wins_in_one};
    use crate::mcts::{Mcts, MctsConfig};
    use crate::nn::ModelSpec;

    fn untrained() -> NetAgent {
        let kind = BackendKind::preferred();
        NetAgent::new(ModelSpec::Linear.init(&kind.device()), kind)
    }

    #[test]
    fn plays_legal_moves_with_a_policy_over_legal_moves() {
        let _device = crate::nn::device_lock();
        let mut agent = untrained();
        let state = x_wins_in_one();
        let (mv, policy) = agent.choose_move_with_policy(&state);
        assert!(state.is_legal(mv));
        assert_eq!(policy.len(), 7);
        assert!(policy.iter().all(|(m, _)| state.is_legal(*m)));
        let sum: f32 = policy.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert_eq!(agent.choose_move(&state), mv);
    }

    #[test]
    fn evaluations_are_exact_on_terminal_states_and_batched() {
        let _device = crate::nn::device_lock();
        let mut agent = untrained();
        let mut won = x_wins_in_one();
        won.apply(X_WINNING_MOVE.into());
        assert_eq!(agent.evaluate(&won).value, -1.0);
        let mut out = Evaluations::default();
        agent.evaluate_batch(&[BitState::initial(), won, x_wins_in_one()], &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out.value(1), -1.0);
        let nonzero = |i: usize| out.priors(i).unwrap().iter().filter(|&&p| p > 0.0).count();
        assert_eq!(nonzero(0), 81);
        assert_eq!(nonzero(1), 0, "terminal states get no priors");
        assert_eq!(nonzero(2), 7);
        assert!((out.priors(2).unwrap().iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!((0..3).all(|i| (-1.0..=1.0).contains(&out.value(i))));
    }

    #[test]
    fn begin_and_finish_match_the_synchronous_batch() {
        let _device = crate::nn::device_lock();
        let mut agent = untrained();
        let states = [BitState::initial(), x_wins_in_one()];
        let mut sync = Evaluations::default();
        agent.evaluate_batch(&states, &mut sync);
        let mut split = Evaluations::default();
        agent.begin_batch(&states);
        agent.finish_batch(&states, &mut split);
        assert_eq!(sync, split);
        agent.begin_batch(&[]);
        agent.finish_batch(&[], &mut split);
        assert_eq!(split.len(), 2);
    }

    #[test]
    fn works_as_mcts_leaf_evaluator_with_priors() {
        let _device = crate::nn::device_lock();
        let mut mcts = Mcts::new(
            untrained(),
            MctsConfig {
                simulations: 64,
                batch_size: 8,
                ..Default::default()
            },
        );
        let r = mcts.search(&x_wins_in_one());
        assert_eq!(r.best_move(), Some(X_WINNING_MOVE));
    }
}
