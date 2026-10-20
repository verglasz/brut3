//! [`NetAgent`]: plays and evaluates with a trained policy/value network.

use std::path::Path;

use burn::DispatchDevice;

use super::any_model::AnyModel;
use super::backend::BackendKind;
use super::model::{HostPrediction, expected_value};
use crate::agent::{Evaluation, UtttAgent};
use crate::game::{BitState, GameState, Move};

pub struct NetAgent {
    model: AnyModel,
    device: DispatchDevice,
    backend: BackendKind,
}

impl NetAgent {
    pub fn new(model: AnyModel, backend: BackendKind) -> Self {
        Self {
            model,
            device: backend.device(),
            backend,
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

    fn evaluate_batch(&mut self, states: &[BitState], out: &mut Vec<Evaluation>) {
        let p = self.predict(states);
        out.extend(
            states
                .iter()
                .zip(p.wdl.iter().zip(p.policy))
                .map(|(s, (wdl, policy))| Self::evaluation(s, wdl, policy)),
        );
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
        won.apply(X_WINNING_MOVE);
        assert_eq!(agent.evaluate(&won).value, -1.0);
        let mut out = Vec::new();
        agent.evaluate_batch(&[BitState::initial(), won, x_wins_in_one()], &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].value, -1.0);
        assert!(out[1].priors.is_empty());
        assert_eq!(out[0].priors.len(), 81);
        assert_eq!(out[2].priors.len(), 7);
        assert!(out.iter().all(|e| (-1.0..=1.0).contains(&e.value)));
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
