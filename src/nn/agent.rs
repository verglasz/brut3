//! [`ValueNetAgent`]: plays and evaluates with a trained [`ValueNet`].

use std::path::Path;

use burn::{Dispatch, DispatchDevice};

use super::backend::BackendKind;
use super::encode::encode_batch;
use super::model::ValueNet;
use crate::agent::{Evaluation, UtttAgent};
use crate::game::{BitState, GameState, Move, MoveSet};

pub struct ValueNetAgent {
    model: ValueNet<Dispatch>,
    device: DispatchDevice,
    backend: BackendKind,
}

impl ValueNetAgent {
    pub fn new(model: ValueNet<Dispatch>, backend: BackendKind) -> Self {
        Self {
            model,
            device: backend.device(),
            backend,
        }
    }

    /// Loads weights saved by [`ValueNet::save`] onto `backend`'s default device.
    pub fn load(path: &Path, backend: BackendKind) -> Result<Self, String> {
        let device = backend.device();
        let model = ValueNet::<Dispatch>::load(path, &device)
            .map_err(|e| format!("cannot load value net from {}: {e}", path.display()))?;
        Ok(Self {
            model,
            device,
            backend,
        })
    }

    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Network values for `states` (side-to-move perspective), one forward pass.
    pub fn values(&self, states: &[BitState]) -> Vec<f32> {
        self.model.values(&encode_batch(states), &self.device)
    }

    /// Exact value for terminal states, network value otherwise.
    fn value_or_exact(state: &BitState, net: f32) -> f32 {
        match state.outcome() {
            Some(outcome) => outcome.value_for(state.to_move()),
            None => net,
        }
    }
}

impl UtttAgent for ValueNetAgent {
    type State = BitState;

    /// One-ply lookahead: evaluates every child in a single batch and plays the move whose
    /// resulting position is worst for the opponent.
    fn choose_move(&mut self, state: &BitState) -> Move {
        let legal = state.legal_moves();
        assert!(!legal.is_empty(), "choose_move called on a terminal state");
        let moves: Vec<Move> = legal.iter().collect();
        let children: Vec<BitState> = moves
            .iter()
            .map(|&mv| {
                let mut child = *state;
                child.apply(mv);
                child
            })
            .collect();
        let values = self.values(&children);
        let mut best = moves[0];
        let mut best_value = f32::NEG_INFINITY;
        for ((mv, child), net) in moves.iter().zip(&children).zip(values) {
            // The child's value is from the opponent's view; negate for ours.
            let value = -Self::value_or_exact(child, net);
            if value > best_value {
                best_value = value;
                best = *mv;
            }
        }
        best
    }

    fn evaluate(&mut self, state: &BitState) -> Evaluation {
        let net = self.values(std::slice::from_ref(state))[0];
        Evaluation::value_only(Self::value_or_exact(state, net))
    }

    fn evaluate_batch(&mut self, states: &[BitState], out: &mut Vec<Evaluation>) {
        let values = self.values(states);
        out.extend(
            states
                .iter()
                .zip(values)
                .map(|(s, net)| Evaluation::value_only(Self::value_or_exact(s, net))),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::test_positions::{X_WINNING_MOVE, x_wins_in_one};
    use crate::mcts::{Mcts, MctsConfig};
    use crate::nn::ValueNetConfig;

    fn untrained() -> ValueNetAgent {
        let device = BackendKind::NdArray.device();
        ValueNetAgent::new(
            ValueNetConfig::default().init(&device),
            BackendKind::NdArray,
        )
    }

    #[test]
    fn takes_an_immediate_win_even_untrained() {
        let _device = crate::nn::device_lock();
        let mut agent = untrained();
        assert_eq!(agent.choose_move(&x_wins_in_one()), X_WINNING_MOVE);
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
        assert!(
            out.iter()
                .all(|e| (-1.0..=1.0).contains(&e.value) && e.priors.is_empty())
        );
    }

    #[test]
    fn works_as_mcts_leaf_evaluator_in_batches() {
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
