//! The [`UtttAgent`] trait: anything that can pick a move and evaluate a position.

pub mod any;
pub mod random;

pub use any::{AgentSpec, AnyAgent};
pub use random::RandomAgent;

use crate::game::{GameState, Move};

/// What an agent thinks of a position, from the perspective of the side to move.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Evaluation {
    /// Expected result in `[-1, 1]` for the side to move: `1` = certain win, `-1` = certain loss.
    pub value: f32,
    /// Prior probability per move. Empty means "uniform over legal moves". Moves not listed get
    /// prior 0; listed moves that are illegal are ignored by the search.
    pub priors: Vec<(Move, f32)>,
}

impl Evaluation {
    pub fn value_only(value: f32) -> Self {
        Self {
            value,
            priors: Vec::new(),
        }
    }
}

/// An Ultimate Tic Tac Toe agent.
///
/// The state representation is the agent's choice ([`UtttAgent::State`]); callers convert
/// through [`crate::game::Position`] when they need to talk to another agent.
///
/// Methods take `&mut self` so agents can own RNGs, caches or network handles directly.
pub trait UtttAgent {
    type State: GameState;

    /// Picks a move. `state` is never terminal, so at least one legal move exists.
    fn choose_move(&mut self, state: &Self::State) -> Move;

    /// Evaluates a position for tree search.
    ///
    /// The default plays the game out with [`UtttAgent::choose_move`] and reports the result
    /// with no priors. Terminal states are evaluated exactly.
    fn evaluate(&mut self, state: &Self::State) -> Evaluation {
        let me = state.to_move();
        let mut s = state.clone();
        loop {
            if let Some(outcome) = s.outcome() {
                return Evaluation::value_only(outcome.value_for(me));
            }
            let mv = self.choose_move(&s);
            s.apply(mv);
        }
    }

    /// Evaluates many positions at once, appending to `out`. The default calls
    /// [`UtttAgent::evaluate`] for each; agents backed by a network override it to batch.
    fn evaluate_batch(&mut self, states: &[Self::State], out: &mut Vec<Evaluation>) {
        out.reserve(states.len());
        for s in states {
            let e = self.evaluate(s);
            out.push(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{BitState, Outcome, Player, Position, sq};

    #[test]
    fn default_evaluate_is_exact_on_terminal_states() {
        let mut p = Position::initial();
        for b in [sq(0, 0), sq(0, 1), sq(0, 2)] {
            for s in [sq(2, 0), sq(2, 1), sq(2, 2)] {
                p.set_cell(Move::new(b, s), Some(Player::X));
            }
        }
        assert_eq!(p.outcome(), Some(Outcome::Win(Player::X)));
        let mut agent = RandomAgent::<BitState>::new(0);
        // O "to move" in a position X already won.
        p.set_to_move(Player::O);
        assert_eq!(agent.evaluate(&BitState::from_position(&p)).value, -1.0);
        p.set_to_move(Player::X);
        assert_eq!(agent.evaluate(&BitState::from_position(&p)).value, 1.0);
    }

    #[test]
    fn default_evaluate_rolls_out_to_a_result() {
        let mut agent = RandomAgent::<BitState>::new(0);
        let e = agent.evaluate(&BitState::initial());
        assert!(e.priors.is_empty());
        assert!([-1.0, 0.0, 1.0].contains(&e.value));
        let mut out = Vec::new();
        agent.evaluate_batch(&[BitState::initial(); 3], &mut out);
        assert_eq!(out.len(), 3);
    }
}
