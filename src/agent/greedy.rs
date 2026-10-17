//! [`GreedyAgent`]: one-ply lookahead on the static heuristic.

use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::{Evaluation, UtttAgent, heuristic};
use crate::game::{BitState, GameState, Move};

/// Plays the move whose resulting position scores best under [`heuristic::score`], breaking
/// ties at random. Takes wins and board wins when offered, but sees nothing deeper.
pub struct GreedyAgent {
    rng: SmallRng,
    buf: Vec<Move>,
}

impl GreedyAgent {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            buf: Vec::new(),
        }
    }
}

impl UtttAgent for GreedyAgent {
    type State = BitState;

    fn choose_move(&mut self, state: &BitState) -> Move {
        let me = state.to_move();
        state.legal_moves(&mut self.buf);
        assert!(
            !self.buf.is_empty(),
            "choose_move called on a terminal state"
        );
        let mut best = self.buf[0];
        let mut best_score = i32::MIN;
        let mut ties = 0u32;
        for &mv in &self.buf {
            let mut next = *state;
            next.apply(mv);
            let s = heuristic::score(&next, me);
            if s > best_score {
                best_score = s;
                best = mv;
                ties = 1;
            } else if s == best_score {
                // Reservoir sampling keeps each tied move equally likely.
                ties += 1;
                if self.rng.random_range(0..ties) == 0 {
                    best = mv;
                }
            }
        }
        best
    }

    /// The static heuristic value; no search, no priors.
    fn evaluate(&mut self, state: &BitState) -> Evaluation {
        Evaluation::value_only(heuristic::value(state, state.to_move()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::test_positions::{X_WINNING_MOVE, o_must_block, o_threatened, x_wins_in_one};

    #[test]
    fn takes_the_win() {
        let mut g = GreedyAgent::new(1);
        assert_eq!(g.choose_move(&x_wins_in_one()), X_WINNING_MOVE);
    }

    #[test]
    fn blocks_an_immediate_threat() {
        // Blocking is the only move that removes X's big-board threat and small-board threat.
        let mut g = GreedyAgent::new(1);
        assert_eq!(g.choose_move(&o_threatened()), X_WINNING_MOVE);
        assert_eq!(g.choose_move(&o_must_block()), X_WINNING_MOVE);
    }

    #[test]
    fn evaluates_statically() {
        let mut g = GreedyAgent::new(1);
        assert!(g.evaluate(&x_wins_in_one()).value > 0.5);
        assert!(g.evaluate(&o_threatened()).value < -0.5);
    }
}
