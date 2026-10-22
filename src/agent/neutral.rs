//! [`NeutralAgent`]: the flat leaf evaluator. Every position that is not over is worth 0 (an
//! even game) with no move priors, so a search built on it learns only from the terminal
//! positions it reaches. Plays uniformly random moves when asked to play.

use super::{Evaluation, RandomAgent, UtttAgent};
use crate::game::{GameState, Move};

pub struct NeutralAgent<S> {
    moves: RandomAgent<S>,
}

impl<S> NeutralAgent<S> {
    pub fn new(seed: u64) -> Self {
        Self {
            moves: RandomAgent::new(seed),
        }
    }
}

impl<S: GameState> UtttAgent for NeutralAgent<S> {
    type State = S;

    fn choose_move(&mut self, state: &S) -> Move {
        self.moves.choose_move(state)
    }

    /// Exact for finished games, `0` for everything else, no priors.
    fn evaluate(&mut self, state: &S) -> Evaluation {
        match state.outcome() {
            Some(outcome) => Evaluation::value_only(outcome.value_for(state.to_move())),
            None => Evaluation::value_only(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::BitState;
    use crate::game::test_positions::x_wins_in_one;

    #[test]
    fn flat_value_except_at_the_end() {
        let mut agent = NeutralAgent::<BitState>::new(1);
        let e = agent.evaluate(&BitState::initial());
        assert_eq!((e.value, e.priors.len()), (0.0, 0));
        let mut won = x_wins_in_one();
        won.apply(crate::game::test_positions::X_WINNING_MOVE);
        assert_eq!(agent.evaluate(&won).value, -1.0, "O to move has lost");
        assert!(BitState::initial().is_legal(agent.choose_move(&BitState::initial())));
    }
}
