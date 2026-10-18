//! [`RandomAgent`]: uniformly random legal moves.

use std::marker::PhantomData;

use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::UtttAgent;
use crate::game::{GameState, Move, MoveSet};

/// Plays a uniformly random legal move. The weakest possible baseline, and the default
/// playout policy when used as the inner agent of [`crate::mcts::Mcts`].
pub struct RandomAgent<S> {
    rng: SmallRng,
    _state: PhantomData<S>,
}

impl<S> RandomAgent<S> {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            _state: PhantomData,
        }
    }
}

impl<S: GameState> UtttAgent for RandomAgent<S> {
    type State = S;

    fn choose_move(&mut self, state: &S) -> Move {
        let legal = state.legal_moves();
        assert!(!legal.is_empty(), "choose_move called on a terminal state");
        let k = self.rng.random_range(0..legal.len());
        legal.iter().nth(k).expect("k < len")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::BitState;

    #[test]
    fn only_plays_legal_moves_and_is_seeded() {
        let mut a = RandomAgent::<BitState>::new(5);
        let mut b = RandomAgent::<BitState>::new(5);
        let mut state = BitState::initial();
        while state.outcome().is_none() {
            let mv = a.choose_move(&state);
            assert_eq!(mv, b.choose_move(&state));
            assert!(state.is_legal(mv));
            state.apply(mv);
        }
    }
}
