//! [`RandomAgent`]: uniformly random legal moves.

use std::marker::PhantomData;

use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::UtttAgent;
use crate::game::{GameState, Move};

/// Plays a uniformly random legal move. The weakest possible baseline, and the default
/// playout policy when used as the inner agent of [`crate::mcts::Mcts`].
pub struct RandomAgent<S> {
    rng: SmallRng,
    buf: Vec<Move>,
    _state: PhantomData<S>,
}

impl<S> RandomAgent<S> {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            buf: Vec::new(),
            _state: PhantomData,
        }
    }
}

impl<S: GameState> UtttAgent for RandomAgent<S> {
    type State = S;

    fn choose_move(&mut self, state: &S) -> Move {
        state.legal_moves(&mut self.buf);
        assert!(
            !self.buf.is_empty(),
            "choose_move called on a terminal state"
        );
        self.buf[self.rng.random_range(0..self.buf.len())]
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
