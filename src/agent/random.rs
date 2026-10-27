//! [`RandomAgent`]: uniformly random legal moves.

use std::marker::PhantomData;

use rand::{SeedableRng, rngs::SmallRng};

use super::{Evaluation, UtttAgent};
use crate::game::{GameState, Move};

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
        assert!(
            !state.is_terminal(),
            "choose_move called on a terminal state"
        );
        state.random_move(&mut self.rng).into()
    }

    /// A random playout to the end, in the state's own moves ([`GameState::random_move`] and
    /// [`GameState::play`]): monomorphised for a state, it is that state's specialised loop.
    fn evaluate(&mut self, state: &S) -> Evaluation {
        let me = state.to_move();
        if let Some(outcome) = state.outcome() {
            return Evaluation::value_only(outcome.value_for(me));
        }
        let mut s = state.clone();
        loop {
            let mv = s.random_move(&mut self.rng);
            if let Some(outcome) = s.play(mv) {
                return Evaluation::value_only(outcome.value_for(me));
            }
        }
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
            state.apply(mv.into());
        }
    }
}
