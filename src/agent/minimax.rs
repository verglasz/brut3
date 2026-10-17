//! [`MinimaxAgent`]: depth-limited alpha-beta search on the static heuristic.

use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::{Evaluation, UtttAgent, heuristic};
use crate::game::{BitState, GameState, Move};

pub struct MinimaxAgent {
    depth: u32,
    rng: SmallRng,
    buf: Vec<Move>,
}

impl MinimaxAgent {
    pub const DEFAULT_DEPTH: u32 = 4;

    /// `depth` plies of search (at least 1); leaves are scored with [`heuristic::score`].
    pub fn new(depth: u32, seed: u64) -> Self {
        Self {
            depth: depth.max(1),
            rng: SmallRng::seed_from_u64(seed),
            buf: Vec::new(),
        }
    }

    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Score of `state` for its side to move, searching `depth` more plies.
    fn negamax(state: &BitState, depth: u32, mut alpha: i32, beta: i32) -> i32 {
        if depth == 0 || state.is_terminal() {
            return heuristic::score(state, state.to_move());
        }
        let mut moves = Vec::with_capacity(9);
        state.legal_moves(&mut moves);
        let mut best = i32::MIN;
        for mv in moves {
            let mut next = *state;
            next.apply(mv);
            let v = -Self::negamax(&next, depth - 1, -beta, -alpha);
            if v > best {
                best = v;
            }
            if v > alpha {
                alpha = v;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }
}

impl UtttAgent for MinimaxAgent {
    type State = BitState;

    fn choose_move(&mut self, state: &BitState) -> Move {
        state.legal_moves(&mut self.buf);
        assert!(
            !self.buf.is_empty(),
            "choose_move called on a terminal state"
        );
        // Root moves are searched with a full window so that ties are exact and can be broken
        // at random.
        let mut best = self.buf[0];
        let mut best_score = i32::MIN;
        let mut ties = 0u32;
        for &mv in &self.buf {
            let mut next = *state;
            next.apply(mv);
            let s = -Self::negamax(&next, self.depth - 1, i32::MIN + 1, i32::MAX);
            if s > best_score {
                best_score = s;
                best = mv;
                ties = 1;
            } else if s == best_score {
                ties += 1;
                if self.rng.random_range(0..ties) == 0 {
                    best = mv;
                }
            }
        }
        best
    }

    /// The static heuristic value of the position itself (no search), so this stays cheap
    /// enough to serve as a leaf evaluator for [`crate::mcts::Mcts`].
    fn evaluate(&mut self, state: &BitState) -> Evaluation {
        Evaluation::value_only(heuristic::value(state, state.to_move()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::RandomAgent;
    use crate::game::test_positions::{X_WINNING_MOVE, o_must_block, x_wins_in_one};
    use crate::game::{Outcome, Player, sq};

    #[test]
    fn takes_the_win_and_blocks() {
        for depth in 1..=4 {
            let mut m = MinimaxAgent::new(depth, 1);
            assert_eq!(
                m.choose_move(&x_wins_in_one()),
                X_WINNING_MOVE,
                "depth {depth}"
            );
            assert_eq!(
                m.choose_move(&o_must_block()),
                X_WINNING_MOVE,
                "depth {depth}"
            );
        }
    }

    #[test]
    fn sees_that_the_block_is_forced_only_by_search() {
        // Not blocking loses by force: X gets a free move and wins. Depth 2 sees it.
        let state = o_must_block();
        let mut next = state;
        next.apply(Move::new(sq(0, 2), sq(2, 2)));
        assert!(next.legal_moves_vec().contains(&X_WINNING_MOVE));
        let v = -MinimaxAgent::negamax(&next, 1, i32::MIN + 1, i32::MAX);
        assert_eq!(v, -heuristic::WIN_SCORE);
    }

    #[test]
    fn beats_random() {
        let mut m = MinimaxAgent::new(2, 1);
        let mut r = RandomAgent::<BitState>::new(2);
        let mut wins = 0;
        let games = 10;
        for game in 0..games {
            let me = if game % 2 == 0 { Player::X } else { Player::O };
            let mut s = BitState::initial();
            let outcome = loop {
                if let Some(o) = s.outcome() {
                    break o;
                }
                let mv = if s.to_move() == me {
                    m.choose_move(&s)
                } else {
                    r.choose_move(&s)
                };
                s.apply(mv);
            };
            if outcome == Outcome::Win(me) {
                wins += 1;
            }
        }
        assert!(wins >= 9, "minimax won {wins}/{games}");
    }
}
