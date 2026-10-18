//! Position -> feature vector, from the side to move's perspective.
//!
//! Layout (all 0/1):
//! - `0..81`     my marks, `board * 9 + square`
//! - `81..162`   opponent's marks
//! - `162..171`  boards I have won
//! - `171..180`  boards the opponent has won
//! - `180..189`  boards full with no winner
//! - `189..198`  one-hot forced board
//! - `198`       free choice of board

use crate::game::{BitState, GameState};

pub const FEATURES: usize = 199;

const THEIRS: usize = 81;
const MY_BOARDS: usize = 162;
const THEIR_BOARDS: usize = 171;
const FULL_BOARDS: usize = 180;
const FORCED: usize = 189;
const ANY_BOARD: usize = 198;

/// Writes the features of `state` into `out`, which must have length [`FEATURES`].
pub fn encode_into(state: &BitState, out: &mut [f32]) {
    assert_eq!(out.len(), FEATURES);
    out.fill(0.0);
    let me = state.to_move();
    let them = me.other();
    for board in 0..9 {
        let mine = state.board_marks(me, board);
        let theirs = state.board_marks(them, board);
        for square in 0..9 {
            if mine >> square & 1 != 0 {
                out[board * 9 + square] = 1.0;
            }
            if theirs >> square & 1 != 0 {
                out[THEIRS + board * 9 + square] = 1.0;
            }
        }
    }
    let my_boards = state.big_marks(me);
    let their_boards = state.big_marks(them);
    let done = state.done_mask();
    for board in 0..9 {
        if my_boards >> board & 1 != 0 {
            out[MY_BOARDS + board] = 1.0;
        } else if their_boards >> board & 1 != 0 {
            out[THEIR_BOARDS + board] = 1.0;
        } else if done >> board & 1 != 0 {
            out[FULL_BOARDS + board] = 1.0;
        }
    }
    match state.forced_board() {
        Some(board) => out[FORCED + board.index()] = 1.0,
        None => out[ANY_BOARD] = 1.0,
    }
}

/// Features of every state, concatenated: `states.len() * FEATURES` values.
pub fn encode_batch(states: &[BitState]) -> Vec<f32> {
    let mut out = vec![0.0; states.len() * FEATURES];
    for (state, chunk) in states.iter().zip(out.chunks_mut(FEATURES)) {
        encode_into(state, chunk);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::test_positions::x_wins_in_one;
    use crate::game::{Move, Player, Position, sq};

    #[test]
    fn initial_position() {
        let mut f = vec![0.0; FEATURES];
        encode_into(&BitState::initial(), &mut f);
        assert_eq!(f.iter().sum::<f32>(), 1.0);
        assert_eq!(f[ANY_BOARD], 1.0);
    }

    #[test]
    fn perspective_follows_side_to_move() {
        let mut s = BitState::initial();
        s.apply(Move::new(sq(1, 1), sq(0, 2))); // X at board 4, square 2; O to move, forced to 2
        let mut f = vec![0.0; FEATURES];
        encode_into(&s, &mut f);
        assert_eq!(f[THEIRS + 4 * 9 + 2], 1.0, "X's mark is 'theirs' for O");
        assert_eq!(f[4 * 9 + 2], 0.0);
        assert_eq!(f[FORCED + 2], 1.0);
        assert_eq!(f[ANY_BOARD], 0.0);

        let mut p: Position = s.to_position();
        p.set_to_move(Player::X);
        encode_into(&BitState::from_position(&p), &mut f);
        assert_eq!(f[4 * 9 + 2], 1.0, "same mark is 'mine' for X");
    }

    #[test]
    fn board_planes() {
        let s = x_wins_in_one(); // X to move, X won boards 0 and 1, forced to board 2
        let mut f = vec![0.0; FEATURES];
        encode_into(&s, &mut f);
        assert_eq!(f[MY_BOARDS], 1.0);
        assert_eq!(f[MY_BOARDS + 1], 1.0);
        assert_eq!(f[THEIR_BOARDS..FULL_BOARDS].iter().sum::<f32>(), 0.0);
        assert_eq!(f[FORCED + 2], 1.0);
        let batch = encode_batch(&[s, BitState::initial()]);
        assert_eq!(batch.len(), 2 * FEATURES);
        assert_eq!(&batch[..FEATURES], &f[..]);
    }
}
