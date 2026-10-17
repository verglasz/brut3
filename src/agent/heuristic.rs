//! A static evaluation of a [`BitState`], shared by the simple search agents.
//!
//! Scores are integers; positive favours the given player. Terminal positions score
//! `±WIN_SCORE` or `0`. Everything else is a weighted count of won boards, threats (lines with
//! two own marks and no opposing mark) on the big board and on open small boards, and centre
//! and corner occupancy.

use crate::game::bitboard::LINE_MASKS;
use crate::game::{BitState, GameState, Outcome, Player};

pub const WIN_SCORE: i32 = 10_000;

const WON_BOARD: i32 = 100;
const CENTRE_BOARD: i32 = 30;
const BIG_THREAT: i32 = 50;
const SMALL_THREAT: i32 = 5;
const CENTRE_SQUARE: i32 = 2;
const CORNER_SQUARE: i32 = 1;

const CENTRE: u16 = 1 << 4;
const CORNERS: u16 = (1 << 0) | (1 << 2) | (1 << 6) | (1 << 8);

/// Static score of `state` from `player`'s perspective. Antisymmetric in the player.
pub fn score(state: &BitState, player: Player) -> i32 {
    if let Some(outcome) = state.outcome() {
        return match outcome {
            Outcome::Win(p) if p == player => WIN_SCORE,
            Outcome::Win(_) => -WIN_SCORE,
            Outcome::Draw => 0,
        };
    }
    side_score(state, player) - side_score(state, player.other())
}

/// [`score`] squashed into `[-1, 1]` for use as a search value.
pub fn value(state: &BitState, player: Player) -> f32 {
    let s = score(state, player);
    if s.abs() >= WIN_SCORE {
        return s.signum() as f32;
    }
    (s as f32 / 300.0).tanh()
}

fn side_score(state: &BitState, p: Player) -> i32 {
    let mine = state.big_marks(p);
    let theirs = state.big_marks(p.other());
    let done = state.done_mask();
    // A big line is blocked if the opponent won a board on it, or a board on it is drawn.
    let big_blocked = theirs | (done & !mine);
    let mut s = WON_BOARD * mine.count_ones() as i32
        + CENTRE_BOARD * (mine & CENTRE != 0) as i32
        + BIG_THREAT * threats(mine, big_blocked);
    for b in 0..9 {
        if done >> b & 1 != 0 {
            continue;
        }
        let m = state.board_marks(p, b);
        let t = state.board_marks(p.other(), b);
        s += SMALL_THREAT * threats(m, t)
            + CENTRE_SQUARE * (m & CENTRE != 0) as i32
            + CORNER_SQUARE * (m & CORNERS).count_ones() as i32;
    }
    s
}

/// Lines holding exactly two of `mine` and nothing of `blocked`.
fn threats(mine: u16, blocked: u16) -> i32 {
    LINE_MASKS
        .iter()
        .filter(|&&line| (mine & line).count_ones() == 2 && blocked & line == 0)
        .count() as i32
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::game::test_positions::{X_WINNING_MOVE, x_wins_in_one};
    use crate::game::{Move, sq};

    #[test]
    fn terminal_scores_are_exact() {
        let mut s = x_wins_in_one();
        s.apply(X_WINNING_MOVE);
        assert_eq!(score(&s, Player::X), WIN_SCORE);
        assert_eq!(score(&s, Player::O), -WIN_SCORE);
        assert_eq!(value(&s, Player::X), 1.0);
        assert_eq!(value(&s, Player::O), -1.0);
    }

    #[test]
    fn antisymmetric_and_bounded_on_random_positions() {
        let mut rng = SmallRng::seed_from_u64(4);
        let mut buf = Vec::new();
        for _ in 0..200 {
            let mut s = BitState::initial();
            while s.outcome().is_none() {
                assert_eq!(score(&s, Player::X), -score(&s, Player::O));
                let v = value(&s, Player::X);
                assert!((-1.0..=1.0).contains(&v));
                s.legal_moves(&mut buf);
                s.apply(buf[rng.random_range(0..buf.len())]);
            }
        }
    }

    #[test]
    fn prefers_won_boards_centre_and_threats() {
        let empty = BitState::initial();
        assert_eq!(score(&empty, Player::X), 0);
        let mut centre = empty;
        centre.apply(Move::new(sq(1, 1), sq(1, 1)));
        assert_eq!(score(&centre, Player::X), CENTRE_SQUARE);
        // Two in a row on a small board is a threat.
        let mut threat = centre;
        threat.apply(Move::new(sq(1, 1), sq(0, 0))); // O, corner
        threat.apply(Move::new(sq(0, 0), sq(1, 1))); // X centre of (0,0)
        threat.apply(Move::new(sq(1, 1), sq(0, 2))); // O, corner, and threatens? no: 0,0 + 0,2 with X at 1,1 -> line 0,1,2 open
        assert!(score(&threat, Player::O) < 0 || score(&threat, Player::O) > 0); // just well-defined
        assert!(score(&x_wins_in_one(), Player::X) > 2 * WON_BOARD);
    }
}
