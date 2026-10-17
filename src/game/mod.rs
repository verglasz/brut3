//! Game rules and the explicit types shared by every agent.
//!
//! Everything here is deliberately plain: coordinates are (row, col) pairs, a move is a
//! (board, square) pair, legal moves are a list. Packed representations live inside the
//! individual [`GameState`] implementations and never leak through these types.

pub mod bitboard;
pub mod position;

pub use bitboard::BitState;
pub use position::Position;

use std::fmt;

/// One of the two players. `X` moves first.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Player {
    X,
    O,
}

impl Player {
    pub const fn other(self) -> Self {
        match self {
            Player::X => Player::O,
            Player::O => Player::X,
        }
    }

    pub const fn symbol(self) -> char {
        match self {
            Player::X => 'X',
            Player::O => 'O',
        }
    }

    /// 0 for `X`, 1 for `O`; for implementations that index arrays by player.
    pub const fn index(self) -> usize {
        match self {
            Player::X => 0,
            Player::O => 1,
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// A coordinate on a 3x3 grid. Used both for "which small board" and "which square in it".
///
/// `row` and `col` are always in `0..3`; `row 0` is the top, `col 0` the left.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct Square {
    row: u8,
    col: u8,
}

impl Square {
    /// All nine squares in row-major order.
    pub const ALL: [Square; 9] = {
        let mut all = [Square { row: 0, col: 0 }; 9];
        let mut i = 0;
        while i < 9 {
            all[i] = Square::from_index(i);
            i += 1;
        }
        all
    };

    /// `None` if either coordinate is out of `0..3`.
    pub const fn new(row: u8, col: u8) -> Option<Self> {
        if row < 3 && col < 3 {
            Some(Self { row, col })
        } else {
            None
        }
    }

    pub const fn row(self) -> u8 {
        self.row
    }

    pub const fn col(self) -> u8 {
        self.col
    }

    /// Row-major index in `0..9`, for implementations that index arrays by square.
    pub const fn index(self) -> usize {
        (self.row * 3 + self.col) as usize
    }

    /// Inverse of [`Square::index`]. Panics if `index >= 9`.
    pub const fn from_index(index: usize) -> Self {
        assert!(index < 9, "square index out of range");
        Self {
            row: (index / 3) as u8,
            col: (index % 3) as u8,
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}c{}", self.row, self.col)
    }
}

/// A move: which small board, and which square inside it.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct Move {
    pub board: Square,
    pub square: Square,
}

impl Move {
    pub const fn new(board: Square, square: Square) -> Self {
        Self { board, square }
    }

    /// Flat index in `0..81` (`board * 9 + square`), for implementations that need it.
    pub const fn index(self) -> usize {
        self.board.index() * 9 + self.square.index()
    }

    /// Inverse of [`Move::index`]. Panics if `index >= 81`.
    pub const fn from_index(index: usize) -> Self {
        assert!(index < 81, "move index out of range");
        Self::new(Square::from_index(index / 9), Square::from_index(index % 9))
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.board, self.square)
    }
}

/// Result of a finished game.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Outcome {
    Win(Player),
    Draw,
}

impl Outcome {
    /// `1.0` if `player` won, `-1.0` if they lost, `0.0` for a draw.
    pub fn value_for(self, player: Player) -> f32 {
        match self {
            Outcome::Win(p) if p == player => 1.0,
            Outcome::Win(_) => -1.0,
            Outcome::Draw => 0.0,
        }
    }
}

/// Status of a single small board.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum BoardStatus {
    #[default]
    Open,
    Won(Player),
    /// Full with no winner: counts for nobody on the big board.
    Full,
}

impl BoardStatus {
    /// Won or full: no further moves are allowed on this board, and it no longer forces anything.
    pub fn is_decided(self) -> bool {
        !matches!(self, BoardStatus::Open)
    }
}

/// The eight winning lines of a 3x3 grid, as row-major cell indices.
pub const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

/// Winner of a 3x3 grid given its nine cells in row-major order.
pub fn line_winner(cells: &[Option<Player>; 9]) -> Option<Player> {
    LINES.iter().find_map(|line| {
        let p = cells[line[0]]?;
        (cells[line[1]] == Some(p) && cells[line[2]] == Some(p)).then_some(p)
    })
}

/// Status of a small board given its nine cells in row-major order.
pub fn board_status_of(cells: &[Option<Player>; 9]) -> BoardStatus {
    if let Some(p) = line_winner(cells) {
        BoardStatus::Won(p)
    } else if cells.iter().all(Option::is_some) {
        BoardStatus::Full
    } else {
        BoardStatus::Open
    }
}

/// A full game state under the standard rules:
///
/// - the square of the previous move names the board the next player must play on;
/// - if that board is decided (won or full), the next player may play on any undecided board;
/// - squares of a decided board are dead;
/// - the game ends with a three-in-a-row of won boards, or a draw once all boards are decided.
///
/// Implementations choose their own internal representation; [`Position`] is the readable
/// reference, [`BitState`] the fast one.
pub trait GameState: Clone {
    /// The empty board, `X` to move.
    fn initial() -> Self;

    fn from_position(position: &Position) -> Self;
    fn to_position(&self) -> Position;

    fn to_move(&self) -> Player;

    /// Clears `out` and fills it with every legal move. Empty only in terminal states.
    fn legal_moves(&self, out: &mut Vec<Move>);

    /// Plays `mv` for the side to move. Must be legal (checked in debug builds).
    fn apply(&mut self, mv: Move);

    /// `None` while the game is ongoing.
    fn outcome(&self) -> Option<Outcome>;

    fn is_terminal(&self) -> bool {
        self.outcome().is_some()
    }

    /// Allocating convenience wrapper around [`GameState::legal_moves`].
    fn legal_moves_vec(&self) -> Vec<Move> {
        let mut out = Vec::new();
        self.legal_moves(&mut out);
        out
    }
}

#[cfg(test)]
pub(crate) fn sq(row: u8, col: u8) -> Square {
    Square::new(row, col).expect("test square in range")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_index_roundtrip() {
        for (i, s) in Square::ALL.iter().enumerate() {
            assert_eq!(s.index(), i);
            assert_eq!(Square::from_index(i), *s);
        }
        assert_eq!(Square::new(3, 0), None);
        assert_eq!(Square::new(0, 3), None);
    }

    #[test]
    fn move_index_roundtrip() {
        for i in 0..81 {
            assert_eq!(Move::from_index(i).index(), i);
        }
        let mv = Move::new(sq(2, 1), sq(0, 2));
        assert_eq!(mv.index(), 7 * 9 + 2);
        assert_eq!(mv.to_string(), "r2c1/r0c2");
    }

    #[test]
    fn outcome_values() {
        assert_eq!(Outcome::Win(Player::X).value_for(Player::X), 1.0);
        assert_eq!(Outcome::Win(Player::X).value_for(Player::O), -1.0);
        assert_eq!(Outcome::Draw.value_for(Player::O), 0.0);
    }

    #[test]
    fn line_winner_and_status() {
        let mut cells = [None; 9];
        assert_eq!(board_status_of(&cells), BoardStatus::Open);
        for line in LINES {
            let mut c = [None; 9];
            for i in line {
                c[i] = Some(Player::O);
            }
            assert_eq!(line_winner(&c), Some(Player::O));
        }
        // X O X / X O O / O X X : full, no winner
        let x = Some(Player::X);
        let o = Some(Player::O);
        cells = [x, o, x, x, o, o, o, x, x];
        assert_eq!(board_status_of(&cells), BoardStatus::Full);
    }
}
