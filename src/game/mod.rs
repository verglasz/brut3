//! Game rules and the explicit types shared by every agent.
//!
//! Everything here is deliberately plain: coordinates are (row, col) pairs, a move is a
//! (board, square) pair, legal moves are a list. Packed representations live inside the
//! individual [`GameState`] implementations and never leak through these types.

pub mod bitboard;
pub mod position;
pub mod symmetry;

pub use bitboard::{BitMoveSet, BitState};
pub use position::Position;
pub use symmetry::Symmetry;

use std::fmt;
use std::str::FromStr;

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

/// Error from parsing a [`Square`] or [`Move`] from text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

impl FromStr for Square {
    type Err = ParseError;

    /// Parses the `Display` form, e.g. `r1c2`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let err = || ParseError(format!("expected a square like r1c2, got {s:?}"));
        let rest = s.strip_prefix('r').ok_or_else(err)?;
        let (row, col) = rest.split_once('c').ok_or_else(err)?;
        let row: u8 = row.parse().map_err(|_| err())?;
        let col: u8 = col.parse().map_err(|_| err())?;
        Square::new(row, col).ok_or_else(err)
    }
}

/// A move: which small board, and which square inside it.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct Move {
    pub board: Square,
    pub square: Square,
}

impl Move {
    /// Number of distinct moves: one per square of the big board.
    pub const COUNT: usize = 81;

    pub const fn new(board: Square, square: Square) -> Self {
        Self { board, square }
    }

    /// Flat index in `0..81` (`board * 9 + square`), for implementations that need it.
    pub const fn index(self) -> usize {
        self.board.index() * 9 + self.square.index()
    }

    /// Inverse of [`Move::index`]. Panics if `index >= Move::COUNT`.
    pub const fn from_index(index: usize) -> Self {
        assert!(index < Self::COUNT, "move index out of range");
        Self::new(Square::from_index(index / 9), Square::from_index(index % 9))
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.board, self.square)
    }
}

impl FromStr for Move {
    type Err = ParseError;

    /// Parses the `Display` form, e.g. `r1c2/r0c0` (board, then square).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (board, square) = s
            .split_once('/')
            .ok_or_else(|| ParseError(format!("expected a move like r1c2/r0c0, got {s:?}")))?;
        Ok(Move::new(board.parse()?, square.parse()?))
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

    /// The outcome as a one-hot `[win, draw, loss]` for `player`.
    pub fn wdl_for(self, player: Player) -> [f32; 3] {
        match self {
            Outcome::Win(p) if p == player => [1.0, 0.0, 0.0],
            Outcome::Win(_) => [0.0, 0.0, 1.0],
            Outcome::Draw => [0.0, 1.0, 0.0],
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

/// A set of moves. The API speaks only in [`Move`]s; the representation is the
/// implementation's business (a bitset for [`BitMoveSet`], a plain `Vec<Move>` as reference).
pub trait MoveSet: Clone + IntoIterator<Item = Move> {
    fn empty() -> Self;
    fn contains(&self, mv: Move) -> bool;
    fn len(&self) -> usize;
    fn insert(&mut self, mv: Move);
    fn remove(&mut self, mv: Move);
    fn iter(&self) -> impl Iterator<Item = Move> + '_;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn to_vec(&self) -> Vec<Move> {
        self.iter().collect()
    }
}

/// Reference [`MoveSet`]: a plain list without duplicates. Used by [`Position`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MoveList(Vec<Move>);

impl MoveList {
    pub fn as_slice(&self) -> &[Move] {
        &self.0
    }
}

impl MoveSet for MoveList {
    fn empty() -> Self {
        Self(Vec::new())
    }

    fn contains(&self, mv: Move) -> bool {
        self.0.contains(&mv)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn insert(&mut self, mv: Move) {
        if !self.0.contains(&mv) {
            self.0.push(mv);
        }
    }

    fn remove(&mut self, mv: Move) {
        self.0.retain(|m| *m != mv);
    }

    fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.0.iter().copied()
    }
}

impl IntoIterator for MoveList {
    type Item = Move;
    type IntoIter = std::vec::IntoIter<Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<Move> for MoveList {
    fn from_iter<I: IntoIterator<Item = Move>>(iter: I) -> Self {
        let mut list = Self::empty();
        for mv in iter {
            list.insert(mv);
        }
        list
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
    /// How this implementation represents a set of moves.
    type Moves: MoveSet;

    /// The empty board, `X` to move.
    fn initial() -> Self;

    fn from_position(position: &Position) -> Self;
    fn to_position(&self) -> Position;

    fn to_move(&self) -> Player;

    /// Every legal move. Empty only in terminal states.
    fn legal_moves(&self) -> Self::Moves;

    /// Plays `mv` for the side to move. Must be legal (checked in debug builds).
    fn apply(&mut self, mv: Move);

    /// `None` while the game is ongoing.
    fn outcome(&self) -> Option<Outcome>;

    /// The same position seen through a symmetry of the board: every mark, the forced board
    /// and the move history move with it, the side to move and the outcome do not.
    fn transform(&self, symmetry: Symmetry) -> Self;

    fn is_terminal(&self) -> bool {
        self.outcome().is_some()
    }

    /// [`GameState::legal_moves`] as a `Vec`, for tests and display.
    fn legal_moves_vec(&self) -> Vec<Move> {
        self.legal_moves().to_vec()
    }
}

#[cfg(test)]
pub(crate) fn sq(row: u8, col: u8) -> Square {
    Square::new(row, col).expect("test square in range")
}

/// Positions shared by tests across modules.
#[cfg(test)]
pub(crate) mod test_positions {
    use super::*;

    /// X has won boards (0,0) and (0,1), has two in a row on (0,2), and is forced to play
    /// there. `r0c2/r0c2` wins on the spot; 7 legal moves.
    pub fn x_wins_in_one() -> BitState {
        let mut p = Position::initial();
        for b in [sq(0, 0), sq(0, 1)] {
            for s in [sq(1, 0), sq(1, 1), sq(1, 2)] {
                p.set_cell(Move::new(b, s), Some(Player::X));
            }
        }
        p.set_cell(Move::new(sq(0, 2), sq(0, 0)), Some(Player::X));
        p.set_cell(Move::new(sq(0, 2), sq(0, 1)), Some(Player::X));
        // Some O marks so O's last move (into square (0,2) of board (1,1)) is plausible.
        for s in [sq(2, 0), sq(2, 1), sq(0, 2)] {
            p.set_cell(Move::new(sq(1, 1), s), Some(Player::O));
        }
        p.set_last_move(Some(Move::new(sq(1, 1), sq(0, 2))));
        p.set_to_move(Player::X);
        let state = BitState::from_position(&p);
        assert_eq!(state.outcome(), None);
        assert_eq!(state.legal_moves_vec().len(), 7);
        state
    }

    pub const X_WINNING_MOVE: Move = Move::new(Square::from_index(2), Square::from_index(2));

    /// Same as [`x_wins_in_one`] but with O to move, forced to board (0,2), where X threatens
    /// to win the game on square (0,2). Blocking is the obvious move, but not forced: O can
    /// instead send X to another board.
    pub fn o_threatened() -> BitState {
        let mut p = x_wins_in_one().to_position();
        p.set_to_move(Player::O);
        BitState::from_position(&p)
    }

    /// O to move, forced to board (0,2), where X has `r0c0 r0c1` and O holds the centre;
    /// every other board is decided with no big line, so any non-blocking move gives X a free
    /// move and the game, while blocking holds (O can parry X's one remaining line).
    ///
    /// Big board: `X X .` / `O X O` / `O # O`.
    pub fn o_must_block() -> BitState {
        let mut p = Position::initial();
        let won = [
            (sq(0, 0), Player::X),
            (sq(0, 1), Player::X),
            (sq(1, 0), Player::O),
            (sq(1, 1), Player::X),
            (sq(1, 2), Player::O),
            (sq(2, 0), Player::O),
            (sq(2, 2), Player::O),
        ];
        for (b, w) in won {
            for s in [sq(1, 0), sq(1, 1), sq(1, 2)] {
                p.set_cell(Move::new(b, s), Some(w));
            }
        }
        let x = Some(Player::X);
        let o = Some(Player::O);
        for (i, v) in [x, o, x, x, o, o, o, x, x].into_iter().enumerate() {
            p.set_cell(Move::new(sq(2, 1), Square::from_index(i)), v);
        }
        p.set_cell(Move::new(sq(0, 2), sq(0, 0)), Some(Player::X));
        p.set_cell(Move::new(sq(0, 2), sq(0, 1)), Some(Player::X));
        p.set_cell(Move::new(sq(0, 2), sq(1, 1)), Some(Player::O));
        // X's last move went to square (0,2) of a board X owns, sending O to board (0,2).
        p.set_cell(Move::new(sq(1, 1), sq(0, 2)), Some(Player::X));
        p.set_last_move(Some(Move::new(sq(1, 1), sq(0, 2))));
        p.set_to_move(Player::O);
        let state = BitState::from_position(&p);
        assert_eq!(state.outcome(), None);
        assert_eq!(state.legal_moves_vec().len(), 6);
        state
    }
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
    fn parse_square_and_move() {
        assert_eq!("r2c1".parse::<Square>(), Ok(sq(2, 1)));
        assert!("r3c1".parse::<Square>().is_err());
        assert!("2,1".parse::<Square>().is_err());
        let mv = Move::new(sq(2, 1), sq(0, 2));
        assert_eq!(mv.to_string().parse::<Move>(), Ok(mv));
        assert!("r2c1".parse::<Move>().is_err());
        assert!("r2c1/r0c3".parse::<Move>().is_err());
    }

    #[test]
    fn move_list_is_a_set() {
        let a = Move::from_index(3);
        let b = Move::from_index(80);
        let mut set = MoveList::empty();
        assert!(set.is_empty());
        set.insert(a);
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 2);
        assert!(set.contains(a) && set.contains(b));
        set.remove(a);
        assert_eq!(set.to_vec(), vec![b]);
        let from: MoveList = [a, b, a].into_iter().collect();
        assert_eq!(from.as_slice(), &[a, b]);
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
