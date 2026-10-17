//! [`Position`]: the readable reference implementation of the rules.
//!
//! Plain arrays of `Option<Player>`, brute-force line checks. Slow but obvious; used for
//! display, tests, and as the conversion hub between faster representations.

use std::fmt;

use super::{BoardStatus, GameState, Move, Outcome, Player, Square, board_status_of, line_winner};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Position {
    /// `cells[board.index()][square.index()]`.
    cells: [[Option<Player>; 9]; 9],
    to_move: Player,
    last_move: Option<Move>,
}

impl Default for Position {
    fn default() -> Self {
        Self::initial()
    }
}

impl Position {
    pub fn cell(&self, mv: Move) -> Option<Player> {
        self.cells[mv.board.index()][mv.square.index()]
    }

    pub fn board_cells(&self, board: Square) -> &[Option<Player>; 9] {
        &self.cells[board.index()]
    }

    pub fn board_status(&self, board: Square) -> BoardStatus {
        board_status_of(self.board_cells(board))
    }

    pub fn last_move(&self) -> Option<Move> {
        self.last_move
    }

    /// The board the side to move is restricted to, if any.
    pub fn forced_board(&self) -> Option<Square> {
        let board = self.last_move?.square;
        (!self.board_status(board).is_decided()).then_some(board)
    }

    /// The big board: each small board's winner, `None` for open or full boards.
    pub fn big_board(&self) -> [Option<Player>; 9] {
        Square::ALL.map(|b| match self.board_status(b) {
            BoardStatus::Won(p) => Some(p),
            _ => None,
        })
    }

    // --- Setup helpers. These bypass the rules; meant for tests and position editors. ---

    pub fn set_cell(&mut self, mv: Move, value: Option<Player>) {
        self.cells[mv.board.index()][mv.square.index()] = value;
    }

    pub fn set_to_move(&mut self, player: Player) {
        self.to_move = player;
    }

    pub fn set_last_move(&mut self, mv: Option<Move>) {
        self.last_move = mv;
    }
}

impl GameState for Position {
    fn initial() -> Self {
        Self {
            cells: [[None; 9]; 9],
            to_move: Player::X,
            last_move: None,
        }
    }

    fn from_position(position: &Position) -> Self {
        position.clone()
    }

    fn to_position(&self) -> Position {
        self.clone()
    }

    fn to_move(&self) -> Player {
        self.to_move
    }

    fn legal_moves(&self, out: &mut Vec<Move>) {
        out.clear();
        let boards: Vec<Square> = match self.forced_board() {
            Some(b) => vec![b],
            None => Square::ALL
                .into_iter()
                .filter(|b| !self.board_status(*b).is_decided())
                .collect(),
        };
        for board in boards {
            for square in Square::ALL {
                let mv = Move::new(board, square);
                if self.cell(mv).is_none() {
                    out.push(mv);
                }
            }
        }
    }

    fn apply(&mut self, mv: Move) {
        debug_assert!(self.legal_moves_vec().contains(&mv), "illegal move {mv}");
        self.set_cell(mv, Some(self.to_move));
        self.last_move = Some(mv);
        self.to_move = self.to_move.other();
    }

    fn outcome(&self) -> Option<Outcome> {
        if let Some(p) = line_winner(&self.big_board()) {
            return Some(Outcome::Win(p));
        }
        if Square::ALL
            .iter()
            .all(|b| self.board_status(*b).is_decided())
        {
            return Some(Outcome::Draw);
        }
        None
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mark = |p: Option<Player>| p.map_or('.', Player::symbol);
        for board_row in 0..3 {
            for row in 0..3 {
                for board_col in 0..3 {
                    if board_col > 0 {
                        write!(f, " | ")?;
                    }
                    let board = Square::new(board_row, board_col).unwrap();
                    for col in 0..3 {
                        let square = Square::new(row, col).unwrap();
                        write!(f, "{}", mark(self.cell(Move::new(board, square))))?;
                    }
                }
                writeln!(f)?;
            }
            if board_row < 2 {
                writeln!(f, "----+-----+----")?;
            }
        }
        let big: String = Square::ALL
            .iter()
            .map(|b| match self.board_status(*b) {
                BoardStatus::Open => '.',
                BoardStatus::Won(p) => p.symbol(),
                BoardStatus::Full => '#',
            })
            .collect();
        writeln!(f, "boards: {} {} {}", &big[0..3], &big[3..6], &big[6..9])?;
        write!(f, "to move: {}", self.to_move)?;
        match self.forced_board() {
            Some(b) => write!(f, ", forced board: {b}")?,
            None => write!(f, ", any board")?,
        }
        if let Some(o) = self.outcome() {
            write!(f, ", game over: {o:?}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::sq;

    fn mv(br: u8, bc: u8, r: u8, c: u8) -> Move {
        Move::new(sq(br, bc), sq(r, c))
    }

    #[test]
    fn initial_has_81_moves_anywhere() {
        let p = Position::initial();
        assert_eq!(p.legal_moves_vec().len(), 81);
        assert_eq!(p.to_move(), Player::X);
        assert_eq!(p.outcome(), None);
        assert_eq!(p.forced_board(), None);
    }

    #[test]
    fn square_forces_next_board() {
        let mut p = Position::initial();
        p.apply(mv(1, 1, 0, 2));
        assert_eq!(p.to_move(), Player::O);
        assert_eq!(p.forced_board(), Some(sq(0, 2)));
        let moves = p.legal_moves_vec();
        assert_eq!(moves.len(), 9);
        assert!(moves.iter().all(|m| m.board == sq(0, 2)));
    }

    #[test]
    fn occupied_square_is_not_legal() {
        let mut p = Position::initial();
        p.apply(mv(1, 1, 1, 1)); // X plays centre of centre, O is sent back to centre
        let moves = p.legal_moves_vec();
        assert_eq!(moves.len(), 8);
        assert!(!moves.contains(&mv(1, 1, 1, 1)));
    }

    #[test]
    fn won_board_is_dead_and_frees_opponent() {
        let mut p = Position::initial();
        // X wins board (0,0) on the top row; O is sent elsewhere each time.
        for (xs, os) in [((0, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (0, 0))] {
            p.set_cell(Move::new(sq(0, 0), sq(xs.0, xs.1)), Some(Player::X));
            // Put O marks in board (1,1) so the position stays plausible.
            p.set_cell(Move::new(sq(1, 1), sq(os.0, os.1)), Some(Player::O));
        }
        assert_eq!(p.board_status(sq(0, 0)), BoardStatus::Won(Player::X));
        // O's last move sent X to board (0,0), which is decided: X may play anywhere open.
        p.set_last_move(Some(mv(1, 1, 0, 0)));
        p.set_to_move(Player::X);
        assert_eq!(p.forced_board(), None);
        let moves = p.legal_moves_vec();
        assert!(moves.iter().all(|m| m.board != sq(0, 0)));
        assert_eq!(moves.len(), 81 - 9 - 2); // minus the dead board, minus O's two marks
    }

    #[test]
    fn full_board_counts_for_nobody() {
        let mut p = Position::initial();
        let x = Some(Player::X);
        let o = Some(Player::O);
        for (i, v) in [x, o, x, x, o, o, o, x, x].into_iter().enumerate() {
            p.set_cell(Move::new(sq(2, 2), Square::from_index(i)), v);
        }
        assert_eq!(p.board_status(sq(2, 2)), BoardStatus::Full);
        assert_eq!(p.big_board()[8], None);
        p.set_last_move(Some(mv(0, 0, 2, 2)));
        assert_eq!(p.forced_board(), None);
    }

    #[test]
    fn big_line_wins_the_game() {
        let mut p = Position::initial();
        for b in [sq(0, 0), sq(1, 1), sq(2, 2)] {
            for s in [sq(1, 0), sq(1, 1), sq(1, 2)] {
                p.set_cell(Move::new(b, s), Some(Player::O));
            }
        }
        assert_eq!(p.outcome(), Some(Outcome::Win(Player::O)));
    }

    #[test]
    fn all_boards_decided_without_line_is_draw() {
        let mut p = Position::initial();
        // Big board: X O X / O X O / O X O -> no line.
        let winners = [
            Player::X,
            Player::O,
            Player::X,
            Player::O,
            Player::X,
            Player::O,
            Player::O,
            Player::X,
            Player::O,
        ];
        for (b, w) in Square::ALL.into_iter().zip(winners) {
            for s in [sq(0, 0), sq(0, 1), sq(0, 2)] {
                p.set_cell(Move::new(b, s), Some(w));
            }
        }
        assert_eq!(p.outcome(), Some(Outcome::Draw));
    }

    #[test]
    fn display_renders() {
        let mut p = Position::initial();
        p.apply(mv(1, 1, 0, 2));
        let s = p.to_string();
        assert!(s.contains("----+-----+----"));
        assert!(s.contains("forced board: r0c2"));
    }
}
