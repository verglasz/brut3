//! [`BitState`]: a compact, fast implementation of the rules.
//!
//! Each small board is a pair of 9-bit masks (one per player); the big board is another pair
//! of 9-bit masks of won boards. Win detection is a 512-entry table lookup. All of this is
//! private: the public surface is the [`GameState`] trait with its explicit [`Move`]s.

use super::{
    BoardStatus,
    GameState,
    LINES,
    Move,
    MoveSet,
    Outcome,
    Player,
    Position,
    Square,
    Symmetry,
};

/// All nine squares set.
const FULL: u16 = 0x1FF;

/// The eight lines as 9-bit masks.
pub(crate) const LINE_MASKS: [u16; 8] = {
    let mut masks = [0u16; 8];
    let mut i = 0;
    while i < 8 {
        masks[i] = (1 << LINES[i][0]) | (1 << LINES[i][1]) | (1 << LINES[i][2]);
        i += 1;
    }
    masks
};

/// `WIN[mask]` is true if the 9-bit `mask` contains a complete line.
const WIN: [bool; 512] = {
    let mut table = [false; 512];
    let mut mask = 0usize;
    while mask < 512 {
        let mut i = 0;
        while i < 8 {
            let line = LINE_MASKS[i] as usize;
            if mask & line == line {
                table[mask] = true;
                break;
            }
            i += 1;
        }
        mask += 1;
    }
    table
};

/// A set of moves as one bit per [`Move::index`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct BitMoveSet(u128);

impl BitMoveSet {
    const fn bit(mv: Move) -> u128 {
        1u128 << mv.index()
    }
}

impl MoveSet for BitMoveSet {
    fn empty() -> Self {
        Self(0)
    }

    #[inline]
    fn contains(&self, mv: Move) -> bool {
        self.0 & Self::bit(mv) != 0
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.count_ones() as usize
    }

    #[inline]
    fn insert(&mut self, mv: Move) {
        self.0 |= Self::bit(mv);
    }

    #[inline]
    fn remove(&mut self, mv: Move) {
        self.0 &= !Self::bit(mv);
    }

    fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        BitMoveIter(self.0)
    }
}

impl IntoIterator for BitMoveSet {
    type Item = Move;
    type IntoIter = BitMoveIter;

    fn into_iter(self) -> BitMoveIter {
        BitMoveIter(self.0)
    }
}

impl FromIterator<Move> for BitMoveSet {
    fn from_iter<I: IntoIterator<Item = Move>>(iter: I) -> Self {
        let mut set = Self::empty();
        for mv in iter {
            set.insert(mv);
        }
        set
    }
}

/// Yields moves in increasing [`Move::index`] order.
pub struct BitMoveIter(u128);

impl Iterator for BitMoveIter {
    type Item = Move;

    #[inline]
    fn next(&mut self) -> Option<Move> {
        if self.0 == 0 {
            return None;
        }
        let index = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(Move::from_index(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for BitMoveIter {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitState {
    /// `marks[player][board]`: 9-bit mask of that player's squares on that board.
    marks: [[u16; 9]; 2],
    /// `big[player]`: 9-bit mask of boards that player has won.
    big: [u16; 2],
    /// 9-bit mask of boards that are decided (won or full).
    done: u16,
    last_move: Option<Move>,
    to_move: Player,
}

impl BitState {
    #[inline]
    fn occupied(&self, board: usize) -> u16 {
        self.marks[0][board] | self.marks[1][board]
    }

    #[inline]
    fn board_done(&self, board: usize) -> bool {
        self.done >> board & 1 != 0
    }

    #[inline]
    fn forced_index(&self) -> Option<usize> {
        let board = self.last_move?.square.index();
        (!self.board_done(board)).then_some(board)
    }

    /// The board the side to move is restricted to, if any.
    pub fn forced_board(&self) -> Option<Square> {
        self.forced_index().map(Square::from_index)
    }

    /// Free squares of `board`, positioned at that board's 9 bits of a move set.
    #[inline]
    fn board_moves(&self, board: usize) -> u128 {
        ((!self.occupied(board) & FULL) as u128) << (board * 9)
    }

    pub fn is_legal(&self, mv: Move) -> bool {
        let board = mv.board.index();
        if self.board_done(board) {
            return false;
        }
        if self.forced_index().is_some_and(|forced| forced != board) {
            return false;
        }
        self.occupied(board) >> mv.square.index() & 1 == 0
    }

    pub fn cell(&self, mv: Move) -> Option<Player> {
        let bit = 1u16 << mv.square.index();
        if self.marks[0][mv.board.index()] & bit != 0 {
            Some(Player::X)
        } else if self.marks[1][mv.board.index()] & bit != 0 {
            Some(Player::O)
        } else {
            None
        }
    }

    pub fn board_status(&self, board: Square) -> BoardStatus {
        let b = board.index();
        if WIN[self.marks[0][b] as usize] {
            BoardStatus::Won(Player::X)
        } else if WIN[self.marks[1][b] as usize] {
            BoardStatus::Won(Player::O)
        } else if self.occupied(b) == FULL {
            BoardStatus::Full
        } else {
            BoardStatus::Open
        }
    }

    pub fn last_move(&self) -> Option<Move> {
        self.last_move
    }

    // --- Mask accessors for in-crate heuristics. Not public: the bit layout is private. ---

    pub(crate) fn board_marks(&self, player: Player, board: usize) -> u16 {
        self.marks[player.index()][board]
    }

    pub(crate) fn big_marks(&self, player: Player) -> u16 {
        self.big[player.index()]
    }

    pub(crate) fn done_mask(&self) -> u16 {
        self.done
    }
}

impl GameState for BitState {
    type Moves = BitMoveSet;

    fn initial() -> Self {
        Self {
            marks: [[0; 9]; 2],
            big: [0; 2],
            done: 0,
            last_move: None,
            to_move: Player::X,
        }
    }

    fn from_position(position: &Position) -> Self {
        let mut s = Self::initial();
        for board in Square::ALL {
            for square in Square::ALL {
                if let Some(p) = position.cell(Move::new(board, square)) {
                    s.marks[p.index()][board.index()] |= 1 << square.index();
                }
            }
        }
        for b in 0..9 {
            for p in 0..2 {
                if WIN[s.marks[p][b] as usize] {
                    s.big[p] |= 1 << b;
                    s.done |= 1 << b;
                }
            }
            if s.occupied(b) == FULL {
                s.done |= 1 << b;
            }
        }
        s.last_move = position.last_move();
        s.to_move = position.to_move();
        s
    }

    fn to_position(&self) -> Position {
        let mut p = Position::initial();
        for board in Square::ALL {
            for square in Square::ALL {
                let mv = Move::new(board, square);
                p.set_cell(mv, self.cell(mv));
            }
        }
        p.set_to_move(self.to_move);
        p.set_last_move(self.last_move);
        p
    }

    #[inline]
    fn to_move(&self) -> Player {
        self.to_move
    }

    fn legal_moves(&self) -> BitMoveSet {
        match self.forced_index() {
            Some(board) => BitMoveSet(self.board_moves(board)),
            None => {
                let mut bits = 0u128;
                for board in 0..9 {
                    if !self.board_done(board) {
                        bits |= self.board_moves(board);
                    }
                }
                BitMoveSet(bits)
            }
        }
    }

    fn apply(&mut self, mv: Move) {
        debug_assert!(self.is_legal(mv), "illegal move {mv}");
        let p = self.to_move.index();
        let b = mv.board.index();
        self.marks[p][b] |= 1 << mv.square.index();
        if WIN[self.marks[p][b] as usize] {
            self.big[p] |= 1 << b;
            self.done |= 1 << b;
        } else if self.occupied(b) == FULL {
            self.done |= 1 << b;
        }
        self.last_move = Some(mv);
        self.to_move = self.to_move.other();
    }

    #[inline]
    fn outcome(&self) -> Option<Outcome> {
        if WIN[self.big[0] as usize] {
            Some(Outcome::Win(Player::X))
        } else if WIN[self.big[1] as usize] {
            Some(Outcome::Win(Player::O))
        } else if self.done == FULL {
            Some(Outcome::Draw)
        } else {
            None
        }
    }

    fn transform(&self, symmetry: Symmetry) -> Self {
        let mut out = Self {
            marks: [[0; 9]; 2],
            big: [
                symmetry.permute_mask(self.big[0]),
                symmetry.permute_mask(self.big[1]),
            ],
            done: symmetry.permute_mask(self.done),
            last_move: self.last_move.map(|mv| symmetry.apply_move(mv)),
            to_move: self.to_move,
        };
        for (b, board) in Square::ALL.iter().enumerate() {
            let image = symmetry.apply(*board).index();
            for p in 0..2 {
                out.marks[p][image] = symmetry.permute_mask(self.marks[p][b]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::game::sq;

    #[test]
    fn win_table_matches_lines() {
        assert!(!WIN[0]);
        for m in LINE_MASKS {
            assert!(WIN[m as usize]);
        }
        assert!(!WIN[0b000_000_011]);
        assert!(WIN[0b100_010_001]);
        assert_eq!(std::mem::size_of::<BitState>(), 48);
    }

    #[test]
    fn bit_move_set_is_a_set() {
        let a = Move::from_index(0);
        let b = Move::from_index(80);
        let mut set = BitMoveSet::empty();
        assert!(set.is_empty());
        set.insert(b);
        set.insert(a);
        set.insert(a);
        assert_eq!(set.len(), 2);
        assert!(set.contains(a) && set.contains(b));
        assert!(!set.contains(Move::from_index(40)));
        assert_eq!(set.to_vec(), vec![a, b]);
        assert_eq!(set.into_iter().len(), 2);
        set.remove(a);
        assert_eq!(set.to_vec(), vec![b]);
        let all: BitMoveSet = (0..81).map(Move::from_index).collect();
        assert_eq!(all.len(), 81);
        assert_eq!(all, BitState::initial().legal_moves());
    }

    #[test]
    fn initial_matches_position() {
        let b = BitState::initial();
        assert_eq!(b.to_position(), Position::initial());
        assert_eq!(BitState::from_position(&Position::initial()), b);
        assert_eq!(b.legal_moves_vec().len(), 81);
    }

    #[test]
    fn forced_board_and_legality() {
        let mut b = BitState::initial();
        let first = Move::new(sq(1, 1), sq(2, 0));
        assert!(b.is_legal(first));
        b.apply(first);
        assert_eq!(b.to_move(), Player::O);
        assert!(b.is_legal(Move::new(sq(2, 0), sq(0, 0))));
        assert!(!b.is_legal(Move::new(sq(0, 0), sq(0, 0))));
        assert_eq!(b.legal_moves_vec().len(), 9);
    }

    /// Plays random games on both implementations in lockstep, comparing after every move.
    fn differential(games: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        for _ in 0..games {
            let mut pos = Position::initial();
            let mut bit = BitState::initial();
            let mut ply = 0;
            loop {
                ply += 1;
                assert_eq!(pos.outcome(), bit.outcome());
                assert_eq!(pos.to_move(), bit.to_move());
                assert_eq!(bit.to_position(), pos);
                assert_eq!(BitState::from_position(&pos), bit);
                if pos.outcome().is_some() {
                    break;
                }
                let mut ref_moves = pos.legal_moves().to_vec();
                ref_moves.sort();
                let bit_moves = bit.legal_moves();
                assert_eq!(ref_moves, bit_moves.to_vec(), "\n{pos}");
                // Symmetries on every eighth position keep the fuzz quick in debug builds.
                for sym in Symmetry::ALL.iter().copied().filter(|_| ply % 8 == 0) {
                    let t_pos = pos.transform(sym);
                    let t_bit = bit.transform(sym);
                    assert_eq!(t_bit.to_position(), t_pos, "{sym:?}\n{pos}");
                    assert_eq!(t_bit.outcome(), bit.outcome());
                    // Legal moves move with the board.
                    let mut mapped: Vec<Move> =
                        ref_moves.iter().map(|m| sym.apply_move(*m)).collect();
                    mapped.sort();
                    assert_eq!(mapped, t_bit.legal_moves().to_vec(), "{sym:?}");
                    assert_eq!(t_bit.transform(sym.inverse()), bit, "{sym:?} inverse");
                }
                for mv in &ref_moves {
                    assert!(bit_moves.contains(*mv) && bit.is_legal(*mv));
                }
                let mv = ref_moves[rng.random_range(0..ref_moves.len())];
                pos.apply(mv);
                bit.apply(mv);
            }
        }
    }

    #[test]
    fn differential_against_position() {
        differential(2000, 1);
    }

    #[test]
    #[ignore = "slow; run in release"]
    fn differential_against_position_long() {
        differential(50_000, 2);
    }

    #[test]
    #[ignore = "throughput measurement; run with --release --nocapture"]
    fn random_playout_throughput() {
        let mut rng = SmallRng::seed_from_u64(7);
        let games = 200_000;
        let mut moves = 0u64;
        let start = std::time::Instant::now();
        for _ in 0..games {
            let mut s = BitState::initial();
            while s.outcome().is_none() {
                let legal = s.legal_moves();
                let k = rng.random_range(0..legal.len());
                s.apply(legal.iter().nth(k).unwrap());
                moves += 1;
            }
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{games} random games, {moves} moves in {secs:.2}s: {:.0} games/s, {:.0} moves/s",
            games as f64 / secs,
            moves as f64 / secs
        );
    }
}
