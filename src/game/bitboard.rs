//! [`BitState`]: a compact, fast implementation of the rules.
//!
//! Each small board is a pair of 9-bit masks (one per player); the big board is another pair
//! of 9-bit masks of won boards. Win detection is a 512-entry table lookup. All of this is
//! private: the public surface is the [`GameState`] trait with its explicit [`Move`]s.

use std::fmt;

use rand::RngCore;

use super::{
    BoardStatus,
    GameMove,
    GameState,
    LINES,
    Move,
    MoveSet,
    Outcome,
    Player,
    Position,
    Square,
    Symmetry,
    uniform_index,
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

/// `FREE_LINE[mask]` is true if some line holds none of the squares of the 9-bit `mask`: a
/// player whose opponent holds `mask` on an open board can still win it.
const FREE_LINE: [bool; 512] = {
    let mut table = [false; 512];
    let mut mask = 0usize;
    while mask < 512 {
        let mut i = 0;
        while i < 8 {
            if mask & LINE_MASKS[i] as usize == 0 {
                table[mask] = true;
                break;
            }
            i += 1;
        }
        mask += 1;
    }
    table
};

/// `SQUARES[mask]`: the squares set in the 9-bit `mask` in increasing order, padded with zeros;
/// entry `9` is how many there are.
const SQUARES: [[u8; 10]; 512] = {
    let mut table = [[0u8; 10]; 512];
    let mut mask = 0usize;
    while mask < 512 {
        let mut n = 0;
        let mut square = 0;
        while square < 9 {
            if mask >> square & 1 != 0 {
                table[mask][n] = square as u8;
                n += 1;
            }
            square += 1;
        }
        table[mask][9] = n as u8;
        mask += 1;
    }
    table
};

/// [`BitState`]'s move: board and square as the indices `0..9` the masks use, so that playing
/// one needs no arithmetic. Converts to and from the explicit [`Move`] by table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitMove {
    board: u8,
    square: u8,
}

impl BitMove {
    #[inline]
    const fn new(board: usize, square: usize) -> Self {
        Self {
            board: board as u8,
            square: square as u8,
        }
    }

    #[inline]
    const fn board(self) -> usize {
        self.board as usize
    }

    #[inline]
    const fn square(self) -> usize {
        self.square as usize
    }

    /// [`Move::index`] of the move.
    #[inline]
    pub const fn index(self) -> usize {
        self.board() * 9 + self.square()
    }
}

/// Every move by [`Move::index`], both ways of writing it.
const MOVES: [Move; Move::COUNT] = {
    let mut table = [Move::from_index(0); Move::COUNT];
    let mut i = 0;
    while i < Move::COUNT {
        table[i] = Move::from_index(i);
        i += 1;
    }
    table
};
const BIT_MOVES: [BitMove; Move::COUNT] = {
    let mut table = [BitMove::new(0, 0); Move::COUNT];
    let mut i = 0;
    while i < Move::COUNT {
        table[i] = BitMove::new(i / 9, i % 9);
        i += 1;
    }
    table
};

impl From<Move> for BitMove {
    #[inline]
    fn from(mv: Move) -> Self {
        Self::new(mv.board.index(), mv.square.index())
    }
}

impl From<BitMove> for Move {
    #[inline]
    fn from(mv: BitMove) -> Self {
        MOVES[mv.index()]
    }
}

impl GameMove for BitMove {
    #[inline]
    fn index(self) -> usize {
        BitMove::index(self)
    }
}

impl fmt::Display for BitMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Move::from(*self).fmt(f)
    }
}

/// A set of moves as one bit per [`Move::index`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct BitMoveSet(u128);

impl BitMoveSet {
    #[inline]
    fn bit(mv: BitMove) -> u128 {
        1u128 << mv.index()
    }
}

impl MoveSet for BitMoveSet {
    fn empty() -> Self {
        Self(0)
    }

    #[inline]
    fn contains(&self, mv: BitMove) -> bool {
        self.0 & Self::bit(mv) != 0
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.count_ones() as usize
    }

    #[inline]
    fn insert(&mut self, mv: BitMove) {
        self.0 |= Self::bit(mv);
    }

    #[inline]
    fn remove(&mut self, mv: BitMove) {
        self.0 &= !Self::bit(mv);
    }

    fn iter(&self) -> impl Iterator<Item = BitMove> + '_ {
        BitMoveIter(self.0)
    }

    /// Steps through the boards' 9-bit chunks from the first one holding a move (in a forced
    /// position the only one), counting and selecting with [`SQUARES`].
    #[inline]
    fn nth(&self, mut k: usize) -> Option<BitMove> {
        if self.0 == 0 {
            return None;
        }
        let mut board = self.0.trailing_zeros() as usize / 9;
        while board < 9 {
            let row = &SQUARES[(self.0 >> (board * 9)) as usize & FULL as usize];
            let n = row[9] as usize;
            if k < n {
                return Some(BitMove::new(board, row[k] as usize));
            }
            k -= n;
            board += 1;
        }
        None
    }
}

impl IntoIterator for BitMoveSet {
    type Item = BitMove;
    type IntoIter = BitMoveIter;

    fn into_iter(self) -> BitMoveIter {
        BitMoveIter(self.0)
    }
}

impl FromIterator<BitMove> for BitMoveSet {
    fn from_iter<I: IntoIterator<Item = BitMove>>(iter: I) -> Self {
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
    type Item = BitMove;

    #[inline]
    fn next(&mut self) -> Option<BitMove> {
        if self.0 == 0 {
            return None;
        }
        let index = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(BIT_MOVES[index])
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
    last_move: Option<BitMove>,
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
        let board = self.last_move?.square();
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
        self.legal(mv.into())
    }

    fn legal(&self, mv: BitMove) -> bool {
        let board = mv.board();
        if self.board_done(board) {
            return false;
        }
        if self.forced_index().is_some_and(|forced| forced != board) {
            return false;
        }
        self.occupied(board) >> mv.square() & 1 == 0
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
        self.last_move.map(Move::from)
    }

    /// Whether neither player can complete a line of boards any more: each line of boards
    /// holds, for each player, a board they cannot win (won by the other, full, or open with
    /// no line free of the other's marks).
    fn dead(&self) -> bool {
        let open = &SQUARES[(!self.done & FULL) as usize];
        let mut winnable = self.big;
        for &b in &open[..open[9] as usize] {
            let b = b as usize;
            for p in 0..2 {
                if FREE_LINE[self.marks[1 - p][b] as usize] {
                    winnable[p] |= 1 << b;
                }
            }
        }
        !WIN[winnable[0] as usize] && !WIN[winnable[1] as usize]
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
    type Move = BitMove;
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
        s.last_move = position.last_move().map(BitMove::from);
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
        p.set_last_move(self.last_move());
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

    /// [`GameState::play`] without its result, which the compiler then drops.
    #[inline]
    fn apply(&mut self, mv: BitMove) {
        let _ = self.play(mv);
    }

    /// Only a move that decides its own board can end the game, so every other move returns
    /// right after updating that board.
    #[inline]
    fn play(&mut self, mv: BitMove) -> Option<Outcome> {
        debug_assert!(self.legal(mv), "illegal move {mv}");
        let mover = self.to_move;
        let (p, b) = (mover.index(), mv.board());
        self.marks[p][b] |= 1 << mv.square();
        self.last_move = Some(mv);
        self.to_move = mover.other();
        if WIN[self.marks[p][b] as usize] {
            self.big[p] |= 1 << b;
            self.done |= 1 << b;
            if WIN[self.big[p] as usize] {
                return Some(Outcome::Win(mover));
            }
        } else if self.occupied(b) == FULL {
            self.done |= 1 << b;
        } else {
            return None;
        }
        (self.done == FULL).then_some(Outcome::Draw)
    }

    /// Sent to one board, the move comes straight from that board's free squares through
    /// [`SQUARES`], without building the move set; free to choose a board, from the set.
    #[inline]
    fn random_move<R: RngCore + ?Sized>(&self, rng: &mut R) -> BitMove {
        debug_assert!(self.outcome().is_none(), "random_move on a finished game");
        match self.forced_index() {
            Some(board) => {
                let row = &SQUARES[(!self.occupied(board) & FULL) as usize];
                BitMove::new(board, row[uniform_index(rng, row[9] as usize)] as usize)
            }
            None => {
                let legal = self.legal_moves();
                legal
                    .nth(uniform_index(rng, legal.len()))
                    .expect("random_move on a finished game")
            }
        }
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

    /// Only a move that decides its own board can end the game, so every other move is
    /// answered from that board's masks alone.
    #[inline]
    fn outcome_after(&self, mv: BitMove) -> Option<Outcome> {
        debug_assert!(self.legal(mv), "illegal move {mv}");
        let p = self.to_move.index();
        let b = mv.board();
        let bit = 1 << mv.square();
        let big = if WIN[(self.marks[p][b] | bit) as usize] {
            self.big[p] | 1 << b
        } else if self.occupied(b) | bit == FULL {
            self.big[p]
        } else {
            return None;
        };
        if WIN[big as usize] {
            Some(Outcome::Win(self.to_move))
        } else if self.done | 1 << b == FULL {
            Some(Outcome::Draw)
        } else {
            None
        }
    }

    fn settled(&self) -> Option<Outcome> {
        self.outcome()
            .or_else(|| self.dead().then_some(Outcome::Draw))
    }

    /// A move that leaves its board open is answered as in [`GameState::outcome_after`];
    /// only one that decides it pays for the copy and the dead check.
    #[inline]
    fn settled_after(&self, mv: BitMove) -> Option<Outcome> {
        let (p, b) = (self.to_move.index(), mv.board());
        let marks = self.marks[p][b] | 1 << mv.square();
        if !WIN[marks as usize] && self.occupied(b) | 1 << mv.square() != FULL {
            return None;
        }
        let mut next = *self;
        next.play_settled(mv)
    }

    #[inline]
    fn play_settled(&mut self, mv: BitMove) -> Option<Outcome> {
        let done = self.done;
        match self.play(mv) {
            None if self.done != done && self.dead() => Some(Outcome::Draw),
            outcome => outcome,
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
            last_move: self
                .last_move
                .map(|mv| symmetry.apply_move(mv.into()).into()),
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
    fn dead_games_settle_as_draws_once_a_board_closes() {
        // Boards won along their top rows: X 0, 2, 3, 7; O 1, 4, 6. O holds two of board 5's
        // top row and is sent there; board 8 is empty. X can still take boards 5 and 8 down the
        // right column, until O wins board 5:
        //   X O X
        //   X O O
        //   O X .
        let mut p = Position::initial();
        for (board, who) in [
            (0, Player::X),
            (2, Player::X),
            (3, Player::X),
            (7, Player::X),
        ]
        .into_iter()
        .chain([(1, Player::O), (4, Player::O), (6, Player::O)])
        {
            for square in 0..3 {
                p.set_cell(
                    Move::new(Square::from_index(board), Square::from_index(square)),
                    Some(who),
                );
            }
        }
        for square in 0..2 {
            p.set_cell(
                Move::new(Square::from_index(5), Square::from_index(square)),
                Some(Player::O),
            );
        }
        let last = Move::new(Square::from_index(0), Square::from_index(5));
        p.set_cell(last, Some(Player::X));
        p.set_to_move(Player::O);
        p.set_last_move(Some(last));
        let s = BitState::from_position(&p);
        assert_eq!((s.outcome(), s.settled()), (None, None));

        let win = BitMove::from(Move::new(Square::from_index(5), Square::from_index(2)));
        assert_eq!(s.outcome_after(win), None);
        assert_eq!(s.settled_after(win), Some(Outcome::Draw));
        let mut after = s;
        assert_eq!(after.play_settled(win), Some(Outcome::Draw));
        assert_eq!(after.outcome(), None, "the rules play on");
        assert_eq!(after.settled(), Some(Outcome::Draw));
        assert!(!after.legal_moves().is_empty());

        // A move that leaves board 5 open is not checked, and does not kill the game here.
        let quiet = BitMove::from(Move::new(Square::from_index(5), Square::from_index(4)));
        assert_eq!(s.settled_after(quiet), None);
    }

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
        set.insert(b.into());
        set.insert(a.into());
        set.insert(a.into());
        assert_eq!(set.len(), 2);
        assert!(set.contains(a.into()) && set.contains(b.into()));
        assert!(!set.contains(Move::from_index(40).into()));
        assert_eq!(set.to_vec(), vec![a, b]);
        assert_eq!(set.into_iter().len(), 2);
        set.remove(a.into());
        assert_eq!(set.to_vec(), vec![b]);
        let all: BitMoveSet = (0..81)
            .map(|i| BitMove::from(Move::from_index(i)))
            .collect();
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
        b.apply(first.into());
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
                    assert!(bit_moves.contains((*mv).into()) && bit.is_legal(*mv));
                    let mut next = pos.clone();
                    next.apply(*mv);
                    assert_eq!(
                        bit.outcome_after((*mv).into()),
                        next.outcome(),
                        "{mv}\n{pos}"
                    );
                }
                // `ref_moves` is sorted, which is the bit set's iteration order.
                for k in 0..=ref_moves.len() {
                    assert_eq!(
                        bit_moves.nth(k).map(Move::from),
                        ref_moves.get(k).copied(),
                        "{k}\n{pos}"
                    );
                }
                let mv = ref_moves[rng.random_range(0..ref_moves.len())];
                pos.apply(mv);
                bit.apply(mv.into());
            }
        }
    }

    /// Random games by `random_move` and `play`, the reference following along: every move
    /// is legal there, converts back and forth, and `play` reports what `outcome` says.
    #[test]
    fn random_move_and_play_follow_the_reference() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut picked = [0u32; Move::COUNT];
        for _ in 0..2000 {
            let mut pos = Position::initial();
            let mut bit = BitState::initial();
            loop {
                let mv = bit.random_move(&mut rng);
                let explicit = Move::from(mv);
                assert_eq!(BitMove::from(explicit), mv);
                assert_eq!(GameMove::index(mv), explicit.index());
                assert!(pos.legal_moves().contains(explicit), "{explicit}\n{pos}");
                picked[explicit.index()] += 1;
                pos.apply(explicit);
                let ended = bit.play(mv);
                assert_eq!(ended, pos.outcome(), "{explicit}\n{pos}");
                assert_eq!(bit.to_position(), pos);
                if ended.is_some() {
                    break;
                }
            }
        }
        assert!(picked.iter().all(|&n| n > 0), "every move gets picked");
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
                s.apply(legal.nth(k).unwrap());
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
