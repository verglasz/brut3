//! An opening book: positions of the first plies with the move a deep search chose, built
//! incrementally.
//!
//! The book serves both colours. For the colour we play, a position needs only our move; for
//! the opponent's turns it needs an entry under every reply we cover. So the book is the
//! union of two trees: from our nodes only the best move is followed, from the opponent's
//! every reply whose share of the search's visits is at least a threshold, once with us as X
//! and once as O. Positions are stored under their canonical form over the eight board
//! symmetries, with the moves in that frame; a lookup canonicalises the live position and
//! maps the move back. A key does not hold a decided board's cells, only who won it (or that
//! it is full): positions that differ only inside decided boards play the same, so they share
//! an entry.
//!
//! Each entry keeps the search's root value and visit counts over the moves, not just the
//! move, so that deepening the book later can extend the frontier from the file alone and
//! re-choose moves from their children's values, and a search with a bigger budget can start
//! from the visit shares as its root priors.
//!
//! File format: comment lines starting with `#`, the line [`FORMAT_LINE`], then lines of
//! `KEY MOVE VALUE VISITS`, where `KEY` is the canonical [`Key`], `MOVE` the chosen move's
//! index, `VALUE` the root value for the side to move, and `VISITS` a comma-separated list of
//! `move:count` over the moves, then `dropped` for an entry pruned from the shipped book; or
//! of `KEY open` for an open position, where only the opponent of the book moves and every
//! reply is covered without a search (see [`BookConfig::dense_plies`]). Files without the format line are older: their lines start with a position's absolute
//! cells (81 digits, 1 X, 2 O) and forced board, or for a while with a key in absolute colours
//! followed by the side to move (83 digits). Loading canonicalises every line and pools the
//! searches of lines that share a key: the positions have the same moves, so their visits add
//! up.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::agent::UtttAgent;
use crate::game::{BitState, BoardStatus, GameState, Move, Player, Position, Square, Symmetry};
use crate::mcts::{MctsConfig, SearchResult};
use crate::selfplay::search_positions;

/// A book position's key, from the side to move's view: per board, nine characters in
/// [`Move::index`] order, the cell digits of an open board (0 empty, 1 the side to move's,
/// 2 the opponent's) or nine times `3` (won by the side to move), `4` (won by the opponent)
/// or `5` (full) for a decided one; then the forced board digit (9 for a free choice). The
/// rules treat both players alike, so a position and its colour-swapped twin share a key.
pub type Key = String;

/// Characters in a [`Key`].
pub const KEY_LEN: usize = Move::COUNT + 1;

/// The line before the entries of a book file whose lines start with keys.
pub const FORMAT_LINE: &str = "format 2";

/// What the search found from a position, in the position's canonical frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub best: Move,
    /// Root value for the side to move, in `[-1, 1]`.
    pub value: f32,
    /// Visit counts of the root's moves.
    pub visits: Vec<(Move, u32)>,
    /// Pruned from the shipped book (`prune-book`): lookups and the export skip the entry,
    /// while the walks over the book (deepening, common lines) still use it.
    pub dropped: bool,
}

impl Entry {
    fn from_result(result: &SearchResult) -> Option<Self> {
        Some(Entry {
            best: result.best_move()?,
            value: result.root_value,
            visits: result.visits.clone(),
            dropped: false,
        })
    }

    fn total_visits(&self) -> u32 {
        self.visits.iter().map(|(_, n)| n).sum()
    }

    /// Whether a search of `simulations` would have nothing to add to this one's choice: it had
    /// that budget (a search of n simulations stores n - 1 child visits), or its most visited
    /// move leads the next by more than the rest of that budget, so that continuing it could
    /// not have changed the move (where a search with [`MctsConfig::early_stop`] stops).
    fn settled(&self, simulations: u32) -> bool {
        let done = self.total_visits() + 1;
        let (mut first, mut second) = (0, 0);
        for &(_, n) in &self.visits {
            if n > first {
                second = first;
                first = n;
            } else if n > second {
                second = n;
            }
        }
        done >= simulations || first - second > simulations - done
    }

    /// The visit shares as a prior row indexed by [`Move::index`], to seed a new search of
    /// the position (in the entry's frame) with what this one found.
    fn prior_row(&self) -> [f32; Move::COUNT] {
        let total = self.total_visits().max(1) as f32;
        let mut row = [0.0; Move::COUNT];
        for (mv, n) in &self.visits {
            row[mv.index()] = *n as f32 / total;
        }
        row
    }

    /// Two searches of positions with one key pooled, moves in the same frame: the visits add
    /// up, the value is the visit-weighted mean, the best move is the most visited one (the
    /// deeper search's on ties), and the entry is live if either was.
    fn pooled(self, other: Entry) -> Entry {
        let (deep, shallow) = if other.total_visits() > self.total_visits() {
            (other, self)
        } else {
            (self, other)
        };
        let (wd, ws) = (deep.total_visits() as f32, shallow.total_visits() as f32);
        let value = if wd + ws > 0.0 {
            (deep.value * wd + shallow.value * ws) / (wd + ws)
        } else {
            deep.value
        };
        let mut visits = deep.visits;
        for (mv, n) in shallow.visits {
            match visits.iter_mut().find(|(m, _)| *m == mv) {
                Some(entry) => entry.1 += n,
                None => visits.push((mv, n)),
            }
        }
        let count = |mv: Move| visits.iter().find(|(m, _)| *m == mv).map_or(0, |e| e.1);
        let top = visits.iter().map(|(_, n)| *n).max().unwrap_or(0);
        let best = if count(deep.best) == top {
            deep.best
        } else {
            visits
                .iter()
                .find(|(_, n)| *n == top)
                .map_or(deep.best, |e| e.0)
        };
        Entry {
            best,
            value,
            visits,
            dropped: deep.dropped && shallow.dropped,
        }
    }
}

impl Entry {
    /// The entry with its moves mapped through `sym`.
    fn mapped(self, sym: Symmetry) -> Self {
        Entry {
            best: sym.apply_move(self.best),
            visits: self
                .visits
                .into_iter()
                .map(|(mv, n)| (sym.apply_move(mv), n))
                .collect(),
            ..self
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Book {
    entries: HashMap<Key, Entry>,
    /// Open positions without an entry: every reply is in the book, and none was searched.
    open: HashSet<Key>,
    /// Lines of the loaded file that shared a key with another line.
    coalesced: usize,
}

fn forced_digit(state: &BitState) -> char {
    match state.forced_board() {
        Some(b) => char::from(b'0' + b.index() as u8),
        None => '9',
    }
}

fn cell_digit(cell: Option<Player>) -> char {
    match cell {
        None => '0',
        Some(Player::X) => '1',
        Some(Player::O) => '2',
    }
}

/// The key of `state` as it stands (no canonicalisation).
pub fn key_of(state: &BitState) -> Key {
    let me = state.to_move();
    let mut key = String::with_capacity(KEY_LEN);
    for board in 0..9 {
        let decided = match state.board_status(Square::from_index(board)) {
            BoardStatus::Open => None,
            BoardStatus::Won(p) if p == me => Some('3'),
            BoardStatus::Won(_) => Some('4'),
            BoardStatus::Full => Some('5'),
        };
        for square in 0..9 {
            key.push(decided.unwrap_or_else(|| {
                match state.cell(Move::from_index(board * 9 + square)) {
                    None => '0',
                    Some(p) if p == me => '1',
                    Some(_) => '2',
                }
            }));
        }
    }
    key.push(forced_digit(state));
    key
}

/// A key as the book wrote it for a while: absolute digits (1 X, 2 O; 3 won by X, 4 won by
/// O), then the side to move (0 X, 1 O). The same key from the side to move's view.
fn key_of_absolute(key: &str) -> Result<Key, String> {
    let (digits, to_move) = key.split_at(Move::COUNT + 1);
    let swap = match to_move {
        "0" => false,
        "1" => true,
        _ => return Err(format!("bad key {key:?}")),
    };
    Ok(digits
        .chars()
        .enumerate()
        .map(|(i, c)| match c {
            '1' | '2' | '3' | '4' if swap && i < Move::COUNT => {
                char::from(if c == '1' || c == '3' {
                    c as u8 + 1
                } else {
                    c as u8 - 1
                })
            }
            c => c,
        })
        .collect())
}

/// The position's cells and forced board as the file stores them: 81 cell digits by
/// [`Move::index`] and the forced board digit.
pub fn cells_of(state: &BitState) -> String {
    let mut cells: String = (0..Move::COUNT)
        .map(|i| cell_digit(state.cell(Move::from_index(i))))
        .collect();
    cells.push(forced_digit(state));
    cells
}

/// Marks on the board: the position's ply.
pub fn marks(state: &BitState) -> usize {
    (0..Move::COUNT)
        .filter(|&i| state.cell(Move::from_index(i)).is_some())
        .count()
}

/// The canonical image of `state`: the symmetry (first in `Symmetry::ALL` on ties) whose
/// image has the smallest key, that image, and its key.
pub fn canonical(state: &BitState) -> (Symmetry, BitState, Key) {
    let mut best: Option<(Symmetry, BitState, Key)> = None;
    for sym in Symmetry::ALL {
        let image = state.transform(sym);
        let key = key_of(&image);
        if best.as_ref().is_none_or(|(_, _, k)| key < *k) {
            best = Some((sym, image, key));
        }
    }
    best.expect("eight symmetries")
}

/// The cells a key's decided board stands for: the winner's top row for a won board,
/// `XOX OXX OXO` (no line) for a full one.
fn decided_cells(status: BoardStatus) -> [Option<Player>; 9] {
    use Player::{O, X};
    match status {
        BoardStatus::Won(winner) => {
            let mut cells = [None; 9];
            cells[..3].fill(Some(winner));
            cells
        }
        BoardStatus::Full => [X, O, X, O, X, X, O, X, O].map(Some),
        BoardStatus::Open => unreachable!("a decided board"),
    }
}

/// The position a key stands for, in the key's frame, with X to move: the open boards as
/// the key gives them, each decided board as [`decided_cells`], and as the last move (whose
/// square is all the rules read of it: it sets the forced board) the first mark of O that
/// sends play to the key's forced board, or to a decided board when the choice is free, else
/// the first cell that does.
pub fn state_of_key(key: &str) -> Result<BitState, String> {
    let digits = key.as_bytes();
    if digits.len() != KEY_LEN || !digits.iter().all(u8::is_ascii_digit) {
        return Err(format!("bad key {key:?}"));
    }
    let to_move = Player::X;
    let mut position = Position::initial();
    let mut status = [BoardStatus::Open; 9];
    for (board, s) in status.iter_mut().enumerate() {
        let cells = &digits[board * 9..board * 9 + 9];
        let cells: [Option<Player>; 9] = match cells[0] {
            b'3' | b'4' | b'5' if cells.iter().all(|&c| c == cells[0]) => {
                *s = match cells[0] {
                    b'3' => BoardStatus::Won(Player::X),
                    b'4' => BoardStatus::Won(Player::O),
                    _ => BoardStatus::Full,
                };
                decided_cells(*s)
            }
            _ => {
                let mut open = [None; 9];
                for (cell, &d) in open.iter_mut().zip(cells) {
                    *cell = match d {
                        b'0' => None,
                        b'1' => Some(Player::X),
                        b'2' => Some(Player::O),
                        _ => return Err(format!("{key:?}: bad board {board}")),
                    };
                }
                open
            }
        };
        for (square, cell) in cells.into_iter().enumerate() {
            position.set_cell(Move::from_index(board * 9 + square), cell);
        }
    }
    position.set_to_move(to_move);
    let forced = usize::from(digits[Move::COUNT] - b'0');
    let leads_there = |mv: &Move| {
        let square = mv.square.index();
        if forced == 9 {
            status[square].is_decided()
        } else {
            square == forced
        }
    };
    let cells = (0..Move::COUNT).map(Move::from_index);
    let mover = to_move.other();
    let last = cells
        .clone()
        .filter(leads_there)
        .find(|&mv| position.cell(mv) == Some(mover))
        .or_else(|| cells.clone().find(leads_there));
    let started = cells.clone().any(|mv| position.cell(mv).is_some());
    if started {
        position.set_last_move(Some(
            last.ok_or_else(|| format!("{key:?}: a free choice with no decided board"))?,
        ));
    } else if forced != 9 {
        return Err(format!("{key:?}: forced board {forced} on an empty board"));
    }
    let state = BitState::from_position(&position);
    if key_of(&state) != key {
        return Err(format!("{key:?} is not a position"));
    }
    Ok(state)
}

/// The state that cells and a forced board (as [`cells_of`] writes them) describe. They hold
/// no move history, so the last move is set to some mark of the side that just moved whose
/// square sends play to the forced board (or to a decided board when the choice is free); the
/// rules read nothing else of it.
pub fn state_of_cells(key: &str) -> Result<BitState, String> {
    let digits = key.as_bytes();
    if digits.len() != Move::COUNT + 1 || !digits.iter().all(u8::is_ascii_digit) {
        return Err(format!("bad key {key:?}"));
    }
    let mut position = Position::initial();
    let (mut xs, mut os) = (0usize, 0usize);
    for (i, &d) in digits[..Move::COUNT].iter().enumerate() {
        let cell = match d {
            b'0' => None,
            b'1' => {
                xs += 1;
                Some(Player::X)
            }
            b'2' => {
                os += 1;
                Some(Player::O)
            }
            _ => return Err(format!("bad cell digit in {key:?}")),
        };
        position.set_cell(Move::from_index(i), cell);
    }
    let to_move = match xs.checked_sub(os) {
        Some(0) => Player::X,
        Some(1) => Player::O,
        _ => return Err(format!("{key:?}: {xs} X marks against {os} O marks")),
    };
    position.set_to_move(to_move);
    let forced = usize::from(digits[Move::COUNT] - b'0');
    if xs + os > 0 {
        let mover = to_move.other();
        let last = (0..Move::COUNT).map(Move::from_index).find(|&mv| {
            position.cell(mv) == Some(mover)
                && if forced == 9 {
                    position.board_status(mv.square).is_decided()
                } else {
                    mv.square.index() == forced
                }
        });
        position.set_last_move(Some(last.ok_or_else(|| {
            format!("{key:?}: no last move of {mover} leads to forced board {forced}")
        })?));
    }
    let state = BitState::from_position(&position);
    if cells_of(&state) != key {
        return Err(format!("{key:?} is not a reachable position"));
    }
    Ok(state)
}

/// The symmetries that leave `state` unchanged (the identity among them).
fn stabiliser(state: &BitState) -> impl Iterator<Item = Symmetry> {
    let key = key_of(state);
    Symmetry::ALL
        .into_iter()
        .filter(move |&sym| key_of(&state.transform(sym)) == key)
}

/// The moves that are the same as `mv` in `state` up to the symmetries that leave `state`
/// unchanged (always including `mv` itself), e.g. all four corners of the empty board.
pub fn equivalent_moves(state: &BitState, mv: Move) -> Vec<Move> {
    let mut moves: Vec<Move> = stabiliser(state).map(|sym| sym.apply_move(mv)).collect();
    moves.sort();
    moves.dedup();
    moves
}

/// Whether an agent that plays `played` with move distribution `policy` at `state` finds the
/// book move `best` anyway: `played` is `best` up to the position's symmetries, and those
/// moves together lead the agent's second-best move (likewise with its symmetric
/// equivalents) by at least `min_advantage` of the policy.
pub fn agent_agrees(
    state: &BitState,
    best: Move,
    played: Move,
    policy: &[(Move, f32)],
    min_advantage: f32,
) -> bool {
    // Collected once: every move of the policy is mapped through them.
    let syms: Vec<Symmetry> = stabiliser(state).collect();
    let class = |mv: Move| {
        syms.iter()
            .map(|sym| sym.apply_move(mv))
            .min()
            .expect("the identity")
    };
    let book_class = class(best);
    if class(played) != book_class {
        return false;
    }
    let total: f32 = policy.iter().map(|(_, p)| p.max(0.0)).sum();
    if total <= 0.0 {
        return true;
    }
    let mut shares: Vec<(Move, f32)> = Vec::new();
    for &(mv, p) in policy {
        let c = class(mv);
        match shares.iter_mut().find(|(k, _)| *k == c) {
            Some(entry) => entry.1 += p.max(0.0),
            None => shares.push((c, p.max(0.0))),
        }
    }
    let book_share = shares
        .iter()
        .find(|(k, _)| *k == book_class)
        .map_or(0.0, |(_, p)| *p);
    let second = shares
        .iter()
        .filter(|(k, _)| *k != book_class)
        .map(|(_, p)| *p)
        .fold(0.0, f32::max);
    book_share - second >= min_advantage * total
}

impl Book {
    /// Entries, dropped ones included.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Entries not dropped: the shipped book.
    pub fn live_len(&self) -> usize {
        self.entries.values().filter(|e| !e.dropped).count()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every entry not dropped by canonical key, in no particular order.
    pub fn entries(&self) -> impl Iterator<Item = (&Key, &Entry)> {
        self.entries.iter().filter(|(_, e)| !e.dropped)
    }

    /// Every entry, dropped ones included, by canonical key in no particular order.
    pub fn all_entries(&self) -> impl Iterator<Item = (&Key, &Entry)> {
        self.entries.iter()
    }

    /// Every entry's position (a position its key stands for, in the canonical frame: see
    /// [`state_of_key`]) with its key, sorted by key, dropped entries included.
    pub fn positions(&self) -> Vec<(Key, BitState)> {
        let mut keys: Vec<&Key> = self.entries.keys().collect();
        keys.sort();
        keys.into_iter()
            .map(|key| {
                (
                    key.clone(),
                    state_of_key(key).expect("book keys are positions"),
                )
            })
            .collect()
    }

    /// Every entry's and open position's ply: the fewest moves from the start that reach it
    /// through the book's moves (every move with visits, dropped entries included, and every
    /// move out of an open position), or for an entry no such path reaches, the marks of its
    /// position.
    pub fn plies(&self) -> HashMap<Key, usize> {
        let mut plies = HashMap::new();
        let mut queue = VecDeque::from([(BitState::initial(), 0)]);
        while let Some((state, ply)) = queue.pop_front() {
            if state.is_terminal() {
                continue;
            }
            let (sym, _, key) = canonical(&state);
            let moves: Vec<Move> = match self.entries.get(&key) {
                Some(entry) => {
                    let back = sym.inverse();
                    entry
                        .visits
                        .iter()
                        .filter(|(_, n)| *n > 0)
                        .map(|(mv, _)| back.apply_move(*mv))
                        .collect()
                }
                None if self.open.contains(&key) => state.legal_moves_vec(),
                None => continue,
            };
            if plies.contains_key(&key) {
                continue;
            }
            plies.insert(key, ply);
            for mv in moves {
                let mut child = state;
                child.apply(mv.into());
                queue.push_back((child, ply + 1));
            }
        }
        for (key, _) in self.positions() {
            plies
                .entry(key)
                .or_insert_with_key(|key| marks(&state_of_key(key).expect("a position")));
        }
        plies
    }

    /// Lines of the loaded file that were merged into another line's entry, their positions
    /// differing only inside decided boards.
    pub fn coalesced(&self) -> usize {
        self.coalesced
    }

    /// Stores `entry`, searched from `state` with its moves in `state`'s frame, under
    /// `state`'s canonical key; returns the entry it replaces.
    pub fn insert(&mut self, state: &BitState, entry: Entry) -> Option<Entry> {
        let (sym, _, key) = canonical(state);
        self.open.remove(&key);
        self.entries.insert(key, entry.mapped(sym))
    }

    /// Marks `state` open: every reply is covered, without a search (unless it has an entry).
    pub fn open(&mut self, state: &BitState) {
        let key = canonical(state).2;
        if !self.entries.contains_key(&key) {
            self.open.insert(key);
        }
    }

    /// Open positions without an entry.
    pub fn open_len(&self) -> usize {
        self.open.len()
    }

    /// The open leaves: where play leaves the book, for the book playing X and for it playing
    /// O. Play follows the book's move where the book's side is to move and every legal reply
    /// where the opponent is, a reply taken with its share of the visits of the position's
    /// search (uniformly out of an open position, which has none); a position it reaches that
    /// the book has nothing for (no live entry where the book is to move; neither that nor an
    /// open mark where the opponent is) is a leaf. Each leaf comes with its reach, the
    /// probability that such play gets there, summed over the ways it can (one side's reaches
    /// add up to at most 1, the rest being games that end inside the book). Sorted by side,
    /// ply and key.
    pub fn open_leaves(&self) -> Vec<Leaf> {
        let mut leaves = Vec::new();
        for book_side in [Player::X, Player::O] {
            // One ply at a time: the positions (canonical images, so that their entries'
            // moves are in their own frame) with their reach and the move that reached them.
            let mut level: HashMap<Key, (BitState, f64, Reached)> = HashMap::new();
            let initial = BitState::initial();
            level.insert(key_of(&initial), (initial, 1.0, Reached::BookMove));
            let mut ply = 0;
            while !level.is_empty() {
                let mut next: HashMap<Key, (BitState, f64, Reached)> = HashMap::new();
                let mut here: Vec<(Key, Leaf)> = Vec::new();
                for (key, (state, reach, reached)) in level {
                    if state.is_terminal() {
                        continue;
                    }
                    let own = state.to_move() == book_side;
                    let entry = self.entries.get(&key).filter(|e| !e.dropped);
                    if entry.is_none() && (own || !self.open.contains(&key)) {
                        let leaf = Leaf {
                            book: book_side,
                            ply,
                            reach,
                            reached,
                        };
                        here.push((key, leaf));
                        continue;
                    }
                    let moves: Vec<(Move, f64, Reached)> = match entry {
                        Some(entry) if own => vec![(entry.best, 1.0, Reached::BookMove)],
                        Some(entry) => {
                            let total = entry.total_visits().max(1) as f64;
                            let mut shares = [0.0; Move::COUNT];
                            for (mv, n) in &entry.visits {
                                shares[mv.index()] = *n as f64 / total;
                            }
                            state
                                .legal_moves_vec()
                                .into_iter()
                                .map(|mv| {
                                    let share = shares[mv.index()];
                                    (mv, share, Reached::Reply(share as f32))
                                })
                                .collect()
                        }
                        None => {
                            let legal = state.legal_moves_vec();
                            let p = 1.0 / legal.len() as f64;
                            legal
                                .into_iter()
                                .map(|mv| (mv, p, Reached::OpenReply))
                                .collect()
                        }
                    };
                    for (mv, p, how) in moves {
                        let mut child = state;
                        child.apply(mv.into());
                        let (_, image, child_key) = canonical(&child);
                        next.entry(child_key)
                            .and_modify(|n| n.1 += reach * p)
                            .or_insert((image, reach * p, how));
                    }
                }
                here.sort_by(|a, b| a.0.cmp(&b.0));
                leaves.extend(here.into_iter().map(|(_, leaf)| leaf));
                level = next;
                ply += 1;
            }
        }
        leaves
    }

    /// The entry stored under the canonical `key`.
    pub fn get(&self, key: &str) -> Option<&Entry> {
        self.entries.get(key)
    }

    /// The entries on the book's common lines: the positions reached from the start when the
    /// side the book plays follows its book moves and the other side only plays replies that
    /// had at least `min_reply` of the book search's visits at their ply (judged reply by reply,
    /// not as a product along the line). Only the positions where the book's side is to move
    /// count, the ones whose entry it plays from.
    pub fn common_lines(&self, min_reply: f32) -> HashSet<Key> {
        let mut common = HashSet::new();
        let mut seen: HashSet<(Key, bool, bool)> = HashSet::new();
        let mut queue: VecDeque<(BitState, bool, bool)> = VecDeque::new();
        queue.push_back((BitState::initial(), true, true));
        while let Some((state, x_line, o_line)) = queue.pop_front() {
            if state.is_terminal() {
                continue;
            }
            let (sym, _, key) = canonical(&state);
            if !seen.insert((key.clone(), x_line, o_line)) {
                continue;
            }
            let mover = state.to_move();
            // The moves with whether each is the book move and whether it is a frequent reply;
            // out of an open position, every move is a frequent reply.
            let moves: Vec<(Move, bool, bool)> = match self.entries.get(&key) {
                Some(entry) => {
                    if (mover == Player::X && x_line) || (mover == Player::O && o_line) {
                        common.insert(key);
                    }
                    let back = sym.inverse();
                    let total = entry.total_visits().max(1) as f32;
                    entry
                        .visits
                        .iter()
                        .map(|(mv, n)| {
                            let frequent = *n as f32 / total >= min_reply;
                            (back.apply_move(*mv), *mv == entry.best, frequent)
                        })
                        .collect()
                }
                None if self.open.contains(&key) => state
                    .legal_moves_vec()
                    .into_iter()
                    .map(|mv| (mv, false, true))
                    .collect(),
                None => continue,
            };
            for (mv, is_best, frequent) in moves {
                let (xl, ol) = match mover {
                    Player::X => (x_line && is_best, o_line && frequent),
                    Player::O => (x_line && frequent, o_line && is_best),
                };
                if xl || ol {
                    let mut child = state;
                    child.apply(mv.into());
                    queue.push_back((child, xl, ol));
                }
            }
        }
        common
    }

    /// Marks the entry under `key` as dropped; returns whether it was live before.
    pub fn drop_entry(&mut self, key: &str) -> bool {
        match self.entries.get_mut(key) {
            Some(entry) if !entry.dropped => {
                entry.dropped = true;
                true
            }
            _ => false,
        }
    }

    /// The book's live entry for `state`, if any, with its moves mapped back to `state`'s
    /// frame.
    pub fn lookup(&self, state: &BitState) -> Option<Entry> {
        let (sym, _, key) = canonical(state);
        let entry = self.entries.get(&key).filter(|e| !e.dropped)?;
        let back = sym.inverse();
        Some(Entry {
            best: back.apply_move(entry.best),
            value: entry.value,
            visits: entry
                .visits
                .iter()
                .map(|(mv, n)| (back.apply_move(*mv), *n))
                .collect(),
            dropped: false,
        })
    }

    /// The book move for `state`, if the position is in the book.
    pub fn best_move(&self, state: &BitState) -> Option<Move> {
        self.lookup(state).map(|e| e.best)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Reads a book file (see the module documentation for the formats).
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut book = Book::default();
        // Whether the lines start with keys: after the format line.
        let mut keys = false;
        let mut lines = 0usize;
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line == FORMAT_LINE {
                if lines > 0 {
                    return Err(format!("line {}: format line after entries", i + 1));
                }
                keys = true;
                continue;
            }
            lines += 1;
            let (position, entry) = parse_line(line).map_err(|e| format!("line {}: {e}", i + 1))?;
            let state = if keys {
                state_of_key(&position)
            } else if position.len() == Move::COUNT + 2 {
                key_of_absolute(&position).and_then(|key| state_of_key(&key))
            } else {
                state_of_cells(&position)
            }
            .map_err(|e| format!("line {}: {e}", i + 1))?;
            // Lines for one key pool their searches.
            let (sym, _, key) = canonical(&state);
            let Some(entry) = entry else {
                book.open.insert(key);
                continue;
            };
            let entry = entry.mapped(sym);
            let entry = match book.entries.remove(&key) {
                Some(old) => {
                    book.coalesced += 1;
                    old.pooled(entry)
                }
                None => entry,
            };
            book.entries.insert(key, entry);
        }
        let Book { entries, open, .. } = &mut book;
        open.retain(|key| !entries.contains_key(key));
        Ok(book)
    }

    /// Writes the book, entries and open positions sorted by key, after `header` comment
    /// lines.
    pub fn save(&self, path: &Path, header: &[String]) -> Result<(), String> {
        let mut keys: Vec<&Key> = self.entries.keys().chain(&self.open).collect();
        keys.sort();
        let mut out = String::new();
        for line in header {
            let _ = writeln!(out, "# {line}");
        }
        let _ = writeln!(out, "{FORMAT_LINE}");
        for key in keys {
            let Some(e) = self.entries.get(key) else {
                let _ = writeln!(out, "{key} open");
                continue;
            };
            let _ = write!(out, "{key} {} {:.4}", e.best.index(), e.value);
            for (i, (mv, n)) in e.visits.iter().enumerate() {
                out.push(if i == 0 { ' ' } else { ',' });
                let _ = write!(out, "{}:{n}", mv.index());
            }
            if e.dropped {
                out.push_str(" dropped");
            }
            out.push('\n');
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    /// The positions of the book's tree under `cfg` that want something the book does not
    /// have yet: a search of `simulations` (or of `cfg.reply_simulations` for an opponent's
    /// position) that the stored one, if any, cannot stand in for (see [`Entry::settled`]), or
    /// a mark that the position is open. Each comes once, at the shallowest ply that reaches
    /// it, with the most it is needed for.
    pub fn pending(&self, cfg: &BookConfig, simulations: u32) -> Vec<Pending> {
        let replies = cfg.reply_simulations.unwrap_or(simulations);
        let mut needs: HashMap<Key, (BitState, usize, Need)> = HashMap::new();
        // A node's children depend on which of the two trees it belongs to, so the visited
        // set is keyed by that too.
        let mut seen: HashSet<(Key, bool, bool)> = HashSet::new();
        // The last flag: reached by its parent's book move (which main-lines mode requires
        // of a position without an entry).
        let mut queue: VecDeque<(BitState, usize, bool, bool, bool)> = VecDeque::new();
        queue.push_back((BitState::initial(), 0, true, true, true));
        while let Some((state, ply, x_strict, o_strict, by_best)) = queue.pop_front() {
            if state.is_terminal() {
                continue;
            }
            let mover = state.to_move();
            let own = if mover == Player::X {
                x_strict
            } else {
                o_strict
            };
            let dense = ply < cfg.dense_plies;
            // The opponent's replies from the last ply would be past it: nothing to cover.
            let need = match (own, dense) {
                (true, _) => Need::Move,
                _ if ply >= cfg.plies => continue,
                (false, true) => Need::Open,
                (false, false) => Need::Replies,
            };
            let (sym, image, key) = canonical(&state);
            // A dropped entry counts as missing: its stored search may come back, or be
            // searched again.
            let entry = self.entries.get(&key).filter(|e| !e.dropped);
            if entry.is_none() && cfg.main_lines && !by_best {
                continue;
            }
            needs
                .entry(key.clone())
                .and_modify(|n| n.2 = n.2.max(need))
                .or_insert((image, ply, need));
            if ply >= cfg.plies || !seen.insert((key, x_strict, o_strict)) {
                continue;
            }
            // The book's side follows its move, which takes an entry; the opponent's side
            // every covered reply: all of them on a dense ply, else those with enough of the
            // entry's visits.
            let back = sym.inverse();
            let mut counts = [None; Move::COUNT];
            for (mv, n) in entry.map_or(&[][..], |e| &e.visits) {
                counts[back.apply_move(*mv).index()] = Some(*n);
            }
            let total = entry.map_or(0, Entry::total_visits).max(1) as f32;
            let best = entry.map(|e| back.apply_move(e.best));
            for mv in state.legal_moves_vec() {
                let covered =
                    dense || counts[mv.index()].is_some_and(|n| n as f32 / total >= cfg.min_prior);
                let is_best = best == Some(mv);
                let (xs, os) = match mover {
                    Player::X => (x_strict && is_best, o_strict && covered),
                    Player::O => (x_strict && covered, o_strict && is_best),
                };
                if xs || os {
                    let mut child = state;
                    child.apply(mv.into());
                    queue.push_back((child, ply + 1, xs, os, is_best));
                }
            }
        }
        let mut pending: Vec<Pending> = needs
            .into_iter()
            .filter_map(|(key, (state, ply, need))| {
                let entry = self.entries.get(&key);
                let live = entry.filter(|e| !e.dropped);
                let met = match need {
                    Need::Open => entry.is_some() || self.open.contains(&key),
                    Need::Replies => live.is_some_and(|e| cfg.keep_searched || e.settled(replies)),
                    Need::Move => live.is_some_and(|e| cfg.keep_searched || e.settled(simulations)),
                };
                (!met).then_some(Pending {
                    state,
                    ply,
                    need,
                    stored: entry.map(|e| !e.dropped),
                })
            })
            .collect();
        pending.sort_by(|a, b| (a.ply, key_of(&a.state)).cmp(&(b.ply, key_of(&b.state))));
        pending
    }

    /// Brings the book up to `cfg` with `search` over `evaluators`, round after round (each
    /// round one ply deeper), until nothing is [`Book::pending`]: missing positions are
    /// searched, with `search` where the book plays and with `cfg.reply_simulations` (and no
    /// early stop: their shares count) where only the opponent does, or marked open on a
    /// dense ply. A dropped entry whose stored search would do is made live again. Entries
    /// whose search falls short are searched again, shallowest ply first and each round no
    /// deeper than the shallowest of them, with their stored visit shares as the root priors;
    /// so are dropped ones. Each round's searches are shared out among `evaluators`, a thread
    /// each. `log` gets a line per round; `on_done` the count of positions searched so far in
    /// the round; `after_round` the book after each round's entries are in (to save it
    /// provisionally), and its error ends the run.
    pub fn extend<A: UtttAgent<State = BitState> + Send>(
        &mut self,
        evaluators: &mut [A],
        search: &MctsConfig,
        cfg: &BookConfig,
        mut log: impl FnMut(String),
        mut on_done: impl FnMut(usize, usize),
        mut after_round: impl FnMut(&Book) -> Result<(), String>,
    ) -> Result<Extension, String> {
        let reply_search = MctsConfig {
            simulations: cfg.reply_simulations.unwrap_or(search.simulations),
            early_stop: 0,
            ..search.clone()
        };
        let mut done = Extension::default();
        // Positions searched in this run, with what for: a search whose root was proven early
        // stores fewer visits than the budget, and must not be searched round after round.
        let mut searched: HashMap<Key, Need> = HashMap::new();
        loop {
            let mut pending = self.pending(cfg, search.simulations);
            pending.retain(|p| {
                p.need == Need::Open || searched.get(&key_of(&p.state)).is_none_or(|n| *n < p.need)
            });
            if pending.is_empty() {
                break;
            }
            // A dropped entry comes back from its stored search when that would do.
            let mut revived = 0;
            pending.retain(|p| {
                let budget = match p.need {
                    Need::Move => search.simulations,
                    Need::Replies => reply_search.simulations,
                    Need::Open => return true,
                };
                match self.entries.get_mut(&key_of(&p.state)) {
                    Some(entry)
                        if entry.dropped && (cfg.keep_searched || entry.settled(budget)) =>
                    {
                        entry.dropped = false;
                        revived += 1;
                        false
                    }
                    _ => true,
                }
            });
            done.revived += revived;
            // A new search can change a best move or a reply's share, and so the tree below
            // it: the rest waits until the live entries searched again above it are done.
            let again = |p: &Pending| p.stored == Some(true) && p.need != Need::Open;
            if let Some(shallowest) = pending.iter().filter(|p| again(p)).map(|p| p.ply).min() {
                pending.retain(|p| p.ply <= shallowest);
            }
            let mut opened = 0;
            for p in pending.iter().filter(|p| p.need == Need::Open) {
                self.open(&p.state);
                opened += 1;
            }
            done.opened += opened;
            let (moves, replies): (Vec<&Pending>, Vec<&Pending>) = pending
                .iter()
                .filter(|p| p.need != Need::Open)
                .partition(|p| p.need == Need::Move);
            let searching = moves.len() + replies.len();
            let plies = moves
                .iter()
                .chain(&replies)
                .map(|p| p.ply)
                .fold((usize::MAX, 0), |(lo, hi), p| (lo.min(p), hi.max(p)));
            let again_count = moves.iter().chain(&replies).filter(|p| again(p)).count();
            let notes = [
                (
                    replies.len(),
                    format!(
                        "{} for their replies, at {} simulations",
                        replies.len(),
                        reply_search.simulations
                    ),
                ),
                (
                    again_count,
                    format!("{again_count} in the book already, searched again from their visits"),
                ),
                (
                    revived,
                    format!("{revived} dropped entries live again from their stored search"),
                ),
                (
                    opened,
                    format!(
                        "{opened} opponent positions open, every reply covered without a search"
                    ),
                ),
            ];
            log(format!(
                "searching {searching} positions{}{}",
                match plies {
                    _ if searching == 0 => String::new(),
                    (lo, hi) if lo == hi => format!(" at ply {lo}"),
                    (lo, hi) => format!(" at ply {lo} to {hi}"),
                },
                notes
                    .iter()
                    .filter(|(n, _)| *n > 0)
                    .map(|(_, note)| format!(" ({note})"))
                    .collect::<String>()
            ));
            let mut finished = 0;
            for (group, budget) in [(&moves, search), (&replies, &reply_search)] {
                if group.is_empty() {
                    continue;
                }
                let states: Vec<BitState> = group.iter().map(|p| p.state).collect();
                let keys: Vec<Key> = states.iter().map(key_of).collect();
                let priors: Vec<_> = keys
                    .iter()
                    .map(|key| self.entries.get(key).map(Entry::prior_row))
                    .collect();
                let results =
                    search_shared(evaluators, &states, &priors, budget, cfg.concurrent, |n| {
                        on_done(finished + n, searching)
                    });
                finished += states.len();
                for ((state, key), (result, p)) in states
                    .iter()
                    .zip(keys)
                    .zip(results.iter().zip(group.iter()))
                {
                    searched.insert(key, p.need);
                    if let Some(entry) = Entry::from_result(result) {
                        match self.insert(state, entry) {
                            Some(_) => done.researched += 1,
                            None => done.added += 1,
                        }
                    }
                }
            }
            after_round(self)?;
        }
        Ok(done)
    }
}

/// Searches `states` (with their root priors) shared out evenly among `evaluators`, a thread
/// each, the calling thread taking the first share, and returns their results in order.
/// `on_done` gets the count finished so far, on the calling thread.
fn search_shared<A: UtttAgent<State = BitState> + Send>(
    evaluators: &mut [A],
    states: &[BitState],
    priors: &[Option<[f32; Move::COUNT]>],
    search: &MctsConfig,
    concurrent: usize,
    mut on_done: impl FnMut(usize),
) -> Vec<SearchResult> {
    if states.is_empty() {
        return Vec::new();
    }
    let share = states.len().div_ceil(evaluators.len().max(1));
    let finished = AtomicUsize::new(0);
    let mut reported = 0;
    let results = std::thread::scope(|scope| {
        let mut shares = states
            .chunks(share)
            .zip(priors.chunks(share))
            .zip(evaluators);
        let ((states, priors), evaluator) = shares.next().expect("an evaluator");
        let others: Vec<_> = shares
            .map(|((states, priors), evaluator)| {
                let finished = &finished;
                scope.spawn(move || {
                    search_positions(
                        evaluator,
                        states,
                        search,
                        concurrent,
                        |i| priors[i],
                        |_| {
                            finished.fetch_add(1, Ordering::Relaxed);
                        },
                    )
                })
            })
            .collect();
        let mut results = search_positions(
            evaluator,
            states,
            search,
            concurrent,
            |i| priors[i],
            |_| {
                reported = 1 + finished.fetch_add(1, Ordering::Relaxed);
                on_done(reported);
            },
        );
        for other in others {
            results.extend(other.join().expect("book search thread panicked"));
        }
        results
    });
    if reported < states.len() {
        on_done(states.len());
    }
    results
}

/// Where play leaves the book: see [`Book::open_leaves`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Leaf {
    /// The side the book plays.
    pub book: Player,
    pub ply: usize,
    /// The probability that play gets there.
    pub reach: f64,
    /// The move that first reached it.
    pub reached: Reached,
}

/// The move that reached an open leaf.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reached {
    /// The book's own move: the book ends after it, as at its last ply.
    BookMove,
    /// An opponent reply from a searched position, with its share of that search's visits
    /// there: one under the book's threshold, or one the book has not got to yet.
    Reply(f32),
    /// An opponent reply from an open position, whose replies were not all searched yet: a
    /// run cut short between marking positions open and searching their replies.
    OpenReply,
}

/// What a position of the book's tree wants, by [`BookConfig`]; ordered by how much it takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Need {
    /// Only the opponent of the book moves from it, on a dense ply: every reply is covered
    /// without a search, so the position is only marked open.
    Open,
    /// Only the opponent moves from it, past the dense plies: a search of the reply budget,
    /// for its replies' shares of the visits.
    Replies,
    /// The book plays from it: a search of the full budget, for its move.
    Move,
}

/// A position [`Book::pending`] finds wanting.
#[derive(Clone, Debug)]
pub struct Pending {
    /// The position in its canonical frame.
    pub state: BitState,
    pub ply: usize,
    pub need: Need,
    /// Whether the book has an entry for it already, and if so whether it is live.
    pub stored: Option<bool>,
}

/// What [`Book::extend`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Extension {
    /// Positions searched and added.
    pub added: usize,
    /// Dropped entries made live again from their stored search.
    pub revived: usize,
    /// Opponent positions marked open, without a search.
    pub opened: usize,
    /// Entries searched again (live or dropped), their stored search having had a smaller
    /// budget and no settled move.
    pub researched: usize,
}

/// A line's first field (a key, or in older files a position: see [`Book::parse`]) and
/// entry, or `None` for an open position.
fn parse_line(line: &str) -> Result<(String, Option<Entry>), String> {
    let mut fields = line.split_whitespace();
    let key = fields.next().ok_or("empty line")?;
    if ![KEY_LEN, Move::COUNT + 2].contains(&key.len()) || !key.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(format!("bad key {key:?}"));
    }
    let mut fields = fields.peekable();
    if fields.next_if(|f| *f == "open").is_some() {
        return match fields.next() {
            Some(extra) => Err(format!("unexpected field {extra:?}")),
            None => Ok((key.to_string(), None)),
        };
    }
    let move_index = |s: &str| -> Result<Move, String> {
        let i: usize = s.parse().map_err(|_| format!("bad move {s:?}"))?;
        if i >= Move::COUNT {
            return Err(format!("move index {i} out of range"));
        }
        Ok(Move::from_index(i))
    };
    let best = move_index(fields.next().ok_or("missing move")?)?;
    let value: f32 = fields
        .next()
        .ok_or("missing value")?
        .parse()
        .map_err(|_| "bad value".to_string())?;
    let mut visits = Vec::new();
    let mut dropped = false;
    let mut rest = fields.peekable();
    if let Some(list) = rest.next_if(|f| *f != "dropped") {
        for item in list.split(',') {
            let (mv, n) = item
                .split_once(':')
                .ok_or_else(|| format!("bad visits {item:?}"))?;
            let n: u32 = n.parse().map_err(|_| format!("bad count {n:?}"))?;
            visits.push((move_index(mv)?, n));
        }
    }
    if rest.next_if(|f| *f == "dropped").is_some() {
        dropped = true;
    }
    if let Some(extra) = rest.next() {
        return Err(format!("unexpected field {extra:?}"));
    }
    Ok((
        key.to_string(),
        Some(Entry {
            best,
            value,
            visits,
            dropped,
        }),
    ))
}

#[derive(Clone, Debug)]
pub struct BookConfig {
    /// Deepest ply to hold entries for (the initial position is ply 0).
    pub plies: usize,
    /// Smallest share of a position's visits for an opponent reply to be covered.
    pub min_prior: f32,
    /// Plies below this one cover every opponent reply, without a search: the book's tree
    /// is complete to it, and its positions where only the opponent moves are marked open.
    pub dense_plies: usize,
    /// Simulations of the searches of positions where only the opponent moves, past the dense
    /// plies, for their replies' shares (no early stop); the book's search's own when `None`.
    pub reply_simulations: Option<u32>,
    /// Positions searched at once.
    pub concurrent: usize,
    /// Only add positions reached by their parent's book move, whichever side plays it: the
    /// existing tree is walked as usual, but new depth grows only along the main line from
    /// each position, not into new reply branches.
    pub main_lines: bool,
    /// Keep every stored search whatever its budget: only positions the book lacks are
    /// searched, and dropped entries come back from their stored search.
    pub keep_searched: bool,
}

impl Default for BookConfig {
    fn default() -> Self {
        Self {
            plies: 8,
            min_prior: 0.01,
            dense_plies: 0,
            reply_simulations: None,
            concurrent: 256,
            main_lines: false,
            keep_searched: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::RandomAgent;
    use crate::game::{Square, sq};

    #[test]
    fn keys_and_canonical_forms() {
        let initial = BitState::initial();
        assert_eq!(key_of(&initial), format!("{}9", "0".repeat(81)));
        let (sym, image, key) = canonical(&initial);
        assert_eq!(sym, Symmetry::Identity);
        assert_eq!(image, initial);
        assert_eq!(key, key_of(&initial));

        // Every image of a position canonicalises to the same key.
        let mut s = initial;
        s.apply(Move::new(sq(0, 1), sq(2, 0)).into());
        s.apply(Move::new(sq(2, 0), sq(0, 0)).into());
        let (_, image, key) = canonical(&s);
        for sym in Symmetry::ALL {
            let (_, img, k) = canonical(&s.transform(sym));
            assert_eq!(k, key, "{sym:?}");
            assert_eq!(key_of(&img), key);
            assert_eq!(img, image);
        }
        // The forced board is board 0's image under the canonical symmetry.
        let forced: usize = key[81..82].parse().unwrap();
        assert!(forced < 9);
        assert_eq!(image.forced_board(), Some(Square::from_index(forced)));
    }

    #[test]
    fn keys_ignore_the_cells_of_decided_boards() {
        // X won board 0 along its top row; O's two marks there sit on different squares. The
        // last move, X's square 0, sends play to the won board: a free choice.
        let rest = "0".repeat(72);
        let a = state_of_cells(&format!("111220000{rest}9")).unwrap();
        let b = state_of_cells(&format!("111200200{rest}9")).unwrap();
        assert_ne!(cells_of(&a), cells_of(&b));
        // O to move: board 0 is won by the opponent.
        assert_eq!(key_of(&a), format!("{}{rest}9", "4".repeat(9)));
        assert_eq!(key_of(&a), key_of(&b));
        assert_eq!(canonical(&a).2, canonical(&b).2);
        // The colour-swapped twin, X to move with O's line on board 0, shares the key.
        let twin = state_of_cells(&format!("222110100{rest}9")).unwrap();
        assert_eq!(twin.to_move(), Player::X);
        assert_eq!(key_of(&twin), key_of(&a));

        // A file holding both pools their searches under one entry.
        let text = format!(
            "{} 40 0.1 40:5,41:3\n{} 41 0.2 41:4,42:2 dropped\n",
            cells_of(&a),
            cells_of(&b)
        );
        let book = Book::parse(&text).unwrap();
        assert_eq!((book.len(), book.coalesced()), (1, 1));
        let entry = book.lookup(&b).unwrap();
        let count = |i: usize| {
            entry
                .visits
                .iter()
                .find(|(m, _)| m.index() == i)
                .map(|e| e.1)
        };
        assert_eq!(
            (count(40), count(41), count(42)),
            (Some(5), Some(7), Some(2))
        );
        assert_eq!(entry.best, Move::from_index(41), "most visits once pooled");
        assert!((entry.value - (0.1 * 8.0 + 0.2 * 6.0) / 14.0).abs() < 1e-6);
        assert_eq!(book.live_len(), 1, "live if either line was");
        // The position the key stands for, with X to move: O's top row on the won board, which
        // the canonical frame puts last.
        let (key, state) = &book.positions()[0];
        assert_eq!(cells_of(state), format!("{rest}2220000009"));
        assert_eq!(*key, key_of(state));
        // Saved, the line holds the key.
        let path = std::env::temp_dir().join(format!("uttt-pool-{}.txt", std::process::id()));
        book.save(&path, &[]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let best = book.get(key).unwrap().best.index();
        assert!(
            text.starts_with(&format!("{FORMAT_LINE}\n{key} {best} ")),
            "{text}"
        );
        let back = Book::parse(&text).unwrap();
        let (saved, read) = (book.get(key).unwrap(), back.get(key).unwrap());
        assert_eq!((&read.best, &read.visits), (&saved.best, &saved.visits));
        assert!((read.value - saved.value).abs() < 1e-4, "four decimals");
    }

    #[test]
    fn states_round_trip_through_their_keys() {
        let mut agent = RandomAgent::<BitState>::new(3);
        for _ in 0..50 {
            let mut s = BitState::initial();
            while !s.is_terminal() {
                for state in [s, canonical(&s).1] {
                    let back = state_of_cells(&cells_of(&state)).expect("reachable");
                    assert_eq!(cells_of(&back), cells_of(&state));
                    assert_eq!(key_of(&back), key_of(&state));
                    // From the key alone: the same key, side to move and moves.
                    let made = state_of_key(&key_of(&state)).expect("a key's position");
                    assert_eq!(key_of(&made), key_of(&state));
                    assert_eq!(made.to_move(), Player::X);
                    assert_eq!(made.legal_moves_vec(), state.legal_moves_vec());
                    assert_eq!(back.to_move(), state.to_move());
                    assert_eq!(back.legal_moves_vec(), state.legal_moves_vec());
                }
                s.apply(agent.choose_move(&s).into());
            }
        }
        assert!(state_of_cells(&format!("{}9", "0".repeat(81))).is_ok());
        assert!(state_of_cells(&format!("{}4", "0".repeat(81))).is_err());
        assert!(state_of_cells(&format!("1{}9", "1".repeat(80))).is_err());
        assert!(state_of_cells("01").is_err());
        assert_eq!(
            state_of_key(&format!("{}9", "0".repeat(81))),
            Ok(BitState::initial())
        );
        assert!(state_of_key(&format!("{}4", "0".repeat(81))).is_err());
        assert!(
            state_of_key(&format!("1{}9", "0".repeat(80))).is_err(),
            "free, nothing decided"
        );
        assert!(state_of_key(&format!("{}{}9", "3".repeat(8), "0".repeat(73))).is_err());
    }

    #[test]
    fn agreement_counts_symmetric_moves() {
        let empty = BitState::initial();
        let corner = Move::new(sq(0, 0), sq(0, 0));
        let far_corner = Move::new(sq(2, 2), sq(2, 2));
        let centre = Move::new(sq(1, 1), sq(1, 1));
        assert_eq!(equivalent_moves(&empty, corner).len(), 4);
        assert_eq!(equivalent_moves(&empty, centre), vec![centre]);
        let policy = [(far_corner, 0.3), (corner, 0.3), (centre, 0.4)];
        // The corners together have 0.6 against the centre's 0.4.
        assert!(agent_agrees(&empty, corner, far_corner, &policy, 0.15));
        assert!(!agent_agrees(&empty, corner, far_corner, &policy, 0.25));
        assert!(!agent_agrees(&empty, corner, centre, &policy, 0.0));
        // X in the top-middle square of the top-middle board keeps only the left-right
        // mirror: O's top corners of that board are the same move, its corner and edge not.
        let mut s = empty;
        s.apply(Move::new(sq(0, 1), sq(0, 1)).into());
        let a = Move::new(sq(0, 1), sq(0, 0));
        let b = Move::new(sq(0, 1), sq(0, 2));
        let edge = Move::new(sq(0, 1), sq(1, 0));
        assert_eq!(equivalent_moves(&s, a), vec![a, b]);
        assert!(agent_agrees(&s, a, b, &[(b, 1.0)], 1.0));
        assert!(!agent_agrees(&s, a, edge, &[(edge, 1.0)], 0.0));
    }

    #[test]
    fn common_lines_judge_each_reply_by_its_own_share() {
        // X's book move a; O replies b (10%) or c (90%); after b, X's book move d; O replies
        // e (10%) or f (90%).
        let mut book = Book::default();
        let insert = |book: &mut Book, state: &BitState, best: Move, visits: &[(Move, u32)]| {
            book.insert(
                state,
                Entry {
                    best,
                    value: 0.0,
                    visits: visits.to_vec(),
                    dropped: false,
                },
            );
        };
        let play = |s: &BitState, mv: Move| {
            let mut next = *s;
            next.apply(mv.into());
            next
        };
        let root = BitState::initial();
        let a = Move::new(sq(1, 1), sq(0, 1));
        let (b, c) = (Move::new(sq(0, 1), sq(0, 0)), Move::new(sq(0, 1), sq(2, 2)));
        let d = Move::new(sq(0, 0), sq(1, 1));
        let (e, f) = (Move::new(sq(1, 1), sq(0, 0)), Move::new(sq(1, 1), sq(2, 2)));
        insert(&mut book, &root, a, &[(a, 100)]);
        let after_a = play(&root, a);
        insert(&mut book, &after_a, c, &[(b, 10), (c, 90)]);
        let after_b = play(&after_a, b);
        insert(&mut book, &after_b, d, &[(d, 100)]);
        let after_d = play(&after_b, d);
        insert(&mut book, &after_d, f, &[(e, 10), (f, 90)]);
        let after_e = play(&after_d, e);
        insert(&mut book, &after_e, a, &[(a, 1)]);
        let key = |s: &BitState| canonical(s).2;

        let common = book.common_lines(0.05);
        for s in [&root, &after_b, &after_e] {
            assert!(common.contains(&key(s)), "X to move on a common line");
        }
        // Positions where O is to move count on the lines where the book plays O: after a
        // (X's only move) yes, after d no, since there O plays its book move c rather than b.
        assert!(common.contains(&key(&after_a)));
        assert!(
            !common.contains(&key(&after_d)),
            "O meets d only after O's book move c"
        );
        let rare = book.common_lines(0.2);
        assert!(rare.contains(&key(&root)));
        assert!(!rare.contains(&key(&after_b)) && !rare.contains(&key(&after_e)));
    }

    #[test]
    fn lookup_maps_moves_back_through_the_symmetry() {
        let mut book = Book::default();
        let mut s = BitState::initial();
        s.apply(Move::new(sq(0, 1), sq(2, 0)).into()); // X top-middle board, bottom-left square
        let reply = Move::new(sq(2, 0), sq(1, 1)); // O in the forced board's centre
        book.insert(
            &s,
            Entry {
                best: reply,
                value: -0.1,
                visits: vec![(reply, 90), (Move::new(sq(2, 0), sq(0, 0)), 10)],
                dropped: false,
            },
        );
        let found = book.lookup(&s).unwrap();
        assert_eq!(found.best, reply);
        assert_eq!(found.visits[0], (reply, 90));
        // And from any symmetric image of the position.
        for t in Symmetry::ALL {
            let got = book.best_move(&s.transform(t)).unwrap();
            assert_eq!(got, t.apply_move(reply), "{t:?}");
        }
        assert_eq!(book.best_move(&BitState::initial()), None);
    }

    #[test]
    fn settled_entries_need_no_search_of_that_budget() {
        let entry = |visits: &[(usize, u32)]| Entry {
            best: Move::from_index(visits[0].0),
            value: 0.0,
            visits: visits
                .iter()
                .map(|&(i, n)| (Move::from_index(i), n))
                .collect(),
            dropped: false,
        };
        // A full search of 1000 simulations: settled for that budget, not for a bigger one.
        let full = entry(&[(40, 600), (0, 399)]);
        assert!(full.settled(1000));
        assert!(!full.settled(2000));
        // Stopped early at 600 of 1000: the lead of 450 is more than the 400 left.
        let stopped = entry(&[(40, 500), (0, 50), (1, 49)]);
        assert!(stopped.settled(1000));
        assert!(!stopped.settled(1100), "a lead of 450 with 500 left");
    }

    #[test]
    fn file_round_trip() {
        let mut book = Book::default();
        let key = cells_of(&BitState::initial());
        book.insert(
            &BitState::initial(),
            Entry {
                best: Move::from_index(40),
                value: 0.0625,
                visits: vec![(Move::from_index(40), 7), (Move::from_index(0), 3)],
                dropped: false,
            },
        );
        let path = std::env::temp_dir().join(format!("uttt-book-{}.txt", std::process::id()));
        book.save(&path, &["test".into()]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# test\n"));
        let back = Book::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(back.entries, book.entries);
        assert!(Book::parse("bad line").is_err());
        assert!(Book::parse(&format!("{key} 81 0.0")).is_err());
        assert!(Book::parse(&format!("{key} 40 0.0 40:1 dropped extra")).is_err());
    }

    #[test]
    fn dropped_entries_are_kept_for_walks_but_not_looked_up() {
        let root = BitState::initial();
        let centre = Move::new(sq(1, 1), sq(1, 1));
        let reply = Move::new(sq(1, 1), sq(0, 0));
        let mut after = root;
        after.apply(centre.into());
        let mut book = Book::default();
        for (state, best) in [(root, centre), (after, reply)] {
            book.insert(
                &state,
                Entry {
                    best,
                    value: 0.0,
                    visits: vec![(best, 10)],
                    dropped: false,
                },
            );
        }
        assert!(book.drop_entry(&key_of(&root)));
        assert!(!book.drop_entry(&key_of(&root)), "already dropped");
        assert_eq!(
            (book.len(), book.live_len(), book.entries().count()),
            (2, 1, 1)
        );
        assert_eq!(book.best_move(&root), None);
        assert_eq!(book.best_move(&after), Some(reply));
        // The walk still passes through the dropped root to reach O's position.
        assert!(book.common_lines(0.5).contains(&canonical(&after).2));

        let path = std::env::temp_dir().join(format!("uttt-drop-{}.txt", std::process::id()));
        book.save(&path, &[]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let back = Book::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(text.lines().next(), Some(FORMAT_LINE));
        assert!(text.lines().nth(1).unwrap().ends_with(" dropped"), "{text}");
        // Keys only count as keys after the format line: without it, the root's key reads
        // as the old format's cells, and the other line's (O to move, so its digits are
        // colour-swapped) does not.
        assert!(Book::parse(text.strip_prefix(FORMAT_LINE).unwrap()).is_err());
        assert_eq!(back.entries, book.entries);
    }

    /// The pending positions' states, the stored entries standing in for any search.
    fn missing(book: &Book, cfg: &BookConfig) -> Vec<BitState> {
        book.pending(cfg, 1).into_iter().map(|p| p.state).collect()
    }

    #[test]
    fn pending_follows_best_moves_and_covered_replies() {
        let cfg = BookConfig {
            plies: 2,
            min_prior: 0.2,
            concurrent: 4,
            ..Default::default()
        };
        let mut book = Book::default();
        assert_eq!(missing(&book, &cfg), vec![BitState::initial()]);

        // Root entry: X's best is the centre; two other first moves have 25% and 5% shares.
        let centre = Move::new(sq(1, 1), sq(1, 1));
        let corner = Move::new(sq(0, 0), sq(0, 0));
        let edge = Move::new(sq(0, 1), sq(0, 1));
        book.insert(
            &BitState::initial(),
            Entry {
                best: centre,
                value: 0.1,
                visits: vec![(centre, 70), (corner, 25), (edge, 5)],
                dropped: false,
            },
        );
        let frontier = missing(&book, &cfg);
        // Ply 1: after the centre (the X book) and after the corner (the O book covers X's
        // replies with at least 20%); the edge is pruned.
        assert_eq!(frontier.len(), 2);
        let keys: HashSet<Key> = frontier.iter().map(key_of).collect();
        let mut after_centre = BitState::initial();
        after_centre.apply(centre.into());
        let mut after_corner = BitState::initial();
        after_corner.apply(corner.into());
        assert!(keys.contains(&canonical(&after_centre).2));
        assert!(keys.contains(&canonical(&after_corner).2));
        assert!(!frontier.iter().any(|s| s.to_move() == Player::X));

        // Main lines only: just the book move's child. (One ply deeper, so that the opponent's
        // positions at ply 2 below want a search for their replies.)
        let main = BookConfig {
            main_lines: true,
            plies: 3,
            ..cfg.clone()
        };
        assert_eq!(
            missing(&book, &main),
            vec![canonical(&after_centre).1],
            "only the centre"
        );
        // An existing reply branch is still walked, and grows along its own main line.
        let reply = Move::new(sq(0, 0), sq(1, 1));
        book.insert(
            &after_corner,
            Entry {
                best: reply,
                value: 0.0,
                visits: vec![(reply, 9), (edge, 1)],
                dropped: false,
            },
        );
        let mut after_reply = after_corner;
        after_reply.apply(reply.into());
        let keys: HashSet<Key> = missing(&book, &main).iter().map(key_of).collect();
        let expected: HashSet<Key> = [canonical(&after_centre).2, canonical(&after_reply).2]
            .into_iter()
            .collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn extend_searches_the_frontier_round_by_round() {
        // With a 20-simulation search nothing reaches a sizeable visit share, so every
        // reply is covered to exercise both trees.
        let cfg = BookConfig {
            plies: 2,
            min_prior: 0.0,
            concurrent: 8,
            ..Default::default()
        };
        let search = MctsConfig {
            simulations: 20,
            batch_size: 4,
            ..Default::default()
        };
        let mut evaluator = RandomAgent::<BitState>::new(1);
        let mut book = Book::default();
        let mut rounds = Vec::new();
        let mut saves = 0;
        let added = book
            .extend(
                std::slice::from_mut(&mut evaluator),
                &search,
                &cfg,
                |l| rounds.push(l),
                |_, _| {},
                |b| {
                    saves += 1;
                    assert!(!b.is_empty());
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(
            added,
            Extension {
                added: book.len(),
                ..Extension::default()
            }
        );
        assert_eq!(saves, 3);
        assert_eq!(rounds.len(), 3, "{rounds:?}");
        assert!(rounds[0].contains("1 positions at ply 0"));
        assert!(book.best_move(&BitState::initial()).is_some());
        // Ply 1 holds the 15 canonical first moves (the O book covers them all).
        assert!(rounds[1].contains("15 positions at ply 1"), "{rounds:?}");
        assert!(book.pending(&cfg, search.simulations).is_empty());

        // Dropped entries are pending again, and come back from their stored search.
        let ply_one: Vec<Key> = book
            .plies()
            .into_iter()
            .filter(|(_, ply)| *ply == 1)
            .map(|(k, _)| k)
            .collect();
        for key in &ply_one {
            book.drop_entry(key);
        }
        let stored = book.entries.clone();
        assert_eq!(book.pending(&cfg, search.simulations).len(), ply_one.len());
        let again = book
            .extend(
                std::slice::from_mut(&mut evaluator),
                &search,
                &cfg,
                |_| {},
                |_, _| {},
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(
            again,
            Extension {
                revived: ply_one.len(),
                ..Extension::default()
            }
        );
        for (key, entry) in &stored {
            assert_eq!(book.entries[key].visits, entry.visits, "stored search kept");
        }
        // Stored searches kept whatever their budget: the dropped ones come back, and nothing
        // is searched again.
        for key in &ply_one {
            book.drop_entry(key);
        }
        let keep = BookConfig {
            keep_searched: true,
            ..cfg.clone()
        };
        let bigger = search.simulations * 10;
        assert_eq!(book.pending(&keep, bigger).len(), ply_one.len());
        let kept = book
            .extend(
                std::slice::from_mut(&mut evaluator),
                &MctsConfig {
                    simulations: bigger,
                    ..search.clone()
                },
                &keep,
                |_| {},
                |_, _| {},
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(
            kept,
            Extension {
                revived: ply_one.len(),
                ..Extension::default()
            }
        );
        // A budget the stored searches cannot stand in for: searched again instead.
        for key in &ply_one {
            book.drop_entry(key);
        }
        let deeper = MctsConfig {
            simulations: search.simulations * 10,
            ..search.clone()
        };
        let mut rounds = Vec::new();
        let again = book
            .extend(
                std::slice::from_mut(&mut evaluator),
                &deeper,
                &cfg,
                |l| rounds.push(l),
                |_, _| {},
                |_| Ok(()),
            )
            .unwrap();
        // The live entries had the smaller budget too: searched again, shallowest first, the
        // root alone before the positions below it. (A new best move opens lines the book did
        // not have, so some entries are added.)
        assert!(
            rounds[0].contains("1 positions at ply 0 (1 in the book already"),
            "{rounds:?}"
        );
        assert!(again.researched > ply_one.len(), "{again:?}");
        assert_eq!(again.revived, 0);
        assert!(book.pending(&cfg, deeper.simulations).is_empty());
        assert_eq!(book.live_len(), book.len());
        assert_eq!(book.live_len(), book.len());
        // Every book position answers from any symmetric image.
        let mut s = BitState::initial();
        s.apply(book.best_move(&s).unwrap().into());
        for sym in Symmetry::ALL {
            assert!(book.best_move(&s.transform(sym)).is_some());
        }
    }

    #[test]
    fn dense_plies_open_opponent_positions_and_replies_get_their_own_budget() {
        let search = MctsConfig {
            simulations: 40,
            batch_size: 4,
            ..Default::default()
        };
        let mut evaluator = RandomAgent::<BitState>::new(2);
        // Dense to ply 3: the O book's positions after its move at ply 1 (X to move at ply 2)
        // are only marked open, and every X reply from them gets O's move at ply 3.
        let cfg = BookConfig {
            plies: 3,
            dense_plies: 3,
            min_prior: 0.5,
            concurrent: 8,
            ..Default::default()
        };
        let mut book = Book::default();
        let done = book
            .extend(
                std::slice::from_mut(&mut evaluator),
                &search,
                &cfg,
                |_| {},
                |_, _| {},
                |_| Ok(()),
            )
            .unwrap();
        assert!(
            done.opened > 0 && done.opened == book.open_len(),
            "{done:?}"
        );
        // Complete to ply 3: play leaves the book after ply 3 at the latest, never at an open
        // position, and every game leaves it (none ends this early).
        let leaves = book.open_leaves();
        assert!(
            leaves
                .iter()
                .all(|l| l.ply <= 4 && l.reached != Reached::OpenReply),
            "{leaves:?}"
        );
        for side in [Player::X, Player::O] {
            let reach: f64 = leaves
                .iter()
                .filter(|l| l.book == side)
                .map(|l| l.reach)
                .sum();
            assert!((reach - 1.0).abs() < 1e-9, "{side:?}: {reach}");
        }
        // X's centre with an O reply share of 10% for the corner; the centre's position is
        // open, with none of O's replies after it searched.
        let centre = Move::new(sq(1, 1), sq(1, 1));
        let corner = Move::new(sq(0, 0), sq(0, 0));
        let mut lone = Book::default();
        lone.insert(
            &BitState::initial(),
            Entry {
                best: centre,
                value: 0.0,
                visits: vec![(centre, 90), (corner, 10)],
                dropped: false,
            },
        );
        let mut after_centre = BitState::initial();
        after_centre.apply(centre.into());
        lone.open(&after_centre);
        let leaves = lone.open_leaves();
        // The X book plays the centre into the open position, and has no move after any of
        // O's replies there, taken uniformly: the corners, edges and centre of the centre
        // board.
        let x: Vec<&Leaf> = leaves.iter().filter(|l| l.book == Player::X).collect();
        assert!(
            x.iter()
                .all(|l| l.ply == 2 && l.reached == Reached::OpenReply)
        );
        let mut reach: Vec<f64> = x.iter().map(|l| l.reach).collect();
        reach.sort_by(f64::total_cmp);
        assert_eq!(reach.len(), 3);
        for (r, expected) in reach.iter().zip([1.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0]) {
            assert!((r - expected).abs() < 1e-9, "{reach:?}");
        }
        // The O book has no move after any first move up to symmetry: the centre (90% of the
        // visits), the corners (10%) and 13 others without visits.
        let o: Vec<&Leaf> = leaves.iter().filter(|l| l.book == Player::O).collect();
        assert_eq!(o.len(), 15);
        assert!(o.iter().all(|l| l.ply == 1));
        assert!(
            o.iter()
                .any(|l| l.reached == Reached::Reply(0.9) && (l.reach - 0.9).abs() < 1e-9)
        );
        assert!(o.iter().any(|l| (l.reach - 0.1).abs() < 1e-9));
        assert_eq!(o.iter().filter(|l| l.reach == 0.0).count(), 13);
        assert!(book.pending(&cfg, search.simulations).is_empty());
        for key in &book.open {
            assert!(book.get(key).is_none());
            let state = state_of_key(key).unwrap();
            assert_eq!(
                book.lookup(&state),
                None,
                "nothing to play from an open position"
            );
            for mv in state.legal_moves_vec() {
                let mut reply = state;
                reply.apply(mv.into());
                assert!(reply.is_terminal() || book.lookup(&reply).is_some(), "{mv}");
            }
        }
        // Saved and loaded, the open positions are kept.
        let path = std::env::temp_dir().join(format!("uttt-open-{}.txt", std::process::id()));
        book.save(&path, &[]).unwrap();
        let back = Book::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(back.open, book.open);
        assert!(back.pending(&cfg, search.simulations).is_empty());

        // Dense to ply 2 only: the opponent's positions at ply 2 are searched for their
        // replies, with their own budget and no early stop.
        let cfg = BookConfig {
            dense_plies: 2,
            reply_simulations: Some(10),
            ..cfg
        };
        // Two threads, each with its evaluator.
        let mut book = Book::default();
        let mut rounds = Vec::new();
        book.extend(
            &mut [RandomAgent::new(3), RandomAgent::new(4)],
            &search,
            &cfg,
            |l| rounds.push(l),
            |_, _| {},
            |_| Ok(()),
        )
        .unwrap();
        assert!(
            rounds
                .iter()
                .any(|r| r.contains("for their replies, at 10 simulations")),
            "{rounds:?}"
        );
        let totals: HashSet<u32> = book.all_entries().map(|(_, e)| e.total_visits()).collect();
        assert_eq!(totals, HashSet::from([9, 39]));
        assert_eq!(book.open_len(), 0);
    }
}
