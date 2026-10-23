//! An opening book: positions of the first plies with the move a deep search chose, built
//! incrementally.
//!
//! The book serves both colours. For the colour we play, a position needs only our move; for
//! the opponent's turns it needs an entry under every reply we cover. So the book is the
//! union of two trees: from our nodes only the best move is followed, from the opponent's
//! every reply whose share of the search's visits is at least a threshold, once with us as X
//! and once as O. Positions are stored under their canonical form over the eight board
//! symmetries, with the moves in that frame; a lookup canonicalises the live position and
//! maps the move back.
//!
//! Each entry keeps the search's root value and visit counts over the moves, not just the
//! move, so that deepening the book later can extend the frontier from the file alone and
//! re-choose moves from their children's values.
//!
//! File format: lines of `KEY MOVE VALUE VISITS`, where `KEY` is 81 cell digits by
//! [`Move::index`] (0 empty, 1 X, 2 O) followed by the forced board's index (9 for a free
//! choice), `MOVE` the chosen move's index, `VALUE` the root value for the side to move, and
//! `VISITS` a comma-separated list of `move:count` over the moves. Lines starting with `#`
//! are comments.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::path::Path;

use crate::agent::UtttAgent;
use crate::game::{BitState, GameState, Move, Player, Symmetry};
use crate::mcts::{MctsConfig, SearchResult};
use crate::selfplay::search_positions;

/// A book position's key: 81 cell digits then the forced board digit.
pub type Key = String;

/// What the search found from a position, in the position's canonical frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub best: Move,
    /// Root value for the side to move, in `[-1, 1]`.
    pub value: f32,
    /// Visit counts of the root's moves.
    pub visits: Vec<(Move, u32)>,
}

impl Entry {
    fn from_result(result: &SearchResult) -> Option<Self> {
        Some(Entry {
            best: result.best_move()?,
            value: result.root_value,
            visits: result.visits.clone(),
        })
    }

    fn total_visits(&self) -> u32 {
        self.visits.iter().map(|(_, n)| n).sum()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Book {
    entries: HashMap<Key, Entry>,
}

/// The key of `state` as it stands (no canonicalisation).
pub fn key_of(state: &BitState) -> Key {
    let mut key = String::with_capacity(82);
    for i in 0..Move::COUNT {
        key.push(match state.cell(Move::from_index(i)) {
            None => '0',
            Some(Player::X) => '1',
            Some(Player::O) => '2',
        });
    }
    key.push(match state.forced_board() {
        Some(b) => char::from(b'0' + b.index() as u8),
        None => '9',
    });
    key
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

impl Book {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The book's entry for `state`, if any, with its moves mapped back to `state`'s frame.
    pub fn lookup(&self, state: &BitState) -> Option<Entry> {
        let (sym, _, key) = canonical(state);
        let entry = self.entries.get(&key)?;
        let back = sym.inverse();
        Some(Entry {
            best: back.apply_move(entry.best),
            value: entry.value,
            visits: entry
                .visits
                .iter()
                .map(|(mv, n)| (back.apply_move(*mv), *n))
                .collect(),
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

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut book = Book::default();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, entry) = parse_entry(line).map_err(|e| format!("line {}: {e}", i + 1))?;
            book.entries.insert(key, entry);
        }
        Ok(book)
    }

    /// Writes the book, entries sorted by key, after `header` comment lines.
    pub fn save(&self, path: &Path, header: &[String]) -> Result<(), String> {
        let mut keys: Vec<&Key> = self.entries.keys().collect();
        keys.sort();
        let mut out = String::new();
        for line in header {
            let _ = writeln!(out, "# {line}");
        }
        for key in keys {
            let e = &self.entries[key];
            let _ = write!(out, "{key} {} {:.4}", e.best.index(), e.value);
            for (i, (mv, n)) in e.visits.iter().enumerate() {
                out.push(if i == 0 { ' ' } else { ',' });
                let _ = write!(out, "{}:{n}", mv.index());
            }
            out.push('\n');
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    /// The positions the book should hold under `cfg` but does not yet, as canonical states:
    /// the frontier of the union tree, at most `cfg.plies` plies deep.
    pub fn frontier(&self, cfg: &BookConfig) -> Vec<BitState> {
        let mut frontier = Vec::new();
        let mut frontier_keys: HashSet<Key> = HashSet::new();
        // A node's children depend on which of the two trees it belongs to, so the visited
        // set is keyed by that too.
        let mut seen: HashSet<(Key, bool, bool)> = HashSet::new();
        let mut queue: VecDeque<(BitState, usize, bool, bool)> = VecDeque::new();
        queue.push_back((BitState::initial(), 0, true, true));
        while let Some((state, ply, x_strict, o_strict)) = queue.pop_front() {
            if state.is_terminal() {
                continue;
            }
            let (sym, image, key) = canonical(&state);
            if !seen.insert((key.clone(), x_strict, o_strict)) {
                continue;
            }
            let Some(entry) = self.entries.get(&key) else {
                if ply <= cfg.plies && frontier_keys.insert(key) {
                    frontier.push(image);
                }
                continue;
            };
            if ply >= cfg.plies {
                continue;
            }
            let back = sym.inverse();
            let best = back.apply_move(entry.best);
            let total = entry.total_visits().max(1) as f32;
            let mover = state.to_move();
            for (mv, n) in &entry.visits {
                let mv = back.apply_move(*mv);
                let covered = *n as f32 / total >= cfg.min_prior;
                let is_best = mv == best;
                let (xs, os) = match mover {
                    Player::X => (x_strict && is_best, o_strict && covered),
                    Player::O => (x_strict && covered, o_strict && is_best),
                };
                if xs || os {
                    let mut child = state;
                    child.apply(mv);
                    queue.push_back((child, ply + 1, xs, os));
                }
            }
        }
        frontier
    }

    /// Searches the frontier under `cfg` with `search` over `evaluator`, round after round
    /// (each round one ply deeper), until the book is complete to `cfg.plies`. `log` gets a
    /// line per round; `on_done` the count of positions searched so far in the round;
    /// `after_round` the book after each round's entries are in (to save it provisionally),
    /// and its error ends the run. Returns the number of entries added.
    pub fn extend<A: UtttAgent<State = BitState>>(
        &mut self,
        evaluator: &mut A,
        search: &MctsConfig,
        cfg: &BookConfig,
        mut log: impl FnMut(String),
        mut on_done: impl FnMut(usize, usize),
        mut after_round: impl FnMut(&Book) -> Result<(), String>,
    ) -> Result<usize, String> {
        let before = self.len();
        loop {
            let frontier = self.frontier(cfg);
            if frontier.is_empty() {
                break;
            }
            let plies = frontier
                .iter()
                .map(|s| {
                    key_of(s)
                        .bytes()
                        .take(Move::COUNT)
                        .filter(|&b| b != b'0')
                        .count()
                })
                .fold((usize::MAX, 0), |(lo, hi), p| (lo.min(p), hi.max(p)));
            log(format!(
                "searching {} positions at ply {}{}",
                frontier.len(),
                plies.0,
                if plies.1 != plies.0 {
                    format!(" to {}", plies.1)
                } else {
                    String::new()
                }
            ));
            let total = frontier.len();
            let results = search_positions(evaluator, &frontier, search, cfg.concurrent, |n| {
                on_done(n, total)
            });
            for (state, result) in frontier.iter().zip(&results) {
                if let Some(entry) = Entry::from_result(result) {
                    self.entries.insert(key_of(state), entry);
                }
            }
            after_round(self)?;
        }
        Ok(self.len() - before)
    }
}

fn parse_entry(line: &str) -> Result<(Key, Entry), String> {
    let mut fields = line.split_whitespace();
    let key = fields.next().ok_or("empty line")?;
    if key.len() != Move::COUNT + 1 || !key.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("bad key {key:?}"));
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
    if let Some(list) = fields.next() {
        for item in list.split(',') {
            let (mv, n) = item
                .split_once(':')
                .ok_or_else(|| format!("bad visits {item:?}"))?;
            let n: u32 = n.parse().map_err(|_| format!("bad count {n:?}"))?;
            visits.push((move_index(mv)?, n));
        }
    }
    if let Some(extra) = fields.next() {
        return Err(format!("unexpected field {extra:?}"));
    }
    Ok((
        key.to_string(),
        Entry {
            best,
            value,
            visits,
        },
    ))
}

#[derive(Clone, Debug)]
pub struct BookConfig {
    /// Deepest ply to hold entries for (the initial position is ply 0).
    pub plies: usize,
    /// Smallest share of a position's visits for an opponent reply to be covered.
    pub min_prior: f32,
    /// Positions searched at once.
    pub concurrent: usize,
}

impl Default for BookConfig {
    fn default() -> Self {
        Self {
            plies: 8,
            min_prior: 0.01,
            concurrent: 256,
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
        s.apply(Move::new(sq(0, 1), sq(2, 0)));
        s.apply(Move::new(sq(2, 0), sq(0, 0)));
        let (_, image, key) = canonical(&s);
        for sym in Symmetry::ALL {
            let (_, img, k) = canonical(&s.transform(sym));
            assert_eq!(k, key, "{sym:?}");
            assert_eq!(key_of(&img), key);
            assert_eq!(img, image);
        }
        // The forced board is board 0's image under the canonical symmetry.
        let forced: usize = key[81..].parse().unwrap();
        assert!(forced < 9);
        assert_eq!(image.forced_board(), Some(Square::from_index(forced)));
    }

    #[test]
    fn lookup_maps_moves_back_through_the_symmetry() {
        let mut book = Book::default();
        let mut s = BitState::initial();
        s.apply(Move::new(sq(0, 1), sq(2, 0))); // X top-middle board, bottom-left square
        let (sym, image, key) = canonical(&s);
        let reply = Move::new(sq(2, 0), sq(1, 1)); // O in the forced board's centre
        let reply_c = sym.apply_move(reply);
        book.entries.insert(
            key,
            Entry {
                best: reply_c,
                value: -0.1,
                visits: vec![
                    (reply_c, 90),
                    (sym.apply_move(Move::new(sq(2, 0), sq(0, 0))), 10),
                ],
            },
        );
        let _ = image;
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
    fn file_round_trip() {
        let mut book = Book::default();
        let key = key_of(&BitState::initial());
        book.entries.insert(
            key.clone(),
            Entry {
                best: Move::from_index(40),
                value: 0.0625,
                visits: vec![(Move::from_index(40), 7), (Move::from_index(0), 3)],
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
    }

    #[test]
    fn frontier_follows_best_moves_and_covered_replies() {
        let cfg = BookConfig {
            plies: 2,
            min_prior: 0.2,
            concurrent: 4,
        };
        let mut book = Book::default();
        assert_eq!(book.frontier(&cfg), vec![BitState::initial()]);

        // Root entry: X's best is the centre; two other first moves have 25% and 5% shares.
        let centre = Move::new(sq(1, 1), sq(1, 1));
        let corner = Move::new(sq(0, 0), sq(0, 0));
        let edge = Move::new(sq(0, 1), sq(0, 1));
        book.entries.insert(
            key_of(&BitState::initial()),
            Entry {
                best: centre,
                value: 0.1,
                visits: vec![(centre, 70), (corner, 25), (edge, 5)],
            },
        );
        let frontier = book.frontier(&cfg);
        // Ply 1: after the centre (the X book) and after the corner (the O book covers X's
        // replies with at least 20%); the edge is pruned.
        assert_eq!(frontier.len(), 2);
        let keys: HashSet<Key> = frontier.iter().map(key_of).collect();
        let mut after_centre = BitState::initial();
        after_centre.apply(centre);
        let mut after_corner = BitState::initial();
        after_corner.apply(corner);
        assert!(keys.contains(&canonical(&after_centre).2));
        assert!(keys.contains(&canonical(&after_corner).2));
        assert!(!frontier.iter().any(|s| s.to_move() == Player::X));
    }

    #[test]
    fn extend_searches_the_frontier_round_by_round() {
        // With a 20-simulation search nothing reaches a sizeable visit share, so every
        // reply is covered to exercise both trees.
        let cfg = BookConfig {
            plies: 2,
            min_prior: 0.0,
            concurrent: 8,
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
                &mut evaluator,
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
        assert_eq!(added, book.len());
        assert_eq!(saves, 3);
        assert_eq!(rounds.len(), 3, "{rounds:?}");
        assert!(rounds[0].contains("1 positions at ply 0"));
        assert!(book.best_move(&BitState::initial()).is_some());
        // Ply 1 holds the 15 canonical first moves (the O book covers them all).
        assert!(rounds[1].contains("15 positions at ply 1"), "{rounds:?}");
        assert!(book.frontier(&cfg).is_empty());
        // Every book position answers from any symmetric image.
        let mut s = BitState::initial();
        s.apply(book.best_move(&s).unwrap());
        for sym in Symmetry::ALL {
            assert!(book.best_move(&s.transform(sym)).is_some());
        }
    }
}
