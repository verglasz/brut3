//! The opening book as a sorted byte stream of canonical keys and moves (`BookContent.hs`),
//! its decoding and the square permutations of the eight symmetries (`Book.hs`), and
//! reference lookups that pin down the canonicalisation (`BookTest.hs`).
//!
//! A book key ([`crate::book::Key`]) becomes nine board codes, an open board's nine digits
//! (0 empty, 1 X, 2 O) as a base-3 number with square 0 most significant (0..19682) or 19683,
//! 19684, 19685 for a board won by X, won by O or full, plus the forced board (9 = free) and
//! the side to move (0 X, 1 O). In Haskell a key is three `Word64`s, each packing three codes
//! in base 19686, the last one times 20 plus `2 * forced + toMove`. Comparing the words in
//! order is the same as comparing the key strings, so "the smallest image" means the same
//! thing on both sides.

use std::fmt::Write as _;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::haskell::{
    banner, bundle_body_begin, bundle_body_end, bundle_imports, data_warning, hs_list,
};
use super::quant::base64;
use crate::book::{Book, KEY_LEN, Key, key_of};
use crate::game::{BitState, GameState, Move, Player, Square, Symmetry};

/// Bytes per entry: nine little-endian `Word16` board codes, `2 * forced + toMove`, the move.
const ENTRY_BYTES: usize = 20;

/// The first code of a decided board: won by X; then won by O, full.
const DECIDED_CODE: u16 = 19683;

/// A key's nine board codes and `2 * forced + to_move`. In this order they compare like the
/// key strings.
pub fn board_codes(key: &Key) -> ([u16; 9], u8) {
    let digits: Vec<u8> = key.bytes().map(|b| b - b'0').collect();
    assert_eq!(digits.len(), KEY_LEN, "book key");
    let codes = std::array::from_fn(|board| {
        let cells = &digits[board * 9..(board + 1) * 9];
        if cells[0] >= 3 {
            DECIDED_CODE + u16::from(cells[0] - 3)
        } else {
            cells.iter().fold(0u16, |acc, &d| acc * 3 + u16::from(d))
        }
    });
    (codes, 2 * digits[81] + digits[82])
}

/// Every entry as `(board codes, 2 * forced + to_move, best move index)`, sorted by key.
pub fn entries(book: &Book) -> Vec<([u16; 9], u8, usize)> {
    let mut out: Vec<_> = book
        .entries()
        .map(|(key, entry)| {
            let (codes, forced) = board_codes(key);
            (codes, forced, entry.best.index())
        })
        .collect();
    out.sort();
    out
}

fn encode(entries: &[([u16; 9], u8, usize)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(entries.len() * ENTRY_BYTES);
    for (codes, forced_to_move, mv) in entries {
        for c in codes {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        bytes.push(*forced_to_move);
        bytes.push(*mv as u8);
    }
    bytes
}

/// Image of square `i` under each symmetry, in `Symmetry::ALL` order.
pub fn square_permutations() -> Vec<[usize; 9]> {
    Symmetry::ALL
        .iter()
        .map(|sym| std::array::from_fn(|i| sym.apply(Square::from_index(i)).index()))
        .collect()
}

/// Positions to look up and the expected answer: book positions reached from the initial
/// position through covered moves, each under a random symmetry, a few book positions with a
/// decided board likewise, plus some positions off the book. Expected moves are in the looked-up position's own frame.
pub fn reference_lookups(book: &Book, seed: u64, n: usize) -> Vec<(BitState, Option<Move>)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut queue = std::collections::VecDeque::from([BitState::initial()]);
    let mut seen = std::collections::HashSet::new();
    let mut in_book = Vec::new();
    while let Some(state) = queue.pop_front() {
        if in_book.len() >= n {
            break;
        }
        let Some(entry) = book.lookup(&state) else {
            continue;
        };
        if !seen.insert(key_of(&state)) {
            continue;
        }
        in_book.push(state);
        let total: u32 = entry.visits.iter().map(|(_, v)| v).sum::<u32>().max(1);
        for (mv, visits) in &entry.visits {
            if *visits as f32 / total as f32 >= 0.01 {
                let mut child = state;
                child.apply(*mv);
                if !child.is_terminal() {
                    queue.push_back(child);
                }
            }
        }
    }
    let mut out: Vec<(BitState, Option<Move>)> = in_book
        .into_iter()
        .map(|state| {
            let sym = Symmetry::ALL[rng.random_range(0..Symmetry::ALL.len())];
            let image = state.transform(sym);
            let expected = book.lookup(&image).map(|e| e.best);
            (image, expected)
        })
        .collect();
    // Book positions with a decided board, whose key holds the board's status only.
    let decided: Vec<BitState> = book
        .positions()
        .into_iter()
        .filter(|(key, _)| book.get(key).is_some_and(|e| !e.dropped))
        .map(|(_, state)| state)
        .filter(|state| (0..9).any(|b| state.board_status(Square::from_index(b)).is_decided()))
        .collect();
    for _ in 0..decided.len().min(8) {
        let state = decided[rng.random_range(0..decided.len())];
        let image = state.transform(Symmetry::ALL[rng.random_range(0..Symmetry::ALL.len())]);
        let expected = book.lookup(&image).map(|e| e.best);
        out.push((image, expected));
    }
    // Off-book positions: random play well past the book's depth.
    for _ in 0..4 {
        let mut state = BitState::initial();
        for _ in 0..30 {
            if state.is_terminal() {
                break;
            }
            let legal = state.legal_moves_vec();
            state.apply(legal[rng.random_range(0..legal.len())]);
        }
        if !state.is_terminal() {
            let expected = book.lookup(&state).map(|e| e.best);
            out.push((state, expected));
        }
    }
    out
}

/// Statistics for the export report.
#[derive(Clone, Debug)]
pub struct BookStats {
    pub entries: usize,
    pub bytes: usize,
    pub literals: usize,
    pub lookups: usize,
}

/// `Book.hs` (what the engine needs), `BookContent.hs` (the entries it decodes) and
/// `BookTest.hs` (its verification).
pub fn book_modules(
    book: &Book,
    prefix: &str,
    chunk: usize,
    seed: u64,
    lookups: usize,
) -> (String, String, String, BookStats) {
    let entries = entries(book);
    let bytes = encode(&entries);
    let encoded = base64(&bytes);
    let pieces: Vec<&str> = {
        let mut out = Vec::new();
        let mut rest = encoded.as_str();
        while !rest.is_empty() {
            let n = chunk.min(rest.len());
            out.push(&rest[..n]);
            rest = &rest[n..];
        }
        if out.is_empty() {
            out.push("");
        }
        out
    };
    let refs = reference_lookups(book, seed, lookups);

    let mut s = String::with_capacity(8192);
    banner(
        &mut s,
        &format!("{prefix}.Book"),
        "(BookKey, boardCode, bookKey, bookKeyOfBoards, bookEntries, squarePermutations)",
    );
    let _ = writeln!(s, "import Decode (decodeBytes)");
    let _ = writeln!(s, "import {prefix}.BookContent (bookText)");
    bundle_imports(&mut s, &["import Data.Word (Word64, Word8)"]);
    bundle_body_begin(
        &mut s,
        "BookKey, boardCode, bookKey, bookKeyOfBoards, squarePermutations, bookEntries",
    );
    s.push_str(
        r#"-- | A position's key: nine board codes (see boardCode), the forced board (0..8, or 9 when
-- free) and the side to move (0 X, 1 O), as three words that each pack three codes in base
-- 19686, the last one times 20 plus 2 * forced + toMove. Positions that differ only inside
-- decided boards share a key.
type BookKey = (Word64, Word64, Word64)

-- | A board's code from its nine cells (0 empty, 1 X, 2 O): 19683 if X has a line, 19684 if
-- O has, 19685 if it is full, otherwise the digits as a base-3 number, square 0 most
-- significant.
boardCode :: [Int] -> Int
boardCode cells
  | won 1 = 19683
  | won 2 = 19684
  | all (/= 0) cells = 19685
  | otherwise = foldl (\acc d -> acc * 3 + d) 0 cells
  where
    won v = any (all (\q -> cells !! q == v)) [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

-- | The key of 81 cells (by board * 9 + square), the forced board (9 when free) and the side
-- to move (0 X, 1 O), in the frame they are given in.
bookKey :: [Int] -> Int -> Int -> BookKey
bookKey cells = bookKeyOfBoards [boardCode (take 9 (drop (9 * b) cells)) | b <- [0 .. 8]]

-- | The key of nine board codes, the forced board and the side to move.
bookKeyOfBoards :: [Int] -> Int -> Int -> BookKey
bookKeyOfBoards codes forcedBoard toMove =
  let word = foldl (\acc c -> acc * 19686 + fromIntegral c) 0
   in (word (take 3 codes), word (take 3 (drop 3 codes)), word (drop 6 codes) * 20 + fromIntegral (2 * forcedBoard + toMove))

-- | Image of square i under each of the eight symmetries (identity first). A symmetry maps
-- the cell at (board b, square q) to (p !! b, p !! q) and the forced board b to p !! b.
squarePermutations :: [[Int]]
"#,
    );
    let perms = square_permutations();
    let _ = writeln!(
        s,
        "squarePermutations = {}\n",
        hs_list(perms.iter().map(|p| hs_list(p.iter())))
    );
    s.push_str(
        r#"-- | Every entry (canonical key, best move index in the canonical frame), ascending by key.
bookEntries :: [(BookKey, Int)]
bookEntries = go (map (\b -> fromIntegral (fromIntegral b :: Word8)) (decodeBytes (concat bookText)))
  where
    -- the bytes as 0..255, whether decodeBytes gives Int8 or Word8
    go :: [Int] -> [(BookKey, Int)]
    go [] = []
    go bytes =
      let (entry, rest) = splitAt 20 bytes
          code i = entry !! (2 * i) + 256 * entry !! (2 * i + 1)
       in if length entry == 20
            then (bookKeyOfBoards (map code [0 .. 8]) (entry !! 18 `div` 2) (entry !! 18 `mod` 2), entry !! 19) : go rest
            else error "bookEntries: ragged book stream"
"#,
    );
    bundle_body_end(&mut s);

    // The data: the entries' base64 text.
    let mut c = String::with_capacity(encoded.len() + 4096);
    banner(&mut c, &format!("{prefix}.BookContent"), "(bookText)");
    bundle_imports(&mut c, &[]);
    bundle_body_begin(
        &mut c,
        &format!("bookText, book_0 .. book_{}", pieces.len() - 1),
    );
    data_warning(
        &mut c,
        &format!("the opening book's {} entries", entries.len()),
        entries.len() * ENTRY_BYTES * 4 / 3,
    );
    let _ = writeln!(
        c,
        "-- | {} entries of 25 bytes each (three little-endian Word64 then the move), base64.",
        entries.len()
    );
    let names: Vec<String> = (0..pieces.len()).map(|i| format!("book_{i}")).collect();
    let _ = writeln!(c, "bookText :: [String]");
    let _ = writeln!(c, "bookText = [{}]\n", names.join(", "));
    for (name, text) in names.iter().zip(&pieces) {
        let _ = writeln!(c, "{name} :: String");
        let _ = writeln!(c, "{name} =");
        super::haskell::write_string_literal(&mut c, text, super::haskell::LITERAL_WIDTH);
        c.push('\n');
    }
    bundle_body_end(&mut c);
    // The verification module: a reference canonicalisation and lookup, and the positions
    // to check them against.
    let mut t = String::with_capacity(16384);
    banner(
        &mut t,
        &format!("{prefix}.BookTest"),
        "(canonicalKey, lookupBook, bookRefs, bookTest)",
    );
    let _ = writeln!(t, "import Data.List (minimumBy)");
    let _ = writeln!(t, "import Data.Ord (comparing)");
    let _ = writeln!(t, "import {prefix}.Book\n");
    t.push_str(
        r#"-- | The cells of a position's image under a permutation: the image's cell at
-- (p b, p q) is the position's cell at (b, q).
permuteCells :: [Int] -> [Int] -> [Int]
permuteCells p cells =
  let inv = [length (takeWhile (/= i) p) | i <- [0 .. 8]]
   in [cells !! ((inv !! b) * 9 + inv !! q) | b <- [0 .. 8], q <- [0 .. 8]]

-- | The smallest key among the position's eight images, with the permutation that produced
-- it (the first one on ties, as the book was built).
canonicalKey :: [Int] -> Int -> Int -> (BookKey, [Int])
canonicalKey cells forced toMove =
  minimumBy (comparing fst)
    [ (bookKey (permuteCells p cells) (if forced == 9 then 9 else p !! forced) toMove, p)
    | p <- squarePermutations
    ]

-- | The book move for a position, in the position's own frame. A linear search: the search
-- engine should build a balanced tree from bookEntries instead.
lookupBook :: [Int] -> Int -> Int -> Maybe Int
lookupBook cells forced toMove =
  let (key, p) = canonicalKey cells forced toMove
      inv = [length (takeWhile (/= i) p) | i <- [0 .. 8]]
   in case lookup key bookEntries of
        Nothing -> Nothing
        Just m -> Just ((inv !! (m `div` 9)) * 9 + inv !! (m `mod` 9))

-- | Positions (cells, forced board, side to move) with the move the Rust book gives for them,
-- -1 if the position is not in the book. Mostly book positions under a random symmetry.
bookRefs :: [([Int], Int, Int, Int)]
bookRefs =
"#,
    );
    let s_main = s;
    let mut s = t;
    for (i, (state, expected)) in refs.iter().enumerate() {
        let cells = (0..Move::COUNT).map(|i| match state.cell(Move::from_index(i)) {
            None => 0,
            Some(Player::X) => 1,
            Some(Player::O) => 2,
        });
        let _ = writeln!(
            s,
            "  {} ({}, {}, {}, {})",
            if i == 0 { "[" } else { "," },
            hs_list(cells),
            state.forced_board().map_or(9, |b| b.index()),
            usize::from(state.to_move() == Player::O),
            expected.map_or(-1, |m| m.index() as i64)
        );
    }
    let _ = writeln!(s, "  ]\n");
    s.push_str(
        r#"-- | Looks every reference position up and compares.
bookTest :: IO Bool
bookTest = do
  let results = [(expected, maybe (-1) id (lookupBook cells forced toMove)) | (cells, forced, toMove, expected) <- bookRefs]
      bad = length (filter (uncurry (/=)) results)
  putStrLn ("book test: " ++ show (length bookEntries) ++ " entries, " ++ show (length results) ++ " lookups, " ++ show bad ++ " mismatches")
  return (bad == 0)
"#,
    );
    let stats = BookStats {
        entries: entries.len(),
        bytes: bytes.len(),
        literals: pieces.len(),
        lookups: refs.len(),
    };
    (s_main, c, s, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::canonical;

    /// The three words from board codes, as `bookKeyOfBoards` computes them.
    fn words(key: &Key) -> [u64; 3] {
        let (codes, forced_to_move) = board_codes(key);
        let word = |bs: &[u16]| bs.iter().fold(0u64, |acc, &c| acc * 19686 + u64::from(c));
        [
            word(&codes[0..3]),
            word(&codes[3..6]),
            word(&codes[6..9]) * 20 + u64::from(forced_to_move),
        ]
    }

    /// Board-code order and word order are key order, decided boards included.
    #[test]
    fn key_words_order_like_keys() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut keys = Vec::new();
        for _ in 0..300 {
            let mut state = BitState::initial();
            for _ in 0..rng.random_range(0..60) {
                if state.is_terminal() {
                    break;
                }
                let legal = state.legal_moves_vec();
                state.apply(legal[rng.random_range(0..legal.len())]);
            }
            keys.push(canonical(&state).2);
        }
        let decided = |k: &Key| board_codes(k).0.iter().any(|&c| c >= DECIDED_CODE);
        assert!(keys.iter().filter(|k| decided(k)).count() > 50);
        for a in &keys {
            for b in &keys {
                assert_eq!(a.cmp(b), board_codes(a).cmp(&board_codes(b)), "{a} vs {b}");
                assert_eq!(a.cmp(b), words(a).cmp(&words(b)), "{a} vs {b}");
            }
            assert!(board_codes(a).0.iter().all(|&c| c < DECIDED_CODE + 3));
        }
    }

    #[test]
    fn permutations_match_the_symmetries() {
        let perms = square_permutations();
        assert_eq!(perms[0], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        for (sym, p) in Symmetry::ALL.iter().zip(&perms) {
            let mut sorted = *p;
            sorted.sort();
            assert_eq!(
                sorted,
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "{sym:?} is a permutation"
            );
            let mv = Move::new(Square::from_index(1), Square::from_index(5));
            assert_eq!(sym.apply_move(mv).index(), p[1] * 9 + p[5]);
        }
    }
}
