//! `Book.hs`: the opening book as a sorted byte stream of canonical keys and moves, the
//! square permutations of the eight symmetries, and reference lookups that pin down the
//! canonicalisation.
//!
//! A key is three `Word64`s: the 81 cell digits (0 empty, 1 X, 2 O, by `board * 9 + square`)
//! as three base-3 numbers of 27 digits each, most significant first, with the last word
//! multiplied by ten plus the forced-board digit (9 = free). Comparing the words in order is
//! the same as comparing the 82-character key strings, so "the smallest image" means the same
//! thing on both sides.

use std::fmt::Write as _;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::haskell::hs_list;
use super::quant::base64;
use crate::book::{Book, Key, key_of};
use crate::game::{BitState, GameState, Move, Player, Square, Symmetry};

/// Bytes per entry: three little-endian `Word64`s and the move index.
const ENTRY_BYTES: usize = 25;

/// The three words of a key string.
pub fn key_words(key: &Key) -> [u64; 3] {
    let digits: Vec<u64> = key.bytes().map(|b| u64::from(b - b'0')).collect();
    assert_eq!(digits.len(), 82, "key of 81 cells and the forced digit");
    let word =
        |range: std::ops::Range<usize>| digits[range].iter().fold(0u64, |acc, &d| acc * 3 + d);
    [word(0..27), word(27..54), word(54..81) * 10 + digits[81]]
}

/// Every entry as `(words, best move index)`, sorted by key.
pub fn entries(book: &Book) -> Vec<([u64; 3], usize)> {
    let mut out: Vec<([u64; 3], usize)> = book
        .entries()
        .map(|(key, entry)| (key_words(key), entry.best.index()))
        .collect();
    out.sort();
    out
}

fn encode(entries: &[([u64; 3], usize)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(entries.len() * ENTRY_BYTES);
    for (words, mv) in entries {
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
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
/// position through covered moves, each under a random symmetry, plus some positions off
/// the book. Expected moves are in the looked-up position's own frame.
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

/// `Book.hs` (what the engine needs) and `BookTest.hs` (its verification).
pub fn book_modules(
    book: &Book,
    prefix: &str,
    chunk: usize,
    seed: u64,
    lookups: usize,
) -> (String, String, BookStats) {
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

    let mut s = String::with_capacity(encoded.len() + 8192);
    let _ = writeln!(s, "{{- FOURMOLU_DISABLE -}}");
    let _ = writeln!(s, "{{-# LANGUAGE Safe #-}}");
    let _ = writeln!(
        s,
        "-- Generated by `ultimate-ttt-burn export`. Do not edit: regenerate instead."
    );
    let _ = writeln!(
        s,
        "module {prefix}.Book (BookKey, bookKey, bookEntries, squarePermutations) where\n"
    );
    let _ = writeln!(s, "import Data.Word (Word64)");
    let _ = writeln!(s, "import {prefix}.Decode (decodeBytes)\n");
    s.push_str(
        r#"-- | A position's key: the 81 cell digits (0 empty, 1 X, 2 O, by board * 9 + square) as
-- three base-3 numbers of 27 digits, most significant first, the last times ten plus the
-- forced board (0..8, or 9 when free). Keys compare like the digit strings.
type BookKey = (Word64, Word64, Word64)

bookKey :: [Int] -> Int -> BookKey
bookKey cells forced =
  let word = foldl (\acc d -> acc * 3 + fromIntegral d) 0
   in (word (take 27 cells), word (take 27 (drop 27 cells)), word (drop 54 cells) * 10 + fromIntegral forced)

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
bookEntries = go (decodeBytes (concat bookText))
  where
    go [] = []
    go bytes =
      let (entry, rest) = splitAt 25 bytes
          word i = foldr (\b acc -> acc * 256 + fromIntegral b) 0 (take 8 (drop (8 * i) entry)) :: Word64
       in if length entry == 25
            then ((word 0, word 1, word 2), entry !! 24) : go rest
            else error "bookEntries: ragged book stream"

"#,
    );
    let _ = writeln!(
        s,
        "-- | {} entries of 25 bytes each (three little-endian Word64 then the move), base64.",
        entries.len()
    );
    let names: Vec<String> = (0..pieces.len()).map(|i| format!("book_{i}")).collect();
    let _ = writeln!(s, "bookText :: [String]");
    let _ = writeln!(s, "bookText = [{}]\n", names.join(", "));
    for (name, text) in names.iter().zip(&pieces) {
        let _ = writeln!(s, "{name} :: String");
        let _ = writeln!(s, "{name} =");
        super::haskell::write_string_literal(&mut s, text, super::haskell::LITERAL_WIDTH);
        s.push('\n');
    }
    // The verification module: a reference canonicalisation and lookup, and the positions
    // to check them against.
    let mut t = String::with_capacity(16384);
    let _ = writeln!(t, "{{- FOURMOLU_DISABLE -}}");
    let _ = writeln!(t, "{{-# LANGUAGE Safe #-}}");
    let _ = writeln!(
        t,
        "-- Generated by `ultimate-ttt-burn export`. Do not edit: regenerate instead."
    );
    let _ = writeln!(
        t,
        "module {prefix}.BookTest (canonicalKey, lookupBook, bookRefs, bookTest) where\n"
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
canonicalKey :: [Int] -> Int -> (BookKey, [Int])
canonicalKey cells forced =
  minimumBy (comparing fst)
    [ (bookKey (permuteCells p cells) (if forced == 9 then 9 else p !! forced), p)
    | p <- squarePermutations
    ]

-- | The book move for a position, in the position's own frame. A linear search: the search
-- engine should build a balanced tree from bookEntries instead.
lookupBook :: [Int] -> Int -> Maybe Int
lookupBook cells forced =
  let (key, p) = canonicalKey cells forced
      inv = [length (takeWhile (/= i) p) | i <- [0 .. 8]]
   in case lookup key bookEntries of
        Nothing -> Nothing
        Just m -> Just ((inv !! (m `div` 9)) * 9 + inv !! (m `mod` 9))

-- | Positions (cells, forced board) with the move the Rust book gives for them, -1 if the
-- position is not in the book. Mostly book positions under a random symmetry.
bookRefs :: [([Int], Int, Int)]
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
            "  {} ({}, {}, {})",
            if i == 0 { "[" } else { "," },
            hs_list(cells),
            state.forced_board().map_or(9, |b| b.index()),
            expected.map_or(-1, |m| m.index() as i64)
        );
    }
    let _ = writeln!(s, "  ]\n");
    s.push_str(
        r#"-- | Looks every reference position up and compares.
bookTest :: IO Bool
bookTest = do
  let results = [(expected, maybe (-1) id (lookupBook cells forced)) | (cells, forced, expected) <- bookRefs]
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
    (s_main, s, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::canonical;

    /// Word order is string order.
    #[test]
    fn key_words_order_like_keys() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut keys = Vec::new();
        for _ in 0..200 {
            let mut state = BitState::initial();
            for _ in 0..rng.random_range(0..20) {
                if state.is_terminal() {
                    break;
                }
                let legal = state.legal_moves_vec();
                state.apply(legal[rng.random_range(0..legal.len())]);
            }
            keys.push(canonical(&state).2);
        }
        for a in &keys {
            for b in &keys {
                assert_eq!(a.cmp(b), key_words(a).cmp(&key_words(b)), "{a} vs {b}");
            }
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
