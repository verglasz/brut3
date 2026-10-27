//! The opening book as a sorted byte stream of canonical keys and moves (`BookContent.hs`),
//! its decoding and the square permutations of the eight symmetries (`Book.hs`), and
//! reference lookups that pin down the canonicalisation (`BookTest.hs`).
//!
//! A book key ([`crate::book::Key`], from the side to move's view) becomes nine board codes,
//! an open board's nine digits (0 empty, 1 the side to move's, 2 the opponent's) as a base-3
//! number with square 0 most significant (0..19682) or 19683, 19684, 19685 for a board won by
//! the side to move, won by the opponent or full, plus the forced board (9 = free). An entry
//! stores them as they are. The Haskell key is a record of five `Word32`s, each two of those
//! fields side by side (`hi << 16 | lo`), whose derived order is the key strings' order:
//! "the smallest image" means the same thing on both sides.

use std::fmt::Write as _;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::haskell::{
    banner,
    bundle_body_begin,
    bundle_body_end,
    bundle_imports,
    data_warning,
    hs_list,
};
use super::quant::base64;
use crate::book::{Book, KEY_LEN, Key, key_of};
use crate::game::{BitState, GameState, Move, Square, Symmetry};

/// Bytes per entry: nine little-endian `Word16` board codes, the forced board, the move.
const ENTRY_BYTES: usize = 20;

/// The first code of a decided board: won by the side to move; then won by the opponent,
/// full.
const DECIDED_CODE: u16 = 19683;

/// A key's nine board codes and forced board. In this order they compare like the key
/// strings.
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
    (codes, digits[81])
}

/// Every entry as `(board codes, forced board, best move index)`, sorted by key.
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
    for (codes, forced, mv) in entries {
        for c in codes {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        bytes.push(*forced);
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
                child.apply((*mv).into());
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
            state.apply(legal[rng.random_range(0..legal.len())].into());
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
        "(BookKey (..), Symmetry, bookKeyOfBoards, bookEntries, squarePermutations)",
    );
    let _ = writeln!(s, "import Decode (decodeU16lePiecesWith)");
    let _ = writeln!(s, "import {prefix}.BookContent (bookText)");
    bundle_imports(
        &mut s,
        &[
            "import Data.Bits ((.&.), (.<<.), (.>>.), (.|.))",
            "import Data.Word (Word16, Word32, Word8)",
        ],
    );
    bundle_body_begin(
        &mut s,
        "BookKey (constructor BookKey), Symmetry, bookKeyOfBoards, squarePermutations, bookEntries",
    );
    s.push_str(
        r#"-- | A position's key, from the side to move's view: the nine boards' codes and the forced
-- board (0..8, or 9 when free), two per word, the first in the high 16 bits. A board's code:
-- an open board's nine cells (0 empty, 1 the side to move's, 2 the opponent's) as a base-3
-- number, square 0 most significant (0..19682); 19683 if the side to move won it, 19684 if
-- the opponent did, 19685 if it is full. The derived order is the Rust book's key-string
-- order. Positions that differ only inside decided boards, or only by swapping colours along
-- with the side to move, share a key.
data BookKey = BookKey !Word32 !Word32 !Word32 !Word32 !Word32
  deriving (Eq, Ord, Show)

-- | The key of nine board codes (by board) and the forced board. Inlined, a call on a
-- list of known length builds the key with no list in between.
bookKeyOfBoards :: [Word16] -> Word8 -> BookKey
{-# INLINE bookKeyOfBoards #-}
bookKeyOfBoards codes forcedBoard = case map fromIntegral codes of
  [c0, c1, c2, c3, c4, c5, c6, c7, c8] ->
    BookKey (pair c0 c1) (pair c2 c3) (pair c4 c5) (pair c6 c7) (pair c8 (fromIntegral forcedBoard))
  _ -> error "bookKeyOfBoards: not nine boards"
  where
    pair hi lo = hi .<<. 16 .|. lo

-- | One of the eight board symmetries, as the image of each square: p !! i for square i. It
-- maps the cell at (board b, square q) to (p !! b, p !! q) and the forced board b to p !! b.
type Symmetry = [Int]

-- | The eight symmetries, identity first, in the order the book breaks ties between images in.
squarePermutations :: [Symmetry]
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
-- The stream is ten little-endian Word16 per entry: the nine board codes, then the forced
-- board in the low byte and the move in the high byte.
bookEntries :: [(BookKey, Int)]
bookEntries = go (decodeU16lePiecesWith id bookText)
  where
    go (c0 : c1 : c2 : c3 : c4 : c5 : c6 : c7 : c8 : forcedAndMove : rest) =
      let !key = bookKeyOfBoards [c0, c1, c2, c3, c4, c5, c6, c7, c8] (fromIntegral (forcedAndMove .&. 0xFF))
          !move = fromIntegral (forcedAndMove .>>. 8)
       in (key, move) : go rest
    go [] = []
    go _ = error "bookEntries: ragged book stream"
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
        "-- | {} entries of {ENTRY_BYTES} bytes each, ascending by key, as one base64 text: the nine board\n\
         -- codes of the key (little-endian Word16, see BookKey in Book), the forced board and the move.",
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
        "(boardCode, canonicalKey, lookupBook, bookRefs, bookTest)",
    );
    let _ = writeln!(t, "import Data.List (minimumBy)");
    let _ = writeln!(t, "import Data.Ord (comparing)");
    let _ = writeln!(t, "import {prefix}.Book\n");
    t.push_str(
        r#"-- | The cells of a position's image under a permutation: the image's cell at
-- (p b, p q) is the position's cell at (b, q).
permuteCells :: Symmetry -> [Int] -> [Int]
permuteCells p cells =
  let inv = [length (takeWhile (/= i) p) | i <- [0 .. 8]]
   in [cells !! ((inv !! b) * 9 + inv !! q) | b <- [0 .. 8], q <- [0 .. 8]]

-- | A board's code from its nine cells (see BookKey).
boardCode :: [Int] -> Int
boardCode cells
  | won 1 = 19683
  | won 2 = 19684
  | all (/= 0) cells = 19685
  | otherwise = foldl (\acc d -> acc * 3 + d) 0 cells
  where
    won v = any (all (\q -> cells !! q == v)) [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

-- | The key of 81 cells (by board * 9 + square, from the side to move's view) and the
-- forced board.
keyOfCells :: [Int] -> Int -> BookKey
keyOfCells cells forced =
  bookKeyOfBoards [fromIntegral (boardCode (take 9 (drop (9 * b) cells))) | b <- [0 .. 8]] (fromIntegral forced)

-- | The smallest key among the position's eight images, with the permutation that produced
-- it (the first one on ties, as the book was built).
canonicalKey :: [Int] -> Int -> (BookKey, Symmetry)
canonicalKey cells forced =
  minimumBy (comparing fst)
    [ (keyOfCells (permuteCells p cells) (if forced == 9 then 9 else p !! forced), p)
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

-- | Positions (cells from the side to move's view: 0 empty, 1 its, 2 the opponent's; forced
-- board) with the move the Rust book gives for them, -1 if the position is not in the book.
-- Mostly book positions under a random symmetry, with either side to move.
bookRefs :: [([Int], Int, Int)]
bookRefs =
"#,
    );
    let s_main = s;
    let mut s = t;
    for (i, (state, expected)) in refs.iter().enumerate() {
        let me = state.to_move();
        let cells = (0..Move::COUNT).map(|i| match state.cell(Move::from_index(i)) {
            None => 0,
            Some(p) if p == me => 1,
            Some(_) => 2,
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
    (s_main, c, s, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::canonical;

    /// The five words of a key, as `bookKeyOfBoards` builds them.
    fn words(key: &Key) -> [u32; 5] {
        let (codes, forced) = board_codes(key);
        let mut fields = codes.map(u32::from).to_vec();
        fields.push(u32::from(forced));
        std::array::from_fn(|i| (fields[2 * i] << 16) | fields[2 * i + 1])
    }

    /// Board-code order and word order (the Haskell key's derived order) are key order,
    /// decided boards included.
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
                state.apply(legal[rng.random_range(0..legal.len())].into());
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
