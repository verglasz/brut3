//! Haskell sources generated from [`Weights`]: the constants a port must agree on and the
//! reference evaluations it is tested against.
//!
//! Every file starts with `{- FOURMOLU_DISABLE -}` and a "generated" banner; none is meant
//! to be edited. Module names are `<prefix>.<Name>` under `<out>/<prefix>/`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::Weights;
use super::codegen;
use super::quant::base64;
use crate::book::Book;
use crate::game::{BitState, GameState, LINES, Move};
use crate::nn::model::MOVES;
use crate::nn::nnue::{
    BOARDS, CELL_BASE, CELL_PAD, CELLS, CONTENT_ROWS, CONTENTS, DECIDED_BASE, FEATURE_IDS,
    FEATURE_ROWS, FORCED_BASE, FULL, IDS, IDS_PER_BOARD, LINE_BASE, LINE_STATES, NO_FORCED,
    OPEN_CONTENTS, OPEN_MARKS_BASE, RAW_CONTENTS, TOKENS, WON_BY_ME, WON_BY_THEM, compact_id,
};

#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// Module prefix and directory name, e.g. `Gen` for `Gen/Spec.hs`.
    pub prefix: String,
    /// Reference positions to emit.
    pub positions: usize,
    /// Seed of the random games the reference positions come from.
    pub seed: u64,
    /// Characters of base64 per weight string literal.
    pub chunk: usize,
    /// Reference book lookups to emit.
    pub lookups: usize,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            prefix: "Gen".into(),
            positions: 16,
            seed: 1,
            chunk: 64 * 1024,
            lookups: 40,
        }
    }
}

/// What was written, for the command's report.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub files: Vec<(PathBuf, usize)>,
    /// Bytes of int16 weights and the number of base64 string literals holding them.
    pub stream_bytes: usize,
    pub chunks: usize,
    /// The `gen/vecgen.py` command that makes the `FixedVec` module the code needs.
    pub vecgen: String,
    /// Largest deviations of the dequantised weights from the exact ones over the reference
    /// positions: win/draw/loss probability, `p_win - p_loss`, legal-move probability.
    pub wdl_error: f32,
    pub value_error: f32,
    pub policy_error: f32,
    /// The book export, if one was written.
    pub book: Option<super::book::BookStats>,
}

/// Writes every generated module under `out`: the network's modules for `weights`, the
/// opening book's for `book`, and the decoder either way.
pub fn write_all(
    weights: Option<&Weights>,
    book: Option<&Book>,
    out: &Path,
    opts: &ExportOptions,
) -> Result<Report, String> {
    let dir = out.join(&opts.prefix);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let mut report = Report::default();
    let mut emit = |name: &str, text: String| -> Result<(), String> {
        let path = dir.join(format!("{name}.hs"));
        std::fs::write(&path, &text)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        report.files.push((path, text.len()));
        Ok(())
    };
    emit("Decode", codegen::decode_module(&opts.prefix))?;
    if let Some(weights) = weights {
        let rounded = weights.dequantised();
        let names = codegen::Names::new(&weights.config);
        emit("Spec", spec_module(weights, &opts.prefix))?;
        let positions = reference_positions(opts.seed, opts.positions);
        // The reference holds what the shipped (int16-rounded) network computes.
        emit(
            "Reference",
            reference_module(&rounded, &positions, &opts.prefix),
        )?;
        let (text, stream_bytes, chunks, manifest_names) =
            weights_module(weights, &opts.prefix, opts.chunk);
        emit("Weights", text)?;
        emit(
            "Net",
            codegen::net_module(&names, &opts.prefix, &manifest_names),
        )?;
        emit("SelfTest", codegen::selftest_module(&names, &opts.prefix))?;
        report.stream_bytes = stream_bytes;
        report.chunks = chunks;
        report.vecgen = codegen::vecgen_command(&names);
        for (_, state) in &positions {
            let exact = weights.evaluate(state);
            let other = rounded.evaluate(state);
            for (a, b) in exact.wdl.iter().zip(&other.wdl) {
                report.wdl_error = report.wdl_error.max((a - b).abs());
            }
            report.value_error = report
                .value_error
                .max((exact.value() - other.value()).abs());
            for (a, b) in exact.policy.iter().zip(&other.policy) {
                report.policy_error = report.policy_error.max((a.2 - b.2).abs());
            }
        }
    }
    if let Some(book) = book {
        let (text, stats) =
            super::book::book_module(book, &opts.prefix, opts.chunk, opts.seed, opts.lookups);
        emit("Book", text)?;
        report.book = Some(stats);
    }
    Ok(report)
}

/// `Weights.hs`: every array with its shape, scale and base64 text, each array's text in
/// its own string literals of at most `chunk` characters, wrapped with string gaps. Returns
/// the text, the total bytes of int16 values, the number of literals and the arrays' names.
pub fn weights_module(
    weights: &Weights,
    prefix: &str,
    chunk: usize,
) -> (String, usize, usize, Vec<String>) {
    let arrays = weights.quantised();
    let bytes: usize = arrays.iter().map(|a| a.bytes.len()).sum();
    let mut s = String::with_capacity(bytes * 4 / 3 + 4096);
    banner(&mut s, &format!("{prefix}.Weights"), "(Array (..), arrays)");
    // Nothing to optimise here, and the literals are large.
    s = s.replacen(
        "{- FOURMOLU_DISABLE -}\n",
        "{- FOURMOLU_DISABLE -}\n{-# LANGUAGE Safe #-}\n{-# OPTIONS_GHC -O0 #-}\n",
        1,
    );
    s.push_str(
        "-- | One weight array: rows * cols little-endian Int16 values, each worth q * scale,\n\
         -- matrices one row per output, as base64 text (standard alphabet, padded) split into\n\
         -- pieces to concatenate.\n\
         data Array = Array\n\
         \x20 { arrayName :: String\n\
         \x20 , arrayRows :: Int\n\
         \x20 , arrayCols :: Int\n\
         \x20 , arrayScale :: Float\n\
         \x20 , arrayText :: [String]\n\
         \x20 }\n\n",
    );
    let mut literals = 0;
    let _ = writeln!(s, "arrays :: [Array]");
    let _ = writeln!(s, "arrays =");
    for (i, a) in arrays.iter().enumerate() {
        let _ = writeln!(
            s,
            "  {} Array {:?} {} {} {} {}_text",
            if i == 0 { "[" } else { "," },
            a.name,
            a.rows,
            a.cols,
            hs_float(a.scale),
            a.name
        );
    }
    let _ = writeln!(s, "  ]\n");
    for a in &arrays {
        let encoded = base64(&a.bytes);
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
        literals += pieces.len();
        let names: Vec<String> = (0..pieces.len())
            .map(|i| format!("{}_{i}", a.name))
            .collect();
        let _ = writeln!(s, "{}_text :: [String]", a.name);
        let _ = writeln!(s, "{}_text = [{}]\n", a.name, names.join(", "));
        for (name, text) in names.iter().zip(&pieces) {
            let _ = writeln!(s, "{name} :: String");
            let _ = writeln!(s, "{name} =");
            write_string_literal(&mut s, text, LITERAL_WIDTH);
            s.push('\n');
        }
    }
    let names = arrays.iter().map(|a| a.name.clone()).collect();
    (s, bytes, literals, names)
}

/// Characters of base64 per line of a string literal.
pub(super) const LITERAL_WIDTH: usize = 1024;

/// A string literal over several lines, joined by string gaps (`\` newline `\`), indented
/// two spaces.
pub(super) fn write_string_literal(s: &mut String, text: &str, width: usize) {
    let lines: Vec<&str> = if text.is_empty() {
        vec![""]
    } else {
        text.as_bytes()
            .chunks(width)
            .map(|c| std::str::from_utf8(c).expect("ascii"))
            .collect()
    };
    for (i, line) in lines.iter().enumerate() {
        let open = if i == 0 { '"' } else { '\\' };
        let close = if i + 1 == lines.len() { "\"" } else { "\\" };
        let _ = writeln!(s, "  {open}{line}{close}");
    }
}

fn banner(out: &mut String, module: &str, exports: &str) {
    let _ = writeln!(out, "{{- FOURMOLU_DISABLE -}}");
    let _ = writeln!(
        out,
        "-- Generated by `ultimate-ttt-burn export`. Do not edit: regenerate instead."
    );
    let exports = if exports.is_empty() {
        String::new()
    } else {
        format!(" {exports}")
    };
    let _ = writeln!(out, "module {module}{exports} where\n");
}

/// Checksum of the open raw codes in increasing order, so that a port's own enumeration can
/// be verified: `h <- (h * 1000003 + code) mod 2^32`, starting from 0.
pub fn open_codes_checksum() -> u64 {
    let mut h: u64 = 0;
    for code in 0..RAW_CONTENTS {
        if matches!(compact_id(code), Some(id) if id < OPEN_CONTENTS) {
            h = (h * 1_000_003 + code as u64) % (1 << 32);
        }
    }
    h
}

/// `Spec.hs`: the model's widths and every id convention, as `Int` constants.
pub fn spec_module(weights: &Weights, prefix: &str) -> String {
    let c = &weights.config;
    let mut s = String::new();
    banner(&mut s, &format!("{prefix}.Spec"), "");
    let unreachable = (0..RAW_CONTENTS)
        .filter(|&code| compact_id(code).is_none())
        .count();
    let groups: [(&str, Vec<(&str, usize)>); 5] = [
        (
            "Model widths.",
            vec![
                ("dEmbed", c.d_embed),
                ("dAcc", c.d_acc),
                ("dHidden", c.d_hidden),
                ("nLayers", c.layers),
            ],
        ),
        (
            "Compact content ids: open contents in increasing raw-code order, then the three \
             decided kinds. Raw code = sum over cells of v * 3^cell, v = 1 mine, 2 theirs.",
            vec![
                ("rawContents", RAW_CONTENTS),
                ("openContents", OPEN_CONTENTS),
                ("fullId", FULL),
                ("wonByMe", WON_BY_ME),
                ("wonByThem", WON_BY_THEM),
                ("contents", CONTENTS),
                ("unreachableContents", unreachable),
            ],
        ),
        (
            "Rows of the content table: the contents, the no-forced row, cell rows \
             (cellBase + cell * 3 + v, v = 0 empty, 1 mine, 2 theirs) and the padding row.",
            vec![
                ("noForced", NO_FORCED),
                ("cellBase", CELL_BASE),
                ("cellPad", CELL_PAD),
                ("contentRows", CONTENT_ROWS),
            ],
        ),
        (
            "Ids per position: ten tokens of (content, nine cells), then the feature ids.",
            vec![
                ("idsPerBoard", IDS_PER_BOARD),
                ("tokens", TOKENS),
                ("featureIds", FEATURE_IDS),
                ("idsPerPosition", IDS),
            ],
        ),
        (
            "Feature rows: forcedBase + forced board (9 = free); for each big line l, \
             lineBase + l * lineStates + 16 a + 4 b + c with statuses 0 open, 1 mine, \
             2 theirs, 3 full; decidedBase + decided boards; openMarksBase + marks on open \
             boards.",
            vec![
                ("forcedBase", FORCED_BASE),
                ("lineBase", LINE_BASE),
                ("lineStates", LINE_STATES),
                ("decidedBase", DECIDED_BASE),
                ("openMarksBase", OPEN_MARKS_BASE),
                ("featureRows", FEATURE_ROWS),
            ],
        ),
    ];
    for (comment, items) in &groups {
        for line in wrap(comment, 96) {
            let _ = writeln!(s, "-- {line}");
        }
        let names: Vec<&str> = items.iter().map(|(n, _)| *n).collect();
        let _ = writeln!(s, "{} :: Int", names.join(", "));
        for (name, value) in items {
            let _ = writeln!(s, "{name} = {value}");
        }
        s.push('\n');
    }
    let _ = writeln!(s, "-- The big-board lines, in feature order.");
    let _ = writeln!(s, "bigLines :: [[Int]]");
    let _ = writeln!(
        s,
        "bigLines = {}",
        hs_list(LINES.iter().map(|l| hs_list(l.iter())))
    );
    s.push('\n');
    let _ = writeln!(
        s,
        "-- Checksum of the open raw codes in increasing order: h <- (h * 1000003 + code) \
         `mod` 2^32, from 0."
    );
    let _ = writeln!(s, "openCodesChecksum :: Int");
    let _ = writeln!(s, "openCodesChecksum = {}", open_codes_checksum());
    s.push('\n');
    let _ = writeln!(
        s,
        "-- Trained parameters and multiply-adds per evaluation once folded."
    );
    let _ = writeln!(s, "parameters, foldedMacs :: Int");
    let _ = writeln!(s, "parameters = {}", weights.parameters());
    let _ = writeln!(s, "foldedMacs = {}", folded_macs(weights));
    s
}

fn folded_macs(weights: &Weights) -> usize {
    let c = &weights.config;
    c.d_acc * c.d_hidden
        + (c.layers - 1) * c.d_hidden * c.d_hidden
        + c.d_hidden * 3
        + c.d_acc * 3
        + c.d_hidden * MOVES
}

/// Named positions for the reference vectors: the initial position, then a mix of early,
/// middle, late (several decided boards) and free-choice positions from random games.
pub fn reference_positions(seed: u64, n: usize) -> Vec<(String, BitState)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut early = Vec::new();
    let mut mid = Vec::new();
    let mut late = Vec::new();
    let mut free = Vec::new();
    for game in 0..64 {
        let mut state = BitState::initial();
        let mut ply = 0;
        while !state.is_terminal() {
            let legal = state.legal_moves_vec();
            state.apply(legal[rng.random_range(0..legal.len())]);
            ply += 1;
            if state.is_terminal() {
                break;
            }
            let mut ids = [0i32; IDS];
            crate::nn::nnue::nnue_ids(&state, &mut ids);
            let decided = ids[TOKENS * IDS_PER_BOARD + 1 + LINES.len()] as usize - DECIDED_BASE;
            let name = format!("game{game}-ply{ply}");
            let entry = (name, state.clone());
            if state.forced_board().is_none() {
                free.push(entry);
            } else if decided >= 3 {
                late.push(entry);
            } else if ply <= 5 {
                early.push(entry);
            } else {
                mid.push(entry);
            }
        }
    }
    for bucket in [&mut early, &mut mid, &mut late, &mut free] {
        bucket.shuffle(&mut rng);
    }
    // Round-robin over the buckets, at most two positions from any one game.
    let mut out = vec![("initial".to_string(), BitState::initial())];
    let mut games: Vec<usize> = Vec::new();
    let mut buckets = [early, mid, late, free];
    let mut i = 0;
    while out.len() < n && buckets.iter().any(|b| !b.is_empty()) {
        if let Some((name, state)) = buckets[i % buckets.len()].pop() {
            let game = name[4..name.find('-').expect("game-ply")]
                .parse::<usize>()
                .expect("game number");
            if games.iter().filter(|&&g| g == game).count() < 2 {
                games.push(game);
                out.push((name, state));
            }
        }
        i += 1;
    }
    out
}

/// `Reference.hs`: each position with its ids and the scalar evaluator's outputs.
pub fn reference_module(
    weights: &Weights,
    positions: &[(String, BitState)],
    prefix: &str,
) -> String {
    let mut s = String::new();
    banner(&mut s, &format!("{prefix}.Reference"), "(Ref (..), refs)");
    s.push_str(
        "-- A position and what the shipped (int16-rounded) network computes for it, from the\n\
         -- side to move's view.\n\
         data Ref = Ref\n\
         \x20 { refName :: String\n\
         \x20 , refCells :: [Int]      -- 81 cells by board * 9 + square: 0 empty, 1 X, 2 O\n\
         \x20 , refToMove :: Int       -- 0 X, 1 O\n\
         \x20 , refLastMove :: Int     -- board * 9 + square of the last move, -1 if none\n\
         \x20 , refForced :: Int       -- forced board, 9 if free\n\
         \x20 , refRawCodes :: [Int]   -- the nine boards' raw content codes\n\
         \x20 , refIds :: [Int]        -- the idsPerPosition input ids\n\
         \x20 , refAcc :: [Float]      -- accumulator after the ReLU\n\
         \x20 , refHidden :: [Float]   -- last hidden layer\n\
         \x20 , refWdlLogits :: [Float]\n\
         \x20 , refWdl :: [Float]      -- softmax of the logits: win, draw, loss\n\
         \x20 , refPolicy :: [(Int, Float, Float)]  -- legal move index, logit, probability\n\
         \x20 }\n\n",
    );
    let _ = writeln!(s, "refs :: [Ref]");
    let _ = writeln!(s, "refs =");
    for (i, (name, state)) in positions.iter().enumerate() {
        let eval = weights.evaluate(state);
        let me = state.to_move();
        let cells = (0..MOVES).map(|i| match state.cell(Move::from_index(i)) {
            None => 0,
            Some(p) if p == crate::game::Player::X => 1,
            Some(_) => 2,
        });
        let raw = (0..BOARDS)
            .map(|b| raw_code(state.board_marks(me, b), state.board_marks(me.other(), b)));
        let _ = writeln!(s, "  {} Ref", if i == 0 { "[" } else { "," });
        let _ = writeln!(s, "      {{ refName = {name:?}");
        let _ = writeln!(s, "      , refCells = {}", hs_list(cells));
        let _ = writeln!(s, "      , refToMove = {}", me.index());
        let _ = writeln!(
            s,
            "      , refLastMove = {}",
            state.last_move().map_or(-1, |m| m.index() as i64)
        );
        let _ = writeln!(
            s,
            "      , refForced = {}",
            state.forced_board().map_or(BOARDS, |b| b.index())
        );
        let _ = writeln!(s, "      , refRawCodes = {}", hs_list(raw));
        let _ = writeln!(s, "      , refIds = {}", hs_list(eval.ids.iter()));
        let _ = writeln!(s, "      , refAcc = {}", hs_floats(&eval.acc));
        let _ = writeln!(s, "      , refHidden = {}", hs_floats(&eval.hidden));
        let _ = writeln!(s, "      , refWdlLogits = {}", hs_floats(&eval.wdl_logits));
        let _ = writeln!(s, "      , refWdl = {}", hs_floats(&eval.wdl));
        let policy = eval.policy.iter().map(|(mv, logit, p)| {
            format!("({}, {}, {})", mv.index(), hs_float(*logit), hs_float(*p))
        });
        let _ = writeln!(s, "      , refPolicy = {}", hs_list(policy));
        let _ = writeln!(s, "      }}");
    }
    let _ = writeln!(s, "  ]");
    s
}

/// Greedy word wrap to `width` characters.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = vec![String::new()];
    for word in text.split_whitespace() {
        let last = lines.last_mut().expect("one line");
        if !last.is_empty() && last.len() + 1 + word.len() > width {
            lines.push(word.to_string());
        } else {
            if !last.is_empty() {
                last.push(' ');
            }
            last.push_str(word);
        }
    }
    lines
}

fn raw_code(mine: u16, theirs: u16) -> usize {
    let mut code = 0;
    let mut pow = 1;
    for cell in 0..CELLS {
        if mine >> cell & 1 != 0 {
            code += pow;
        } else if theirs >> cell & 1 != 0 {
            code += 2 * pow;
        }
        pow *= 3;
    }
    code
}

pub(super) fn hs_list<T: std::fmt::Display>(items: impl Iterator<Item = T>) -> String {
    let items: Vec<String> = items.map(|x| x.to_string()).collect();
    format!("[{}]", items.join(", "))
}

/// A `Float` literal Haskell accepts: shortest round-trip form, always with a `.` or `e`.
pub fn hs_float(x: f32) -> String {
    assert!(x.is_finite(), "non-finite weight {x}");
    format!("{x:?}")
}

fn hs_floats(xs: &[f32]) -> String {
    hs_list(xs.iter().map(|&x| hs_float(x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_is_stable_and_counts_match() {
        assert_eq!(
            (0..RAW_CONTENTS)
                .filter(|&c| matches!(compact_id(c), Some(id) if id < OPEN_CONTENTS))
                .count(),
            OPEN_CONTENTS
        );
        assert_eq!(open_codes_checksum(), open_codes_checksum());
        assert!(open_codes_checksum() < 1 << 32);
    }

    #[test]
    fn reference_positions_cover_the_cases() {
        let positions = reference_positions(1, 16);
        assert_eq!(positions.len(), 16);
        assert_eq!(positions[0].0, "initial");
        assert!(
            positions
                .iter()
                .filter(|(_, s)| s.forced_board().is_none())
                .count()
                >= 3
        );
        let names: std::collections::HashSet<&str> =
            positions.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), 16, "distinct positions");
        assert_eq!(positions, reference_positions(1, 16), "deterministic");
    }

    #[test]
    fn float_literals_parse_as_haskell() {
        for x in [0.0f32, 1.0, -2.5, 1e-7, -3.25e9, 0.1] {
            let lit = hs_float(x);
            assert!(lit.contains('.') || lit.contains('e'), "{lit}");
            assert!(!lit.contains("inf") && !lit.contains("NaN"));
        }
    }
}
