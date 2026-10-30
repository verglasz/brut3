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
use super::vecgen::VecSpec;
use crate::book::Book;
use crate::game::{BitState, GameState, LINES, Move};
use crate::nn::model::MOVES;
use crate::nn::nnue::{
    BOARDS,
    CELL_BASE,
    CELL_PAD,
    CONTENT_ROWS,
    CONTENTS,
    DECIDED_BASE,
    FEATURE_IDS,
    FEATURE_ROWS,
    FORCED_BASE,
    FULL,
    IDS,
    IDS_PER_BOARD,
    LINE_BASE,
    LINE_STATES,
    NO_FORCED,
    OPEN_CONTENTS,
    OPEN_MARKS_BASE,
    RAW_CONTENTS,
    TOKENS,
    WON_BY_ME,
    WON_BY_THEM,
    compact_id,
    raw_code,
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
    /// Extra `FixedVec` shapes beyond the model's (the export's FixedVec flags).
    pub vec_extra: Option<VecSpec>,
    /// Whether to write `<prefix>/FixedVec.hs` at all.
    pub fixed_vec: bool,
    /// Ship every array as int8 with a scale per row (see
    /// [`quantise_int8`](super::quant::quantise_int8)) instead of int16: about half the text.
    pub int8: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            prefix: "Gen".into(),
            positions: 16,
            seed: 1,
            chunk: 64 * 1024,
            lookups: 40,
            vec_extra: None,
            fixed_vec: true,
            int8: false,
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
    /// The `FixedVec` module's shapes, if one was generated: the FixedVec flags that make
    /// the same module without a model.
    pub vec_flags: Option<String>,
    /// Largest deviations of the dequantised weights from the exact ones over the reference
    /// positions: win/draw/loss probability, `p_win - p_loss`, legal-move probability.
    pub wdl_error: f32,
    pub value_error: f32,
    pub policy_error: f32,
    /// The book export, if one was written.
    pub book: Option<super::book::BookStats>,
}

/// Writes every generated module under `out` (or prints them all to stdout if `out` is `-`,
/// each after a line `-- file: <module>/<name>.hs` with a blank line around it): the
/// network's modules for `weights`, the
/// opening book's for `book`, and `FixedVec` for the network's shapes plus `opts.vec_extra`
/// (without a network, for the extra shapes alone, if any). The network and book modules
/// import the Haskell project's hand-written `Decode`.
pub fn write_all(
    weights: Option<&Weights>,
    book: Option<&Book>,
    out: &Path,
    opts: &ExportOptions,
) -> Result<Report, String> {
    // `-`: every file to stdout instead, each after a `-- file: <module>/<name>.hs` line.
    let to_stdout = out == Path::new("-");
    let dir = if to_stdout {
        PathBuf::from(&opts.prefix)
    } else {
        out.join(&opts.prefix)
    };
    if !to_stdout {
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let mut report = Report::default();
    let mut emit = |name: &str, text: String| -> Result<(), String> {
        let path = dir.join(format!("{name}.hs"));
        if to_stdout {
            use std::io::Write as _;
            let mut stdout = std::io::stdout().lock();
            write!(stdout, "\n-- file: {}\n\n{text}", path.display())
                .and_then(|()| stdout.flush())
                .map_err(|e| format!("cannot print {}: {e}", path.display()))?;
        } else {
            std::fs::write(&path, &text)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        report.files.push((path, text.len()));
        Ok(())
    };
    if let Some(weights) = weights {
        let rounded = weights.dequantised(opts.int8);
        let names = codegen::Names::new(weights, opts.int8);
        emit("Spec", spec_module(weights, &opts.prefix))?;
        let positions = reference_positions(opts.seed, opts.positions);
        // The reference holds what the shipped (int16-rounded) network computes.
        emit(
            "Reference",
            reference_module(&rounded, &positions, &opts.prefix),
        )?;
        let (text, stream_bytes, chunks, manifest_names) =
            weights_module(weights, &opts.prefix, opts.chunk, opts.int8);
        emit("Weights", text)?;
        emit(
            "Net",
            codegen::net_module(&names, &opts.prefix, &manifest_names),
        )?;
        emit("SelfTest", codegen::selftest_module(&names, &opts.prefix))?;
        report.stream_bytes = stream_bytes;
        report.chunks = chunks;
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
    // FixedVec: the network's shapes, plus the extra ones for hand-written code.
    let vec_spec = match (weights, &opts.vec_extra) {
        (None, None) => None,
        (weights, extra) => {
            let mut spec = match weights {
                Some(w) => codegen::vec_spec(&codegen::Names::new(w, opts.int8), &opts.prefix),
                None => VecSpec::new(&format!("{}.FixedVec", opts.prefix)),
            };
            if let Some(extra) = extra {
                spec.merge(extra);
            }
            Some(spec)
        }
    };
    if let Some(spec) = vec_spec {
        report.vec_flags = Some(spec.flags());
        if opts.fixed_vec {
            emit("FixedVec", spec.generate_bundled())?;
        }
    }
    if let Some(book) = book {
        let (text, content, test, stats) =
            super::book::book_modules(book, &opts.prefix, opts.chunk, opts.seed, opts.lookups);
        emit("Book", text)?;
        emit("BookContent", content)?;
        emit("BookTest", test)?;
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
    int8: bool,
) -> (String, usize, usize, Vec<String>) {
    // Pieces cut at a multiple of 4 characters decode on their own (int8 arrays are decoded
    // piece by piece).
    assert!(
        chunk % 4 == 0,
        "literal pieces of a multiple of 4 characters"
    );
    let arrays = weights.quantised(int8);
    let bytes: usize = arrays.iter().map(|a| a.bytes.len()).sum();
    let mut s = String::with_capacity(bytes * 4 / 3 + 4096);
    // Not -O0: at -O0 every literal is an updatable CAF whose unpacked characters stay
    // alive while the network is built; at -O1 and above they are re-unpacked on demand.
    banner(&mut s, &format!("{prefix}.Weights"), "(Array (..), arrays)");
    bundle_imports(&mut s, &[]);
    bundle_body_begin(
        &mut s,
        "Array (arrayName, arrayRows, arrayCols, arrayScale, arrayText), arrays, and per array \
         weights_<array>_text with its literals weights_<array>_<k>",
    );
    // Four base64 characters per three bytes.
    data_warning(
        &mut s,
        &format!(
            "the network's {} weights ({})",
            weights.parameters(),
            if int8 { "int8" } else { "int16" }
        ),
        bytes * 4 / 3,
    );
    s.push_str(
        "-- | One weight array: rows * cols little-endian Int16 values, each worth q * scale, or in\n\
         -- an int8 export each row an exponent byte e then its signed bytes, each worth\n\
         -- q * scale * 2^(-e/8); matrices one row per output, as base64 text (standard alphabet,\n\
         -- padded) split into pieces of a multiple of four characters.\n\
         data Array = Array\n\
         \x20 { arrayName :: String -- ^ the array's name, checked by the decoder\n\
         \x20 , arrayRows :: Int -- ^ rows: outputs of a matrix, ids of a table, 1 for a bias\n\
         \x20 , arrayCols :: Int -- ^ values per row\n\
         \x20 , arrayScale :: Float -- ^ what one Int16 step is worth\n\
         \x20 , arrayText :: [String] -- ^ the base64 text, in pieces\n\
         \x20 }\n\n",
    );
    let mut literals = 0;
    let _ = writeln!(
        s,
        "-- | Every weight array, in the order the generated decoder in Net expects."
    );
    let _ = writeln!(s, "arrays :: [Array]");
    let _ = writeln!(s, "arrays =");
    for (i, a) in arrays.iter().enumerate() {
        let _ = writeln!(
            s,
            "  {} Array {:?} {} {} {} weights_{}_text",
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
            .map(|i| format!("weights_{}_{i}", a.name))
            .collect();
        let _ = writeln!(s, "weights_{}_text :: [String]", a.name);
        let _ = writeln!(s, "weights_{}_text = [{}]\n", a.name, names.join(", "));
        for (name, text) in names.iter().zip(&pieces) {
            let _ = writeln!(s, "{name} :: String");
            let _ = writeln!(s, "{name} =");
            write_string_literal(&mut s, text, LITERAL_WIDTH);
            s.push('\n');
        }
    }
    bundle_body_end(&mut s);
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

pub(super) fn banner(out: &mut String, module: &str, exports: &str) {
    let _ = writeln!(out, "{{- FOURMOLU_DISABLE -}}");
    let _ = writeln!(out, "{{-# LANGUAGE Safe #-}}");
    let _ = writeln!(
        out,
        "-- Generated by `{} export`. Do not edit: regenerate instead.",
        env!("CARGO_PKG_NAME")
    );
    let exports = if exports.is_empty() {
        String::new()
    } else {
        format!(" {exports}")
    };
    let _ = writeln!(out, "module {module}{exports} where\n");
}

// Markers around the sections of a module that the Haskell project's bundler concatenates
// into the single contest file: the imports from base (hoisted into the file's import list)
// and the top-level declarations. Everything outside them (pragmas, module header, imports of
// other bundled modules) is dropped, so every bundled body shares one namespace.
pub(super) const IMPORTS_BEGIN: &str = "{- BUNDLE IMPORTS BEGIN -}";
pub(super) const IMPORTS_END: &str = "{- BUNDLE IMPORTS END -}";
pub(super) const BODY_BEGIN: &str = "{- BUNDLE BODY BEGIN -}";
pub(super) const BODY_END: &str = "{- BUNDLE BODY END -}";

/// A module's IMPORTS section: its imports from base, one per line with explicit lists.
pub(super) fn bundle_imports(out: &mut String, imports: &[&str]) {
    let _ = writeln!(out, "{IMPORTS_BEGIN}");
    for import in imports {
        let _ = writeln!(out, "{import}");
    }
    let _ = writeln!(out, "{IMPORTS_END}\n");
}

/// Opens a module's BODY section, after a comment listing the top-level names it defines (in
/// the bundle they share one namespace with every other body and the engine).
pub(super) fn bundle_body_begin(out: &mut String, names: &str) {
    for line in wrap(&format!("Top-level names: {names}."), 96) {
        let _ = writeln!(out, "-- {line}");
    }
    let _ = writeln!(out, "{BODY_BEGIN}");
}

/// A warning opening the body of a module that is mostly data: what the data is and roughly
/// how much text it takes, so that a reader knows to skip it rather than read it.
pub(super) fn data_warning(out: &mut String, what: &str, text_bytes: usize) {
    let warning = format!(
        "DATA BLOB WARNING: this module is mostly {what} as base64 string literals, about {} \
         of text. Read the declarations up to the first literal and skip the rest; the code \
         that decodes the data is in other modules.",
        rough_size(text_bytes)
    );
    for line in wrap(&warning, 96) {
        let _ = writeln!(out, "-- {line}");
    }
    out.push('\n');
}

/// A size to one or two significant digits: `2.6 MB`, `140 KB`.
fn rough_size(bytes: usize) -> String {
    let b = bytes as f64;
    if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} KB", b / 1e3)
    } else {
        format!("{bytes} bytes")
    }
}

pub(super) fn bundle_body_end(out: &mut String) {
    if !out.ends_with('\n') {
        out.push('\n');
    }
    let _ = writeln!(out, "{BODY_END}");
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
    let c = weights;
    let mut s = String::new();
    banner(&mut s, &format!("{prefix}.Spec"), "");
    bundle_imports(&mut s, &[]);
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
                ("nLayers", c.trunk.len()),
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
    let mut top_level: Vec<&str> = groups
        .iter()
        .flat_map(|(_, items)| items.iter().map(|(n, _)| *n))
        .collect();
    top_level.extend(["bigLines", "openCodesChecksum", "parameters", "foldedMacs"]);
    bundle_body_begin(&mut s, &top_level.join(", "));
    for (comment, items) in &groups {
        for (i, line) in wrap(comment, 96).iter().enumerate() {
            let _ = writeln!(s, "-- {}{line}", if i == 0 { "| " } else { "" });
        }
        let names: Vec<&str> = items.iter().map(|(n, _)| *n).collect();
        let _ = writeln!(s, "{} :: Int", names.join(", "));
        for (name, value) in items {
            let _ = writeln!(s, "{name} = {value}");
        }
        s.push('\n');
    }
    let _ = writeln!(s, "-- | The big-board lines, in feature order.");
    let _ = writeln!(s, "bigLines :: [[Int]]");
    let _ = writeln!(
        s,
        "bigLines = {}",
        hs_list(LINES.iter().map(|l| hs_list(l.iter())))
    );
    s.push('\n');
    let _ = writeln!(
        s,
        "-- | Checksum of the open raw codes in increasing order: h <- (h * 1000003 + code) \
         `mod` 2^32, from 0."
    );
    let _ = writeln!(s, "openCodesChecksum :: Int");
    let _ = writeln!(s, "openCodesChecksum = {}", open_codes_checksum());
    s.push('\n');
    let _ = writeln!(
        s,
        "-- | Trained parameters and multiply-adds per evaluation once folded."
    );
    let _ = writeln!(s, "parameters, foldedMacs :: Int");
    let _ = writeln!(s, "parameters = {}", weights.parameters());
    let _ = writeln!(s, "foldedMacs = {}", weights.folded_macs());
    bundle_body_end(&mut s);
    s
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
            state.apply(legal[rng.random_range(0..legal.len())].into());
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
        "-- | A position and what the shipped (int16-rounded) network computes for it, from the\n\
         -- side to move's view.\n\
         data Ref = Ref\n\
         \x20 { refName :: String -- ^ where the position comes from, e.g. game17-ply2\n\
         \x20 , refCells :: [Int] -- ^ 81 cells by board * 9 + square: 0 empty, 1 X, 2 O\n\
         \x20 , refToMove :: Int -- ^ 0 X, 1 O\n\
         \x20 , refLastMove :: Int -- ^ board * 9 + square of the last move, -1 if none\n\
         \x20 , refForced :: Int -- ^ forced board, 9 if free\n\
         \x20 , refRawCodes :: [Int] -- ^ the nine boards' raw content codes\n\
         \x20 , refIds :: [Int] -- ^ the idsPerPosition input ids\n\
         \x20 , refAcc :: [Float] -- ^ accumulator after the ReLU\n\
         \x20 , refHidden :: [Float] -- ^ last hidden layer\n\
         \x20 , refWdlLogits :: [Float] -- ^ win, draw, loss logits\n\
         \x20 , refWdl :: [Float] -- ^ softmax of the logits: win, draw, loss\n\
         \x20 , refPolicy :: [(Int, Float, Float)] -- ^ legal move index, logit, probability\n\
         \x20 }\n\n",
    );
    let _ = writeln!(
        s,
        "-- | The reference positions, the initial position first."
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
