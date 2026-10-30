//! Generated Haskell code for the network: the decoded weights and the forward pass over the
//! fixed-width vector family, plus a self-test. The literal decoder is the Haskell project's
//! hand-written `Decode` module (`decodeRowsWith`, `decodeFloats`, `decodeBytes`, ...).
//!
//! The vector types come from `<prefix>.FixedVec`, written by the same export from
//! [`vec_spec`] through [`super::vecgen`] (strict records `VecN`, wide ones nested as chunks
//! of `Vec32`, row containers `RowsM_N`, `dotN`, `selM_N`, `linearM_N`/`denseM_N`/
//! `residualM_M`, `concatK_N`, ...); the generated modules import it and alias the role
//! widths (`type VecA = Vec512`). The layout follows the Haskell project's vector codegen notes: the fold
//! runs in 64-row pieces, the feature table is split by row range into four tables indexed
//! by local ids, the policy head is grouped per board, and sums are folds of binary adds.

use std::fmt::Write as _;

use super::haskell::{bundle_body_begin, bundle_body_end, bundle_imports};
use super::vecgen::SPARSE_K;
use super::vecgen::VecSpec;
use super::weights::{Layer, Weights};
use crate::game::LINES;
use crate::nn::model::MOVES;
use crate::nn::nnue::{
    BOARDS,
    CELL_BASE,
    CELL_PAD,
    CELLS,
    CONTENTS,
    DECIDED_BASE,
    FEATURE_IDS,
    FEATURE_ROWS,
    FORCED_BASE,
    FULL,
    GATES,
    IDS_PER_BOARD,
    LINE_BASE,
    LINE_STATES,
    NO_FORCED,
    OPEN_CONTENTS,
    OPEN_MARKS_BASE,
    TOKENS,
    WON_BY_ME,
    WON_BY_THEM,
};

/// Inputs the accumulator sums: the bias, the folded rows of the board tokens and the feature
/// rows.
const ACC_INPUTS: usize = 1 + TOKENS + FEATURE_IDS;

/// Width of the fold's row pieces, and the widest vector kept flat.
pub const CHUNK: usize = 64;

/// Width of the chunks the wider vectors (the accumulator, and the hidden layers when wider
/// than [`CHUNK`]) are nested in. Every chunk function builds one result with all its lanes
/// live, and GHC's native code generator spills most of 64 (round 4 of the Haskell project's
/// exporter notes: 15% fewer cycles per search with 32 than with 64; 16 is 3% faster again
/// but costs a third more compile time, and stays reachable with `--nest`).
pub const NEST: usize = 32;

/// Rows of the four feature tables, in row order: forced board, the eight lines, decided
/// boards, marks on open boards.
const FORCED_FEATURES: usize = LINE_BASE - FORCED_BASE;
const DECIDED_FEATURES: usize = OPEN_MARKS_BASE - DECIDED_BASE;
const MARKS_FEATURES: usize = FEATURE_ROWS - OPEN_MARKS_BASE;

/// The kind of a hidden layer (see [`Layer`]), with a ReGLU block's inner width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Dense,
    Linear,
    Residual,
    ReluResidual,
    SelfGated,
    ReGlu(usize),
}

impl Kind {
    fn of(layer: &Layer) -> Self {
        match layer {
            Layer::Dense(_) => Kind::Dense,
            Layer::Linear(_) => Kind::Linear,
            Layer::Residual(_) => Kind::Residual,
            Layer::ReluResidual(_) => Kind::ReluResidual,
            Layer::SelfGated { .. } => Kind::SelfGated,
            Layer::ReGlu { gate, .. } => Kind::ReGlu(gate.weight.rows),
        }
    }

    /// Whether the layer is computed by columns over its nonzero inputs in sparseHiddenOf:
    /// the layers from the accumulator, and the residual ones, whose inputs are rectified.
    fn sparse(self) -> bool {
        matches!(self, Kind::Dense | Kind::Linear | Kind::Residual)
    }
}

/// The widths and layers a model's code is generated with.
pub struct Names {
    pub e: usize,
    pub a: usize,
    pub h: usize,
    /// The hidden layers, the first from the accumulator.
    pub trunk: Vec<Kind>,
    /// Whether the heads are gated.
    pub gated: bool,
    /// Whether the weights ship as int8 (see [`super::quant::quantise_int8`]).
    pub int8: bool,
}

impl Names {
    pub fn new(w: &Weights, int8: bool) -> Self {
        assert!(
            w.d_acc % CHUNK == 0,
            "d_acc must be a multiple of {CHUNK} for the chunked export"
        );
        let trunk: Vec<Kind> = w.trunk.iter().map(Kind::of).collect();
        let mut widths = vec![("d_hidden", w.d_hidden)];
        widths.extend(w.d_ff().map(|d| ("the ReGLU width", d)));
        for (what, n) in widths {
            assert!(
                n <= CHUNK || n % NEST == 0,
                "{what} must be at most {CHUNK} or a multiple of {NEST}"
            );
        }
        Self {
            e: w.d_embed,
            a: w.d_acc,
            h: w.d_hidden,
            trunk,
            gated: w.gates.is_some(),
            int8,
        }
    }

    /// The ReGLU blocks' inner widths.
    fn ff_widths(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .trunk
            .iter()
            .filter_map(|k| match k {
                Kind::ReGlu(d) => Some(*d),
                _ => None,
            })
            .collect();
        out.dedup();
        out
    }

    fn vec(w: usize) -> String {
        format!("Vec{w}")
    }

    fn rows(m: usize, n: usize) -> String {
        format!("Rows{m}_{n}")
    }

    /// Chunks of the accumulator.
    fn pieces(&self) -> usize {
        self.a / CHUNK
    }

    /// `(m, n)` dense shapes: the fold pieces, the hidden layers, the value head, the skip
    /// and the per-board policy (and policy gates).
    fn dense(&self) -> Vec<(usize, usize)> {
        let h = self.h;
        let mut shapes = vec![(CHUNK, self.e), (h, self.a)];
        if self.trunk[1..].iter().any(|k| !matches!(k, Kind::ReGlu(_))) {
            shapes.push((h, h));
        }
        for d in self.ff_widths() {
            shapes.extend([(d, h), (h, d)]);
        }
        shapes.extend([(3, h), (3, self.a), (CELLS, h)]);
        // Shapes can coincide (a ReGLU width equal to the hidden one, a 9-wide layer).
        let mut seen = std::collections::BTreeSet::new();
        shapes.retain(|&shape| seen.insert(shape));
        shapes
    }

    /// Row containers read by index: the feature tables.
    fn selected(&self) -> Vec<(usize, usize)> {
        vec![
            (FORCED_FEATURES, self.a),
            (LINE_STATES, self.a),
            (MARKS_FEATURES, self.a),
        ]
    }
}

/// The `FixedVec` shapes the configuration's generated code needs, as module
/// `<prefix>.FixedVec`.
pub fn vec_spec(names: &Names, prefix: &str) -> VecSpec {
    let mut spec = VecSpec::new(&format!("{prefix}.FixedVec"));
    spec.floats
        .extend([3, CELLS, CHUNK, names.e, names.a, names.h]);
    spec.floats.extend(names.ff_widths());
    spec.nest.insert(names.a, (names.a / NEST, NEST));
    for n in std::iter::once(names.h).chain(names.ff_widths()) {
        if n > CHUNK {
            spec.nest.insert(n, (n / NEST, NEST));
        }
    }
    spec.rows = names.selected();
    spec.dense = names.dense();
    spec.sums = vec![(ACC_INPUTS, names.a)];
    // The hidden layers by columns over their nonzero inputs (sparseHiddenOf).
    spec.sparse = vec![(names.h, names.a, SPARSE_K)];
    if names.trunk.contains(&Kind::Residual) {
        spec.sparse.push((names.h, names.h, SPARSE_K));
    }
    spec.concats = vec![(names.pieces(), CHUNK)];
    spec
}

fn header(out: &mut String, module: &str, exports: &str, opts: &str) {
    let _ = writeln!(out, "{{- FOURMOLU_DISABLE -}}");
    let _ = writeln!(out, "{{-# LANGUAGE Safe #-}}");
    if !opts.is_empty() {
        let _ = writeln!(out, "{{-# OPTIONS_GHC {opts} #-}}");
    }
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

fn vars(prefix: &str, n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("{prefix}{}", super::vecgen::pad_index(i, n)))
        .collect()
}

/// `Net.hs`: the decoded network and its forward pass.
pub fn net_module(names: &Names, prefix: &str, manifest_names: &[String]) -> String {
    let (e, a, h) = (names.e, names.a, names.h);
    let k = names.pieces();
    let mut s = String::new();
    header(&mut s, &format!("{prefix}.Net"), "", "-O2 -fobject-code");
    let _ = writeln!(
        s,
        "-- {prefix}.FixedVec is written by the same export; without a model, the same module is\n\
         -- `export --out <dir> -- {}`.\n",
        vec_spec(names, prefix).flags()
    );
    let _ = writeln!(
        s,
        "import Decode (chunksOf, {})",
        if names.int8 {
            "decodeBytes"
        } else {
            "decodeFloats, decodeRowsWith"
        }
    );
    let _ = writeln!(s, "import {prefix}.FixedVec");
    let _ = writeln!(s, "import {prefix}.Weights (Array (..), arrays)");
    let base_imports: &[&str] = match names.int8 {
        true => &["import Data.Int (Int8)"],
        false => &[],
    };
    bundle_imports(&mut s, base_imports);
    bundle_body_begin(
        &mut s,
        "VecE, VecA, VecH, FoldPiece, OpenClosed (ocOpen, ocFull, ocWonByUs, ocWonByThem), \
         Net (every field starts with net), net, genZeroChunk, genZero3, \
         foldRow, lineFeature, accumulateRows, genZeroH, hiddenOf, HiddenColumns (constructor \
         HiddenColumns), hiddenColumns, sparseHiddenOf, genSigmoid or genOnes9, wdlOf, \
         PolicyGates (pgLocal, pgSend), policyGates, policyLogits",
    );
    let _ = writeln!(
        s,
        "-- | A content-table row and a board token (width dEmbed)."
    );
    let _ = writeln!(s, "type VecE = {}", Names::vec(e));
    let _ = writeln!(
        s,
        "-- | The accumulator and a folded first-layer row (width dAcc)."
    );
    let _ = writeln!(s, "type VecA = {}", Names::vec(a));
    let _ = writeln!(s, "-- | A hidden layer's output (width dHidden).");
    let _ = writeln!(s, "type VecH = {}", Names::vec(h));
    let _ = writeln!(
        s,
        "-- | {CHUNK} rows of the fold: a piece of a board's first-layer slice."
    );
    let _ = writeln!(s, "type FoldPiece = {}\n", Names::rows(CHUNK, e));
    let _ = writeln!(
        s,
        r#"-- | A per-content table: the open contents (ids 0..{last_open}, raw-code order) and the three
-- decided kinds (ids {FULL} fullId, {WON_BY_ME} wonByMe, {WON_BY_THEM} wonByThem).
data OpenClosed a = OpenClosed
  {{ ocOpen :: [a] -- ^ the open contents, by compact id
  , ocFull :: !a -- ^ a full board without a line
  , ocWonByUs :: !a -- ^ a board won by the side to move
  , ocWonByThem :: !a -- ^ a board won by the opponent
  }}
  deriving (Functor)
"#,
        last_open = OPEN_CONTENTS - 1
    );

    // The record.
    let content_ids = format!("content rows, ids 0..{}", CONTENTS - 1);
    let no_forced_id =
        format!("the forced token's content row when the move is free, id {NO_FORCED} (noForced)");
    let cell_ids = format!(
        "{} cell rows, ids {CELL_BASE}..{}, at 3 * cell + v (v: 0 empty, 1 mine, 2 theirs)",
        CELL_PAD - CELL_BASE,
        CELL_PAD - 1
    );
    let pad_id = format!(
        "the cell padding row, id {CELL_PAD} (cellPad): a decided board's or no-forced token adds it nine times instead of cell rows"
    );
    let _ = writeln!(
        s,
        "-- | The weights, laid out for the evaluation: first layer, features, hidden, heads."
    );
    let _ = writeln!(s, "data Net = Net");
    let mut fields: Vec<(String, String, String)> = vec![
        ("netContent".into(), "(OpenClosed VecE)".into(), content_ids),
        ("netNoForced".into(), "VecE".into(), no_forced_id),
        ("netCells".into(), "[VecE]".into(), cell_ids),
        ("netPad".into(), "VecE".into(), pad_id),
        (
            "netMix".into(),
            "[[FoldPiece]]".into(),
            "board k's first-layer slice, transposed, in pieces of 64 rows".into(),
        ),
        (
            "netMixBias".into(),
            "VecA".into(),
            "the first layer's bias".into(),
        ),
        (
            "netForced".into(),
            "[FoldPiece]".into(),
            "the forced board's slice".into(),
        ),
        (
            "netForcedFeature".into(),
            Names::rows(FORCED_FEATURES, a),
            "feature rows by forced board (9 = free)".into(),
        ),
    ];
    for l in 0..LINES.len() {
        fields.push((
            format!("netLineFeature{l}"),
            Names::rows(LINE_STATES, a),
            "feature rows of a big-board line by 16 a + 4 b + c (0 open, 1 mine, 2 theirs, 3 full)"
                .into(),
        ));
    }
    fields.push((
        "netDecidedFeature".into(),
        Names::rows(DECIDED_FEATURES, a),
        "feature rows by number of decided boards".into(),
    ));
    fields.push((
        "netMarksFeature".into(),
        Names::rows(MARKS_FEATURES, a),
        "feature rows by marks on open boards".into(),
    ));
    for (i, kind) in names.trunk.iter().enumerate() {
        let field =
            |name: &str, ty: String, what: String| (format!("netHidden{i}{name}"), ty, what);
        let plain = |what: &str| {
            let shape = if i == 0 {
                Names::rows(h, a)
            } else {
                Names::rows(h, h)
            };
            [
                field("", shape, format!("hidden layer {i}, {what}")),
                field("Bias", "VecH".into(), format!("hidden layer {i}'s bias")),
            ]
        };
        match *kind {
            Kind::Dense => fields.extend(plain("relu (W acc + b)")),
            Kind::Linear => fields.extend(plain("W acc + b, no relu")),
            Kind::Residual => fields.extend(plain("residual: relu (x + W x + b)")),
            Kind::ReluResidual => fields.extend(plain("relu (x + W (relu x) + b)")),
            Kind::SelfGated => fields.extend([
                field(
                    "Up",
                    Names::rows(h, h),
                    format!("hidden layer {i}, x + Down (relu x * (Up x + b)): Up"),
                ),
                field(
                    "UpBias",
                    "VecH".into(),
                    format!("hidden layer {i}'s Up bias"),
                ),
                field(
                    "Down",
                    Names::rows(h, h),
                    format!("hidden layer {i}'s Down, no bias"),
                ),
            ]),
            Kind::ReGlu(d) => fields.extend([
                field(
                    "Gate",
                    Names::rows(d, h),
                    format!("hidden layer {i}, x + Down (relu (Gate x + b) * (Up x + b)): Gate"),
                ),
                field(
                    "GateBias",
                    Names::vec(d),
                    format!("hidden layer {i}'s Gate bias"),
                ),
                field("Up", Names::rows(d, h), format!("hidden layer {i}'s Up")),
                field(
                    "UpBias",
                    Names::vec(d),
                    format!("hidden layer {i}'s Up bias"),
                ),
                field(
                    "Down",
                    Names::rows(h, d),
                    format!("hidden layer {i}'s Down, no bias"),
                ),
            ]),
        }
    }
    fields.extend([
        (
            "netValue".into(),
            Names::rows(3, h),
            "win/draw/loss logits from the last hidden layer".into(),
        ),
        (
            "netValueBias".into(),
            Names::vec(3),
            "win/draw/loss bias".into(),
        ),
        (
            "netSkip".into(),
            Names::rows(3, a),
            "win/draw/loss logits straight from the accumulator".into(),
        ),
    ]);
    for b in 0..BOARDS {
        fields.push((
            format!("netPolicy{b}"),
            Names::rows(CELLS, h),
            format!("global policy rows of board {b}'s nine squares"),
        ));
        fields.push((
            format!("netPolicyBias{b}"),
            Names::vec(CELLS),
            format!("global policy bias of board {b}'s nine squares"),
        ));
    }
    fields.extend([
        (
            "netLocal".into(),
            format!("(OpenClosed {})", Names::vec(CELLS)),
            "each content's own cell logits".into(),
        ),
        (
            "netSend".into(),
            "(OpenClosed Float)".into(),
            "each content's send logit".into(),
        ),
    ]);
    if names.gated {
        fields.extend([
            (
                "netValueGate".into(),
                "VecH".into(),
                "the value's mix gate: sigmoid (g . h + b) of the skip, the rest the value's own"
                    .into(),
            ),
            (
                "netValueGateBias".into(),
                "Float".into(),
                "the value's mix gate's bias".into(),
            ),
            (
                "netLocalGates".into(),
                Names::rows(CELLS, h),
                "per board b, the gate of its own cell logits".into(),
            ),
            (
                "netLocalGatesBias".into(),
                Names::vec(CELLS),
                "the local gates' bias".into(),
            ),
            (
                "netSendGates".into(),
                Names::rows(CELLS, h),
                "per board s, the gate of the logit of sending to it".into(),
            ),
            (
                "netSendGatesBias".into(),
                Names::vec(CELLS),
                "the send gates' bias".into(),
            ),
        ]);
    }
    for (i, (name, ty, comment)) in fields.iter().enumerate() {
        let sep = if i == 0 { "{" } else { "," };
        let _ = writeln!(s, "  {sep} {name} :: !{ty} -- ^ {comment}");
    }
    let _ = writeln!(s, "  }}\n");

    // Decoding: every array from its own text, after checking names and shapes.
    let position = |name: &str| {
        manifest_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("{name} array"))
    };
    // Array `i` as `m` rows of width `n` (a list), or as its `m * n` values.
    let rows_at = |m: usize, n: usize, i: usize| match names.int8 {
        true => format!("int8Rows {m} {n} fromListRest{n} a{i}"),
        false => format!("rows {m} {n} fromListRest{n} a{i}"),
    };
    let floats_at = |m: usize, n: usize, i: usize| match names.int8 {
        true => format!("int8Values {m} {n} a{i}"),
        false => format!("floats {m} {n} a{i}"),
    };
    let _ = writeln!(
        s,
        "-- | The network, decoded from the weight arrays on first use."
    );
    let _ = writeln!(s, "net :: Net");
    let _ = writeln!(s, "net =");
    let _ = writeln!(
        s,
        "  let shaped r c a = if arrayRows a == r && arrayCols a == c then a else error (\"weights: \" ++ arrayName a ++ \" is not \" ++ show r ++ \"x\" ++ show c)"
    );
    if names.int8 {
        let _ = writeln!(
            s,
            "      -- Each row an exponent byte e, then its values as signed bytes, each worth the\n\
             \x20     -- array's scale times 2^(-e/8). The text's pieces are whole groups of four characters,\n\
             \x20     -- so they decode one by one.\n\
             \x20     int8Values r c a =\n\
             \x20       let a' = shaped r c a\n\
             \x20           bad what = error (\"weights: \" ++ arrayName a' ++ \" has \" ++ what ++ \" values than rows * cols\")\n\
             \x20           go 0 _ _ [] = []\n\
             \x20           go 0 _ _ _ = bad \"more\"\n\
             \x20           go n 0 _ (e : rest) = go n c (arrayScale a' * 2 ** negate (fromIntegral e / 8)) rest\n\
             \x20           go n k sc (w : rest) =\n\
             \x20             sc * fromIntegral (fromIntegral w :: Int8) : (if k == 1 then go (n - 1) 0 sc rest else go n (k - 1) sc rest)\n\
             \x20           go _ _ _ [] = bad \"fewer\"\n\
             \x20        in go r 0 0 (concatMap decodeBytes (arrayText a'))\n\
             \x20     int8Rows r c split a =\n\
             \x20       let go 0 _ = []\n\
             \x20           go n xs = case split xs of (v, rest) -> v `seq` (v : go (n - 1 :: Int) rest)\n\
             \x20        in go r (int8Values r c a)"
        );
    } else {
        let _ = writeln!(
            s,
            "      rows r c split a = let a' = shaped r c a in decodeRowsWith (arrayName a') r c (arrayScale a') split (arrayText a')"
        );
        let _ = writeln!(
            s,
            "      floats r c a = let a' = shaped r c a in decodeFloats (arrayName a') (r * c) (arrayScale a') (arrayText a')"
        );
    }
    let _ = writeln!(
        s,
        "      table name open decided = case decided of\n\
         \x20       [full, me, them] -> OpenClosed open full me them\n\
         \x20       _ -> error (\"net: array \" ++ name ++ \"_decided is not 3 rows\")"
    );
    let _ = writeln!(s, "      expected = {}", hs_strings(manifest_names));
    let _ = writeln!(s, "   in case arrays of");
    let vars_: Vec<String> = (0..manifest_names.len()).map(|i| format!("a{i}")).collect();
    let _ = writeln!(
        s,
        "        [{}] | map arrayName arrays == expected ->",
        vars_.join(", ")
    );
    // The policy rows are grouped per board.
    let (policy_index, policy_bias) = (position("policy"), position("policy_bias"));
    let _ = writeln!(
        s,
        "          let policyTables = map fromListRows{CELLS}_{h} (chunksOf {CELLS} ({}))",
        rows_at(MOVES, h, policy_index)
    );
    let _ = writeln!(
        s,
        "              policyBiases = map fromList{CELLS} (chunksOf {CELLS} ({}))",
        floats_at(1, MOVES, policy_bias)
    );
    if names.gated {
        let (gates, gates_bias) = (position("gates"), position("gates_bias"));
        let _ = writeln!(
            s,
            "              (valueGate, localGates, sendGates) = case {} of\n\
             \x20               v : rest | (l, r) <- splitAt {CELLS} rest -> (v, l, r)\n\
             \x20               _ -> error \"net: array gates is not {GATES} rows\"\n\
             \x20             (valueGateBias, localGatesBias, sendGatesBias) = case {} of\n\
             \x20               v : rest | (l, r) <- splitAt {CELLS} rest -> (v, l, r)\n\
             \x20               _ -> error \"net: array gates_bias is not {GATES} values\"",
            rows_at(GATES, h, gates),
            floats_at(1, GATES, gates_bias)
        );
    }
    let _ = writeln!(s, "           in Net");
    // Every array decodes straight into its field, in order: `step` counts the arrays taken.
    let step = std::cell::Cell::new(0usize);
    let next = || step.replace(step.get() + 1);
    // The next array's `m` rows of width `n` as a list.
    let list = |m: usize, n: usize| rows_at(m, n, next());
    let matrix = |m: usize, n: usize| format!("fromListRows{m}_{n} ({})", list(m, n));
    let vector = |n: usize| format!("fromList{n} ({})", floats_at(1, n, next()));
    let mut inits = Vec::new();
    let open = list(OPEN_CONTENTS, e);
    let decided = list(CONTENTS - OPEN_CONTENTS, e);
    inits.push(format!(
        "netContent = table \"content\" ({open}) ({decided})"
    ));
    inits.push(format!("netNoForced = {}", vector(e)));
    inits.push(format!("netCells = {}", list(CELL_PAD - CELL_BASE, e)));
    inits.push(format!("netPad = {}", vector(e)));
    // The first layer's slices in pieces of CHUNK rows.
    let pieces = || {
        let p: Vec<String> = (0..k).map(|_| matrix(CHUNK, e)).collect();
        format!("[{}]", p.join(", "))
    };
    let mix: Vec<String> = (0..BOARDS).map(|_| pieces()).collect();
    inits.push(format!(
        "netMix =\n                    [ {}\n                    ]",
        mix.join("\n                    , ")
    ));
    inits.push(format!("netMixBias = {}", vector(a)));
    inits.push(format!("netForced = {}", pieces()));
    inits.push(format!("netForcedFeature = {}", matrix(FORCED_FEATURES, a)));
    for l in 0..LINES.len() {
        inits.push(format!("netLineFeature{l} = {}", matrix(LINE_STATES, a)));
    }
    inits.push(format!(
        "netDecidedFeature = {}",
        matrix(DECIDED_FEATURES, a)
    ));
    inits.push(format!("netMarksFeature = {}", matrix(MARKS_FEATURES, a)));
    for (i, kind) in names.trunk.iter().enumerate() {
        let d_in = if i == 0 { a } else { h };
        let mut parts = Vec::new();
        match *kind {
            Kind::SelfGated => parts.extend([("Up", h, h, true), ("Down", h, h, false)]),
            Kind::ReGlu(d) => parts.extend([
                ("Gate", d, h, true),
                ("Up", d, h, true),
                ("Down", h, d, false),
            ]),
            _ => parts.push(("", h, d_in, true)),
        }
        for (name, m, n, bias) in parts {
            inits.push(format!("netHidden{i}{name} = {}", matrix(m, n)));
            if bias {
                inits.push(format!("netHidden{i}{name}Bias = {}", vector(m)));
            }
        }
    }
    inits.push(format!("netValue = {}", matrix(3, h)));
    inits.push(format!("netValueBias = {}", vector(3)));
    inits.push(format!("netSkip = {}", matrix(3, a)));
    // The policy arrays, decoded above.
    assert_eq!(step.get(), policy_index, "policy array position");
    let _ = list(MOVES, h);
    assert_eq!(next(), policy_bias, "policy bias array position");
    for b in 0..BOARDS {
        inits.push(format!("netPolicy{b} = policyTables !! {b}"));
        inits.push(format!("netPolicyBias{b} = policyBiases !! {b}"));
    }
    let open = list(OPEN_CONTENTS, CELLS);
    let decided = list(CONTENTS - OPEN_CONTENTS, CELLS);
    inits.push(format!("netLocal = table \"local\" ({open}) ({decided})"));
    let open = floats_at(1, OPEN_CONTENTS, next());
    let decided = floats_at(1, CONTENTS - OPEN_CONTENTS, next());
    inits.push(format!("netSend = table \"send\" ({open}) ({decided})"));
    if names.gated {
        // One row (and bias) for the value, then nine per board for each policy table,
        // decoded above.
        assert_eq!(step.get(), position("gates"), "gates array position");
        let _ = list(GATES, h);
        assert_eq!(next(), position("gates_bias"), "gates bias array position");
        inits.extend([
            "netValueGate = valueGate".to_string(),
            "netValueGateBias = valueGateBias".to_string(),
            format!("netLocalGates = fromListRows{CELLS}_{h} localGates"),
            format!("netLocalGatesBias = fromList{CELLS} localGatesBias"),
            format!("netSendGates = fromListRows{CELLS}_{h} sendGates"),
            format!("netSendGatesBias = fromList{CELLS} sendGatesBias"),
        ]);
    }
    assert_eq!(step.get(), manifest_names.len(), "one decode per array");
    for (i, init) in inits.iter().enumerate() {
        let sep = if i == 0 { "{" } else { "," };
        let _ = writeln!(s, "                {sep} {init}");
    }
    let _ = writeln!(s, "                }}");
    let _ = writeln!(
        s,
        "        _ -> error \"weights: the arrays do not match the generated decoder\"\n"
    );

    // Forward pieces.
    let _ = writeln!(s, "-- | A zero chunk: the bias of the fold's dot products.");
    let _ = writeln!(s, "genZeroChunk :: Vec{CHUNK}");
    let _ = writeln!(s, "genZeroChunk = replicate{CHUNK} 0\n");
    let _ = writeln!(
        s,
        "-- | Three zeros: the bias of the skip connection's logits."
    );
    let _ = writeln!(s, "genZero3 :: Vec3");
    let _ = writeln!(s, "genZero3 = replicate3 0\n");
    let _ = writeln!(
        s,
        "-- | The folded first-layer row of a board token at board k (k = {BOARDS} for the forced\n\
         -- token): what a table of rows per content caches. {k} x {CHUNK} dot products of width {e}."
    );
    let _ = writeln!(s, "foldRow :: Net -> Int -> VecE -> VecA");
    let ps = vars("p", k);
    let parts: Vec<String> = ps
        .iter()
        .map(|p| format!("(linear{CHUNK}_{e} {p} genZeroChunk t)"))
        .collect();
    let _ = writeln!(
        s,
        "foldRow n k t = case (if k == {BOARDS} then netForced n else netMix n !! k) of\n\
         \x20 [{}] -> concat{k}_{CHUNK} {}\n\
         \x20 _ -> error \"foldRow: {k} pieces\"\n",
        ps.join(", "),
        parts.join(" ")
    );
    let _ = writeln!(
        s,
        "-- | The feature rows of big-board line l (0..{}).",
        LINES.len() - 1
    );
    let _ = writeln!(
        s,
        "lineFeature :: Net -> Int -> {}",
        Names::rows(LINE_STATES, a)
    );
    let _ = writeln!(s, "lineFeature n l = case l of");
    for l in 0..LINES.len() {
        let _ = writeln!(s, "  {l} -> netLineFeature{l} n");
    }
    let _ = writeln!(s, "  _ -> error \"lineFeature: line out of range\"\n");
    let _ = writeln!(
        s,
        "-- | The accumulator: the ReLU of the bias, the ten folded rows and the eleven feature\n\
         -- rows (forced board 0..9, the eight lines' status codes 0..63 in bigLines order, decided\n\
         -- boards 0..9, marks on open boards 0..72)."
    );
    let _ = writeln!(
        s,
        "accumulateRows :: Net -> [VecA] -> Int -> [Int] -> Int -> Int -> VecA"
    );
    // One ACC_INPUTS-way sum, in the order of `foldl add` over bias, rows, features (so the
    // result is the same to the bit as that fold's).
    let rows = vars("r", TOKENS);
    let lines = vars("l", LINES.len());
    let mut terms = vec!["(netMixBias n)".to_string()];
    terms.extend(rows.iter().cloned());
    terms.push(format!(
        "(sel{FORCED_FEATURES}_{a} forcedBoard (netForcedFeature n))"
    ));
    terms.extend(
        lines
            .iter()
            .enumerate()
            .map(|(l, code)| format!("(sel{LINE_STATES}_{a} {code} (netLineFeature{l} n))")),
    );
    terms.push(format!(
        "(sel{DECIDED_FEATURES}_{a} decided (netDecidedFeature n))"
    ));
    terms.push(format!(
        "(sel{MARKS_FEATURES}_{a} marks (netMarksFeature n))"
    ));
    assert_eq!(terms.len(), ACC_INPUTS, "accumulator inputs");
    let _ = writeln!(
        s,
        "accumulateRows n rows forcedBoard lineCodes decided marks = case (rows, lineCodes) of\n\
         \x20 ([{}], [{}]) ->",
        rows.join(", "),
        lines.join(", ")
    );
    let sum = super::vecgen::Expr::app(format!("sum{ACC_INPUTS}_{a}"), &terms);
    let acc = super::vecgen::Expr::App(format!("relu{a}"), vec![sum]);
    for line in acc.layout(4) {
        let _ = writeln!(s, "{line}");
    }
    let _ = writeln!(
        s,
        "  _ -> error \"accumulateRows: not {TOKENS} rows and {} line codes\"\n",
        LINES.len()
    );
    let _ = writeln!(
        s,
        "-- | Zeros of the hidden width: the bias of the gated layers' down projections."
    );
    let _ = writeln!(s, "genZeroH :: VecH");
    let _ = writeln!(s, "genZeroH = replicate{h} 0\n");
    // Layer i's output from `x`, the previous one's (the accumulator for layer 0), or with
    // `cols` by columns over the nonzero inputs.
    let layer = |i: usize, kind: Kind, x: &str, cols: Option<&str>| -> String {
        let w = |name: &str| format!("(netHidden{i}{name} n)");
        match (kind, cols) {
            (Kind::Dense, None) => format!("dense{h}_{a} {} {} {x}", w(""), w("Bias")),
            (Kind::Linear, None) => format!("linear{h}_{a} {} {} {x}", w(""), w("Bias")),
            (Kind::Residual, None) => format!("residual{h}_{h} {} {} {x}", w(""), w("Bias")),
            (Kind::Dense, Some(c)) => format!("sparseDense{h}_{a} {c} {} {x}", w("Bias")),
            (Kind::Linear, Some(c)) => format!("sparseLinear{h}_{a} {c} {} {x}", w("Bias")),
            (Kind::Residual, Some(c)) => format!("sparseResidual{h} {c} {} {x}", w("Bias")),
            (Kind::ReluResidual, _) => format!(
                "relu{h} (add{h} {x} (linear{h}_{h} {} {} (relu{h} {x})))",
                w(""),
                w("Bias")
            ),
            (Kind::SelfGated, _) => format!(
                "add{h} {x} (linear{h}_{h} {} genZeroH (combine{h} (*) (relu{h} {x}) (linear{h}_{h} {} {} {x})))",
                w("Down"),
                w("Up"),
                w("UpBias")
            ),
            (Kind::ReGlu(d), _) => format!(
                "add{h} {x} (linear{h}_{d} {} genZeroH (combine{d} (*) (dense{d}_{h} {} {} {x}) (linear{d}_{h} {} {} {x})))",
                w("Down"),
                w("Gate"),
                w("GateBias"),
                w("Up"),
                w("UpBias")
            ),
        }
    };
    // The layers as a let chain, h0 from the accumulator.
    let chain = |cols: &[Option<String>]| -> String {
        let hs = vars("h", names.trunk.len());
        let mut out = String::from("  let\n");
        for (i, kind) in names.trunk.iter().enumerate() {
            let x = if i == 0 { "acc" } else { hs[i - 1].as_str() };
            let _ = writeln!(
                out,
                "    {} = {}",
                hs[i],
                layer(i, *kind, x, cols[i].as_deref())
            );
        }
        let _ = write!(out, "  in {}", hs[hs.len() - 1]);
        out
    };
    let _ = writeln!(s, "-- | The last hidden layer.");
    let _ = writeln!(s, "hiddenOf :: Net -> VecA -> VecH");
    let _ = writeln!(
        s,
        "hiddenOf n acc =\n{}\n",
        chain(&vec![None; names.trunk.len()])
    );
    // The same layers, the sparse ones by columns over their nonzero inputs.
    let sparse: Vec<usize> = (0..names.trunk.len())
        .filter(|&i| names.trunk[i].sparse())
        .collect();
    let d_in = |i: usize| if i == 0 { a } else { h };
    let col_types: Vec<String> = sparse
        .iter()
        .map(|&i| format!("!Cols{}_{h}", d_in(i)))
        .collect();
    let _ = writeln!(
        s,
        "-- | The columns of the layers sparseHiddenOf computes by columns: build them once\n\
         -- (hiddenColumns) and keep them with the other precomputed data.\n\
         data HiddenColumns = HiddenColumns {}\n",
        col_types.join(" ")
    );
    let built: Vec<String> = sparse
        .iter()
        .map(|&i| format!("(columns{h}_{} (netHidden{i} n))", d_in(i)))
        .collect();
    let _ = writeln!(
        s,
        "-- | The columns of those layers, transposed from their rows.\n\
         hiddenColumns :: Net -> HiddenColumns\n\
         hiddenColumns n = HiddenColumns {}\n",
        built.join(" ")
    );
    let cs = vars("c", sparse.len());
    let mut cols = vec![None; names.trunk.len()];
    for (c, &i) in cs.iter().zip(&sparse) {
        cols[i] = Some(c.clone());
    }
    let _ = writeln!(
        s,
        "-- | hiddenOf with the layers from the accumulator (and the residual ones) by columns over\n\
         -- their nonzero inputs (equal to it up to rounding).\n\
         sparseHiddenOf :: Net -> HiddenColumns -> VecA -> VecH\n\
         sparseHiddenOf n (HiddenColumns {}) acc =\n{}\n",
        cs.join(" "),
        chain(&cols)
    );
    let value = format!("linear3_{h} (netValue n) (netValueBias n) h");
    let skip = format!("linear3_{a} (netSkip n) genZero3 acc");
    if names.gated {
        let _ = writeln!(s, "-- | The logistic function.");
        let _ = writeln!(s, "genSigmoid :: Float -> Float");
        let _ = writeln!(s, "genSigmoid x = 1 / (1 + exp (negate x))\n");
        let _ = writeln!(
            s,
            "-- | Win, draw and loss logits: the value's own and the skip's, mixed by a gate of h."
        );
        let _ = writeln!(s, "wdlOf :: Net -> VecA -> VecH -> Vec3");
        let _ = writeln!(
            s,
            "wdlOf n acc h =\n\
             \x20 let v = {value}\n\
             \x20     s = {skip}\n\
             \x20     g = genSigmoid (dot{h} (netValueGate n) h + netValueGateBias n)\n\
             \x20  in add3 v (scale3 g (sub3 s v))\n"
        );
    } else {
        let _ = writeln!(s, "-- | Win, draw and loss logits.");
        let _ = writeln!(s, "wdlOf :: Net -> VecA -> VecH -> Vec3");
        let _ = writeln!(s, "wdlOf n acc h = add3 ({value}) ({skip})\n");
    }
    let _ = writeln!(
        s,
        "-- | The weights of the policy tables in a position: board b's own cell logits (netLocal n\n\
         -- at its content) count index9 b (pgLocal g) times, the send logit of board s (netSend n\n\
         -- at its content, on square s of every board) index9 s (pgSend g) times.\n\
         data PolicyGates = PolicyGates {{pgLocal :: !Vec{CELLS}, pgSend :: !Vec{CELLS}}}\n"
    );
    let _ = writeln!(s, "-- | The policy gates of a last hidden layer.");
    let _ = writeln!(s, "policyGates :: Net -> VecH -> PolicyGates");
    if names.gated {
        let _ = writeln!(
            s,
            "policyGates n h =\n\
             \x20 PolicyGates\n\
             \x20   (map{CELLS} genSigmoid (linear{CELLS}_{h} (netLocalGates n) (netLocalGatesBias n) h))\n\
             \x20   (map{CELLS} genSigmoid (linear{CELLS}_{h} (netSendGates n) (netSendGatesBias n) h))\n"
        );
    } else {
        let _ = writeln!(
            s,
            "policyGates _ _ = PolicyGates genOnes{CELLS} genOnes{CELLS}\n\n\
             -- | Ones: this network's tables are not gated.\n\
             genOnes{CELLS} :: Vec{CELLS}\n\
             genOnes{CELLS} = replicate{CELLS} 1\n"
        );
    }
    let _ = writeln!(
        s,
        "-- | The global part of the nine move logits of board b; the full logits add the\n\
         -- board's own cell logits (netLocal n at content b) and, at square j, the send logit\n\
         -- (netSend n at content j), each scaled by its gate (policyGates)."
    );
    let _ = writeln!(s, "policyLogits :: Net -> VecH -> Int -> Vec{CELLS}");
    let _ = writeln!(s, "policyLogits n h b = case b of");
    for b in 0..BOARDS {
        let _ = writeln!(
            s,
            "  {b} -> linear{CELLS}_{h} (netPolicy{b} n) (netPolicyBias{b} n) h"
        );
    }
    let _ = writeln!(s, "  _ -> error \"policyLogits: board out of range\"");
    bundle_body_end(&mut s);
    s
}

fn hs_strings(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|s| format!("{s:?}")).collect();
    format!("[{}]", quoted.join(", "))
}

/// `SelfTest.hs`: evaluates the reference positions from their ids and compares.
pub fn selftest_module(names: &Names, prefix: &str) -> String {
    let (e, a, h) = (names.e, names.a, names.h);
    let mut s = String::new();
    header(
        &mut s,
        &format!("{prefix}.SelfTest"),
        "(selfTest, evalIds, bench)",
        "-O2 -fobject-code",
    );
    let _ = writeln!(s, "import GHC.Clock (getMonotonicTimeNSec)");
    let _ = writeln!(s, "import {prefix}.FixedVec");
    let _ = writeln!(s, "import {prefix}.Net");
    let _ = writeln!(s, "import {prefix}.Reference");
    let _ = writeln!(s, "import {prefix}.Spec\n");
    let text = format!(
        r#"-- | Entry i of a per-content table, in the training numbering (ids 0..{last_content}).
byId :: OpenClosed a -> Int -> a
byId t i
  | i < openContents = ocOpen t !! i
  | i == fullId = ocFull t
  | i == wonByMe = ocWonByUs t
  | i == wonByThem = ocWonByThem t
  | otherwise = error ("byId: " ++ show i ++ " is not a content id")

-- | Row i of the content array in the training numbering: a content, the no-forced row, a
-- cell row or the padding row.
contentRow :: Net -> Int -> VecE
contentRow n i
  | i < noForced = byId (netContent n) i
  | i == noForced = netNoForced n
  | i < cellPad = netCells n !! (i - cellBase)
  | i == cellPad = netPad n
  | otherwise = error ("contentRow: " ++ show i ++ " is not a content row")

-- | A board token from its {IDS_PER_BOARD} ids, looking the rows up in the content lists
-- (slow: for tests and for building a table, not for the search).
tokenOf :: Net -> [Int] -> VecE
tokenOf n ids = foldl1 add{e} (map (contentRow n) ids)
{{-# NOINLINE tokenOf #-}}

-- | The eleven global feature ids of a position's ids as the local indices the
-- feature tables take: forced board, the eight line codes, decided boards, open marks.
localFeatures :: [Int] -> (Int, [Int], Int, Int)
localFeatures ids =
  case drop (tokens * idsPerBoard) ids of
    (forced : rest) ->
      ( forced - forcedBase
      , zipWith (\l f -> f - lineBase - l * lineStates) [0 ..] (take 8 rest)
      , rest !! 8 - decidedBase
      , rest !! 9 - openMarksBase
      )
    [] -> error "localFeatures: no feature ids"

-- | The ten folded rows of a position's ids.
foldedRows :: Net -> [Int] -> [VecA]
foldedRows n ids =
  let tokenIds = take (tokens * idsPerBoard) ids
      toks = [tokenOf n (take idsPerBoard (drop (k * idsPerBoard) tokenIds)) | k <- [0 .. tokens - 1]]
   in zipWith (foldRow n) [0 ..] toks

-- | The accumulator from a position's folded rows and its feature ids.
accumulateIds :: Net -> [VecA] -> [Int] -> VecA
accumulateIds n rows ids =
  let (forced, lineCodes, decided, marks) = localFeatures ids
   in accumulateRows n rows forced lineCodes decided marks

-- | Accumulator, last hidden layer and the three logits from a position's ids.
evalIds :: Net -> [Int] -> (VecA, VecH, Vec3)
evalIds n ids =
  let acc = accumulateIds n (foldedRows n ids) ids
      hid = hiddenOf n acc
   in (acc, hid, wdlOf n acc hid)

-- | A move's full logit: the global head plus the board's own cell logit and the send logit,
-- each through its gate.
moveLogit :: Net -> [Int] -> VecH -> Int -> Float
moveLogit n ids hid m =
  let board = m `div` 9
      square = m `mod` 9
      content b = ids !! (b * idsPerBoard)
      PolicyGates localGates sendGates = policyGates n hid
   in index9 square (policyLogits n hid board) + index9 board localGates * index9 square (byId (netLocal n) (content board)) + index9 square sendGates * byId (netSend n) (content square)

-- | Softmax, shifted by the maximum for stability.
softmax :: [Float] -> [Float]
softmax zs = let m = maximum zs; es = map (\z -> exp (z - m)) zs; t = sum es in map (/ t) es

-- | Largest absolute difference between two lists, 0 for empty ones.
maxAbsDiff :: [Float] -> [Float] -> Float
maxAbsDiff xs ys = maximum (0 : zipWith (\x y -> abs (x - y)) xs ys)

-- | Compares every reference position; prints the largest deviations and whether they are
-- within tolerance (1e-2 on pre-activation sums, 1e-3 on probabilities and between
-- sparseHiddenOf and hiddenOf, which sum in different orders).
selfTest :: IO Bool
selfTest = do
  let n = net
      cols = hiddenColumns n
      results =
        [ [maxAbsDiff (toList{a} acc) (refAcc r), maxAbsDiff (toList{h} hid) (refHidden r), maxAbsDiff (toList{h} (sparseHiddenOf n cols acc)) (toList{h} hid), maxAbsDiff (toList3 wdl) (refWdlLogits r), maxAbsDiff (softmax (toList3 wdl)) (refWdl r), maxAbsDiff logits refLogits, maxAbsDiff (softmax logits) refProbs]
        | r <- refs
        , let (acc, hid, wdl) = evalIds n (refIds r)
        , let moves = [m | (m, _, _) <- refPolicy r]
        , let refLogits = [l | (_, l, _) <- refPolicy r]
        , let refProbs = [p | (_, _, p) <- refPolicy r]
        , let logits = map (moveLogit n (refIds r) hid) moves
        ]
      worst = foldr1 (zipWith max) results
      labels = ["acc", "hidden", "sparse vs dense hidden", "wdl logits", "wdl", "policy logits", "policy"]
      limits = [1e-2, 1e-2, 1e-3, 1e-2, 1e-3, 1e-2, 1e-3]
      ok = and (zipWith (<) worst limits)
  putStrLn ("self-test over " ++ show (length refs) ++ " positions, largest deviations: " ++ unwords [l ++ " " ++ show w | (l, w) <- zip labels worst])
  putStrLn (if ok then "self-test passed" else "self-test FAILED")
  return ok

-- | Times the per-evaluation work after the fold (accumulator, hidden layers, value head,
-- policy gates and one board's policy logits) over the reference positions, with the dense hidden layers
-- (hiddenOf) and with the sparse ones (sparseHiddenOf), printing microseconds per evaluation.
bench :: Int -> IO ()
bench iters = do
  let n = net
      cols = hiddenColumns n
      inputs = [(foldedRows n (refIds r), refIds r) | r <- refs]
      count = length inputs
      step :: (VecA -> VecH) -> Int -> Float -> Float
      step hiddenFn i total =
        let (rows, ids) = inputs !! (i `mod` count)
            acc = accumulateIds n rows ids
            hid = hiddenFn acc
            wdl = wdlOf n acc hid
            pol = sum9 (policyLogits n hid (i `mod` 9)) + index9 0 (pgLocal (policyGates n hid))
         in total + index3 0 wdl + pol
      loop :: (VecA -> VecH) -> Int -> Float -> Float
      loop hiddenFn i total
        | i >= iters = total
        | otherwise = let total' = step hiddenFn i total in total' `seq` loop hiddenFn (i + 1) total'
      time :: String -> (VecA -> VecH) -> Float -> IO ()
      time label hiddenFn warm = do
        t0 <- getMonotonicTimeNSec
        total <- return $! loop hiddenFn 0 warm
        t1 <- getMonotonicTimeNSec
        putStrLn ("bench: " ++ show (fromIntegral (t1 - t0) / 1000 / fromIntegral iters :: Double) ++ " us per evaluation, " ++ label ++ " hidden layers (checksum " ++ show total ++ ")")
  -- Decode everything first: the strict fields of Net force a list-valued table only to its
  -- first element, so without this most of the decoding would land inside the timed loop.
  let walkLists = sum (map (index{e} 0) (ocOpen (netContent n) ++ netCells n)) + sum (ocOpen (netSend n)) + sum (map (index9 0) (ocOpen (netLocal n))) + sum [index{e} 0 (sel{CHUNK}_{e} 0 p) | ps <- netForced n : netMix n, p <- ps]
      walkRefs = sum [index3 0 wdl + index{a} 0 acc + index{h} 0 hid | r <- refs, let (acc, hid, wdl) = evalIds n (refIds r)]
      -- The columns are built once, outside the timed loop, as the engine keeps them.
      walkCols = sum [index{h} 0 (sparseHiddenOf n cols acc) | r <- refs, let (acc, _, _) = evalIds n (refIds r)]
  warm <- return $! walkLists + walkRefs + walkCols + sum (map (\(rows, _) -> sum (map (index{a} 0) rows)) inputs)
  time "dense" (hiddenOf n) warm
  time "sparse" (sparseHiddenOf n cols) warm
"#,
        last_content = CONTENTS - 1
    );
    s.push_str(&text);
    s
}
