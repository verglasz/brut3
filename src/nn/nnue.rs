//! [`Nnue`]: a table-driven network whose first layer folds into lookups at export time.
//!
//! Each small board is identified by its content from the side to move's view: the base-3
//! code of its nine cells, compacted so that every won board maps to one of two ids (won by
//! me, won by them) and every full drawn board to a third, since nothing about a decided
//! board matters beyond who holds it. Open contents keep their own id: [`CONTENTS`] ids in
//! all. A content embedding `E[c]` (plus, for open boards, per-cell embeddings so that rare
//! contents still get a sensible row) goes through a board-specific matrix `W_k`, and the nine
//! results are summed with a few feature rows (the forced board, the forced board's own
//! content, the state of the eight big-board lines, the game phase) into the accumulator.
//! That sum is one `Linear` over the concatenated embeddings here; an exported evaluator
//! precomputes `T_k[c] = E[c] W_k` for every content and adds rows instead.
//!
//! A ReLU MLP reads the accumulator. The value head is linear in the last hidden layer plus a
//! linear skip from the accumulator. The policy adds three terms: a global linear map from
//! the hidden layer to the 81 moves, a local table giving each open content's nine cell
//! logits, and a send table: playing square `j` on any board sends the opponent to board `j`,
//! so a logit read from board `j`'s content is added to square `j` of every board.

use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid};

use super::model::{MOVES, Model, Prediction, Predictor, Verdict};
use crate::game::{BitState, GameState, LINES};

pub const BOARDS: usize = 9;
pub const CELLS: usize = 9;
/// Raw contents: each of 9 cells is empty, mine or theirs.
pub const RAW_CONTENTS: usize = 3usize.pow(9);

/// Compact content ids: the open contents in increasing raw-code order, then the three
/// decided kinds.
pub const OPEN_CONTENTS: usize = count_open();
pub const FULL: usize = OPEN_CONTENTS;
pub const WON_BY_ME: usize = OPEN_CONTENTS + 1;
pub const WON_BY_THEM: usize = OPEN_CONTENTS + 2;
/// Number of compact content ids.
pub const CONTENTS: usize = OPEN_CONTENTS + 3;

/// Compact id of a raw code; `u16::MAX` for the unreachable codes where both players have a
/// line.
static COMPACT: [u16; RAW_CONTENTS] = compact_table();

const fn winner(code: usize) -> (bool, bool) {
    let mut mine = 0u16;
    let mut theirs = 0u16;
    let mut rest = code;
    let mut cell = 0;
    while cell < CELLS {
        match rest % 3 {
            1 => mine |= 1 << cell,
            2 => theirs |= 1 << cell,
            _ => {}
        }
        rest /= 3;
        cell += 1;
    }
    (has_line(mine), has_line(theirs))
}

const fn has_line(mask: u16) -> bool {
    let mut i = 0;
    while i < LINES.len() {
        let line = (1 << LINES[i][0]) | (1 << LINES[i][1]) | (1 << LINES[i][2]);
        if mask & line == line {
            return true;
        }
        i += 1;
    }
    false
}

const fn marks(code: usize) -> usize {
    let mut rest = code;
    let mut n = 0;
    while rest > 0 {
        if rest % 3 != 0 {
            n += 1;
        }
        rest /= 3;
    }
    n
}

const fn count_open() -> usize {
    let mut n = 0;
    let mut code = 0;
    while code < RAW_CONTENTS {
        let (m, t) = winner(code);
        if !m && !t && marks(code) < CELLS {
            n += 1;
        }
        code += 1;
    }
    n
}

const fn compact_table() -> [u16; RAW_CONTENTS] {
    let mut table = [u16::MAX; RAW_CONTENTS];
    let mut next = 0u16;
    let mut code = 0;
    while code < RAW_CONTENTS {
        table[code] = match winner(code) {
            (true, true) => u16::MAX,
            (true, false) => WON_BY_ME as u16,
            (false, true) => WON_BY_THEM as u16,
            (false, false) if marks(code) == CELLS => FULL as u16,
            (false, false) => {
                next += 1;
                next - 1
            }
        };
        code += 1;
    }
    table
}

/// Compact id of a raw content code (`sum of v * 3^cell`, `v` = 1 mine, 2 theirs), or `None`
/// where both players hold a line.
pub fn compact_id(code: usize) -> Option<usize> {
    match COMPACT[code] {
        u16::MAX => None,
        id => Some(id as usize),
    }
}

/// `TERNARY[mask]`: the base-3 number with digit 1 where the 9-bit `mask` has its bits; a
/// board's raw code is `TERNARY[mine] + 2 * TERNARY[theirs]` (see [`raw_code`]).
pub static TERNARY: [u16; 512] = {
    let mut table = [0u16; 512];
    let mut mask = 0;
    while mask < 512 {
        let (mut code, mut pow, mut cell) = (0u16, 1u16, 0);
        while cell < CELLS {
            if mask >> cell & 1 != 0 {
                code += pow;
            }
            pow *= 3;
            cell += 1;
        }
        table[mask] = code;
        mask += 1;
    }
    table
};

/// The raw code of a board from its two mark masks (bit `i` = square `i`, the masks
/// disjoint): the base-3 number whose digit `i` is 0 for an empty square, 1 for mine, 2 for
/// theirs. Two table loads.
#[inline]
pub fn raw_code(mine: u16, theirs: u16) -> usize {
    debug_assert_eq!(mine & theirs, 0, "a square marked by both players");
    TERNARY[mine as usize] as usize + 2 * TERNARY[theirs as usize] as usize
}

/// Compact content id of a board from its two mark masks (bit `i` = square `i`).
#[inline]
pub fn content_id(mine: u16, theirs: u16) -> usize {
    let id = COMPACT[raw_code(mine, theirs)];
    debug_assert!(id != u16::MAX, "unreachable board content");
    id as usize
}

// The content table: compact contents, a row for "no forced board", per-cell rows for open
// boards (cell times empty/mine/theirs) and a padding row standing in for the cells of
// decided boards and of a missing forced board.
pub const NO_FORCED: usize = CONTENTS;
pub const CELL_BASE: usize = NO_FORCED + 1;
pub const CELL_PAD: usize = CELL_BASE + CELLS * 3;
pub const CONTENT_ROWS: usize = CELL_PAD + 1;
/// Ids per board token: its content and its nine cells.
pub const IDS_PER_BOARD: usize = 1 + CELLS;
/// Board tokens: the nine boards, then the forced board again (or the "none" token).
pub const TOKENS: usize = BOARDS + 1;

// The feature table, already at accumulator width: the forced board (9 boards or none), the
// state of each big-board line (three statuses in 0..4), and the game phase as the number of
// decided boards and the marks on the open ones (both functions of the compact contents, so
// that dead cells never matter).
pub const FORCED_BASE: usize = 0;
pub const LINE_BASE: usize = FORCED_BASE + BOARDS + 1;
pub const LINE_STATES: usize = 4 * 4 * 4;
pub const DECIDED_BASE: usize = LINE_BASE + LINES.len() * LINE_STATES;
pub const OPEN_MARKS_BASE: usize = DECIDED_BASE + BOARDS + 1;
pub const FEATURE_ROWS: usize = OPEN_MARKS_BASE + BOARDS * (CELLS - 1) + 1;
pub const FEATURE_IDS: usize = 1 + LINES.len() + 2;

/// Ids per position: the board tokens, then the feature ids.
pub const IDS: usize = TOKENS * IDS_PER_BOARD + FEATURE_IDS;

/// Encoded positions: `[batch, IDS]`, see [`nnue_ids`].
pub type NnueInput = Tensor<2, Int>;

/// Status of a board for the line features: open, mine, theirs, full.
fn status(content: usize) -> usize {
    match content {
        c if c < OPEN_CONTENTS => 0,
        WON_BY_ME => 1,
        WON_BY_THEM => 2,
        _ => 3,
    }
}

/// Writes the ids of `state`, from the side to move's view, into `out`.
pub fn nnue_ids(state: &BitState, out: &mut [i32; IDS]) {
    let me = state.to_move();
    let them = me.other();
    let mut contents = [0usize; BOARDS];
    let mut open_marks = 0u32;
    let mut decided = 0usize;
    for (b, content) in contents.iter_mut().enumerate() {
        let mine = state.board_marks(me, b);
        let theirs = state.board_marks(them, b);
        *content = content_id(mine, theirs);
        if *content < OPEN_CONTENTS {
            open_marks += mine.count_ones() + theirs.count_ones();
        } else {
            decided += 1;
        }
        write_token(
            &mut out[b * IDS_PER_BOARD..(b + 1) * IDS_PER_BOARD],
            *content,
            mine,
            theirs,
        );
    }
    let forced = state.forced_board().map(|sq| sq.index());
    let forced_token = &mut out[BOARDS * IDS_PER_BOARD..TOKENS * IDS_PER_BOARD];
    match forced {
        Some(b) => write_token(
            forced_token,
            contents[b],
            state.board_marks(me, b),
            state.board_marks(them, b),
        ),
        None => {
            forced_token[0] = NO_FORCED as i32;
            forced_token[1..].fill(CELL_PAD as i32);
        }
    }
    let features = &mut out[TOKENS * IDS_PER_BOARD..];
    features[0] = (FORCED_BASE + forced.unwrap_or(BOARDS)) as i32;
    for (l, line) in LINES.iter().enumerate() {
        let code = status(contents[line[0]]) * 16
            + status(contents[line[1]]) * 4
            + status(contents[line[2]]);
        features[1 + l] = (LINE_BASE + l * LINE_STATES + code) as i32;
    }
    features[1 + LINES.len()] = (DECIDED_BASE + decided) as i32;
    features[2 + LINES.len()] = (OPEN_MARKS_BASE + open_marks as usize) as i32;
}

/// One board token's ids: the content, then the nine cell rows for an open board or padding
/// for a decided one (whose cells must not matter, so that a row is a function of the
/// compact content alone).
fn write_token(token: &mut [i32], content: usize, mine: u16, theirs: u16) {
    token[0] = content as i32;
    if content < OPEN_CONTENTS {
        for cell in 0..CELLS {
            let value = if mine >> cell & 1 != 0 {
                1
            } else if theirs >> cell & 1 != 0 {
                2
            } else {
                0
            };
            token[1 + cell] = (CELL_BASE + cell * 3 + value) as i32;
        }
    } else {
        token[1..].fill(CELL_PAD as i32);
    }
}

/// Rows `ids` (`[n, k]`) of an embedding table, as `[n, k, d]`.
///
/// Not `Embedding::forward`: on the CubeCL backends its gradient goes through `select_assign`,
/// a kernel with one thread per table column that loops over every index in the batch, which
/// made the embedding gradients most of a training step. `gather` on indices broadcast across
/// the row has an atomic scatter-add gradient that is parallel over every element.
fn lookup(table: &Embedding, ids: Tensor<2, Int>) -> Tensor<3> {
    rows(table.weight.val(), ids)
}

/// Rows `ids` (`[n, k]`) of `table`, as `[n, k, d]` (see [`lookup`]).
fn rows(table: Tensor<2>, ids: Tensor<2, Int>) -> Tensor<3> {
    let [n, k] = ids.dims();
    let [_, d] = table.dims();
    let indices = ids.reshape([n * k, 1]).expand([n * k, d]);
    table.gather(0, indices).reshape([n, k, d])
}

/// Encodes `states` as `[batch, IDS]` ids.
pub fn nnue_input(states: &[BitState], device: &Device) -> NnueInput {
    let n = states.len();
    let mut ids = vec![0i32; n * IDS];
    for (state, out) in states.iter().zip(ids.chunks_exact_mut(IDS)) {
        nnue_ids(state, out.try_into().expect("IDS ids per position"));
    }
    Tensor::from_data(TensorData::new(ids, [n, IDS]), device)
}

#[derive(Config, Debug, PartialEq)]
pub struct NnueConfig {
    /// Width of the content embedding (the rank of the factored first layer).
    #[config(default = 64)]
    pub d_embed: usize,
    /// Width of the accumulator (the folded tables' rows).
    #[config(default = 256)]
    pub d_acc: usize,
    /// Width of the hidden layers.
    #[config(default = 32)]
    pub d_hidden: usize,
    /// Hidden layers: one from the accumulator, the rest residual layers or ReGLU blocks.
    #[config(default = 2)]
    pub layers: usize,
    /// Inner width of the layers after the first as ReGLU blocks,
    /// `h <- h + W_down (ReLU(W_gate h + b) * (W_up h + b))` with `W_gate` and `W_up`
    /// mapping to this width (see [`ReGlu`]); `None` keeps residual layers
    /// `h <- ReLU(h + W h + b)`. An `Option` so that specs saved before it existed, which lack
    /// the field, load as residual networks.
    pub reglu: Option<usize>,
}

impl Default for NnueConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl NnueConfig {
    pub fn init(&self, device: &Device) -> Nnue {
        assert!(self.layers >= 1, "at least one hidden layer");
        let mut hidden = vec![LinearConfig::new(self.d_acc, self.d_hidden).init(device)];
        let mut glu = Vec::new();
        for _ in 1..self.layers {
            match self.reglu {
                Some(d_ff) => glu.push(ReGlu::new(self.d_hidden, d_ff, device)),
                None => hidden.push(LinearConfig::new(self.d_hidden, self.d_hidden).init(device)),
            }
        }
        let e = EndLayers::new(self.d_embed, self.d_acc, self.d_hidden, device);
        Nnue {
            config: self.clone(),
            content: e.content,
            mix: e.mix,
            forced_mix: e.forced_mix,
            features: e.features,
            hidden,
            glu,
            value: e.value,
            value_skip: e.value_skip,
            policy: e.policy,
            local_policy: e.local_policy,
            send_policy: e.send_policy,
            qat: Qat::default(),
        }
    }
}

/// The layers every NNUE model has around its hidden ones, freshly initialised: the first
/// layer's tables and matrices, and the heads. The models keep them as fields of their own,
/// under these names, so that saved weights keep their names, and lend them out as [`Ends`].
pub struct EndLayers {
    pub content: Embedding,
    pub mix: Linear,
    pub forced_mix: Linear,
    pub features: Embedding,
    pub value: Linear,
    pub value_skip: Linear,
    pub policy: Linear,
    pub local_policy: Embedding,
    pub send_policy: Embedding,
}

impl EndLayers {
    pub fn new(d_embed: usize, d_acc: usize, d_hidden: usize, device: &Device) -> Self {
        let normal = |std: f64| Initializer::Normal { mean: 0.0, std };
        Self {
            content: EmbeddingConfig::new(CONTENT_ROWS, d_embed)
                .with_initializer(normal(0.02))
                .init(device),
            mix: LinearConfig::new(BOARDS * d_embed, d_acc).init(device),
            forced_mix: LinearConfig::new(d_embed, d_acc)
                .with_bias(false)
                .init(device),
            features: EmbeddingConfig::new(FEATURE_ROWS, d_acc)
                .with_initializer(normal(0.02))
                .init(device),
            value: LinearConfig::new(d_hidden, Verdict::ALL.len()).init(device),
            value_skip: LinearConfig::new(d_acc, Verdict::ALL.len())
                .with_bias(false)
                .init(device),
            policy: LinearConfig::new(d_hidden, MOVES).init(device),
            local_policy: EmbeddingConfig::new(CONTENTS, CELLS)
                .with_initializer(normal(0.02))
                .init(device),
            send_policy: EmbeddingConfig::new(CONTENTS, 1)
                .with_initializer(normal(0.02))
                .init(device),
        }
    }
}

/// An NNUE model's layers around its hidden ones, borrowed (see [`EndLayers`]).
pub struct Ends<'a> {
    pub content: &'a Embedding,
    pub mix: &'a Linear,
    pub forced_mix: &'a Linear,
    pub features: &'a Embedding,
    pub value: &'a Linear,
    pub value_skip: &'a Linear,
    pub policy: &'a Linear,
    pub local_policy: &'a Embedding,
    pub send_policy: &'a Embedding,
    /// The head gates, `[d_hidden, GATES]` (see [`GATES`]); `None` for heads that add every
    /// term as it is.
    pub gates: Option<&'a Linear>,
}

/// Outputs of the head gates' layer, all through a sigmoid: the value's mix `g`, then the
/// nine boards' local-table gates `γ_b`, then the nine destinations' send-table gates `γ′_s`.
/// The value is `(1 − g) V h + g S acc`, and a move's logit `(P h)[b, s] + γ_b L[c_b][s] +
/// γ′_s T[c_s]`.
pub const GATES: usize = 1 + 2 * BOARDS;

/// An NNUE model: the shared first layer and heads ([`Ends`]) around hidden layers of its
/// own, which is all the prediction, the folding and [`FoldedNnue`] need.
pub trait NnueNet: Module {
    fn ends(&self) -> Ends<'_>;

    /// The tables that train fake-quantised (see [`Qat`]).
    fn qat(&self) -> &Qat;

    /// The hidden layers: from the rectified accumulator to the state the heads read.
    fn trunk(&self, acc: Tensor<2>) -> Tensor<2>;
}

/// A gated hidden block: `h + W_down (ReLU(W_gate h + b) * (W_up h + b))`, the ReLU-gated
/// linear unit of Shazeer's "GLU variants" (ReGLU) on a residual connection. Unlike the
/// residual layers, the stream itself is not rectified, so it carries negative values on.
/// `W_down` has no bias: whatever reads the stream next is affine (the next block's
/// projections, the heads), so a constant added to it would fold into their biases.
#[derive(Module, Debug)]
pub struct ReGlu {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl ReGlu {
    /// A block on a `d_hidden` stream with `d_ff`-wide gate and up projections.
    pub fn new(d_hidden: usize, d_ff: usize, device: &Device) -> Self {
        Self {
            gate: LinearConfig::new(d_hidden, d_ff).init(device),
            up: LinearConfig::new(d_hidden, d_ff).init(device),
            down: LinearConfig::new(d_ff, d_hidden)
                .with_bias(false)
                .init(device),
        }
    }

    /// The block's branch, to add to `h`.
    pub fn forward(&self, h: Tensor<2>) -> Tensor<2> {
        self.down
            .forward(relu(self.gate.forward(h.clone())) * self.up.forward(h))
    }
}

#[derive(Module, Debug)]
pub struct Nnue {
    #[module(skip)]
    config: NnueConfig,
    /// `[CONTENT_ROWS, d_embed]`: contents, the no-forced row, cell rows, padding.
    pub content: Embedding,
    /// `[9 * d_embed, d_acc]`: the nine board-specific matrices side by side.
    pub mix: Linear,
    /// `[d_embed, d_acc]`: the forced board's content.
    pub forced_mix: Linear,
    /// `[FEATURE_ROWS, d_acc]`.
    pub features: Embedding,
    /// The first hidden layer, then the residual layers when `reglu` is `None`.
    pub hidden: Vec<Linear>,
    /// The layers after the first when `reglu` is set.
    pub glu: Vec<ReGlu>,
    pub value: Linear,
    pub value_skip: Linear,
    pub policy: Linear,
    /// `[CONTENTS, 9]`: a board's own cell logits.
    pub local_policy: Embedding,
    /// `[CONTENTS, 1]`: the logit of sending the opponent to a board with this content.
    pub send_policy: Embedding,
    #[module(skip)]
    qat: Qat,
}

impl Nnue {
    pub fn config(&self) -> NnueConfig {
        self.config.clone()
    }

    /// Multiply-adds of one evaluation once the first layer is folded into tables: the MLP,
    /// the value heads and the policy over all 81 moves.
    pub fn folded_macs(&self) -> usize {
        let c = &self.config;
        let later = match c.reglu {
            None => c.d_hidden * c.d_hidden,
            Some(d_ff) => 3 * c.d_hidden * d_ff,
        };
        c.d_acc * c.d_hidden
            + (c.layers - 1) * later
            + c.d_hidden * 3
            + c.d_acc * 3
            + c.d_hidden * MOVES
    }
}

impl Nnue {
    /// The same network for inference, with its first layer folded into one table of
    /// accumulator rows indexed by raw board code (see [`FoldedNnue`]). Computed here on the
    /// model's device.
    pub fn fold(&self) -> FoldedNnue<Nnue> {
        fold(self)
    }
}

impl NnueNet for Nnue {
    fn ends(&self) -> Ends<'_> {
        Ends {
            content: &self.content,
            mix: &self.mix,
            forced_mix: &self.forced_mix,
            features: &self.features,
            value: &self.value,
            value_skip: &self.value_skip,
            policy: &self.policy,
            local_policy: &self.local_policy,
            send_policy: &self.send_policy,
            gates: None,
        }
    }

    fn qat(&self) -> &Qat {
        &self.qat
    }

    fn trunk(&self, acc: Tensor<2>) -> Tensor<2> {
        let mut h = relu(self.hidden[0].forward(acc));
        for layer in &self.hidden[1..] {
            h = relu(h.clone() + layer.forward(h));
        }
        for block in &self.glu {
            h = h.clone() + block.forward(h);
        }
        h
    }
}

impl Model for Nnue {
    type Input = NnueInput;

    fn encode(states: &[BitState], device: &Device) -> NnueInput {
        nnue_input(states, device)
    }

    fn predict(&self, ids: NnueInput) -> Prediction {
        predict(self, ids)
    }

    fn set_qat(&mut self, int8: &crate::nn::export::Int8Arrays) {
        self.qat = Qat::of(int8);
    }
}

/// The accumulator before its ReLU: the board tokens through their matrices, the forced token
/// through its own, the feature rows and the bias.
fn accumulate(e: &Ends, qat: &Qat, ids: NnueInput) -> Tensor<2> {
    let [n, _] = ids.dims();
    let [_, d_e] = e.content.weight.dims();
    let [_, d_acc] = e.features.weight.dims();
    let token_ids = ids.clone().slice([0..n, 0..TOKENS * IDS_PER_BOARD]);
    // Every token is the sum of its content row and its nine cell rows.
    let content = qat.rows(qat.content, e.content.weight.val());
    let tokens = rows(content, token_ids)
        .reshape([n, TOKENS, IDS_PER_BOARD, d_e])
        .sum_dim(2)
        .reshape([n, TOKENS, d_e]);
    let boards = tokens
        .clone()
        .slice([0..n, 0..BOARDS])
        .reshape([n, BOARDS * d_e]);
    let forced = tokens.slice([0..n, BOARDS..TOKENS]).reshape([n, d_e]);
    let features = qat.rows(qat.features, e.features.weight.val());
    let features = rows(features, ids.slice([0..n, TOKENS * IDS_PER_BOARD..IDS]))
        .sum_dim(1)
        .reshape([n, d_acc]);
    let mix = qat.mix_weight(e.mix.weight.val(), d_e);
    let bias = e.mix.bias.as_ref().expect("mix has a bias").val();
    let bias = qat.vector(qat.mix_bias, bias);
    let forced_mix = qat.forced_weight(e.forced_mix.weight.val());
    boards.matmul(mix) + bias.unsqueeze() + forced.matmul(forced_mix) + features
}

/// The heads: the value from the hidden state `h` and the rectified accumulator `acc`, and the
/// policy, which also reads the nine boards' contents (`[n, BOARDS]`) in the policy tables
/// `local_policy` (`[_, CELLS]`) and `send_policy` (`[_, 1]`), the model's own or re-indexed
/// copies. With gates, `h` weighs the skip and the tables (see [`GATES`]).
pub fn heads(
    e: &Ends,
    h: Tensor<2>,
    acc: Tensor<2>,
    contents: Tensor<2, Int>,
    local_policy: Tensor<2>,
    send_policy: Tensor<2>,
) -> Prediction {
    let n = contents.dims()[0];
    let value = e.value.forward(h.clone());
    let skip = e.value_skip.forward(acc);
    // Board-major `[n, 9, 9]` is `Move::index` order (`board * 9 + square`).
    let local = rows(local_policy, contents.clone());
    let send = rows(send_policy, contents).reshape([n, 1, CELLS]);
    let global = e.policy.forward(h.clone()).reshape([n, BOARDS, CELLS]);
    let (wdl, policy) = match e.gates {
        None => (value + skip, global + local + send),
        Some(gates) => {
            let g = sigmoid(gates.forward(h));
            let mix = g.clone().slice([0..n, 0..1]);
            let local_gate = g
                .clone()
                .slice([0..n, 1..1 + BOARDS])
                .reshape([n, BOARDS, 1]);
            let send_gate = g.slice([0..n, 1 + BOARDS..GATES]).reshape([n, 1, CELLS]);
            (
                value.clone() + (skip - value) * mix,
                global + local * local_gate + send * send_gate,
            )
        }
    };
    Prediction {
        wdl,
        policy: policy.reshape([n, MOVES]),
    }
}

/// An NNUE model's prediction from [`nnue_ids`] input: the accumulator, the model's trunk,
/// the heads.
pub fn predict<N: NnueNet>(net: &N, ids: NnueInput) -> Prediction {
    let [n, _] = ids.dims();
    // Content ids of the nine boards: every tenth id of the first nine tokens.
    let contents = ids
        .clone()
        .slice([0..n, 0..BOARDS * IDS_PER_BOARD])
        .reshape([n, BOARDS, IDS_PER_BOARD])
        .slice([0..n, 0..BOARDS, 0..1])
        .reshape([n, BOARDS]);
    let (e, qat) = (net.ends(), net.qat());
    let acc = relu(accumulate(&e, qat, ids));
    let h = net.trunk(acc.clone());
    heads(
        &e,
        h,
        acc,
        contents,
        qat.rows(qat.local, e.local_policy.weight.val()),
        qat.send_column(e.send_policy.weight.val()),
    )
}

/// Which of an NNUE's tables train fake-quantised (quantisation-aware training): their
/// weights in the forward pass are what the export's int8 grid makes of them, as the export
/// lays them out in rows (see [`crate::nn::export::Int8Arrays`]), while the gradient passes
/// through the rounding as if it were the identity (the straight-through estimator), so the
/// full-precision weights learn to do well once rounded. Covers the first layer's tables and
/// matrices and the policy tables, by the export's array names; set by
/// [`Model::set_qat`](super::model::Model::set_qat), off by default.
#[derive(Clone, Debug, Default)]
pub struct Qat {
    pub content: bool,
    pub mix: bool,
    pub mix_bias: bool,
    pub forced: bool,
    pub features: bool,
    pub local: bool,
    pub send: bool,
}

impl Qat {
    /// The tables whose export arrays `int8` covers.
    pub fn of(int8: &crate::nn::export::Int8Arrays) -> Self {
        Self {
            content: int8.covers("content_open"),
            mix: int8.covers("mix0_0"),
            mix_bias: int8.covers("mix_bias"),
            forced: int8.covers("forced_0"),
            features: int8.covers("features_forced"),
            local: int8.covers("local_open"),
            send: int8.covers("send_open"),
        }
    }

    /// `table` with each row on the int8 grid when `on`.
    fn rows(&self, on: bool, table: Tensor<2>) -> Tensor<2> {
        if on { fake_quantise(table, 1) } else { table }
    }

    /// A vector shipped as one row.
    fn vector(&self, on: bool, v: Tensor<1>) -> Tensor<1> {
        if !on {
            return v;
        }
        let [d] = v.dims();
        fake_quantise(v.reshape([1, d]), 1).reshape([d])
    }

    /// The mix's `[9 d_embed, d_acc]` weight: the export's rows are a board's slice of one
    /// output, `d_embed` inputs.
    fn mix_weight(&self, w: Tensor<2>, d_e: usize) -> Tensor<2> {
        if !self.mix {
            return w;
        }
        let [inputs, outputs] = w.dims();
        let w = w.reshape([inputs / d_e, d_e, outputs]);
        fake_quantise(w, 1).reshape([inputs, outputs])
    }

    /// The forced board's `[d_embed, d_acc]` slice: a row per output.
    fn forced_weight(&self, w: Tensor<2>) -> Tensor<2> {
        if self.forced { fake_quantise(w, 0) } else { w }
    }

    /// The send logits, `[CONTENTS, 1]`, shipped as two rows: the open contents and the
    /// decided ones.
    fn send_column(&self, w: Tensor<2>) -> Tensor<2> {
        if !self.send {
            return w;
        }
        let open = fake_quantise(w.clone().slice([0..OPEN_CONTENTS, 0..1]), 0);
        let decided = fake_quantise(w.slice([OPEN_CONTENTS..CONTENTS, 0..1]), 0);
        Tensor::cat(vec![open, decided], 0)
    }
}

/// `w` on the export's int8 grid with the export's rows along `dim` (each row the values that
/// share a scale): each row's scale is `2^(k / 8)` with the smallest `k` that covers its
/// largest value in 127 steps, and the values round to the nearest step. The gradient goes
/// straight through, as if `w` were returned (see [`Qat`]).
fn fake_quantise<const D: usize>(w: Tensor<D>, dim: usize) -> Tensor<D> {
    let need = w
        .clone()
        .abs()
        .max_dim(dim)
        .div_scalar(127.0)
        .clamp_min(1e-30);
    let steps = need
        .log()
        .div_scalar(std::f32::consts::LN_2)
        .mul_scalar(8.0)
        .ceil();
    let scale = steps.mul_scalar(std::f32::consts::LN_2 / 8.0).exp();
    let rounded = (w.clone() / scale.clone()).round().clamp(-127.0, 127.0) * scale;
    w.clone() + (rounded - w).detach()
}

/// `net` for inference, with its first layer folded into one table of accumulator rows
/// indexed by raw board code (see [`FoldedNnue`]). Computed on the model's device.
pub fn fold<N: NnueNet + Clone>(net: &N) -> FoldedNnue<N> {
    let e = net.ends();
    let device = e.content.weight.device();
    let [_, d_e] = e.content.weight.dims();
    let ids = Tensor::<2, Int>::from_data(
        TensorData::new(token_ids(), [CONTENTS + 1, IDS_PER_BOARD]),
        &device,
    );
    // The token of every content (rows 0..CONTENTS) and of "no forced board" (the last).
    let tokens = lookup(e.content, ids)
        .sum_dim(1)
        .reshape([CONTENTS + 1, d_e]);
    // Spread over the raw codes; codes no game reaches get zero rows (and so zero products),
    // never read.
    let (to_compact, reachable) = raw_contents();
    let to_compact =
        Tensor::<1, Int>::from_data(TensorData::new(to_compact, [RAW_CONTENTS]), &device);
    let reachable = Tensor::<2>::from_data(TensorData::new(reachable, [RAW_CONTENTS, 1]), &device);
    let spread = |table: Tensor<2>| table.select(0, to_compact.clone()) * reachable.clone();
    let raw_tokens = spread(tokens.clone().slice([0..CONTENTS]));
    let mix = e.mix.weight.val();
    let mut parts: Vec<Tensor<2>> = (0..BOARDS)
        .map(|k| {
            raw_tokens
                .clone()
                .matmul(mix.clone().slice([k * d_e..(k + 1) * d_e]))
        })
        .collect();
    // Every position has exactly one forced token, so the layer's bias rides on its rows (only
    // on the reachable ones and "no forced board", the others staying zero).
    let forced_tokens = Tensor::cat(vec![raw_tokens, tokens.slice([CONTENTS..CONTENTS + 1])], 0);
    let mut forced = forced_tokens.matmul(e.forced_mix.weight.val());
    if let Some(bias) = &e.mix.bias {
        let used = Tensor::cat(vec![reachable.clone(), Tensor::ones([1, 1], &device)], 0);
        forced = forced + bias.val().unsqueeze::<2>() * used;
    }
    parts.push(forced);
    parts.push(e.features.weight.val());
    let offsets = (0..BOARDS).map(|k| (k * RAW_CONTENTS) as i32).collect();
    FoldedNnue {
        table: Tensor::cat(parts, 0),
        offsets: Tensor::from_data(TensorData::new(offsets, [1, BOARDS]), &device),
        local_policy: spread(e.local_policy.weight.val()),
        send_policy: spread(e.send_policy.weight.val()),
        net: net.clone(),
    }
}

/// The token ids of every compact content in order, then of "no forced board": rows of
/// [`IDS_PER_BOARD`] ids as [`write_token`] writes them for a board holding that content.
fn token_ids() -> Vec<i32> {
    let mut ids = vec![0i32; (CONTENTS + 1) * IDS_PER_BOARD];
    let row = |c: usize| c * IDS_PER_BOARD..(c + 1) * IDS_PER_BOARD;
    // Open contents from their raw codes; a decided content's cells are padding anyway.
    for code in 0..RAW_CONTENTS {
        let Some(c) = compact_id(code) else { continue };
        let (mut mine, mut theirs, mut rest) = (0u16, 0u16, code);
        for cell in 0..CELLS {
            match rest % 3 {
                1 => mine |= 1 << cell,
                2 => theirs |= 1 << cell,
                _ => {}
            }
            rest /= 3;
        }
        write_token(&mut ids[row(c)], c, mine, theirs);
    }
    let none = row(CONTENTS);
    ids[none.start] = NO_FORCED as i32;
    ids[none.start + 1..none.end].fill(CELL_PAD as i32);
    ids
}

/// For every raw code, its compact content id (0 where unreachable) and whether a game can
/// reach it (1 or 0).
fn raw_contents() -> (Vec<i32>, Vec<f32>) {
    (0..RAW_CONTENTS)
        .map(|code| match compact_id(code) {
            Some(c) => (c as i32, 1.0),
            None => (0, 0.0),
        })
        .unzip()
}

/// Rows of the folded table: the nine board tables, the forced table, the feature rows.
const FORCED_TABLE: usize = BOARDS * RAW_CONTENTS;
const NO_FORCED_ROW: usize = FORCED_TABLE + RAW_CONTENTS;
const FEATURE_TABLE: usize = NO_FORCED_ROW + 1;
/// Rows of the folded table one position reads: one per board, the forced token, the
/// features.
pub const FOLDED_IDS: usize = BOARDS + 1 + FEATURE_IDS;

/// [`Nnue`] for inference with the first layer folded into a table, as the Haskell export's
/// engine computes it: a board's token only depends on its content, so `W_k token(c)` is
/// precomputed for every board `k` and content `c` (and the forced token's product, with the
/// bias), and the accumulator is a sum of [`FOLDED_IDS`] table rows instead of the
/// `9 d_embed x d_acc` product.
///
/// The tables are indexed by raw base-3 board code, which [`folded_ids`] reads straight from
/// the bitboard's masks, not by compact content id: that skips a lookup per board for about
/// `9 x 19,683 x d_acc` floats (346 MiB at `d_acc` 512, against 195 MiB compact; the rows of
/// unreachable codes stay zero). Built eagerly, by [`Nnue::fold`], on the model's device.
#[derive(Debug)]
pub struct FoldedNnue<N> {
    /// `[10 RAW_CONTENTS + 1 + FEATURE_ROWS, d_acc]`: board `k` with raw code `r` at row
    /// `k RAW_CONTENTS + r`, then the forced tokens with the bias (raw codes, then "no forced
    /// board"), then the feature rows.
    table: Tensor<2>,
    /// `[1, BOARDS]`: the first row of each board table, to read the raw codes back.
    offsets: Tensor<2, Int>,
    /// The network's policy tables by raw code: `[RAW_CONTENTS, CELLS]` and `[RAW_CONTENTS, 1]`.
    local_policy: Tensor<2>,
    send_policy: Tensor<2>,
    /// The network the table was folded from, for everything after the accumulator.
    net: N,
}

/// Writes the rows of the folded table that `state` reads into `out`: the same position as
/// [`nnue_ids`] describes, from the bitboard's masks directly.
pub fn folded_ids(state: &BitState, out: &mut [i32; FOLDED_IDS]) {
    let me = state.to_move();
    let them = me.other();
    let (won_by_me, won_by_them) = (state.big_marks(me), state.big_marks(them));
    let done = state.done_mask();
    let mut codes = [0usize; BOARDS];
    let mut status = [0usize; BOARDS];
    let mut open_marks = 0;
    for b in 0..BOARDS {
        let (mine, theirs) = (state.board_marks(me, b), state.board_marks(them, b));
        codes[b] = raw_code(mine, theirs);
        out[b] = (b * RAW_CONTENTS + codes[b]) as i32;
        // As `status` reads the compact content: open, mine, theirs, full.
        status[b] = if won_by_me >> b & 1 != 0 {
            1
        } else if won_by_them >> b & 1 != 0 {
            2
        } else if done >> b & 1 != 0 {
            3
        } else {
            open_marks += (mine | theirs).count_ones() as usize;
            0
        };
    }
    let forced = state.forced_board().map(|sq| sq.index());
    out[BOARDS] = match forced {
        Some(b) => FORCED_TABLE + codes[b],
        None => NO_FORCED_ROW,
    } as i32;
    let features = &mut out[BOARDS + 1..];
    features[0] = (FEATURE_TABLE + FORCED_BASE + forced.unwrap_or(BOARDS)) as i32;
    for (l, line) in LINES.iter().enumerate() {
        let code = status[line[0]] * 16 + status[line[1]] * 4 + status[line[2]];
        features[1 + l] = (FEATURE_TABLE + LINE_BASE + l * LINE_STATES + code) as i32;
    }
    features[1 + LINES.len()] = (FEATURE_TABLE + DECIDED_BASE + done.count_ones() as usize) as i32;
    features[2 + LINES.len()] = (FEATURE_TABLE + OPEN_MARKS_BASE + open_marks) as i32;
}

impl<N: NnueNet> Predictor for FoldedNnue<N> {
    type Input = Tensor<2, Int>;

    fn encode(states: &[BitState], device: &Device) -> Tensor<2, Int> {
        let n = states.len();
        let mut ids = vec![0i32; n * FOLDED_IDS];
        for (state, out) in states.iter().zip(ids.chunks_exact_mut(FOLDED_IDS)) {
            folded_ids(state, out.try_into().expect("FOLDED_IDS ids per position"));
        }
        Tensor::from_data(TensorData::new(ids, [n, FOLDED_IDS]), device)
    }

    fn predict(&self, ids: Tensor<2, Int>) -> Prediction {
        let [n, _] = ids.dims();
        let [_, d_acc] = self.table.dims();
        let acc = relu(
            rows(self.table.clone(), ids.clone())
                .sum_dim(1)
                .reshape([n, d_acc]),
        );
        let codes = ids.slice([0..n, 0..BOARDS]) - self.offsets.clone().expand([n, BOARDS]);
        let h = self.net.trunk(acc.clone());
        heads(
            &self.net.ends(),
            h,
            acc,
            codes,
            self.local_policy.clone(),
            self.send_policy.clone(),
        )
    }
}

/// Asserts that `net` folded computes what `net` does, up to rounding, on positions from
/// random games (forced and free moves, decided boards). The caller holds the device lock.
#[cfg(test)]
pub(crate) fn assert_folds<N>(net: &N, device: &Device)
where
    N: NnueNet + Model<Input = NnueInput> + Clone,
{
    use rand::{SeedableRng, rngs::SmallRng};

    use super::model::predict_logits;

    let mut rng = SmallRng::seed_from_u64(5);
    let mut states = Vec::new();
    for _ in 0..8 {
        let mut s = BitState::initial();
        loop {
            states.push(s);
            let mv = s.random_move(&mut rng);
            if s.play(mv).is_some() {
                break;
            }
        }
    }
    let expected = predict_logits(net, &states, device);
    let folded = predict_logits(&fold(net), &states, device);
    assert_eq!(expected.len(), folded.len());
    for (i, (e, f)) in expected.iter().zip(&folded).enumerate() {
        assert!(
            (e - f).abs() <= 1e-4 * (1.0 + e.abs()),
            "logit {i}: {e} vs {f}"
        );
    }
}

crate::nn::train::impl_train_steps!(Nnue);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::test_positions::{o_must_block, x_wins_in_one};
    use crate::game::{Move, Player, Square, Symmetry};
    use crate::nn::BackendKind;
    use crate::nn::model::predict_host;

    #[test]
    fn compact_contents_count_and_classify() {
        // 11125 line-free contents, of which 32 are full.
        assert_eq!(OPEN_CONTENTS, 11093);
        assert_eq!(CONTENTS, 11096);
        assert_eq!(content_id(0, 0), 0);
        assert_eq!(content_id(0b111, 0), WON_BY_ME);
        assert_eq!(content_id(0, 0b100_010_001), WON_BY_THEM);
        // X O X / X O O / O X X: full, no line.
        assert_eq!(content_id(0b110_001_101, 0b001_110_010), FULL);
        let open: std::collections::HashSet<usize> = (0..RAW_CONTENTS)
            .filter_map(|code| {
                let id = COMPACT[code];
                (id != u16::MAX && (id as usize) < OPEN_CONTENTS).then_some(id as usize)
            })
            .collect();
        assert_eq!(open.len(), OPEN_CONTENTS, "open ids are distinct and dense");
        assert_eq!(
            COMPACT.iter().filter(|&&id| id == u16::MAX).count(),
            unreachable()
        );
    }

    /// Codes where both players hold a line.
    fn unreachable() -> usize {
        (0..RAW_CONTENTS)
            .filter(|&code| matches!(winner(code), (true, true)))
            .count()
    }

    #[test]
    fn ids_describe_the_position() {
        let mut ids = [0i32; IDS];
        nnue_ids(&BitState::initial(), &mut ids);
        assert!(ids.iter().all(|&id| id >= 0));
        for b in 0..BOARDS {
            assert_eq!(ids[b * IDS_PER_BOARD], 0, "empty content");
            assert_eq!(ids[b * IDS_PER_BOARD + 1], CELL_BASE as i32, "cell 0 empty");
        }
        assert_eq!(ids[BOARDS * IDS_PER_BOARD], NO_FORCED as i32);
        assert_eq!(ids[BOARDS * IDS_PER_BOARD + 1], CELL_PAD as i32);
        let features = &ids[TOKENS * IDS_PER_BOARD..];
        assert_eq!(features[0], (FORCED_BASE + BOARDS) as i32);
        assert_eq!(features[1], LINE_BASE as i32);
        assert_eq!(features[1 + LINES.len()], DECIDED_BASE as i32);
        assert_eq!(features[2 + LINES.len()], OPEN_MARKS_BASE as i32);

        // X to move, boards 0 and 1 won by X, forced to board 2 where X has squares 0 and 1.
        let state = x_wins_in_one();
        nnue_ids(&state, &mut ids);
        assert_eq!(ids[0], WON_BY_ME as i32);
        assert_eq!(
            ids[1], CELL_PAD as i32,
            "a decided board's cells are padding"
        );
        assert_eq!(ids[IDS_PER_BOARD], WON_BY_ME as i32);
        let board2 = ids[2 * IDS_PER_BOARD] as usize;
        assert!(board2 < OPEN_CONTENTS);
        assert_eq!(
            ids[2 * IDS_PER_BOARD + 1],
            (CELL_BASE + 1) as i32,
            "square 0 mine"
        );
        assert_eq!(
            ids[2 * IDS_PER_BOARD + 3],
            (CELL_BASE + 6) as i32,
            "square 2 empty"
        );
        assert_eq!(
            ids[BOARDS * IDS_PER_BOARD],
            board2 as i32,
            "forced token is board 2"
        );
        let features = &ids[TOKENS * IDS_PER_BOARD..];
        assert_eq!(features[0], (FORCED_BASE + 2) as i32);
        // Top row: mine, mine, open.
        assert_eq!(features[1], (LINE_BASE + 16 + 4) as i32);
        assert_eq!(features[1 + LINES.len()], (DECIDED_BASE + 2) as i32);
        // Board 2 holds two X marks, board 4 three O marks; boards 0 and 1 are decided.
        assert_eq!(features[2 + LINES.len()], (OPEN_MARKS_BASE + 5) as i32);

        // The same position from O's view swaps mine and theirs.
        let mut p = state.to_position();
        p.set_to_move(Player::O);
        nnue_ids(&BitState::from_position(&p), &mut ids);
        assert_eq!(ids[0], WON_BY_THEM as i32);
        assert_eq!(ids[2 * IDS_PER_BOARD + 1], (CELL_BASE + 2) as i32);
    }

    fn folded_matches(config: NnueConfig) {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        assert_folds(&config.init(&device), &device);
    }

    fn small() -> NnueConfig {
        NnueConfig::new()
            .with_d_embed(16)
            .with_d_acc(32)
            .with_d_hidden(16)
            .with_layers(2)
    }

    #[test]
    fn folded_network_matches_the_network() {
        folded_matches(small());
    }

    #[test]
    fn folded_reglu_network_matches_the_network() {
        folded_matches(small().with_layers(3).with_reglu(Some(24)));
    }

    #[test]
    fn reglu_shapes_macs_and_old_specs() {
        let device = BackendKind::preferred().device();
        let _lock = crate::nn::device_lock();
        let model = small().with_layers(3).with_reglu(Some(24)).init(&device);
        assert_eq!(model.hidden.len(), 1);
        assert_eq!(model.glu.len(), 2);
        assert_eq!(
            model.folded_macs(),
            32 * 16 + 2 * 3 * 16 * 24 + 16 * 3 + 32 * 3 + 16 * 81
        );
        // Specs saved before the option existed load as residual networks.
        let old: NnueConfig =
            serde_json::from_str(r#"{"d_embed":64,"d_acc":512,"d_hidden":128,"layers":4}"#)
                .expect("an old spec parses");
        assert_eq!(old.reglu, None);
    }

    #[test]
    fn predicts_shapes_and_legal_policies() {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let model = NnueConfig::new()
            .with_d_embed(16)
            .with_d_acc(32)
            .with_d_hidden(16)
            .with_layers(2)
            .init(&device);
        assert_eq!(
            model.folded_macs(),
            32 * 16 + 16 * 16 + 16 * 3 + 32 * 3 + 16 * 81
        );
        let states = [BitState::initial(), x_wins_in_one(), o_must_block()];
        let p = predict_host(&model, &states, &device);
        assert_eq!(p.wdl.len(), 3);
        assert_eq!(p.policy[0].len(), 81);
        assert_eq!(p.policy[1].len(), 7);
        assert_eq!(p.policy[2].len(), 6);
        for (wdl, policy) in p.wdl.iter().zip(&p.policy) {
            assert!((wdl.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            assert!((policy.iter().map(|(_, p)| p).sum::<f32>() - 1.0).abs() < 1e-3);
        }
        let _ = Move::new(Square::from_index(0), Square::from_index(0));
    }

    /// Decided boards contribute the same rows whatever their dead cells: two positions
    /// differing only inside a won board get identical predictions.
    #[test]
    fn dead_cells_do_not_matter() {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let model = NnueConfig::new()
            .with_d_embed(16)
            .with_d_acc(32)
            .with_d_hidden(16)
            .init(&device);
        let a = x_wins_in_one();
        let mut p = a.to_position();
        // Board 0 is won by X through its middle row; add an X in its corner.
        p.set_cell(
            Move::new(Square::from_index(0), Square::from_index(0)),
            Some(Player::X),
        );
        let b = BitState::from_position(&p);
        assert_ne!(a, b);
        let pa = predict_host(&model, &[a], &device);
        let pb = predict_host(&model, &[b], &device);
        assert_eq!(pa, pb);
        // Whereas a symmetry of the position changes the ids (the net is not equivariant).
        let mut ids_a = [0; IDS];
        let mut ids_c = [0; IDS];
        nnue_ids(&a, &mut ids_a);
        nnue_ids(&a.transform(Symmetry::Rotate90), &mut ids_c);
        assert_ne!(ids_a, ids_c);
    }
}
