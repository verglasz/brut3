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
use burn::tensor::activation::relu;

use super::model::{MOVES, Model, Prediction, Verdict};
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

/// Compact content id of a board from its two mark masks (bit `i` = square `i`).
pub fn content_id(mine: u16, theirs: u16) -> usize {
    let mut code = 0usize;
    let mut pow = 1usize;
    for cell in 0..CELLS {
        if mine >> cell & 1 != 0 {
            code += pow;
        } else if theirs >> cell & 1 != 0 {
            code += 2 * pow;
        }
        pow *= 3;
    }
    let id = COMPACT[code];
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
    let [n, k] = ids.dims();
    let [_, d] = table.weight.dims();
    let indices = ids.reshape([n * k, 1]).expand([n * k, d]);
    table.weight.val().gather(0, indices).reshape([n, k, d])
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
    /// Hidden layers: one from the accumulator, the rest residual.
    #[config(default = 2)]
    pub layers: usize,
}

impl Default for NnueConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl NnueConfig {
    pub fn init(&self, device: &Device) -> Nnue {
        assert!(self.layers >= 1, "at least one hidden layer");
        let normal = |std: f64| Initializer::Normal { mean: 0.0, std };
        let mut hidden = vec![LinearConfig::new(self.d_acc, self.d_hidden).init(device)];
        for _ in 1..self.layers {
            hidden.push(LinearConfig::new(self.d_hidden, self.d_hidden).init(device));
        }
        Nnue {
            config: self.clone(),
            content: EmbeddingConfig::new(CONTENT_ROWS, self.d_embed)
                .with_initializer(normal(0.02))
                .init(device),
            mix: LinearConfig::new(BOARDS * self.d_embed, self.d_acc).init(device),
            forced_mix: LinearConfig::new(self.d_embed, self.d_acc)
                .with_bias(false)
                .init(device),
            features: EmbeddingConfig::new(FEATURE_ROWS, self.d_acc)
                .with_initializer(normal(0.02))
                .init(device),
            hidden,
            value: LinearConfig::new(self.d_hidden, Verdict::ALL.len()).init(device),
            value_skip: LinearConfig::new(self.d_acc, Verdict::ALL.len())
                .with_bias(false)
                .init(device),
            policy: LinearConfig::new(self.d_hidden, MOVES).init(device),
            local_policy: EmbeddingConfig::new(CONTENTS, CELLS)
                .with_initializer(normal(0.02))
                .init(device),
            send_policy: EmbeddingConfig::new(CONTENTS, 1)
                .with_initializer(normal(0.02))
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Nnue {
    #[module(skip)]
    config: NnueConfig,
    /// `[CONTENT_ROWS, d_embed]`: contents, the no-forced row, cell rows, padding.
    pub(crate) content: Embedding,
    /// `[9 * d_embed, d_acc]`: the nine board-specific matrices side by side.
    pub(crate) mix: Linear,
    /// `[d_embed, d_acc]`: the forced board's content.
    pub(crate) forced_mix: Linear,
    /// `[FEATURE_ROWS, d_acc]`.
    pub(crate) features: Embedding,
    pub(crate) hidden: Vec<Linear>,
    pub(crate) value: Linear,
    pub(crate) value_skip: Linear,
    pub(crate) policy: Linear,
    /// `[CONTENTS, 9]`: a board's own cell logits.
    pub(crate) local_policy: Embedding,
    /// `[CONTENTS, 1]`: the logit of sending the opponent to a board with this content.
    pub(crate) send_policy: Embedding,
}

impl Nnue {
    pub fn config(&self) -> NnueConfig {
        self.config.clone()
    }

    /// Multiply-adds of one evaluation once the first layer is folded into tables: the MLP,
    /// the value heads and the policy over all 81 moves.
    pub fn folded_macs(&self) -> usize {
        let c = &self.config;
        c.d_acc * c.d_hidden
            + (c.layers - 1) * c.d_hidden * c.d_hidden
            + c.d_hidden * 3
            + c.d_acc * 3
            + c.d_hidden * MOVES
    }
}

impl Model for Nnue {
    type Input = NnueInput;

    fn encode(states: &[BitState], device: &Device) -> NnueInput {
        nnue_input(states, device)
    }

    fn predict(&self, ids: NnueInput) -> Prediction {
        let [n, _] = ids.dims();
        let d_e = self.config.d_embed;
        let d_acc = self.config.d_acc;
        let token_ids = ids.clone().slice([0..n, 0..TOKENS * IDS_PER_BOARD]);
        // Every token is the sum of its content row and its nine cell rows.
        let tokens = lookup(&self.content, token_ids)
            .reshape([n, TOKENS, IDS_PER_BOARD, d_e])
            .sum_dim(2)
            .reshape([n, TOKENS, d_e]);
        let boards = tokens
            .clone()
            .slice([0..n, 0..BOARDS])
            .reshape([n, BOARDS * d_e]);
        let forced = tokens.slice([0..n, BOARDS..TOKENS]).reshape([n, d_e]);
        let features = lookup(
            &self.features,
            ids.clone().slice([0..n, TOKENS * IDS_PER_BOARD..IDS]),
        )
        .sum_dim(1)
        .reshape([n, d_acc]);
        let acc = relu(self.mix.forward(boards) + self.forced_mix.forward(forced) + features);

        let mut h = relu(self.hidden[0].forward(acc.clone()));
        for layer in &self.hidden[1..] {
            h = relu(h.clone() + layer.forward(h));
        }

        let wdl = self.value.forward(h.clone()) + self.value_skip.forward(acc);

        // Content ids of the nine boards: every tenth id of the first nine tokens.
        let contents = ids
            .slice([0..n, 0..BOARDS * IDS_PER_BOARD])
            .reshape([n, BOARDS, IDS_PER_BOARD])
            .slice([0..n, 0..BOARDS, 0..1])
            .reshape([n, BOARDS]);
        // Board-major `[n, 9, 9]` is `Move::index` order (`board * 9 + square`).
        let local = lookup(&self.local_policy, contents.clone());
        let send = lookup(&self.send_policy, contents).reshape([n, 1, CELLS]);
        let policy = self.policy.forward(h).reshape([n, BOARDS, CELLS]) + local + send;
        Prediction {
            wdl,
            policy: policy.reshape([n, MOVES]),
        }
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
