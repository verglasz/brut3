//! [`CellTransformer`]: a transformer over the 81 cells of the board.
//!
//! Each cell is one token: an embedding of its content (empty, mine, theirs, from the side to
//! move's view) plus embeddings of its position among the 81 cells, the status of the small
//! board it belongs to (open, won by me, won by them, full), whether that board is the forced
//! one, and whether the cell is a legal move right now. All five live in one embedding table
//! at distinct offsets, so a position is `81 * 5` ids in one upload and one gather, summed per
//! cell. The tokens go through pre-norm encoder layers with all-to-all attention, and two
//! perceptron heads read the flattened final residual stream to produce the move and outcome
//! logits.

use burn::nn::Embedding;
use burn::nn::transformer::TransformerEncoder;
use burn::prelude::*;

use super::heads::{
    BodyConfig,
    ClsReadout,
    ClsReadoutConfig,
    Perceptron,
    PerceptronConfig,
    embedding,
    encode_all_to_all,
};
use super::model::{MOVES, Model, Prediction, Verdict};
use crate::game::{BitState, BoardStatus, GameState, Move, MoveSet, Square};

const CELLS: usize = MOVES;
/// Cell contents: empty, mine, theirs.
const CONTENTS: usize = 3;
/// Board statuses: open, won by me, won by them, full.
const STATUSES: usize = 4;

// One embedding table holds every kind of id, each kind at its own offset.
const CONTENT_BASE: usize = 0;
const POSITION_BASE: usize = CONTENT_BASE + CONTENTS;
const STATUS_BASE: usize = POSITION_BASE + CELLS;
const FORCED_BASE: usize = STATUS_BASE + STATUSES;
const LEGAL_BASE: usize = FORCED_BASE + 2;
const VOCABULARY: usize = LEGAL_BASE + 2;
/// Ids per cell token: content, position, status, forced, legal.
const IDS_PER_CELL: usize = 5;
/// Ids per position.
pub const IDS: usize = CELLS * IDS_PER_CELL;

/// The embedding table and encoder body shared by the cell models: `[batch, IDS]` ids in,
/// `[batch, 81, d_model]` residual stream out.
#[derive(Module, Debug)]
pub struct CellEncoder {
    /// Content, position, status, forced and legal embeddings, at the `*_BASE` offsets.
    embed: Embedding,
    body: TransformerEncoder,
    #[module(skip)]
    d_model: usize,
}

impl CellEncoder {
    pub fn init(body: &BodyConfig, device: &Device) -> Self {
        Self {
            embed: embedding(VOCABULARY, body.d_model, device),
            body: body.init(device),
            d_model: body.d_model,
        }
    }

    /// One gather for every id, then the 5 embeddings of each cell summed into its token:
    /// `[n, 405, d] -> [n, 81, 5, d] -> [n, 81, d]`.
    pub fn tokens(&self, input: CellTokens) -> Tensor<3> {
        let n = input.dims()[0];
        let d = self.d_model;
        self.embed
            .forward(input)
            .reshape([n, CELLS, IDS_PER_CELL, d])
            .sum_dim(2)
            .reshape([n, CELLS, d])
    }

    /// The encoder body over any token sequence `[n, tokens, d]`.
    pub fn body(&self, tokens: Tensor<3>) -> Tensor<3> {
        encode_all_to_all(&self.body, tokens)
    }

    /// Ids in, residual stream `[n, 81, d]` out.
    pub fn forward(&self, input: CellTokens) -> Tensor<3> {
        self.body(self.tokens(input))
    }
}

/// [`CellTransformer`]: the encoder read by two perceptrons over the flattened stream.
#[derive(Config, Debug, PartialEq)]
pub struct CellTransformerConfig {
    pub body: BodyConfig,
    /// Hidden width of the two heads; `0` means `2 * d_model`.
    #[config(default = 0)]
    pub d_head: usize,
}

impl Default for CellTransformerConfig {
    fn default() -> Self {
        Self::new(BodyConfig::new())
    }
}

impl CellTransformerConfig {
    fn d_head(&self) -> usize {
        if self.d_head == 0 {
            2 * self.body.d_model
        } else {
            self.d_head
        }
    }

    pub fn init(&self, device: &Device) -> CellTransformer {
        let d = self.body.d_model;
        let head = |out: usize| {
            PerceptronConfig::new(CELLS * d, self.d_head(), out)
                .with_dropout(self.body.dropout)
                .init(device)
        };
        CellTransformer {
            config: self.clone(),
            encoder: CellEncoder::init(&self.body, device),
            policy_head: head(MOVES),
            wdl_head: head(Verdict::ALL.len()),
        }
    }
}

#[derive(Module, Debug)]
pub struct CellTransformer {
    #[module(skip)]
    config: CellTransformerConfig,
    encoder: CellEncoder,
    policy_head: Perceptron,
    wdl_head: Perceptron,
}

/// [`CellTransformerV2`]: the same encoder with per-token heads. A class token
/// ([`ClsReadout`]) goes through the encoder with the 81 cell tokens and is read out as the
/// value; each cell token is one move, so the policy is one perceptron, shared by the cell
/// tokens, from a token and the class token's readout to the token's logit. Both read the
/// stream through one final layer norm.
#[derive(Config, Debug, PartialEq)]
pub struct CellTransformerV2Config {
    pub body: BodyConfig,
    /// Hidden width of the policy and value perceptrons; `0` means `d_model`.
    #[config(default = 0)]
    pub d_head: usize,
}

impl Default for CellTransformerV2Config {
    fn default() -> Self {
        Self::new(BodyConfig::new())
    }
}

impl CellTransformerV2Config {
    fn d_head(&self) -> usize {
        if self.d_head == 0 {
            self.body.d_model
        } else {
            self.d_head
        }
    }

    pub fn init(&self, device: &Device) -> CellTransformerV2 {
        let d = self.body.d_model;
        CellTransformerV2 {
            config: self.clone(),
            encoder: CellEncoder::init(&self.body, device),
            policy: PerceptronConfig::new(2 * d, self.d_head(), 1)
                .with_dropout(self.body.dropout)
                .init(device),
            cls: ClsReadoutConfig::new(d, Verdict::ALL.len(), self.d_head())
                .with_dropout(self.body.dropout)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct CellTransformerV2 {
    #[module(skip)]
    config: CellTransformerV2Config,
    encoder: CellEncoder,
    /// Shared by the tokens: a token with the global readout appended, to its move logits.
    policy: Perceptron,
    cls: ClsReadout,
}

/// Embedding ids for a batch: `[batch, IDS]`, see [`cell_token_ids`].
pub type CellTokens = Tensor<2, Int>;

/// Embedding ids of one position, cell-major by `Move::index`: for each cell its content
/// (`0` empty, `1` mine, `2` theirs), position, its board's status (`0` open, `1` won by me,
/// `2` won by them, `3` full), whether that board is the forced one and whether the cell is a
/// legal move now, each at its table offset.
pub type CellTokenIds = [i32; IDS];

/// Writes the ids of `state`, from the side to move's view, into `out`.
pub fn cell_token_ids(state: &BitState, out: &mut CellTokenIds) {
    let me = state.to_move();
    let them = me.other();
    let forced_board = state.forced_board();
    let legal_moves = state.legal_moves();
    for (b, board) in Square::ALL.iter().enumerate() {
        let mine = state.board_marks(me, b);
        let theirs = state.board_marks(them, b);
        let status = STATUS_BASE as i32
            + match state.board_status(*board) {
                BoardStatus::Open => 0,
                BoardStatus::Won(p) if p == me => 1,
                BoardStatus::Won(_) => 2,
                BoardStatus::Full => 3,
            };
        let forced = FORCED_BASE as i32 + i32::from(forced_board == Some(*board));
        for (c, square) in Square::ALL.iter().enumerate() {
            let mv = Move::new(*board, *square);
            let i = mv.index();
            let token = &mut out[i * IDS_PER_CELL..(i + 1) * IDS_PER_CELL];
            token[0] = CONTENT_BASE as i32
                + if mine >> c & 1 != 0 {
                    1
                } else if theirs >> c & 1 != 0 {
                    2
                } else {
                    0
                };
            token[1] = (POSITION_BASE + i) as i32;
            token[2] = status;
            token[3] = forced;
            token[4] = LEGAL_BASE as i32 + i32::from(legal_moves.contains(mv));
        }
    }
}

/// Encodes `states` as `[batch, IDS]` embedding ids.
pub fn cell_tokens(states: &[BitState], device: &Device) -> CellTokens {
    let n = states.len();
    let mut ids = vec![0i32; n * IDS];
    for (state, out) in states.iter().zip(ids.chunks_exact_mut(IDS)) {
        cell_token_ids(state, out.try_into().expect("IDS ids per position"));
    }
    Tensor::from_data(TensorData::new(ids, [n, IDS]), device)
}

impl CellTransformer {
    pub fn config(&self) -> CellTransformerConfig {
        self.config.clone()
    }
}

impl Model for CellTransformer {
    type Input = CellTokens;

    fn encode(states: &[BitState], device: &Device) -> CellTokens {
        cell_tokens(states, device)
    }

    fn predict(&self, input: CellTokens) -> Prediction {
        let stream = self.encoder.forward(input).flatten::<2>(1, 2);
        Prediction {
            wdl: self.wdl_head.forward(stream.clone()),
            policy: self.policy_head.forward(stream),
        }
    }
}

crate::nn::train::impl_train_steps!(CellTransformer);

impl CellTransformerV2 {
    pub fn config(&self) -> CellTransformerV2Config {
        self.config.clone()
    }
}

impl Model for CellTransformerV2 {
    type Input = CellTokens;

    fn encode(states: &[BitState], device: &Device) -> CellTokens {
        cell_tokens(states, device)
    }

    fn predict(&self, input: CellTokens) -> Prediction {
        let tokens = self.encoder.tokens(input);
        let n = tokens.dims()[0];
        let stream = self.encoder.body(self.cls.prepend(tokens));
        let (global, cells) = self.cls.split(stream);
        let wdl = self.cls.output(global.clone());
        // Cell tokens are in `Move::index` order, one logit each.
        let policy = self
            .policy
            .forward(ClsReadout::with_global(global, cells))
            .reshape([n, MOVES]);
        Prediction { wdl, policy }
    }
}

crate::nn::train::impl_train_steps!(CellTransformerV2);

#[cfg(test)]
mod tests {

    use super::*;
    use crate::game::sq;
    use crate::game::test_positions::x_wins_in_one;
    use crate::nn::BackendKind;
    use crate::nn::model::predict_host;
    use crate::nn::train::tests::{opts, synthetic_dataset};
    use crate::nn::train::{evaluate, train};

    fn small() -> CellTransformerConfig {
        CellTransformerConfig::new(
            BodyConfig::new()
                .with_d_model(16)
                .with_n_layers(1)
                .with_n_heads(2),
        )
    }

    fn small_v2() -> CellTransformerV2Config {
        CellTransformerV2Config::new(small().body.with_n_layers(2))
    }

    #[test]
    fn v2_predicts_shapes_and_legal_policies() {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let model = small_v2().init(&device);
        let states = [BitState::initial(), x_wins_in_one()];
        let p = predict_host(&model, &states, &device);
        assert_eq!(p.wdl.len(), 2);
        assert_eq!(p.policy[0].len(), 81);
        assert_eq!(p.policy[1].len(), 7);
        for (wdl, policy) in p.wdl.iter().zip(&p.policy) {
            assert!((wdl.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            assert!((policy.iter().map(|(_, p)| p).sum::<f32>() - 1.0).abs() < 1e-3);
        }
    }

    /// The ids of a position, unpacked into `(content, status, forced, legal)` with the table
    /// offsets removed; checks the position ids on the way.
    fn unpacked(state: &BitState) -> ([i32; 81], [i32; 81], [i32; 81], [i32; 81]) {
        let mut ids = [0; IDS];
        cell_token_ids(state, &mut ids);
        assert!(ids.iter().all(|&id| (id as usize) < VOCABULARY));
        let (mut content, mut status, mut forced, mut legal) = ([0; 81], [0; 81], [0; 81], [0; 81]);
        for i in 0..CELLS {
            let token = &ids[i * IDS_PER_CELL..(i + 1) * IDS_PER_CELL];
            content[i] = token[0] - CONTENT_BASE as i32;
            assert_eq!(token[1] as usize, POSITION_BASE + i);
            status[i] = token[2] - STATUS_BASE as i32;
            forced[i] = token[3] - FORCED_BASE as i32;
            legal[i] = token[4] - LEGAL_BASE as i32;
        }
        (content, status, forced, legal)
    }

    #[test]
    fn token_ids_describe_the_position() {
        let (content, status, forced, legal) = unpacked(&BitState::initial());
        assert_eq!(content, [0; 81]);
        assert_eq!(status, [0; 81]);
        assert_eq!(forced, [0; 81]);
        assert_eq!(legal, [1; 81]);

        let state = x_wins_in_one(); // X to move, boards 0 and 1 won by X, forced to board 2
        let (content, status, forced, legal) = unpacked(&state);
        let at = |b: Square, s: Square| Move::new(b, s).index();
        assert_eq!(content[at(sq(0, 2), sq(0, 0))], 1, "X's own mark");
        assert_eq!(content[at(sq(1, 1), sq(2, 0))], 2, "O's mark");
        assert_eq!(content[at(sq(0, 2), sq(0, 2))], 0);
        assert!(Square::ALL.iter().all(|s| status[at(sq(0, 0), *s)] == 1));
        assert!(Square::ALL.iter().all(|s| status[at(sq(1, 1), *s)] == 0));
        assert_eq!(forced.iter().sum::<i32>(), 9);
        assert!(Square::ALL.iter().all(|s| forced[at(sq(0, 2), *s)] == 1));
        assert_eq!(legal.iter().sum::<i32>(), 7);
        assert_eq!(legal[at(sq(0, 2), sq(0, 2))], 1);
        assert_eq!(legal[at(sq(0, 2), sq(0, 0))], 0);
    }

    #[test]
    fn predicts_shapes_and_legal_policies() {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let model = small().init(&device);
        let states = [BitState::initial(), x_wins_in_one()];
        let p = predict_host(&model, &states, &device);
        assert_eq!(p.wdl.len(), 2);
        assert_eq!(p.policy[0].len(), 81);
        assert_eq!(p.policy[1].len(), 7);
        for (wdl, policy) in p.wdl.iter().zip(&p.policy) {
            assert!((wdl.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            assert!((policy.iter().map(|(_, p)| p).sum::<f32>() - 1.0).abs() < 1e-3);
        }
    }

    /// Identical weights must give identical predictions on every compiled backend.
    #[test]
    fn backends_agree() {
        let _device = crate::nn::device_lock();
        let kinds = BackendKind::for_tests();
        let reference = kinds[kinds.len() - 1].device(); // flex when compiled: not CubeCL
        let model = small().init(&reference);
        let states: Vec<BitState> = synthetic_dataset(16, 5)
            .samples
            .iter()
            .map(|s| s.state)
            .collect();
        let want = predict_host(&model, &states, &reference);
        let dir = std::env::temp_dir().join(format!("uttt-cell-tf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        model.save_file(dir.join("m.bpk")).unwrap();
        for kind in kinds {
            let device = kind.device();
            let loaded = small()
                .init(&device)
                .try_load_file(dir.join("m.bpk"))
                .unwrap()
                .valid();
            let got = predict_host(&loaded, &states, &device);
            for (i, (g, w)) in got.wdl.iter().zip(&want.wdl).enumerate() {
                for c in 0..3 {
                    assert!(
                        (g[c] - w[c]).abs() < 1e-3,
                        "{kind}: wdl row {i}: {g:?} vs {w:?}"
                    );
                }
            }
            for (i, (g, w)) in got.policy.iter().zip(&want.policy).enumerate() {
                for ((mv, gp), (_, wp)) in g.iter().zip(w) {
                    assert!(
                        (gp - wp).abs() < 1e-3,
                        "{kind}: policy row {i} move {mv}: {gp} vs {wp}"
                    );
                }
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "trains a network; run with: cargo test --release -- --include-ignored"]
    fn learns_the_synthetic_dataset() {
        let _device = crate::nn::device_lock();
        let kind = BackendKind::preferred();
        let device = kind.device();
        let data = synthetic_dataset(8000, 4);
        let untrained = small().init(&device);
        let (valid, _) = data.clone().split(0.9, 0);
        let before = evaluate(&untrained, &device, &valid.samples);
        let options = crate::nn::train::TrainOptions {
            batch_size: 64,
            learning_rate: 1e-3,
            ..opts(kind, "cell-tf", 10)
        };
        let (_, report) = train(|d| small().init(d), &device, data, &options);
        assert!(
            report.validation.verdict_accuracy > report.baseline.verdict_accuracy + 0.05
                && report.validation.verdict_accuracy > before.verdict_accuracy,
            "{report:?} (untrained {before:?})"
        );
        assert!(
            report.validation.policy_accuracy > report.baseline.policy_accuracy,
            "{report:?}"
        );
    }

    #[test]
    #[ignore = "trains a network; run with: cargo test --release -- --include-ignored"]
    fn v2_learns_the_synthetic_dataset() {
        let _device = crate::nn::device_lock();
        let kind = BackendKind::preferred();
        let device = kind.device();
        let data = synthetic_dataset(8000, 4);
        let untrained = small_v2().init(&device);
        let (valid, _) = data.clone().split(0.9, 0);
        let before = evaluate(&untrained, &device, &valid.samples);
        let options = crate::nn::train::TrainOptions {
            batch_size: 64,
            learning_rate: 1e-3,
            ..opts(kind, "cell-tf-v2", 20)
        };
        let (_, report) = train(|d| small_v2().init(d), &device, data, &options);
        assert!(
            report.validation.verdict_accuracy > report.baseline.verdict_accuracy + 0.05
                && report.validation.verdict_accuracy > before.verdict_accuracy,
            "{report:?} (untrained {before:?})"
        );
        assert!(
            report.validation.policy_accuracy > report.baseline.policy_accuracy,
            "{report:?}"
        );
    }
}
