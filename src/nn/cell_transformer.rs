//! [`CellTransformer`]: a transformer over the 81 cells of the board.
//!
//! Each cell is one token: an embedding of its content (empty, mine, theirs, from the side to
//! move's view) plus embeddings of its position among the 81 cells, the status of the small
//! board it belongs to (open, won by me, won by them, full), whether that board is the forced
//! one, and whether the cell is a legal move right now. The tokens go through pre-norm encoder
//! layers with all-to-all attention, and two perceptron heads read the flattened final
//! residual stream to produce the move and outcome logits.

use burn::nn::transformer::TransformerEncoder;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;

use super::heads::{BodyConfig, Perceptron, PerceptronConfig, encode_all_to_all, positions};
use super::model::{MOVES, Model, Prediction, Verdict};
use crate::game::{BitState, BoardStatus, GameState, Move, MoveSet, Square};

const CELLS: usize = MOVES;
/// Cell contents: empty, mine, theirs.
const CONTENTS: usize = 3;
/// Board statuses: open, won by me, won by them, full.
const STATUSES: usize = 4;

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

    pub fn init<B: Backend>(&self, device: &B::Device) -> CellTransformer<B> {
        let d = self.body.d_model;
        let embedding = |n: usize| {
            EmbeddingConfig::new(n, d)
                .with_initializer(burn::nn::Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device)
        };
        let head = |out: usize| {
            PerceptronConfig::new(CELLS * d, self.d_head(), out)
                .with_dropout(self.body.dropout)
                .init(device)
        };
        CellTransformer {
            config: self.clone(),
            content: embedding(CONTENTS),
            position: embedding(CELLS),
            status: embedding(STATUSES),
            forced: embedding(2),
            legal: embedding(2),
            body: self.body.init(device),
            policy_head: head(MOVES),
            wdl_head: head(Verdict::ALL.len()),
        }
    }
}

#[derive(Module, Debug)]
pub struct CellTransformer<B: Backend> {
    #[module(skip)]
    config: CellTransformerConfig,
    content: Embedding<B>,
    position: Embedding<B>,
    status: Embedding<B>,
    forced: Embedding<B>,
    legal: Embedding<B>,
    body: TransformerEncoder<B>,
    policy_head: Perceptron<B>,
    wdl_head: Perceptron<B>,
}

/// Per-cell token ids for a batch, each `[batch, 81]` indexed by `Move::index`.
#[derive(Clone, Debug)]
pub struct CellTokens<B: Backend> {
    /// `0` empty, `1` mine, `2` theirs.
    pub content: Tensor<B, 2, Int>,
    /// Status of the cell's board: `0` open, `1` won by me, `2` won by them, `3` full.
    pub status: Tensor<B, 2, Int>,
    /// `1` if the cell's board is the forced board.
    pub forced: Tensor<B, 2, Int>,
    /// `1` if playing this cell is legal now.
    pub legal: Tensor<B, 2, Int>,
}

/// Token ids of one position, from the side to move's view: `(content, status, forced, legal)`.
pub fn cell_token_ids(
    state: &BitState,
) -> ([i32; CELLS], [i32; CELLS], [i32; CELLS], [i32; CELLS]) {
    let me = state.to_move();
    let them = me.other();
    let forced_board = state.forced_board();
    let legal_moves = state.legal_moves();
    let mut content = [0; CELLS];
    let mut status = [0; CELLS];
    let mut forced = [0; CELLS];
    let mut legal = [0; CELLS];
    for (b, board) in Square::ALL.iter().enumerate() {
        let mine = state.board_marks(me, b);
        let theirs = state.board_marks(them, b);
        let board_status = match state.board_status(*board) {
            BoardStatus::Open => 0,
            BoardStatus::Won(p) if p == me => 1,
            BoardStatus::Won(_) => 2,
            BoardStatus::Full => 3,
        };
        let board_forced = i32::from(forced_board == Some(*board));
        for (c, square) in Square::ALL.iter().enumerate() {
            let i = Move::new(*board, *square).index();
            content[i] = if mine >> c & 1 != 0 {
                1
            } else if theirs >> c & 1 != 0 {
                2
            } else {
                0
            };
            status[i] = board_status;
            forced[i] = board_forced;
            legal[i] = i32::from(legal_moves.contains(Move::new(*board, *square)));
        }
    }
    (content, status, forced, legal)
}

impl<B: Backend> CellTransformer<B> {
    pub fn config(&self) -> CellTransformerConfig {
        self.config.clone()
    }
}

impl<B: Backend> Model<B> for CellTransformer<B> {
    type Input = CellTokens<B>;

    fn encode(states: &[BitState], device: &B::Device) -> CellTokens<B> {
        let n = states.len();
        let mut content = Vec::with_capacity(n * CELLS);
        let mut status = Vec::with_capacity(n * CELLS);
        let mut forced = Vec::with_capacity(n * CELLS);
        let mut legal = Vec::with_capacity(n * CELLS);
        for state in states {
            let (c, s, f, l) = cell_token_ids(state);
            content.extend_from_slice(&c);
            status.extend_from_slice(&s);
            forced.extend_from_slice(&f);
            legal.extend_from_slice(&l);
        }
        let tensor = |v: Vec<i32>| Tensor::from_data(TensorData::new(v, [n, CELLS]), device);
        CellTokens {
            content: tensor(content),
            status: tensor(status),
            forced: tensor(forced),
            legal: tensor(legal),
        }
    }

    fn predict(&self, input: CellTokens<B>) -> Prediction<B> {
        let device = input.content.device();
        let tokens = self.content.forward(input.content)
            + self.status.forward(input.status)
            + self.forced.forward(input.forced)
            + self.legal.forward(input.legal)
            + self.position.forward(positions(CELLS, &device));
        let stream = encode_all_to_all(&self.body, tokens).flatten::<2>(1, 2);
        Prediction {
            wdl: self.wdl_head.forward(stream.clone()),
            policy: self.policy_head.forward(stream),
        }
    }
}

crate::nn::train::impl_train_steps!(CellTransformer);

#[cfg(test)]
mod tests {
    use burn::Dispatch;

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

    #[test]
    fn token_ids_describe_the_position() {
        let (content, status, forced, legal) = cell_token_ids(&BitState::initial());
        assert_eq!(content, [0; 81]);
        assert_eq!(status, [0; 81]);
        assert_eq!(forced, [0; 81]);
        assert_eq!(legal, [1; 81]);

        let state = x_wins_in_one(); // X to move, boards 0 and 1 won by X, forced to board 2
        let (content, status, forced, legal) = cell_token_ids(&state);
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
        let model = small().init::<Dispatch>(&device);
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
        let reference = kinds[kinds.len() - 1].device(); // ndarray when compiled: not CubeCL
        let model = small().init::<Dispatch>(&reference);
        let states: Vec<BitState> = synthetic_dataset(16, 5)
            .samples
            .iter()
            .map(|s| s.state)
            .collect();
        let want = predict_host(&model, &states, &reference);
        let dir = std::env::temp_dir().join(format!("uttt-cell-tf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        model
            .save_file(dir.join("m"), &crate::nn::any_model::Recorder::new())
            .unwrap();
        for kind in kinds {
            let device = kind.device();
            let loaded = small()
                .init::<Dispatch>(&device)
                .load_file(
                    dir.join("m"),
                    &crate::nn::any_model::Recorder::new(),
                    &device,
                )
                .unwrap();
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
    fn learns_the_synthetic_dataset() {
        let _device = crate::nn::device_lock();
        let kind = BackendKind::preferred();
        let device = kind.device();
        let data = synthetic_dataset(8000, 4);
        let untrained = small().init::<Dispatch>(&device);
        let (valid, _) = data.clone().split(0.9, 0);
        let before = evaluate(&untrained, &device, &valid.samples);
        let options = crate::nn::train::TrainOptions {
            batch_size: 64,
            learning_rate: 1e-3,
            ..opts(kind, "cell-tf", 10)
        };
        let (_, report) = train(|d| small().init::<Dispatch>(d), &device, data, &options);
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
