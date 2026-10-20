//! [`BoardTransformer`]: a transformer over the nine small boards.
//!
//! Each small board is one token: an embedding of its full cell configuration (`3^9` possible
//! contents, from the side to move's view) plus embeddings of its position on the big board,
//! its status (open, won by me, won by them, full) and whether it is the forced board. The
//! tokens go through pre-norm encoder layers with all-to-all attention, and two perceptron
//! heads read the flattened final residual stream to produce the move and outcome logits.

use burn::nn::transformer::TransformerEncoder;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;

use super::heads::{BodyConfig, Perceptron, PerceptronConfig, encode_all_to_all, positions};
use super::model::{MOVES, Model, Prediction, Verdict};
use crate::game::{BitState, BoardStatus, GameState, Square};

/// Number of small-board contents: each of 9 cells is empty, mine or theirs.
const CONTENTS: usize = 3usize.pow(9);
const BOARDS: usize = 9;
/// Board statuses: open, won by me, won by them, full.
const STATUSES: usize = 4;

#[derive(Config, Debug, PartialEq)]
pub struct BoardTransformerConfig {
    pub body: BodyConfig,
    /// Hidden width of the two heads; `0` means `2 * d_model`.
    #[config(default = 0)]
    pub d_head: usize,
}

impl Default for BoardTransformerConfig {
    fn default() -> Self {
        Self::new(BodyConfig::new())
    }
}

impl BoardTransformerConfig {
    fn d_head(&self) -> usize {
        if self.d_head == 0 {
            2 * self.body.d_model
        } else {
            self.d_head
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BoardTransformer<B> {
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
            PerceptronConfig::new(BOARDS * d, self.d_head(), out)
                .with_dropout(self.body.dropout)
                .init(device)
        };
        BoardTransformer {
            config: self.clone(),
            content: embedding(CONTENTS),
            position: embedding(BOARDS),
            status: embedding(STATUSES),
            forced: embedding(2),
            body: self.body.init(device),
            policy_head: head(MOVES),
            wdl_head: head(Verdict::ALL.len()),
        }
    }
}

#[derive(Module, Debug)]
pub struct BoardTransformer<B: Backend> {
    #[module(skip)]
    config: BoardTransformerConfig,
    content: Embedding<B>,
    position: Embedding<B>,
    status: Embedding<B>,
    forced: Embedding<B>,
    body: TransformerEncoder<B>,
    policy_head: Perceptron<B>,
    wdl_head: Perceptron<B>,
}

/// Per-board token ids for a batch, each `[batch, 9]`.
#[derive(Clone, Debug)]
pub struct BoardTokens<B: Backend> {
    /// Cell configuration index in `0..3^9`.
    pub content: Tensor<B, 2, Int>,
    /// `0` open, `1` won by the side to move, `2` won by the opponent, `3` full.
    pub status: Tensor<B, 2, Int>,
    /// `1` if this is the board the side to move must play on.
    pub forced: Tensor<B, 2, Int>,
}

/// Token ids of one position, from the side to move's view.
pub fn board_token_ids(state: &BitState) -> ([i32; BOARDS], [i32; BOARDS], [i32; BOARDS]) {
    let me = state.to_move();
    let them = me.other();
    let forced_board = state.forced_board();
    let mut content = [0; BOARDS];
    let mut status = [0; BOARDS];
    let mut forced = [0; BOARDS];
    for (b, square) in Square::ALL.iter().enumerate() {
        let mine = state.board_marks(me, b);
        let theirs = state.board_marks(them, b);
        let mut index = 0i32;
        for cell in (0..9).rev() {
            let value = if mine >> cell & 1 != 0 {
                1
            } else if theirs >> cell & 1 != 0 {
                2
            } else {
                0
            };
            index = index * 3 + value;
        }
        content[b] = index;
        status[b] = match state.board_status(*square) {
            BoardStatus::Open => 0,
            BoardStatus::Won(p) if p == me => 1,
            BoardStatus::Won(_) => 2,
            BoardStatus::Full => 3,
        };
        forced[b] = i32::from(forced_board == Some(*square));
    }
    (content, status, forced)
}

impl<B: Backend> BoardTransformer<B> {
    pub fn config(&self) -> BoardTransformerConfig {
        self.config.clone()
    }
}

impl<B: Backend> Model<B> for BoardTransformer<B> {
    type Input = BoardTokens<B>;

    fn encode(states: &[BitState], device: &B::Device) -> BoardTokens<B> {
        let n = states.len();
        let mut content = Vec::with_capacity(n * BOARDS);
        let mut status = Vec::with_capacity(n * BOARDS);
        let mut forced = Vec::with_capacity(n * BOARDS);
        for state in states {
            let (c, s, f) = board_token_ids(state);
            content.extend_from_slice(&c);
            status.extend_from_slice(&s);
            forced.extend_from_slice(&f);
        }
        let tensor = |v: Vec<i32>| Tensor::from_data(TensorData::new(v, [n, BOARDS]), device);
        BoardTokens {
            content: tensor(content),
            status: tensor(status),
            forced: tensor(forced),
        }
    }

    fn predict(&self, input: BoardTokens<B>) -> Prediction<B> {
        let device = input.content.device();
        let tokens = self.content.forward(input.content)
            + self.status.forward(input.status)
            + self.forced.forward(input.forced)
            + self.position.forward(positions(BOARDS, &device));
        let stream = encode_all_to_all(&self.body, tokens).flatten::<2>(1, 2);
        Prediction {
            wdl: self.wdl_head.forward(stream.clone()),
            policy: self.policy_head.forward(stream),
        }
    }
}

crate::nn::train::impl_train_steps!(BoardTransformer);

#[cfg(test)]
mod tests {
    use burn::Dispatch;

    use super::*;
    use crate::game::test_positions::x_wins_in_one;
    use crate::game::{Move, sq};
    use crate::nn::BackendKind;
    use crate::nn::model::predict_host;
    use crate::nn::train::tests::{opts, synthetic_dataset};
    use crate::nn::train::{evaluate, train};

    fn small() -> BoardTransformerConfig {
        BoardTransformerConfig::new(
            BodyConfig::new()
                .with_d_model(32)
                .with_n_layers(1)
                .with_n_heads(2),
        )
    }

    #[test]
    fn token_ids_describe_the_position() {
        let (content, status, forced) = board_token_ids(&BitState::initial());
        assert_eq!(content, [0; 9]);
        assert_eq!(status, [0; 9]);
        assert_eq!(forced, [0; 9]);

        let state = x_wins_in_one(); // X to move, boards 0 and 1 won by X, forced to board 2
        let (content, status, forced) = board_token_ids(&state);
        assert_eq!(status[0], 1);
        assert_eq!(status[1], 1);
        assert_eq!(status[4], 0);
        assert_eq!(forced, [0, 0, 1, 0, 0, 0, 0, 0, 0]);
        // Board 2 holds X at cells 0 and 1 (mine = 1): 1 * 3^0 + 1 * 3^1.
        assert_eq!(content[2], 4);
        // Board 4 holds O (theirs = 2) at cells 6, 7, 2: 2 * (3^6 + 3^7 + 3^2).
        assert_eq!(content[4], 2 * (729 + 2187 + 9));
        assert!(content.iter().all(|&c| (c as usize) < CONTENTS));

        // Same position from O's view: X's boards are "theirs", its marks are 2s.
        let mut p = state.to_position();
        p.set_to_move(crate::game::Player::O);
        let (content_o, status_o, _) = board_token_ids(&BitState::from_position(&p));
        assert_eq!(status_o[0], 2);
        assert_eq!(content_o[2], 8);
        let _ = Move::new(sq(0, 0), sq(0, 0));
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
        let dir = std::env::temp_dir().join(format!("uttt-board-tf-{}", std::process::id()));
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
        let data = synthetic_dataset(1500, 4);
        let untrained = small().init::<Dispatch>(&device);
        let (valid, _) = data.clone().split(0.9, 0);
        let before = evaluate(&untrained, &device, &valid.samples);
        let options = crate::nn::train::TrainOptions {
            batch_size: 64,
            learning_rate: 1e-3,
            ..opts(kind, "board-tf", 6)
        };
        let (_, report) = train(|d| small().init::<Dispatch>(d), &device, data, &options);
        assert!(
            report.validation.verdict_accuracy > report.baseline.verdict_accuracy
                && report.validation.verdict_accuracy > before.verdict_accuracy,
            "{report:?} (untrained {before:?})"
        );
        assert!(
            report.validation.policy_accuracy > report.baseline.policy_accuracy,
            "{report:?}"
        );
    }
}
