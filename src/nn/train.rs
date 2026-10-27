//! Self-play data generation and policy/value training with burn's `Learner`.

use std::marker::PhantomData;
use std::path::PathBuf;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset as BurnDataset, DatasetError, InMemDataset};
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::train::logger::{EvaluationProgressLogger, TrainingProgressLogger};
use burn::train::metric::{AccuracyMetric, LossMetric, MetricDefinition};
use burn::train::renderer::{
    EvaluationName,
    MetricState,
    MetricsRenderer,
    MetricsRendererEvaluation,
    MetricsRendererTraining,
};
use burn::train::{
    ClassificationOutput,
    ExecutionStrategy,
    InferenceStep,
    Learner,
    SupervisedTraining,
    TrainStep,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::linear::LinearNet;
use super::model::{MOVES, Model, Verdict, expected_value, legal_mask, most_likely, predict_host};
use crate::agent::{AgentSpec, UtttAgent};
use crate::game::{BitState, GameState, Move, Outcome, Symmetry};

/// How training data is multiplied by the eight board symmetries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Augment {
    /// Positions as played only.
    None,
    /// Every sample under all eight symmetries: an epoch sees each eight times, transformed
    /// on access. Right for small datasets (self-play generations).
    #[default]
    All,
    /// Each sample under one symmetry drawn at random every time it is batched: an epoch
    /// sees the data once and the symmetries over the epochs, in the memory of one copy.
    /// Right for datasets of millions of positions.
    Random,
}

impl std::str::FromStr for Augment {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "all" => Ok(Self::All),
            "random" => Ok(Self::Random),
            other => Err(format!("augmentation {other:?} is not none, all or random")),
        }
    }
}
use crate::selfplay::{SelfPlayConfig, SelfPlayStats, self_play};

/// One position with its targets, all from the side to move's view: a move distribution, and
/// a win/draw/loss distribution and/or a scalar value.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub state: BitState,
    /// Probabilities over the legal moves; a search agent's visit distribution, or a single
    /// move with probability 1.
    pub policy: Vec<(Move, f32)>,
    /// Which game the position came from. Splits keep a game's positions together, since they
    /// share the outcome and a model could otherwise memorise trajectories.
    pub game: usize,
    /// Win/draw/loss target: the game's outcome as a one-hot, or a search's win/draw/loss
    /// frequencies. `None` when only a scalar value is known.
    pub wdl: Option<[f32; 3]>,
    /// A search's root value in `[-1, 1]`: a lower-variance value target than the outcome
    /// alone. At least one of `wdl` and `search_value` is set.
    pub search_value: Option<f32>,
}

impl Sample {
    /// A position of a finished game: the outcome as the win/draw/loss target.
    pub fn from_outcome(
        state: BitState,
        policy: Vec<(Move, f32)>,
        game: usize,
        outcome: Outcome,
        search_value: Option<f32>,
    ) -> Self {
        Self {
            wdl: Some(Verdict::from_outcome(outcome, state.to_move()).one_hot()),
            state,
            policy,
            game,
            search_value,
        }
    }

    /// The same sample seen through a board symmetry: the position and every move of the
    /// policy move with it, the targets and game id stay.
    pub fn transform(&self, symmetry: Symmetry) -> Sample {
        Sample {
            state: self.state.transform(symmetry),
            policy: self
                .policy
                .iter()
                .map(|(mv, p)| (symmetry.apply_move(*mv), *p))
                .collect(),
            game: self.game,
            wdl: self.wdl,
            search_value: self.search_value,
        }
    }

    /// The most likely verdict of the win/draw/loss target, if there is one.
    pub fn verdict(&self) -> Option<Verdict> {
        self.wdl.as_ref().map(most_likely)
    }

    /// The scalar value target: the search's root value when there is one, else the
    /// expectation of the win/draw/loss target.
    pub fn value_target(&self) -> f32 {
        match (self.search_value, &self.wdl) {
            (Some(v), _) => v,
            (None, Some(wdl)) => expected_value(wdl),
            (None, None) => panic!("a sample needs a win/draw/loss or a scalar value target"),
        }
    }

    /// The verdict class for the accuracy metric: the win/draw/loss argmax, else the scalar
    /// value's sign with a dead zone.
    fn verdict_class(&self) -> Verdict {
        self.verdict().unwrap_or_else(|| {
            let v = self.value_target();
            if v > 1.0 / 3.0 {
                Verdict::Win
            } else if v < -1.0 / 3.0 {
                Verdict::Loss
            } else {
                Verdict::Draw
            }
        })
    }

    /// The target move: the most probable one.
    pub fn best_move(&self) -> Move {
        self.policy
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(mv, _)| *mv)
            .expect("policy has a move")
    }
}

/// A set of samples.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    /// Number of distinct game ids.
    pub fn games(&self) -> usize {
        self.samples
            .iter()
            .map(|s| s.game)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// How many samples with a win/draw/loss target have each [`Verdict`] as the most likely,
    /// in `Verdict::ALL` order.
    pub fn verdict_counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for s in &self.samples {
            if let Some(v) = s.verdict() {
                counts[v.index()] += 1;
            }
        }
        counts
    }

    /// Accuracy of always predicting the most common verdict, over the samples that have one.
    pub fn baseline_accuracy(&self) -> f32 {
        let counts = self.verdict_counts();
        let total: usize = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        *counts.iter().max().unwrap() as f32 / total as f32
    }

    /// Mean squared error (on the `1 / 0 / -1` value scale) of always predicting the mean
    /// value target.
    pub fn baseline_mse(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let n = self.len() as f32;
        let mean = self.samples.iter().map(Sample::value_target).sum::<f32>() / n;
        self.samples
            .iter()
            .map(|s| (s.value_target() - mean).powi(2))
            .sum::<f32>()
            / n
    }

    /// Expected accuracy of picking a legal move uniformly at random.
    pub fn baseline_policy_accuracy(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        self.samples
            .iter()
            .map(|s| 1.0 / s.state.legal_moves_vec().len() as f32)
            .sum::<f32>()
            / self.len() as f32
    }

    /// Splits off about `fraction` of the positions as validation data, whole games at a
    /// time (games are shuffled with `seed`). When there are at least two games, each part
    /// gets at least one.
    pub fn split(self, fraction: f32, seed: u64) -> (Dataset, Dataset) {
        let mut games: Vec<usize> = self.samples.iter().map(|s| s.game).collect();
        games.sort_unstable();
        games.dedup();
        let mut rng = SmallRng::seed_from_u64(seed);
        games.shuffle(&mut rng);
        let n_val = ((games.len() as f32 * fraction).round() as usize)
            .max(usize::from(games.len() >= 2))
            .min(games.len().saturating_sub(1));
        let validation_games: std::collections::HashSet<usize> =
            games.iter().take(n_val).copied().collect();
        let (valid, train): (Vec<Sample>, Vec<Sample>) = self
            .samples
            .into_iter()
            .partition(|s| validation_games.contains(&s.game));
        (Dataset { samples: train }, Dataset { samples: valid })
    }
}

/// Plays `games` games of `spec` against itself and records every position with the final
/// result and the agent's move distribution. `on_game` is called after each game with the
/// number played so far.
pub fn generate(
    spec: &AgentSpec,
    games: usize,
    mut on_game: impl FnMut(usize),
) -> Result<Dataset, String> {
    let mut agent = spec.build()?;
    let mut data = Dataset::default();
    let mut positions: Vec<(BitState, Vec<(Move, f32)>)> = Vec::new();
    for game in 0..games {
        positions.clear();
        let mut state = BitState::initial();
        let outcome = loop {
            if let Some(outcome) = state.outcome() {
                break outcome;
            }
            let (mv, policy) = agent.choose_move_with_policy(&state);
            positions.push((state, policy));
            state.apply(mv);
        };
        for (state, policy) in positions.drain(..) {
            data.samples
                .push(Sample::from_outcome(state, policy, game, outcome, None));
        }
        on_game(game + 1);
    }
    Ok(data)
}

/// Plays `games` games of search with `evaluator` as the leaf evaluator, many at once so the
/// evaluator sees large batches (see [`crate::selfplay`]), and records every position with the
/// final result and the search's move distribution.
pub fn generate_parallel(
    evaluator: &AgentSpec,
    games: usize,
    cfg: &SelfPlayConfig,
    on_game: impl FnMut(usize),
) -> Result<(Dataset, SelfPlayStats), String> {
    let mut evaluator = evaluator.build()?;
    let (records, stats) = self_play(&mut evaluator, games, cfg, on_game);
    let mut data = Dataset::default();
    for (game, record) in records.into_iter().enumerate() {
        for (state, policy, value) in record.positions {
            data.samples.push(Sample::from_outcome(
                state,
                policy,
                game,
                record.outcome,
                Some(value),
            ));
        }
    }
    Ok((data, stats))
}

/// Every sample of a [`Dataset`] under all eight board symmetries, as a burn dataset eight
/// times as long ([`Augment::All`]): with `n` samples, item `i` is sample `i % n` under
/// symmetry `i / n` (the identity first), transformed on access, so the memory is that of
/// one copy.
pub struct Augmented {
    samples: Vec<Sample>,
}

impl Augmented {
    pub fn new(data: Dataset) -> Self {
        Self {
            samples: data.samples,
        }
    }
}

impl BurnDataset<Sample> for Augmented {
    fn get(&self, index: usize) -> Result<Sample, DatasetError> {
        let n = self.samples.len();
        match Symmetry::ALL.get(index / n.max(1)) {
            Some(&symmetry) => Ok(self.samples[index % n].transform(symmetry)),
            None => panic!(
                "index out of bounds for Augmented: {index} >= {}",
                self.len()
            ),
        }
    }

    fn len(&self) -> usize {
        self.samples.len() * Symmetry::ALL.len()
    }
}

/// A minibatch on a device, encoded for model `M`.
pub struct Batch<M: Model> {
    pub input: M::Input,
    /// `[batch, 3]` win/draw/loss targets; zeros where the sample has none.
    pub wdl: Tensor<2>,
    /// `[batch, 1]` one where the sample has a win/draw/loss target, else zero.
    pub wdl_mask: Tensor<2>,
    /// `[batch]` verdict classes, for the accuracy metric (see [`Sample::verdict_class`]).
    pub verdicts: Tensor<1, Int>,
    /// `[batch, 1]` scalar value targets (see [`Sample::value_target`]).
    pub values: Tensor<2>,
    /// Weight of the win/draw/loss cross-entropy in the loss.
    pub outcome_weight: f32,
    /// Weight of the squared error against `values` in the loss.
    pub search_value_weight: f32,
    /// `[batch, MOVES]` target move probabilities.
    pub policy: Tensor<2>,
    /// `[batch, MOVES]` legal moves.
    pub legal: Tensor<2, Bool>,
}

impl<M: Model> std::fmt::Debug for Batch<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batch")
            .field("input", &self.input)
            .field("wdl", &self.wdl)
            .field("values", &self.values)
            .field("policy", &self.policy)
            .field("legal", &self.legal)
            .finish()
    }
}

impl<M: Model> Clone for Batch<M> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            wdl: self.wdl.clone(),
            wdl_mask: self.wdl_mask.clone(),
            verdicts: self.verdicts.clone(),
            values: self.values.clone(),
            outcome_weight: self.outcome_weight,
            search_value_weight: self.search_value_weight,
            policy: self.policy.clone(),
            legal: self.legal.clone(),
        }
    }
}

/// Turns samples into a [`Batch`] for model `M`, carrying the loss weights. With `random_symmetry`
/// each sample is seen through a board symmetry drawn at random every time it is batched
/// ([`Augment::Random`]).
pub struct ModelBatcher<M> {
    pub outcome_weight: f32,
    pub search_value_weight: f32,
    pub random_symmetry: bool,
    _model: PhantomData<fn() -> M>,
}

impl<M> ModelBatcher<M> {
    pub fn new(outcome_weight: f32, search_value_weight: f32, random_symmetry: bool) -> Self {
        Self {
            outcome_weight,
            search_value_weight,
            random_symmetry,
            _model: PhantomData,
        }
    }
}

impl<M> Default for ModelBatcher<M> {
    fn default() -> Self {
        Self::new(1.0, 0.0, false)
    }
}

impl<M> Clone for ModelBatcher<M> {
    fn clone(&self) -> Self {
        Self::new(
            self.outcome_weight,
            self.search_value_weight,
            self.random_symmetry,
        )
    }
}

impl<M: Model> Batcher<Sample, Batch<M>> for ModelBatcher<M> {
    fn batch(&self, items: Vec<Sample>, device: &Device) -> Batch<M> {
        let items: Vec<Sample> = if self.random_symmetry {
            let mut rng = rand::rng();
            items
                .into_iter()
                .map(|s| {
                    let sym = Symmetry::ALL[rng.random_range(0..Symmetry::ALL.len())];
                    s.transform(sym)
                })
                .collect()
        } else {
            items
        };
        let n = items.len();
        let states: Vec<BitState> = items.iter().map(|s| s.state).collect();
        let mut wdl = vec![0.0f32; n * 3];
        let mut wdl_mask = vec![0.0f32; n];
        let mut policy = vec![0.0f32; n * MOVES];
        for (i, item) in items.iter().enumerate() {
            if let Some(target) = &item.wdl {
                wdl[i * 3..i * 3 + 3].copy_from_slice(target);
                wdl_mask[i] = 1.0;
            }
            for (mv, p) in &item.policy {
                policy[i * MOVES + mv.index()] = *p;
            }
        }
        let verdicts: Vec<i32> = items
            .iter()
            .map(|s| s.verdict_class().index() as i32)
            .collect();
        let values: Vec<f32> = items.iter().map(Sample::value_target).collect();
        Batch {
            input: M::encode(&states, device),
            wdl: Tensor::from_data(TensorData::new(wdl, [n, 3]), device),
            wdl_mask: Tensor::from_data(TensorData::new(wdl_mask, [n, 1]), device),
            verdicts: Tensor::from_data(TensorData::new(verdicts, [n]), device),
            values: Tensor::from_data(TensorData::new(values, [n, 1]), device),
            outcome_weight: self.outcome_weight,
            search_value_weight: self.search_value_weight,
            policy: Tensor::from_data(TensorData::new(policy, [n, MOVES]), device),
            legal: legal_mask(&states, device),
        }
    }
}

/// Output of one training or validation step: the summed loss, the `[batch, 3]` verdict
/// logits and the `[batch]` verdict targets, which burn's loss and accuracy metrics read.
pub type StepOutput = ClassificationOutput;

/// Forward pass plus the loss: the weighted cross-entropy between the win/draw/loss target
/// and the verdict logits (averaged over the samples that have a target), the weighted squared error
/// of the expected value (`p_win - p_loss`) against the scalar value target, and the
/// cross-entropy between the target move distribution and the legal-masked policy.
pub fn forward_step<M: Model>(model: &M, batch: Batch<M>) -> StepOutput {
    let prediction = model.predict(batch.input);
    let n = batch.verdicts.dims()[0];
    // Averaged over the positions that have a win/draw/loss target, so that the term keeps
    // its weight when only some of a batch carry one.
    let labelled = batch.wdl_mask.clone().sum().clamp_min(1.0);
    let wdl_loss = (-(log_softmax(prediction.wdl.clone(), 1) * batch.wdl).sum_dim(1)
        * batch.wdl_mask)
        .sum()
        .div(labelled)
        .mul_scalar(batch.outcome_weight);
    let probabilities = softmax(prediction.wdl.clone(), 1);
    let expected = probabilities.clone().slice([0..n, 0..1]) - probabilities.slice([0..n, 2..3]);
    let value_loss = (expected - batch.values)
        .powf_scalar(2.0)
        .mean()
        .mul_scalar(batch.search_value_weight);
    let policy_loss = -(prediction.policy_log_probabilities(batch.legal) * batch.policy)
        .sum_dim(1)
        .mean();
    let loss = wdl_loss + value_loss + policy_loss;
    ClassificationOutput::new(loss, prediction.wdl, batch.verdicts)
}

/// Implements burn's step traits for a model type.
macro_rules! impl_train_steps {
    ($model:ident) => {
        impl ::burn::train::TrainStep for $model {
            type Input = $crate::nn::train::Batch<$model>;
            type Output = $crate::nn::train::StepOutput;

            fn step(
                &self,
                batch: Self::Input,
            ) -> ::burn::train::TrainOutput<$crate::nn::train::StepOutput> {
                let item = $crate::nn::train::forward_step(self, batch);
                ::burn::train::TrainOutput::new(self, item.loss.backward(), item)
            }
        }

        impl ::burn::train::InferenceStep for $model {
            type Input = $crate::nn::train::Batch<$model>;
            type Output = $crate::nn::train::StepOutput;

            fn step(&self, batch: Self::Input) -> $crate::nn::train::StepOutput {
                $crate::nn::train::forward_step(self, batch)
            }
        }
    };
}
pub(crate) use impl_train_steps;

impl_train_steps!(LinearNet);

#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Cosine-anneal the learning rate from `learning_rate` down to this over the whole run
    /// (every step of every epoch); `0` keeps it constant.
    pub lr_min: f64,
    /// AdamW weight decay.
    pub weight_decay: f32,
    /// Fraction of positions held out for validation.
    pub validation_fraction: f32,
    /// Symmetry augmentation of the training split. Validation is never augmented.
    pub augment: Augment,
    /// Weight of the win/draw/loss cross-entropy in the loss; `0` with a search-value weight
    /// trains the value head purely on scalar values (distillation).
    pub outcome_weight: f32,
    /// Weight of the squared error between the expected value and the scalar value target
    /// (a search's root value, or the outcome's value where there was no search) in the
    /// loss; `0` ignores it.
    pub search_value_weight: f32,
    /// Data loader threads that encode batches; `0` means one per core.
    pub workers: usize,
    pub seed: u64,
    /// Where the Learner writes its logs and checkpoints.
    pub artifact_dir: PathBuf,
    /// Show the live metrics dashboard (needs an interactive terminal); otherwise log lines.
    pub tui: bool,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 1024,
            learning_rate: 1e-3,
            lr_min: 0.0,
            weight_decay: 1e-4,
            validation_fraction: 0.1,
            augment: Augment::All,
            outcome_weight: 1.0,
            search_value_weight: 0.5,
            workers: 0,
            seed: 0,
            artifact_dir: PathBuf::from("artifacts/train"),
            tui: true,
        }
    }
}

impl TrainOptions {
    fn workers(&self) -> usize {
        match self.workers {
            0 => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            n => n,
        }
    }
}

/// Quality of a model over a set of samples.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evaluation {
    /// Fraction of positions with a win/draw/loss target whose most likely verdict is the
    /// target's.
    pub verdict_accuracy: f32,
    /// MSE of `p_win - p_loss` against the scalar value target.
    pub value_mse: f32,
    /// Fraction of positions whose most likely move is the target's most likely move.
    pub policy_accuracy: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainReport {
    pub train_positions: usize,
    pub validation_positions: usize,
    /// Epochs run.
    pub epochs: usize,
    /// The epoch whose checkpoint the returned model comes from: the one with the lowest
    /// validation loss when checkpoints were kept, otherwise the last.
    pub best_epoch: usize,
    /// Always predicting the most common verdict, the mean value, and a uniformly random
    /// legal move, over the validation split.
    pub baseline: Evaluation,
    /// The trained model over the validation split.
    pub validation: Evaluation,
}

/// A renderer that shows nothing: callers print what they need from the [`TrainReport`], and
/// the Learner's metric logs under the artifact directory keep the full curves.
struct QuietRenderer;

impl MetricsRendererTraining for QuietRenderer {
    fn update_train(&mut self, _state: MetricState) {}
    fn update_valid(&mut self, _state: MetricState) {}
}

impl MetricsRendererEvaluation for QuietRenderer {
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}
}

impl TrainingProgressLogger for QuietRenderer {
    fn start(
        &mut self,
        _total_epochs: usize,
        _starting_epoch: usize,
        _total_items: Option<usize>,
        _label: Option<&str>,
    ) {
    }
    fn update_epoch(&mut self, _epoch: usize) {}
    fn start_split(&mut self, _split: &str, _total_items: usize) {}
    fn update_split(&mut self, _items_processed: usize) {}
    fn end_split(&mut self) {}
    fn end(&mut self) {}
    fn log_event_training(&mut self, _event: String) {}
}

impl EvaluationProgressLogger for QuietRenderer {
    fn start_global_progress(&mut self, _total_tests: usize) {}
    fn start_test(&mut self, _name: &str, _total_items: usize) {}
    fn update_test_progress(&mut self, _items_processed: usize) {}
    fn end_test(&mut self) {}
    fn end_global_progress(&mut self) {}
    fn log_event_evaluation(&mut self, _event: String) {}
}

impl MetricsRenderer for QuietRenderer {
    fn manual_close(&mut self) {}
    fn register_metric(&mut self, _definition: MetricDefinition) {}
}

/// Models that can go through [`train`].
pub trait Trainable:
    Model
    + Module
    + TrainStep<Input = Batch<Self>, Output = StepOutput>
    + InferenceStep<Input = Batch<Self>, Output = StepOutput>
    + std::fmt::Display
    + 'static
{
}

impl<M> Trainable for M where
    M: Model
        + Module
        + TrainStep<Input = Batch<M>, Output = StepOutput>
        + InferenceStep<Input = Batch<M>, Output = StepOutput>
        + std::fmt::Display
        + 'static
{
}

/// Trains a model built by `init` on the autodiff form of `device` on `data` with burn's
/// `Learner`, rendering progress to the terminal, and returns it ready for inference on
/// `device` together with a report.
pub fn train<M: Trainable>(
    init: impl Fn(&Device) -> M,
    device: &Device,
    data: Dataset,
    opts: &TrainOptions,
) -> (M, TrainReport) {
    assert!(!data.is_empty(), "cannot train on an empty dataset");
    let ad_device = device.clone().autodiff();
    ad_device.seed(opts.seed);

    let (train_data, valid_data) = data.split(opts.validation_fraction, opts.seed);
    let train_positions = match opts.augment {
        Augment::All => train_data.len() * Symmetry::ALL.len(),
        Augment::None | Augment::Random => train_data.len(),
    };
    let report_base = TrainReport {
        train_positions,
        validation_positions: valid_data.len(),
        baseline: Evaluation {
            verdict_accuracy: valid_data.baseline_accuracy(),
            value_mse: valid_data.baseline_mse(),
            policy_accuracy: valid_data.baseline_policy_accuracy(),
        },
        ..Default::default()
    };
    let batch = opts.batch_size.max(1);
    let steps_per_epoch = train_positions.div_ceil(batch);
    let batcher = ModelBatcher::<M>::new(
        opts.outcome_weight,
        opts.search_value_weight,
        opts.augment == Augment::Random,
    );
    let loader_train = DataLoaderBuilder::<Sample, Batch<M>>::new(batcher)
        .batch_size(batch)
        .shuffle(opts.seed)
        .num_workers(opts.workers())
        .set_device(ad_device.clone());
    let loader_train = match opts.augment {
        Augment::All => loader_train.build(Augmented::new(train_data)),
        Augment::None | Augment::Random => {
            loader_train.build(InMemDataset::new(train_data.samples))
        }
    };
    let valid_samples = valid_data.samples;
    let valid_batcher =
        ModelBatcher::<M>::new(opts.outcome_weight, opts.search_value_weight, false);
    let loader_valid = DataLoaderBuilder::<Sample, Batch<M>>::new(valid_batcher)
        .batch_size(1024)
        .num_workers(opts.workers())
        .set_device(device.clone())
        .build(InMemDataset::new(valid_samples.clone()));

    let model = init(&ad_device);
    let optim = AdamWConfig::new()
        .with_weight_decay(opts.weight_decay)
        .init();
    let learner = if opts.lr_min > 0.0 && opts.lr_min < opts.learning_rate {
        let schedule = CosineAnnealingLrSchedulerConfig::new(
            opts.learning_rate,
            (steps_per_epoch * opts.epochs).max(1),
        )
        .with_min_lr(opts.lr_min)
        .init()
        .expect("valid cosine schedule");
        Learner::new(model, optim, schedule)
    } else {
        Learner::new(model, optim, opts.learning_rate)
    };

    let training = SupervisedTraining::new(&opts.artifact_dir, loader_train, loader_valid)
        .num_epochs(opts.epochs)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_training_strategy(ExecutionStrategy::single(ad_device).into())
        .with_default_checkpointers();
    let training = if opts.tui {
        training.summary()
    } else {
        training.renderer(Box::new(QuietRenderer))
    };
    let result = training.launch(learner);
    if let Some(error) = &result.error {
        panic!("training failed: {error}");
    }
    // The trained weights without autodiff, on `device`.
    let mut model = result.model.valid();

    // The Learner returns the last epoch's model; prefer the epoch with the lowest validation
    // loss, whose checkpoint burn keeps, since large models overfit well before the end.
    let mut best_epoch = opts.epochs;
    if let Some(best) =
        best_validation_epoch(&opts.artifact_dir, opts.epochs).filter(|&b| b != opts.epochs)
    {
        let path = opts
            .artifact_dir
            .join("checkpoint")
            .join(format!("model-{best}.bpk"));
        if path.exists() {
            match init(device).try_load_file(&path) {
                Ok(m) => {
                    model = m.valid();
                    best_epoch = best;
                }
                Err(e) => eprintln!("cannot load best checkpoint (epoch {best}): {e}"),
            }
        }
    }

    let validation = evaluate(&model, device, &valid_samples);
    (
        model,
        TrainReport {
            epochs: opts.epochs,
            best_epoch,
            validation,
            ..report_base
        },
    )
}

/// The epoch with the lowest mean validation loss, read from the Learner's metric logs.
fn best_validation_epoch(artifact_dir: &std::path::Path, epochs: usize) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for epoch in 1..=epochs {
        let path = artifact_dir
            .join("valid")
            .join(format!("epoch-{epoch}"))
            .join("Loss.log");
        let text = std::fs::read_to_string(path).ok()?;
        let values: Vec<f64> = text
            .lines()
            .filter_map(|l| l.split(',').next()?.trim().parse().ok())
            .collect();
        if values.is_empty() {
            return None;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        if best.is_none_or(|(_, b)| mean < b) {
            best = Some((epoch, mean));
        }
    }
    best.map(|(e, _)| e)
}

/// Evaluates `model` over `samples`; zeros if there are none.
pub fn evaluate<M: Model>(model: &M, device: &Device, samples: &[Sample]) -> Evaluation {
    if samples.is_empty() {
        return Evaluation::default();
    }
    let mut verdict_hits = 0usize;
    let mut verdict_total = 0usize;
    let mut policy_hits = 0usize;
    let mut value_err = 0.0f64;
    let mut states = Vec::new();
    for chunk in samples.chunks(1024) {
        states.clear();
        states.extend(chunk.iter().map(|s| s.state));
        let prediction = predict_host(model, &states, device);
        let best_moves = prediction.best_moves();
        for (i, s) in chunk.iter().enumerate() {
            let p = &prediction.wdl[i];
            if let Some(target) = s.verdict() {
                verdict_total += 1;
                verdict_hits += usize::from(most_likely(p) == target);
            }
            policy_hits += usize::from(best_moves[i] == s.best_move());
            value_err += (expected_value(p) - s.value_target()).powi(2) as f64;
        }
    }
    let n = samples.len();
    Evaluation {
        verdict_accuracy: if verdict_total == 0 {
            0.0
        } else {
            verdict_hits as f32 / verdict_total as f32
        },
        value_mse: (value_err / n as f64) as f32,
        policy_accuracy: policy_hits as f32 / n as f32,
    }
}

/// A fresh temporary artifact directory for one run (tests, throwaway runs).
pub fn temp_artifact_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("uttt-train-{tag}-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::game::MoveSet;
    use crate::nn::BackendKind;
    use crate::nn::encode::{FEATURES, encode_into};
    use crate::nn::linear::LinearNetConfig;

    /// Options for tests: no dashboard, a temporary artifact directory, and no symmetry
    /// augmentation, because the synthetic labels are not symmetric (the self-play test turns
    /// it back on).
    pub fn opts(kind: BackendKind, tag: &str, epochs: usize) -> TrainOptions {
        TrainOptions {
            epochs,
            tui: false,
            augment: Augment::None,
            workers: 1,
            artifact_dir: temp_artifact_dir(&format!("{kind}-{tag}")),
            ..Default::default()
        }
    }

    /// Positions from random play, labelled by a hidden linear model of the flat features:
    /// the verdict is the argmax of three linear scores, the policy is one-hot on the legal
    /// move with the highest linear score. Any model that can represent a linear function of
    /// the board must fit this.
    pub fn synthetic_dataset(n: usize, seed: u64) -> Dataset {
        let mut rng = SmallRng::seed_from_u64(seed);
        let w_wdl: Vec<f32> = (0..FEATURES * 3)
            .map(|_| rng.random_range(-0.3..0.3))
            .collect();
        let w_policy: Vec<f32> = (0..FEATURES * MOVES)
            .map(|_| rng.random_range(-0.3..0.3))
            .collect();
        let mut data = Dataset::default();
        let mut features = vec![0.0; FEATURES];
        let mut game = 0;
        while data.len() < n {
            let mut state = BitState::initial();
            game += 1;
            while state.outcome().is_none() && data.len() < n {
                encode_into(&state, &mut features);
                let mut wdl = [0.0f32; 3];
                let mut policy = [0.0f32; MOVES];
                for (i, x) in features.iter().enumerate() {
                    if *x == 0.0 {
                        continue;
                    }
                    for (c, s) in wdl.iter_mut().enumerate() {
                        *s += x * w_wdl[i * 3 + c];
                    }
                    for (m, s) in policy.iter_mut().enumerate() {
                        *s += x * w_policy[i * MOVES + m];
                    }
                }
                let verdict = (0..3).max_by(|&a, &b| wdl[a].total_cmp(&wdl[b])).unwrap();
                let legal = state.legal_moves();
                let best = legal
                    .iter()
                    .max_by(|a, b| policy[a.index()].total_cmp(&policy[b.index()]))
                    .unwrap();
                data.samples.push(Sample {
                    state,
                    policy: vec![(best, 1.0)],
                    game,
                    wdl: Some(Verdict::from_index(verdict).one_hot()),
                    search_value: None,
                });
                let k = rng.random_range(0..legal.len());
                state.apply(legal.nth(k).unwrap());
            }
        }
        data
    }

    #[test]
    fn generate_records_every_position_with_result_and_policy() {
        let data = generate(
            &AgentSpec::Mcts {
                simulations: 8,
                seed: 1,
                book: None,
            },
            2,
            |_| {},
        )
        .unwrap();
        assert!(data.len() >= 2 * 17, "{} positions", data.len());
        for s in &data.samples {
            let sum: f32 = s.policy.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-4);
            assert!(s.policy.iter().all(|(mv, _)| s.state.is_legal(*mv)));
            assert!(s.wdl.is_some());
        }
        // Consecutive positions of one game alternate perspective, so values alternate (or
        // are all draws).
        let (a, b) = (
            data.samples[0].value_target(),
            data.samples[1].value_target(),
        );
        assert!(a == -b);
        assert_eq!(data.verdict_counts().iter().sum::<usize>(), data.len());
        assert!(data.baseline_policy_accuracy() > 0.0);
        assert_eq!(data.samples.first().unwrap().game, 0);
        assert_eq!(data.samples.last().unwrap().game, 1);
    }

    #[test]
    fn value_targets_and_verdict_classes() {
        let s = Sample {
            state: BitState::initial(),
            policy: vec![(Move::from_index(40), 1.0)],
            game: 0,
            wdl: Some([0.6, 0.3, 0.1]),
            search_value: None,
        };
        assert_eq!(s.verdict(), Some(Verdict::Win));
        assert!((s.value_target() - 0.5).abs() < 1e-6);
        let scalar = Sample {
            wdl: None,
            search_value: Some(-0.8),
            ..s.clone()
        };
        assert_eq!(scalar.verdict(), None);
        assert_eq!(scalar.verdict_class(), Verdict::Loss);
        assert_eq!(scalar.value_target(), -0.8);
        let both = Sample {
            search_value: Some(0.1),
            ..s
        };
        assert_eq!(both.value_target(), 0.1);
        assert_eq!(both.verdict_class(), Verdict::Win);
    }

    #[test]
    fn symmetries_map_positions_and_policies_together() {
        let data = generate(
            &AgentSpec::Mcts {
                simulations: 8,
                seed: 2,
                book: None,
            },
            1,
            |_| {},
        )
        .unwrap();
        let aug = Augmented::new(data.clone());
        assert_eq!(aug.len(), 8 * data.len());
        let n = data.len();
        for (i, original) in data.samples.iter().enumerate() {
            assert_eq!(aug.get(i).as_ref().ok(), Some(original), "identity first");
            assert_eq!(
                aug.get(n + i).ok(),
                Some(original.transform(Symmetry::ALL[1]))
            );
            for sym in Symmetry::ALL {
                let s = original.transform(sym);
                assert_eq!(s.wdl, original.wdl);
                assert_eq!(s.game, original.game);
                assert_eq!(s.policy.len(), original.policy.len());
                for (mv, p) in &s.policy {
                    assert!(
                        s.state.is_legal(*mv),
                        "policy move {mv} legal after transform"
                    );
                    assert!(original.policy.iter().any(|(_, q)| q == p));
                }
                assert_eq!(s.transform(sym.inverse()), *original);
            }
        }
        // A symmetric position is its own image.
        let s = Sample {
            state: BitState::initial(),
            policy: vec![(Move::new(crate::game::sq(1, 1), crate::game::sq(1, 1)), 1.0)],
            game: 0,
            wdl: Some(Verdict::Draw.one_hot()),
            search_value: None,
        };
        assert!(Symmetry::ALL.iter().all(|&sym| s.transform(sym) == s));
    }

    #[test]
    fn generate_parallel_records_games_with_search_policies() {
        let cfg = SelfPlayConfig {
            simulations: 16,
            tree_batch: 4,
            concurrent: 2,
            ..Default::default()
        };
        let (data, stats) =
            generate_parallel(&AgentSpec::Random { seed: 1 }, 3, &cfg, |_| {}).unwrap();
        assert_eq!(stats.games, 3);
        assert_eq!(data.len(), stats.positions);
        assert_eq!(data.samples.last().unwrap().game, 2);
        for s in &data.samples {
            assert!(s.policy.iter().all(|(mv, _)| s.state.is_legal(*mv)));
            assert!(s.wdl.is_some() && s.search_value.is_some());
        }
    }

    #[test]
    fn split_is_by_game_and_keeps_both_parts_non_empty() {
        let mut data = Dataset::default();
        for i in 0..20 {
            data.samples.push(Sample {
                state: BitState::initial(),
                policy: vec![(Move::from_index(i), 1.0)],
                game: i / 4, // 5 games of 4 positions
                wdl: Some(Verdict::from_index(i % 3).one_hot()),
                search_value: None,
            });
        }
        let (train, valid) = data.clone().split(0.2, 1);
        assert_eq!((train.len(), valid.len()), (16, 4));
        let valid_games: std::collections::HashSet<usize> =
            valid.samples.iter().map(|s| s.game).collect();
        assert_eq!(valid_games.len(), 1);
        assert!(train.samples.iter().all(|s| !valid_games.contains(&s.game)));
        let (train, valid) = data.split(0.0, 1);
        assert_eq!((train.len(), valid.len()), (16, 4));
    }

    #[test]
    #[ignore = "trains a network; run with: cargo test --release -- --include-ignored"]
    fn linear_fits_the_synthetic_dataset_on_every_backend() {
        let _device = crate::nn::device_lock();
        let data = synthetic_dataset(4000, 3);
        for kind in BackendKind::for_tests() {
            let device = kind.device();
            let opts = TrainOptions {
                batch_size: 128,
                learning_rate: 1e-2,
                ..opts(kind, "linear-synthetic", 20)
            };
            let (_, report) = train(
                |d| LinearNetConfig::default().init(d),
                &device,
                data.clone(),
                &opts,
            );
            assert!(
                report.validation.verdict_accuracy > 0.8
                    && report.validation.verdict_accuracy > report.baseline.verdict_accuracy + 0.2,
                "{kind}: {report:?}"
            );
            assert!(
                report.validation.policy_accuracy > 0.5
                    && report.validation.policy_accuracy > report.baseline.policy_accuracy + 0.2,
                "{kind}: {report:?}"
            );
        }
    }

    #[test]
    #[ignore = "trains a network; run with: cargo test --release -- --include-ignored"]
    fn training_on_self_play_beats_the_baselines() {
        let _device = crate::nn::device_lock();
        let kind = BackendKind::preferred();
        let device = kind.device();
        let data = generate(&AgentSpec::Greedy { seed: 1 }, 300, |_| {}).unwrap();
        let first = data.samples[0].state;
        let options = TrainOptions {
            augment: Augment::All,
            ..opts(kind, "linear-selfplay", 8)
        };
        let (model, report) = train(
            |d| LinearNetConfig::default().init(d),
            &device,
            data,
            &options,
        );
        // Greedy play is a deterministic function of the position, so the policy is learnable.
        // The outcome labels of a few hundred greedy games are close to noise, so the value
        // only has to stay in the neighbourhood of the mean predictor.
        assert!(
            report.validation.policy_accuracy > report.baseline.policy_accuracy + 0.1,
            "{report:?}"
        );
        assert!(
            report.validation.value_mse < report.baseline.value_mse * 1.2,
            "{report:?}"
        );
        assert!((1..=8).contains(&report.best_epoch), "{report:?}");
        let p = predict_host(&model, &[first], &device);
        assert!((p.wdl[0].iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!((-1.0..=1.0).contains(&p.values()[0]));
    }
}
