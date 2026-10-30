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
use super::train_stats::{LossPartMetric, LossParts, LrMetric, StatMetric};
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
use crate::seed::derive_seed;
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

    /// Always predicting the most common verdict (and the mean win/draw/loss target), the mean
    /// value, and a uniformly random legal move.
    pub fn baseline(&self) -> Evaluation {
        Evaluation {
            verdict_accuracy: self.baseline_accuracy(),
            value_mse: self.baseline_mse(),
            policy_accuracy: self.baseline_policy_accuracy(),
            wdl_loss: self.baseline_wdl_loss(),
            policy_loss: self.baseline_policy_loss(),
            policy_entropy: self.policy_entropy(),
        }
    }

    /// Mean entropy of the target move distributions (see [`Evaluation::policy_entropy`]).
    pub fn policy_entropy(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let total: f64 = self.samples.iter().map(|s| target_entropy(&s.policy)).sum();
        (total / self.len() as f64) as f32
    }

    /// Cross-entropy of always predicting the mean win/draw/loss target, over the samples that
    /// have one.
    pub fn baseline_wdl_loss(&self) -> f32 {
        let targets: Vec<[f32; 3]> = self.samples.iter().filter_map(|s| s.wdl).collect();
        if targets.is_empty() {
            return 0.0;
        }
        let n = targets.len() as f32;
        let mean: [f32; 3] = std::array::from_fn(|k| targets.iter().map(|t| t[k]).sum::<f32>() / n);
        targets.iter().map(|t| cross_entropy(t, &mean)).sum::<f32>() / n
    }

    /// Cross-entropy of a uniform policy over the legal moves: the mean of their log count.
    pub fn baseline_policy_loss(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        self.samples
            .iter()
            .map(|s| (s.state.legal_moves_vec().len() as f32).ln())
            .sum::<f32>()
            / self.len() as f32
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
            state.apply(mv.into());
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
    Ok(generate_parallel_with(
        &mut evaluator.build()?,
        games,
        cfg,
        on_game,
    ))
}

/// The search threads [`generate_threaded`] uses for `games` games when asked for `threads`:
/// 0 means one per core, and never more threads than games.
pub fn self_play_threads(threads: usize, games: usize) -> usize {
    crate::selfplay::search_threads(threads, games)
}

/// Plays `games` games of search with `evaluator` as the leaf evaluator on
/// [`self_play_threads`]`(threads, games)` search threads, each playing a contiguous share of
/// the games with its own seed (`cfg.seed` mixed with the thread's index, see
/// [`derive_seed`]), and returns them as one
/// dataset (game ids distinct, in thread order) with the statistics summed.
///
/// With several threads, a network evaluator (`nn ...`) runs on one [`super::EvalServer`]
/// that batches every thread's leaves together, so the device's per-batch costs are paid once
/// for all threads; other evaluators (random playouts, ...), and any evaluator on a single
/// thread, get an agent per thread, as [`generate_parallel`] builds it. `on_game` is called
/// from the search threads with the number of games finished so far.
pub fn generate_threaded(
    evaluator: &AgentSpec,
    games: usize,
    threads: usize,
    cfg: &SelfPlayConfig,
    on_game: impl Fn(usize) + Sync,
) -> Result<(Dataset, SelfPlayStats), String> {
    let start = std::time::Instant::now();
    let threads = self_play_threads(threads, games);
    let server = match evaluator {
        AgentSpec::Nn { .. } if threads > 1 => {
            let spec = evaluator.clone();
            Some(super::EvalServer::spawn(
                move || spec.build(),
                super::MAX_BATCH,
            )?)
        }
        _ => None,
    };
    let done = std::sync::atomic::AtomicUsize::new(0);
    let runs: Vec<Result<(Dataset, SelfPlayStats), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let games = games / threads + usize::from(t < games % threads);
                let cfg = SelfPlayConfig {
                    seed: derive_seed(cfg.seed, t as u64),
                    ..cfg.clone()
                };
                let (server, done, on_game) = (server.as_ref(), &done, &on_game);
                scope.spawn(move || {
                    let finished = |_| {
                        on_game(1 + done.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
                    };
                    match server {
                        Some(server) => Ok(generate_parallel_with(
                            &mut server.client(),
                            games,
                            &cfg,
                            finished,
                        )),
                        None => generate_parallel(evaluator, games, &cfg, finished),
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("self-play thread panicked"))
            .collect()
    });
    let mut data = Dataset::default();
    let mut stats = SelfPlayStats::default();
    for run in runs {
        let (part, part_stats) = run?;
        let offset = data.games();
        data.samples
            .extend(part.samples.into_iter().map(|s| Sample {
                game: s.game + offset,
                ..s
            }));
        stats.games += part_stats.games;
        stats.positions += part_stats.positions;
        stats.simulations += part_stats.simulations;
        stats.evaluator_calls += part_stats.evaluator_calls;
        stats.leaves += part_stats.leaves;
    }
    stats.seconds = start.elapsed().as_secs_f32();
    Ok((data, stats))
}

/// [`generate_parallel`] with an evaluator already built, such as a
/// [`super::RemoteEvaluator`] shared with other threads.
pub fn generate_parallel_with<A: UtttAgent<State = BitState>>(
    evaluator: &mut A,
    games: usize,
    cfg: &SelfPlayConfig,
    on_game: impl FnMut(usize),
) -> (Dataset, SelfPlayStats) {
    let (records, stats) = self_play(evaluator, games, cfg, on_game);
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
    (data, stats)
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

pub use super::train_stats::StepOutput;

/// Forward pass plus the loss: the weighted cross-entropy between the win/draw/loss target
/// and the verdict logits (averaged over the samples that have a target), the weighted squared error
/// of the expected value (`p_win - p_loss`) against the scalar value target, and the
/// cross-entropy between the target move distribution and the legal-masked policy. The
/// output holds the summed loss, the `[batch, 3]` verdict logits and the `[batch]` verdict
/// targets, which burn's loss and accuracy metrics read, and the three terms unweighted (see
/// [`LossParts`](super::train_stats::LossParts)).
pub fn forward_step<M: Model>(model: &M, batch: Batch<M>) -> (ClassificationOutput, LossParts) {
    let prediction = model.predict(batch.input);
    let n = batch.verdicts.dims()[0];
    // Averaged over the positions that have a win/draw/loss target, so that the term keeps
    // its weight when only some of a batch carry one.
    let labelled = batch.wdl_mask.clone().sum().clamp_min(1.0);
    let wdl_loss = (-(log_softmax(prediction.wdl.clone(), 1) * batch.wdl).sum_dim(1)
        * batch.wdl_mask)
        .sum()
        .div(labelled);
    let probabilities = softmax(prediction.wdl.clone(), 1);
    let expected = probabilities.clone().slice([0..n, 0..1]) - probabilities.slice([0..n, 2..3]);
    let value_loss = (expected - batch.values).powf_scalar(2.0).mean();
    // The targets' own entropy, the least the policy's cross-entropy can be (0 ln 0 = 0).
    let target_entropy = -(batch.policy.clone() * batch.policy.clone().clamp_min(1e-12).log())
        .sum_dim(1)
        .mean()
        .detach();
    let policy_loss = -(prediction.policy_log_probabilities(batch.legal) * batch.policy)
        .sum_dim(1)
        .mean();
    let loss = wdl_loss.clone().mul_scalar(batch.outcome_weight)
        + value_loss.clone().mul_scalar(batch.search_value_weight)
        + policy_loss.clone();
    let policy_kl = policy_loss.clone() - target_entropy;
    let parts = LossParts::new(
        Tensor::cat(vec![wdl_loss, value_loss, policy_loss, policy_kl], 0).detach(),
        n,
    );
    (
        ClassificationOutput::new(loss, prediction.wdl, batch.verdicts),
        parts,
    )
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
                let (item, parts) = $crate::nn::train::forward_step(self, batch);
                let grads = ::burn::optim::GradientsParams::from_grads(item.loss.backward(), self);
                let stats = $crate::nn::train_stats::TrainStats::new(self, &grads)
                    .map(::std::sync::Arc::new);
                ::burn::train::TrainOutput {
                    grads,
                    item: $crate::nn::train::StepOutput {
                        classification: item,
                        stats,
                        parts: ::std::sync::Arc::new(parts),
                    },
                }
            }

            fn optimize(
                self,
                optim: &mut ::burn::optim::ModuleOptimizer,
                lr: ::burn::lr_scheduler::module_lr_scheduler::ModuleLearningRate,
                grads: ::burn::optim::GradientsParams,
            ) -> Self {
                $crate::nn::train_stats::optimize_recorded(self, optim, lr, grads)
            }
        }

        impl ::burn::train::InferenceStep for $model {
            type Input = $crate::nn::train::Batch<$model>;
            type Output = $crate::nn::train::StepOutput;

            fn step(&self, batch: Self::Input) -> $crate::nn::train::StepOutput {
                let (classification, parts) = $crate::nn::train::forward_step(self, batch);
                $crate::nn::train::StepOutput {
                    classification,
                    stats: None,
                    parts: ::std::sync::Arc::new(parts),
                }
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
    /// Training steps between two samples of the gradient and update statistics, 0 for none
    /// (see [`super::train_stats::set_every`]).
    pub stats_every: usize,
    /// Tables trained fake-quantised to the export's int8 grid, by the export's array names
    /// (see [`Model::set_qat`]); none by default.
    pub qat: super::export::Int8Arrays,
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
            stats_every: super::train_stats::DEFAULT_EVERY,
            qat: super::export::Int8Arrays::default(),
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
    /// MSE of `p_win - p_loss` against the scalar value target: the loss's value term.
    pub value_mse: f32,
    /// Fraction of positions whose most likely move is the target's most likely move.
    pub policy_accuracy: f32,
    /// Cross-entropy of the win/draw/loss prediction against the target, over the positions
    /// that have one: the loss's win/draw/loss term.
    pub wdl_loss: f32,
    /// Cross-entropy of the policy over the legal moves against the target distribution: the
    /// loss's policy term.
    pub policy_loss: f32,
    /// Mean entropy of the target move distributions: the least `policy_loss` can be, so that
    /// `policy_loss - policy_entropy` is the KL divergence left to learn.
    pub policy_entropy: f32,
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
        baseline: valid_data.baseline(),
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

    super::train_stats::set_every(opts.stats_every);
    let mut model = init(&ad_device);
    model.set_qat(&opts.qat);
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

    let mut training = SupervisedTraining::new(&opts.artifact_dir, loader_train, loader_valid)
        .num_epochs(opts.epochs)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LrMetric::new());
    // The loss's terms, unweighted: which of the heads a change helps.
    for part in 0..LossParts::NAMES.len() {
        training = training
            .metric_train_numeric(LossPartMetric::new(part))
            .metric_valid_numeric(LossPartMetric::new(part));
    }
    // Per part of the network: the gradient norm and the size of each step against the weights.
    for metric in StatMetric::all() {
        training = training.metric_train_numeric(metric);
    }
    let training = training
        .with_training_strategy(ExecutionStrategy::single(ad_device).into())
        .with_default_checkpointers()
        .with_application_logger(Some(Box::new(ExperimentLog(
            opts.artifact_dir.join("experiment.log"),
        ))));
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
    model.set_qat(&opts.qat);

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
                    model.set_qat(&opts.qat);
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

/// The Learner's `experiment.log`, as burn's own file logger writes it (INFO and above, wgpu
/// quieter) but with the kernel compiler's IR left out: CubeCL compiles CUDA kernels through
/// pliron, which logs every pass's IR at INFO, about 400 MB per training run.
struct ExperimentLog(PathBuf);

impl burn::train::ApplicationLoggerInstaller for ExperimentLog {
    fn install(&self) -> Result<(), String> {
        use tracing_core::{Level, LevelFilter};
        use tracing_subscriber::Layer;
        use tracing_subscriber::filter::filter_fn;
        use tracing_subscriber::prelude::*;

        let dir = self.0.parent().unwrap_or_else(|| std::path::Path::new("."));
        let file = self.0.file_name().expect("experiment.log is a file");
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(tracing_appender::rolling::never(dir, file))
            .with_filter(LevelFilter::INFO)
            .with_filter(filter_fn(|m| {
                let chatty = m
                    .module_path()
                    .is_some_and(|p| p.starts_with("wgpu") || p.starts_with("pliron"));
                !(chatty && *m.level() >= Level::INFO)
            }));
        tracing_subscriber::registry()
            .with(layer)
            .try_init()
            .map_err(|_| "Failed to install the file logger.".to_string())?;
        let hook = std::panic::take_hook();
        let path = self.0.clone();
        std::panic::set_hook(Box::new(move |info| {
            log::error!("PANIC => {info}");
            eprintln!(
                "=== PANIC ===\nA fatal error happened, you can check the experiment logs here => \
                 '{}'\n=============",
                path.display()
            );
            hook(info);
        }));
        Ok(())
    }
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
    let mut sums = EvalSums::default();
    let mut states = Vec::new();
    for chunk in samples.chunks(1024) {
        states.clear();
        states.extend(chunk.iter().map(|s| s.state));
        let prediction = predict_host(model, &states, device);
        for (i, s) in chunk.iter().enumerate() {
            sums.add(s, &prediction.wdl[i], &prediction.policy[i]);
        }
    }
    sums.finish()
}

/// The sums behind an [`Evaluation`], one prediction at a time.
#[derive(Default)]
struct EvalSums {
    n: usize,
    verdict_hits: usize,
    verdict_total: usize,
    policy_hits: usize,
    value_err: f64,
    wdl_loss: f64,
    policy_loss: f64,
    policy_entropy: f64,
}

impl EvalSums {
    /// Adds a prediction for `s`: win/draw/loss probabilities and the legal moves'.
    fn add(&mut self, s: &Sample, p: &[f32; 3], moves: &[(Move, f32)]) {
        self.n += 1;
        if let Some(target) = s.verdict() {
            self.verdict_total += 1;
            self.verdict_hits += usize::from(most_likely(p) == target);
        }
        if let Some(target) = &s.wdl {
            self.wdl_loss += f64::from(cross_entropy(target, p));
        }
        let best = moves
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(mv, _)| *mv)
            .expect("at least one legal move");
        self.policy_hits += usize::from(best == s.best_move());
        self.value_err += (expected_value(p) - s.value_target()).powi(2) as f64;
        self.policy_entropy += target_entropy(&s.policy);
        self.policy_loss -= s
            .policy
            .iter()
            .map(|&(mv, t)| {
                let q = moves.iter().find(|m| m.0 == mv).map_or(0.0, |m| m.1);
                f64::from(t) * f64::from(q.max(1e-12)).ln()
            })
            .sum::<f64>();
    }

    fn finish(&self) -> Evaluation {
        if self.n == 0 {
            return Evaluation::default();
        }
        let n = self.n as f64;
        Evaluation {
            verdict_accuracy: if self.verdict_total == 0 {
                0.0
            } else {
                self.verdict_hits as f32 / self.verdict_total as f32
            },
            value_mse: (self.value_err / n) as f32,
            policy_accuracy: (self.policy_hits as f64 / n) as f32,
            wdl_loss: (self.wdl_loss / self.verdict_total.max(1) as f64) as f32,
            policy_loss: (self.policy_loss / n) as f32,
            policy_entropy: (self.policy_entropy / n) as f32,
        }
    }
}

/// How far a model is from respecting the board's eight symmetries, over `samples`: the model
/// run on every symmetric image of each position, its outputs mapped back.
#[derive(Clone, Debug, Default)]
pub struct SymmetryEvaluation {
    /// Each image evaluated on its own, averaged over the eight.
    pub each: Evaluation,
    /// The eight images' probabilities averaged.
    pub mean_probabilities: Evaluation,
    /// The eight images' log-probabilities averaged (and renormalised): their logits averaged.
    pub mean_logits: Evaluation,
    /// Largest and RMS deviation of an image's outputs from the eight's mean probabilities:
    /// win/draw/loss probabilities, value, legal move probabilities.
    pub largest: [f32; 3],
    pub rms: [f32; 3],
}

/// The model's [`SymmetryEvaluation`] over `samples`.
pub fn evaluate_symmetries<M: Model>(
    model: &M,
    device: &Device,
    samples: &[Sample],
) -> SymmetryEvaluation {
    let (mut each, mut mean_p, mut mean_log) =
        (EvalSums::default(), EvalSums::default(), EvalSums::default());
    let mut largest = [0f32; 3];
    let mut squares = [(0f64, 0usize); 3];
    let mut deviate = |k: usize, d: f32| {
        largest[k] = largest[k].max(d.abs());
        squares[k].0 += f64::from(d) * f64::from(d);
        squares[k].1 += 1;
    };
    for chunk in samples.chunks(1024) {
        let predictions: Vec<_> = Symmetry::ALL
            .iter()
            .map(|&g| {
                let states: Vec<BitState> = chunk.iter().map(|s| s.state.transform(g)).collect();
                predict_host(model, &states, device)
            })
            .collect();
        for (i, s) in chunk.iter().enumerate() {
            // Each image's outputs in the position's own frame, the moves in index order so
            // that the images line up.
            let images: Vec<([f32; 3], Vec<(Move, f32)>)> = Symmetry::ALL
                .iter()
                .zip(&predictions)
                .map(|(&g, p)| {
                    let back = g.inverse();
                    let mut moves: Vec<(Move, f32)> = p.policy[i]
                        .iter()
                        .map(|&(mv, q)| (back.apply_move(mv), q))
                        .collect();
                    moves.sort_by_key(|m| m.0.index());
                    (p.wdl[i], moves)
                })
                .collect();
            for (wdl, moves) in &images {
                each.add(s, wdl, moves);
            }
            let k = images.len() as f32;
            let mut wdl = [0f32; 3];
            let mut log_wdl = [0f32; 3];
            for (w, _) in &images {
                for c in 0..3 {
                    wdl[c] += w[c] / k;
                    log_wdl[c] += w[c].max(1e-12).ln() / k;
                }
            }
            let (moves0, n_moves) = (&images[0].1, images[0].1.len());
            let mut moves: Vec<(Move, f32)> = moves0.iter().map(|&(mv, _)| (mv, 0.0)).collect();
            let mut log_moves = moves.clone();
            for (_, m) in &images {
                debug_assert_eq!(m.len(), n_moves);
                for j in 0..n_moves {
                    debug_assert_eq!(m[j].0, moves[j].0);
                    moves[j].1 += m[j].1 / k;
                    log_moves[j].1 += m[j].1.max(1e-12).ln() / k;
                }
            }
            mean_p.add(s, &wdl, &moves);
            mean_log.add(s, &normalised_exp(log_wdl), &normalised_exp_moves(log_moves));
            let value = expected_value(&wdl);
            for (w, m) in &images {
                for c in 0..3 {
                    deviate(0, w[c] - wdl[c]);
                }
                deviate(1, expected_value(w) - value);
                for (a, b) in m.iter().zip(&moves) {
                    deviate(2, a.1 - b.1);
                }
            }
        }
    }
    let rms = |(sum, n): (f64, usize)| (sum / n.max(1) as f64).sqrt() as f32;
    SymmetryEvaluation {
        each: each.finish(),
        mean_probabilities: mean_p.finish(),
        mean_logits: mean_log.finish(),
        largest,
        rms: squares.map(rms),
    }
}

/// Probabilities from log-probabilities that need renormalising.
fn normalised_exp(logs: [f32; 3]) -> [f32; 3] {
    let top = logs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e = logs.map(|l| (l - top).exp());
    let total: f32 = e.iter().sum();
    e.map(|x| x / total)
}

fn normalised_exp_moves(mut moves: Vec<(Move, f32)>) -> Vec<(Move, f32)> {
    let top = moves.iter().map(|m| m.1).fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for m in &mut moves {
        m.1 = (m.1 - top).exp();
        total += m.1;
    }
    for m in &mut moves {
        m.1 /= total;
    }
    moves
}

/// `-sum p ln p` of a move distribution.
fn target_entropy(policy: &[(Move, f32)]) -> f64 {
    -policy
        .iter()
        .filter(|&&(_, p)| p > 0.0)
        .map(|&(_, p)| f64::from(p) * f64::from(p).ln())
        .sum::<f64>()
}

/// `-sum_k target_k ln p_k`, with the probabilities floored to keep it finite.
fn cross_entropy(target: &[f32; 3], p: &[f32; 3]) -> f32 {
    -(0..3)
        .map(|k| target[k] * p[k].max(1e-12).ln())
        .sum::<f32>()
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
                    policy: vec![(best.into(), 1.0)],
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
