//! Self-play data generation and policy/value training with burn's `Learner`.

use std::marker::PhantomData;
use std::path::PathBuf;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::InMemDataset;
use burn::module::AutodiffModule;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::train::metric::{AccuracyInput, AccuracyMetric, Adaptor, LossInput, LossMetric};
use burn::train::renderer::CliMetricsRenderer;
use burn::train::{
    ExecutionStrategy, InferenceStep, ItemLazy, Learner, SupervisedTraining, TrainOutput, TrainStep,
};
use burn::{Dispatch, DispatchDevice};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};

use super::any_model::Recorder;
use super::linear::LinearNet;
use super::model::{MOVES, Model, Verdict, expected_value, legal_mask, predict_host};
use crate::agent::{AgentSpec, UtttAgent};
use crate::game::{BitState, GameState, Move};

/// One position with how its game ended (side to move's view) and the move distribution the
/// generating agent played from it.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub state: BitState,
    pub verdict: Verdict,
    /// Probabilities over the legal moves; a search agent's visit distribution, or a single
    /// move with probability 1.
    pub policy: Vec<(Move, f32)>,
}

impl Sample {
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
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// How many samples ended in each [`Verdict`], in `Verdict::ALL` order.
    pub fn verdict_counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for s in &self.samples {
            counts[s.verdict.index()] += 1;
        }
        counts
    }

    /// Accuracy of always predicting the most common verdict.
    pub fn baseline_accuracy(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let counts = self.verdict_counts();
        *counts.iter().max().unwrap() as f32 / self.len() as f32
    }

    /// Mean squared error (on the `1 / 0 / -1` value scale) of always predicting the mean.
    pub fn baseline_mse(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let n = self.len() as f32;
        let mean = self.samples.iter().map(|s| s.verdict.value()).sum::<f32>() / n;
        self.samples
            .iter()
            .map(|s| (s.verdict.value() - mean).powi(2))
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

    /// Shuffles with `seed` and splits off `fraction` of the samples as validation data.
    /// At least one sample stays in each part when there are two or more.
    pub fn split(mut self, fraction: f32, seed: u64) -> (Dataset, Dataset) {
        let mut rng = SmallRng::seed_from_u64(seed);
        self.samples.shuffle(&mut rng);
        let n_val = ((self.len() as f32 * fraction) as usize)
            .max(usize::from(self.len() >= 2))
            .min(self.len().saturating_sub(1));
        let train = self.samples.split_off(n_val);
        (Dataset { samples: train }, self)
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
            data.samples.push(Sample {
                verdict: Verdict::from_outcome(outcome, state.to_move()),
                state,
                policy,
            });
        }
        on_game(game + 1);
    }
    Ok(data)
}

/// A minibatch on a device, encoded for model `M`.
pub struct Batch<B: Backend, M: Model<B>> {
    pub input: M::Input,
    /// `[batch]` verdict class indices.
    pub verdicts: Tensor<B, 1, Int>,
    /// `[batch, MOVES]` target move probabilities.
    pub policy: Tensor<B, 2>,
    /// `[batch, MOVES]` legal moves.
    pub legal: Tensor<B, 2, Bool>,
}

impl<B: Backend, M: Model<B>> std::fmt::Debug for Batch<B, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batch")
            .field("input", &self.input)
            .field("verdicts", &self.verdicts)
            .field("policy", &self.policy)
            .field("legal", &self.legal)
            .finish()
    }
}

impl<B: Backend, M: Model<B>> Clone for Batch<B, M> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            verdicts: self.verdicts.clone(),
            policy: self.policy.clone(),
            legal: self.legal.clone(),
        }
    }
}

/// Turns samples into a [`Batch`] for model `M`.
pub struct ModelBatcher<M>(PhantomData<fn() -> M>);

impl<M> Default for ModelBatcher<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<M> Clone for ModelBatcher<M> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<B: Backend, M: Model<B>> Batcher<B, Sample, Batch<B, M>> for ModelBatcher<M> {
    fn batch(&self, items: Vec<Sample>, device: &B::Device) -> Batch<B, M> {
        let n = items.len();
        let states: Vec<BitState> = items.iter().map(|s| s.state).collect();
        let verdicts: Vec<i32> = items.iter().map(|s| s.verdict.index() as i32).collect();
        let mut policy = vec![0.0f32; n * MOVES];
        for (i, item) in items.iter().enumerate() {
            for (mv, p) in &item.policy {
                policy[i * MOVES + mv.index()] = *p;
            }
        }
        Batch {
            input: M::encode(&states, device),
            verdicts: Tensor::from_data(TensorData::new(verdicts, [n]), device),
            policy: Tensor::from_data(TensorData::new(policy, [n, MOVES]), device),
            legal: legal_mask(&states, device),
        }
    }
}

/// Output of one training or validation step.
pub struct StepOutput<B: Backend> {
    /// Verdict cross-entropy plus policy cross-entropy.
    pub loss: Tensor<B, 1>,
    /// `[batch, 3]` verdict logits.
    pub wdl: Tensor<B, 2>,
    /// `[batch]` verdict targets.
    pub verdicts: Tensor<B, 1, Int>,
    /// The tensors already read back on the thread that computed them; see
    /// [`eager_host_read`].
    host: Option<(TensorData, TensorData, TensorData)>,
}

/// A [`StepOutput`] re-materialised on the plain (non-autodiff) device for the metrics.
///
/// This replaces burn's `ClassificationOutput`, whose sync reads through a multi-tensor
/// `Transaction`. On the MLIR CPU backend (cubecl-cpu 0.10) that read, and any read of a
/// still-lazy tensor from a thread other than the one that produced it, panics inside the
/// memory pool. So the tensors are read on the producing thread there ([`eager_host_read`])
/// and only host data crosses threads.
pub struct StepOutputSync {
    loss: Tensor<Dispatch, 1>,
    wdl: Tensor<Dispatch, 2>,
    verdicts: Tensor<Dispatch, 1, Int>,
}

impl ItemLazy for StepOutput<Dispatch> {
    type ItemSync = StepOutputSync;

    fn sync(self) -> StepOutputSync {
        let device = plain_device(&self.loss.device());
        let (loss, wdl, verdicts) = match self.host {
            Some(host) => host,
            None => (
                self.loss.into_data(),
                self.wdl.into_data(),
                self.verdicts.into_data(),
            ),
        };
        StepOutputSync {
            loss: Tensor::from_data(loss, &device),
            wdl: Tensor::from_data(wdl, &device),
            verdicts: Tensor::from_data(verdicts, &device),
        }
    }
}

impl Adaptor<LossInput<Dispatch>> for StepOutputSync {
    fn adapt(&self) -> LossInput<Dispatch> {
        LossInput::new(self.loss.clone())
    }
}

impl Adaptor<AccuracyInput<Dispatch>> for StepOutputSync {
    fn adapt(&self) -> AccuracyInput<Dispatch> {
        AccuracyInput::new(self.wdl.clone(), self.verdicts.clone())
    }
}

/// The device itself, or the device wrapped by an autodiff device.
fn plain_device(device: &DispatchDevice) -> DispatchDevice {
    match device {
        DispatchDevice::Autodiff(inner) => (**inner).clone(),
        other => other.clone(),
    }
}

/// Whether tensors must be read back on the thread that computed them (MLIR CPU backend).
fn eager_host_read(device: &DispatchDevice) -> bool {
    #[cfg(feature = "cpu-mlir")]
    if matches!(plain_device(device), DispatchDevice::Cpu(_)) {
        return true;
    }
    let _ = device;
    false
}

/// Forward pass plus the loss: cross-entropy of the verdict and cross-entropy between the
/// target move distribution and the legal-masked policy.
pub fn forward_step<M: Model<Dispatch>>(
    model: &M,
    batch: Batch<Dispatch, M>,
) -> StepOutput<Dispatch> {
    let prediction = model.predict(batch.input);
    let n = batch.verdicts.dims()[0];
    let wdl_loss = -log_softmax(prediction.wdl.clone(), 1)
        .gather(1, batch.verdicts.clone().reshape([n, 1]))
        .mean();
    let policy_loss = -(prediction.policy_log_probabilities(batch.legal) * batch.policy)
        .sum_dim(1)
        .mean();
    let loss = wdl_loss + policy_loss;
    let host = eager_host_read(&loss.device()).then(|| {
        (
            loss.clone().into_data(),
            prediction.wdl.clone().into_data(),
            batch.verdicts.clone().into_data(),
        )
    });
    StepOutput {
        loss,
        wdl: prediction.wdl,
        verdicts: batch.verdicts,
        host,
    }
}

/// Implements burn's step traits for a model type on the [`Dispatch`] backend.
macro_rules! impl_train_steps {
    ($model:ident) => {
        impl TrainStep for $model<Dispatch> {
            type Input = Batch<Dispatch, $model<Dispatch>>;
            type Output = StepOutput<Dispatch>;

            fn step(&self, batch: Self::Input) -> TrainOutput<StepOutput<Dispatch>> {
                let item = forward_step(self, batch);
                TrainOutput::new(self, item.loss.backward(), item)
            }
        }

        impl InferenceStep for $model<Dispatch> {
            type Input = Batch<Dispatch, $model<Dispatch>>;
            type Output = StepOutput<Dispatch>;

            fn step(&self, batch: Self::Input) -> StepOutput<Dispatch> {
                forward_step(self, batch)
            }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use impl_train_steps;

impl_train_steps!(LinearNet);

#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Fraction of positions held out for validation.
    pub validation_fraction: f32,
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
            batch_size: 256,
            learning_rate: 1e-3,
            validation_fraction: 0.1,
            seed: 0,
            artifact_dir: PathBuf::from("artifacts/train"),
            tui: true,
        }
    }
}

/// Quality of a model over a set of samples.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evaluation {
    /// Fraction of positions whose most likely verdict is the true one.
    pub verdict_accuracy: f32,
    /// MSE of `p_win - p_loss` against the `1 / 0 / -1` outcome.
    pub value_mse: f32,
    /// Fraction of positions whose most likely move is the target's most likely move.
    pub policy_accuracy: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainReport {
    pub train_positions: usize,
    pub validation_positions: usize,
    /// Always predicting the most common verdict, the mean value, and a uniformly random
    /// legal move, over the validation split.
    pub baseline: Evaluation,
    /// The trained model over the validation split.
    pub validation: Evaluation,
}

/// Models that can go through [`train`].
pub trait Trainable:
    Model<Dispatch>
    + AutodiffModule<Dispatch, InnerModule = Self>
    + TrainStep<Input = Batch<Dispatch, Self>, Output = StepOutput<Dispatch>>
    + InferenceStep<Input = Batch<Dispatch, Self>, Output = StepOutput<Dispatch>>
    + std::fmt::Display
    + 'static
{
}

impl<M> Trainable for M where
    M: Model<Dispatch>
        + AutodiffModule<Dispatch, InnerModule = M>
        + TrainStep<Input = Batch<Dispatch, M>, Output = StepOutput<Dispatch>>
        + InferenceStep<Input = Batch<Dispatch, M>, Output = StepOutput<Dispatch>>
        + std::fmt::Display
        + 'static
{
}

/// Trains a model built by `init` on the autodiff form of `device` (a `Dispatch` model cannot
/// be moved between the two) on `data` with burn's `Learner`, rendering progress to the
/// terminal, and returns it ready for inference on `device` together with a report.
pub fn train<M: Trainable>(
    init: impl FnOnce(&DispatchDevice) -> M,
    device: &DispatchDevice,
    data: Dataset,
    opts: &TrainOptions,
) -> (M, TrainReport) {
    assert!(!data.is_empty(), "cannot train on an empty dataset");
    let ad_device = DispatchDevice::autodiff(device.clone());
    Dispatch::seed(&ad_device, opts.seed);

    let (train_data, valid_data) = data.split(opts.validation_fraction, opts.seed);
    let report_base = TrainReport {
        train_positions: train_data.len(),
        validation_positions: valid_data.len(),
        baseline: Evaluation {
            verdict_accuracy: valid_data.baseline_accuracy(),
            value_mse: valid_data.baseline_mse(),
            policy_accuracy: valid_data.baseline_policy_accuracy(),
        },
        validation: Evaluation::default(),
    };
    let batch = opts.batch_size.max(1);
    let loader_train = DataLoaderBuilder::<Dispatch, Sample, Batch<Dispatch, M>>::new(
        ModelBatcher::<M>::default(),
    )
    .batch_size(batch)
    .shuffle(opts.seed)
    .build(InMemDataset::new(train_data.samples));
    let valid_samples = valid_data.samples;
    let loader_valid = DataLoaderBuilder::<Dispatch, Sample, Batch<Dispatch, M>>::new(
        ModelBatcher::<M>::default(),
    )
    .batch_size(1024)
    .build(InMemDataset::new(valid_samples.clone()));

    let model = init(&ad_device);
    let optim = AdamWConfig::new().init::<Dispatch, M>();
    let learner = Learner::new(model, optim, opts.learning_rate);

    let mut training = SupervisedTraining::new(&opts.artifact_dir, loader_train, loader_valid)
        .num_epochs(opts.epochs)
        .metric_train_numeric(LossMetric::<Dispatch>::new())
        .metric_valid_numeric(LossMetric::<Dispatch>::new())
        .metric_train_numeric(AccuracyMetric::<Dispatch>::new())
        .metric_valid_numeric(AccuracyMetric::<Dispatch>::new())
        .with_training_strategy(ExecutionStrategy::single(ad_device).into())
        .summary();
    // The checkpointer saves from its own thread, which the MLIR CPU backend cannot serve
    // (see `StepOutputSync`); it is skipped there.
    if !eager_host_read(device) {
        training = training.with_file_checkpointer(Recorder::new());
    }
    if !opts.tui {
        training = training.renderer(CliMetricsRenderer::new());
    }
    let model = training.launch(learner).model;

    let validation = evaluate(&model, device, &valid_samples);
    (
        model,
        TrainReport {
            validation,
            ..report_base
        },
    )
}

/// Evaluates `model` over `samples`; zeros if there are none.
pub fn evaluate<M: Model<Dispatch>>(
    model: &M,
    device: &DispatchDevice,
    samples: &[Sample],
) -> Evaluation {
    if samples.is_empty() {
        return Evaluation::default();
    }
    let mut verdict_hits = 0usize;
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
            let predicted = (0..3).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
            verdict_hits += usize::from(predicted == s.verdict.index());
            policy_hits += usize::from(best_moves[i] == s.best_move());
            value_err += (expected_value(p) - s.verdict.value()).powi(2) as f64;
        }
    }
    let n = samples.len();
    Evaluation {
        verdict_accuracy: verdict_hits as f32 / n as f32,
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
    use rand::Rng;

    use super::*;
    use crate::game::MoveSet;
    use crate::nn::BackendKind;
    use crate::nn::encode::{FEATURES, encode_into};
    use crate::nn::linear::LinearNetConfig;

    pub fn opts(kind: BackendKind, tag: &str, epochs: usize) -> TrainOptions {
        TrainOptions {
            epochs,
            tui: false,
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
        while data.len() < n {
            let mut state = BitState::initial();
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
                    verdict: Verdict::from_index(verdict),
                    policy: vec![(best, 1.0)],
                });
                let k = rng.random_range(0..legal.len());
                state.apply(legal.iter().nth(k).unwrap());
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
        }
        // Consecutive positions of one game alternate perspective, so verdicts alternate
        // (or are all draws).
        let (a, b) = (data.samples[0].verdict, data.samples[1].verdict);
        assert!(a.value() == -b.value());
        assert_eq!(data.verdict_counts().iter().sum::<usize>(), data.len());
        assert!(data.baseline_policy_accuracy() > 0.0);
    }

    #[test]
    fn split_keeps_both_parts_non_empty() {
        let mut data = Dataset::default();
        for i in 0..10 {
            data.samples.push(Sample {
                state: BitState::initial(),
                verdict: Verdict::from_index(i % 3),
                policy: vec![(Move::from_index(i), 1.0)],
            });
        }
        let (train, valid) = data.clone().split(0.1, 1);
        assert_eq!((train.len(), valid.len()), (9, 1));
        let (train, valid) = data.split(0.0, 1);
        assert_eq!((train.len(), valid.len()), (9, 1));
    }

    #[test]
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
                |d| LinearNetConfig::default().init::<Dispatch>(d),
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
    fn training_on_self_play_beats_the_baselines() {
        let _device = crate::nn::device_lock();
        let kind = BackendKind::preferred();
        let device = kind.device();
        let data = generate(&AgentSpec::Greedy { seed: 1 }, 60, |_| {}).unwrap();
        let first = data.samples[0].state;
        let (model, report) = train(
            |d| LinearNetConfig::default().init::<Dispatch>(d),
            &device,
            data,
            &opts(kind, "linear-selfplay", 5),
        );
        assert!(
            report.validation.value_mse < report.baseline.value_mse,
            "{report:?}"
        );
        assert!(
            report.validation.policy_accuracy > report.baseline.policy_accuracy,
            "{report:?}"
        );
        let p = predict_host(&model, &[first], &device);
        assert!((p.wdl[0].iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!((-1.0..=1.0).contains(&p.values()[0]));
    }
}
