//! Self-play data generation and value-network training with burn's `Learner`.

use std::path::PathBuf;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::InMemDataset;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::train::metric::{Adaptor, LossInput, LossMetric};
use burn::train::renderer::CliMetricsRenderer;
use burn::train::{
    ExecutionStrategy, InferenceStep, ItemLazy, Learner, SupervisedTraining, TrainOutput, TrainStep,
};
use burn::{Dispatch, DispatchDevice};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};

use super::encode::{FEATURES, encode_into};
use super::model::{Recorder, ValueNet, ValueNetConfig};
use crate::agent::{AgentSpec, UtttAgent};
use crate::game::{BitState, GameState};

/// One encoded position and the result the game it came from ended in, from the side to
/// move's view: `1` won, `-1` lost, `0` drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// `FEATURES` values.
    pub features: Vec<f32>,
    pub target: f32,
}

impl Sample {
    pub fn new(state: &BitState, target: f32) -> Self {
        let mut features = vec![0.0; FEATURES];
        encode_into(state, &mut features);
        Self { features, target }
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

    pub fn push(&mut self, state: &BitState, target: f32) {
        self.samples.push(Sample::new(state, target));
    }

    /// Mean squared error of always predicting the mean target.
    pub fn baseline_mse(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let n = self.len() as f32;
        let mean = self.samples.iter().map(|s| s.target).sum::<f32>() / n;
        self.samples
            .iter()
            .map(|s| (s.target - mean).powi(2))
            .sum::<f32>()
            / n
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
/// result. `on_game` is called after each game with the number played so far.
pub fn generate(
    spec: &AgentSpec,
    games: usize,
    mut on_game: impl FnMut(usize),
) -> Result<Dataset, String> {
    let mut agent = spec.build()?;
    let mut data = Dataset::default();
    let mut positions = Vec::new();
    for game in 0..games {
        positions.clear();
        let mut state = BitState::initial();
        let outcome = loop {
            if let Some(outcome) = state.outcome() {
                break outcome;
            }
            positions.push(state);
            state.apply(agent.choose_move(&state));
        };
        for p in &positions {
            data.push(p, outcome.value_for(p.to_move()));
        }
        on_game(game + 1);
    }
    Ok(data)
}

/// A minibatch on a device.
#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    /// `[batch, FEATURES]`
    pub features: Tensor<B, 2>,
    /// `[batch, 1]`
    pub targets: Tensor<B, 2>,
}

/// Turns samples into a [`Batch`].
#[derive(Clone, Debug, Default)]
pub struct SampleBatcher;

impl<B: Backend> Batcher<B, Sample, Batch<B>> for SampleBatcher {
    fn batch(&self, items: Vec<Sample>, device: &B::Device) -> Batch<B> {
        let n = items.len();
        let mut features = Vec::with_capacity(n * FEATURES);
        let mut targets = Vec::with_capacity(n);
        for item in items {
            features.extend_from_slice(&item.features);
            targets.push(item.target);
        }
        Batch {
            features: Tensor::from_data(TensorData::new(features, [n, FEATURES]), device),
            targets: Tensor::from_data(TensorData::new(targets, [n, 1]), device),
        }
    }
}

/// Output of one training or validation step: the batch loss and the predictions.
pub struct StepOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    /// `[batch, 1]`
    pub output: Tensor<B, 2>,
    /// `[batch, 1]`
    pub targets: Tensor<B, 2>,
    /// The loss already read back on the thread that computed it; see [`eager_loss_read`].
    loss_host: Option<TensorData>,
}

/// The part of a [`StepOutput`] the metrics need, re-materialised on the plain (non-autodiff)
/// device.
///
/// This replaces burn's `RegressionOutput`, whose sync reads through a multi-tensor
/// `Transaction`. On the MLIR CPU backend (cubecl-cpu 0.10) that read, and any read of a
/// still-lazy tensor from a thread other than the one that produced it, panics inside the
/// memory pool. So the loss is read on the producing thread there ([`eager_loss_read`]) and
/// only host data crosses threads.
pub struct StepOutputSync {
    loss: Tensor<Dispatch, 1>,
}

impl ItemLazy for StepOutput<Dispatch> {
    type ItemSync = StepOutputSync;

    fn sync(self) -> StepOutputSync {
        let device = plain_device(&self.loss.device());
        let data = match self.loss_host {
            Some(data) => data,
            None => self.loss.into_data(),
        };
        StepOutputSync {
            loss: Tensor::from_data(data, &device),
        }
    }
}

/// Whether tensors must be read back on the thread that computed them (MLIR CPU backend).
fn eager_loss_read(device: &DispatchDevice) -> bool {
    #[cfg(feature = "cpu-mlir")]
    if matches!(plain_device(device), DispatchDevice::Cpu(_)) {
        return true;
    }
    let _ = device;
    false
}

impl Adaptor<LossInput<Dispatch>> for StepOutputSync {
    fn adapt(&self) -> LossInput<Dispatch> {
        LossInput::new(self.loss.clone())
    }
}

/// The device itself, or the device wrapped by an autodiff device.
fn plain_device(device: &DispatchDevice) -> DispatchDevice {
    match device {
        DispatchDevice::Autodiff(inner) => (**inner).clone(),
        other => other.clone(),
    }
}

impl<B: Backend> ValueNet<B> {
    /// Forward pass plus mean squared error.
    pub fn forward_step(&self, batch: Batch<B>) -> StepOutput<B> {
        let output = self.forward(batch.features);
        let loss = (output.clone() - batch.targets.clone())
            .powf_scalar(2.0)
            .mean();
        StepOutput {
            loss,
            output,
            targets: batch.targets,
            loss_host: None,
        }
    }
}

impl ValueNet<Dispatch> {
    fn forward_step_synced(&self, batch: Batch<Dispatch>) -> StepOutput<Dispatch> {
        let mut out = self.forward_step(batch);
        if eager_loss_read(&out.loss.device()) {
            out.loss_host = Some(out.loss.clone().into_data());
        }
        out
    }
}

impl TrainStep for ValueNet<Dispatch> {
    type Input = Batch<Dispatch>;
    type Output = StepOutput<Dispatch>;

    fn step(&self, batch: Batch<Dispatch>) -> TrainOutput<StepOutput<Dispatch>> {
        let item = self.forward_step_synced(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for ValueNet<Dispatch> {
    type Input = Batch<Dispatch>;
    type Output = StepOutput<Dispatch>;

    fn step(&self, batch: Batch<Dispatch>) -> StepOutput<Dispatch> {
        self.forward_step_synced(batch)
    }
}

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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainReport {
    pub train_positions: usize,
    pub validation_positions: usize,
    /// MSE of predicting the mean target over the validation split, for comparison.
    pub baseline_mse: f32,
    /// MSE of the trained network over the validation split.
    pub validation_mse: f32,
}

/// Trains a fresh [`ValueNet`] on `data` on `device` with burn's `Learner`, rendering
/// progress to the terminal, and returns the network ready for inference on `device`.
pub fn train(
    device: &DispatchDevice,
    data: Dataset,
    opts: &TrainOptions,
) -> (ValueNet<Dispatch>, TrainReport) {
    assert!(!data.is_empty(), "cannot train on an empty dataset");
    let ad_device = DispatchDevice::autodiff(device.clone());
    Dispatch::seed(&ad_device, opts.seed);

    let (train_data, valid_data) = data.split(opts.validation_fraction, opts.seed);
    let report_base = TrainReport {
        train_positions: train_data.len(),
        validation_positions: valid_data.len(),
        baseline_mse: valid_data.baseline_mse(),
        validation_mse: 0.0,
    };
    let batch = opts.batch_size.max(1);
    let loader_train = DataLoaderBuilder::<Dispatch, Sample, Batch<Dispatch>>::new(SampleBatcher)
        .batch_size(batch)
        .shuffle(opts.seed)
        .build(InMemDataset::new(train_data.samples));
    let valid_samples = valid_data.samples;
    let loader_valid = DataLoaderBuilder::<Dispatch, Sample, Batch<Dispatch>>::new(SampleBatcher)
        .batch_size(4096)
        .build(InMemDataset::new(valid_samples.clone()));

    let model = ValueNetConfig::default().init::<Dispatch>(&ad_device);
    let optim = AdamWConfig::new().init::<Dispatch, ValueNet<Dispatch>>();
    let learner = Learner::new(model, optim, opts.learning_rate);

    let mut training = SupervisedTraining::new(&opts.artifact_dir, loader_train, loader_valid)
        .num_epochs(opts.epochs)
        .metric_train_numeric(LossMetric::<Dispatch>::new())
        .metric_valid_numeric(LossMetric::<Dispatch>::new())
        .with_training_strategy(ExecutionStrategy::single(ad_device).into())
        .summary();
    // The checkpointer saves from its own thread, which the MLIR CPU backend cannot serve
    // (see `StepOutputSync`); it is skipped there.
    if !eager_loss_read(device) {
        training = training.with_file_checkpointer(Recorder::new());
    }
    if !opts.tui {
        training = training.renderer(CliMetricsRenderer::new());
    }
    let result = training.launch(learner);
    let model = result.model;

    let validation_mse = mse(&model, device, &valid_samples);
    (
        model,
        TrainReport {
            validation_mse,
            ..report_base
        },
    )
}

/// Mean squared error of `model` over `samples`; `0` if there are none.
pub fn mse(model: &ValueNet<Dispatch>, device: &DispatchDevice, samples: &[Sample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut xs = Vec::new();
    for chunk in samples.chunks(4096) {
        xs.clear();
        for s in chunk {
            xs.extend_from_slice(&s.features);
        }
        let predicted = model.values(&xs, device);
        for (p, s) in predicted.iter().zip(chunk) {
            sum += (p - s.target).powi(2) as f64;
        }
    }
    (sum / samples.len() as f64) as f32
}

/// A fresh temporary artifact directory for one run (tests, throwaway runs).
pub fn temp_artifact_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("uttt-train-{tag}-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::BackendKind;

    fn opts(kind: BackendKind, epochs: usize) -> TrainOptions {
        TrainOptions {
            epochs,
            tui: false,
            artifact_dir: temp_artifact_dir(&format!("{kind}-{epochs}")),
            ..Default::default()
        }
    }

    #[test]
    fn generate_records_every_position_with_its_result() {
        let data = generate(&AgentSpec::Random { seed: 1 }, 3, |_| {}).unwrap();
        assert!(data.len() >= 3 * 17, "{} positions", data.len());
        assert!(data.samples.iter().all(|s| s.features.len() == FEATURES));
        assert!(
            data.samples
                .iter()
                .all(|s| [-1.0, 0.0, 1.0].contains(&s.target))
        );
        // Consecutive positions of one game alternate perspective, so targets alternate sign
        // (or are all zero for a draw).
        let (a, b) = (data.samples[0].target, data.samples[1].target);
        assert!(a == -b || a == 0.0);
    }

    #[test]
    fn split_keeps_both_parts_non_empty() {
        let mut data = Dataset::default();
        for i in 0..10 {
            data.push(&BitState::initial(), i as f32);
        }
        let (train, valid) = data.clone().split(0.1, 1);
        assert_eq!((train.len(), valid.len()), (9, 1));
        let (train, valid) = data.split(0.0, 1);
        assert_eq!((train.len(), valid.len()), (9, 1));
    }

    #[test]
    fn training_fits_a_linear_target_on_every_backend() {
        let _device = crate::nn::device_lock();
        for kind in BackendKind::compiled() {
            training_fits_a_linear_target(kind);
        }
    }

    fn training_fits_a_linear_target(kind: BackendKind) {
        // Synthetic data: value = tanh(w . x) for a fixed random w in the linear regime of
        // tanh, with enough samples for 199 weights; a linear model must fit it.
        let device = kind.device();
        let mut rng = SmallRng::seed_from_u64(3);
        let w: Vec<f32> = (0..FEATURES)
            .map(|_| rand::Rng::random_range(&mut rng, -0.1..0.1))
            .collect();
        let mut data = Dataset::default();
        for _ in 0..4000 {
            let x: Vec<f32> = (0..FEATURES)
                .map(|_| rand::Rng::random_range(&mut rng, 0..2) as f32)
                .collect();
            let target = w.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>().tanh();
            data.samples.push(Sample {
                features: x,
                target,
            });
        }
        let opts = TrainOptions {
            batch_size: 128,
            ..opts(kind, 20)
        };
        let (_, report) = train(&device, data, &opts);
        assert!(
            report.validation_mse < report.baseline_mse * 0.5,
            "{kind}: val {} vs baseline {}",
            report.validation_mse,
            report.baseline_mse
        );
    }

    #[test]
    fn training_on_self_play_beats_the_baseline() {
        let _device = crate::nn::device_lock();
        let kind = BackendKind::preferred();
        let device = kind.device();
        let data = generate(&AgentSpec::Greedy { seed: 1 }, 60, |_| {}).unwrap();
        let first = data.samples[0].clone();
        let (model, report) = train(&device, data, &opts(kind, 5));
        assert!(report.validation_mse < report.baseline_mse, "{report:?}");
        let v = model.values(&first.features, &device);
        assert!((-1.0..=1.0).contains(&v[0]));
    }
}
