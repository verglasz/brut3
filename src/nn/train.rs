//! Self-play data generation and value-network training.

use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::{Dispatch, DispatchDevice};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};

use super::encode::{FEATURES, encode_into};
use super::model::{ValueNet, ValueNetConfig};
use crate::agent::{AgentSpec, UtttAgent};
use crate::game::{BitState, GameState};

/// Encoded positions with the game result each led to, from the side to move's view.
#[derive(Clone, Debug, Default)]
pub struct Dataset {
    /// `len() * FEATURES` values.
    pub features: Vec<f32>,
    /// One target in `{-1, 0, 1}` per position.
    pub targets: Vec<f32>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    pub fn push(&mut self, state: &BitState, target: f32) {
        let start = self.features.len();
        self.features.resize(start + FEATURES, 0.0);
        encode_into(state, &mut self.features[start..]);
        self.targets.push(target);
    }

    fn features_of(&self, i: usize) -> &[f32] {
        &self.features[i * FEATURES..(i + 1) * FEATURES]
    }

    /// Mean squared error of always predicting the mean target.
    pub fn baseline_mse(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let mean = self.targets.iter().sum::<f32>() / self.len() as f32;
        self.targets.iter().map(|t| (t - mean).powi(2)).sum::<f32>() / self.len() as f32
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

#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Fraction of positions held out to report validation error.
    pub validation_fraction: f32,
    pub seed: u64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 256,
            learning_rate: 1e-3,
            validation_fraction: 0.1,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpochReport {
    pub epoch: usize,
    pub train_mse: f32,
    pub validation_mse: f32,
    pub seconds: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainReport {
    pub train_positions: usize,
    pub validation_positions: usize,
    /// MSE of predicting the mean target, for comparison.
    pub baseline_mse: f32,
    pub epochs: Vec<EpochReport>,
}

/// Trains a fresh [`ValueNet`] on `data` on `device`, calling `on_epoch` after each epoch.
/// Returns the trained network (ready for inference on `device`) and the report.
pub fn train(
    device: &DispatchDevice,
    data: &Dataset,
    opts: &TrainOptions,
    mut on_epoch: impl FnMut(&EpochReport),
) -> (ValueNet<Dispatch>, TrainReport) {
    assert!(!data.is_empty(), "cannot train on an empty dataset");
    let ad_device = DispatchDevice::autodiff(device.clone());
    Dispatch::seed(&ad_device, opts.seed);
    let mut rng = SmallRng::seed_from_u64(opts.seed);

    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.shuffle(&mut rng);
    let n_val = ((data.len() as f32 * opts.validation_fraction) as usize).min(data.len() - 1);
    let (val_idx, train_idx) = indices.split_at(n_val);
    let mut train_idx = train_idx.to_vec();

    let mut model = ValueNetConfig::default().init::<Dispatch>(&ad_device);
    let mut optim = AdamWConfig::new().init::<Dispatch, ValueNet<Dispatch>>();
    let mut report = TrainReport {
        train_positions: train_idx.len(),
        validation_positions: val_idx.len(),
        baseline_mse: data.baseline_mse(),
        epochs: Vec::new(),
    };

    let batch = opts.batch_size.max(1);
    let mut xs = Vec::with_capacity(batch * FEATURES);
    let mut ys = Vec::with_capacity(batch);
    for epoch in 1..=opts.epochs {
        let start = Instant::now();
        train_idx.shuffle(&mut rng);
        let mut loss_sum = 0.0f64;
        for chunk in train_idx.chunks(batch) {
            xs.clear();
            ys.clear();
            for &i in chunk {
                xs.extend_from_slice(data.features_of(i));
                ys.push(data.targets[i]);
            }
            let n = chunk.len();
            let x = Tensor::<Dispatch, 2>::from_data(
                TensorData::new(xs.clone(), [n, FEATURES]),
                &ad_device,
            );
            let y =
                Tensor::<Dispatch, 2>::from_data(TensorData::new(ys.clone(), [n, 1]), &ad_device);
            let loss = (model.forward(x) - y).powf_scalar(2.0).mean();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(opts.learning_rate, model, grads);
            loss_sum += loss.into_scalar().to_f64() * n as f64;
        }
        let inference = model.valid();
        let epoch_report = EpochReport {
            epoch,
            train_mse: (loss_sum / train_idx.len() as f64) as f32,
            validation_mse: mse(&inference, device, data, val_idx),
            seconds: start.elapsed().as_secs_f32(),
        };
        on_epoch(&epoch_report);
        report.epochs.push(epoch_report);
    }
    (model.valid(), report)
}

/// Mean squared error of `model` over the given positions; `0` if there are none.
pub fn mse(
    model: &ValueNet<Dispatch>,
    device: &DispatchDevice,
    data: &Dataset,
    idx: &[usize],
) -> f32 {
    if idx.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut xs = Vec::new();
    for chunk in idx.chunks(4096) {
        xs.clear();
        for &i in chunk {
            xs.extend_from_slice(data.features_of(i));
        }
        let predicted = model.values(&xs, device);
        for (p, &i) in predicted.iter().zip(chunk) {
            sum += (p - data.targets[i]).powi(2) as f64;
        }
    }
    (sum / idx.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::BackendKind;

    #[test]
    fn generate_records_every_position_with_its_result() {
        let data = generate(&AgentSpec::Random { seed: 1 }, 3, |_| {}).unwrap();
        assert!(data.len() >= 3 * 17, "{} positions", data.len());
        assert_eq!(data.features.len(), data.len() * FEATURES);
        assert!(data.targets.iter().all(|t| [-1.0, 0.0, 1.0].contains(t)));
        // Consecutive positions of one game alternate perspective, so targets alternate sign
        // (or are all zero for a draw).
        assert!(data.targets[0] == -data.targets[1] || data.targets[0] == 0.0);
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
            let v: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>().tanh();
            data.features.extend_from_slice(&x);
            data.targets.push(v);
        }
        let opts = TrainOptions {
            epochs: 20,
            batch_size: 128,
            learning_rate: 1e-3,
            ..Default::default()
        };
        let (_, report) = train(&device, &data, &opts, |_| {});
        let first = report.epochs.first().unwrap().train_mse;
        let last = report.epochs.last().unwrap();
        assert!(
            last.train_mse < first * 0.3,
            "{kind}: train mse {first} -> {}",
            last.train_mse
        );
        assert!(
            last.validation_mse < report.baseline_mse * 0.5,
            "{kind}: val {} vs baseline {}",
            last.validation_mse,
            report.baseline_mse
        );
    }

    #[test]
    fn training_on_self_play_beats_the_baseline() {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let data = generate(&AgentSpec::Greedy { seed: 1 }, 60, |_| {}).unwrap();
        let (model, report) = train(
            &device,
            &data,
            &TrainOptions {
                epochs: 5,
                ..Default::default()
            },
            |_| {},
        );
        let last = report.epochs.last().unwrap();
        assert!(
            last.validation_mse < report.baseline_mse,
            "{last:?} vs baseline {}",
            report.baseline_mse
        );
        let v = model.values(data.features_of(0), &device);
        assert!((-1.0..=1.0).contains(&v[0]));
    }
}
