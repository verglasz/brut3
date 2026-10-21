//! The self-play training loop: generate games with the current best model, train a candidate
//! on a replay buffer of recent games, promote it if it beats the best, repeat.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::module::Module;
use burn::prelude::Device;
use serde_json::json;

use super::any_model::{AnyModel, ModelSpec};
use super::backend::BackendKind;
use super::train::{Dataset, Sample, TrainOptions, TrainReport, generate_parallel, train};
use crate::agent::AgentSpec;
use crate::benchmark::{MatchResult, play_match_with};
use crate::selfplay::{MatchExploration, SelfPlayConfig, SelfPlayStats};

/// Trains a model of kind `spec` on `data`: fresh weights, or continuing from the weights at
/// `start_from` (saved by [`AnyModel::save`]).
pub fn train_model(
    spec: &ModelSpec,
    start_from: Option<&Path>,
    device: &Device,
    data: Dataset,
    opts: &TrainOptions,
) -> Result<(AnyModel, TrainReport), String> {
    if let Some(path) = start_from {
        // Surface a bad path or a mismatched spec here rather than inside the init closure.
        let existing = AnyModel::load(path, device)?;
        if existing.spec() != *spec {
            return Err(format!(
                "weights at {} are a {} model, not {spec}",
                path.display(),
                existing.spec()
            ));
        }
    }
    fn load<M: Module>(m: M, path: &Path) -> Result<M, String> {
        m.try_load_file(ModelSpec::weights_path(path))
            .map_err(|e| format!("cannot load weights from {}: {e}", path.display()))
    }
    macro_rules! run {
        ($variant:ident, $init:expr) => {{
            let (m, report) = train(
                |d| {
                    let m = $init(d);
                    match start_from {
                        Some(path) => load(m, path).expect("weights were loaded above"),
                        None => m,
                    }
                },
                device,
                data,
                opts,
            );
            Ok((AnyModel::$variant(m), report))
        }};
    }
    match spec {
        ModelSpec::Linear => run!(Linear, |d| super::linear::LinearNetConfig::default()
            .init(d)),
        ModelSpec::BoardTransformer(cfg) => run!(BoardTransformer, |d| cfg.init(d)),
        ModelSpec::CellTransformer(cfg) => run!(CellTransformer, |d| cfg.init(d)),
        ModelSpec::BoardTransformerV2(cfg) => run!(BoardTransformerV2, |d| cfg.init(d)),
        ModelSpec::CellTransformerV2(cfg) => run!(CellTransformerV2, |d| cfg.init(d)),
    }
}

/// The last `capacity` games' positions.
#[derive(Clone, Debug, Default)]
pub struct ReplayBuffer {
    games: VecDeque<Vec<Sample>>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            games: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Changes the capacity; the oldest games beyond it are dropped at the next push.
    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
    }

    /// Adds every game of `data` (grouped by its game id), dropping the oldest beyond capacity.
    pub fn push(&mut self, data: Dataset) {
        let mut by_game: std::collections::BTreeMap<usize, Vec<Sample>> = Default::default();
        for s in data.samples {
            by_game.entry(s.game).or_default().push(s);
        }
        for (_, game) in by_game {
            self.games.push_back(game);
            while self.games.len() > self.capacity {
                self.games.pop_front();
            }
        }
    }

    pub fn games(&self) -> usize {
        self.games.len()
    }

    pub fn positions(&self) -> usize {
        self.games.iter().map(Vec::len).sum()
    }

    /// Everything in the buffer as one dataset, game ids renumbered from 0.
    pub fn dataset(&self) -> Dataset {
        let mut samples = Vec::with_capacity(self.positions());
        for (game, positions) in self.games.iter().enumerate() {
            samples.extend(positions.iter().map(|s| Sample { game, ..s.clone() }));
        }
        Dataset { samples }
    }
}

#[derive(Clone, Debug)]
pub struct LoopConfig {
    pub spec: ModelSpec,
    pub backend: BackendKind,
    /// Where `best.*`, `gen-<n>.*` and `history.jsonl` live.
    pub dir: PathBuf,
    /// Continue from `dir/best` and the existing history instead of fresh weights.
    pub resume: bool,
    pub generations: usize,
    /// Self-play games of the first generation.
    pub games: usize,
    /// Games schedule: the count grows by this factor per generation (`1.0` keeps it fixed)
    /// up to `max_games`.
    pub games_growth: f64,
    pub max_games: usize,
    pub selfplay: SelfPlayConfig,
    /// Games kept in the replay buffer when `buffer_generations` is zero.
    pub buffer_games: usize,
    /// When nonzero, the buffer keeps the games of the last this many generations instead
    /// of a fixed game count (so that it follows the games schedule).
    pub buffer_generations: usize,
    /// Training options of the first generation; the learning rate then follows the schedule.
    pub train: TrainOptions,
    /// Learning-rate schedule: multiplied by this factor per generation (`1.0` keeps it
    /// fixed), never below `lr_min`.
    pub lr_decay: f64,
    pub lr_min: f64,
    /// Promotion match: games, simulations per move, per-tree batch.
    pub eval_games: usize,
    pub eval_simulations: u32,
    pub eval_batch: usize,
    /// Opening randomness of the promotion and baseline matches, so that their games differ.
    pub eval_exploration: MatchExploration,
    /// The candidate replaces the best model when its match score reaches this.
    pub promote_threshold: f32,
    /// Fixed opponents the best model is also measured against each generation.
    pub baselines: Vec<AgentSpec>,
    /// Games of each baseline match.
    pub baseline_games: usize,
    /// Stop after the generation that is expected to exhaust this much wall-clock time (the
    /// next generation is estimated from the mean so far).
    pub time_budget: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationReport {
    pub generation: usize,
    /// Learning rate of this generation's training, from the schedule.
    pub learning_rate: f64,
    pub selfplay: SelfPlayStats,
    pub buffer_games: usize,
    pub buffer_positions: usize,
    pub train: TrainReport,
    /// Candidate against the previous best.
    pub eval: MatchResult,
    pub promoted: bool,
    /// Best model (after promotion) against each baseline, in `LoopConfig::baselines` order.
    pub baseline_scores: Vec<(AgentSpec, f32)>,
    pub seconds: f32,
}

impl GenerationReport {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "generation": self.generation,
            "learning_rate": self.learning_rate,
            "selfplay": {
                "games": self.selfplay.games,
                "positions": self.selfplay.positions,
                "simulations_per_second": self.selfplay.simulations_per_second(),
                "mean_batch": self.selfplay.mean_batch(),
                "seconds": self.selfplay.seconds,
            },
            "buffer": { "games": self.buffer_games, "positions": self.buffer_positions },
            "train": {
                "train_positions": self.train.train_positions,
                "validation_positions": self.train.validation_positions,
                "best_epoch": self.train.best_epoch,
                "validation": {
                    "verdict_accuracy": self.train.validation.verdict_accuracy,
                    "value_mse": self.train.validation.value_mse,
                    "policy_accuracy": self.train.validation.policy_accuracy,
                },
                "baseline": {
                    "verdict_accuracy": self.train.baseline.verdict_accuracy,
                    "value_mse": self.train.baseline.value_mse,
                    "policy_accuracy": self.train.baseline.policy_accuracy,
                },
            },
            "eval": {
                "candidate_score": self.eval.score_a(),
                "wins": self.eval.a_wins,
                "draws": self.eval.draws,
                "losses": self.eval.b_wins,
            },
            "promoted": self.promoted,
            "baselines": self
                .baseline_scores
                .iter()
                .map(|(spec, score)| (spec.to_string(), json!(score)))
                .collect::<serde_json::Map<_, _>>(),
            "seconds": self.seconds,
        })
    }
}

impl LoopConfig {
    /// Self-play games of `generation` (1-based) under the games schedule.
    pub fn games_at(&self, generation: usize) -> usize {
        let grown = self.games as f64 * self.games_growth.powi(generation as i32 - 1);
        (grown.round() as usize).clamp(1, self.max_games.max(1))
    }

    /// Learning rate of `generation` (1-based) under the schedule.
    pub fn learning_rate_at(&self, generation: usize) -> f64 {
        let decayed = self.train.learning_rate * self.lr_decay.powi(generation as i32 - 1);
        decayed.max(self.lr_min)
    }
}

fn best_path(dir: &Path) -> PathBuf {
    dir.join("best")
}

fn generation_path(dir: &Path, generation: usize) -> PathBuf {
    dir.join(format!("gen-{generation}"))
}

/// Copies a saved model (`.bpk` and `.json`) from one path stem to another.
fn copy_model(from: &Path, to: &Path) -> Result<(), String> {
    for ext in ["bpk", "json"] {
        std::fs::copy(from.with_extension(ext), to.with_extension(ext)).map_err(|e| {
            format!(
                "cannot copy {} to {}: {e}",
                from.with_extension(ext).display(),
                to.with_extension(ext).display()
            )
        })?;
    }
    Ok(())
}

/// Number of generations already recorded in `dir/history.jsonl`.
fn recorded_generations(dir: &Path) -> usize {
    std::fs::read_to_string(dir.join("history.jsonl"))
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

/// Runs `cfg.generations` generations. `log` receives progress lines.
pub fn run_loop(
    cfg: &LoopConfig,
    mut log: impl FnMut(String),
) -> Result<Vec<GenerationReport>, String> {
    std::fs::create_dir_all(&cfg.dir)
        .map_err(|e| format!("cannot create {}: {e}", cfg.dir.display()))?;
    let device = cfg.backend.device();
    let best = best_path(&cfg.dir);

    let mut first = 0;
    if cfg.resume && best.with_extension("bpk").exists() {
        let existing = AnyModel::load(&best, &device)?;
        if existing.spec() != cfg.spec {
            return Err(format!(
                "{} holds a {} model, not {}",
                best.display(),
                existing.spec(),
                cfg.spec
            ));
        }
        first = recorded_generations(&cfg.dir);
        log(format!(
            "resuming from {} after {first} generations",
            best.display()
        ));
    } else {
        let model = cfg.spec.init(&device);
        log(format!(
            "generation 0: fresh {} ({} parameters) as the first best model",
            cfg.spec,
            model.num_params()
        ));
        model.save(&best)?;
        let _ = std::fs::remove_file(cfg.dir.join("history.jsonl"));
    }

    // Self-play always uses the best model; training continues from the previous
    // generation's candidate, promoted or not, so that the weights keep moving and a
    // candidate that just missed the threshold is not thrown away.
    let mut last = best.clone();
    if first > 0 {
        let previous = generation_path(&cfg.dir, first);
        if previous.with_extension("bpk").exists() {
            last = previous;
        }
    }

    let mut buffer = ReplayBuffer::new(cfg.buffer_games);
    // Games per generation so far, for a buffer measured in generations.
    let mut recent_games: VecDeque<usize> = VecDeque::new();
    let mut reports = Vec::new();
    let loop_started = Instant::now();
    for generation in first + 1..=first + cfg.generations {
        let started = Instant::now();
        let games = cfg.games_at(generation);
        let learning_rate = cfg.learning_rate_at(generation);
        log(format!(
            "generation {generation}: self-play of {games} games with the best model"
        ));
        let evaluator = AgentSpec::Nn {
            path: best.clone(),
            backend: cfg.backend,
        };
        let selfplay = SelfPlayConfig {
            seed: cfg.selfplay.seed.wrapping_add(generation as u64),
            ..cfg.selfplay.clone()
        };
        let (data, stats) = generate_parallel(&evaluator, games, &selfplay, |_| {})?;
        log(format!(
            "  {} positions from {} games, {:.0} simulations/s, mean batch {:.1}",
            data.len(),
            stats.games,
            stats.simulations_per_second(),
            stats.mean_batch()
        ));
        if cfg.buffer_generations > 0 {
            recent_games.push_back(games);
            while recent_games.len() > cfg.buffer_generations {
                recent_games.pop_front();
            }
            buffer.set_capacity(recent_games.iter().sum());
        }
        buffer.push(data);

        log(format!(
            "  training {} on {} positions from {} buffered games at lr {learning_rate:.2e}",
            last.display(),
            buffer.positions(),
            buffer.games()
        ));
        let train_opts = TrainOptions {
            learning_rate,
            seed: cfg.train.seed.wrapping_add(generation as u64),
            artifact_dir: cfg.dir.join(format!("artifacts-gen-{generation}")),
            ..cfg.train.clone()
        };
        let (candidate, train_report) = train_model(
            &cfg.spec,
            Some(&last),
            &device,
            buffer.dataset(),
            &train_opts,
        )?;
        let candidate_path = generation_path(&cfg.dir, generation);
        candidate.save(&candidate_path)?;
        last = candidate_path.clone();
        log(format!(
            "  validation: verdict accuracy {:.3} (baseline {:.3}), value mse {:.4} (baseline {:.4}), policy accuracy {:.3} (baseline {:.3}); kept epoch {}",
            train_report.validation.verdict_accuracy,
            train_report.baseline.verdict_accuracy,
            train_report.validation.value_mse,
            train_report.baseline.value_mse,
            train_report.validation.policy_accuracy,
            train_report.baseline.policy_accuracy,
            train_report.best_epoch
        ));

        let searcher = |path: &Path| AgentSpec::MctsNn {
            path: path.to_path_buf(),
            backend: cfg.backend,
            simulations: cfg.eval_simulations,
            batch_size: cfg.eval_batch,
            seed: generation as u64,
        };
        let eval = play_match_with(
            &searcher(&candidate_path),
            &searcher(&best),
            cfg.eval_games,
            &cfg.eval_exploration,
            |_| {},
        )?;
        let promoted = eval.score_a() >= cfg.promote_threshold;
        log(format!(
            "  candidate vs best: {}-{}-{} (score {:.3}) -> {}",
            eval.a_wins,
            eval.draws,
            eval.b_wins,
            eval.score_a(),
            if promoted {
                "promoted"
            } else {
                "kept the old best"
            }
        ));
        if promoted {
            copy_model(&candidate_path, &best)?;
        }

        let mut baseline_scores = Vec::with_capacity(cfg.baselines.len());
        for baseline in &cfg.baselines {
            let m = play_match_with(
                &searcher(&best),
                baseline,
                cfg.baseline_games,
                &cfg.eval_exploration,
                |_| {},
            )?;
            log(format!(
                "  best vs {baseline}: {}-{}-{} (score {:.3})",
                m.a_wins,
                m.draws,
                m.b_wins,
                m.score_a()
            ));
            baseline_scores.push((baseline.clone(), m.score_a()));
        }

        let report = GenerationReport {
            generation,
            learning_rate,
            selfplay: stats,
            buffer_games: buffer.games(),
            buffer_positions: buffer.positions(),
            train: train_report,
            eval,
            promoted,
            baseline_scores,
            seconds: started.elapsed().as_secs_f32(),
        };
        append_history(&cfg.dir, &report)?;
        log(format!(
            "  generation {generation} done in {:.0}s",
            report.seconds
        ));
        reports.push(report);

        if let Some(budget) = cfg.time_budget {
            let elapsed = loop_started.elapsed();
            let mean = elapsed / reports.len() as u32;
            if elapsed + mean > budget {
                log(format!(
                    "time budget of {:.0} min reached after {} generations ({:.0} min); the \
                     next one would take about {:.0} min",
                    budget.as_secs_f64() / 60.0,
                    reports.len(),
                    elapsed.as_secs_f64() / 60.0,
                    mean.as_secs_f64() / 60.0
                ));
                break;
            }
        }
    }
    Ok(reports)
}

fn append_history(dir: &Path, report: &GenerationReport) -> Result<(), String> {
    use std::io::Write;
    let path = dir.join("history.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    writeln!(file, "{}", report.to_json())
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;
    use crate::nn::train::temp_artifact_dir;

    #[test]
    fn schedules_grow_games_and_decay_the_learning_rate() {
        let cfg = LoopConfig {
            spec: ModelSpec::Linear,
            backend: BackendKind::preferred(),
            dir: PathBuf::new(),
            resume: false,
            generations: 1,
            games: 100,
            games_growth: 2.0,
            max_games: 500,
            selfplay: SelfPlayConfig::default(),
            buffer_games: 1,
            buffer_generations: 0,
            train: TrainOptions {
                learning_rate: 1e-3,
                ..Default::default()
            },
            lr_decay: 0.5,
            lr_min: 2e-4,
            eval_games: 1,
            eval_simulations: 1,
            eval_batch: 1,
            eval_exploration: MatchExploration::default(),
            promote_threshold: 0.5,
            baselines: vec![],
            baseline_games: 1,
            time_budget: None,
        };
        assert_eq!(
            (1..=5).map(|g| cfg.games_at(g)).collect::<Vec<_>>(),
            [100, 200, 400, 500, 500]
        );
        let lrs: Vec<f64> = (1..=5).map(|g| cfg.learning_rate_at(g)).collect();
        assert_eq!(lrs, [1e-3, 5e-4, 2.5e-4, 2e-4, 2e-4]);
    }

    #[test]
    fn replay_buffer_keeps_the_latest_games() {
        let mut buffer = ReplayBuffer::new(2);
        let mut data = Dataset::default();
        for game in 0..3 {
            for _ in 0..4 {
                data.samples.push(Sample {
                    state: crate::game::BitState::initial(),
                    verdict: super::super::model::Verdict::Draw,
                    policy: vec![],
                    game,
                });
            }
        }
        buffer.push(data);
        assert_eq!((buffer.games(), buffer.positions()), (2, 8));
        let d = buffer.dataset();
        assert_eq!(d.len(), 8);
        assert!(d.samples.iter().all(|s| s.game < 2));
    }

    #[test]
    #[ignore = "trains a network; run with: cargo test --release -- --include-ignored"]
    fn two_tiny_generations_run_end_to_end() {
        let _device = crate::nn::device_lock();
        let dir = temp_artifact_dir("loop");
        let cfg = LoopConfig {
            spec: ModelSpec::Linear,
            backend: BackendKind::preferred(),
            dir: dir.clone(),
            resume: false,
            generations: 2,
            games: 6,
            games_growth: 1.0,
            max_games: usize::MAX,
            selfplay: SelfPlayConfig {
                simulations: 12,
                tree_batch: 4,
                concurrent: 3,
                ..Default::default()
            },
            buffer_games: 8,
            buffer_generations: 0,
            train: TrainOptions {
                epochs: 1,
                tui: false,
                ..Default::default()
            },
            lr_decay: 1.0,
            lr_min: 0.0,
            eval_games: 2,
            eval_simulations: 8,
            eval_batch: 4,
            eval_exploration: MatchExploration {
                temperature_moves: 2,
                root_noise: Some((0.3, 0.25)),
            },
            promote_threshold: 0.5,
            baselines: vec![AgentSpec::Random { seed: 1 }],
            baseline_games: 2,
            time_budget: None,
        };
        let mut lines = Vec::new();
        let reports = run_loop(&cfg, |l| lines.push(l)).unwrap();
        assert_eq!(reports.len(), 2);
        assert!(reports[1].buffer_games <= 8 && reports[1].buffer_games >= reports[0].buffer_games);
        assert!(dir.join("best.bpk").exists() && dir.join("best.json").exists());
        assert!(dir.join("gen-1.bpk").exists() && dir.join("gen-2.bpk").exists());
        assert_eq!(recorded_generations(&dir), 2);
        assert!(reports.iter().all(|r| r.baseline_scores.len() == 1));
        assert!(lines.iter().any(|l| l.contains("candidate vs best")));

        // Resuming continues the numbering and reuses the best model.
        let resumed = run_loop(
            &LoopConfig {
                resume: true,
                generations: 1,
                ..cfg
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(resumed[0].generation, 3);
        assert_eq!(recorded_generations(&dir), 3);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
