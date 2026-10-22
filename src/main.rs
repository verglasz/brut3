use std::io;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use ultimate_ttt_burn::agent::AgentSpec;
use ultimate_ttt_burn::repl;

/// Ultimate Tic Tac Toe agents: play, train, benchmark.
#[derive(Parser)]
#[command(name = "ultimate-ttt-burn", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Play against an agent in the terminal (the default command).
    ///
    /// SPEC is e.g. `mcts 800 3`, `minimax 4`, `greedy`, `random`, `nn models/value cuda`,
    /// `mcts-nn models/value cuda 400 16`. Defaults to `mcts 400`.
    Play {
        #[arg(trailing_var_arg = true)]
        spec: Vec<String>,
    },
    /// Generate self-play games and train the value network on them.
    #[cfg(feature = "nn")]
    Train(TrainArgs),
    /// Play lockstep self-play games with a search agent and save them, without training:
    /// e.g. `selfplay --evaluator random --games 4000 --out models/mcts-200.games`, for
    /// `loop --initial-games` or `train --data`.
    #[cfg(feature = "nn")]
    Selfplay(SelfplayArgs),
    /// Self-play training loop: generate games with the best model, train a candidate on a
    /// replay buffer, promote it if it beats the best; repeat.
    #[cfg(feature = "nn")]
    Loop(LoopArgs),
    /// List the backends compiled into this binary.
    #[cfg(feature = "nn")]
    Backends,
    /// Round-robin tournament between agents, e.g. `benchmark "mcts 400" "nn models/value" random`.
    Benchmark {
        /// Games per pair of agents (colours alternate).
        #[arg(long, default_value_t = 20)]
        games: usize,
        /// Agent specs, one per argument, quoted when they contain spaces.
        #[arg(required = true, num_args = 2..)]
        specs: Vec<String>,
    },
}

#[cfg(feature = "nn")]
#[derive(clap::Args)]
struct LoopArgs {
    /// Backend to run on (default: fastest compiled in).
    #[arg(long)]
    backend: Option<String>,
    /// Model kind for a fresh start, e.g. `board-transformer 64 2 4`.
    #[arg(long, default_value = "board-transformer 64 2 4")]
    model: String,
    /// Directory for `best.*`, `gen-<n>.*`, per-generation artifacts and `history.jsonl`.
    #[arg(long, default_value = "models/loop")]
    dir: std::path::PathBuf,
    /// Continue from `<dir>/best` and the recorded history.
    #[arg(long)]
    resume: bool,
    /// Start a fresh run from this saved model (path stem of its `.bpk`/`.json`) instead of
    /// random weights; its shape must match `--model`.
    #[arg(long, conflicts_with = "resume")]
    init: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 10)]
    generations: usize,
    /// Self-play games of the first generation.
    #[arg(long, default_value_t = 1000)]
    games: usize,
    /// Games per generation grow by this factor each generation (1 keeps them fixed).
    #[arg(long, default_value_t = 1.0)]
    games_growth: f64,
    /// Cap of the games schedule (0 for none).
    #[arg(long, default_value_t = 0)]
    max_games: usize,
    #[arg(long, default_value_t = 200)]
    simulations: u32,
    #[arg(long, default_value_t = 256)]
    concurrent: usize,
    #[arg(long, default_value_t = 8)]
    tree_batch: usize,
    #[arg(long, default_value_t = 16)]
    temperature_moves: usize,
    #[arg(long, default_value_t = 0.3)]
    dirichlet_alpha: f32,
    /// Games kept in the replay buffer (when `--buffer-generations` is 0).
    #[arg(long, default_value_t = 5000)]
    buffer_games: usize,
    /// Keep the games of the last this many generations in the replay buffer instead of a
    /// fixed count, so that the buffer follows the games schedule (0 to use --buffer-games).
    #[arg(long, default_value_t = 0)]
    buffer_generations: usize,
    /// Training epochs per generation, continuing from the previous generation's weights.
    #[arg(long, default_value_t = 2)]
    epochs: usize,
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,
    /// Learning rate of the first generation.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    /// The learning rate is multiplied by this each generation (1 keeps it fixed) ...
    #[arg(long, default_value_t = 1.0)]
    lr_decay: f64,
    /// ... but never goes below this.
    #[arg(long, default_value_t = 0.0)]
    lr_min: f64,
    #[arg(long, default_value_t = 1e-4)]
    weight_decay: f32,
    #[arg(long)]
    no_augment: bool,
    /// Show the Learner dashboard while training each generation.
    #[arg(long)]
    tui: bool,
    /// Promotion match length (colours alternate).
    #[arg(long, default_value_t = 20)]
    eval_games: usize,
    #[arg(long, default_value_t = 200)]
    eval_simulations: u32,
    /// Matches: plies per game sampled by visit count before argmax, so that the games of a
    /// match differ (two deterministic searches otherwise replay one game per colour).
    #[arg(long, default_value_t = 6)]
    eval_temperature_moves: usize,
    /// Matches: Dirichlet alpha of the root noise (0 disables it).
    #[arg(long, default_value_t = 0.0)]
    eval_dirichlet_alpha: f32,
    /// Candidate score (wins + draws/2 over games) needed to replace the best model.
    #[arg(long, default_value_t = 0.55)]
    promote_threshold: f32,
    /// Games files to put into the replay buffer before the first generation (e.g. saved by
    /// `train --data-out` from another agent's games), so that training starts from them.
    #[arg(long, num_args = 1..)]
    initial_games: Vec<std::path::PathBuf>,
    /// Fixed opponent to measure the candidate against each generation, e.g. `mcts 200`;
    /// repeat the flag for several, `none` to skip.
    #[arg(long, default_value = "mcts 200")]
    baseline: Vec<String>,
    /// Games of each baseline match (colours alternate).
    #[arg(long, default_value_t = 10)]
    baseline_games: usize,
    /// Stop once this much wall-clock time is spent, checked between generations (the next
    /// generation is estimated from the mean so far); `--generations` is the other limit.
    #[arg(long)]
    max_minutes: Option<f64>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[cfg(feature = "nn")]
#[derive(clap::Args)]
struct SelfplayArgs {
    /// Leaf evaluator of the search: `random` for playouts, or `nn <weights> [backend]`.
    #[arg(long, default_value = "random")]
    evaluator: String,
    #[arg(long, default_value_t = 1000)]
    games: usize,
    /// Simulations per move.
    #[arg(long, default_value_t = 200)]
    simulations: u32,
    /// Games in flight (the evaluator batch is concurrent / 2 x tree-batch).
    #[arg(long, default_value_t = 256)]
    concurrent: usize,
    #[arg(long, default_value_t = 8)]
    tree_batch: usize,
    /// Plies sampled by visit count before playing the most visited move.
    #[arg(long, default_value_t = 16)]
    temperature_moves: usize,
    /// Dirichlet alpha of the root noise (0 disables it).
    #[arg(long, default_value_t = 0.3)]
    dirichlet_alpha: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Independent self-play runs in parallel, each playing its share of the games with its
    /// own evaluator and seed (0 = one per core).
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Games file to write.
    #[arg(long)]
    out: std::path::PathBuf,
}

#[cfg(feature = "nn")]
#[derive(clap::Args)]
struct TrainArgs {
    /// Backend to train on (default: fastest compiled in). See `backends`.
    #[arg(long)]
    backend: Option<String>,
    /// Agent that plays the self-play games, as a spec like `mcts 50` or `greedy`. Ignored
    /// when `--selfplay-evaluator` or `--data` is given.
    #[arg(long, default_value = "mcts 50")]
    data_agent: String,
    /// Train on these saved games files instead of generating games.
    #[arg(long, num_args = 1..)]
    data: Vec<std::path::PathBuf>,
    /// Save the games trained on to this file (loadable with `--data` or `loop
    /// --initial-games`).
    #[arg(long)]
    data_out: Option<std::path::PathBuf>,
    /// Generate the games by lockstep self-play: MCTS with this agent as the leaf evaluator
    /// (e.g. `nn models/board cuda`, or `random` for playouts), `--concurrent` games at once.
    #[arg(long)]
    selfplay_evaluator: Option<String>,
    /// Self-play: simulations per move.
    #[arg(long, default_value_t = 200)]
    simulations: u32,
    /// Self-play: games in flight, in two groups that alternate between tree work and
    /// evaluation (the evaluator batch is concurrent / 2 x tree-batch).
    #[arg(long, default_value_t = 256)]
    concurrent: usize,
    /// Self-play: leaves claimed per tree per round.
    #[arg(long, default_value_t = 8)]
    tree_batch: usize,
    /// Self-play: plies per game with temperature-1 move sampling before switching to argmax.
    #[arg(long, default_value_t = 16)]
    temperature_moves: usize,
    /// Self-play: Dirichlet alpha of the root noise (0 disables it).
    #[arg(long, default_value_t = 0.3)]
    dirichlet_alpha: f32,
    /// Model to train: `linear`, or `board-transformer`, `cell-transformer`, `board-v2` or
    /// `cell-v2` followed by `[d_model] [layers] [heads]`.
    #[arg(long, default_value = "linear")]
    model: String,
    /// Number of self-play games to generate.
    #[arg(long, default_value_t = 1000)]
    games: usize,
    #[arg(long, default_value_t = 10)]
    epochs: usize,
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,
    /// Learning rate.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    /// AdamW weight decay; raise it (e.g. 1e-2) when a large model overfits a small dataset.
    #[arg(long, default_value_t = 1e-4)]
    weight_decay: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Where to save the weights (the `.bpk` extension is added).
    #[arg(long, default_value = "models/value")]
    out: std::path::PathBuf,
    /// Where the Learner writes logs and checkpoints.
    #[arg(long, default_value = "artifacts/train")]
    artifact_dir: std::path::PathBuf,
    /// Log metrics as plain lines instead of the live dashboard (for scripts and CI).
    #[arg(long)]
    no_tui: bool,
    /// Train only on the positions as played, without the eight board symmetries.
    #[arg(long)]
    no_augment: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command.unwrap_or(Command::Play { spec: Vec::new() }) {
        Command::Play { spec } => play(&spec),
        #[cfg(feature = "nn")]
        Command::Train(args) => train(&args),
        #[cfg(feature = "nn")]
        Command::Selfplay(args) => selfplay(&args),
        #[cfg(feature = "nn")]
        Command::Loop(args) => run_loop(&args),
        Command::Benchmark { games, specs } => benchmark(games, &specs),
        #[cfg(feature = "nn")]
        Command::Backends => {
            use ultimate_ttt_burn::nn::BackendKind;
            for kind in BackendKind::ALL {
                let state = if kind.is_compiled() {
                    "compiled"
                } else {
                    "not compiled"
                };
                let default = if kind == BackendKind::preferred() {
                    " (default)"
                } else {
                    ""
                };
                println!("{kind:<9} {state}{default}");
            }
            Ok(())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn benchmark(games: usize, specs: &[String]) -> Result<(), String> {
    use ultimate_ttt_burn::benchmark::round_robin;
    let specs = specs
        .iter()
        .map(|s| {
            let words: Vec<&str> = s.split_whitespace().collect();
            AgentSpec::parse(&words)
        })
        .collect::<Result<Vec<_>, _>>()?;
    println!("round robin, {games} games per pair:");
    let tournament = round_robin(&specs, games, |m| println!("  {m}"))?;
    print!("\n{}", tournament.table());
    Ok(())
}

fn play(spec: &[String]) -> Result<(), String> {
    let words: Vec<&str> = spec.iter().map(String::as_str).collect();
    let spec = if words.is_empty() {
        AgentSpec::default()
    } else {
        AgentSpec::parse(&words)?
    };
    let stdin = io::stdin();
    repl::run(stdin.lock(), io::stdout().lock(), spec).map_err(|e| e.to_string())
}

#[cfg(feature = "nn")]
fn run_loop(args: &LoopArgs) -> Result<(), String> {
    use ultimate_ttt_burn::nn::{BackendKind, LoopConfig, ModelSpec, TrainOptions, run_loop};
    use ultimate_ttt_burn::selfplay::{MatchExploration, SelfPlayConfig};

    let backend = match &args.backend {
        Some(name) => name.parse::<BackendKind>()?,
        None => BackendKind::preferred(),
    };
    let model_words: Vec<&str> = args.model.split_whitespace().collect();
    let mut baselines = Vec::new();
    for spec in &args.baseline {
        match spec.trim() {
            "" | "none" => {}
            words => {
                let words: Vec<&str> = words.split_whitespace().collect();
                baselines.push(AgentSpec::parse(&words)?);
            }
        }
    }
    let cfg = LoopConfig {
        spec: ModelSpec::parse(&model_words)?,
        backend,
        dir: args.dir.clone(),
        resume: args.resume,
        init: args.init.clone(),
        generations: args.generations,
        games: args.games,
        selfplay: SelfPlayConfig {
            simulations: args.simulations,
            tree_batch: args.tree_batch,
            concurrent: args.concurrent,
            temperature_moves: args.temperature_moves,
            root_noise: (args.dirichlet_alpha > 0.0).then_some((args.dirichlet_alpha, 0.25)),
            seed: args.seed,
            ..Default::default()
        },
        games_growth: args.games_growth,
        max_games: if args.max_games == 0 {
            usize::MAX
        } else {
            args.max_games
        },
        buffer_games: args.buffer_games,
        buffer_generations: args.buffer_generations,
        lr_decay: args.lr_decay,
        lr_min: args.lr_min,
        train: TrainOptions {
            epochs: args.epochs,
            batch_size: args.batch_size,
            learning_rate: args.lr,
            weight_decay: args.weight_decay,
            augment: !args.no_augment,
            tui: args.tui,
            seed: args.seed,
            ..Default::default()
        },
        eval_games: args.eval_games,
        eval_simulations: args.eval_simulations,
        eval_batch: args.tree_batch,
        eval_exploration: MatchExploration {
            temperature_moves: args.eval_temperature_moves,
            root_noise: (args.eval_dirichlet_alpha > 0.0)
                .then_some((args.eval_dirichlet_alpha, 0.25)),
        },
        promote_threshold: args.promote_threshold,
        initial_games: args.initial_games.clone(),
        baselines,
        baseline_games: args.baseline_games,
        time_budget: args
            .max_minutes
            .map(|m| std::time::Duration::from_secs_f64(m * 60.0)),
    };
    let reports = run_loop(&cfg, |line| println!("{line}"))?;
    println!(
        "{} generations done; best model at {}; history in {}",
        reports.len(),
        args.dir.join("best.bpk").display(),
        args.dir.join("history.jsonl").display()
    );
    Ok(())
}

#[cfg(feature = "nn")]
fn selfplay(args: &SelfplayArgs) -> Result<(), String> {
    use std::io::Write;
    use std::time::Instant;
    use ultimate_ttt_burn::nn::{games, generate_parallel};
    use ultimate_ttt_burn::selfplay::SelfPlayConfig;

    let words: Vec<&str> = args.evaluator.split_whitespace().collect();
    let evaluator = AgentSpec::parse(&words)?;
    let cfg = SelfPlayConfig {
        simulations: args.simulations,
        tree_batch: args.tree_batch,
        concurrent: args.concurrent,
        temperature_moves: args.temperature_moves,
        root_noise: (args.dirichlet_alpha > 0.0).then_some((args.dirichlet_alpha, 0.25)),
        seed: args.seed,
        ..Default::default()
    };
    let threads = match args.threads {
        0 => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        n => n,
    }
    .clamp(1, args.games.max(1));
    println!(
        "self-play: {} games, {} simulations/move, {} concurrent x {} leaves, evaluator {evaluator}, {threads} thread(s)",
        args.games, cfg.simulations, cfg.concurrent, cfg.tree_batch
    );
    let start = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let report = |n: usize| {
        if n.is_multiple_of(100) || n == args.games {
            print!("\r  {n}/{} games", args.games);
            let _ = io::stdout().flush();
        }
    };
    // Each thread plays a contiguous share of the games with its own seed; the datasets are
    // concatenated with distinct game ids and the statistics summed.
    let runs: Vec<Result<_, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let games = args.games / threads + usize::from(t < args.games % threads);
                let cfg = SelfPlayConfig {
                    seed: cfg.seed.wrapping_add(t as u64),
                    ..cfg.clone()
                };
                let (evaluator, done, report) = (&evaluator, &done, &report);
                scope.spawn(move || {
                    generate_parallel(evaluator, games, &cfg, |_| {
                        report(1 + done.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("self-play thread panicked"))
            .collect()
    });
    let mut data = ultimate_ttt_burn::nn::Dataset::default();
    let mut stats = ultimate_ttt_burn::selfplay::SelfPlayStats::default();
    for run in runs {
        let (part, part_stats) = run?;
        let offset = data.games();
        data.samples.extend(part.samples.into_iter().map(|s| {
            ultimate_ttt_burn::nn::Sample {
                game: s.game + offset,
                ..s
            }
        }));
        stats.games += part_stats.games;
        stats.positions += part_stats.positions;
        stats.simulations += part_stats.simulations;
        stats.evaluator_calls += part_stats.evaluator_calls;
        stats.leaves += part_stats.leaves;
    }
    stats.seconds = start.elapsed().as_secs_f32();
    games::save(&args.out, &data)?;
    let [wins, draws, losses] = data.verdict_counts();
    println!(
        "\n  {} positions ({wins} win / {draws} draw / {losses} loss) in {:.0}s, {:.0} simulations/s; saved to {}",
        data.len(),
        start.elapsed().as_secs_f32(),
        stats.simulations_per_second(),
        args.out.display()
    );
    Ok(())
}

#[cfg(feature = "nn")]
fn train(args: &TrainArgs) -> Result<(), String> {
    use std::io::Write;
    use std::time::Instant;
    use ultimate_ttt_burn::nn::{BackendKind, ModelSpec, TrainOptions, generate};

    let backend = match &args.backend {
        Some(name) => name.parse::<BackendKind>()?,
        None => BackendKind::preferred(),
    };
    let data_words: Vec<&str> = args.data_agent.split_whitespace().collect();
    let data_spec = AgentSpec::parse(&data_words)?;
    let model_words: Vec<&str> = args.model.split_whitespace().collect();
    let model_spec = ModelSpec::parse(&model_words)?;

    let start = Instant::now();
    let progress = |n: usize| {
        if n.is_multiple_of(100) || n == args.games {
            print!("\r  {n}/{} games", args.games);
            let _ = io::stdout().flush();
        }
    };
    let data = match &args.selfplay_evaluator {
        _ if !args.data.is_empty() => {
            let data = ultimate_ttt_burn::nn::games::load_many(&args.data)?;
            println!(
                "loaded {} games from {} file(s)",
                data.games(),
                args.data.len()
            );
            data
        }
        Some(evaluator) => {
            use ultimate_ttt_burn::nn::generate_parallel;
            use ultimate_ttt_burn::selfplay::SelfPlayConfig;
            let words: Vec<&str> = evaluator.split_whitespace().collect();
            let evaluator_spec = AgentSpec::parse(&words)?;
            let cfg = SelfPlayConfig {
                simulations: args.simulations,
                tree_batch: args.tree_batch,
                concurrent: args.concurrent,
                temperature_moves: args.temperature_moves,
                root_noise: (args.dirichlet_alpha > 0.0).then_some((args.dirichlet_alpha, 0.25)),
                seed: args.seed,
                ..Default::default()
            };
            println!(
                "self-play: {} games, {} simulations/move, {} concurrent x {} leaves, evaluator {evaluator_spec}",
                args.games, cfg.simulations, cfg.concurrent, cfg.tree_batch
            );
            let (data, stats) = generate_parallel(&evaluator_spec, args.games, &cfg, progress)?;
            println!(
                "\n  {:.0} simulations/s, mean evaluator batch {:.1} over {} calls",
                stats.simulations_per_second(),
                stats.mean_batch(),
                stats.evaluator_calls
            );
            data
        }
        None => {
            println!("generating {} self-play games with {data_spec}", args.games);
            generate(&data_spec, args.games, progress)?
        }
    };
    let [wins, draws, losses] = data.verdict_counts();
    println!(
        "\n  {} positions ({wins} win / {draws} draw / {losses} loss) in {:.1}s",
        data.len(),
        start.elapsed().as_secs_f32(),
    );
    if let Some(out) = &args.data_out {
        ultimate_ttt_burn::nn::games::save(out, &data)?;
        println!("  saved to {}", out.display());
    }
    let device = backend.device();
    println!(
        "training {model_spec} ({} parameters) on {backend} for {} epochs",
        model_spec.init(&device).num_params(),
        args.epochs
    );
    let opts = TrainOptions {
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        weight_decay: args.weight_decay,
        seed: args.seed,
        artifact_dir: args.artifact_dir.clone(),
        tui: !args.no_tui,
        augment: !args.no_augment,
        ..Default::default()
    };
    let (model, report) =
        ultimate_ttt_burn::nn::train_model(&model_spec, None, &device, data, &opts)?;
    println!(
        "  {} train / {} validation positions; kept epoch {} of {} (lowest validation loss)",
        report.train_positions, report.validation_positions, report.best_epoch, report.epochs
    );
    println!(
        "  validation: verdict accuracy {:.3} (baseline {:.3}), value mse {:.4} (baseline {:.4}), policy accuracy {:.3} (baseline {:.3})",
        report.validation.verdict_accuracy,
        report.baseline.verdict_accuracy,
        report.validation.value_mse,
        report.baseline.value_mse,
        report.validation.policy_accuracy,
        report.baseline.policy_accuracy
    );

    if let Some(dir) = args.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    model.save(&args.out)?;
    println!(
        "saved weights to {}.bpk (spec in .json)",
        args.out.display()
    );
    Ok(())
}
