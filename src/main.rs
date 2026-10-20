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
    /// List the backends compiled into this binary.
    #[cfg(feature = "nn")]
    Backends,
}

#[cfg(feature = "nn")]
#[derive(clap::Args)]
struct TrainArgs {
    /// Backend to train on (default: fastest compiled in). See `backends`.
    #[arg(long)]
    backend: Option<String>,
    /// Agent that plays the self-play games, as a spec like `mcts 50` or `greedy`.
    #[arg(long, default_value = "mcts 50")]
    data_agent: String,
    /// Number of self-play games to generate.
    #[arg(long, default_value_t = 1000)]
    games: usize,
    #[arg(long, default_value_t = 10)]
    epochs: usize,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    /// Learning rate. Much above 1e-3 saturates the tanh output and training stalls.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Where to save the weights (the `.mpk` extension is added).
    #[arg(long, default_value = "models/value")]
    out: std::path::PathBuf,
    /// Where the Learner writes logs and checkpoints.
    #[arg(long, default_value = "artifacts/train")]
    artifact_dir: std::path::PathBuf,
    /// Log metrics as plain lines instead of the live dashboard (for scripts and CI).
    #[arg(long)]
    no_tui: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command.unwrap_or(Command::Play { spec: Vec::new() }) {
        Command::Play { spec } => play(&spec),
        #[cfg(feature = "nn")]
        Command::Train(args) => train(&args),
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
fn train(args: &TrainArgs) -> Result<(), String> {
    use std::io::Write;
    use std::time::Instant;
    use ultimate_ttt_burn::nn::{BackendKind, TrainOptions, generate, train};

    let backend = match &args.backend {
        Some(name) => name.parse::<BackendKind>()?,
        None => BackendKind::preferred(),
    };
    let data_words: Vec<&str> = args.data_agent.split_whitespace().collect();
    let data_spec = AgentSpec::parse(&data_words)?;

    println!("generating {} self-play games with {data_spec}", args.games);
    let start = Instant::now();
    let data = generate(&data_spec, args.games, |n| {
        if n % 100 == 0 || n == args.games {
            print!("\r  {n}/{} games", args.games);
            let _ = io::stdout().flush();
        }
    })?;
    println!(
        "\n  {} positions in {:.1}s; baseline mse (predict mean) {:.4}",
        data.len(),
        start.elapsed().as_secs_f32(),
        data.baseline_mse()
    );

    println!("training on {backend} for {} epochs", args.epochs);
    let device = backend.device();
    let opts = TrainOptions {
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        seed: args.seed,
        artifact_dir: args.artifact_dir.clone(),
        tui: !args.no_tui,
        ..Default::default()
    };
    let (model, report) = train(&device, data, &opts);
    println!(
        "  {} train / {} validation positions; validation mse {:.4} (baseline {:.4})",
        report.train_positions,
        report.validation_positions,
        report.validation_mse,
        report.baseline_mse
    );

    if let Some(dir) = args.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    model
        .save(&args.out)
        .map_err(|e| format!("cannot save weights to {}: {e}", args.out.display()))?;
    println!("saved weights to {}.mpk", args.out.display());
    Ok(())
}
