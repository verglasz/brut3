use std::io;
use std::process::ExitCode;

use brut3::agent::AgentSpec;
use brut3::repl;
use clap::{Parser, Subcommand};

/// Ultimate Tic Tac Toe agents: play, train, benchmark.
#[derive(Parser)]
#[command(name = "brut3", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Play against an agent in the terminal (the default command).
    ///
    /// SPEC is e.g. `mcts 800 3`, `minimax 4`, `greedy`, `random`, `nn models/value cuda`,
    /// `mcts-nn models/value cuda 400 16`; `mcts`, `nn` and `mcts-nn` may end with
    /// `book <file>` to play an opening book's moves without searching. Defaults to
    /// `mcts 400`.
    #[command(after_help = brut3::agent::SPEC_SYNTAX)]
    Play {
        #[arg(trailing_var_arg = true)]
        spec: Vec<String>,
    },
    /// Generate self-play games and train the value network on them.
    #[cfg(feature = "nn")]
    #[command(after_help = brut3::agent::SPEC_SYNTAX)]
    Train(TrainArgs),
    /// Play lockstep self-play games with a search agent and save them, without training:
    /// e.g. `selfplay --evaluator random --games 4000 --out models/mcts-200.games`, for
    /// `loop --initial-games` or `train --data`.
    #[cfg(feature = "nn")]
    #[command(after_help = brut3::agent::SPEC_SYNTAX)]
    Selfplay(SelfplayArgs),
    /// Self-play training loop: generate games with the best model, train a candidate on a
    /// replay buffer, promote it if it beats the best; repeat.
    #[cfg(feature = "nn")]
    #[command(after_help = brut3::agent::SPEC_SYNTAX)]
    Loop(LoopArgs),
    /// Convert the uttt.ai text datasets (stage1-mcts, stage2-nmcts) into a games file for
    /// `train --data`: e.g. `import-utttai --out datasets/stage1.games datasets/stage1-mcts`.
    #[cfg(feature = "nn")]
    ImportUtttai {
        /// Dataset directories (every `.txt` inside) or files.
        #[arg(required = true, num_args = 1..)]
        paths: Vec<std::path::PathBuf>,
        /// Games file to write.
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// List the backends compiled into this binary.
    #[cfg(feature = "nn")]
    Backends,
    /// Export an nnue or nnue-v2 model and/or an opening book as generated Haskell sources for the
    /// base-only engine.
    #[cfg(feature = "nn")]
    Export {
        /// Model stem (`<stem>.bpk` + `<stem>.json`), an nnue or nnue-v2 model.
        #[arg(long, required_unless_present_any = ["book", "vec_extra", "vec_flags"])]
        model: Option<std::path::PathBuf>,
        /// Opening book file (from `book`) to export as Gen/Book.hs.
        #[arg(long)]
        book: Option<std::path::PathBuf>,
        /// Directory to write `<module>/*.hs` under, or `-` to print every file to stdout
        /// instead, each after a line `-- file: <module>/<name>.hs` with a blank line around
        /// it (messages always go to stderr).
        #[arg(long)]
        out: std::path::PathBuf,
        /// Module prefix and directory name.
        #[arg(long, default_value = "Gen")]
        module: String,
        /// Backend to load the model with (default: fastest compiled in).
        #[arg(long)]
        backend: Option<String>,
        /// Reference positions to emit for the port's tests.
        #[arg(long, default_value_t = 16)]
        positions: usize,
        /// Seed of the random games the reference positions come from.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Reference book lookups to emit for the port's tests.
        #[arg(long, default_value_t = 40)]
        lookups: usize,
        /// The same flags as VEC_FLAGS (after `--`), as one quoted string, to give them among
        /// the other options: `--vec-extra "--int32 512 --rows 9:128"`. Both may be given.
        #[arg(long, allow_hyphen_values = true)]
        vec_extra: Option<String>,
        /// Do not write FixedVec.hs (keep a hand-generated one).
        #[arg(long)]
        no_fixed_vec: bool,
        /// Extra FixedVec.hs shapes for hand-written code, on top of the model's, after `--`:
        /// `export --model m --out lib -- --int32 512 --rows 9:128`. Without --model,
        /// FixedVec.hs holds only these. Widths that a flag needs get their vector type too.
        /// Flags:
        ///   --float N ..     VecN of Float: fromListN, fromListRestN, toListN, indexN,
        ///                    replicateN, mapN, combineN, addN, subN, sumN, dotN, scaleN,
        ///                    reluN, and for flat N <= 64 maskedMaxN, maskedExpN
        ///   --int32 N ..     I32VecN of Int32: i32FromListN .. i32DotN (no scale, relu,
        ///                    masked)
        ///   --int16 N ..     I16VecN of Int16: i16FromListN .. i16DotN, likewise
        ///   --nest N=KxM,..  VecN as K boxed chunks of VecM, its functions built from VecM's
        ///                    (GHC compiles wide flat vectors badly: nest widths above 64,
        ///                    in 32-lane chunks as the model's export does)
        ///   --rows M:N,..    RowsM_N, M rows of VecN (one row per output), with
        ///                    fromListRowsM_N and selM_N i (row i)
        ///   --dense M:N,..   layers from N inputs to M outputs: linearM_N rows bias x =
        ///                    bias + rows . x, denseM_N = relu of that, and when M = N
        ///                    residualM_M = relu (x + bias + rows . x); adds RowsM_N
        ///   --concat K:N,..  concatK_N: K vectors of width N into one of width K * N
        ///   --sum K:N,..     sumK_N: the componentwise sum of K vectors of width N
        ///   --axpy K:N,..    axpyK_N s0 c0 .. acc = acc + s0 * c0 + .. (K scaled vectors)
        ///   --cols M:N,..    ColsN_M, the columns of a layer with M outputs and N inputs, and
        ///                    columnsM_N, their transposition from RowsM_N (run once, at load)
        ///   --sparse M:N[:K],.. sparseDenseM_N, and when M = N sparseResidualM: denseM_N and
        ///                    residualM_M computed by columns over the nonzero inputs only,
        ///                    K columns per pass (default 8), the leftovers in passes of
        ///                    K/2, K/4, .., 1; adds the columns, their PairsM list and the axpys
        ///   --alias NAME=N,.. type NAME = VecN
        #[arg(last = true, verbatim_doc_comment)]
        vec_flags: Vec<String>,
    },
    /// Build or deepen an opening book by searching the positions of the first plies with a
    /// search agent, e.g. `book --search "mcts-nn models/big cuda 100000 16" --out models/book.txt`.
    /// An existing book file is extended, and its entries to --plies whose search had fewer
    /// simulations are searched again with their visit shares as the root priors.
    #[command(after_help = brut3::agent::SPEC_SYNTAX)]
    Book {
        /// The search that evaluates each position: `mcts [simulations] [seed]` or
        /// `mcts-nn <weights> [backend] [simulations] [batch] [seed]`.
        #[arg(long)]
        search: String,
        /// Book file to extend (if it exists) and write.
        #[arg(long)]
        out: std::path::PathBuf,
        /// Deepest ply to hold entries for (the initial position is ply 0).
        #[arg(long, default_value_t = 8)]
        plies: usize,
        /// Smallest share of a position's visits for an opponent reply to be covered.
        #[arg(long, default_value_t = 0.01)]
        min_prior: f32,
        /// Cover every opponent reply at the plies below this one, without searching the
        /// positions where only the opponent moves (they are marked open).
        #[arg(long, default_value_t = 0)]
        dense_plies: usize,
        /// Simulations for the positions where only the opponent moves, past --dense-plies,
        /// searched for their replies' shares (default: the search's own).
        #[arg(long)]
        reply_simulations: Option<u32>,
        /// Positions searched at once per thread (the evaluator batch is concurrent / 2 x the
        /// search's batch, per thread).
        #[arg(long, default_value_t = 256)]
        concurrent: usize,
        /// Search threads, each taking its share of every round's positions (0 = one per
        /// core). With a network they share one evaluator thread, which batches all their
        /// leaves together.
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Only extend the main lines: new positions must be reached by their parent's book
        /// move, whichever side plays it, so each round deepens the principal variation from
        /// every leaf (to --plies) without opening new reply branches.
        #[arg(long)]
        main_lines: bool,
        /// Keep every entry already in the book whatever its search's budget: only missing
        /// positions are searched (dropped entries come back from their stored search). By
        /// default, entries a search of this budget could improve on are searched again.
        #[arg(long)]
        keep_searched: bool,
    },
    /// Mark the entries of an opening book that the shipped book can do without, e.g.
    /// `prune-book --book models/book.txt --agent "mcts-nn models/m cuda 800 16" --out
    /// models/book.txt`. Dropped entries stay in the file, marked: lookups and the export
    /// skip them, while `book` and the walks of later passes still use them, so the book can
    /// be pruned in passes (say, different settings from different plies) and deepened again.
    #[command(after_help = brut3::agent::SPEC_SYNTAX)]
    PruneBook(PruneArgs),
    /// Show what an opening book file holds, without searching: its header, its entries,
    /// dropped entries and open positions by ply, and the visits of their searches.
    BookStats {
        /// Book file to read.
        book: std::path::PathBuf,
    },
    /// Round-robin tournament between agents, e.g. `benchmark "mcts 400" "nn models/value" random`.
    #[command(after_help = brut3::agent::SPEC_SYNTAX)]
    Benchmark {
        /// Games per pair of agents (colours alternate; game g uses each agent's seed + g).
        #[arg(long, default_value_t = 20)]
        games: usize,
        /// Search vs search: opening plies per game sampled by visit count before argmax, so
        /// that the games of a match differ (0 plays every move greedily, which between two
        /// deterministic searches repeats one game per colour).
        #[arg(long, default_value_t = 6)]
        temperature_moves: usize,
        /// Search vs search: Dirichlet alpha of the root noise (0 disables it).
        #[arg(long, default_value_t = 0.0)]
        dirichlet_alpha: f32,
        /// Search vs search: threads each playing its share of a match's games (0 = one per
        /// core). A network runs on one thread of its own that batches them all.
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Agent specs, one per argument, quoted when they contain spaces.
        #[arg(required = true, num_args = 2..)]
        specs: Vec<String>,
    },
    /// Serve a web page for playing against agents, e.g.
    /// `serve "mcts 800" "mcts-nn models/x cuda 400 16 book models/book.txt"`. Each side of
    /// the board is a human or one of the agents, chosen on the page.
    #[command(after_help = brut3::agent::SPEC_SYNTAX)]
    Serve {
        /// Address to listen on.
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Agent specs, one per argument, quoted when they contain spaces.
        #[arg(required = true, num_args = 1..)]
        specs: Vec<String>,
    },
}

#[derive(clap::Args)]
struct PruneArgs {
    /// Book file to prune.
    #[arg(long)]
    book: std::path::PathBuf,
    /// Book file to write, with the drops marked; may be the same as --book.
    #[arg(long)]
    out: std::path::PathBuf,
    /// Drop the entries this agent plays anyway (up to the position's symmetries), as on
    /// the playing side: search agents (`mcts`, `mcts-nn <weights> [backend] [simulations]
    /// [batch]`) run batched, any other agent spec position by position.
    #[arg(long)]
    agent: Option<String>,
    /// Margin on the agent's agreement: drop an entry only if the book move's share of the
    /// agent's visits or policy leads its second-best move by at least this much (each with
    /// its symmetric equivalents), so that moves it barely prefers stay in the book for a
    /// weaker or noisier search at play time.
    #[arg(long, default_value_t = 0.0)]
    min_advantage: f32,
    /// Keep the common lines whatever the agent plays, to save search time where games
    /// usually go: every entry reached from the start through opponent replies that each had
    /// at least this share of the book search's visits at their ply (judged reply by reply,
    /// so two 10% replies in a row stay with 0.05). These are not searched.
    #[arg(long)]
    keep_replies: Option<f32>,
    /// Drop, without searching, the entries known to lie behind a rarer opponent reply: those
    /// the book's lines reach (walking every reply the book has) but not the lines of
    /// replies with at least this share of the visits each (the same reply-by-reply walk as
    /// --keep-replies). Entries off the book's lines are left alone.
    #[arg(long)]
    drop_rare: Option<f32>,
    /// Only drop entries at this ply or deeper (marks on the board).
    #[arg(long, alias = "ply", default_value_t = 0)]
    from_ply: usize,
    /// Positions searched at once by a search agent.
    #[arg(long, default_value_t = 256)]
    concurrent: usize,
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
    /// Self-play games in flight per thread.
    #[arg(long, default_value_t = 1024)]
    concurrent: usize,
    /// Self-play search threads (0 = one per core); the network runs on one thread of its own
    /// that batches them all.
    #[arg(long, default_value_t = 1)]
    threads: usize,
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
    /// Training steps between two samples of the gradient-norm and update-size metrics (0
    /// for none): sampling every step makes a step about six times slower.
    #[arg(long, default_value_t = brut3::nn::train_stats::DEFAULT_EVERY)]
    stats_every: usize,
    /// Symmetry augmentation: `all` (eight copies, for small datasets), `random` (one random
    /// symmetry per sample per epoch, for large ones) or `none`.
    #[arg(long, default_value = "all")]
    augment: String,
    /// Weight of the outcome cross-entropy in the loss (0 with --search-value-weight trains
    /// the value head on the search's values only).
    #[arg(long, default_value_t = 1.0)]
    outcome_weight: f32,
    /// Weight of the squared error against the self-play search's root value in the loss
    /// (0 ignores the search values).
    #[arg(long, default_value_t = 0.5)]
    search_value_weight: f32,
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
    /// Games in flight per thread (the evaluator batch is concurrent / 2 x tree-batch).
    #[arg(long, default_value_t = 1024)]
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
    /// Search threads, each playing its share of the games with its own seed (0 = one per
    /// core). A network evaluator runs on one thread of its own that batches them all.
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
    #[arg(long, default_value_t = 1024)]
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
    /// Model to train: `linear`; `board-transformer`, `cell-transformer`, `board-v2` or
    /// `cell-v2` followed by `[d_model] [layers] [heads]`; or `nnue` followed by
    /// `[d_embed] [d_acc] [d_hidden] [layers]`.
    #[arg(long, default_value = "linear")]
    model: String,
    /// Continue training from this saved model (path stem of its `.bpk`/`.json`); its shape
    /// must match `--model`.
    #[arg(long)]
    init: Option<std::path::PathBuf>,
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
    /// Cosine-anneal the learning rate down to this over the run (0 keeps it constant).
    #[arg(long, default_value_t = 0.0)]
    lr_min: f64,
    /// AdamW weight decay; raise it (e.g. 1e-2) when a large model overfits a small dataset.
    #[arg(long, default_value_t = 1e-4)]
    weight_decay: f32,
    /// Training steps between two samples of the gradient-norm and update-size metrics (0
    /// for none): sampling every step makes a step about six times slower.
    #[arg(long, default_value_t = brut3::nn::train_stats::DEFAULT_EVERY)]
    stats_every: usize,
    /// Fraction of the positions held out for validation.
    #[arg(long, default_value_t = 0.1)]
    validation_fraction: f32,
    /// Data loader threads (0 = one per core).
    #[arg(long, default_value_t = 0)]
    workers: usize,
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
    /// Symmetry augmentation: `all` (eight copies, for small datasets), `random` (one random
    /// symmetry per sample per epoch, for large ones) or `none`.
    #[arg(long, default_value = "all")]
    augment: String,
    /// Weight of the outcome cross-entropy in the loss (0 with --search-value-weight is
    /// distillation: the value head learns the generating search's values only).
    #[arg(long, default_value_t = 1.0)]
    outcome_weight: f32,
    /// Weight of the squared error against the generating search's root value in the loss
    /// (0 ignores them; positions without a search value use the outcome).
    #[arg(long, default_value_t = 0.5)]
    search_value_weight: f32,
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
        Command::Benchmark {
            games,
            temperature_moves,
            dirichlet_alpha,
            threads,
            specs,
        } => benchmark(games, temperature_moves, dirichlet_alpha, threads, &specs),
        Command::Serve { addr, specs } => serve(&addr, &specs),
        Command::Book {
            search,
            out,
            plies,
            min_prior,
            dense_plies,
            reply_simulations,
            concurrent,
            threads,
            main_lines,
            keep_searched,
        } => book(
            &search,
            &out,
            brut3::book::BookConfig {
                plies,
                min_prior,
                dense_plies,
                reply_simulations,
                concurrent,
                main_lines,
                keep_searched,
            },
            threads,
        ),
        Command::PruneBook(args) => prune_book(&args),
        Command::BookStats { book } => book_stats(&book),
        #[cfg(feature = "nn")]
        Command::ImportUtttai { paths, out } => import_utttai(&paths, &out),
        #[cfg(feature = "nn")]
        Command::Export {
            model,
            book,
            out,
            module,
            backend,
            positions,
            seed,
            lookups,
            vec_extra,
            no_fixed_vec,
            vec_flags,
        } => export(
            model.as_deref(),
            book.as_deref(),
            &out,
            module,
            backend.as_deref(),
            positions,
            seed,
            lookups,
            vec_extra
                .iter()
                .flat_map(|s| s.split_whitespace().map(str::to_string))
                .chain(vec_flags)
                .collect(),
            !no_fixed_vec,
        ),
        #[cfg(feature = "nn")]
        Command::Backends => {
            use brut3::nn::BackendKind;
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

fn book_stats(path: &std::path::Path) -> Result<(), String> {
    use std::collections::{BTreeMap, HashMap};

    use brut3::book::{Book, Reached};

    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let book = Book::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("{}", path.display());
    for line in text.lines().map_while(|l| l.strip_prefix('#')) {
        println!("  # {}", line.trim());
    }
    println!(
        "{} entries ({} live, {} dropped), {} open positions{}",
        book.len(),
        book.live_len(),
        book.len() - book.live_len(),
        book.open_len(),
        if book.coalesced() > 0 {
            format!(", {} lines pooled into others' entries", book.coalesced())
        } else {
            String::new()
        }
    );

    #[derive(Default)]
    struct Ply {
        live: usize,
        dropped: usize,
        open: usize,
        visits: Vec<u32>,
    }
    let mut by_ply: BTreeMap<usize, Ply> = BTreeMap::new();
    // How many searches had each budget (visits + 1: the root's own evaluation).
    let mut budgets: HashMap<u32, usize> = HashMap::new();
    let mut open_reached = 0;
    for (key, ply) in book.plies() {
        let row = by_ply.entry(ply).or_default();
        match book.get(&key) {
            Some(entry) => {
                let visits: u32 = entry.visits.iter().map(|(_, n)| n).sum();
                if entry.dropped {
                    row.dropped += 1;
                } else {
                    row.live += 1;
                }
                row.visits.push(visits);
                *budgets.entry(visits + 1).or_default() += 1;
            }
            None => {
                row.open += 1;
                open_reached += 1;
            }
        }
    }
    println!();
    println!(
        "{:>4} {:>8} {:>8} {:>8}   {:>9} {:>9} {:>9}",
        "ply", "live", "dropped", "open", "min", "median", "max"
    );
    for (ply, row) in &mut by_ply {
        row.visits.sort_unstable();
        let at = |q: f64| {
            row.visits
                .get(((row.visits.len().saturating_sub(1)) as f64 * q) as usize)
                .map_or("-".to_string(), |v| v.to_string())
        };
        println!(
            "{ply:>4} {:>8} {:>8} {:>8}   {:>9} {:>9} {:>9}",
            row.live,
            row.dropped,
            row.open,
            at(0.0),
            at(0.5),
            at(1.0)
        );
    }
    println!(
        "(min, median and max of the entries' visits; a ply is the fewest moves reaching a position)"
    );
    if open_reached < book.open_len() {
        println!(
            "{} open positions not reached from the start",
            book.open_len() - open_reached
        );
    }

    // Where play leaves the book, by side and ply: how often (the reach of its leaves) and
    // how (by the book's own last move, by a reply it does not cover, or out of an open
    // position not finished yet).
    #[derive(Default)]
    struct Leaves {
        count: usize,
        reach: f64,
        book_move: f64,
        reply: f64,
        open_reply: f64,
    }
    let mut leaves: BTreeMap<(usize, usize), Leaves> = BTreeMap::new();
    let mut unfinished = 0;
    for leaf in book.open_leaves() {
        let side = usize::from(leaf.book == brut3::game::Player::O);
        let row = leaves.entry((leaf.ply, side)).or_default();
        row.count += 1;
        row.reach += leaf.reach;
        match leaf.reached {
            Reached::BookMove => row.book_move += leaf.reach,
            Reached::Reply(_) => row.reply += leaf.reach,
            Reached::OpenReply => {
                row.open_reply += leaf.reach;
                unfinished += 1;
            }
        }
    }
    println!();
    println!(
        "open leaves: where play leaves the book (the book's moves against an opponent replying by the searches' visit shares)"
    );
    let pct = |p: f64| format!("{:.2}%", 100.0 * p);
    for (side, name) in [(0, "X"), (1, "O")] {
        println!("  book as {name}:");
        println!(
            "  {:>4} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "ply", "leaves", "leave", "by move", "by reply", "open", "in book"
        );
        let mut inside = 1.0;
        for ((ply, _), row) in leaves.iter().filter(|((_, s), _)| *s == side) {
            inside -= row.reach;
            println!(
                "  {ply:>4} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
                row.count,
                pct(row.reach),
                pct(row.book_move),
                pct(row.reply),
                pct(row.open_reply),
                pct(inside.max(0.0))
            );
        }
    }
    println!(
        "(leave: the probability that play leaves the book at that ply, by the book's own last move, by a reply it does not cover or by one out of an open position not searched yet; in book: still in it after that ply, or in a game that ended inside it)"
    );
    if unfinished > 0 {
        println!("{unfinished} open leaves after open positions: run `book` again to search them");
    }

    // The budgets most searches had; the rest (early stops, proven roots, pooled lines) in one.
    let mut common: Vec<(u32, usize)> = budgets.into_iter().collect();
    common.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let floor = (book.len() / 200).max(1);
    let shown = common
        .iter()
        .take(8)
        .take_while(|(_, n)| *n >= floor)
        .count();
    println!();
    println!("searches by simulations (visits + 1):");
    for (budget, n) in &common[..shown] {
        println!("  {budget:>9} {n:>8}");
    }
    let rest = &common[shown..];
    if !rest.is_empty() {
        let count: usize = rest.iter().map(|(_, n)| n).sum();
        let lo = rest.iter().map(|(b, _)| b).min().expect("some");
        let hi = rest.iter().map(|(b, _)| b).max().expect("some");
        println!("  {count:>18} more, from {lo} to {hi}");
    }
    Ok(())
}

fn prune_book(args: &PruneArgs) -> Result<(), String> {
    use std::collections::{BTreeMap, HashSet};
    use std::io::Write;
    use std::time::Instant;

    use brut3::agent::UtttAgent;
    use brut3::book::{Book, agent_agrees};
    use brut3::game::{BitState, Move};
    use brut3::selfplay::search_positions;

    if args.agent.is_none() && args.drop_rare.is_none() {
        return Err("nothing to drop: give --agent and/or --drop-rare".into());
    }
    let spec = match &args.agent {
        Some(agent) => {
            let words: Vec<&str> = agent.split_whitespace().collect();
            let spec = AgentSpec::parse(&words)?;
            if spec.book_path().is_some() {
                return Err(format!(
                    "{spec} plays from a book itself; give it without `book`"
                ));
            }
            Some(spec)
        }
        None => None,
    };
    let path = &args.book;
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut header: Vec<String> = text
        .lines()
        .map_while(|l| l.strip_prefix('#'))
        .map(|l| l.trim().to_string())
        .collect();
    let mut book = Book::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let live_before = book.live_len();
    if book.coalesced() > 0 {
        println!(
            "{} lines pooled into others' entries (the same position up to decided boards' cells and colours)",
            book.coalesced()
        );
    }
    let plies = book.plies();
    let ply_of = |key: &str| plies[key];
    let deep = |key: &str| ply_of(key) >= args.from_ply;

    // Per ply: entries, live before, common, dropped as rare, dropped by the agent.
    let mut per_ply: BTreeMap<usize, [usize; 5]> = BTreeMap::new();
    for (key, entry) in book.all_entries() {
        let counts = per_ply.entry(ply_of(key)).or_default();
        counts[0] += 1;
        counts[1] += usize::from(!entry.dropped);
    }
    let start = Instant::now();

    // Rare lines go without a search: entries on the book's lines but off the frequent ones.
    if let Some(share) = args.drop_rare {
        let all_lines = book.common_lines(0.0);
        let lines = book.common_lines(share);
        let rare: Vec<String> = book
            .entries()
            .filter(|(key, _)| deep(key) && all_lines.contains(*key) && !lines.contains(*key))
            .map(|(key, _)| key.clone())
            .collect();
        for key in rare {
            if book.drop_entry(&key) {
                per_ply.entry(ply_of(&key)).or_default()[3] += 1;
            }
        }
    }

    let common = args
        .keep_replies
        .map_or_else(HashSet::new, |share| book.common_lines(share));
    for key in &common {
        if book.get(key).is_some_and(|e| !e.dropped) {
            per_ply.entry(ply_of(key)).or_default()[2] += 1;
        }
    }
    if let Some(spec) = &spec {
        // Only the live entries off the common lines from --from-ply on are searched.
        let positions: Vec<_> = book
            .positions()
            .into_iter()
            .filter(|(key, _)| {
                deep(key) && !common.contains(key) && book.get(key).is_some_and(|e| !e.dropped)
            })
            .collect();
        let states: Vec<BitState> = positions.iter().map(|(_, s)| *s).collect();
        let total = states.len();
        println!(
            "pruning {} ({live_before} live entries) against {spec}: searching {total}",
            path.display()
        );
        let progress = |done: usize| {
            if done.is_multiple_of(200) || done == total {
                print!("\r  {done}/{total} ({:.0}s)", start.elapsed().as_secs_f32());
                let _ = io::stdout().flush();
                if done == total {
                    println!();
                }
            }
        };
        let choices: Vec<Option<(Move, Vec<(Move, f32)>)>> = match spec.search_parts()? {
            Some((mut evaluator, search, _)) => search_positions(
                &mut evaluator,
                &states,
                &search,
                args.concurrent,
                |_| None,
                progress,
            )
            .into_iter()
            .map(|r| Some((r.best_move()?, r.policy())))
            .collect(),
            None => {
                let mut agent = spec.build()?;
                states
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        let choice = agent.choose_move_with_policy(s);
                        progress(i + 1);
                        Some(choice)
                    })
                    .collect()
            }
        };
        let mut agreed = Vec::new();
        for ((key, state), choice) in positions.iter().zip(&choices) {
            let entry = book.get(key).expect("position of an entry");
            if let Some((played, policy)) = choice
                && agent_agrees(state, entry.best, *played, policy, args.min_advantage)
            {
                agreed.push(key.clone());
            }
        }
        for key in agreed {
            if book.drop_entry(&key) {
                per_ply.entry(ply_of(&key)).or_default()[4] += 1;
            }
        }
    }

    println!("ply  entries     live   common     rare    agent  live now");
    for (ply, [n, live, common, rare, agent]) in &per_ply {
        println!(
            "{ply:>3}  {n:>7}  {live:>7}  {common:>7}  {rare:>7}  {agent:>7}  {:>8}",
            live - rare - agent
        );
    }
    let live_after = book.live_len();
    println!(
        "{live_after} of {} entries live (was {live_before}), in {:.0}s",
        book.len(),
        start.elapsed().as_secs_f32()
    );

    let mut how = Vec::new();
    if let Some(share) = args.drop_rare {
        how.push(format!("the lines of replies under {share} of the visits"));
    }
    if let Some(spec) = &spec {
        let mut agree = format!("the moves {spec} plays anyway");
        if args.min_advantage > 0.0 {
            agree.push_str(&format!(
                " leading its second-best by at least {} of its policy",
                args.min_advantage
            ));
        }
        if let Some(share) = args.keep_replies {
            agree.push_str(&format!(
                " (off the lines of replies with at least {share} of the visits)"
            ));
        }
        how.push(agree);
    }
    header.push(format!(
        "prune pass{}: dropped {}; {live_before} -> {live_after} live entries",
        if args.from_ply > 0 {
            format!(" from ply {}", args.from_ply)
        } else {
            String::new()
        },
        how.join(" and ")
    ));
    book.save(&args.out, &header)?;
    println!("wrote {}", args.out.display());
    Ok(())
}

fn book(
    search: &str,
    out: &std::path::Path,
    cfg: brut3::book::BookConfig,
    threads: usize,
) -> Result<(), String> {
    use brut3::book::Book;

    let words: Vec<&str> = search.split_whitespace().collect();
    let spec = AgentSpec::parse(&words)?;
    let search_cfg = spec
        .search_config()
        .ok_or_else(|| format!("{spec} is not a search; use mcts or mcts-nn"))?;
    let threads = match threads {
        0 => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        n => n,
    };
    // The header keeps the lines of earlier runs and prune passes, after a fresh count.
    let (book, mut history) = if out.exists() {
        let text = std::fs::read_to_string(out)
            .map_err(|e| format!("cannot read {}: {e}", out.display()))?;
        let book = Book::parse(&text).map_err(|e| format!("{}: {e}", out.display()))?;
        println!("extending {} ({} entries)", out.display(), book.len());
        if book.coalesced() > 0 {
            println!(
                "  {} lines pooled into others' entries (the same position up to decided boards' cells and colours)",
                book.coalesced()
            );
        }
        let history: Vec<String> = text
            .lines()
            .map_while(|l| l.strip_prefix('#'))
            .map(|l| l.trim().to_string())
            .filter(|l| !l.starts_with("ultimate-ttt opening book:"))
            .collect();
        (book, history)
    } else {
        (Book::default(), Vec::new())
    };
    let plies = cfg.plies;
    let mut run = if cfg.main_lines {
        format!("main lines to ply {plies} searched by {spec}")
    } else {
        format!(
            "to ply {plies} searched by {spec}; replies covered from {} of the visits",
            cfg.min_prior
        )
    };
    if cfg.keep_searched {
        run.push_str("; existing entries kept");
    }
    if cfg.dense_plies > 0 {
        run.push_str(&format!(", all of them below ply {}", cfg.dense_plies));
    }
    if let Some(n) = cfg.reply_simulations {
        run.push_str(&format!(
            "; the opponent's positions searched with {n} simulations"
        ));
    }
    println!("book {run} ({threads} search thread(s))");
    if !history.contains(&run) {
        history.push(run);
    }
    #[cfg(feature = "nn")]
    if threads > 1 && matches!(spec, AgentSpec::MctsNn { .. }) {
        // The network evaluates on a thread of its own that batches every search thread's
        // leaves together (see EvalServer).
        let spec = spec.clone();
        let server = brut3::nn::EvalServer::spawn(
            move || {
                spec.leaf_evaluator()?
                    .ok_or_else(|| format!("{spec} is not a search"))
            },
            brut3::nn::MAX_BATCH,
        )?;
        let mut clients: Vec<_> = (0..threads).map(|_| server.client()).collect();
        return extend_book(book, &mut clients, &search_cfg, &cfg, out, &history);
    }
    let mut evaluators = (0..threads)
        .map(|t| {
            spec.offset_seed(t as u64)
                .leaf_evaluator()?
                .ok_or_else(|| format!("{spec} is not a search"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    extend_book(book, &mut evaluators, &search_cfg, &cfg, out, &history)
}

/// Extends `book` with a search thread per evaluator, saving it to `out` after every round
/// under a header of `history`.
fn extend_book<A: brut3::agent::UtttAgent<State = brut3::game::BitState> + Send>(
    mut book: brut3::book::Book,
    evaluators: &mut [A],
    search_cfg: &brut3::mcts::MctsConfig,
    cfg: &brut3::book::BookConfig,
    out: &std::path::Path,
    history: &[String],
) -> Result<(), String> {
    use std::cell::Cell;
    use std::io::Write;
    use std::time::Instant;

    use brut3::book::Book;

    let start = Instant::now();
    let round_start = Cell::new(Instant::now());
    // Threads report their searches in bursts, so the progress line is redrawn every 50
    // positions crossed rather than on multiples of 50.
    let shown = Cell::new(0);
    let save = |book: &Book| {
        let mut header = vec![format!(
            "ultimate-ttt opening book: {} entries, {} open positions",
            book.len(),
            book.open_len()
        )];
        header.extend(history.iter().cloned());
        book.save(out, &header)
    };
    let added = book.extend(
        evaluators,
        search_cfg,
        cfg,
        |line| {
            round_start.set(Instant::now());
            shown.set(0);
            println!("{line}");
        },
        |done, total| {
            if done / 50 != shown.get() / 50 || done == total {
                shown.set(done);
                print!(
                    "\r  {done}/{total} ({:.0}s)",
                    round_start.get().elapsed().as_secs_f32()
                );
                let _ = io::stdout().flush();
                if done == total {
                    println!();
                }
            }
        },
        |book| {
            save(book)?;
            println!("  saved {} entries to {}", book.len(), out.display());
            Ok(())
        },
    )?;
    println!(
        "added {} entries, made {} dropped ones live again, searched {} again and opened {} opponent positions in {:.0}s; {} entries and {} open positions saved to {}",
        added.added,
        added.revived,
        added.researched,
        added.opened,
        start.elapsed().as_secs_f32(),
        book.len(),
        book.open_len(),
        out.display()
    );
    Ok(())
}

fn benchmark(
    games: usize,
    temperature_moves: usize,
    dirichlet_alpha: f32,
    threads: usize,
    specs: &[String],
) -> Result<(), String> {
    use brut3::benchmark::round_robin_with;
    use brut3::selfplay::MatchExploration;
    let exploration = MatchExploration {
        temperature_moves,
        root_noise: (dirichlet_alpha > 0.0).then_some((dirichlet_alpha, 0.25)),
    };
    let specs = specs
        .iter()
        .map(|s| {
            let words: Vec<&str> = s.split_whitespace().collect();
            AgentSpec::parse(&words)
        })
        .collect::<Result<Vec<_>, _>>()?;
    println!("round robin, {games} games per pair:");
    let tournament = round_robin_with(&specs, games, &exploration, threads, |m| println!("  {m}"))?;
    print!("\n{}", tournament.table());
    Ok(())
}

fn serve(addr: &str, specs: &[String]) -> Result<(), String> {
    let specs = specs
        .iter()
        .map(|s| {
            let words: Vec<&str> = s.split_whitespace().collect();
            AgentSpec::parse(&words)
        })
        .collect::<Result<Vec<_>, _>>()?;
    brut3::serve::run(&specs, addr).map_err(|e| e.to_string())
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
    use brut3::nn::{BackendKind, LoopConfig, ModelSpec, TrainOptions, run_loop};
    use brut3::selfplay::{MatchExploration, SelfPlayConfig};

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
        threads: args.threads,
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
            stats_every: args.stats_every,
            augment: args.augment.parse()?,
            outcome_weight: args.outcome_weight,
            search_value_weight: args.search_value_weight,
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

    use brut3::nn::{games, generate_threaded, self_play_threads};
    use brut3::selfplay::SelfPlayConfig;

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
    println!(
        "self-play: {} games, {} simulations/move, {} concurrent x {} leaves, evaluator {evaluator}, {} thread(s)",
        args.games,
        cfg.simulations,
        cfg.concurrent,
        cfg.tree_batch,
        self_play_threads(args.threads, args.games)
    );
    let start = Instant::now();
    let report = |n: usize| {
        if n.is_multiple_of(100) || n == args.games {
            print!("\r  {n}/{} games", args.games);
            let _ = io::stdout().flush();
        }
    };
    let (data, stats) = generate_threaded(&evaluator, args.games, args.threads, &cfg, report)?;
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
fn import_utttai(paths: &[std::path::PathBuf], out: &std::path::Path) -> Result<(), String> {
    use std::time::Instant;

    use brut3::nn::{games, utttai};

    let start = Instant::now();
    let data = utttai::import(paths)?;
    let with_wdl = data.samples.iter().filter(|s| s.wdl.is_some()).count();
    println!(
        "parsed {} positions ({with_wdl} with win/draw/loss counts) in {:.0}s",
        data.len(),
        start.elapsed().as_secs_f32()
    );
    games::save(out, &data)?;
    println!("saved to {}", out.display());
    Ok(())
}

#[cfg(feature = "nn")]
#[allow(clippy::too_many_arguments)]
fn export(
    model: Option<&std::path::Path>,
    book: Option<&std::path::Path>,
    out: &std::path::Path,
    module: String,
    backend: Option<&str>,
    positions: usize,
    seed: u64,
    lookups: usize,
    vec_flags: Vec<String>,
    fixed_vec: bool,
) -> Result<(), String> {
    use brut3::book::Book;
    use brut3::nn::export::{ExportOptions, VecSpec, Weights, write_all};
    use brut3::nn::{AnyModel, BackendKind};

    // Messages go to stderr: with `--out -`, stdout holds the files.
    let to_stdout = out == std::path::Path::new("-");

    let weights = match model {
        Some(model) => {
            let backend = match backend {
                Some(name) => name.parse::<BackendKind>()?,
                None => BackendKind::preferred(),
            };
            let weights = Weights::from_any(&AnyModel::load(model, &backend.device())?)
                .map_err(|e| format!("{} is {e}", model.display()))?;
            eprintln!(
                "{}: {} parameters",
                weights.config_display(),
                weights.parameters()
            );
            Some(weights)
        }
        None => None,
    };
    let book = match book {
        Some(path) => {
            let book = Book::load(path)?;
            eprintln!(
                "book {}: {} entries ({} dropped)",
                path.display(),
                book.live_len(),
                book.len() - book.live_len()
            );
            Some(book)
        }
        None => None,
    };
    // --vec-extra's words, then the ones after `--`.
    let vec_extra = if vec_flags.is_empty() {
        None
    } else {
        Some(VecSpec::parse_args(
            &vec_flags.iter().map(String::as_str).collect::<Vec<_>>(),
        )?)
    };
    let opts = ExportOptions {
        prefix: module,
        positions,
        seed,
        lookups,
        vec_extra,
        fixed_vec,
        ..ExportOptions::default()
    };
    let report = write_all(weights.as_ref(), book.as_ref(), out, &opts)?;
    for (path, bytes) in &report.files {
        eprintln!(
            "{} {} ({} bytes)",
            if to_stdout { "printed" } else { "wrote" },
            path.display(),
            bytes
        );
    }
    if weights.is_some() {
        eprintln!(
            "int16 weights: {} bytes in {} literals; quantisation error over the reference positions: \
             wdl {:.2e}, value {:.2e}, policy {:.2e}",
            report.stream_bytes,
            report.chunks,
            report.wdl_error,
            report.value_error,
            report.policy_error
        );
    }
    if let Some(flags) = &report.vec_flags {
        eprintln!(
            "FixedVec shapes (the same module without a model: export --out <dir> -- {flags})"
        );
    }
    if let Some(stats) = &report.book {
        eprintln!(
            "book: {} entries, {} bytes in {} literals, {} reference lookups",
            stats.entries, stats.bytes, stats.literals, stats.lookups
        );
    }
    Ok(())
}

#[cfg(feature = "nn")]
fn train(args: &TrainArgs) -> Result<(), String> {
    use std::io::Write;
    use std::time::Instant;

    use brut3::nn::{BackendKind, ModelSpec, TrainOptions, generate};

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
            let data = brut3::nn::games::load_many(&args.data)?;
            println!(
                "loaded {} games from {} file(s)",
                data.games(),
                args.data.len()
            );
            data
        }
        Some(evaluator) => {
            use brut3::nn::generate_threaded;
            use brut3::selfplay::SelfPlayConfig;
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
            let (data, stats) = generate_threaded(&evaluator_spec, args.games, 1, &cfg, progress)?;
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
        brut3::nn::games::save(out, &data)?;
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
        lr_min: args.lr_min,
        weight_decay: args.weight_decay,
        stats_every: args.stats_every,
        validation_fraction: args.validation_fraction,
        workers: args.workers,
        seed: args.seed,
        artifact_dir: args.artifact_dir.clone(),
        tui: !args.no_tui,
        augment: args.augment.parse()?,
        outcome_weight: args.outcome_weight,
        search_value_weight: args.search_value_weight,
    };
    let (model, report) =
        brut3::nn::train_model(&model_spec, args.init.as_deref(), &device, data, &opts)?;
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
