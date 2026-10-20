# ultimate-ttt-burn

A Monte Carlo Tree Search agent for [Ultimate Tic Tac Toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe),
written in Rust with [burn](https://burn.dev) for the neural-network side.

The end goal is an AlphaZero-style agent: MCTS guided by a learned policy/value
network, trained from self-play. Along the way the project also provides a plain
MCTS (random-rollout) baseline to measure against.

## Game

Ultimate Tic Tac Toe is a 3x3 grid of 3x3 tic-tac-toe boards (81 cells).

- Players alternate placing X / O in a cell of one of the small boards.
- Winning a small board claims that cell of the big board; winning three in a
  row on the big board wins the game.
- The cell you play in dictates which small board the opponent must play in
  next. If that board is already decided (won or full), the opponent may play
  anywhere open.
- A full small board with no winner is a draw for that cell; a full big board
  with no three-in-a-row is a drawn game.

The game is small enough to iterate on quickly but has a branching factor and
depth where plain search is not enough, which makes it a good target for
learned evaluation.

## Architecture

The code is organised in layers, each usable without the ones above it:

1. **Game engine** (`game`) — rules, legal-move generation, terminal detection,
   behind the `GameState` trait. Pure Rust with no burn dependency; the bitboard
   implementation is what search and self-play run on.
2. **Agents and search** (`agent`, `mcts`) — the `UtttAgent` trait and a PUCT
   MCTS generic over the agent that evaluates leaves. With playouts and no
   priors it is classic UCT; with a network it is the AlphaZero search. Leaf
   evaluations will be batched across many concurrent searches so the GPU stays busy.
3. **Network** (planned) — a small residual conv/MLP network built with burn
   that takes the encoded position and outputs a policy over the 81 moves
   (masked to legal ones) and a scalar value in `[-1, 1]` from the side-to-move
   perspective.
4. **Training loop** (planned) — self-play generates `(position, search policy,
   outcome)` triples into a replay buffer; the network is trained on them; the
   new network is pitted against the previous one before being promoted.
5. **CLI** (planned) — play against the agent, run agent-vs-agent matches,
   drive training, and benchmark backends.

## Backends

burn is backend-agnostic, so the same model and training code runs on:

| Backend | Cargo feature | Use |
|---------|---------------|-----|
| `Cuda`  | `cuda` (opt-in) | Primary target for training and large self-play runs. Needs the CUDA toolkit at build time. |
| `Wgpu`  | `wgpu` (default) | Portable GPU fallback (Vulkan/Metal/DX12). |
| `NdArray` | `ndarray` (default) | Pure-Rust CPU backend for tests and GPU-less machines. |
| CubeCL CPU | `cpu-mlir` (opt-in) | MLIR-based CPU backend; downloads a prebuilt LLVM on first build. |

Default features are everything that builds with no extra system requirements. All tensors go
through burn's `Dispatch` backend, whose device enum selects among the compiled-in backends at
runtime (`--backend cuda`, `nn models/value wgpu`, ...), so one binary can exercise all of them;
`backends` lists what a binary has. `cargo build --no-default-features` skips burn entirely for
fast iteration on the engine. Development builds use `--all-features`.

Known upstream issues worked around:

- Kernel fusion is off on the CubeCL backends (cuda, wgpu, cpu-mlir): burn is built without its
  default features and the pieces we need are enabled one by one (`nn` feature). In burn 0.21
  fusion returns wrong values when an elementwise op is fused onto a matmul with a single
  output column (which broke `burn::nn::Linear` with one output), and its multi-stream
  bookkeeping races under the Learner's threads with the transformer models ("Should have
  handle for tensor"). The `fusion` cargo feature restores burn's defaults for experiments.
  `nn/linear.rs` keeps a test of both heads against a host computation on every compiled
  backend.
- `nn/train.rs`: on the MLIR CPU backend (cubecl-cpu 0.10), reading a still-lazy tensor from a
  thread other than the one that produced it panics in the memory pool. The Learner's metric
  and checkpoint threads do exactly that, so on that backend the step outputs are read back on
  the training thread before they are handed over, and file checkpointing is skipped. Treat
  cpu-mlir as experimental; `ndarray` is the dependable CPU backend.
- The per-backend tests honour `UTTT_SKIP_BACKENDS=wgpu,...` for machines where a compiled
  backend cannot initialise (for example an NVIDIA driver update pending a reboot).

## Building and running

Requires a recent stable Rust toolchain (edition 2024).

```sh
# Engine only, no burn: fast builds for game/search work
cargo build --no-default-features
cargo test --no-default-features

# Default: burn with ndarray + wgpu
cargo build --release

# Everything, including CUDA (needs nvcc / CUDA_PATH) and the MLIR CPU backend
cargo build --release --all-features
cargo test --all-features
cargo run --release --all-features -- backends

# Train the value network on self-play data (live metrics dashboard), then play against it
cargo run --release --all-features -- train --backend cuda --model linear --data-agent "mcts 50" --games 2000 --epochs 15
cargo run --release --all-features -- play mcts-nn models/value cuda 400 16

# For scripts and CI: plain log lines instead of the dashboard
cargo run --release --all-features -- train --backend ndarray --no-tui --artifact-dir artifacts/run1

# Slow tests, and throughput numbers for engine, agents and search
cargo test --release --no-default-features -- --ignored --nocapture
cargo run --release --no-default-features --example bench

# Play against an agent in the terminal (default: mcts 400)
cargo run --release --no-default-features -- mcts 800 3
```

In the REPL, moves are entered as `r1c2/r0c0` (board, then square) or as four digits
`1 2 0 0`; `help` lists the commands (`new o` to play O, `agent nn models/value`, `hint`, `undo`).

## Design

The search is generic over the agent that evaluates leaves:

- `game::GameState` is the rules trait. Shared types are explicit: `Square { row, col }`,
  `Move { board, square }`, `Outcome`, `BoardStatus`. Legal moves come back as the
  implementation's own `MoveSet` type (associated type `Moves`), whose API speaks only in
  `Move`s. Each implementation picks its representations: `Position` is the readable reference
  (arrays of `Option<Player>`, a `MoveList`), `BitState` the fast bitboard used by search with a
  `u128`-backed `BitMoveSet`.
- `agent::UtttAgent` has an associated `State: GameState`, `choose_move`, and `evaluate`, which
  returns a value in `[-1, 1]` plus optional move priors. The default `evaluate` is a playout
  using `choose_move`; the heuristic agents return their static value; a network agent will
  override `evaluate_batch` to run one forward pass per batch of leaves.
- `mcts::Mcts<A: UtttAgent>` is a PUCT search over an arena tree. With no priors it behaves as
  plain UCT with rollouts; with network priors it is the AlphaZero search. `Mcts<A>` itself
  implements `UtttAgent`, so searches compose and its visit distribution can feed training.
- `nn` holds the network side. Every model implements `Model`: it encodes raw `BitState`s into
  its own input tensors and predicts a `Prediction` with two heads, three win/draw/loss logits
  from the side to move's view (search uses `p_win - p_loss` as the value) and 81 move logits
  (masked to legal moves). `LinearNet` is the baseline: one linear layer per head over 199
  binary features from `encode`. `ModelSpec` names a model kind and is saved as JSON next to
  the weights, so `AnyModel::load` rebuilds the right model from a path. `train` generates
  self-play positions labelled with the game result and the generating agent's move
  distribution (a search's visit counts), holds out whole games for validation (positions of
  one game share their outcome), then fits the model with burn's `Learner`: AdamW,
  the summed cross-entropies as the loss, loss and verdict-accuracy metrics for both splits,
  checkpoints, a TUI dashboard or plain logs, and an `experiment.log` under `--artifact-dir`.
  `NetAgent` plays the policy head's best move and serves `Mcts` as a batched leaf evaluator
  with priors, which makes `mcts-nn` the AlphaZero-style search. `BackendKind` names the
  compiled backends and gives their `Dispatch` device.
- Two transformer models share `heads` (pre-norm encoder body with all-to-all attention, and
  two-layer perceptron heads reading the flattened final residual stream). `BoardTransformer`
  uses nine tokens, one per small board: an embedding of the board's full cell configuration
  (3^9 entries) plus a sum of per-cell embeddings (so unseen configurations still get a
  meaningful token), plus position, status and forced-board embeddings. `CellTransformer` uses 81
  cell tokens: content, position, the cell's board status, forced flag and legality
  embeddings. Both are selected with `--model board-transformer [d_model] [layers] [heads]` or
  `--model cell-transformer ...`.
- `benchmark` plays round-robin matches between any agents (colours alternate) and prints
  pairwise results and standings, for comparing trained models with each other and with the
  search and heuristic baselines.

## Status

Present:

- Game rules with two `GameState` implementations, cross-checked by a differential fuzz test.
- `UtttAgent` trait and simple agents: random, greedy (one ply of a static heuristic), and
  depth-limited alpha-beta minimax on the same heuristic.
- `Mcts` search with batched leaf evaluation and virtual loss (`batch_size` in `MctsConfig`),
  tested against the random agent.
- `AnyAgent`, one concrete enum over the built agents, and a REPL for interactive play.
- A policy/value model interface with a linear baseline, trained from self-play on any compiled
  backend (ndarray, wgpu, cuda, cpu-mlir) with burn's `Learner`, and `nn` / `mcts-nn` agents
  using it. The linear model only shaves a little off the baselines; it is the plumbing for
  the real networks.

Planned next, roughly in order:

- [ ] Tree reuse between moves
- [x] Real networks: a transformer over small-board tokens and one over cell tokens
- [ ] Iterated self-play with the current network, replay buffer, evaluation against the
      previous generation before promotion
- [ ] Shared batching inference service so many searches feed one GPU batch
- [x] CLI for agent-vs-agent matches (`benchmark`)

## Layout

```
src/
  lib.rs            crate root
  game/mod.rs       Player, Square, Move, Outcome, BoardStatus, GameState trait
  game/position.rs  Position: reference implementation + Display
  game/bitboard.rs  BitState: bitboard implementation
  agent/mod.rs      UtttAgent trait, Evaluation
  agent/random.rs   RandomAgent
  agent/heuristic.rs static evaluation shared by greedy and minimax
  agent/greedy.rs   GreedyAgent
  agent/minimax.rs  MinimaxAgent
  agent/any.rs      AgentSpec, AnyAgent (concrete enum over all agents)
  repl.rs           interactive play
  benchmark.rs      matches and round-robin tournaments
  mcts.rs           Mcts, MctsConfig, SearchResult
  main.rs           launches the REPL
examples/bench.rs   throughput benchmark
Cargo.toml          burn 0.21 (Dispatch + autodiff) behind backend features, clap, rand
```
