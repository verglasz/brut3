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

Default features are everything that builds with no extra system requirements. Every backend
compiled in will be selectable at runtime, so one binary can exercise all of them.
`cargo build --no-default-features` skips burn entirely for fast iteration on the engine.

## Building and running

Requires a recent stable Rust toolchain (edition 2024).

```sh
# Engine only, no burn: fast builds for game/search work
cargo build --no-default-features
cargo test --no-default-features

# Default: burn with ndarray + wgpu
cargo build --release

# With CUDA (needs nvcc / CUDA_PATH)
cargo build --release --features cuda

# Slow tests and throughput numbers
cargo test --release --no-default-features -- --ignored --nocapture

# Play against an agent in the terminal (default: mcts 400)
cargo run --release --no-default-features -- mcts 800 3
```

In the REPL, moves are entered as `r1c2/r0c0` (board, then square) or as four digits
`1 2 0 0`; `help` lists the commands (`new o` to play O, `agent random`, `hint`, `undo`).

## Design

The search is generic over the agent that evaluates leaves:

- `game::GameState` is the rules trait. Shared types are explicit: `Square { row, col }`,
  `Move { board, square }`, `Outcome`, `BoardStatus`. Each implementation picks its own internal
  representation: `Position` is the readable reference (arrays of `Option<Player>`), `BitState`
  the fast bitboard used by search.
- `agent::UtttAgent` has an associated `State: GameState`, `choose_move`, and `evaluate`, which
  returns a value in `[-1, 1]` plus optional move priors. The default `evaluate` is a playout
  using `choose_move`; a network agent overrides it (and `evaluate_batch`).
- `mcts::Mcts<A: UtttAgent>` is a PUCT search over an arena tree. With no priors it behaves as
  plain UCT with rollouts; with network priors it is the AlphaZero search. `Mcts<A>` itself
  implements `UtttAgent`, so searches compose and its visit distribution can feed training.

## Status

Present:

- Game rules with two `GameState` implementations, cross-checked by a differential fuzz test.
- `UtttAgent` trait, `RandomAgent`, and `Mcts` search, tested against the random agent.
- `AnyAgent`, one concrete enum over the built agents, and a REPL for interactive play.
- Cargo feature layout for the burn backends (no network code yet).

Planned next, roughly in order:

- [ ] More simple agents: one-ply greedy, shallow minimax
- [ ] Batched leaf evaluation with virtual loss, tree reuse between moves
- [ ] Network definition in burn, position -> tensor encoding, runtime backend selection
- [ ] Self-play, replay buffer, training loop, model checkpointing
- [ ] CLI for agent-vs-agent matches, training, benchmarks

## Layout

```
src/
  lib.rs            crate root
  game/mod.rs       Player, Square, Move, Outcome, BoardStatus, GameState trait
  game/position.rs  Position: reference implementation + Display
  game/bitboard.rs  BitState: bitboard implementation
  agent/mod.rs      UtttAgent trait, Evaluation
  agent/random.rs   RandomAgent
  agent/any.rs      AgentSpec, AnyAgent (concrete enum over all agents)
  repl.rs           interactive play
  mcts.rs           Mcts, MctsConfig, SearchResult
  main.rs           launches the REPL
Cargo.toml          burn 0.21 behind backend features
```
