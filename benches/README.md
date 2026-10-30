# Benchmarks

Microbenchmarks of single components, for finding where time goes and checking that a change
helped. `examples/bench.rs` has the end-to-end throughput numbers (agents, whole searches,
self-play, network evaluators).

```sh
benches/run.sh [FILTER...]
```

runs `micro.rs` and writes its output to `results/micro-<date>-<commit>[-dirty].txt`, so runs
at different commits can be diffed. A filter keeps the benchmarks whose name contains it
(`benches/run.sh mcts/ pick/`); filtered runs go to a file named after the filters too, so they
do not replace a full run. Without filters everything runs but the `deep/` benchmarks, which
take seconds and run only when a filter names them (`benches/run.sh deep/`). `cargo bench --features full --bench micro [-- FILTER...]` runs
the same thing without saving it.

## How it measures

- Inputs are pools of 4096 distinct non-terminal positions from seeded random games, in the
  order met (`all`) and split by whether the side to move is sent to one board (`forced`) or
  may play anywhere (`free`). The pools are the same on every run.
- Each benchmark is calibrated to samples of about 5 ms, each made of whole passes over its
  inputs, and reports the median time per unit over 15 samples with the min and max.
- The checksum is the result summed over one pass of the inputs. It does not depend on timing
  or on the filters, so equal checksums at two commits mean both computed the same thing.
- `baseline/loop overhead` is the cost of the harness itself, to subtract from the smallest
  numbers.

| benchmark | measures |
|---|---|
| `game/legal_moves` | move generation, forced and free positions |
| `game/outcome` | result check of a position |
| `game/copy + apply + outcome`, `game/copy + play`, `game/outcome_after` | the result after a move: the long way, `play` (applies and reports the end, as playouts do) and the shortcut without applying that `expand` uses |
| `game/transform` | one board symmetry (book and data augmentation) |
| `rng/random_range`, `rng/uniform_index` | drawing a move number: `rand`'s way and the multiply-shift that `random_move` uses |
| `pick/MoveSet::nth(k)` | turning that number into a move |
| `game/random_move` | a uniform legal move straight from the state, as playouts pick them |
| `agent/RandomAgent::choose_move` | the two above plus move generation: one playout step |
| `eval/random playout` | one rollout to the end from a pool position |
| `eval/heuristic::score` | the static evaluation used by the greedy and minimax agents |
| `mcts/search 1k per sim` | a 1,000-simulation search from 16 roots with fake leaf values, per simulation: selection, expansion, backup and the claim/resolve round trip, without any evaluator. With uniform priors, with a prior row per leaf (as a network gives), and claiming 16 leaves per round |
| `mcts/search 10k per sim` | the same (prior rows, batch 16) at 10,000 simulations from 2 roots: a tree that stays less in cache |
| `mcts/search 1k ..., reset tree` | the 1k search in one tree reset between searches, as a game's moves reuse it |
| `mcts/game 256 per move` | whole games from 2 roots, 256 simulations per move in one reset tree, per simulation run |
| `mem/search`, `mem/game` | not timed: bytes per simulation of the nodes and edges in use and allocated, after the 1k and 10k searches and at the largest allocation over the games' searches |
| `deep/...` | only when named (`benches/run.sh deep/`), seconds rather than milliseconds: a 100k-simulation search, games at 4096 simulations per move, and the 100k tree's bytes per simulation |
| `mcts/dirichlet` | root noise for 81 moves |
| `nn/encode_into`, `nn/nnue_ids`, `nn/folded_ids` | network inputs for one position (with the `nn` feature); `folded_ids` is the folded NNUE's 21 table rows |

## Pitfalls met so far

- Wrapping a result in `black_box` costs time of its own when the result is a whole position:
  "copy + apply" alone measured slower than "copy + apply + outcome". Return something small
  derived from the result instead where possible.
- Inputs of uneven cost (searches from different roots, playouts of different lengths) look
  like noise unless every sample covers the same inputs; hence whole passes.
- Single short runs are not comparable: `examples/bench.rs` cases of about 0.4 s varied by
  about 25% between back-to-back runs of the same binary.
