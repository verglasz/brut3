# The nnue network, the folding trick, and what a Haskell export needs

Reference for porting the trained network (and the opening book) to the base-only Haskell
contest engine. The Rust implementation is `src/nn/nnue.rs`; the book is `src/book.rs`.

## Why this architecture

- uttt.ai's network (5M params, ~18M MACs per eval) spends over half its cost on a policy head
  that goes back to 9x9 resolution, and its "convolutions" at 3x3 resolution are per-token
  MLPs plus a pool-and-rebroadcast bottleneck. A small board has under 20k contents, so a
  lookup table represents any per-board function exactly, at zero FLOPs.
- With 9 fixed positions, weight sharing (convolution) costs 9 MACs per parameter; a
  fully-connected first layer costs 1 and mixes all boards at once. Attention over 9 tokens
  buys nothing over that.
- The first layer over table rows can be *folded*: precompute each board's contribution for
  every content, and the first layer becomes row additions. This is what makes a ~12k-MAC
  evaluation possible in Haskell without SIMD.
- Measured: the one-epoch smoke model (d_acc 256) at 10k simulations beat 100k random playouts
  16-4-0; the lab model (d_acc 512) beats them at 1k simulations.

## Inputs (all ids, from the side to move's view)

**Compact content id of a board.** Raw code = sum over cells of `v * 3^cell`, cell 0 least
significant, `v` = 1 mine, 2 theirs, 0 empty. Mapped through a 19683-entry table built by
enumeration: open contents (no line, not full) get ids `0..11093` in increasing raw-code
order; then `FULL = 11093`, `WON_BY_ME = 11094`, `WON_BY_THEM = 11095`; `CONTENTS = 11096`.
Any content where exactly one player has a line is "won" by that player (the exact cells of a
decided board never matter); both players with a line is unreachable. The Haskell side builds
the same table at startup with the same enumeration (`compact_table` in nnue.rs).

**Content table rows** (`CONTENT_ROWS = 11125`, width `d_embed`):
`0..11096` contents; `11096` NO_FORCED; `11097 + cell*3 + v` per-cell rows (27);
`11124` CELL_PAD.

**Board token** `t_k` for board `k` (and a tenth token `t_f` for the forced board, or NO_FORCED):
`E[content] + sum over the 9 cells of E[cell row]` for an *open* board, where the cell row
is `CELL_BASE + cell*3 + v` (v=0 for empty cells too); for a decided board or no forced
board, `E[content] + 9 * E[CELL_PAD]`. Decided boards must not depend on their cells so
that a folded row is a function of the compact id alone.

**Feature ids** (table of `FEATURE_ROWS = 605` rows, width `d_acc`), 11 per position:
- `0 + forced board index`, or `9` for a free choice;
- for each of the 8 big-board lines `l` (order of `game::LINES`): `10 + l*64 + s_a*16 + s_b*4 + s_c`
  with status `s` = 0 open, 1 won by me, 2 won by them, 3 full (statuses derive from the
  compact ids);
- `522 + number of decided boards` (0..9);
- `532 + number of marks on open boards` (0..72).

## Forward pass

```
acc    = relu( concat(t_0..t_8) · W_mix + b_mix  +  t_f · W_forced  +  Σ F[feature ids] )
h_1    = relu(acc · W_1 + b_1)                         # d_acc -> d_hidden
h_i    = relu(h_{i-1} + h_{i-1} · W_i + b_i)           # residual, i = 2..layers
wdl    = h · W_value + b_value + acc · W_skip          # 3 logits: win, draw, loss
policy[k*9+j] = (h · W_policy + b_policy)[k*9+j] + P[c_k][j] + Q[c_j]
```
`P` is the local policy table (`CONTENTS x 9`), `Q` the send table (`CONTENTS x 1`): playing
square `j` sends the opponent to board `j`, so `Q[c_j]` is added to square `j` of every
board. Value for the search = `softmax(wdl)[win] - softmax(wdl)[loss]`. Policy = softmax over
legal moves only. Burn's `Linear` weight is `[d_in, d_out]`, applied as `x · W`.

Parameter tensors (burn record names in `Nnue`): `content.weight`, `mix.weight/bias`,
`forced_mix.weight`, `features.weight`, `hidden[i].weight/bias`, `value.weight/bias`,
`value_skip.weight`, `policy.weight/bias`, `local_policy.weight`, `send_policy.weight`.
Saved as `<stem>.bpk` (burnpack) + `<stem>.json` (spec); exported to Haskell by the `export`
command (see "Haskell export" below).

Sizes trained so far: `nnue 64 128 32 2`, `nnue 64 256 32 2`, lab `nnue 64 512 512 3`.

## The folding trick

Board `k`'s share of the first layer is `t_k · W_k`, where `W_k` is rows `k*d_embed ..
(k+1)*d_embed` of `W_mix`, and `t_k` depends only on the compact content. So precompute

```
T_k[c] = t(c) · W_k        for every content c, each board k   (9 tables of CONTENTS x d_acc)
T_f[c] = t(c) · W_forced   plus the row for NO_FORCED
```
with `t(c)` the token defined above (content row + cell rows for open `c`, content row +
9·pad otherwise). Then

```
acc = relu( Σ_k T_k[c_k] + T_f[c_f or NO_FORCED] + Σ F[feature ids] + b_mix )
```
is 21 row additions of width `d_acc`, no multiplies. The trainable model is unchanged; the
tables are a cache of the first layer over its whole input domain.

Cost per evaluation after folding, `A = d_acc`, `H = d_hidden`, `L` layers, `m` legal moves:
`A·H + (L-1)·H² + 3H + 3A + m·H` MACs plus 21 row adds of A.

| variant | MACs | folded tables f32 | int8 |
|---|---|---|---|
| A 256, H 32, L 2 | ~12k (vs ~175k unfolded) | 10 × 11096 × 256 × 4 ≈ 110 MB | 28 MB |
| A 128, H 64, L 2 | ~13k | 55 MB | 14 MB |
| A 512, H 512, L 3 (lab) | ~1M | 220 MB | 55 MB |

A random playout is a few hundred to ~2k simple ops, so a folded 12k-MAC eval ≈ 5–10
playouts of compute: 10k network simulations ≈ 50–100k playouts, i.e. around the bar from
the previous notes. Levers if tight: fewer simulations (1k already beats 100k playouts),
A = 128.

### Folding options (export-time, same trained weights)

1. **Ship folded tables.** Largest artifact; quantise to int8 per table (NNUE-style, int32
   accumulator) for 4x less. Problem: the weights must live in Haskell source; 45M numbers
   is not a source file.
2. **Fold at startup.** Ship the factored weights (~1.1M numbers for A 256, a few MB as
   text), compute `T_k = t(c)·W_k` for all `c` into RAM at program start: ~3 GMAC at A 256,
   tens of seconds in Haskell; half at A 128. Affordable against a 5-minute clock, not free.
3. **Fold lazily.** Memoise `T_k[c]` per (board, content) on first use. A miss costs
   `d_embed·A` ≈ 16k MACs (one eval's worth); a game touches a small fraction of contents and
   the search revisits the same boards, so the hit rate is high after a few moves. No startup
   cost, graceful worst case (~2x eval). Probably the right default.
4. Shrink `A` (128 halves everything).

### Architectural options (must be trained in; not implemented)

5. **One shared table + per-board diagonal gate**: `T[c] ⊙ s_k + bias_k`. One table's memory,
   +9·A multiplies, loses content-by-position mixing beyond per-channel gating.
6. **Shared table + low-rank per-board correction**: `T[c] + S_k[c]·R_k`, rank r. One big
   table + tiny ones, +9·r·A MACs (~9k at A 256, r 4).
7. The three symmetry classes (corner/edge/centre) could share tables with permuted content
   codes, but that conflates e.g. the top-left and top-right boards' contents; avoid.

## Haskell-side constraints (Safe Haskell, base only)

- `GHC.Exts`, `GHC.Int`, `GHC.ST`, `System.IO.Unsafe` are Unsafe: a `Safe` module cannot
  import them; `Trustworthy` on our own module works only if the contest checks just `Main`
  (a global `-XSafe` rejects it). Unlifted types (`Int32#`) cannot be class-instance heads
  (kind `TYPE Int32Rep`, not `Type`); instances go on boxed `Int32` with `I32#` at the
  boundary, `INLINE`d.
- Under strict Safe: the `Foreign` modules are `Trustworthy`. Tables as `ForeignPtr`s from
  `mallocForeignPtrBytes`/`newArray`; hot loops in `IO` with bang patterns over
  `peekElemOff`/`pokeElemOff`; `-O2`; no class dictionaries in the hot path (or `INLINE`).
  Check with `-ddump-simpl` that loop variables are unboxed. No pure indexing (no
  `unsafePerformIO`), and `GHC.Arr`/`array` are unavailable, so everything that touches the
  tables lives in `IO`, which the search loop does anyway.
- Weights in source as string literals (static C strings; split into chunks of a few hundred
  KB), decoded at startup into the arrays: ~1 s for 10M characters. Element types: int8/int16
  for folded rows if quantised, `Float`/`Int32` for the small layers, `Int` accumulators.

## Opening book (for embedding too)

`book --search "mcts-nn <stem> cuda <sims> <batch>" --out models/book.txt --plies 8
--min-prior 0.01`: the union of the X-book and the O-book (our best move at our nodes, every
reply with visit share ≥ min_prior at the opponent's), searched ply by ply in lockstep,
re-runnable to deepen (`--plies 10`). Depth 8 ≈ 74k raw entries before symmetry/pruning,
~10k after symmetry, ~3–5k after pruning. Recommended first build: 30k simulations per node
(the lab model beats 100k playouts at 1k), then re-search the first few plies at 300k–1M.

**Canonical form**: of the 8 images of a position (the same square mapping applied to board
and square coordinates and to the forced board; mappings in `game/symmetry.rs`), the one with
the lexicographically smallest key, key = 81 cell digits by `board*9+square` (0 empty, 1 X,
2 O) + forced board digit (9 = free). Entries store the move in that frame; lookup maps back
through the inverse symmetry (the quarter turns are each other's inverses, the rest are
self-inverse). Ties between symmetries are harmless (symmetric positions, equivalent moves).
File lines: `KEY MOVE VALUE move:visits,...`; the Haskell table can be a stripped `KEY MOVE`
export (~100 bytes per entry).

## Training notes (for reproducing weights)

- Data: uttt.ai stage1-mcts (8.0M positions, W/D/L counts from 1M random playouts, mover's
  view) and stage2-nmcts (8.0M positions, scalar mean values), imported to
  `datasets/stage{1,2}.games` (format v3). Stage 2 has no W/D/L targets: the three-way term is
  masked, only the expectation `p_win - p_loss` is trained, so the draw probability drifts and
  "verdict accuracy" reads 0 by design; pass both files to keep it calibrated.
- `scripts/train-nnue-utttai.sh`; throughput >200k positions/s at batch 2048 after switching
  table lookups to `gather` (burn's `Embedding` gradient on CubeCL is a one-thread-per-column
  serial scatter; see commit d2a74c5). Keep batch 2048, spend speed on epochs (utttai used
  ~25 per stage); cosine `--lr-min`.

## Haskell export (`export` command, `src/nn/export/`)

`export --model <stem> --out <dir> [--module Gen] [--positions 16] [--seed 1]` writes, under
`<dir>/Gen/`:

- `Spec.hs`: widths, every id convention, the open-content checksum (`h <- (h * 1000003 +
  code) mod 2^32` over open raw codes ascending), parameter count and folded MACs.
- `Weights.hs`: `arrays :: [Array]`, one `Array name rows cols scale text` per weight array,
  int16-quantised with one scale per array, little-endian, base64 in string literals of
  64 KB (lines of 1024 characters joined by string gaps). Matrices are one row per output
  (burn's `[d_in, d_out]` transposed), `W_mix` split into `mix0..mix8`.
- `Decode.hs`: base64 to signed 16-bit values; `decodeRows` for one array, which checks the
  value count.
- `Net.hs`: the `Net` record decoded from the stream on first use (`net`), `foldRow`,
  `accumulateRows`, `hiddenOf`, `wdlOf`, `policyLogit`. Imports `FixedVec` (the
  Haskell project's `gen/vecgen.py` output: `VecN`, `RowsM_N`, `dotN`, `selM_N`,
  `linear/dense/residualM_N`, ...) and aliases `VecE`/`VecA`/`VecH`; the export prints the
  exact `vecgen.py` command line for the shapes it needs.
- `Reference.hs`: positions with cells, side to move, last move, forced board, raw codes,
  the input ids, accumulator, hidden layer, W/D/L logits and probabilities, and every legal
  move's logit and probability, as computed by the **int16-rounded** weights (so the port
  should match to Float rounding, not to quantisation error).
- `SelfTest.hs` (verification only): `tokenOf` (tokens by list lookup), `selfTest`, which
  replays the references from their ids through `Net` and checks the deviations, and
  `bench n`, which times accumulator + MLP + value head + nine policy logits.
- `Book.hs` (with `--book models/book.txt`, with or without a model): the canonical entries
  as a sorted byte stream (25 bytes each: three little-endian `Word64` key words, then the
  move index) in base64 literals, `bookKey` and the eight `squarePermutations`. The engine
  should build a balanced tree from `bookEntries` and compute keys from its own bitboards.
- `BookTest.hs` (verification only): a reference `canonicalKey`/`lookupBook` by linear
  search, and `bookRefs`/`bookTest`: book positions under random symmetries plus a few
  off-book ones, with the move Rust's `Book::lookup` gives. Key words: the 81 cell
  digits as three base-3 numbers of 27 digits, most significant first, the last times ten
  plus the forced digit, so word order is key-string order and "smallest image, first
  symmetry on ties" means the same on both sides. The 14k-entry book is 354 KB of bytes
  (8 literals), compiles in 3 s and passes its 44 lookups.

The Rust side of the same arithmetic is `export::scalar` (`Weights::evaluate`), checked
against burn's forward pass; the export report prints the quantisation error of the
rounded weights over the reference positions.

Measured (GHC 9.10.3, `-O2 -XSafe`, one core), including `FixedVec`:

| model | params | int16 stream | quantisation error (wdl / policy) | compile | self-test | µs per eval after the fold |
|---|---|---|---|---|---|---|
| `nnue 64 128 32 2` | 0.99M | 2.0 MB, 59 literals | 1.8e-4 / 3.6e-4 | 12 s, 0.35 GB | max dev 9e-6 | 7–10 |
| `nnue 64 512 128 4` | 1.59M | 3.2 MB, 84 literals | 2.8e-4 / 1.5e-3 | 60 s, 1.8–2.3 GB | max dev 1e-5 | 100–140 |

The chunked layout (8 x `Vec64` for A 512) is what makes the medium net cheap to compile
and fast to run: flat 512-wide functions were 60 s / 2.3 GB to compile and ~100 µs per
evaluation. Startup (decode of the arrays) is well under a second for both.

