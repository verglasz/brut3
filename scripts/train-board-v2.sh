#!/usr/bin/env sh
# A two-hour self-play training run of the board-v2 model, measured each generation against
# three fixed opponents (random, greedy, random-playout MCTS), ten games each.
#
# Environment overrides: MODEL, DIR, MINUTES, CONCURRENT, GAMES, GAMES_GROWTH, MAX_GAMES,
# BUFFER_GENERATIONS, EPOCHS, LR, LR_DECAY, LR_MIN, EVAL_GAMES, BACKEND. Extra arguments are
# passed to the `loop` command, e.g. `scripts/train-board-v2.sh --resume` to continue a
# stopped run (flags already set below cannot be repeated; use the environment variables).
#
# Sizing: the evaluator sees CONCURRENT / 2 x 8 positions per forward pass, and a generation
# idles its slots for about half a game at the end. Generations start at two games per slot,
# refreshing the self-play network often while it improves fastest, and grow by 10% per
# generation to eight games per slot, where the fixed per-generation costs (tail, matches,
# training setup) are small. The learning rate decays 5% per generation down to a tenth of
# its start. The replay buffer holds the last five generations and is trained over once per
# generation.
#
# Stopping early is safe at any point: Ctrl-C during self-play or a match kills the process
# and the files on disk (best.*, history.jsonl) describe the last finished generation. While
# the training dashboard is up the terminal is in raw mode, so Ctrl-C does nothing there:
# press `q` to end that generation's training early, or kill the process from another shell.
set -eu
cd "$(dirname "$0")/.."

MODEL=${MODEL:-"board-v2 128 4 4"}
DIR=${DIR:-models/board-v2}
MINUTES=${MINUTES:-120}
CONCURRENT=${CONCURRENT:-512}
GAMES=${GAMES:-$((CONCURRENT * 2))}
GAMES_GROWTH=${GAMES_GROWTH:-1.1}
MAX_GAMES=${MAX_GAMES:-$((CONCURRENT * 8))}
BUFFER_GENERATIONS=${BUFFER_GENERATIONS:-5}
EPOCHS=${EPOCHS:-1}
LR=${LR:-1e-3}
LR_DECAY=${LR_DECAY:-0.95}
LR_MIN=${LR_MIN:-1e-4}
EVAL_GAMES=${EVAL_GAMES:-60}
BACKEND=${BACKEND:-cuda}

cargo build --release --features "$BACKEND"
exec target/release/brut3 loop \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --dir "$DIR" \
    --generations 100000 \
    --max-minutes "$MINUTES" \
    --games "$GAMES" \
    --games-growth "$GAMES_GROWTH" \
    --max-games "$MAX_GAMES" \
    --concurrent "$CONCURRENT" \
    --buffer-generations "$BUFFER_GENERATIONS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lr-decay "$LR_DECAY" \
    --lr-min "$LR_MIN" \
    --simulations 200 \
    --eval-games "$EVAL_GAMES" \
    --baseline random \
    --baseline greedy \
    --baseline "mcts 200" \
    --baseline-games 30 \
    --tui \
    "$@"
