#!/usr/bin/env sh
# A two-hour self-play training run of the board-v2 model, measured each generation against
# three fixed opponents (random, greedy, random-playout MCTS), ten games each.
#
# Environment overrides: MODEL, DIR, MINUTES, GAMES, BACKEND. Extra arguments are passed to
# the `loop` command, e.g. `scripts/train-board-v2.sh --resume` to continue a stopped run.
#
# Stopping early is safe at any point: Ctrl-C during self-play or a match kills the process
# and the files on disk (best.*, history.jsonl) describe the last finished generation. While
# the training dashboard is up the terminal is in raw mode, so Ctrl-C does nothing there:
# press `q` to end that generation's training early, or kill the process from another shell.
set -eu
cd "$(dirname "$0")/.."

MODEL=${MODEL:-"board-v2 64 2 4"}
DIR=${DIR:-models/board-v2}
MINUTES=${MINUTES:-120}
GAMES=${GAMES:-500}
BACKEND=${BACKEND:-cuda}

cargo build --release --features "$BACKEND"
exec target/release/ultimate-ttt-burn loop \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --dir "$DIR" \
    --generations 100000 \
    --max-minutes "$MINUTES" \
    --games "$GAMES" \
    --simulations 200 \
    --eval-games 20 \
    --baseline random \
    --baseline greedy \
    --baseline "mcts 200" \
    --baseline-games 10 \
    --tui \
    "$@"
