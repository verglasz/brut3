#!/usr/bin/env sh
# A two-hour self-play training run of the board-v2 model, measured each generation against
# three fixed opponents (random, greedy, random-playout MCTS), ten games each.
#
# Environment overrides: MODEL, DIR, MINUTES, GAMES, CONCURRENT, BUFFER_GAMES, EPOCHS,
# EVAL_GAMES, BACKEND. Extra arguments are passed to the `loop` command, e.g.
# `scripts/train-board-v2.sh --resume` to continue a stopped run (flags already set below
# cannot be repeated; use the environment variables for those).
#
# Sizing: the evaluator sees CONCURRENT / 2 x 8 positions per forward pass, and a generation
# idles its slots for about half a game at the end, so GAMES is kept at 4 x CONCURRENT. The
# replay buffer holds five generations and is trained over once per generation.
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
GAMES=${GAMES:-$((CONCURRENT * 4))}
BUFFER_GAMES=${BUFFER_GAMES:-$((GAMES * 5))}
EPOCHS=${EPOCHS:-1}
EVAL_GAMES=${EVAL_GAMES:-60}
BACKEND=${BACKEND:-cuda}

cargo build --release --features "$BACKEND"
exec target/release/ultimate-ttt-burn loop \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --dir "$DIR" \
    --generations 100000 \
    --max-minutes "$MINUTES" \
    --games "$GAMES" \
    --concurrent "$CONCURRENT" \
    --buffer-games "$BUFFER_GAMES" \
    --epochs "$EPOCHS" \
    --simulations 200 \
    --eval-games "$EVAL_GAMES" \
    --baseline random \
    --baseline greedy \
    --baseline "mcts 200" \
    --baseline-games 10 \
    --tui \
    "$@"
