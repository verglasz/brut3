#!/usr/bin/env sh
# Train an nnue model on the uttt.ai datasets: first on stage1-mcts (random-playout search
# with win/draw/loss counts), then continuing on stage2-nmcts (network-guided search with
# mean values). Expects the games files made by `import-utttai` under datasets/.
#
# Environment overrides: MODEL, NAME, EPOCHS1, EPOCHS2, BATCH, LR, LR_MIN, VALID, BACKEND,
# STAGE1, STAGE2, WORKERS. Extra arguments go to both `train` commands. Set EPOCHS1=0 to skip
# the first stage and continue from models/$NAME-s1.
set -eu
cd "$(dirname "$0")/.."

MODEL=${MODEL:-"nnue 64 128 32 2"}
NAME=${NAME:-$(echo "$MODEL" | tr ' ' '-')_$(date -Is)}
EPOCHS1=${EPOCHS1:-10}
EPOCHS2=${EPOCHS2:-10}
BATCH=${BATCH:-2048}
LR=${LR:-2e-3}
LR_MIN=${LR_MIN:-2e-5}
VALID=${VALID:-0.02}
BACKEND=${BACKEND:-cuda}
STAGE1=${STAGE1:-datasets/stage1.games}
STAGE2=${STAGE2:-datasets/stage2.games}
WORKERS=${WORKERS:-0}

# cargo build --release --features "$BACKEND"
BIN=target/release/ultimate-ttt-burn

# The burn dashboard assumes a dark background. When it is going to be shown (a terminal,
# and no --no-tui among the extra arguments), give the terminal a dark background and a light
# foreground for the run and restore its own colours on exit. Inside tmux this styles the
# pane through tmux (OSC sequences would need allow-passthrough); elsewhere it uses OSC 11
# and 10, reset with OSC 111 and 110.
dark_on() {
    if [ -n "${TMUX:-}" ]; then
        tmux select-pane -P 'bg=#1c1c1c,fg=#d7d7d7'
    else
        printf '\033]11;#1c1c1c\007\033]10;#d7d7d7\007'
    fi
}
dark_off() {
    if [ -n "${TMUX:-}" ]; then
        tmux select-pane -P 'default'
    else
        printf '\033]111\007\033]110\007'
    fi
}
if [ -t 1 ]; then
    case " $* " in
        *" --no-tui "*) ;;
        *) dark_on; trap dark_off EXIT INT TERM ;;
    esac
fi

if [ "$EPOCHS1" -gt 0 ]; then
    "$BIN" train --backend "$BACKEND" --data "$STAGE1" --model "$MODEL" \
        --epochs "$EPOCHS1" --batch-size "$BATCH" --lr "$LR" --lr-min "$LR_MIN" \
        --augment all --validation-fraction "$VALID" --workers "$WORKERS" \
        --outcome-weight 1 --search-value-weight 0.5 \
        --out "models/$NAME-s1" --artifact-dir "artifacts/$NAME-s1" "$@"
fi
if [ "$EPOCHS2" -gt 0 ]; then
    "$BIN" train --backend "$BACKEND" --data "$STAGE2" --model "$MODEL" --init "models/$NAME-s1" \
        --epochs "$EPOCHS2" --batch-size "$BATCH" --lr "$LR" --lr-min "$LR_MIN" \
        --augment all --validation-fraction "$VALID" --workers "$WORKERS" \
        --outcome-weight 1 --search-value-weight 1 \
        --out "models/$NAME-s2" --artifact-dir "artifacts/$NAME-s2" "$@"
fi
