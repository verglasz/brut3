#!/usr/bin/env bash
# Microbenchmarks (benches/micro.rs). Writes benches/results/micro-<date>-<commit>[-dirty].txt.
# Usage: benches/run.sh [FILTER...]   (only benchmarks whose name contains a filter)
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
mkdir -p "$here/results"

commit=$(git -C "$root" rev-parse --short HEAD)
dirty=$(git -C "$root" diff --quiet HEAD -- src benches Cargo.toml Cargo.lock || echo "-dirty")
filters=$(printf '%s' "${*:+-$*}" | tr -c 'A-Za-z0-9._-' '_')
out="$here/results/micro-$(date +%F)-$commit$dirty$filters.txt"

cd "$root"
cargo bench --all-features --bench micro --no-run 2>/dev/null

{
  echo "# microbenchmarks, $(date -Iseconds), commit $commit$dirty${*:+, filters: $*}"
  echo "# $(rustc --version), $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  echo "# median ns per unit over 15 samples [min .. max]"
  cargo bench --all-features --bench micro -- "$@" 2>/dev/null
} | tee "$out"

echo "wrote $out" >&2
