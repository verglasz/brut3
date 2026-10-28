//! Throughput numbers for the engine, agents and (with the `nn` feature) network evaluators.
//! Run with `cargo run --release --features full --example bench [-- FILTER...]`: with
//! filters, only the cases whose label contains one of them.

use std::sync::OnceLock;
use std::time::Instant;

use brut3::agent::{GreedyAgent, MinimaxAgent, RandomAgent, UtttAgent};
use brut3::game::{BitState, GameState, MoveSet};
use brut3::mcts::{Mcts, MctsConfig};
use brut3::selfplay::{SelfPlayConfig, self_play};
use rand::{Rng, SeedableRng, rngs::SmallRng};

/// Whether a case labelled `label` is selected by the command line's filters.
fn wanted(label: &str) -> bool {
    static FILTERS: OnceLock<Vec<String>> = OnceLock::new();
    let filters = FILTERS.get_or_init(|| {
        std::env::args()
            .skip(1)
            .filter(|a| !a.starts_with("--"))
            .collect()
    });
    filters.is_empty() || filters.iter().any(|f| label.contains(f.as_str()))
}

fn time<R: Default>(label: &str, f: impl FnOnce() -> (R, u64, &'static str)) -> R {
    if !wanted(label) {
        return R::default();
    }
    let start = Instant::now();
    let (r, count, unit) = f();
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{label:<34} {:>12.0} {unit}/s  ({count} in {secs:.2}s)",
        count as f64 / secs
    );
    r
}

/// Plays `agent` against itself from the initial position, returning moves made.
fn solo_play<A: UtttAgent<State = BitState>>(agent: &mut A, games: u64) -> u64 {
    let mut moves = 0;
    for _ in 0..games {
        let mut s = BitState::initial();
        while s.outcome().is_none() {
            s.apply(agent.choose_move(&s).into());
            moves += 1;
        }
    }
    moves
}

fn main() {
    time("random playouts (engine only)", || {
        let mut rng = SmallRng::seed_from_u64(1);
        let mut moves = 0u64;
        for _ in 0..200_000 {
            let mut s = BitState::initial();
            while s.outcome().is_none() {
                let legal = s.legal_moves();
                let k = rng.random_range(0..legal.len());
                s.apply(legal.nth(k).unwrap());
                moves += 1;
            }
        }
        ((), moves, "moves")
    });

    time("RandomAgent self-play", || {
        let mut a = RandomAgent::<BitState>::new(1);
        ((), solo_play(&mut a, 100_000), "moves")
    });

    time("GreedyAgent self-play", || {
        let mut a = GreedyAgent::new(1);
        ((), solo_play(&mut a, 5_000), "moves")
    });

    for depth in [2, 4] {
        time(&format!("MinimaxAgent depth {depth} self-play"), || {
            let mut a = MinimaxAgent::new(depth, 1);
            let games = if depth <= 2 { 200 } else { 10 };
            ((), solo_play(&mut a, games), "moves")
        });
    }

    for batch_size in [1, 16] {
        time(&format!("MCTS random rollouts, batch {batch_size}"), || {
            let mut m = Mcts::new(
                RandomAgent::<BitState>::new(1),
                MctsConfig {
                    simulations: 10_000,
                    batch_size,
                    ..Default::default()
                },
            );
            let mut sims = 0u64;
            for _ in 0..20 {
                m.search(&BitState::initial());
                sims += 10_000;
            }
            ((), sims, "simulations")
        });
    }

    time("lockstep self-play, random leaves, 64 games x 8", || {
        let mut evaluator = RandomAgent::<BitState>::new(1);
        let cfg = SelfPlayConfig {
            simulations: 100,
            tree_batch: 8,
            concurrent: 64,
            ..Default::default()
        };
        let (_, stats) = self_play(&mut evaluator, 64, &cfg, |_| {});
        ((), stats.simulations, "simulations")
    });

    #[cfg(feature = "nn")]
    network_benchmarks();

    time("MCTS greedy-heuristic leaves", || {
        let mut m = Mcts::new(
            GreedyAgent::new(1),
            MctsConfig {
                simulations: 10_000,
                ..Default::default()
            },
        );
        let mut sims = 0u64;
        for _ in 0..20 {
            m.search(&BitState::initial());
            sims += 10_000;
        }
        ((), sims, "simulations")
    });
}

#[cfg(feature = "nn")]
fn network_benchmarks() {
    use brut3::agent::{Evaluations, UtttAgent};
    use brut3::nn::{BackendKind, ModelSpec, NetAgent};

    let kind = BackendKind::preferred();
    let mut rng = SmallRng::seed_from_u64(1);
    // A pool of random positions to evaluate.
    let mut positions = Vec::new();
    while positions.len() < 4096 {
        let mut s = BitState::initial();
        while s.outcome().is_none() && positions.len() < 4096 {
            positions.push(s);
            let legal = s.legal_moves();
            let k = rng.random_range(0..legal.len());
            s.apply(legal.nth(k).unwrap());
        }
    }
    let nnue = vec!["nnue", "64", "512", "128", "4"];
    // (name, spec, backend, folded): NNUE models are folded for inference unless asked not to.
    let mut runs = vec![
        ("linear", vec!["linear"], kind, true),
        (
            "board-transformer 64 2 4",
            vec!["board-transformer", "64", "2", "4"],
            kind,
            true,
        ),
        (
            "cell-transformer 64 2 4",
            vec!["cell-transformer", "64", "2", "4"],
            kind,
            true,
        ),
    ];
    // On the preferred backend, and on the pure-Rust CPU one when that is another.
    let mut backends = vec![kind];
    if kind != BackendKind::Flex && BackendKind::Flex.is_compiled() {
        backends.push(BackendKind::Flex);
    }
    for backend in backends {
        runs.push(("nnue 64 512 128 4, folded", nnue.clone(), backend, true));
        runs.push(("nnue 64 512 128 4, unfolded", nnue.clone(), backend, false));
    }
    for (name, words, kind, folded) in runs {
        if !wanted(&format!("{name} on {kind}")) && !wanted(&format!("self-play {name} on {kind}"))
        {
            continue;
        }
        let spec = ModelSpec::parse(&words).unwrap();
        let agent = NetAgent::new(spec.init(&kind.device()), kind);
        let mut agent = if folded { agent } else { agent.unfolded() };
        let mut out = Evaluations::default();
        for batch in [16usize, 256, 2048] {
            // Warm up this shape first: kernel compilation and autotune happen on first use.
            out.clear();
            agent.evaluate_batch(&positions[..batch], &mut out);
            time(&format!("{name} on {kind}, batch {batch}"), || {
                let mut n = 0u64;
                let rounds = (65536 / batch).max(16);
                for i in 0..rounds {
                    let from = (i * batch) % (positions.len() - batch);
                    out.clear();
                    agent.evaluate_batch(&positions[from..from + batch], &mut out);
                    n += batch as u64;
                }
                ((), n, "leaves")
            });
        }
        time(
            &format!("self-play {name} on {kind}, 64 games x 8, 100 sims"),
            || {
                let cfg = SelfPlayConfig {
                    simulations: 100,
                    tree_batch: 8,
                    concurrent: 64,
                    ..Default::default()
                };
                let (_, stats) = self_play(&mut agent, 64, &cfg, |_| {});
                ((), stats.simulations, "simulations")
            },
        );
    }
}
