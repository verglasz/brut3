//! Microbenchmarks: each component timed on its own, over pools of distinct positions from
//! random games so that no single input gets specialised on. Run with `benches/run.sh`, or
//! `cargo bench --features full --bench micro [-- FILTER...]` to see the output without
//! saving it. See `benches/README.md`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use brut3::agent::{Evaluations, RandomAgent, UtttAgent, heuristic};
use brut3::game::{BitMove, BitState, GameState, Move, MoveSet, Symmetry, uniform_index};
use brut3::mcts::{Footprint, SearchTree, dirichlet};
use rand::{Rng, SeedableRng, rngs::SmallRng};

/// Positions per pool.
const POOL: usize = 4096;
/// Timed samples per benchmark; the median is reported.
const SAMPLES: usize = 15;
/// Wall time of one sample.
const SAMPLE_TIME: Duration = Duration::from_millis(5);
/// Simulations per search in the `mcts/search` benchmarks.
const SIMULATIONS: u32 = 1_000;
/// Simulations per move in the `mcts/game` benchmarks.
const GAME_SIMULATIONS: u32 = 256;

struct Harness {
    filters: Vec<String>,
}

impl Harness {
    fn from_args() -> Self {
        // `cargo bench` passes `--bench` to custom harnesses; flags are not filters.
        let filters = std::env::args()
            .skip(1)
            .filter(|a| !a.starts_with("--"))
            .collect();
        Self { filters }
    }

    /// Times `f` called on `inputs` in turn, one call being `units` units of work, and prints
    /// the median time per unit with the range over all samples. The checksum is the sum of
    /// `f` over one pass of `inputs`, so runs at different commits can be checked to compute
    /// the same thing.
    fn bench<T>(&self, name: &str, units: u64, inputs: &[T], mut f: impl FnMut(&T) -> u64) {
        if !self.selects(name) {
            return;
        }
        let checksum = inputs.iter().fold(0u64, |acc, x| acc.wrapping_add(f(x)));

        // Calibrate: double the passes over `inputs` until a run is long enough to time, then
        // scale. Samples are whole passes, so that every sample sees every input equally often
        // and inputs of uneven cost do not show up as noise.
        let mut passes = 1;
        let elapsed = loop {
            let t = run(inputs, &mut f, passes);
            if t >= Duration::from_millis(2) {
                break t;
            }
            passes *= 2;
        };
        let passes = ((passes as f64 * SAMPLE_TIME.as_secs_f64() / elapsed.as_secs_f64()).round()
            as usize)
            .max(1);

        let per_sample = (passes * inputs.len()) as f64 * units as f64;
        let mut per_unit: Vec<f64> = (0..SAMPLES)
            .map(|_| run(inputs, &mut f, passes).as_nanos() as f64 / per_sample)
            .collect();
        per_unit.sort_by(f64::total_cmp);
        println!(
            "{name:<56} {:>10.2} ns  [{:>9.2} .. {:>9.2}]  checksum {}",
            per_unit[SAMPLES / 2],
            per_unit[0],
            per_unit[SAMPLES - 1],
            checksum as i64,
        );
    }

    /// Whether a filter names `name`; with none, every benchmark but the `deep/` ones, which
    /// run only when a filter names them.
    fn selects(&self, name: &str) -> bool {
        if self.filters.is_empty() {
            return !name.starts_with("deep/");
        }
        self.filters.iter().any(|w| name.contains(w.as_str()))
    }

    /// Prints a measurement that is not a time (`text`, computed only if `name` is selected).
    fn report(&self, name: &str, text: impl FnOnce() -> String) {
        if self.selects(name) {
            println!("{name:<56} {}", text());
        }
    }
}

/// `passes` passes of `f` over `inputs`.
fn run<T>(inputs: &[T], f: &mut impl FnMut(&T) -> u64, passes: usize) -> Duration {
    let mut sum = 0u64;
    let start = Instant::now();
    for _ in 0..passes {
        for x in inputs {
            sum = sum.wrapping_add(f(black_box(x)));
        }
    }
    let elapsed = start.elapsed();
    black_box(sum);
    elapsed
}

/// Non-terminal positions from random games: in the order met, and split by whether the side
/// to move is sent to one board or may play anywhere.
struct Pools {
    all: Vec<BitState>,
    forced: Vec<BitState>,
    free: Vec<BitState>,
}

fn pools() -> Pools {
    let mut rng = SmallRng::seed_from_u64(1);
    let (mut all, mut forced, mut free) = (Vec::new(), Vec::new(), Vec::new());
    while all.len() < POOL || forced.len() < POOL || free.len() < POOL {
        let mut s = BitState::initial();
        while s.outcome().is_none() {
            if all.len() < POOL {
                all.push(s);
            }
            match s.forced_board() {
                Some(_) if forced.len() < POOL => forced.push(s),
                None if free.len() < POOL => free.push(s),
                _ => {}
            }
            s.apply(random_move(&s, &mut rng));
        }
    }
    Pools { all, forced, free }
}

/// The pools' own random moves (`random_range` then `nth`), kept as they were so that the
/// pools, and the checksums over them, stay the same across commits.
fn random_move(s: &BitState, rng: &mut SmallRng) -> BitMove {
    let legal = s.legal_moves();
    legal.nth(rng.random_range(0..legal.len())).unwrap()
}

/// A cheap, deterministic, position-dependent leaf value, so that the trees get some shape
/// without paying for a real evaluator.
fn fake_value(s: &BitState) -> f32 {
    let m = s.last_move().map_or(0, |m| m.index() as u32);
    (m.wrapping_mul(0x9E37_79B9) >> 16) as f32 / 32768.0 - 1.0
}

/// One full search of `simulations` from `root` with fake leaf values, claiming up to `batch`
/// leaves per round; `priors` gives every leaf the same non-uniform prior row.
fn search(
    root: &BitState,
    simulations: u32,
    batch: usize,
    priors: Option<&[f32; Move::COUNT]>,
    states: &mut Vec<BitState>,
    evals: &mut Evaluations,
) -> u64 {
    let mut tree = SearchTree::new(*root, simulations, 1.4, 1.0);
    drive(&mut tree, batch, priors, states, evals);
    tree.result().best_move().map_or(0, |m| m.index() as u64)
}

/// Runs `tree`'s search to the end with fake leaf values (see [`search`]).
fn drive(
    tree: &mut SearchTree<BitState>,
    batch: usize,
    priors: Option<&[f32; Move::COUNT]>,
    states: &mut Vec<BitState>,
    evals: &mut Evaluations,
) {
    while !tree.is_done() {
        states.clear();
        if tree.claim(batch, states) > 0 {
            evals.clear();
            for s in states.iter() {
                match priors {
                    Some(row) => evals.push_row(fake_value(s)).copy_from_slice(row),
                    None => evals.push_value(fake_value(s)),
                }
            }
            tree.resolve(evals, 0);
        }
    }
}

/// A whole game from `root`, each side searching `simulations` per move in one tree reset
/// between moves (as self-play and matches do) and playing its most visited move. Calls
/// `each` with the tree after every search. Returns the simulations run and a checksum of the
/// moves played.
fn game(
    root: &BitState,
    simulations: u32,
    batch: usize,
    priors: Option<&[f32; Move::COUNT]>,
    states: &mut Vec<BitState>,
    evals: &mut Evaluations,
    mut each: impl FnMut(&SearchTree<BitState>),
) -> (u64, u64) {
    let mut tree = SearchTree::new(*root, simulations, 1.4, 1.0);
    let (mut sims, mut checksum) = (0u64, 0u64);
    let mut state = *root;
    loop {
        drive(&mut tree, batch, priors, states, evals);
        each(&tree);
        sims += u64::from(tree.completed());
        let Some(mv) = tree.result().best_move() else {
            break;
        };
        checksum = checksum.wrapping_mul(31).wrapping_add(mv.index() as u64);
        state.apply(BitMove::from(mv));
        if state.outcome().is_some() {
            break;
        }
        tree.reset(state, simulations, 1.4, 1.0);
    }
    (sims, checksum)
}

fn main() {
    let h = Harness::from_args();
    let p = pools();
    let mut rng = SmallRng::seed_from_u64(2);

    h.bench("baseline/loop overhead", 1, &p.all, |_| 0);

    // --- Rules ---
    for (pool, name) in [(&p.forced, "forced"), (&p.free, "free")] {
        h.bench(&format!("game/legal_moves ({name})"), 1, pool, |s| {
            s.legal_moves().len() as u64
        });
    }
    let with_move: Vec<(BitState, BitMove)> = p
        .all
        .iter()
        .map(|s| (*s, random_move(s, &mut rng)))
        .collect();
    // No "copy + apply" on its own: keeping the new position alive for `black_box` costs
    // more than the apply, so it measured slower than the version below.
    h.bench("game/outcome", 1, &p.all, |s| s.outcome().is_some() as u64);
    h.bench("game/copy + apply + outcome", 1, &with_move, |(s, mv)| {
        let mut next = *s;
        next.apply(*mv);
        next.outcome().is_some() as u64
    });
    h.bench("game/copy + play", 1, &with_move, |(s, mv)| {
        let mut next = *s;
        next.play(*mv).is_some() as u64
    });
    h.bench("game/outcome_after", 1, &with_move, |(s, mv)| {
        s.outcome_after(*mv).is_some() as u64
    });
    let with_sym: Vec<(BitState, Symmetry)> = p
        .all
        .iter()
        .enumerate()
        .map(|(i, s)| (*s, Symmetry::ALL[i % 8]))
        .collect();
    h.bench("game/transform", 1, &with_sym, |(s, sym)| {
        black_box(s.transform(*sym));
        0
    });

    // --- Picking a random move ---
    let counts: Vec<usize> = p.all.iter().map(|s| s.legal_moves().len()).collect();
    // Benchmarks that draw numbers get generators of their own, so that the inputs and
    // checksums do not depend on which benchmarks the filters select.
    let mut range_rng = SmallRng::seed_from_u64(6);
    h.bench("rng/random_range(0..legal)", 1, &counts, |n| {
        range_rng.random_range(0..*n) as u64
    });
    let mut index_rng = SmallRng::seed_from_u64(7);
    h.bench("rng/uniform_index(legal)", 1, &counts, |n| {
        uniform_index(&mut index_rng, *n) as u64
    });
    for (pool, name) in [(&p.forced, "forced"), (&p.free, "free")] {
        let picks: Vec<(<BitState as GameState>::Moves, usize)> = pool
            .iter()
            .map(|s| {
                let legal = s.legal_moves();
                (legal, rng.random_range(0..legal.len()))
            })
            .collect();
        h.bench(
            &format!("pick/MoveSet::nth(k) ({name})"),
            1,
            &picks,
            |(legal, k)| legal.nth(*k).unwrap().index() as u64,
        );
        let mut move_rng = SmallRng::seed_from_u64(8);
        h.bench(&format!("game/random_move ({name})"), 1, pool, |s| {
            s.random_move(&mut move_rng).index() as u64
        });
        let mut agent = RandomAgent::<BitState>::new(3);
        h.bench(
            &format!("agent/RandomAgent::choose_move ({name})"),
            1,
            pool,
            |s| agent.choose_move(s).index() as u64,
        );
    }

    // --- Leaf evaluators ---
    let mut agent = RandomAgent::<BitState>::new(4);
    h.bench("eval/random playout", 1, &p.all, |s| {
        (agent.evaluate(s).value + 1.0) as u64
    });
    h.bench("eval/heuristic::score", 1, &p.all, |s| {
        heuristic::score(s, s.to_move()) as i64 as u64
    });

    // --- Search, with fake leaf values: the tree's own cost per simulation ---
    // Every pass is about 16k simulations (a few ms), so that the whole section runs well
    // under a second.
    let roots = &p.all[..16];
    let mut row = [0.0f32; Move::COUNT];
    for (i, x) in row.iter_mut().enumerate() {
        *x = 1.0 + (i * 37 % 11) as f32;
    }
    let (mut states, mut evals) = (Vec::new(), Evaluations::default());
    for (batch, priors, label) in [
        (1, None, "uniform priors, batch 1"),
        (1, Some(&row), "prior rows, batch 1"),
        (16, Some(&row), "prior rows, batch 16"),
    ] {
        h.bench(
            &format!("mcts/search 1k per sim, {label}"),
            SIMULATIONS as u64,
            roots,
            |s| search(s, SIMULATIONS, batch, priors, &mut states, &mut evals),
        );
    }
    // A larger tree, which stays less in cache.
    h.bench(
        "mcts/search 10k per sim, prior rows, batch 16",
        10 * SIMULATIONS as u64,
        &roots[..2],
        |s| search(s, 10 * SIMULATIONS, 16, Some(&row), &mut states, &mut evals),
    );
    // One tree reused across searches, as a game's moves reuse it.
    let mut reused = SearchTree::new(roots[0], SIMULATIONS, 1.4, 1.0);
    h.bench(
        "mcts/search 1k per sim, prior rows, batch 16, reset tree",
        SIMULATIONS as u64,
        roots,
        |s| {
            reused.reset(*s, SIMULATIONS, 1.4, 1.0);
            drive(&mut reused, 16, Some(&row), &mut states, &mut evals);
            reused.result().best_move().map_or(0, |m| m.index() as u64)
        },
    );
    // Whole games: proven positions near the end stop their searches early, so the unit is
    // the mean number of simulations a game runs.
    let game_roots = &roots[..2];
    let game_sims: u64 = game_roots
        .iter()
        .map(|s| game(s, GAME_SIMULATIONS, 16, Some(&row), &mut states, &mut evals, |_| {}).0)
        .sum();
    h.bench(
        &format!("mcts/game {GAME_SIMULATIONS} per move, per sim, batch 16"),
        game_sims / game_roots.len() as u64,
        game_roots,
        |s| game(s, GAME_SIMULATIONS, 16, Some(&row), &mut states, &mut evals, |_| {}).1,
    );
    tree_memory(&h, roots, &row);
    deep_trees(&h, roots, &row);

    let alphas = [0.3f32; 64];
    let mut noise_rng = SmallRng::seed_from_u64(5);
    h.bench("mcts/dirichlet(0.3, 81)", 1, &alphas, |a| {
        (dirichlet(&mut noise_rng, *a, 81)[0] * 1e6) as u64
    });

    #[cfg(feature = "nn")]
    network_inputs(&h, &p);
}

/// The trees' size per simulation: what the nodes and edges in use take, and what is allocated
/// for them. Fresh trees allocate by doubling; a reset tree keeps its buffers, so over a game
/// the allocation is the largest search's (resized once, after the first).
fn tree_memory(h: &Harness, roots: &[BitState], row: &[f32; Move::COUNT]) {
    let (mut states, mut evals) = (Vec::new(), Evaluations::default());
    let per_sim = |f: &Footprint, sims: u64| {
        format!(
            "{:>7.1} B/sim used  {:>7.1} B/sim allocated  ({} B nodes x {:.2}/sim, {} B edges x {:.2}/sim)",
            f.bytes() as f64 / sims as f64,
            f.allocated() as f64 / sims as f64,
            f.node_size,
            f.nodes as f64 / sims as f64,
            f.edge_size,
            f.edges as f64 / sims as f64,
        )
    };
    for (sims, n_roots) in [(SIMULATIONS, 16), (10 * SIMULATIONS, 2)] {
        let name = format!("mem/search {}k", sims / 1000);
        h.report(&name, || {
            let mut total = Footprint::default();
            let mut completed = 0;
            for root in &roots[..n_roots] {
                let mut tree = SearchTree::new(*root, sims, 1.4, 1.0);
                drive(&mut tree, 16, Some(row), &mut states, &mut evals);
                let f = tree.footprint();
                total = Footprint {
                    nodes: total.nodes + f.nodes,
                    edges: total.edges + f.edges,
                    node_capacity: total.node_capacity + f.node_capacity,
                    edge_capacity: total.edge_capacity + f.edge_capacity,
                    ..f
                };
                completed += u64::from(tree.completed());
            }
            per_sim(&total, completed)
        });
    }
    let name = format!("mem/game {GAME_SIMULATIONS} per move, peak");
    h.report(&name, || {
        // The largest search of each game and the allocation at the end of it.
        let mut peak = Footprint::default();
        for root in &roots[..2] {
            game(
                root,
                GAME_SIMULATIONS,
                16,
                Some(row),
                &mut states,
                &mut evals,
                |tree| {
                    let f = tree.footprint();
                    if f.allocated() > peak.allocated() {
                        peak = f;
                    }
                },
            );
        }
        per_sim(&peak, u64::from(GAME_SIMULATIONS))
    });
}

/// Large trees, out of cache, and games at match strength: seconds rather than milliseconds,
/// so they run only when named (`benches/run.sh deep/`).
fn deep_trees(h: &Harness, roots: &[BitState], row: &[f32; Move::COUNT]) {
    let (mut states, mut evals) = (Vec::new(), Evaluations::default());
    const DEEP: u32 = 100_000;
    h.bench(
        "deep/search 100k per sim, prior rows, batch 16",
        u64::from(DEEP),
        &roots[..2],
        |s| search(s, DEEP, 16, Some(row), &mut states, &mut evals),
    );
    let game_roots = &roots[..2];
    let sims: u64 = game_roots
        .iter()
        .map(|s| game(s, 4096, 16, Some(row), &mut states, &mut evals, |_| {}).0)
        .sum();
    h.bench(
        "deep/game 4096 per move, per sim, batch 16",
        sims / game_roots.len() as u64,
        game_roots,
        |s| game(s, 4096, 16, Some(row), &mut states, &mut evals, |_| {}).1,
    );
    h.report("deep/mem search 100k", || {
        let mut tree = SearchTree::new(roots[0], DEEP, 1.4, 1.0);
        drive(&mut tree, 16, Some(row), &mut states, &mut evals);
        let f = tree.footprint();
        let sims = f64::from(tree.completed());
        format!(
            "{:>7.1} B/sim used  {:>7.1} B/sim allocated",
            f.bytes() as f64 / sims,
            f.allocated() as f64 / sims
        )
    });
}

/// Encoding positions into network inputs, which self-play does on the CPU for every leaf.
#[cfg(feature = "nn")]
fn network_inputs(h: &Harness, p: &Pools) {
    use brut3::nn::nnue::{FOLDED_IDS, IDS, folded_ids, nnue_ids};
    use brut3::nn::{FEATURES, encode_into};

    let mut features = [0.0f32; FEATURES];
    h.bench("nn/encode_into (199 floats)", 1, &p.all, |s| {
        encode_into(s, &mut features);
        features[0] as u64
    });
    let mut ids = [0i32; IDS];
    h.bench("nn/nnue_ids", 1, &p.all, |s| {
        nnue_ids(s, &mut ids);
        ids[0] as u64
    });
    let mut folded = [0i32; FOLDED_IDS];
    h.bench("nn/folded_ids", 1, &p.all, |s| {
        folded_ids(s, &mut folded);
        folded.iter().map(|&i| i as u64).sum()
    });
}
