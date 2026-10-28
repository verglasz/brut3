//! Throughput of the evaluator server alone, without any search: `THREADS` clients send
//! requests of `SIZE` positions back to back, each waiting for its answer before sending the
//! next (a self-play thread with one batch out), and the positions evaluated per second are
//! reported after a warm-up. In self-play a request is `concurrent / 2 x tree batch`
//! positions: 4096 at 1024 games in flight, 8192 at 2048.
//!
//! `cargo run --release --all-features --example server_bench -- MODEL [THREADS] [SIZE]
//! [DEPTH] [SECONDS] [BACKEND] [unfolded] [local]`, DEPTH being the server's batches in
//! flight (1 or 2); `unfolded` evaluates an NNUE without its folded table; `local` skips the
//! server and has every client evaluate its own requests with a network of its own (DEPTH is
//! then ignored), as a CPU backend can run several in parallel where the server would merge
//! them into one batch.

#[cfg(feature = "nn")]
fn main() -> Result<(), String> {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use brut3::agent::{Evaluations, UtttAgent};
    use brut3::game::{BitState, GameState};
    use brut3::nn::{BackendKind, EvalServer, MAX_BATCH, NetAgent};
    use rand::{SeedableRng, rngs::SmallRng};

    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: usize| -> Result<usize, String> {
        args.get(i).map_or(Ok(default), |a| {
            a.parse().map_err(|_| format!("bad number {a:?}"))
        })
    };
    let model = args
        .first()
        .ok_or(
            "usage: server_bench MODEL [THREADS] [SIZE] [DEPTH] [SECONDS] [BACKEND] [unfolded] \
             [local]",
        )?
        .clone();
    let (threads, size, depth, seconds) = (arg(1, 4)?, arg(2, 4096)?, arg(3, 2)?, arg(4, 10)?);
    let backend: BackendKind = args.get(5).map_or("cuda", String::as_str).parse()?;
    let flag = |name: &str| {
        args.get(6..)
            .is_some_and(|rest| rest.iter().any(|a| a == name))
    };
    let (unfolded, local) = (flag("unfolded"), flag("local"));

    // Positions from random games, from which every request takes a slice.
    let mut rng = SmallRng::seed_from_u64(1);
    let mut pool = Vec::new();
    while pool.len() < size.max(1 << 14) * 2 {
        let mut s = BitState::initial();
        loop {
            pool.push(s);
            let mv = s.random_move(&mut rng);
            if s.play(mv).is_some() {
                break;
            }
        }
    }

    let build = move || {
        let agent = NetAgent::load(std::path::Path::new(&model), backend)?;
        Ok(if unfolded { agent.unfolded() } else { agent })
    };
    let server = match local {
        true => None,
        false => Some(EvalServer::spawn_with(build.clone(), MAX_BATCH, depth)?),
    };
    // With `local`, every client's own network, built before the clock starts.
    let mut locals = Vec::new();
    if local {
        for _ in 0..threads {
            locals.push(build()?);
        }
    }
    let (stop, evaluated) = (AtomicBool::new(false), AtomicU64::new(0));
    // Requests of `size` positions back to back from client `t` until `stop`.
    fn client<E: UtttAgent<State = BitState>>(
        evaluator: &mut E,
        (t, threads, size): (usize, usize, usize),
        pool: &[BitState],
        stop: &AtomicBool,
        evaluated: &AtomicU64,
    ) {
        let mut out = Evaluations::default();
        let mut from = t * size % (pool.len() - size);
        while !stop.load(Ordering::Relaxed) {
            out.clear();
            evaluator.evaluate_batch(&pool[from..from + size], &mut out);
            evaluated.fetch_add(size as u64, Ordering::Relaxed);
            from = (from + size * threads) % (pool.len() - size);
        }
    }
    let warm_up = Duration::from_secs(3);
    let (start, end) = std::thread::scope(|scope| {
        let (pool, stop, evaluated) = (&pool, &stop, &evaluated);
        match &server {
            Some(server) => {
                for t in 0..threads {
                    scope.spawn(move || {
                        let mut remote = server.client();
                        client(&mut remote, (t, threads, size), pool, stop, evaluated);
                    });
                }
            }
            None => {
                for (t, mut agent) in locals.drain(..).enumerate() {
                    scope.spawn(move || {
                        client(&mut agent, (t, threads, size), pool, stop, evaluated);
                    });
                }
            }
        }
        std::thread::sleep(warm_up);
        let start = (Instant::now(), evaluated.load(Ordering::Relaxed));
        std::thread::sleep(Duration::from_secs(seconds as u64));
        let end = (Instant::now(), evaluated.load(Ordering::Relaxed));
        stop.store(true, Ordering::Relaxed);
        (start, end)
    });
    let secs = (end.0 - start.0).as_secs_f64();
    println!(
        "{threads} clients x {size} positions, {}{}: {:.0} positions/s over {secs:.1}s",
        if local {
            "local networks".to_string()
        } else {
            format!("depth {depth}")
        },
        if unfolded { ", unfolded" } else { "" },
        (end.1 - start.1) as f64 / secs
    );
    Ok(())
}

#[cfg(not(feature = "nn"))]
fn main() {
    eprintln!("server_bench needs the nn feature");
}
