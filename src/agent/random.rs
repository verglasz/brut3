//! [`RandomAgent`]: uniformly random legal moves, and random playouts on every core.

use std::marker::PhantomData;
use std::sync::mpsc::{Receiver, Sender, channel};

use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::{Evaluations, UtttAgent};
use crate::game::{GameState, Move, MoveSet};

/// Plays a uniformly random legal move. The weakest possible baseline, and the default
/// playout policy when used as the inner agent of [`crate::mcts::Mcts`]: a batch of leaves is
/// played out in parallel on a pool of worker threads, one per core.
pub struct RandomAgent<S> {
    rng: SmallRng,
    seed: u64,
    pool: Option<Pool<S>>,
    _state: PhantomData<S>,
}

impl<S> RandomAgent<S> {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            seed,
            pool: None,
            _state: PhantomData,
        }
    }
}

/// Worker threads available for playouts.
pub fn rollout_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Plays `state` out with uniformly random moves and returns the result as a one-hot
/// `[win, draw, loss]` for the side to move at `state`.
pub fn rollout<S: GameState>(state: &S, rng: &mut SmallRng) -> [f32; 3] {
    let me = state.to_move();
    let mut s = state.clone();
    loop {
        if let Some(outcome) = s.outcome() {
            return outcome.wdl_for(me);
        }
        let legal = s.legal_moves();
        let k = rng.random_range(0..legal.len());
        s.apply(legal.iter().nth(k).expect("k < len"));
    }
}

/// Playout workers, each with its own generator seeded from the agent's seed and its index,
/// fed contiguous chunks of a batch so that results are deterministic for a given thread
/// count. Workers exit when the pool is dropped.
struct Pool<S> {
    jobs: Vec<Sender<Vec<S>>>,
    results: Vec<Receiver<Vec<[f32; 3]>>>,
}

impl<S: GameState + Send + 'static> Pool<S> {
    fn new(seed: u64, threads: usize) -> Self {
        let mut jobs = Vec::with_capacity(threads);
        let mut results = Vec::with_capacity(threads);
        for worker in 0..threads as u64 {
            let (job_tx, job_rx) = channel::<Vec<S>>();
            let (result_tx, result_rx) = channel::<Vec<[f32; 3]>>();
            let mut rng =
                SmallRng::seed_from_u64(seed ^ (worker + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            std::thread::spawn(move || {
                for states in job_rx {
                    let out: Vec<[f32; 3]> = states.iter().map(|s| rollout(s, &mut rng)).collect();
                    if result_tx.send(out).is_err() {
                        break;
                    }
                }
            });
            jobs.push(job_tx);
            results.push(result_rx);
        }
        Self { jobs, results }
    }

    fn evaluate(&self, states: &[S], out: &mut Evaluations) {
        let chunk = states.len().div_ceil(self.jobs.len()).max(1);
        let mut sent = 0;
        for (worker, states) in states.chunks(chunk).enumerate() {
            self.jobs[worker]
                .send(states.to_vec())
                .expect("playout worker alive");
            sent += 1;
        }
        for worker in 0..sent {
            for wdl in self.results[worker].recv().expect("playout worker alive") {
                out.push_value(wdl);
            }
        }
    }
}

impl<S: GameState + Send + 'static> UtttAgent for RandomAgent<S> {
    type State = S;

    fn choose_move(&mut self, state: &S) -> Move {
        let legal = state.legal_moves();
        assert!(!legal.is_empty(), "choose_move called on a terminal state");
        let k = self.rng.random_range(0..legal.len());
        legal.iter().nth(k).expect("k < len")
    }

    /// Random playouts of the whole batch, spread over the worker pool when the batch is big
    /// enough to be worth the hand-off; small batches play out on the calling thread.
    fn evaluate_batch(&mut self, states: &[S], out: &mut Evaluations) {
        let threads = rollout_threads();
        if threads < 2 || states.len() < 2 * threads {
            for s in states {
                let e = self.evaluate(s);
                out.push(&e);
            }
            return;
        }
        let seed = self.seed;
        let pool = self.pool.get_or_insert_with(|| Pool::new(seed, threads));
        pool.evaluate(states, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::BitState;

    #[test]
    fn only_plays_legal_moves_and_is_seeded() {
        let mut a = RandomAgent::<BitState>::new(5);
        let mut b = RandomAgent::<BitState>::new(5);
        let mut state = BitState::initial();
        while state.outcome().is_none() {
            let mv = a.choose_move(&state);
            assert_eq!(mv, b.choose_move(&state));
            assert!(state.is_legal(mv));
            state.apply(mv);
        }
    }

    #[test]
    fn parallel_playouts_cover_the_batch_with_results() {
        let mut agent = RandomAgent::<BitState>::new(3);
        let states = vec![BitState::initial(); 4 * rollout_threads().max(2)];
        let mut out = Evaluations::default();
        agent.evaluate_batch(&states, &mut out);
        assert_eq!(out.len(), states.len());
        for i in 0..out.len() {
            let wdl = out.wdl(i);
            assert!(wdl.iter().all(|&p| p == 0.0 || p == 1.0), "{wdl:?}");
            assert_eq!(wdl.iter().sum::<f32>(), 1.0);
        }
        // The same seed gives the same results again.
        let mut again = Evaluations::default();
        RandomAgent::<BitState>::new(3).evaluate_batch(&states, &mut again);
        assert_eq!(out, again);
    }
}
