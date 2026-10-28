//! [`EvalServer`]: one thread that owns a batched evaluator (a network on its device) and
//! evaluates the leaves of many search threads together, which talk to it through
//! [`RemoteEvaluator`]s.
//!
//! A network agent per search thread made every thread pay for launching its own forward
//! passes: about 0.75 ms of host time per batch whatever its size, half of a self-play
//! thread's time, and with several threads the launches contended (1.5 ms each at four
//! threads) while every thread also held its own folded table. Here only the server thread
//! touches the device: it takes whatever requests are queued, evaluates them as one batch and
//! hands each its rows back, while the search threads keep searching. It can keep a second
//! batch in flight, launching one while the other is read back ([`EvalServer::spawn_with`]),
//! which measured no faster.

use std::ops::Range;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use crate::agent::{Evaluation, Evaluations, UtttAgent};
use crate::game::{BitState, GameState, Move, MoveSet};

/// Most positions the server gathers into one batch. The folded network's accumulator gather
/// holds `positions x 21 x d_acc` floats on the device (350 MB at 8192 and `d_acc` 512).
pub const MAX_BATCH: usize = 8192;

/// Positions to evaluate and where to send their evaluations.
struct Request {
    states: Vec<BitState>,
    reply: Sender<Evaluations>,
}

/// The evaluator thread and the way to reach it. Dropping it waits for the thread, which ends
/// once every [`RemoteEvaluator`] is gone too.
pub struct EvalServer {
    requests: Option<Sender<Request>>,
    thread: Option<JoinHandle<()>>,
}

impl EvalServer {
    /// [`EvalServer::spawn_with`] one batch deep: a second batch in flight measured the same
    /// (`examples/server_bench`: 1.36-1.44M positions/s both ways at 4 clients of 4096 or 8192
    /// positions, 64-512-128-4 on an RTX 2080), and costs a second agent and folded table.
    pub fn spawn<A, F>(build: F, max_batch: usize) -> Result<Self, String>
    where
        A: UtttAgent<State = BitState>,
        F: Fn() -> Result<A, String> + Send + 'static,
    {
        Self::spawn_with(build, max_batch, 1)
    }

    /// Starts the evaluator thread, which builds `depth` agents (1 or 2) with `build` (there,
    /// so that a device agent's work stays on that thread), and returns once the builds have
    /// succeeded or failed. With two, one batch is launched while the other is read back; with
    /// one, each batch is finished before the next is gathered. Batches hold at most
    /// `max_batch` positions (a single larger request is still evaluated whole).
    pub fn spawn_with<A, F>(build: F, max_batch: usize, depth: usize) -> Result<Self, String>
    where
        A: UtttAgent<State = BitState>,
        F: Fn() -> Result<A, String> + Send + 'static,
    {
        assert!(matches!(depth, 1 | 2), "server depth must be 1 or 2");
        let (requests, queue) = mpsc::channel::<Request>();
        let (ready, built) = mpsc::channel::<Result<(), String>>();
        let thread = std::thread::Builder::new()
            .name("evaluator".to_string())
            .spawn(
                move || match (0..depth).map(|_| build()).collect::<Result<Vec<A>, _>>() {
                    Ok(agents) => {
                        let _ = ready.send(Ok(()));
                        serve(agents, &queue, max_batch);
                    }
                    Err(e) => {
                        let _ = ready.send(Err(e));
                    }
                },
            )
            .map_err(|e| format!("cannot start the evaluator thread: {e}"))?;
        built
            .recv()
            .map_err(|_| "the evaluator thread died while building its agents".to_string())??;
        Ok(Self {
            requests: Some(requests),
            thread: Some(thread),
        })
    }

    /// A new handle for one search thread.
    pub fn client(&self) -> RemoteEvaluator {
        let (reply_to, replies) = mpsc::channel();
        RemoteEvaluator {
            requests: self.requests.clone().expect("server running"),
            reply_to,
            replies,
            pending: false,
        }
    }
}

impl Drop for EvalServer {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A batch on its way through one of the server's agents: the requests it answers, in order,
/// and their positions.
#[derive(Default)]
struct Slot {
    requests: Vec<Request>,
    states: Vec<BitState>,
    in_flight: bool,
}

/// The evaluator thread's loop, one batch in flight per agent: gather the queued requests into
/// one batch and launch it on one agent, then read back the other agent's batch (with one
/// agent, that same batch) and answer it, so that with two the device works on the new batch
/// during the readback. It only waits for requests when nothing is in flight, and ends when
/// the queue is closed and every slot is empty.
fn serve<A: UtttAgent<State = BitState>>(
    mut agents: Vec<A>,
    queue: &Receiver<Request>,
    max_batch: usize,
) {
    let mut slots: Vec<Slot> = agents.iter().map(|_| Slot::default()).collect();
    let mut evals = Evaluations::default();
    let mut current = 0;
    loop {
        let other = (current + 1) % agents.len();
        let wait = !slots[other].in_flight;
        if !gather(queue, &mut slots[current].requests, max_batch, wait) {
            break;
        }
        let slot = &mut slots[current];
        if !slot.requests.is_empty() {
            slot.states.clear();
            for request in &slot.requests {
                slot.states.extend_from_slice(&request.states);
            }
            agents[current].begin_batch(&slot.states);
            slot.in_flight = true;
        }
        let done = &mut slots[other];
        if done.in_flight {
            evals.clear();
            agents[other].finish_batch(&done.states, &mut evals);
            done.in_flight = false;
            let mut offset = 0;
            for request in done.requests.drain(..) {
                let n = request.states.len();
                let mut part = Evaluations::default();
                copy_rows(&evals, offset..offset + n, &mut part);
                offset += n;
                // A client that went away no longer needs its answer.
                let _ = request.reply.send(part);
            }
        }
        current = other;
    }
}

/// Moves the queued requests into `batch`, up to `max_batch` positions, first waiting for one
/// if `wait`. Returns false when waiting finds the queue closed.
fn gather(
    queue: &Receiver<Request>,
    batch: &mut Vec<Request>,
    max_batch: usize,
    wait: bool,
) -> bool {
    let mut positions = 0;
    if wait {
        match queue.recv() {
            Ok(request) => {
                positions += request.states.len();
                batch.push(request);
            }
            Err(_) => return false,
        }
    }
    while positions < max_batch {
        match queue.try_recv() {
            Ok(request) => {
                positions += request.states.len();
                batch.push(request);
            }
            Err(_) => break,
        }
    }
    true
}

/// Appends `src`'s evaluations `rows` to `dst`.
fn copy_rows(src: &Evaluations, rows: Range<usize>, dst: &mut Evaluations) {
    for i in rows {
        match src.priors(i) {
            Some(priors) => dst.push_row(src.value(i)).copy_from_slice(priors),
            None => dst.push_value(src.value(i)),
        }
    }
}

/// A search thread's handle on an [`EvalServer`]: an agent whose batches are evaluated by the
/// server, together with other threads'. One request in flight at a time, like
/// [`UtttAgent::begin_batch`] asks.
pub struct RemoteEvaluator {
    requests: Sender<Request>,
    reply_to: Sender<Evaluations>,
    replies: Receiver<Evaluations>,
    pending: bool,
}

impl RemoteEvaluator {
    fn one(&mut self, state: &BitState) -> Evaluations {
        let mut out = Evaluations::default();
        self.evaluate_batch(std::slice::from_ref(state), &mut out);
        out
    }
}

impl UtttAgent for RemoteEvaluator {
    type State = BitState;

    /// The legal move with the highest prior (the first legal one without priors).
    fn choose_move(&mut self, state: &BitState) -> Move {
        let out = self.one(state);
        let legal = state.legal_moves();
        let best = match out.priors(0) {
            Some(p) => legal
                .iter()
                .max_by(|a, b| p[a.index()].total_cmp(&p[b.index()])),
            None => legal.iter().next(),
        };
        best.expect("choose_move on a finished game").into()
    }

    fn evaluate(&mut self, state: &BitState) -> Evaluation {
        let out = self.one(state);
        Evaluation {
            value: out.value(0),
            priors: match out.priors(0) {
                Some(p) => state
                    .legal_moves()
                    .iter()
                    .map(|mv| (mv.into(), p[mv.index()]))
                    .collect(),
                None => Vec::new(),
            },
        }
    }

    fn evaluate_batch(&mut self, states: &[BitState], out: &mut Evaluations) {
        self.begin_batch(states);
        self.finish_batch(states, out);
    }

    /// Sends the positions to the server; the search carries on until `finish_batch`.
    fn begin_batch(&mut self, states: &[BitState]) {
        assert!(!self.pending, "finish the previous batch first");
        if states.is_empty() {
            return;
        }
        self.requests
            .send(Request {
                states: states.to_vec(),
                reply: self.reply_to.clone(),
            })
            .expect("the evaluator thread stopped");
        self.pending = true;
    }

    /// Waits for the server's answer and appends it to `out`.
    fn finish_batch(&mut self, states: &[BitState], out: &mut Evaluations) {
        if !self.pending {
            assert!(states.is_empty(), "finish_batch without begin_batch");
            return;
        }
        self.pending = false;
        let evals = self.replies.recv().expect("the evaluator thread stopped");
        assert_eq!(evals.len(), states.len(), "one evaluation per position");
        copy_rows(&evals, 0..evals.len(), out);
    }
}
