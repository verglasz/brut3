//! PUCT Monte Carlo Tree Search.
//!
//! [`SearchTree`] is the search itself, written as a resumable state machine: it hands out
//! leaves to evaluate ([`SearchTree::claim`]) and takes their evaluations back
//! ([`SearchTree::resolve`]), so that any number of trees can share one batched evaluator.
//! [`Mcts`] drives a single tree with its own agent, which is what the REPL and the
//! benchmark use.
//!
//! With an agent whose [`UtttAgent::evaluate`] does random playouts and returns no priors this
//! is classic rollout MCTS; with a network agent returning priors and a value it is the
//! AlphaZero search. One code path covers both.
//!
//! Within one tree, up to `batch_size` leaves are selected per round, each claimed leaf
//! carrying a virtual loss so the next selection goes elsewhere. With `batch_size = 1` this is
//! plain sequential MCTS.

use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::agent::{Evaluation, UtttAgent};
use crate::game::{GameState, Move, MoveSet, Outcome};

#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Leaf evaluations per search (terminal leaves count too).
    pub simulations: u32,
    /// Exploration constant in the PUCT formula.
    pub c_puct: f32,
    /// Move selection: `0` picks the most visited move; `> 0` samples proportionally to
    /// `visits^(1/temperature)` (used for self-play diversity).
    pub temperature: f32,
    /// Leaves selected per evaluation round. `1` = sequential search.
    pub batch_size: usize,
    /// Value a claimed-but-unevaluated leaf is assumed to have, from its parent's view, while
    /// the rest of the batch is selected. Keeps one round from piling onto a single leaf.
    pub virtual_loss: f32,
    pub seed: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            simulations: 800,
            c_puct: 1.4,
            temperature: 0.0,
            batch_size: 1,
            virtual_loss: 1.0,
            seed: 0x5EED,
        }
    }
}

/// Root statistics of a finished search. `visits` is empty if the root was terminal.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchResult {
    /// Mean value of the root, from the perspective of the side to move.
    pub root_value: f32,
    pub visits: Vec<(Move, u32)>,
}

impl SearchResult {
    /// Most visited move; ties go to the first.
    pub fn best_move(&self) -> Option<Move> {
        self.visits.iter().max_by_key(|(_, n)| *n).map(|(m, _)| *m)
    }

    /// Visit counts normalised to a distribution.
    pub fn policy(&self) -> Vec<(Move, f32)> {
        let total: u32 = self.visits.iter().map(|(_, n)| n).sum();
        if total == 0 {
            return Vec::new();
        }
        self.visits
            .iter()
            .map(|(m, n)| (*m, *n as f32 / total as f32))
            .collect()
    }

    /// Samples a move with probability proportional to `visits^(1/temperature)`; with
    /// temperature `0` (or below) the most visited move.
    pub fn sample_move(&self, temperature: f32, rng: &mut impl Rng) -> Option<Move> {
        if self.visits.is_empty() {
            return None;
        }
        if temperature <= 0.0 {
            return self.best_move();
        }
        let inv_t = 1.0 / temperature;
        let weights: Vec<f32> = self
            .visits
            .iter()
            .map(|(_, n)| (*n as f32).powf(inv_t))
            .collect();
        let total: f32 = weights.iter().sum();
        let mut r = rng.random::<f32>() * total;
        for ((mv, _), w) in self.visits.iter().zip(&weights) {
            r -= w;
            if r <= 0.0 {
                return Some(*mv);
            }
        }
        self.visits.last().map(|(mv, _)| *mv)
    }
}

const NO_NODE: u32 = u32::MAX;

struct Node<S> {
    state: S,
    /// Move that led here; `None` for the root.
    mv: Option<Move>,
    parent: u32,
    /// Children are contiguous: `first_child..first_child + n_children`.
    first_child: u32,
    n_children: u8,
    expanded: bool,
    /// Claimed by the current round, awaiting evaluation.
    pending: bool,
    visits: u32,
    /// Sum of backed-up values, from the perspective of the side to move *at this node*.
    value_sum: f32,
    /// Virtual losses currently applied (paths through here claimed in the current round).
    virtual_losses: u32,
    prior: f32,
    outcome: Option<Outcome>,
}

impl<S: GameState> Node<S> {
    fn new(state: S, mv: Option<Move>, parent: u32, prior: f32) -> Self {
        let outcome = state.outcome();
        Self {
            state,
            mv,
            parent,
            first_child: NO_NODE,
            n_children: 0,
            expanded: false,
            pending: false,
            visits: 0,
            value_sum: 0.0,
            virtual_losses: 0,
            prior,
            outcome,
        }
    }

    fn children(&self) -> std::ops::Range<u32> {
        self.first_child..self.first_child + self.n_children as u32
    }

    /// Visits including virtual ones.
    fn effective_visits(&self) -> u32 {
        self.visits + self.virtual_losses
    }

    /// Mean value from the perspective of the *parent's* side to move, with each virtual loss
    /// counted as `virtual_loss` for this node's side (i.e. a loss for the parent).
    fn q_for_parent(&self, virtual_loss: f32) -> f32 {
        let n = self.effective_visits();
        if n == 0 {
            0.0
        } else {
            -(self.value_sum + virtual_loss * self.virtual_losses as f32) / n as f32
        }
    }
}

/// Dirichlet noise mixed into the root priors once the root is expanded (AlphaZero's
/// self-play exploration).
#[derive(Clone, Debug)]
pub struct RootNoise {
    /// Dirichlet concentration; AlphaZero used `0.3` for chess, smaller for larger boards.
    pub alpha: f32,
    /// Weight of the noise in the mixed prior, `0.25` in AlphaZero.
    pub fraction: f32,
    pub seed: u64,
}

enum Selection {
    /// An unexpanded, unclaimed leaf.
    Leaf(u32),
    /// A terminal node and its value for the side to move there.
    Terminal(u32, f32),
    /// Selection ran into a leaf already claimed in this round.
    Blocked,
}

/// One search, resumable between leaf evaluations.
///
/// Call [`SearchTree::claim`] to get leaves to evaluate, evaluate them however you like, and
/// hand the results to [`SearchTree::resolve`]; repeat until [`SearchTree::is_done`].
pub struct SearchTree<S: GameState> {
    nodes: Vec<Node<S>>,
    c_puct: f32,
    virtual_loss: f32,
    /// Leaves claimed in the current round, in claim order.
    claimed: Vec<u32>,
    /// Simulations completed (leaf evaluations plus terminal hits).
    completed: u32,
    budget: u32,
    noise: Option<(RootNoise, SmallRng)>,
    prior_buf: [f32; 81],
}

impl<S: GameState> SearchTree<S> {
    /// A search from `root` of `simulations` leaf evaluations.
    pub fn new(root: S, simulations: u32, c_puct: f32, virtual_loss: f32) -> Self {
        Self {
            nodes: vec![Node::new(root, None, NO_NODE, 1.0)],
            c_puct,
            virtual_loss,
            claimed: Vec::new(),
            completed: 0,
            budget: simulations,
            noise: None,
            prior_buf: [0.0; 81],
        }
    }

    /// Mixes Dirichlet noise into the root priors when the root gets expanded.
    pub fn with_root_noise(mut self, noise: RootNoise) -> Self {
        let rng = SmallRng::seed_from_u64(noise.seed);
        self.noise = Some((noise, rng));
        self
    }

    pub fn root(&self) -> &S {
        &self.nodes[0].state
    }

    pub fn is_done(&self) -> bool {
        self.completed >= self.budget || self.nodes[0].outcome.is_some()
    }

    pub fn completed(&self) -> u32 {
        self.completed
    }

    /// Selects up to `max` leaves (never beyond the remaining budget) and appends their
    /// states to `out`, in the order [`SearchTree::resolve`] expects evaluations. Terminal
    /// nodes met on the way are backed up immediately and count as simulations. Returns how
    /// many leaves were claimed; `0` with the search not done means only terminals were hit
    /// this round, so just call again.
    pub fn claim(&mut self, max: usize, out: &mut Vec<S>) -> usize {
        assert!(self.claimed.is_empty(), "resolve the previous round first");
        if self.is_done() {
            return 0;
        }
        let want = max.max(1).min((self.budget - self.completed) as usize);
        for _ in 0..want {
            match self.select_leaf() {
                Selection::Terminal(idx, value) => {
                    self.backup(idx, value);
                    self.completed += 1;
                    if self.completed >= self.budget {
                        break;
                    }
                }
                Selection::Leaf(idx) => {
                    self.nodes[idx as usize].pending = true;
                    self.add_virtual_loss(idx);
                    self.claimed.push(idx);
                    out.push(self.nodes[idx as usize].state.clone());
                }
                // Every path leads to an already-claimed leaf: evaluate what we have.
                Selection::Blocked => break,
            }
        }
        self.claimed.len()
    }

    /// Expands the leaves of the last [`SearchTree::claim`] with `evaluations` (same order)
    /// and backs their values up.
    pub fn resolve(&mut self, evaluations: &[Evaluation]) {
        assert_eq!(
            evaluations.len(),
            self.claimed.len(),
            "one evaluation per claimed leaf"
        );
        let claimed = std::mem::take(&mut self.claimed);
        for (&idx, eval) in claimed.iter().zip(evaluations) {
            self.remove_virtual_loss(idx);
            self.nodes[idx as usize].pending = false;
            let value = self.expand(idx, eval);
            self.backup(idx, value);
            self.completed += 1;
        }
        self.claimed = claimed;
        self.claimed.clear();
    }

    /// Root statistics so far.
    pub fn result(&self) -> SearchResult {
        let root = &self.nodes[0];
        if let Some(outcome) = root.outcome {
            return SearchResult {
                root_value: outcome.value_for(root.state.to_move()),
                visits: Vec::new(),
            };
        }
        let visits = root
            .children()
            .map(|c| {
                let child = &self.nodes[c as usize];
                (child.mv.expect("child has a move"), child.visits)
            })
            .collect();
        SearchResult {
            root_value: root.value_sum / root.visits.max(1) as f32,
            visits,
        }
    }

    fn select_leaf(&self) -> Selection {
        let mut idx = 0u32;
        loop {
            let node = &self.nodes[idx as usize];
            if let Some(outcome) = node.outcome {
                return Selection::Terminal(idx, outcome.value_for(node.state.to_move()));
            }
            if !node.expanded {
                return if node.pending {
                    Selection::Blocked
                } else {
                    Selection::Leaf(idx)
                };
            }
            idx = self.select_child(idx);
        }
    }

    fn select_child(&self, idx: u32) -> u32 {
        let node = &self.nodes[idx as usize];
        let sqrt_n = (node.effective_visits() as f32).sqrt();
        let mut best = node.first_child;
        let mut best_score = f32::NEG_INFINITY;
        for c in node.children() {
            let child = &self.nodes[c as usize];
            let exploration =
                self.c_puct * child.prior * sqrt_n / (1.0 + child.effective_visits() as f32);
            let score = child.q_for_parent(self.virtual_loss) + exploration;
            if score > best_score {
                best_score = score;
                best = c;
            }
        }
        best
    }

    fn add_virtual_loss(&mut self, mut idx: u32) {
        while idx != NO_NODE {
            let node = &mut self.nodes[idx as usize];
            node.virtual_losses += 1;
            idx = node.parent;
        }
    }

    fn remove_virtual_loss(&mut self, mut idx: u32) {
        while idx != NO_NODE {
            let node = &mut self.nodes[idx as usize];
            node.virtual_losses -= 1;
            idx = node.parent;
        }
    }

    /// Adds `value` (for the side to move at `idx`) along the path to the root, flipping sign
    /// at each step.
    fn backup(&mut self, mut idx: u32, mut value: f32) {
        while idx != NO_NODE {
            let node = &mut self.nodes[idx as usize];
            node.visits += 1;
            node.value_sum += value;
            value = -value;
            idx = node.parent;
        }
    }

    /// Creates the children of `idx` from `eval`'s priors; returns the clamped value.
    fn expand(&mut self, idx: u32, eval: &Evaluation) -> f32 {
        let state = self.nodes[idx as usize].state.clone();
        let moves = state.legal_moves();
        debug_assert!(!moves.is_empty(), "non-terminal state without legal moves");

        // Keep only legal moves' priors; fall back to uniform if nothing usable was given.
        let mut total = 0.0f32;
        if !eval.priors.is_empty() {
            self.prior_buf.fill(0.0);
            for (mv, p) in &eval.priors {
                if moves.contains(*mv) {
                    self.prior_buf[mv.index()] += p.max(0.0);
                    total += p.max(0.0);
                }
            }
        }
        let uniform = total <= 1e-12;
        let n = moves.len();

        let first_child = self.nodes.len() as u32;
        for mv in moves {
            let prior = if uniform {
                1.0 / n as f32
            } else {
                self.prior_buf[mv.index()] / total
            };
            let mut child_state = state.clone();
            child_state.apply(mv);
            self.nodes
                .push(Node::new(child_state, Some(mv), idx, prior));
        }

        let node = &mut self.nodes[idx as usize];
        node.first_child = first_child;
        node.n_children = n as u8;
        node.expanded = true;
        if idx == 0 {
            self.apply_root_noise();
        }
        eval.value.clamp(-1.0, 1.0)
    }

    fn apply_root_noise(&mut self) {
        let Some((noise, rng)) = self.noise.as_mut() else {
            return;
        };
        let children = self.nodes[0].children();
        let sample = dirichlet(rng, noise.alpha, children.len());
        for (c, d) in children.zip(sample) {
            let child = &mut self.nodes[c as usize];
            child.prior = (1.0 - noise.fraction) * child.prior + noise.fraction * d;
        }
    }

    #[cfg(test)]
    pub(crate) fn root_children(&self) -> Vec<(Move, f32, u32)> {
        self.nodes[0]
            .children()
            .map(|c| {
                let n = &self.nodes[c as usize];
                (n.mv.unwrap(), n.prior, n.visits)
            })
            .collect()
    }

    /// Checks the tree between rounds: no virtual losses or pending leaves remain, and every
    /// expanded node's visits equal its own evaluation plus its children's visits.
    #[cfg(test)]
    pub(crate) fn assert_consistent(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            assert_eq!(node.virtual_losses, 0, "node {i} has virtual losses");
            assert!(!node.pending, "node {i} still pending");
            if node.expanded {
                let child_visits: u32 =
                    node.children().map(|c| self.nodes[c as usize].visits).sum();
                assert_eq!(node.visits, child_visits + 1, "node {i} visit mismatch");
            }
        }
    }
}

/// A sample from a symmetric Dirichlet distribution with concentration `alpha`, as
/// normalised gamma variates (Marsaglia and Tsang, with the `alpha < 1` boost).
pub fn dirichlet(rng: &mut impl Rng, alpha: f32, n: usize) -> Vec<f32> {
    let mut sample: Vec<f32> = (0..n).map(|_| gamma(rng, alpha)).collect();
    let total: f32 = sample.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return vec![1.0 / n.max(1) as f32; n];
    }
    for x in &mut sample {
        *x /= total;
    }
    sample
}

fn gamma(rng: &mut impl Rng, alpha: f32) -> f32 {
    if alpha < 1.0 {
        // Gamma(a) = Gamma(a + 1) * U^(1/a)
        let u: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
        return gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Standard normal via Box-Muller.
        let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
        let u2: f32 = rng.random();
        let x = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f32 = rng.random();
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// A single search driven by its own agent.
pub struct Mcts<A: UtttAgent> {
    agent: A,
    cfg: MctsConfig,
    rng: SmallRng,
    states: Vec<A::State>,
    evals: Vec<Evaluation>,
    /// The last search, kept for inspection.
    tree: Option<SearchTree<A::State>>,
}

impl<A: UtttAgent> Mcts<A> {
    pub fn new(agent: A, cfg: MctsConfig) -> Self {
        let rng = SmallRng::seed_from_u64(cfg.seed);
        Self {
            agent,
            cfg,
            rng,
            states: Vec::new(),
            evals: Vec::new(),
            tree: None,
        }
    }

    pub fn config(&self) -> &MctsConfig {
        &self.cfg
    }

    pub fn agent(&self) -> &A {
        &self.agent
    }

    pub fn agent_mut(&mut self) -> &mut A {
        &mut self.agent
    }

    pub fn into_agent(self) -> A {
        self.agent
    }

    /// Runs a full search from `state` and returns the root statistics.
    pub fn search(&mut self, state: &A::State) -> SearchResult {
        let mut tree = SearchTree::new(
            state.clone(),
            self.cfg.simulations,
            self.cfg.c_puct,
            self.cfg.virtual_loss,
        );
        while !tree.is_done() {
            self.states.clear();
            if tree.claim(self.cfg.batch_size, &mut self.states) > 0 {
                self.evals.clear();
                self.agent.evaluate_batch(&self.states, &mut self.evals);
                tree.resolve(&self.evals);
            }
        }
        let result = tree.result();
        self.tree = Some(tree);
        result
    }

    fn pick_move(&mut self, result: &SearchResult) -> Move {
        result
            .sample_move(self.cfg.temperature, &mut self.rng)
            .expect("search on a non-terminal state has moves")
    }

    #[cfg(test)]
    fn root_children(&self) -> Vec<(Move, f32, u32)> {
        self.tree.as_ref().expect("searched").root_children()
    }

    #[cfg(test)]
    fn assert_tree_consistent(&self) {
        self.tree.as_ref().expect("searched").assert_consistent();
    }
}

impl<A: UtttAgent> UtttAgent for Mcts<A> {
    type State = A::State;

    fn choose_move(&mut self, state: &Self::State) -> Move {
        let result = self.search(state);
        self.pick_move(&result)
    }

    /// One search: the picked move plus the visit distribution.
    fn choose_move_with_policy(&mut self, state: &Self::State) -> (Move, Vec<(Move, f32)>) {
        let result = self.search(state);
        let mv = self.pick_move(&result);
        (mv, result.policy())
    }

    /// Root value plus the visit distribution as priors: a search-improved evaluation.
    fn evaluate(&mut self, state: &Self::State) -> Evaluation {
        let result = self.search(state);
        Evaluation {
            value: result.root_value,
            priors: result.policy(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::RandomAgent;
    use crate::game::test_positions::x_wins_in_one;
    use crate::game::{BitState, Outcome, Player, Position, sq};

    fn cfg(simulations: u32) -> MctsConfig {
        MctsConfig {
            simulations,
            ..Default::default()
        }
    }

    #[test]
    fn finds_win_in_one() {
        let state = x_wins_in_one();
        let mut mcts = Mcts::new(RandomAgent::<BitState>::new(1), cfg(100));
        let result = mcts.search(&state);
        assert_eq!(result.best_move(), Some(Move::new(sq(0, 2), sq(0, 2))));
        assert!(result.root_value > 0.5, "root value {}", result.root_value);
        assert_eq!(mcts.choose_move(&state), Move::new(sq(0, 2), sq(0, 2)));

        // The first simulation expands the root itself; the other 99 each visit one child.
        let total: u32 = result.visits.iter().map(|(_, n)| n).sum();
        assert_eq!(total, 99);
        // The winning child is terminal and gets the lion's share of visits.
        let (_, wins) = result
            .visits
            .iter()
            .find(|(m, _)| *m == Move::new(sq(0, 2), sq(0, 2)))
            .unwrap();
        assert!(*wins > 50);
    }

    #[test]
    fn batched_search_finds_win_and_keeps_tree_consistent() {
        let state = x_wins_in_one();
        for batch_size in [1, 4, 16, 64] {
            let mut mcts = Mcts::new(
                RandomAgent::<BitState>::new(1),
                MctsConfig {
                    simulations: 120,
                    batch_size,
                    ..Default::default()
                },
            );
            let result = mcts.search(&state);
            mcts.assert_tree_consistent();
            let total: u32 = result.visits.iter().map(|(_, n)| n).sum();
            assert_eq!(total, 119, "batch {batch_size}");
            assert_eq!(
                result.best_move(),
                Some(Move::new(sq(0, 2), sq(0, 2))),
                "batch {batch_size}"
            );
            assert!(
                result.root_value > 0.5,
                "batch {batch_size}: {}",
                result.root_value
            );
        }
    }

    /// Records the largest batch handed to `evaluate_batch`.
    struct Counting {
        inner: RandomAgent<BitState>,
        max_batch: usize,
        calls: usize,
    }

    impl UtttAgent for Counting {
        type State = BitState;

        fn choose_move(&mut self, state: &BitState) -> Move {
            self.inner.choose_move(state)
        }

        fn evaluate_batch(&mut self, states: &[BitState], out: &mut Vec<Evaluation>) {
            self.max_batch = self.max_batch.max(states.len());
            self.calls += 1;
            self.inner.evaluate_batch(states, out);
        }
    }

    #[test]
    fn batches_really_form() {
        let agent = Counting {
            inner: RandomAgent::new(1),
            max_batch: 0,
            calls: 0,
        };
        let mut mcts = Mcts::new(
            agent,
            MctsConfig {
                simulations: 200,
                batch_size: 8,
                ..Default::default()
            },
        );
        mcts.search(&BitState::initial());
        mcts.assert_tree_consistent();
        let agent = mcts.agent();
        assert!(
            agent.max_batch >= 4,
            "largest batch was {}",
            agent.max_batch
        );
        assert!(
            agent.calls < 200,
            "{} calls for 200 simulations",
            agent.calls
        );
    }

    #[test]
    fn batch_of_one_matches_sequential_search() {
        // batch_size 1 must be bit-for-bit the old sequential behaviour, which the
        // deterministic seed makes checkable against itself under the same config.
        let state = BitState::initial();
        let run = |seed| {
            let mut m = Mcts::new(RandomAgent::<BitState>::new(seed), cfg(60));
            let r = m.search(&state);
            m.assert_tree_consistent();
            r
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn search_tree_is_resumable_and_bounded_by_budget() {
        let mut agent = RandomAgent::<BitState>::new(4);
        let mut tree = SearchTree::new(x_wins_in_one(), 50, 1.4, 1.0);
        let mut states = Vec::new();
        let mut evals = Vec::new();
        let mut rounds = 0;
        while !tree.is_done() {
            states.clear();
            let claimed = tree.claim(8, &mut states);
            assert!(claimed <= 8);
            assert_eq!(claimed, states.len());
            if claimed > 0 {
                evals.clear();
                agent.evaluate_batch(&states, &mut evals);
                tree.resolve(&evals);
            }
            tree.assert_consistent();
            rounds += 1;
        }
        assert_eq!(tree.completed(), 50);
        assert!(rounds >= 7, "{rounds} rounds");
        assert_eq!(
            tree.result().best_move(),
            Some(Move::new(sq(0, 2), sq(0, 2)))
        );
        assert_eq!(tree.claim(8, &mut states), 0, "done trees claim nothing");
    }

    #[test]
    fn root_noise_perturbs_priors_but_keeps_them_a_distribution() {
        let mut agent = RandomAgent::<BitState>::new(1);
        let mut states = Vec::new();
        let mut evals = Vec::new();
        let noise = RootNoise {
            alpha: 0.3,
            fraction: 0.25,
            seed: 9,
        };
        let mut tree = SearchTree::new(BitState::initial(), 1, 1.4, 1.0).with_root_noise(noise);
        tree.claim(1, &mut states);
        agent.evaluate_batch(&states, &mut evals);
        tree.resolve(&evals);
        let priors: Vec<f32> = tree.root_children().iter().map(|c| c.1).collect();
        assert_eq!(priors.len(), 81);
        assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(
            priors.iter().any(|p| (p - 1.0 / 81.0).abs() > 1e-3),
            "noise applied"
        );
        assert!(
            priors.iter().all(|p| *p >= 0.75 / 81.0 - 1e-6),
            "uniform part kept"
        );
    }

    #[test]
    fn dirichlet_samples_are_distributions() {
        let mut rng = SmallRng::seed_from_u64(3);
        for alpha in [0.1, 0.3, 1.0, 5.0] {
            for n in [1, 2, 9, 81] {
                let d = dirichlet(&mut rng, alpha, n);
                assert_eq!(d.len(), n);
                assert!(
                    (d.iter().sum::<f32>() - 1.0).abs() < 1e-4,
                    "alpha {alpha} n {n}"
                );
                assert!(d.iter().all(|x| *x >= 0.0 && x.is_finite()));
            }
        }
        // Small alpha concentrates mass, large alpha spreads it.
        let peaky = dirichlet(&mut rng, 0.05, 81);
        let flat = dirichlet(&mut rng, 50.0, 81);
        let max = |v: &Vec<f32>| v.iter().cloned().fold(0.0, f32::max);
        assert!(max(&peaky) > max(&flat));
    }

    #[test]
    fn evaluate_as_agent_returns_policy() {
        let state = x_wins_in_one();
        let mut mcts = Mcts::new(RandomAgent::<BitState>::new(1), cfg(50));
        let e = mcts.evaluate(&state);
        let sum: f32 = e.priors.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(e.priors.len(), 7);
        assert!(e.value > 0.5);
    }

    #[test]
    fn terminal_root_is_exact() {
        let mut p = Position::initial();
        for b in [sq(0, 0), sq(1, 1), sq(2, 2)] {
            for s in [sq(0, 0), sq(0, 1), sq(0, 2)] {
                p.set_cell(Move::new(b, s), Some(Player::O));
            }
        }
        let state = BitState::from_position(&p); // X to move, O has won
        assert_eq!(state.outcome(), Some(Outcome::Win(Player::O)));
        let mut mcts = Mcts::new(RandomAgent::<BitState>::new(1), cfg(10));
        let r = mcts.search(&state);
        assert_eq!(r.root_value, -1.0);
        assert!(r.visits.is_empty());
        assert_eq!(r.best_move(), None);
    }

    /// An agent returning deliberately messy priors: an illegal move, a negative weight, and
    /// only some of the legal moves listed.
    struct MessyPriors;

    impl UtttAgent for MessyPriors {
        type State = BitState;

        fn choose_move(&mut self, state: &BitState) -> Move {
            state.legal_moves_vec()[0]
        }

        fn evaluate(&mut self, _state: &BitState) -> Evaluation {
            Evaluation {
                value: 7.0, // out of range; must be clamped
                priors: vec![
                    (Move::new(sq(1, 1), sq(0, 0)), 3.0),
                    (Move::new(sq(1, 1), sq(0, 1)), 1.0),
                    (Move::new(sq(1, 1), sq(0, 2)), -5.0),
                    (Move::new(sq(2, 2), sq(2, 2)), 100.0), // illegal after the first move
                ],
            }
        }
    }

    #[test]
    fn priors_are_masked_and_normalised() {
        let mut state = BitState::initial();
        state.apply(Move::new(sq(0, 0), sq(1, 1))); // O forced to board (1,1)
        let mut mcts = Mcts::new(MessyPriors, cfg(1));
        let r = mcts.search(&state);
        assert_eq!(r.root_value, 1.0, "value clamped to 1");
        let children = mcts.root_children();
        assert_eq!(children.len(), 9);
        let prior_of = |m: Move| children.iter().find(|(c, _, _)| *c == m).unwrap().1;
        assert!((prior_of(Move::new(sq(1, 1), sq(0, 0))) - 0.75).abs() < 1e-6);
        assert!((prior_of(Move::new(sq(1, 1), sq(0, 1))) - 0.25).abs() < 1e-6);
        assert_eq!(prior_of(Move::new(sq(1, 1), sq(0, 2))), 0.0);
        assert_eq!(prior_of(Move::new(sq(1, 1), sq(2, 2))), 0.0);
        let total: f32 = children.iter().map(|c| c.1).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn uniform_priors_when_none_given() {
        let state = BitState::initial();
        let mut mcts = Mcts::new(RandomAgent::<BitState>::new(3), cfg(1));
        mcts.search(&state);
        let children = mcts.root_children();
        assert_eq!(children.len(), 81);
        assert!(children.iter().all(|c| (c.1 - 1.0 / 81.0).abs() < 1e-6));
    }

    fn play_match(games: usize, simulations: u32) -> (usize, usize, usize) {
        let mut mcts = Mcts::new(RandomAgent::<BitState>::new(11), cfg(simulations));
        let mut random = RandomAgent::<BitState>::new(12);
        let (mut wins, mut draws, mut losses) = (0, 0, 0);
        for game in 0..games {
            let mcts_player = if game % 2 == 0 { Player::X } else { Player::O };
            let mut state = BitState::initial();
            let outcome = loop {
                if let Some(o) = state.outcome() {
                    break o;
                }
                let mv = if state.to_move() == mcts_player {
                    mcts.choose_move(&state)
                } else {
                    random.choose_move(&state)
                };
                state.apply(mv);
            };
            match outcome {
                Outcome::Win(p) if p == mcts_player => wins += 1,
                Outcome::Win(_) => losses += 1,
                Outcome::Draw => draws += 1,
            }
        }
        (wins, draws, losses)
    }

    #[test]
    #[ignore = "plays many search games; run with: cargo test --release -- --include-ignored"]
    fn beats_random_opponent() {
        let (wins, draws, losses) = play_match(12, 100);
        assert!(
            wins >= 11,
            "MCTS vs random: {wins} wins, {draws} draws, {losses} losses"
        );
    }

    #[test]
    #[ignore = "slow; run in release"]
    fn beats_random_opponent_long() {
        let (wins, draws, losses) = play_match(100, 400);
        assert!(
            wins >= 95,
            "MCTS vs random: {wins} wins, {draws} draws, {losses} losses"
        );
    }

    #[test]
    fn temperature_sampling_only_returns_searched_moves() {
        let state = x_wins_in_one();
        let legal = state.legal_moves_vec();
        let mut mcts = Mcts::new(
            RandomAgent::<BitState>::new(1),
            MctsConfig {
                simulations: 30,
                temperature: 1.0,
                ..Default::default()
            },
        );
        for _ in 0..20 {
            let mv = mcts.choose_move(&state);
            assert!(legal.contains(&mv));
        }
    }
}
