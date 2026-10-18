//! [`Mcts`]: PUCT Monte Carlo Tree Search, generic over the agent used to evaluate leaves.
//!
//! With an agent whose [`UtttAgent::evaluate`] does random playouts and returns no priors this
//! is classic rollout MCTS; with a network agent returning priors and a value it is the
//! AlphaZero search. One code path covers both.
//!
//! Leaves are evaluated in batches: up to `batch_size` paths are selected per round, each
//! claimed leaf carrying a virtual loss so the next selection goes elsewhere, and the agent's
//! [`UtttAgent::evaluate_batch`] sees them all at once. With `batch_size = 1` this is plain
//! sequential MCTS.

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
    /// Claimed by the current batch, awaiting evaluation.
    pending: bool,
    visits: u32,
    /// Sum of backed-up values, from the perspective of the side to move *at this node*.
    value_sum: f32,
    /// Virtual losses currently applied (paths through here claimed in the current batch).
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

pub struct Mcts<A: UtttAgent> {
    agent: A,
    cfg: MctsConfig,
    nodes: Vec<Node<A::State>>,
    rng: SmallRng,
    leaves: Vec<u32>,
    states: Vec<A::State>,
    evals: Vec<Evaluation>,
    prior_buf: [f32; 81],
}

impl<A: UtttAgent> Mcts<A> {
    pub fn new(agent: A, cfg: MctsConfig) -> Self {
        let rng = SmallRng::seed_from_u64(cfg.seed);
        Self {
            agent,
            cfg,
            nodes: Vec::new(),
            rng,
            leaves: Vec::new(),
            states: Vec::new(),
            evals: Vec::new(),
            prior_buf: [0.0; 81],
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
        self.nodes.clear();
        self.nodes
            .push(Node::new(state.clone(), None, NO_NODE, 1.0));
        if let Some(outcome) = self.nodes[0].outcome {
            return SearchResult {
                root_value: outcome.value_for(state.to_move()),
                visits: Vec::new(),
            };
        }
        let mut done = 0u32;
        while done < self.cfg.simulations {
            done += self.run_batch(self.cfg.simulations - done);
        }
        let root = &self.nodes[0];
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

    /// Selects up to `min(batch_size, budget)` leaves, evaluates them together, backs up.
    /// Returns how many simulations were completed (terminal hits included).
    fn run_batch(&mut self, budget: u32) -> u32 {
        let want = self.cfg.batch_size.max(1).min(budget as usize);
        let mut completed = 0u32;
        self.leaves.clear();
        for _ in 0..want {
            match self.select_leaf() {
                Selection::Terminal(idx, value) => {
                    self.backup(idx, value);
                    completed += 1;
                }
                Selection::Leaf(idx) => {
                    self.nodes[idx as usize].pending = true;
                    self.add_virtual_loss(idx);
                    self.leaves.push(idx);
                }
                // Every path leads to an already-claimed leaf: evaluate what we have.
                Selection::Blocked => break,
            }
        }
        if self.leaves.is_empty() {
            return completed;
        }

        self.states.clear();
        self.states.extend(
            self.leaves
                .iter()
                .map(|&i| self.nodes[i as usize].state.clone()),
        );
        self.evals.clear();
        self.agent.evaluate_batch(&self.states, &mut self.evals);
        debug_assert_eq!(self.evals.len(), self.leaves.len());

        let leaves = std::mem::take(&mut self.leaves);
        let evals = std::mem::take(&mut self.evals);
        for (&idx, eval) in leaves.iter().zip(&evals) {
            self.remove_virtual_loss(idx);
            self.nodes[idx as usize].pending = false;
            let value = self.expand(idx, eval);
            self.backup(idx, value);
            completed += 1;
        }
        self.leaves = leaves;
        self.evals = evals;
        completed
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
                self.cfg.c_puct * child.prior * sqrt_n / (1.0 + child.effective_visits() as f32);
            let score = child.q_for_parent(self.cfg.virtual_loss) + exploration;
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
                self.prior_buf[mv.index()] += p.max(0.0);
            }
            total = moves.iter().map(|m| self.prior_buf[m.index()]).sum();
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
        eval.value.clamp(-1.0, 1.0)
    }

    fn pick_move(&mut self, result: &SearchResult) -> Move {
        if self.cfg.temperature <= 0.0 {
            return result
                .best_move()
                .expect("search on a non-terminal state has moves");
        }
        let inv_t = 1.0 / self.cfg.temperature;
        let weights: Vec<f32> = result
            .visits
            .iter()
            .map(|(_, n)| (*n as f32).powf(inv_t))
            .collect();
        let total: f32 = weights.iter().sum();
        let mut r = self.rng.random::<f32>() * total;
        for ((mv, _), w) in result.visits.iter().zip(&weights) {
            r -= w;
            if r <= 0.0 {
                return *mv;
            }
        }
        result.visits.last().expect("non-empty").0
    }

    #[cfg(test)]
    fn root_children(&self) -> Vec<(Move, f32, u32)> {
        self.nodes[0]
            .children()
            .map(|c| {
                let n = &self.nodes[c as usize];
                (n.mv.unwrap(), n.prior, n.visits)
            })
            .collect()
    }

    /// Checks the tree after a search: no virtual losses or pending leaves remain, and every
    /// expanded node's visits equal its own evaluation plus its children's visits.
    #[cfg(test)]
    fn assert_tree_consistent(&self) {
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

enum Selection {
    /// An unexpanded, unclaimed leaf.
    Leaf(u32),
    /// A terminal node and its value for the side to move there.
    Terminal(u32, f32),
    /// Selection ran into a leaf already claimed in this batch.
    Blocked,
}

impl<A: UtttAgent> UtttAgent for Mcts<A> {
    type State = A::State;

    fn choose_move(&mut self, state: &Self::State) -> Move {
        let result = self.search(state);
        self.pick_move(&result)
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
