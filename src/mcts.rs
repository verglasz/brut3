//! [`Mcts`]: PUCT Monte Carlo Tree Search, generic over the agent used to evaluate leaves.
//!
//! With an agent whose [`UtttAgent::evaluate`] does random playouts and returns no priors this
//! is classic rollout MCTS; with a network agent returning priors and a value it is the
//! AlphaZero search. One code path covers both.

use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::agent::{Evaluation, UtttAgent};
use crate::game::{GameState, Move, Outcome};

#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Simulations per search (one leaf expansion each).
    pub simulations: u32,
    /// Exploration constant in the PUCT formula.
    pub c_puct: f32,
    /// Move selection: `0` picks the most visited move; `> 0` samples proportionally to
    /// `visits^(1/temperature)` (used for self-play diversity).
    pub temperature: f32,
    pub seed: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            simulations: 800,
            c_puct: 1.4,
            temperature: 0.0,
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
    /// Children are contiguous: `first_child..first_child + n_children`.
    first_child: u32,
    n_children: u8,
    expanded: bool,
    visits: u32,
    /// Sum of backed-up values, from the perspective of the side to move *at this node*.
    value_sum: f32,
    prior: f32,
    outcome: Option<Outcome>,
}

impl<S: GameState> Node<S> {
    fn new(state: S, mv: Option<Move>, prior: f32) -> Self {
        let outcome = state.outcome();
        Self {
            state,
            mv,
            first_child: NO_NODE,
            n_children: 0,
            expanded: false,
            visits: 0,
            value_sum: 0.0,
            prior,
            outcome,
        }
    }

    fn children(&self) -> std::ops::Range<u32> {
        self.first_child..self.first_child + self.n_children as u32
    }

    /// Mean value from the perspective of the *parent's* side to move.
    fn q_for_parent(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            -self.value_sum / self.visits as f32
        }
    }
}

pub struct Mcts<A: UtttAgent> {
    agent: A,
    cfg: MctsConfig,
    nodes: Vec<Node<A::State>>,
    rng: SmallRng,
    move_buf: Vec<Move>,
    path: Vec<u32>,
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
            move_buf: Vec::new(),
            path: Vec::new(),
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
        self.nodes.push(Node::new(state.clone(), None, 1.0));
        if let Some(outcome) = self.nodes[0].outcome {
            return SearchResult {
                root_value: outcome.value_for(state.to_move()),
                visits: Vec::new(),
            };
        }
        for _ in 0..self.cfg.simulations {
            self.simulate();
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

    fn simulate(&mut self) {
        self.path.clear();
        let mut idx = 0u32;
        self.path.push(idx);
        let leaf_value = loop {
            let node = &self.nodes[idx as usize];
            if let Some(outcome) = node.outcome {
                break outcome.value_for(node.state.to_move());
            }
            if !node.expanded {
                break self.expand(idx);
            }
            idx = self.select_child(idx);
            self.path.push(idx);
        };
        let mut value = leaf_value;
        for &n in self.path.iter().rev() {
            let node = &mut self.nodes[n as usize];
            node.visits += 1;
            node.value_sum += value;
            value = -value;
        }
    }

    /// Evaluates the node, creates its children, returns the value for the node's side to move.
    fn expand(&mut self, idx: u32) -> f32 {
        let state = self.nodes[idx as usize].state.clone();
        let mut moves = std::mem::take(&mut self.move_buf);
        state.legal_moves(&mut moves);
        debug_assert!(!moves.is_empty(), "non-terminal state without legal moves");
        let Evaluation { value, priors } = self.agent.evaluate(&state);

        // Keep only legal moves' priors; fall back to uniform if nothing usable was given.
        let mut total = 0.0f32;
        if !priors.is_empty() {
            self.prior_buf.fill(0.0);
            for (mv, p) in &priors {
                self.prior_buf[mv.index()] += p.max(0.0);
            }
            total = moves.iter().map(|m| self.prior_buf[m.index()]).sum();
        }
        let uniform = total <= 1e-12;
        let n = moves.len();

        let first_child = self.nodes.len() as u32;
        for &mv in &moves {
            let prior = if uniform {
                1.0 / n as f32
            } else {
                self.prior_buf[mv.index()] / total
            };
            let mut child_state = state.clone();
            child_state.apply(mv);
            self.nodes.push(Node::new(child_state, Some(mv), prior));
        }
        self.move_buf = moves;

        let node = &mut self.nodes[idx as usize];
        node.first_child = first_child;
        node.n_children = n as u8;
        node.expanded = true;
        value.clamp(-1.0, 1.0)
    }

    fn select_child(&self, idx: u32) -> u32 {
        let node = &self.nodes[idx as usize];
        let sqrt_n = (node.visits as f32).sqrt();
        let mut best = node.first_child;
        let mut best_score = f32::NEG_INFINITY;
        for c in node.children() {
            let child = &self.nodes[c as usize];
            let exploration = self.cfg.c_puct * child.prior * sqrt_n / (1.0 + child.visits as f32);
            let score = child.q_for_parent() + exploration;
            if score > best_score {
                best_score = score;
                best = c;
            }
        }
        best
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
    use crate::agent::test_agents::RandomAgent;
    use crate::game::{BitState, Outcome, Player, Position, sq};

    fn cfg(simulations: u32) -> MctsConfig {
        MctsConfig {
            simulations,
            ..Default::default()
        }
    }

    /// X has won boards (0,0) and (0,1), has two in a row on (0,2), and is forced to play there.
    fn x_wins_in_one() -> BitState {
        let mut p = Position::initial();
        for b in [sq(0, 0), sq(0, 1)] {
            for s in [sq(1, 0), sq(1, 1), sq(1, 2)] {
                p.set_cell(Move::new(b, s), Some(Player::X));
            }
        }
        p.set_cell(Move::new(sq(0, 2), sq(0, 0)), Some(Player::X));
        p.set_cell(Move::new(sq(0, 2), sq(0, 1)), Some(Player::X));
        // Some O marks so O's last move (into square (0,2) of board (1,1)) is plausible.
        for s in [sq(2, 0), sq(2, 1), sq(0, 2)] {
            p.set_cell(Move::new(sq(1, 1), s), Some(Player::O));
        }
        p.set_last_move(Some(Move::new(sq(1, 1), sq(0, 2))));
        p.set_to_move(Player::X);
        let state = BitState::from_position(&p);
        assert_eq!(state.outcome(), None);
        assert_eq!(state.legal_moves_vec().len(), 7);
        state
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
