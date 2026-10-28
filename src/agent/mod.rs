//! The [`UtttAgent`] trait: anything that can pick a move and evaluate a position.

pub mod any;
pub mod greedy;
pub mod heuristic;
pub mod minimax;
pub mod random;

pub use any::{AgentSpec, AnyAgent, SPEC_SYNTAX};
pub use greedy::GreedyAgent;
pub use minimax::MinimaxAgent;
pub use random::RandomAgent;

use crate::game::{GameState, Move};

/// What an agent thinks of a position, from the perspective of the side to move.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Evaluation {
    /// Expected result in `[-1, 1]` for the side to move: `1` = certain win, `-1` = certain loss.
    pub value: f32,
    /// Prior probability per move. Empty means "uniform over legal moves". Moves not listed get
    /// prior 0; listed moves that are illegal are ignored by the search.
    pub priors: Vec<(Move, f32)>,
}

impl Evaluation {
    pub fn value_only(value: f32) -> Self {
        Self {
            value,
            priors: Vec::new(),
        }
    }
}

/// Evaluations of a batch of positions, laid out for the search rather than for reading: one
/// value per position and, once any position has priors, one dense row of [`Move::COUNT`]
/// floats per position indexed by [`Move::index`]. The search only reads the entries of legal
/// moves, so a row only needs those filled; a position without priors in a batch that has
/// them gets an all-zero row, which the search treats as uniform.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evaluations {
    values: Vec<f32>,
    /// Empty while no position has priors, otherwise `values.len() * Move::COUNT`.
    priors: Vec<f32>,
}

impl Evaluations {
    pub fn clear(&mut self) {
        self.values.clear();
        self.priors.clear();
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn has_priors(&self) -> bool {
        !self.priors.is_empty()
    }

    /// Appends a position with no priors.
    pub fn push_value(&mut self, value: f32) {
        self.values.push(value);
        if self.has_priors() {
            self.priors.resize(self.values.len() * Move::COUNT, 0.0);
        }
    }

    /// Appends a position and returns its zeroed prior row to fill in.
    pub fn push_row(&mut self, value: f32) -> &mut [f32; Move::COUNT] {
        self.values.push(value);
        self.priors.resize(self.values.len() * Move::COUNT, 0.0);
        let start = (self.values.len() - 1) * Move::COUNT;
        (&mut self.priors[start..start + Move::COUNT])
            .try_into()
            .expect("row of Move::COUNT entries")
    }

    /// Appends a single evaluation, scattering its priors into a row.
    pub fn push(&mut self, eval: &Evaluation) {
        if eval.priors.is_empty() {
            self.push_value(eval.value);
        } else {
            let row = self.push_row(eval.value);
            for (mv, p) in &eval.priors {
                row[mv.index()] += p;
            }
        }
    }

    pub fn value(&self, i: usize) -> f32 {
        self.values[i]
    }

    /// Prior row of position `i`, or `None` if the batch carries no priors at all.
    pub fn priors(&self, i: usize) -> Option<&[f32; Move::COUNT]> {
        if !self.has_priors() {
            return None;
        }
        let start = i * Move::COUNT;
        Some(
            (&self.priors[start..start + Move::COUNT])
                .try_into()
                .expect("row of Move::COUNT entries"),
        )
    }
}

/// An Ultimate Tic Tac Toe agent.
///
/// The state representation is the agent's choice ([`UtttAgent::State`]); callers convert
/// through [`crate::game::Position`] when they need to talk to another agent.
///
/// Methods take `&mut self` so agents can own RNGs, caches or network handles directly.
pub trait UtttAgent {
    type State: GameState;

    /// Picks a move. `state` is never terminal, so at least one legal move exists.
    fn choose_move(&mut self, state: &Self::State) -> Move;

    /// Picks a move and also reports the move distribution the agent considered, as a
    /// training target. The default is the chosen move with probability 1; search agents
    /// return their visit distribution.
    fn choose_move_with_policy(&mut self, state: &Self::State) -> (Move, Vec<(Move, f32)>) {
        let mv = self.choose_move(state);
        (mv, vec![(mv, 1.0)])
    }

    /// Evaluates a position for tree search.
    ///
    /// The default plays the game out with [`UtttAgent::choose_move`] (until it is settled:
    /// see [`GameState::play_settled`]) and reports the result with no priors. Terminal states
    /// are evaluated exactly.
    fn evaluate(&mut self, state: &Self::State) -> Evaluation {
        let me = state.to_move();
        if let Some(outcome) = state.outcome() {
            return Evaluation::value_only(outcome.value_for(me));
        }
        let mut s = state.clone();
        loop {
            let mv = self.choose_move(&s);
            if let Some(outcome) = s.play_settled(mv.into()) {
                return Evaluation::value_only(outcome.value_for(me));
            }
        }
    }

    /// Evaluates many positions at once, appending to `out`. The default calls
    /// [`UtttAgent::evaluate`] for each; agents backed by a network override it to batch.
    fn evaluate_batch(&mut self, states: &[Self::State], out: &mut Evaluations) {
        for s in states {
            let e = self.evaluate(s);
            out.push(&e);
        }
    }

    /// Starts evaluating `states` without waiting for the result, so the caller can do other
    /// work until [`UtttAgent::finish_batch`], which must follow with the same `states`
    /// before any other evaluation. Agents backed by a device queue the forward pass here and
    /// read it back in `finish_batch`; the default does nothing and lets `finish_batch`
    /// evaluate synchronously.
    fn begin_batch(&mut self, states: &[Self::State]) {
        let _ = states;
    }

    /// Completes the evaluation started by [`UtttAgent::begin_batch`] for the same `states`,
    /// appending to `out` like [`UtttAgent::evaluate_batch`].
    fn finish_batch(&mut self, states: &[Self::State], out: &mut Evaluations) {
        self.evaluate_batch(states, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{BitState, Outcome, Player, Position, sq};

    #[test]
    fn default_evaluate_is_exact_on_terminal_states() {
        let mut p = Position::initial();
        for b in [sq(0, 0), sq(0, 1), sq(0, 2)] {
            for s in [sq(2, 0), sq(2, 1), sq(2, 2)] {
                p.set_cell(Move::new(b, s), Some(Player::X));
            }
        }
        assert_eq!(p.outcome(), Some(Outcome::Win(Player::X)));
        let mut agent = RandomAgent::<BitState>::new(0);
        // O "to move" in a position X already won.
        p.set_to_move(Player::O);
        assert_eq!(agent.evaluate(&BitState::from_position(&p)).value, -1.0);
        p.set_to_move(Player::X);
        assert_eq!(agent.evaluate(&BitState::from_position(&p)).value, 1.0);
    }

    #[test]
    fn default_evaluate_rolls_out_to_a_result() {
        let mut agent = RandomAgent::<BitState>::new(0);
        let e = agent.evaluate(&BitState::initial());
        assert!(e.priors.is_empty());
        assert!([-1.0, 0.0, 1.0].contains(&e.value));
        let mut out = Evaluations::default();
        agent.evaluate_batch(&[BitState::initial(); 3], &mut out);
        assert_eq!(out.len(), 3);
        assert!(!out.has_priors());
        assert!(out.priors(0).is_none());
    }

    #[test]
    fn evaluations_mix_rows_and_values() {
        let mut out = Evaluations::default();
        out.push_value(0.5);
        let row = out.push_row(-0.25);
        row[Move::from_index(3).index()] = 1.0;
        out.push(&Evaluation {
            value: 0.0,
            priors: vec![(Move::from_index(80), 0.5), (Move::from_index(80), 0.5)],
        });
        assert_eq!(out.len(), 3);
        assert_eq!(
            (out.value(0), out.value(1), out.value(2)),
            (0.5, -0.25, 0.0)
        );
        assert!(out.priors(0).unwrap().iter().all(|&p| p == 0.0));
        assert_eq!(out.priors(1).unwrap()[3], 1.0);
        assert_eq!(out.priors(2).unwrap()[80], 1.0);
        out.clear();
        assert!(out.is_empty() && !out.has_priors());
    }
}
