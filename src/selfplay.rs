//! Lockstep self-play: many games are searched at once and, every round, the leaves claimed
//! by all of their trees go to the evaluator in a single batch.
//!
//! This is how a network evaluator gets large batches without pushing the per-tree batch (and
//! its virtual-loss distortion) up: `concurrent` games times `tree_batch` leaves per round.
//! Exploration follows AlphaZero: Dirichlet noise on the root priors and moves sampled in
//! proportion to visit counts for the first plies of each game.

use std::time::Instant;

use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::agent::{Evaluations, UtttAgent};
use crate::game::{BitState, GameState, Move, Outcome};
use crate::mcts::{RootNoise, SearchTree};

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Simulations per move.
    pub simulations: u32,
    /// Leaves each tree claims per round (virtual loss applies within a tree).
    pub tree_batch: usize,
    /// Games in flight; the evaluator sees up to `concurrent * tree_batch` leaves per call.
    pub concurrent: usize,
    pub c_puct: f32,
    pub virtual_loss: f32,
    /// For the first `temperature_moves` plies of a game, moves are sampled in proportion to
    /// `visits^(1/temperature)`; afterwards the most visited move is played.
    pub temperature_moves: usize,
    pub temperature: f32,
    /// Dirichlet noise on the root priors, `None` to disable.
    pub root_noise: Option<(f32, f32)>,
    pub seed: u64,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            simulations: 200,
            tree_batch: 8,
            concurrent: 64,
            c_puct: 1.4,
            virtual_loss: 1.0,
            temperature_moves: 16,
            temperature: 1.0,
            root_noise: Some((0.3, 0.25)),
            seed: 0,
        }
    }
}

/// One finished game: every position with the search's move distribution from it, and how
/// the game ended.
#[derive(Clone, Debug, PartialEq)]
pub struct GameRecord {
    pub positions: Vec<(BitState, Vec<(Move, f32)>)>,
    pub outcome: Outcome,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelfPlayStats {
    pub games: usize,
    pub positions: usize,
    pub simulations: u64,
    /// Calls to the evaluator.
    pub evaluator_calls: u64,
    /// Leaves evaluated over all calls.
    pub leaves: u64,
    pub seconds: f32,
}

impl SelfPlayStats {
    pub fn simulations_per_second(&self) -> f32 {
        self.simulations as f32 / self.seconds.max(1e-9)
    }

    pub fn mean_batch(&self) -> f32 {
        self.leaves as f32 / self.evaluator_calls.max(1) as f32
    }
}

struct Active {
    state: BitState,
    positions: Vec<(BitState, Vec<(Move, f32)>)>,
    tree: SearchTree<BitState>,
    plies: usize,
}

/// Plays `games` games of search-with-`evaluator` against itself and returns their records.
/// `on_game` is called after each finished game with the number finished so far.
pub fn self_play<A: UtttAgent<State = BitState>>(
    evaluator: &mut A,
    games: usize,
    cfg: &SelfPlayConfig,
    mut on_game: impl FnMut(usize),
) -> (Vec<GameRecord>, SelfPlayStats) {
    let start = Instant::now();
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let mut stats = SelfPlayStats::default();
    let mut records = Vec::with_capacity(games);

    let new_tree = |state: BitState, rng: &mut SmallRng| {
        let tree = SearchTree::new(state, cfg.simulations, cfg.c_puct, cfg.virtual_loss);
        match cfg.root_noise {
            Some((alpha, fraction)) => tree.with_root_noise(RootNoise {
                alpha,
                fraction,
                seed: rng.random(),
            }),
            None => tree,
        }
    };
    let start_game = |rng: &mut SmallRng| Active {
        state: BitState::initial(),
        positions: Vec::new(),
        tree: new_tree(BitState::initial(), rng),
        plies: 0,
    };

    let mut active: Vec<Active> = (0..cfg.concurrent.max(1).min(games))
        .map(|_| start_game(&mut rng))
        .collect();
    let mut started = active.len();
    let mut states: Vec<BitState> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut evals = Evaluations::default();

    while !active.is_empty() {
        // One round: every tree claims its leaves, one batched evaluation, every tree resolves.
        states.clear();
        spans.clear();
        for game in &mut active {
            let from = states.len();
            game.tree.claim(cfg.tree_batch, &mut states);
            spans.push((from, states.len()));
        }
        if !states.is_empty() {
            evals.clear();
            evaluator.evaluate_batch(&states, &mut evals);
            debug_assert_eq!(evals.len(), states.len());
            stats.evaluator_calls += 1;
            stats.leaves += states.len() as u64;
            for (game, &(from, to)) in active.iter_mut().zip(&spans) {
                if to > from {
                    game.tree.resolve(&evals, from);
                }
            }
        }

        // Trees that have spent their budget play a move; finished games are replaced.
        for i in (0..active.len()).rev() {
            if !active[i].tree.is_done() {
                continue;
            }
            let game = &mut active[i];
            let result = game.tree.result();
            stats.simulations += game.tree.completed() as u64;
            let temperature = if game.plies < cfg.temperature_moves {
                cfg.temperature
            } else {
                0.0
            };
            let mv = result
                .sample_move(temperature, &mut rng)
                .expect("non-terminal position has moves");
            game.positions.push((game.state, result.policy()));
            game.state.apply(mv);
            game.plies += 1;
            match game.state.outcome() {
                Some(outcome) => {
                    stats.games += 1;
                    stats.positions += game.positions.len();
                    records.push(GameRecord {
                        positions: std::mem::take(&mut game.positions),
                        outcome,
                    });
                    on_game(stats.games);
                    if started < games {
                        active[i] = start_game(&mut rng);
                        started += 1;
                    } else {
                        active.swap_remove(i);
                    }
                }
                None => {
                    let state = game.state;
                    game.tree = new_tree(state, &mut rng);
                }
            }
        }
    }
    stats.seconds = start.elapsed().as_secs_f32();
    (records, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::RandomAgent;
    use crate::game::MoveSet;

    fn small(concurrent: usize, seed: u64) -> SelfPlayConfig {
        SelfPlayConfig {
            simulations: 24,
            tree_batch: 4,
            concurrent,
            temperature_moves: 6,
            seed,
            ..Default::default()
        }
    }

    #[test]
    fn records_are_complete_consistent_games() {
        let mut evaluator = RandomAgent::<BitState>::new(1);
        let mut seen = 0;
        let (records, stats) = self_play(&mut evaluator, 5, &small(3, 7), |n| seen = n);
        assert_eq!(records.len(), 5);
        assert_eq!(seen, 5);
        assert_eq!(stats.games, 5);
        assert_eq!(
            stats.positions,
            records.iter().map(|r| r.positions.len()).sum::<usize>()
        );
        assert_eq!(stats.simulations, 24 * stats.positions as u64);
        for record in &records {
            assert!(record.positions.len() >= 17);
            // Each position follows the previous by a legal move, and ends in the outcome.
            let mut last: Option<BitState> = None;
            for (state, policy) in &record.positions {
                if let Some(prev) = last {
                    assert!(prev.legal_moves().iter().any(|mv| {
                        let mut next = prev;
                        next.apply(mv);
                        next == *state
                    }));
                }
                let sum: f32 = policy.iter().map(|(_, p)| p).sum();
                assert!((sum - 1.0).abs() < 1e-4);
                assert!(policy.iter().all(|(mv, _)| state.is_legal(*mv)));
                last = Some(*state);
            }
            let (final_state, policy) = record.positions.last().unwrap();
            let mut end = *final_state;
            let played = policy
                .iter()
                .find(|(mv, _)| {
                    let mut next = end;
                    next.apply(*mv);
                    next.outcome() == Some(record.outcome)
                })
                .map(|(mv, _)| *mv);
            assert!(
                played.is_some(),
                "the last move ends the game with the recorded outcome"
            );
            end.apply(played.unwrap());
            assert_eq!(end.outcome(), Some(record.outcome));
        }
    }

    #[test]
    fn concurrent_games_share_evaluator_calls() {
        let mut solo = RandomAgent::<BitState>::new(2);
        let (_, one) = self_play(&mut solo, 4, &small(1, 3), |_| {});
        let mut shared = RandomAgent::<BitState>::new(2);
        let (_, many) = self_play(&mut shared, 4, &small(4, 3), |_| {});
        assert!(one.mean_batch() <= 4.0 + 1e-6);
        assert!(many.mean_batch() > one.mean_batch(), "{many:?} vs {one:?}");
        assert!(many.evaluator_calls < one.evaluator_calls);
    }

    #[test]
    fn deterministic_for_a_seed() {
        let run = |seed| {
            let mut e = RandomAgent::<BitState>::new(5);
            self_play(&mut e, 3, &small(2, seed), |_| {}).0
        };
        assert_eq!(run(11), run(11));
        assert_ne!(run(11), run(12));
    }
}
