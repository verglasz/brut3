//! Lockstep self-play: many games are searched at once and, every round, the leaves claimed
//! by all of their trees go to the evaluator in a single batch.
//!
//! This is how a network evaluator gets large batches without pushing the per-tree batch (and
//! its virtual-loss distortion) up: `concurrent / 2` games times `tree_batch` leaves per
//! round. The games in flight form two groups that take turns, so one group's tree work on the
//! CPU overlaps the other group's evaluation on the device. Exploration follows AlphaZero:
//! Dirichlet noise on the root priors and moves sampled in proportion to visit counts for the
//! first plies of each game.

use std::time::Instant;

use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::agent::{Evaluations, UtttAgent};
use crate::book::Book;
use crate::game::{BitState, GameState, Move, Outcome, Player};
use crate::mcts::{MctsConfig, RootNoise, SearchTree};

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Simulations per move.
    pub simulations: u32,
    /// Leaves each tree claims per round (virtual loss applies within a tree).
    pub tree_batch: usize,
    /// Games in flight, in two alternating groups; the evaluator sees up to
    /// `concurrent / 2 * tree_batch` leaves per call.
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
            concurrent: 256,
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
    /// Each position with the search's move distribution and its root value for the side to
    /// move (the mean of the backed-up results).
    pub positions: Vec<(BitState, Vec<(Move, f32)>, f32)>,
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
    positions: Vec<(BitState, Vec<(Move, f32)>, f32)>,
    tree: SearchTree<BitState>,
    plies: usize,
}

fn new_tree(cfg: &SelfPlayConfig, state: BitState, rng: &mut SmallRng) -> SearchTree<BitState> {
    let tree = SearchTree::new(state, cfg.simulations, cfg.c_puct, cfg.virtual_loss);
    match cfg.root_noise {
        Some((alpha, fraction)) => tree.with_root_noise(RootNoise {
            alpha,
            fraction,
            seed: rng.random(),
        }),
        None => tree,
    }
}

fn start_game(cfg: &SelfPlayConfig, rng: &mut SmallRng) -> Active {
    Active {
        state: BitState::initial(),
        positions: Vec::new(),
        tree: new_tree(cfg, BitState::initial(), rng),
        plies: 0,
    }
}

/// Half of the games in flight. The two groups take turns, so that while the evaluator works
/// on one group's leaves the other group's trees select their next leaves and back up their
/// last results: with a device-backed evaluator the CPU and the device overlap instead of
/// waiting on each other.
#[derive(Default)]
struct Group {
    active: Vec<Active>,
    /// Leaves claimed this round, and each game's span in them.
    states: Vec<BitState>,
    spans: Vec<(usize, usize)>,
    evals: Evaluations,
    /// `states` has been handed to `begin_batch` and awaits `finish_batch`.
    in_flight: bool,
}

impl Group {
    /// Every tree claims its leaves for this round.
    fn claim(&mut self, cfg: &SelfPlayConfig) {
        self.states.clear();
        self.spans.clear();
        for game in &mut self.active {
            let from = self.states.len();
            game.tree.claim(cfg.tree_batch, &mut self.states);
            self.spans.push((from, self.states.len()));
        }
    }

    /// Hands the claimed leaves to the evaluator without waiting for the result.
    fn begin<A: UtttAgent<State = BitState>>(
        &mut self,
        evaluator: &mut A,
        stats: &mut SelfPlayStats,
    ) {
        if self.states.is_empty() {
            return;
        }
        evaluator.begin_batch(&self.states);
        self.in_flight = true;
        stats.evaluator_calls += 1;
        stats.leaves += self.states.len() as u64;
    }

    /// Collects the evaluations and resolves every tree's leaves.
    fn finish<A: UtttAgent<State = BitState>>(&mut self, evaluator: &mut A) {
        if !self.in_flight {
            return;
        }
        self.in_flight = false;
        self.evals.clear();
        evaluator.finish_batch(&self.states, &mut self.evals);
        debug_assert_eq!(self.evals.len(), self.states.len());
        for (game, &(from, to)) in self.active.iter_mut().zip(&self.spans) {
            if to > from {
                game.tree.resolve(&self.evals, from);
            }
        }
    }
}

/// Trees that have spent their budget play a move; finished games are recorded and replaced
/// while there are games left to start.
#[allow(clippy::too_many_arguments)]
fn advance(
    group: &mut Group,
    cfg: &SelfPlayConfig,
    games: usize,
    started: &mut usize,
    rng: &mut SmallRng,
    stats: &mut SelfPlayStats,
    records: &mut Vec<GameRecord>,
    on_game: &mut impl FnMut(usize),
) {
    for i in (0..group.active.len()).rev() {
        if !group.active[i].tree.is_done() {
            continue;
        }
        let game = &mut group.active[i];
        let result = game.tree.result();
        stats.simulations += game.tree.completed() as u64;
        let temperature = if game.plies < cfg.temperature_moves {
            cfg.temperature
        } else {
            0.0
        };
        let mv = result
            .sample_move(temperature, rng)
            .expect("non-terminal position has moves");
        game.positions
            .push((game.state, result.policy(), result.root_value));
        game.state.apply(mv.into());
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
                if *started < games {
                    group.active[i] = start_game(cfg, rng);
                    *started += 1;
                } else {
                    group.active.swap_remove(i);
                }
            }
            None => {
                let state = game.state;
                game.tree = new_tree(cfg, state, rng);
            }
        }
    }
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

    let in_flight = cfg.concurrent.max(1).min(games);
    let mut groups = [Group::default(), Group::default()];
    for i in 0..in_flight {
        groups[i % 2].active.push(start_game(cfg, &mut rng));
    }
    let mut started = in_flight;

    // Interleaving: while `current`'s evaluation is in flight, `other` claims; once `current`
    // is finished, `other` goes in flight and `current` resolves and plays its moves.
    let mut current = 0;
    groups[current].claim(cfg);
    groups[current].begin(evaluator, &mut stats);
    while !(groups[0].active.is_empty() && groups[1].active.is_empty()) {
        let other = 1 - current;
        groups[other].claim(cfg);
        groups[current].finish(evaluator);
        groups[other].begin(evaluator, &mut stats);
        advance(
            &mut groups[current],
            cfg,
            games,
            &mut started,
            &mut rng,
            &mut stats,
            &mut records,
            &mut on_game,
        );
        current = other;
    }
    stats.seconds = start.elapsed().as_secs_f32();
    (records, stats)
}

/// Searches every position of `states` with `search`'s settings over `evaluator`, up to
/// `concurrent` at once in two alternating groups as in [`self_play`], and returns their
/// root results in order. `on_done` is called after each finished search with the count so
/// far.
pub fn search_positions<A: UtttAgent<State = BitState>>(
    evaluator: &mut A,
    states: &[BitState],
    search: &MctsConfig,
    concurrent: usize,
    mut on_done: impl FnMut(usize),
) -> Vec<crate::mcts::SearchResult> {
    struct Pending {
        index: usize,
        tree: SearchTree<BitState>,
    }
    #[derive(Default)]
    struct Trees {
        active: Vec<Pending>,
        states: Vec<BitState>,
        spans: Vec<(usize, usize)>,
        evals: Evaluations,
        in_flight: bool,
    }
    impl Trees {
        fn claim(&mut self, batch: usize) {
            self.states.clear();
            self.spans.clear();
            for p in &mut self.active {
                let from = self.states.len();
                p.tree.claim(batch, &mut self.states);
                self.spans.push((from, self.states.len()));
            }
        }
        fn begin<A: UtttAgent<State = BitState>>(&mut self, evaluator: &mut A) {
            if !self.states.is_empty() {
                evaluator.begin_batch(&self.states);
                self.in_flight = true;
            }
        }
        fn finish<A: UtttAgent<State = BitState>>(&mut self, evaluator: &mut A) {
            if !self.in_flight {
                return;
            }
            self.in_flight = false;
            self.evals.clear();
            evaluator.finish_batch(&self.states, &mut self.evals);
            for (p, &(from, to)) in self.active.iter_mut().zip(&self.spans) {
                if to > from {
                    p.tree.resolve(&self.evals, from);
                }
            }
        }
    }

    let new_tree = |index: usize| Pending {
        index,
        tree: SearchTree::new(
            states[index],
            search.simulations,
            search.c_puct,
            search.virtual_loss,
        ),
    };
    let mut results: Vec<Option<crate::mcts::SearchResult>> = vec![None; states.len()];
    let mut done = 0;
    let in_flight = concurrent.max(1).min(states.len());
    let mut groups = [Trees::default(), Trees::default()];
    for i in 0..in_flight {
        groups[i % 2].active.push(new_tree(i));
    }
    let mut started = in_flight;
    let mut current = 0;
    groups[current].claim(search.batch_size);
    groups[current].begin(evaluator);
    while !(groups[0].active.is_empty() && groups[1].active.is_empty()) {
        let other = 1 - current;
        groups[other].claim(search.batch_size);
        groups[current].finish(evaluator);
        groups[other].begin(evaluator);
        let group = &mut groups[current];
        for i in (0..group.active.len()).rev() {
            if !group.active[i].tree.is_done() {
                continue;
            }
            let p = &group.active[i];
            results[p.index] = Some(p.tree.result());
            done += 1;
            on_done(done);
            if started < states.len() {
                group.active[i] = new_tree(started);
                started += 1;
            } else {
                group.active.swap_remove(i);
            }
        }
        current = other;
    }
    results
        .into_iter()
        .map(|r| r.expect("every position was searched"))
        .collect()
}

/// One side of a [`lockstep_match`]: a search with `search`'s simulations, per-tree batch,
/// exploration constant, virtual loss and move temperature, over `evaluator`'s leaf
/// evaluations. This is what an `mcts` or `mcts-nn` agent is, taken apart so that many of
/// its games can share one evaluator call.
pub struct Side<'a, A> {
    pub evaluator: &'a mut A,
    pub search: MctsConfig,
    /// Opening book: while the position is in it, the side plays its move without searching.
    pub book: Option<&'a Book>,
}

/// Exploration applied to both sides of a [`lockstep_match`] so that its games differ. Two
/// deterministic searches otherwise play the same game every time, one as X and one as O.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatchExploration {
    /// Plies per game sampled in proportion to visit counts (temperature 1) before each side
    /// falls back to its own `temperature` (argmax at `0`).
    pub temperature_moves: usize,
    /// Dirichlet `(alpha, fraction)` noise on the root priors of every search, as in self-play.
    pub root_noise: Option<(f32, f32)>,
}

/// How a match went, from `a`'s side.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatchTally {
    pub games: usize,
    pub a_wins: usize,
    pub b_wins: usize,
    pub draws: usize,
    /// Wins of `a` in the games it played as X, and as O.
    pub a_wins_as_x: usize,
    pub a_wins_as_o: usize,
    pub moves: usize,
}

struct MatchGame {
    state: BitState,
    /// The search of the side to move.
    tree: SearchTree<BitState>,
    a_plays: Player,
    /// Each side's randomness in this game (move sampling, root noise): `[a, b]`, seeded
    /// with the side's seed plus the game number, so that game `g` of a match is
    /// reproducible on its own.
    rngs: [SmallRng; 2],
    plies: usize,
    /// Finished games stay in place (a batch in flight refers to games by index) and are
    /// skipped.
    finished: bool,
}

impl MatchGame {
    fn a_to_move(&self) -> bool {
        self.state.to_move() == self.a_plays
    }

    fn rng(&mut self, side_a: bool) -> &mut SmallRng {
        &mut self.rngs[usize::from(!side_a)]
    }

    /// Applies `mv`, counting it and the game's end if it ends the game.
    fn play(&mut self, mv: Move, tally: &mut MatchTally, on_game: &mut impl FnMut(usize)) {
        self.state.apply(mv.into());
        self.plies += 1;
        tally.moves += 1;
        if let Some(outcome) = self.state.outcome() {
            tally.games += 1;
            match outcome {
                Outcome::Win(p) if p == self.a_plays => {
                    tally.a_wins += 1;
                    if self.a_plays == Player::X {
                        tally.a_wins_as_x += 1;
                    } else {
                        tally.a_wins_as_o += 1;
                    }
                }
                Outcome::Win(_) => tally.b_wins += 1,
                Outcome::Draw => tally.draws += 1,
            }
            self.finished = true;
            on_game(tally.games);
        }
    }

    /// Plays book moves for as long as the side to move has one, instantly.
    fn play_book_moves(
        &mut self,
        books: (Option<&Book>, Option<&Book>),
        tally: &mut MatchTally,
        on_game: &mut impl FnMut(usize),
    ) {
        while !self.finished {
            let book = if self.a_to_move() { books.0 } else { books.1 };
            match book.and_then(|b| b.best_move(&self.state)) {
                Some(mv) => self.play(mv, tally, on_game),
                None => break,
            }
        }
    }
}

/// The leaves one side's trees claimed this round: the states, each game's span in them
/// (`(game, from, to)`), and the evaluations once they are back.
#[derive(Default)]
struct SideBatch {
    states: Vec<BitState>,
    spans: Vec<(usize, usize, usize)>,
    evals: Evaluations,
    in_flight: bool,
}

impl SideBatch {
    fn claim(&mut self, games: &mut [MatchGame], side_a: bool, tree_batch: usize) {
        self.states.clear();
        self.spans.clear();
        for (g, game) in games.iter_mut().enumerate() {
            if game.finished || game.a_to_move() != side_a {
                continue;
            }
            let from = self.states.len();
            game.tree.claim(tree_batch, &mut self.states);
            self.spans.push((g, from, self.states.len()));
        }
    }

    fn begin<E: UtttAgent<State = BitState>>(&mut self, evaluator: &mut E) {
        if self.states.is_empty() {
            return;
        }
        evaluator.begin_batch(&self.states);
        self.in_flight = true;
    }

    fn finish<E: UtttAgent<State = BitState>>(
        &mut self,
        games: &mut [MatchGame],
        evaluator: &mut E,
    ) {
        if !self.in_flight {
            return;
        }
        self.in_flight = false;
        self.evals.clear();
        evaluator.finish_batch(&self.states, &mut self.evals);
        debug_assert_eq!(self.evals.len(), self.states.len());
        for &(g, from, to) in &self.spans {
            if to > from {
                games[g].tree.resolve(&self.evals, from);
            }
        }
    }
}

fn search_tree(
    state: BitState,
    search: &MctsConfig,
    root_noise: Option<(f32, f32)>,
    rng: &mut SmallRng,
) -> SearchTree<BitState> {
    let tree = SearchTree::new(
        state,
        search.simulations,
        search.c_puct,
        search.virtual_loss,
    );
    match root_noise {
        Some((alpha, fraction)) => tree.with_root_noise(RootNoise {
            alpha,
            fraction,
            seed: rng.random(),
        }),
        None => tree,
    }
}

/// Games of `side_a` whose search is done play their move; the side to move changes, so the
/// game's next tree belongs to the other side. Finished games are tallied and marked.
fn advance_match(
    games: &mut [MatchGame],
    side_a: bool,
    searches: (&MctsConfig, &MctsConfig),
    books: (Option<&Book>, Option<&Book>),
    exploration: &MatchExploration,
    tally: &mut MatchTally,
    on_game: &mut impl FnMut(usize),
) {
    let search = if side_a { searches.0 } else { searches.1 };
    for game in games.iter_mut() {
        if game.finished || game.a_to_move() != side_a || !game.tree.is_done() {
            continue;
        }
        let temperature = if game.plies < exploration.temperature_moves {
            1.0
        } else {
            search.temperature
        };
        let mv = game
            .tree
            .result()
            .sample_move(temperature, &mut game.rngs[usize::from(!side_a)])
            .expect("non-terminal position has moves");
        game.play(mv, tally, on_game);
        if !game.finished {
            game.play_book_moves(books, tally, on_game);
        }
        if !game.finished {
            let a_next = game.a_to_move();
            let next = if a_next { searches.0 } else { searches.1 };
            game.tree = search_tree(game.state, next, exploration.root_noise, game.rng(a_next));
        }
    }
}

/// Plays `games` games between two searches, all at once: `a` takes X in even-numbered games
/// and O in odd-numbered ones. Game `g` seeds each side's randomness with the side's seed
/// plus `g`. Every round, the leaves claimed by all trees of one side go to
/// that side's evaluator in one call, and as in [`self_play`] the two sides take turns so one
/// side's tree work overlaps the other's evaluation. Moves follow `exploration` for the
/// opening plies and each side's `temperature` after that (argmax at `0`). `on_game` is
/// called after each finished game with the number so far.
pub fn lockstep_match<A: UtttAgent<State = BitState>, B: UtttAgent<State = BitState>>(
    a: Side<'_, A>,
    b: Side<'_, B>,
    games: usize,
    exploration: &MatchExploration,
    mut on_game: impl FnMut(usize),
) -> MatchTally {
    let mut tally = MatchTally::default();
    let mut active: Vec<MatchGame> = Vec::with_capacity(games);
    let books = (a.book, b.book);
    for g in 0..games {
        let a_plays = if g % 2 == 0 { Player::X } else { Player::O };
        let rngs = [
            SmallRng::seed_from_u64(a.search.seed.wrapping_add(g as u64)),
            SmallRng::seed_from_u64(b.search.seed.wrapping_add(g as u64)),
        ];
        let mut game = MatchGame {
            state: BitState::initial(),
            tree: SearchTree::new(BitState::initial(), 0, 1.0, 0.0),
            a_plays,
            rngs,
            plies: 0,
            finished: false,
        };
        // Book moves first; the first position outside both books gets the search.
        game.play_book_moves(books, &mut tally, &mut on_game);
        if !game.finished {
            let a_next = game.a_to_move();
            let next = if a_next { &a.search } else { &b.search };
            game.tree = search_tree(game.state, next, exploration.root_noise, game.rng(a_next));
        }
        active.push(game);
    }
    let (mut batch_a, mut batch_b) = (SideBatch::default(), SideBatch::default());
    let searches = (&a.search, &b.search);

    // Same interleaving as self-play, with the two sides as the two groups.
    let mut current_a = true;
    batch_a.claim(&mut active, true, a.search.batch_size);
    batch_a.begin(a.evaluator);
    while active.iter().any(|g| !g.finished) {
        if current_a {
            batch_b.claim(&mut active, false, b.search.batch_size);
            batch_a.finish(&mut active, a.evaluator);
            batch_b.begin(b.evaluator);
            advance_match(
                &mut active,
                true,
                searches,
                books,
                exploration,
                &mut tally,
                &mut on_game,
            );
        } else {
            batch_a.claim(&mut active, true, a.search.batch_size);
            batch_b.finish(&mut active, b.evaluator);
            batch_a.begin(a.evaluator);
            advance_match(
                &mut active,
                false,
                searches,
                books,
                exploration,
                &mut tally,
                &mut on_game,
            );
        }
        current_a = !current_a;
    }
    tally
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
            for (state, policy, value) in &record.positions {
                assert!((-1.0..=1.0).contains(value), "{value}");
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
            let (final_state, policy, _) = record.positions.last().unwrap();
            let mut end = *final_state;
            let played = policy
                .iter()
                .find(|(mv, _)| {
                    let mut next = end;
                    next.apply((*mv).into());
                    next.outcome() == Some(record.outcome)
                })
                .map(|(mv, _)| *mv);
            assert!(
                played.is_some(),
                "the last move ends the game with the recorded outcome"
            );
            end.apply(played.unwrap().into());
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

    #[test]
    fn lockstep_match_alternates_colours_and_counts_every_game() {
        let mut strong = RandomAgent::<BitState>::new(1);
        let mut weak = RandomAgent::<BitState>::new(2);
        let search = |simulations, seed| MctsConfig {
            simulations,
            batch_size: 4,
            seed,
            ..Default::default()
        };
        let mut seen = 0;
        let tally = lockstep_match(
            Side {
                evaluator: &mut strong,
                search: search(40, 3),
                book: None,
            },
            Side {
                evaluator: &mut weak,
                search: search(1, 4),
                book: None,
            },
            6,
            &MatchExploration::default(),
            |n| seen = n,
        );
        assert_eq!(seen, 6);
        assert_eq!(tally.games, 6);
        assert_eq!(tally.a_wins + tally.b_wins + tally.draws, 6);
        assert_eq!(tally.a_wins, tally.a_wins_as_x + tally.a_wins_as_o);
        assert!(tally.moves >= 6 * 17);
        assert!(tally.a_wins >= 4, "{tally:?}");
    }

    /// Two deterministic searches play one game per colour; exploration makes the games differ.
    #[test]
    fn match_exploration_diversifies_the_games() {
        // Rollout-free evaluator: value 0, no priors, so the search is deterministic.
        struct Flat;
        impl UtttAgent for Flat {
            type State = BitState;
            fn choose_move(&mut self, state: &BitState) -> Move {
                state.legal_moves().iter().next().unwrap().into()
            }
            fn evaluate(&mut self, _state: &BitState) -> crate::agent::Evaluation {
                crate::agent::Evaluation::value_only(0.0)
            }
        }
        let search = MctsConfig {
            simulations: 12,
            batch_size: 4,
            ..Default::default()
        };
        let moves = |exploration: &MatchExploration| {
            let tally = lockstep_match(
                Side {
                    evaluator: &mut Flat,
                    search: search.clone(),
                    book: None,
                },
                Side {
                    evaluator: &mut Flat,
                    search: search.clone(),
                    book: None,
                },
                4,
                exploration,
                |_| {},
            );
            tally.moves
        };
        let plain = moves(&MatchExploration::default());
        // Without exploration games 0 and 2 (and 1 and 3) are identical, so the move count is
        // twice that of two games.
        assert_eq!(plain % 2, 0);
        let noisy = moves(&MatchExploration {
            temperature_moves: 4,
            root_noise: Some((0.3, 0.25)),
        });
        assert!(noisy >= 4 * 17);
        // Not a strict guarantee, but with 4 sampled plies and root noise four 12-simulation
        // games collapsing to two identical pairs is vanishingly unlikely.
        assert_ne!(noisy, plain);
    }
}
