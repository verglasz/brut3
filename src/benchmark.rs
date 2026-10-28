//! Matches and round-robin tournaments between agents, for comparing trained models with each
//! other and with the search and heuristic baselines.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::Instant;

use crate::agent::{AgentSpec, AnyAgent, UtttAgent};
use crate::book::Book;
use crate::game::{BitState, GameState, Outcome, Player};
use crate::mcts::MctsConfig;
use crate::selfplay::{MatchExploration, MatchTally, Side, lockstep_match_games, search_threads};

/// Outcome of a series of games between two agents, alternating colours.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchResult {
    pub a: AgentSpec,
    pub b: AgentSpec,
    pub games: usize,
    pub a_wins: usize,
    pub b_wins: usize,
    pub draws: usize,
    /// Wins of `a` in the games it played as X, and as O.
    pub a_wins_as_x: usize,
    pub a_wins_as_o: usize,
    pub moves: usize,
    pub seconds: f32,
}

impl MatchResult {
    /// `a`'s score in `[0, 1]`: a win counts 1, a draw 0.5.
    pub fn score_a(&self) -> f32 {
        if self.games == 0 {
            return 0.5;
        }
        (self.a_wins as f32 + 0.5 * self.draws as f32) / self.games as f32
    }

    pub fn score_b(&self) -> f32 {
        1.0 - self.score_a()
    }
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} vs {}: {}-{}-{} (W-D-L for the first, score {:.3}; as X {} wins, as O {} wins) in {:.1}s",
            self.a,
            self.b,
            self.a_wins,
            self.draws,
            self.b_wins,
            self.score_a(),
            self.a_wins_as_x,
            self.a_wins_as_o,
            self.seconds
        )
    }
}

/// How the matches of [`play_match_with`] and [`round_robin_with`] are played.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchSettings {
    /// Opening exploration for matches between searches, so that two deterministic searches
    /// do not replay the same game.
    pub exploration: MatchExploration,
    /// Threads playing (0 = one per core; see [`search_threads`]). Each works through the
    /// pairings in order, drawing game numbers from the pairing it is on; one that finds a
    /// pairing's games all handed out moves on to the next, so with several threads more
    /// than one pairing can be playing at a time.
    pub threads: usize,
    /// Most games in flight at once over all threads, shared out among them; 0 for no limit
    /// (each thread takes its share of a pairing's games at once).
    pub concurrent: usize,
}

impl Default for MatchSettings {
    fn default() -> Self {
        Self {
            exploration: MatchExploration::default(),
            threads: 1,
            concurrent: 0,
        }
    }
}

/// A match in progress, as [`round_robin_with`] reports it after each finished game.
#[derive(Clone, Debug)]
pub struct Progress<'a> {
    pub a: &'a AgentSpec,
    pub b: &'a AgentSpec,
    /// The games the match will have.
    pub games: usize,
    /// The games finished so far, from `a`'s side.
    pub tally: MatchTally,
}

/// Plays `games` games between `a` and `b`, `a` taking X in even-numbered games and O in
/// odd-numbered ones. Game `g` uses each agent's seed plus `g`, so that any game of a match
/// can be replayed on its own. When both are searches (`mcts`, `mcts-nn`) the games are
/// played in lockstep, each side's leaves batched to its evaluator ([`lockstep_match_games`]),
/// where only a search's own randomness is seeded per game (a shared playout evaluator keeps
/// one stream); otherwise one game at a time with the agents rebuilt for each game's seed.
/// `on_game` is called after each game with the number played so far.
pub fn play_match(
    a: &AgentSpec,
    b: &AgentSpec,
    games: usize,
    on_game: impl FnMut(usize) + Send,
) -> Result<MatchResult, String> {
    play_match_with(a, b, games, &MatchSettings::default(), on_game)
}

/// [`play_match`] with its games played as `settings` says.
pub fn play_match_with(
    a: &AgentSpec,
    b: &AgentSpec,
    games: usize,
    settings: &MatchSettings,
    on_game: impl FnMut(usize) + Send,
) -> Result<MatchResult, String> {
    let on_game = Mutex::new(on_game);
    let mut results = play_pairings(
        vec![Pairing::new(a, b, games)?],
        settings,
        |p: &Progress| (on_game.lock().expect("on_game panicked"))(p.tally.games),
        |_| {},
    )?;
    Ok(results.remove(0))
}

/// One pairing as its games are played: their numbers are handed out one by one to whichever
/// thread asks, and what the threads finish is added up as it finishes.
struct Pairing<'a> {
    a: &'a AgentSpec,
    b: &'a AgentSpec,
    games: usize,
    /// The next game number to hand out.
    next: AtomicUsize,
    /// Both sides' search settings and opening books, when both are searches.
    searches: Option<(MctsConfig, MctsConfig, Option<Book>, Option<Book>)>,
    total: Mutex<MatchTally>,
    /// When its first game started.
    start: OnceLock<Instant>,
}

impl<'a> Pairing<'a> {
    fn new(a: &'a AgentSpec, b: &'a AgentSpec, games: usize) -> Result<Self, String> {
        let searches = match (a.search_config(), b.search_config()) {
            (Some(search_a), Some(search_b)) => {
                Some((search_a, search_b, a.load_book()?, b.load_book()?))
            }
            _ => None,
        };
        Ok(Self {
            a,
            b,
            games,
            next: AtomicUsize::new(0),
            searches,
            total: Mutex::new(MatchTally::default()),
            start: OnceLock::new(),
        })
    }

    fn take(&self) -> Option<usize> {
        let g = self.next.fetch_add(1, Ordering::Relaxed);
        (g < self.games).then_some(g)
    }

    fn handed_out(&self) -> bool {
        self.next.load(Ordering::Relaxed) >= self.games
    }

    fn result(&self, tally: &MatchTally) -> MatchResult {
        MatchResult {
            a: self.a.clone(),
            b: self.b.clone(),
            games: self.games,
            a_wins: tally.a_wins,
            b_wins: tally.b_wins,
            draws: tally.draws,
            a_wins_as_x: tally.a_wins_as_x,
            a_wins_as_o: tally.a_wins_as_o,
            moves: tally.moves,
            seconds: self
                .start
                .get()
                .map_or(0.0, |start| start.elapsed().as_secs_f32()),
        }
    }
}

/// Where the threads' searches evaluate their leaves: with several threads, every distinct
/// network on one [`EvalServer`](crate::nn::EvalServer) that batches every thread's leaves
/// (its per-batch costs paid once for all, one folded table on the device); otherwise, and
/// for random playouts, an evaluator per thread.
struct Pool {
    #[cfg(feature = "nn")]
    servers: Vec<(
        std::path::PathBuf,
        crate::nn::BackendKind,
        crate::nn::EvalServer,
    )>,
}

impl Pool {
    fn new(pairings: &[Pairing], shared: bool) -> Result<Self, String> {
        #[cfg(feature = "nn")]
        {
            let mut servers: Vec<(_, _, crate::nn::EvalServer)> = Vec::new();
            let specs = pairings
                .iter()
                .filter(|p| p.searches.is_some())
                .flat_map(|p| [p.a, p.b]);
            for spec in specs.filter(|_| shared) {
                if let AgentSpec::MctsNn { path, backend, .. } = spec {
                    if !servers.iter().any(|(p, b, _)| p == path && b == backend) {
                        let spec = spec.clone();
                        let server = crate::nn::EvalServer::spawn(
                            move || Ok(spec.leaf_evaluator()?.expect("a search's leaf evaluator")),
                            crate::nn::MAX_BATCH,
                        )?;
                        servers.push((path.clone(), *backend, server));
                    }
                }
            }
            Ok(Self { servers })
        }
        #[cfg(not(feature = "nn"))]
        {
            let _ = (pairings, shared);
            Ok(Self {})
        }
    }

    /// Thread `thread`'s evaluator for the leaves of `spec`'s search.
    fn for_thread(&self, spec: &AgentSpec, thread: usize) -> Result<AnyAgent, String> {
        #[cfg(feature = "nn")]
        if let AgentSpec::MctsNn { path, backend, .. } = spec {
            if let Some((_, _, server)) = self
                .servers
                .iter()
                .find(|(p, b, _)| p == path && b == backend)
            {
                return Ok(AnyAgent::Remote(server.client()));
            }
        }
        Ok(spec
            .thread_leaf_evaluator(thread as u64)?
            .expect("a search's leaf evaluator"))
    }
}

/// Plays game `g` between `a` and `b` (not both searches) with the agents built for the
/// game's seed, adding it to `tally`.
fn play_single(
    a: &AgentSpec,
    b: &AgentSpec,
    g: usize,
    tally: &mut MatchTally,
) -> Result<(), String> {
    let mut agent_a = a.offset_seed(g as u64).build()?;
    let mut agent_b = b.offset_seed(g as u64).build()?;
    let a_plays = if g % 2 == 0 { Player::X } else { Player::O };
    let mut state = BitState::initial();
    let outcome = loop {
        if let Some(outcome) = state.outcome() {
            break outcome;
        }
        let mv = if state.to_move() == a_plays {
            agent_a.choose_move(&state)
        } else {
            agent_b.choose_move(&state)
        };
        state.apply(mv.into());
        tally.moves += 1;
    };
    tally.games += 1;
    match outcome {
        Outcome::Win(p) if p == a_plays => {
            tally.a_wins += 1;
            if a_plays == Player::X {
                tally.a_wins_as_x += 1;
            } else {
                tally.a_wins_as_o += 1;
            }
        }
        Outcome::Win(_) => tally.b_wins += 1,
        Outcome::Draw => tally.draws += 1,
    }
    Ok(())
}

/// Plays every pairing as `settings` says (see [`MatchSettings`]): `on_progress` after each
/// finished game, from the playing threads, and `on_match` with each match once its last game
/// is in, on the calling thread. Returns the results in pairing order.
fn play_pairings(
    pairings: Vec<Pairing<'_>>,
    settings: &MatchSettings,
    on_progress: impl Fn(&Progress) + Sync,
    mut on_match: impl FnMut(&MatchResult),
) -> Result<Vec<MatchResult>, String> {
    let total_games = pairings.iter().map(|p| p.games).sum();
    let mut threads = search_threads(settings.threads, total_games);
    if settings.concurrent > 0 {
        threads = threads.min(settings.concurrent);
    }
    let pool = Pool::new(&pairings, threads > 1)?;
    let mut results: Vec<Option<MatchResult>> = vec![None; pairings.len()];
    for (i, p) in pairings.iter().enumerate().filter(|(_, p)| p.games == 0) {
        let result = p.result(&MatchTally::default());
        on_match(&result);
        results[i] = Some(result);
    }
    let stop = AtomicBool::new(false);
    let mut error = None;
    let (tx, rx) = mpsc::channel::<Result<(usize, MatchResult), String>>();
    std::thread::scope(|scope| {
        for t in 0..threads {
            // Thread t's slots: its share of the concurrency, or of each pairing's games.
            let slots = match settings.concurrent {
                0 => None,
                n => Some(n / threads + usize::from(t < n % threads)),
            };
            let (tx, pairings, pool, stop, on_progress) =
                (tx.clone(), &pairings, &pool, &stop, &on_progress);
            scope.spawn(move || {
                for (i, p) in pairings.iter().enumerate() {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    if p.handed_out() {
                        continue;
                    }
                    p.start.get_or_init(Instant::now);
                    // Each finished game goes into the pairing's total at once; whoever adds
                    // the last one reports the match.
                    let mut seen = MatchTally::default();
                    let mut on_game = |tally: &MatchTally| {
                        let new = tally.since(&seen);
                        seen = tally.clone();
                        let total = {
                            let mut total = p.total.lock().expect("a match thread panicked");
                            total.add(&new);
                            total.clone()
                        };
                        on_progress(&Progress {
                            a: p.a,
                            b: p.b,
                            games: p.games,
                            tally: total.clone(),
                        });
                        if total.games == p.games {
                            let _ = tx.send(Ok((i, p.result(&total))));
                        }
                    };
                    let played = match &p.searches {
                        Some((search_a, search_b, book_a, book_b)) => {
                            let evaluators = (pool.for_thread(p.a, t), pool.for_thread(p.b, t));
                            match evaluators {
                                (Ok(mut eval_a), Ok(mut eval_b)) => {
                                    lockstep_match_games(
                                        Side {
                                            evaluator: &mut eval_a,
                                            search: search_a.clone(),
                                            book: book_a.as_ref(),
                                        },
                                        Side {
                                            evaluator: &mut eval_b,
                                            search: search_b.clone(),
                                            book: book_b.as_ref(),
                                        },
                                        || p.take(),
                                        slots.unwrap_or(p.games.div_ceil(threads)).max(1),
                                        &settings.exploration,
                                        &mut on_game,
                                    );
                                    Ok(())
                                }
                                (Err(e), _) | (_, Err(e)) => Err(e),
                            }
                        }
                        None => (|| {
                            let mut tally = MatchTally::default();
                            while let Some(g) = p.take() {
                                play_single(p.a, p.b, g, &mut tally)?;
                                on_game(&tally);
                            }
                            Ok(())
                        })(),
                    };
                    if let Err(e) = played {
                        stop.store(true, Ordering::Relaxed);
                        let _ = tx.send(Err(e));
                        return;
                    }
                }
            });
        }
        drop(tx);
        for message in rx {
            match message {
                Ok((i, result)) => {
                    on_match(&result);
                    results[i] = Some(result);
                }
                Err(e) => {
                    error.get_or_insert(e);
                }
            }
        }
    });
    if let Some(e) = error {
        return Err(e);
    }
    Ok(results
        .into_iter()
        .map(|r| r.expect("every match finished"))
        .collect())
}

/// Every pair of agents plays a match.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tournament {
    pub specs: Vec<AgentSpec>,
    pub matches: Vec<MatchResult>,
}

/// One agent's line in the standings.
#[derive(Clone, Debug, PartialEq)]
pub struct Standing {
    pub spec: AgentSpec,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    /// Points over games played: a win 1, a draw 0.5.
    pub score: f32,
}

impl Tournament {
    /// Standings sorted by score, best first.
    pub fn standings(&self) -> Vec<Standing> {
        let mut standings: Vec<Standing> = self
            .specs
            .iter()
            .map(|spec| {
                let (mut wins, mut draws, mut losses) = (0, 0, 0);
                for m in &self.matches {
                    if m.a == *spec {
                        wins += m.a_wins;
                        draws += m.draws;
                        losses += m.b_wins;
                    } else if m.b == *spec {
                        wins += m.b_wins;
                        draws += m.draws;
                        losses += m.a_wins;
                    }
                }
                let games = wins + draws + losses;
                let score = if games == 0 {
                    0.0
                } else {
                    (wins as f32 + 0.5 * draws as f32) / games as f32
                };
                Standing {
                    spec: spec.clone(),
                    wins,
                    draws,
                    losses,
                    score,
                }
            })
            .collect();
        standings.sort_by(|x, y| y.score.total_cmp(&x.score));
        standings
    }

    /// Standings and the pairwise results as text.
    pub fn table(&self) -> String {
        let mut out = String::new();
        out.push_str("standings (score = wins + draws/2, over games played):\n");
        for (rank, s) in self.standings().iter().enumerate() {
            out.push_str(&format!(
                "  {:>2}. {:.3}  {:>4}W {:>4}D {:>4}L  {}\n",
                rank + 1,
                s.score,
                s.wins,
                s.draws,
                s.losses,
                s.spec
            ));
        }
        out.push_str("matches (W-D-L for the first agent):\n");
        for m in &self.matches {
            out.push_str(&format!(
                "  {}-{}-{}  {} vs {}  ({:.1}s)\n",
                m.a_wins, m.draws, m.b_wins, m.a, m.b, m.seconds
            ));
        }
        out
    }
}

/// Plays `games_per_pair` games between every pair of `specs`. `on_match` is called after
/// each match with its result.
pub fn round_robin(
    specs: &[AgentSpec],
    games_per_pair: usize,
    on_match: impl FnMut(&MatchResult),
) -> Result<Tournament, String> {
    round_robin_with(
        specs,
        games_per_pair,
        &MatchSettings::default(),
        |_| {},
        on_match,
    )
}

/// [`round_robin`] with its matches played as `settings` says, `on_progress` called from the
/// playing threads after each finished game and `on_match` with each match as it completes
/// (with several threads, not necessarily in pairing order).
pub fn round_robin_with(
    specs: &[AgentSpec],
    games_per_pair: usize,
    settings: &MatchSettings,
    on_progress: impl Fn(&Progress) + Sync,
    on_match: impl FnMut(&MatchResult),
) -> Result<Tournament, String> {
    let mut pairings = Vec::new();
    for i in 0..specs.len() {
        for j in i + 1..specs.len() {
            pairings.push(Pairing::new(&specs[i], &specs[j], games_per_pair)?);
        }
    }
    Ok(Tournament {
        specs: specs.to_vec(),
        matches: play_pairings(pairings, settings, on_progress, on_match)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_alternates_colours_and_counts_every_game() {
        let a = AgentSpec::Mcts {
            simulations: 60,
            seed: 1,
            book: None,
        };
        let b = AgentSpec::Random { seed: 2 };
        let mut seen = 0;
        let r = play_match(&a, &b, 6, |n| seen = n).unwrap();
        assert_eq!(seen, 6);
        assert_eq!(r.a_wins + r.b_wins + r.draws, 6);
        assert_eq!(r.a_wins, r.a_wins_as_x + r.a_wins_as_o);
        assert!(r.a_wins >= 5, "{r}");
        assert!(r.moves >= 6 * 17);
        assert!((r.score_a() + r.score_b() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn two_searches_play_in_lockstep() {
        let a = AgentSpec::Mcts {
            simulations: 40,
            seed: 1,
            book: None,
        };
        let b = AgentSpec::Mcts {
            simulations: 1,
            seed: 2,
            book: None,
        };
        let mut seen = 0;
        let r = play_match(&a, &b, 4, |n| seen = n).unwrap();
        assert_eq!(seen, 4);
        assert_eq!(r.a_wins + r.b_wins + r.draws, 4);
        assert_eq!(r.a_wins, r.a_wins_as_x + r.a_wins_as_o);
        assert!(r.a_wins >= r.b_wins, "{r}");
        assert!(r.moves >= 4 * 17);
    }

    #[test]
    fn threaded_searches_play_every_game_once() {
        let search = |simulations, seed| AgentSpec::Mcts {
            simulations,
            seed,
            book: None,
        };
        let (a, b) = (search(40, 1), search(1, 2));
        let mut seen = Vec::new();
        let settings = MatchSettings {
            threads: 3,
            concurrent: 2,
            ..MatchSettings::default()
        };
        let r = play_match_with(&a, &b, 7, &settings, |n| seen.push(n)).unwrap();
        // Threads report as they finish, so the counts may arrive out of order.
        seen.sort_unstable();
        assert_eq!(seen, (1..=7).collect::<Vec<_>>());
        assert_eq!(r.a_wins + r.b_wins + r.draws, 7);
        assert_eq!(r.a_wins, r.a_wins_as_x + r.a_wins_as_o);
        assert!(r.moves >= 7 * 17);
    }

    #[test]
    fn several_pairings_play_at_once_and_report_in_order() {
        let specs = [
            AgentSpec::Mcts {
                simulations: 8,
                seed: 1,
                book: None,
            },
            AgentSpec::Mcts {
                simulations: 4,
                seed: 2,
                book: None,
            },
            AgentSpec::Random { seed: 3 },
        ];
        let settings = MatchSettings {
            threads: 2,
            concurrent: 4,
            ..MatchSettings::default()
        };
        let progress = std::sync::atomic::AtomicUsize::new(0);
        let mut done = 0;
        let t = round_robin_with(
            &specs,
            3,
            &settings,
            |_| {
                progress.fetch_add(1, Ordering::Relaxed);
            },
            |_| done += 1,
        )
        .unwrap();
        assert_eq!((done, progress.into_inner()), (3, 9));
        let pairs: Vec<_> = t.matches.iter().map(|m| (&m.a, &m.b)).collect();
        assert_eq!(
            pairs,
            [
                (&specs[0], &specs[1]),
                (&specs[0], &specs[2]),
                (&specs[1], &specs[2])
            ]
        );
        assert!(t.matches.iter().all(|m| m.a_wins + m.b_wins + m.draws == 3));
    }

    #[test]
    #[ignore = "plays many search games; run with: cargo test --release -- --include-ignored"]
    fn round_robin_ranks_stronger_agents_higher() {
        let specs = [
            AgentSpec::Random { seed: 1 },
            AgentSpec::Mcts {
                simulations: 60,
                seed: 2,
                book: None,
            },
            AgentSpec::Greedy { seed: 3 },
        ];
        let mut matches = 0;
        let t = round_robin(&specs, 4, |_| matches += 1).unwrap();
        assert_eq!(matches, 3);
        assert_eq!(t.matches.len(), 3);
        let standings = t.standings();
        assert_eq!(standings.len(), 3);
        assert!(standings.windows(2).all(|w| w[0].score >= w[1].score));
        assert_eq!(
            standings.last().unwrap().spec,
            AgentSpec::Random { seed: 1 },
            "{}",
            t.table()
        );
        assert!(t.table().contains("standings"));
    }

    #[test]
    fn bad_spec_is_an_error() {
        #[cfg(feature = "nn")]
        {
            let bad = AgentSpec::Nn {
                path: "/nonexistent".into(),
                backend: crate::nn::BackendKind::preferred(),
                book: None,
            };
            assert!(play_match(&bad, &AgentSpec::Random { seed: 0 }, 1, |_| {}).is_err());
        }
    }
}
