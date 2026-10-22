//! Matches and round-robin tournaments between agents, for comparing trained models with each
//! other and with the search and heuristic baselines.

use std::fmt;
use std::time::Instant;

use crate::agent::{AgentSpec, UtttAgent};
use crate::game::{BitState, GameState, Outcome, Player};
use crate::selfplay::{MatchExploration, Side, lockstep_match};

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

/// Plays `games` games between `a` and `b`, `a` taking X in even-numbered games and O in
/// odd-numbered ones. Game `g` uses each agent's seed plus `g`, so that any game of a match
/// can be replayed on its own. When both are searches (`mcts`, `mcts-nn`) the games are
/// played in lockstep, all at once with each side's leaves batched to its evaluator
/// ([`lockstep_match`]), where only a search's own randomness is seeded per game (a shared
/// playout evaluator keeps one stream); otherwise one game at a time with the agents rebuilt
/// for each game's seed. `on_game` is called after each game with the number played so far.
pub fn play_match(
    a: &AgentSpec,
    b: &AgentSpec,
    games: usize,
    on_game: impl FnMut(usize),
) -> Result<MatchResult, String> {
    play_match_with(a, b, games, &MatchExploration::default(), on_game)
}

/// [`play_match`] with opening exploration for the lockstep case, so that two deterministic
/// searches do not replay the same game. Ignored when either agent is not a search.
pub fn play_match_with(
    a: &AgentSpec,
    b: &AgentSpec,
    games: usize,
    exploration: &MatchExploration,
    mut on_game: impl FnMut(usize),
) -> Result<MatchResult, String> {
    let start = Instant::now();
    let mut result = MatchResult {
        a: a.clone(),
        b: b.clone(),
        games,
        a_wins: 0,
        b_wins: 0,
        draws: 0,
        a_wins_as_x: 0,
        a_wins_as_o: 0,
        moves: 0,
        seconds: 0.0,
    };
    if let (Some((mut eval_a, search_a)), Some((mut eval_b, search_b))) =
        (a.search_parts()?, b.search_parts()?)
    {
        let tally = lockstep_match(
            Side {
                evaluator: &mut eval_a,
                search: search_a,
            },
            Side {
                evaluator: &mut eval_b,
                search: search_b,
            },
            games,
            exploration,
            on_game,
        );
        result.a_wins = tally.a_wins;
        result.b_wins = tally.b_wins;
        result.draws = tally.draws;
        result.a_wins_as_x = tally.a_wins_as_x;
        result.a_wins_as_o = tally.a_wins_as_o;
        result.moves = tally.moves;
        result.seconds = start.elapsed().as_secs_f32();
        return Ok(result);
    }
    for game in 0..games {
        let mut agent_a = a.offset_seed(game as u64).build()?;
        let mut agent_b = b.offset_seed(game as u64).build()?;
        let a_plays = if game % 2 == 0 { Player::X } else { Player::O };
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
            state.apply(mv);
            result.moves += 1;
        };
        match outcome {
            Outcome::Win(p) if p == a_plays => {
                result.a_wins += 1;
                if a_plays == Player::X {
                    result.a_wins_as_x += 1;
                } else {
                    result.a_wins_as_o += 1;
                }
            }
            Outcome::Win(_) => result.b_wins += 1,
            Outcome::Draw => result.draws += 1,
        }
        on_game(game + 1);
    }
    result.seconds = start.elapsed().as_secs_f32();
    Ok(result)
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
    round_robin_with(specs, games_per_pair, &MatchExploration::default(), on_match)
}

/// [`round_robin`] with opening exploration for the lockstep matches
/// ([`play_match_with`]).
pub fn round_robin_with(
    specs: &[AgentSpec],
    games_per_pair: usize,
    exploration: &MatchExploration,
    mut on_match: impl FnMut(&MatchResult),
) -> Result<Tournament, String> {
    let mut tournament = Tournament {
        specs: specs.to_vec(),
        matches: Vec::new(),
    };
    for i in 0..specs.len() {
        for j in i + 1..specs.len() {
            let result =
                play_match_with(&specs[i], &specs[j], games_per_pair, exploration, |_| {})?;
            on_match(&result);
            tournament.matches.push(result);
        }
    }
    Ok(tournament)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_alternates_colours_and_counts_every_game() {
        let a = AgentSpec::Mcts {
            simulations: 60,
            seed: 1,
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
        };
        let b = AgentSpec::Mcts {
            simulations: 1,
            seed: 2,
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
    #[ignore = "plays many search games; run with: cargo test --release -- --include-ignored"]
    fn round_robin_ranks_stronger_agents_higher() {
        let specs = [
            AgentSpec::Random { seed: 1 },
            AgentSpec::Mcts {
                simulations: 60,
                seed: 2,
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
            };
            assert!(play_match(&bad, &AgentSpec::Random { seed: 0 }, 1, |_| {}).is_err());
        }
    }
}
