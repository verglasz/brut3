//! [`AnyAgent`]: one concrete type covering every built agent, chosen by an [`AgentSpec`].
//!
//! An enum rather than `dyn UtttAgent` so the state type stays concrete ([`BitState`]) and
//! every call is a plain match.

use std::fmt;
#[cfg(feature = "nn")]
use std::path::PathBuf;

use super::{Evaluation, Evaluations, GreedyAgent, MinimaxAgent, RandomAgent, UtttAgent};
use crate::game::{BitState, Move};
use crate::mcts::{Mcts, MctsConfig};
#[cfg(feature = "nn")]
use crate::nn::{BackendKind, NetAgent};

/// How to build an agent. Parsed from words like `random`, `greedy 7`, `minimax 4`, `mcts 800 3`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentSpec {
    Random {
        seed: u64,
    },
    /// One-ply lookahead on the static heuristic.
    Greedy {
        seed: u64,
    },
    /// Alpha-beta search of `depth` plies on the static heuristic.
    Minimax {
        depth: u32,
        seed: u64,
    },
    /// MCTS with random playouts.
    Mcts {
        simulations: u32,
        seed: u64,
    },
    /// One-ply lookahead with a trained value network.
    #[cfg(feature = "nn")]
    Nn {
        path: PathBuf,
        backend: BackendKind,
    },
    /// MCTS with value-network leaf evaluation, batched.
    #[cfg(feature = "nn")]
    MctsNn {
        path: PathBuf,
        backend: BackendKind,
        simulations: u32,
        batch_size: usize,
        seed: u64,
    },
}

/// Leaves per round for a random-playout search: enough to keep every core busy on big
/// searches, while a small search keeps a small batch so that it is not reduced to
/// breadth-first exploration under virtual loss.
fn rollout_batch(simulations: u32) -> usize {
    (simulations as usize / 32).clamp(1, 16 * crate::agent::random::rollout_threads())
}

#[cfg(not(feature = "nn"))]
const NAMES: &str = "`random`, `greedy`, `minimax` or `mcts`";
#[cfg(feature = "nn")]
const NAMES: &str = "`random`, `greedy`, `minimax`, `mcts`, `nn` or `mcts-nn`";

impl AgentSpec {
    pub const DEFAULT_SIMULATIONS: u32 = 400;
    #[cfg(feature = "nn")]
    pub const DEFAULT_NN_BATCH: usize = 16;

    /// Parses `random [seed]`, `greedy [seed]`, `minimax [depth] [seed]`,
    /// `mcts [simulations] [seed]`, `nn <weights> [backend]` or
    /// `mcts-nn <weights> [backend] [simulations] [batch] [seed]`.
    pub fn parse(words: &[&str]) -> Result<Self, String> {
        fn num<T: std::str::FromStr>(
            words: &[&str],
            i: usize,
            what: &str,
            default: T,
        ) -> Result<T, String> {
            match words.get(i) {
                None => Ok(default),
                Some(w) => w
                    .parse()
                    .map_err(|_| format!("{what} must be a number, got {w:?}")),
            }
        }
        match words.first().copied() {
            Some("random") => Ok(Self::Random {
                seed: num(words, 1, "seed", 0)?,
            }),
            Some("greedy") => Ok(Self::Greedy {
                seed: num(words, 1, "seed", 0)?,
            }),
            Some("minimax") => Ok(Self::Minimax {
                depth: num(words, 1, "depth", MinimaxAgent::DEFAULT_DEPTH)?,
                seed: num(words, 2, "seed", 0)?,
            }),
            Some("mcts") => Ok(Self::Mcts {
                simulations: num(words, 1, "simulations", Self::DEFAULT_SIMULATIONS)?,
                seed: num(words, 2, "seed", 0)?,
            }),
            #[cfg(feature = "nn")]
            Some("nn") => {
                let (path, backend, rest) = nn_target(&words[1..])?;
                if let Some(extra) = rest.first() {
                    return Err(format!("unexpected argument {extra:?} after nn backend"));
                }
                Ok(Self::Nn { path, backend })
            }
            #[cfg(feature = "nn")]
            Some("mcts-nn") => {
                let (path, backend, rest) = nn_target(&words[1..])?;
                Ok(Self::MctsNn {
                    path,
                    backend,
                    simulations: num(rest, 0, "simulations", Self::DEFAULT_SIMULATIONS)?,
                    batch_size: num(rest, 1, "batch", Self::DEFAULT_NN_BATCH)?,
                    seed: num(rest, 2, "seed", 0)?,
                })
            }
            Some(other) => Err(format!("unknown agent {other:?}; try {NAMES}")),
            None => Err(format!("missing agent name; try {NAMES}")),
        }
    }

    /// For a search agent (`mcts`, `mcts-nn`): the leaf evaluator it searches over and its
    /// search settings, so that many of its games can be played in lockstep
    /// ([`crate::selfplay::lockstep_match`]). `None` for agents that are not searches. Fails
    /// only for network agents whose weights cannot be loaded.
    pub fn search_parts(&self) -> Result<Option<(AnyAgent, MctsConfig)>, String> {
        Ok(match self {
            AgentSpec::Mcts { simulations, seed } => Some((
                AnyAgent::Random(RandomAgent::new(seed ^ 0x9E37_79B9_7F4A_7C15)),
                MctsConfig {
                    simulations: *simulations,
                    batch_size: rollout_batch(*simulations),
                    seed: *seed,
                    ..Default::default()
                },
            )),
            #[cfg(feature = "nn")]
            AgentSpec::MctsNn {
                path,
                backend,
                simulations,
                batch_size,
                seed,
            } => Some((
                AnyAgent::Nn(NetAgent::load(path, *backend)?),
                MctsConfig {
                    simulations: *simulations,
                    batch_size: *batch_size,
                    seed: *seed,
                    ..Default::default()
                },
            )),
            _ => None,
        })
    }

    /// Builds the agent. Fails only for network agents whose weights cannot be loaded.
    pub fn build(&self) -> Result<AnyAgent, String> {
        Ok(match self {
            AgentSpec::Random { seed } => AnyAgent::Random(RandomAgent::new(*seed)),
            AgentSpec::Greedy { seed } => AnyAgent::Greedy(GreedyAgent::new(*seed)),
            AgentSpec::Minimax { depth, seed } => {
                AnyAgent::Minimax(MinimaxAgent::new(*depth, *seed))
            }
            AgentSpec::Mcts { simulations, seed } => AnyAgent::Mcts(Mcts::new(
                RandomAgent::new(seed ^ 0x9E37_79B9_7F4A_7C15),
                MctsConfig {
                    simulations: *simulations,
                    batch_size: rollout_batch(*simulations),
                    seed: *seed,
                    ..Default::default()
                },
            )),
            #[cfg(feature = "nn")]
            AgentSpec::Nn { path, backend } => AnyAgent::Nn(NetAgent::load(path, *backend)?),
            #[cfg(feature = "nn")]
            AgentSpec::MctsNn {
                path,
                backend,
                simulations,
                batch_size,
                seed,
            } => AnyAgent::MctsNn(Mcts::new(
                NetAgent::load(path, *backend)?,
                MctsConfig {
                    simulations: *simulations,
                    batch_size: *batch_size,
                    seed: *seed,
                    ..Default::default()
                },
            )),
        })
    }
}

/// Parses `<weights> [backend]` and returns the remaining words.
#[cfg(feature = "nn")]
fn nn_target<'w, 'a>(
    words: &'w [&'a str],
) -> Result<(PathBuf, BackendKind, &'w [&'a str]), String> {
    let path = words
        .first()
        .ok_or("missing weights path; try `nn models/value`")?;
    let (backend, rest) = match words.get(1) {
        Some(w) if w.parse::<u64>().is_err() => (w.parse::<BackendKind>()?, &words[2..]),
        _ => (BackendKind::preferred(), &words[1..]),
    };
    Ok((PathBuf::from(path), backend, rest))
}

impl Default for AgentSpec {
    fn default() -> Self {
        Self::Mcts {
            simulations: Self::DEFAULT_SIMULATIONS,
            seed: 0,
        }
    }
}

impl fmt::Display for AgentSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentSpec::Random { seed } => write!(f, "random (seed {seed})"),
            AgentSpec::Greedy { seed } => write!(f, "greedy (seed {seed})"),
            AgentSpec::Minimax { depth, seed } => write!(f, "minimax (depth {depth}, seed {seed})"),
            AgentSpec::Mcts { simulations, seed } => {
                write!(
                    f,
                    "mcts ({simulations} simulations, random playouts, seed {seed})"
                )
            }
            #[cfg(feature = "nn")]
            AgentSpec::Nn { path, backend } => {
                write!(f, "nn ({} on {backend})", path.display())
            }
            #[cfg(feature = "nn")]
            AgentSpec::MctsNn {
                path,
                backend,
                simulations,
                batch_size,
                seed,
            } => write!(
                f,
                "mcts-nn ({simulations} simulations, batch {batch_size}, {} on {backend}, seed {seed})",
                path.display()
            ),
        }
    }
}

/// Every agent that can be built from an [`AgentSpec`].
// One long-lived value per process; the Mcts variant's size does not matter.
#[allow(clippy::large_enum_variant)]
pub enum AnyAgent {
    Random(RandomAgent<BitState>),
    Greedy(GreedyAgent),
    Minimax(MinimaxAgent),
    Mcts(Mcts<RandomAgent<BitState>>),
    #[cfg(feature = "nn")]
    Nn(NetAgent),
    #[cfg(feature = "nn")]
    MctsNn(Mcts<NetAgent>),
}

impl UtttAgent for AnyAgent {
    type State = BitState;

    fn choose_move(&mut self, state: &BitState) -> Move {
        match self {
            AnyAgent::Random(a) => a.choose_move(state),
            AnyAgent::Greedy(a) => a.choose_move(state),
            AnyAgent::Minimax(a) => a.choose_move(state),
            AnyAgent::Mcts(a) => a.choose_move(state),
            #[cfg(feature = "nn")]
            AnyAgent::Nn(a) => a.choose_move(state),
            #[cfg(feature = "nn")]
            AnyAgent::MctsNn(a) => a.choose_move(state),
        }
    }

    fn choose_move_with_policy(&mut self, state: &BitState) -> (Move, Vec<(Move, f32)>) {
        match self {
            AnyAgent::Random(a) => a.choose_move_with_policy(state),
            AnyAgent::Greedy(a) => a.choose_move_with_policy(state),
            AnyAgent::Minimax(a) => a.choose_move_with_policy(state),
            AnyAgent::Mcts(a) => a.choose_move_with_policy(state),
            #[cfg(feature = "nn")]
            AnyAgent::Nn(a) => a.choose_move_with_policy(state),
            #[cfg(feature = "nn")]
            AnyAgent::MctsNn(a) => a.choose_move_with_policy(state),
        }
    }

    fn evaluate(&mut self, state: &BitState) -> Evaluation {
        match self {
            AnyAgent::Random(a) => a.evaluate(state),
            AnyAgent::Greedy(a) => a.evaluate(state),
            AnyAgent::Minimax(a) => a.evaluate(state),
            AnyAgent::Mcts(a) => a.evaluate(state),
            #[cfg(feature = "nn")]
            AnyAgent::Nn(a) => a.evaluate(state),
            #[cfg(feature = "nn")]
            AnyAgent::MctsNn(a) => a.evaluate(state),
        }
    }

    fn evaluate_batch(&mut self, states: &[BitState], out: &mut Evaluations) {
        match self {
            AnyAgent::Random(a) => a.evaluate_batch(states, out),
            AnyAgent::Greedy(a) => a.evaluate_batch(states, out),
            AnyAgent::Minimax(a) => a.evaluate_batch(states, out),
            AnyAgent::Mcts(a) => a.evaluate_batch(states, out),
            #[cfg(feature = "nn")]
            AnyAgent::Nn(a) => a.evaluate_batch(states, out),
            #[cfg(feature = "nn")]
            AnyAgent::MctsNn(a) => a.evaluate_batch(states, out),
        }
    }

    fn begin_batch(&mut self, states: &[BitState]) {
        match self {
            AnyAgent::Random(a) => a.begin_batch(states),
            AnyAgent::Greedy(a) => a.begin_batch(states),
            AnyAgent::Minimax(a) => a.begin_batch(states),
            AnyAgent::Mcts(a) => a.begin_batch(states),
            #[cfg(feature = "nn")]
            AnyAgent::Nn(a) => a.begin_batch(states),
            #[cfg(feature = "nn")]
            AnyAgent::MctsNn(a) => a.begin_batch(states),
        }
    }

    fn finish_batch(&mut self, states: &[BitState], out: &mut Evaluations) {
        match self {
            AnyAgent::Random(a) => a.finish_batch(states, out),
            AnyAgent::Greedy(a) => a.finish_batch(states, out),
            AnyAgent::Minimax(a) => a.finish_batch(states, out),
            AnyAgent::Mcts(a) => a.finish_batch(states, out),
            #[cfg(feature = "nn")]
            AnyAgent::Nn(a) => a.finish_batch(states, out),
            #[cfg(feature = "nn")]
            AnyAgent::MctsNn(a) => a.finish_batch(states, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;

    #[test]
    fn parse_specs() {
        assert_eq!(
            AgentSpec::parse(&["random"]),
            Ok(AgentSpec::Random { seed: 0 })
        );
        assert_eq!(
            AgentSpec::parse(&["random", "9"]),
            Ok(AgentSpec::Random { seed: 9 })
        );
        assert_eq!(
            AgentSpec::parse(&["mcts"]),
            Ok(AgentSpec::Mcts {
                simulations: 400,
                seed: 0
            })
        );
        assert_eq!(
            AgentSpec::parse(&["mcts", "50", "3"]),
            Ok(AgentSpec::Mcts {
                simulations: 50,
                seed: 3
            })
        );
        assert!(AgentSpec::parse(&["mcts", "lots"]).is_err());
        assert_eq!(
            AgentSpec::parse(&["greedy"]),
            Ok(AgentSpec::Greedy { seed: 0 })
        );
        assert_eq!(
            AgentSpec::parse(&["minimax", "3", "2"]),
            Ok(AgentSpec::Minimax { depth: 3, seed: 2 })
        );
        assert!(AgentSpec::parse(&["alphabeta"]).is_err());
        #[cfg(feature = "nn")]
        {
            let backend = BackendKind::preferred();
            assert_eq!(
                AgentSpec::parse(&["nn", "models/value"]),
                Ok(AgentSpec::Nn {
                    path: "models/value".into(),
                    backend
                })
            );
            assert_eq!(
                AgentSpec::parse(&["mcts-nn", "w", backend.name(), "100", "8", "3"]),
                Ok(AgentSpec::MctsNn {
                    path: "w".into(),
                    backend,
                    simulations: 100,
                    batch_size: 8,
                    seed: 3
                })
            );
            assert_eq!(
                AgentSpec::parse(&["mcts-nn", "w", "100"]),
                Ok(AgentSpec::MctsNn {
                    path: "w".into(),
                    backend,
                    simulations: 100,
                    batch_size: AgentSpec::DEFAULT_NN_BATCH,
                    seed: 0
                })
            );
            assert!(AgentSpec::parse(&["nn"]).is_err());
            assert!(AgentSpec::parse(&["nn", "w", "tpu"]).is_err());
            assert!(
                AgentSpec::Nn {
                    path: "/nonexistent".into(),
                    backend
                }
                .build()
                .is_err()
            );
        }
        assert!(AgentSpec::parse(&[]).is_err());
    }

    #[test]
    fn built_agents_play_legal_moves() {
        for spec in [
            AgentSpec::Random { seed: 1 },
            AgentSpec::Greedy { seed: 1 },
            AgentSpec::Minimax { depth: 2, seed: 1 },
            AgentSpec::Mcts {
                simulations: 20,
                seed: 1,
            },
        ] {
            let mut agent = spec.build().unwrap();
            let state = BitState::initial();
            assert!(state.is_legal(agent.choose_move(&state)));
            let e = agent.evaluate(&state);
            assert!((-1.0..=1.0).contains(&e.value));
        }
    }
}
