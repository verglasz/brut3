//! [`AnyAgent`]: one concrete type covering every built agent, chosen by an [`AgentSpec`].
//!
//! An enum rather than `dyn UtttAgent` so the state type stays concrete ([`BitState`]) and
//! every call is a plain match.

use std::fmt;

use super::{Evaluation, RandomAgent, UtttAgent};
use crate::game::{BitState, Move};
use crate::mcts::{Mcts, MctsConfig};

/// How to build an agent. Parsed from words like `random`, `random 7`, `mcts`, `mcts 800 3`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentSpec {
    Random {
        seed: u64,
    },
    /// MCTS with random playouts.
    Mcts {
        simulations: u32,
        seed: u64,
    },
}

impl AgentSpec {
    pub const DEFAULT_SIMULATIONS: u32 = 400;

    /// Parses `random [seed]` or `mcts [simulations] [seed]`.
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
            Some("mcts") => Ok(Self::Mcts {
                simulations: num(words, 1, "simulations", Self::DEFAULT_SIMULATIONS)?,
                seed: num(words, 2, "seed", 0)?,
            }),
            Some(other) => Err(format!("unknown agent {other:?}; try `random` or `mcts`")),
            None => Err("missing agent name; try `random` or `mcts`".into()),
        }
    }

    pub fn build(&self) -> AnyAgent {
        match *self {
            AgentSpec::Random { seed } => AnyAgent::Random(RandomAgent::new(seed)),
            AgentSpec::Mcts { simulations, seed } => AnyAgent::Mcts(Mcts::new(
                RandomAgent::new(seed ^ 0x9E37_79B9_7F4A_7C15),
                MctsConfig {
                    simulations,
                    seed,
                    ..Default::default()
                },
            )),
        }
    }
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
            AgentSpec::Mcts { simulations, seed } => {
                write!(
                    f,
                    "mcts ({simulations} simulations, random playouts, seed {seed})"
                )
            }
        }
    }
}

/// Every agent that can be built from an [`AgentSpec`].
// One long-lived value per process; the Mcts variant's size does not matter.
#[allow(clippy::large_enum_variant)]
pub enum AnyAgent {
    Random(RandomAgent<BitState>),
    Mcts(Mcts<RandomAgent<BitState>>),
}

impl UtttAgent for AnyAgent {
    type State = BitState;

    fn choose_move(&mut self, state: &BitState) -> Move {
        match self {
            AnyAgent::Random(a) => a.choose_move(state),
            AnyAgent::Mcts(a) => a.choose_move(state),
        }
    }

    fn evaluate(&mut self, state: &BitState) -> Evaluation {
        match self {
            AnyAgent::Random(a) => a.evaluate(state),
            AnyAgent::Mcts(a) => a.evaluate(state),
        }
    }

    fn evaluate_batch(&mut self, states: &[BitState], out: &mut Vec<Evaluation>) {
        match self {
            AnyAgent::Random(a) => a.evaluate_batch(states, out),
            AnyAgent::Mcts(a) => a.evaluate_batch(states, out),
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
        assert!(AgentSpec::parse(&["minimax"]).is_err());
        assert!(AgentSpec::parse(&[]).is_err());
    }

    #[test]
    fn built_agents_play_legal_moves() {
        for spec in [
            AgentSpec::Random { seed: 1 },
            AgentSpec::Mcts {
                simulations: 20,
                seed: 1,
            },
        ] {
            let mut agent = spec.build();
            let state = BitState::initial();
            assert!(state.is_legal(agent.choose_move(&state)));
            let e = agent.evaluate(&state);
            assert!((-1.0..=1.0).contains(&e.value));
        }
    }
}
