//! Ultimate Tic Tac Toe agents built around Monte Carlo Tree Search.
//!
//! - [`game`]: rules and the explicit shared types ([`game::Move`], [`game::Square`], ...),
//!   plus two [`game::GameState`] implementations: the readable reference [`game::Position`]
//!   and the fast [`game::BitState`].
//! - [`agent`]: the [`agent::UtttAgent`] trait every agent implements.
//! - [`mcts`]: [`mcts::Mcts`], a PUCT tree search generic over any agent used as leaf evaluator.
//! - [`repl`]: interactive play against a chosen agent.
//! - [`nn`] (feature `nn`, on by default): the value network, its training, and backend selection.

pub mod agent;
pub mod game;
pub mod mcts;
#[cfg(feature = "nn")]
pub mod nn;
pub mod repl;
