//! Ultimate Tic Tac Toe agents built around Monte Carlo Tree Search.
//!
//! - [`game`]: rules and the explicit shared types ([`game::Move`], [`game::Square`], ...),
//!   plus two [`game::GameState`] implementations: the readable reference [`game::Position`]
//!   and the fast [`game::BitState`].
//! - [`agent`]: the [`agent::UtttAgent`] trait every agent implements.
//! - [`mcts`]: [`mcts::Mcts`], a PUCT tree search generic over any agent used as leaf evaluator.
//! - [`repl`]: interactive play against a chosen agent.

pub mod agent;
pub mod game;
pub mod mcts;
pub mod repl;
