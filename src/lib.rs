//! Ultimate Tic Tac Toe agents built around Monte Carlo Tree Search.
//!
//! - [`game`]: rules and the explicit shared types ([`game::Move`], [`game::Square`], ...),
//!   plus two [`game::GameState`] implementations: the readable reference [`game::Position`]
//!   and the fast [`game::BitState`].
//! - [`agent`]: the [`agent::UtttAgent`] trait every agent implements.
//! - [`mcts`]: [`mcts::Mcts`], a PUCT tree search generic over any agent used as leaf evaluator.
//! - [`repl`]: interactive play against a chosen agent.
//! - [`benchmark`]: matches and round-robin tournaments between agents.
//! - [`book`]: an opening book built by searching the positions of the first plies.
//! - [`selfplay`]: lockstep self-play over many games with one batched evaluator.
//! - [`nn`] (feature `nn`, on by default): the value network, its training, and backend selection.

pub mod agent;
pub mod benchmark;
pub mod book;
pub mod game;
pub mod mcts;
#[cfg(feature = "nn")]
pub mod nn;
pub mod repl;
pub mod selfplay;
