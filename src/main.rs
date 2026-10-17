use ultimate_ttt_burn::agent::{RandomAgent, UtttAgent};
use ultimate_ttt_burn::game::{BitState, GameState, Player};
use ultimate_ttt_burn::mcts::{Mcts, MctsConfig};

/// Demo: MCTS with random playouts (as X) against a random agent (as O).
fn main() {
    let mut mcts = Mcts::new(
        RandomAgent::<BitState>::new(1),
        MctsConfig {
            simulations: 400,
            ..Default::default()
        },
    );
    let mut random = RandomAgent::<BitState>::new(2);
    let mut state = BitState::initial();
    let outcome = loop {
        if let Some(outcome) = state.outcome() {
            break outcome;
        }
        let mv = match state.to_move() {
            Player::X => mcts.choose_move(&state),
            Player::O => random.choose_move(&state),
        };
        println!("{} plays {mv}", state.to_move());
        state.apply(mv);
    };
    println!("\n{}\n\nresult: {outcome:?}", state.to_position());
}
