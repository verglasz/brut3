use ultimate_ttt_burn::game::{BitState, GameState, Move, Position, Square};

fn main() {
    let sq = |r, c| Square::new(r, c).expect("in range");
    let mut state = BitState::initial();
    // A short opening showing the forced-board rule.
    for mv in [
        Move::new(sq(1, 1), sq(0, 2)),
        Move::new(sq(0, 2), sq(1, 1)),
        Move::new(sq(1, 1), sq(2, 0)),
        Move::new(sq(2, 0), sq(1, 1)),
    ] {
        state.apply(mv);
    }
    let position: Position = state.to_position();
    println!("{position}");
    let moves = state.legal_moves_vec();
    println!(
        "{} legal moves: {}",
        moves.len(),
        moves
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    );
}
