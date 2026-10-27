//! Import of the uttt.ai datasets (<https://github.com/arnowaczynski/utttai>): positions
//! evaluated by search, one per line, in two text formats.
//!
//! `stage1-mcts`, random-playout MCTS with win/draw/loss counts:
//! ```text
//! evaluatedState{STATE VISITS WINS DRAWS LOSSES} evaluatedActions{SYMBOL INDEX VISITS WINS DRAWS LOSSES,...}
//! ```
//! `stage2-nmcts`, network-guided search with mean values:
//! ```text
//! evaluatedState{STATE VISITS VALUE} evaluatedActions{SYMBOL INDEX VISITS VALUE,...}
//! ```
//! `STATE` is 93 digits: 81 cells by `board * 9 + square` (0 empty, 1 X, 2 O), the same order
//! as [`Move::index`]; 9 board results; the side to move (1 X, 2 O); the forced board (9 for
//! none); the game result. Counts and values are from the side to move's view. The format is
//! told apart per line by the number of fields.
//!
//! Each position becomes a [`Sample`] with the visit distribution as its policy, the
//! win/draw/loss frequencies (stage 1) or the mean value (stage 2) as its value target, and
//! its own game id, since the games the positions came from are not recorded.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::train::{Dataset, Sample};
use crate::game::{BitState, GameState, Move, MoveSet, Player, Position, Square};

const STATE_DIGITS: usize = 93;
const TO_MOVE: usize = 90;
const CONSTRAINT: usize = 91;
const RESULT: usize = 92;
const UNCONSTRAINED: u8 = 9;

/// Reads every file (text, `.txt`) under each path, in parallel, one thread per file.
pub fn import(paths: &[PathBuf]) -> Result<Dataset, String> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "txt"))
                .collect();
            entries.sort();
            files.extend(entries);
        } else {
            files.push(path.clone());
        }
    }
    if files.is_empty() {
        return Err("no dataset files".into());
    }
    let parts: Vec<Result<Vec<Sample>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = files
            .iter()
            .map(|file| scope.spawn(move || import_file(file)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("import thread panicked"))
            .collect()
    });
    let mut data = Dataset::default();
    for part in parts {
        let part = part?;
        let offset = data.len();
        data.samples
            .extend(part.into_iter().enumerate().map(|(i, s)| Sample {
                game: offset + i,
                ..s
            }));
    }
    Ok(data)
}

fn import_file(path: &Path) -> Result<Vec<Sample>, String> {
    let file =
        std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut samples = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        match parse_line(&line) {
            Ok(Some(sample)) => samples.push(sample),
            Ok(None) => {}
            Err(e) => return Err(format!("{}:{}: {e}", path.display(), i + 1)),
        }
    }
    Ok(samples)
}

/// Parses one line; `None` for a terminal position (nothing to learn from it). The game id is
/// left at 0 for the caller to assign.
pub fn parse_line(line: &str) -> Result<Option<Sample>, String> {
    let (state_fields, action_fields) = split_line(line)?;
    let digits = state_fields.first().ok_or("empty evaluatedState")?;
    let state = parse_state(digits)?;
    let Some(state) = state else {
        return Ok(None);
    };
    let (wdl, search_value) = match state_fields.len() {
        // STATE VISITS WINS DRAWS LOSSES
        5 => {
            // The counts occasionally sum to one less than the visit count, so the counts'
            // own sum is the denominator.
            let w: f32 = num(state_fields[2])?;
            let d: f32 = num(state_fields[3])?;
            let l: f32 = num(state_fields[4])?;
            let n = w + d + l;
            if n <= 0.0 {
                return Err(format!("win/draw/loss counts {w}/{d}/{l} are empty"));
            }
            (Some([w / n, d / n, l / n]), Some((w - l) / n))
        }
        // STATE VISITS VALUE
        3 => (None, Some(num::<f32>(state_fields[2])?.clamp(-1.0, 1.0))),
        other => return Err(format!("evaluatedState with {other} fields")),
    };
    let legal = state.legal_moves();
    let mut policy = Vec::with_capacity(action_fields.len());
    let mut total = 0.0f32;
    for action in &action_fields {
        let fields: Vec<&str> = action.split_whitespace().collect();
        if fields.len() != 4 && fields.len() != 6 {
            return Err(format!("evaluatedAction with {} fields", fields.len()));
        }
        let index: usize = num(fields[1])?;
        if index >= Move::COUNT {
            return Err(format!("action index {index}"));
        }
        let mv = Move::from_index(index);
        if !legal.contains(mv.into()) {
            return Err(format!("action {mv} is not legal in the position"));
        }
        let visits: f32 = num(fields[2])?;
        total += visits;
        policy.push((mv, visits));
    }
    if policy.is_empty() || total <= 0.0 {
        return Err("no visited actions".into());
    }
    for (_, p) in &mut policy {
        *p /= total;
    }
    Ok(Some(Sample {
        state,
        policy,
        game: 0,
        wdl,
        search_value,
    }))
}

fn num<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("bad number {s:?}"))
}

/// The space-separated fields inside `evaluatedState{...}` and the comma-separated entries
/// inside `evaluatedActions{...}`.
fn split_line(line: &str) -> Result<(Vec<&str>, Vec<&str>), String> {
    let inner = |tag: &str| -> Result<&str, String> {
        let start = line.find(tag).ok_or_else(|| format!("missing {tag}"))? + tag.len();
        let end = line[start..]
            .find('}')
            .ok_or_else(|| format!("unterminated {tag}"))?;
        Ok(&line[start..start + end])
    };
    let state = inner("evaluatedState{")?.split_whitespace().collect();
    let actions = inner("evaluatedActions{")?
        .split(',')
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .collect();
    Ok((state, actions))
}

/// The position of a 93-digit state string; `None` if the game is over.
pub fn parse_state(digits: &str) -> Result<Option<BitState>, String> {
    let bytes = digits.as_bytes();
    if bytes.len() != STATE_DIGITS || !bytes.iter().all(u8::is_ascii_digit) {
        return Err(format!(
            "state must be {STATE_DIGITS} digits, got {digits:?}"
        ));
    }
    let digit = |i: usize| bytes[i] - b'0';
    if digit(RESULT) != 0 {
        return Ok(None);
    }
    let mut position = Position::initial();
    for i in 0..Move::COUNT {
        let cell = match digit(i) {
            0 => None,
            1 => Some(Player::X),
            2 => Some(Player::O),
            other => return Err(format!("cell value {other}")),
        };
        position.set_cell(Move::from_index(i), cell);
    }
    let to_move = match digit(TO_MOVE) {
        1 => Player::X,
        2 => Player::O,
        other => return Err(format!("side to move {other}")),
    };
    position.set_to_move(to_move);
    let constraint = digit(CONSTRAINT);
    let last_move = last_move_for(&position, to_move.other(), constraint)?;
    position.set_last_move(last_move);
    let state = BitState::from_position(&position);
    let forced = state.forced_board().map(|b| b.index() as u8);
    let expected = (constraint != UNCONSTRAINED).then_some(constraint);
    if forced != expected {
        return Err(format!(
            "forced board {forced:?} does not match the constraint {constraint}"
        ));
    }
    if state.is_terminal() {
        return Err("position is over but the state says otherwise".into());
    }
    Ok(Some(state))
}

/// A move by `mover` consistent with `constraint`: one on square `constraint` for a forced
/// board, or on the square of a decided board for a free choice; `None` for the initial
/// position. Which board it was played on is not recorded and does not matter to the rules.
fn last_move_for(
    position: &Position,
    mover: Player,
    constraint: u8,
) -> Result<Option<Move>, String> {
    let mark_at = |square: Square| {
        Square::ALL
            .iter()
            .map(|board| Move::new(*board, square))
            .find(|mv| position.cell(*mv) == Some(mover))
    };
    if constraint != UNCONSTRAINED {
        let square = Square::from_index(constraint as usize);
        return mark_at(square)
            .map(Some)
            .ok_or_else(|| format!("no {mover} mark on square {square} for the constraint"));
    }
    let any_marks = Square::ALL
        .iter()
        .flat_map(|b| Square::ALL.iter().map(move |s| Move::new(*b, *s)))
        .any(|mv| position.cell(mv).is_some());
    if !any_marks {
        return Ok(None);
    }
    let decided = Square::ALL
        .iter()
        .copied()
        .filter(|b| position.board_status(*b).is_decided());
    for square in decided {
        if let Some(mv) = mark_at(square) {
            return Ok(Some(mv));
        }
    }
    Err("free choice without a decided board the last move could have pointed to".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::sq;

    const INITIAL: &str = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000190";

    #[test]
    fn parses_the_initial_position_in_both_formats() {
        let mcts = format!(
            "evaluatedState{{{INITIAL} 1000 400 300 300}} evaluatedActions{{1 0 500 200 150 150,1 40 500 200 150 150}}"
        );
        let s = parse_line(&mcts).unwrap().unwrap();
        assert_eq!(s.state, BitState::initial());
        let wdl = s.wdl.unwrap();
        assert!((wdl[0] - 0.4).abs() < 1e-6 && (wdl[1] - 0.3).abs() < 1e-6);
        assert!((s.search_value.unwrap() - 0.1).abs() < 1e-6);
        assert_eq!(s.policy.len(), 2);
        assert_eq!(s.policy[1], (Move::from_index(40), 0.5));

        let nmcts = format!(
            "evaluatedState{{{INITIAL} 30000 0.084027}} evaluatedActions{{1 0 177 0.058632,1 40 13174 0.115555}}"
        );
        let s = parse_line(&nmcts).unwrap().unwrap();
        assert_eq!(s.wdl, None);
        assert!((s.search_value.unwrap() - 0.084027).abs() < 1e-6);
        assert!((s.policy[1].1 - 13174.0 / (13174.0 + 177.0)).abs() < 1e-6);
    }

    #[test]
    fn reconstructs_forced_boards_and_free_choice() {
        // X played board 4 square 2 (index 38): O to move, forced to board 2.
        let mut digits = vec![b'0'; STATE_DIGITS];
        digits[38] = b'1';
        digits[TO_MOVE] = b'2';
        digits[CONSTRAINT] = b'2';
        let state = parse_state(std::str::from_utf8(&digits).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(state.to_move(), Player::O);
        assert_eq!(state.forced_board(), Some(sq(0, 2)));
        assert_eq!(state.legal_moves_vec().len(), 9);

        // Board 0 won by X (top row), X's last move went to square 0, so O chooses freely.
        let mut digits = vec![b'0'; STATE_DIGITS];
        for i in [0, 1, 2] {
            digits[i] = b'1';
        }
        for i in [9, 10] {
            digits[i] = b'2';
        }
        digits[81] = b'1';
        digits[TO_MOVE] = b'2';
        digits[CONSTRAINT] = b'9';
        let state = parse_state(std::str::from_utf8(&digits).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(state.forced_board(), None);
        assert_eq!(state.legal_moves_vec().len(), 81 - 9 - 2);

        // A contradictory constraint is rejected: board 2 open but no X mark on square 2.
        let mut digits = vec![b'0'; STATE_DIGITS];
        digits[40] = b'1';
        digits[TO_MOVE] = b'2';
        digits[CONSTRAINT] = b'2';
        assert!(parse_state(std::str::from_utf8(&digits).unwrap()).is_err());
        // Terminal positions are skipped.
        let mut digits = vec![b'0'; STATE_DIGITS];
        digits[RESULT] = b'1';
        digits[TO_MOVE] = b'1';
        digits[CONSTRAINT] = b'9';
        assert_eq!(parse_state(std::str::from_utf8(&digits).unwrap()), Ok(None));
    }

    #[test]
    fn rejects_illegal_actions_and_bad_lines() {
        let bad = format!("evaluatedState{{{INITIAL} 10 5 3 2}} evaluatedActions{{1 81 10 5 3 2}}");
        assert!(parse_line(&bad).is_err());
        let empty = format!("evaluatedState{{{INITIAL} 0 0 0 0}} evaluatedActions{{1 0 10 5 3 2}}");
        assert!(parse_line(&empty).is_err());
        assert!(parse_line("nonsense").is_err());
    }

    #[test]
    fn imports_a_file_with_sequential_game_ids() {
        let dir = std::env::temp_dir().join(format!("uttt-utttai-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let line =
            format!("evaluatedState{{{INITIAL} 10 5 3 2}} evaluatedActions{{1 0 10 5 3 2}}\n");
        std::fs::write(dir.join("depth00.txt"), format!("{line}{line}")).unwrap();
        std::fs::write(dir.join("depth01.txt"), &line).unwrap();
        std::fs::write(dir.join("notes.md"), "ignored").unwrap();
        let data = import(&[dir.clone()]).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(data.len(), 3);
        assert_eq!(data.games(), 3);
    }
}
