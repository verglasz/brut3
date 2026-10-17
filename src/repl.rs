//! Interactive play against a chosen agent, over any line-based input and output.

use std::io::{self, BufRead, Write};

use crate::agent::{AgentSpec, AnyAgent, UtttAgent};
use crate::game::{BitState, GameState, Move, Player, Position};

const HELP: &str = "\
commands:
  <move>               play a move: r1c2/r0c0 (board/square), or four digits: 1 2 0 0
  new [x|o]            new game; you play X (default) or O
  agent <spec>         choose the opponent: random [seed] | mcts [simulations] [seed]
  moves                list legal moves
  show                 print the board
  hint                 ask the agent what it would play here
  go                   let the agent play the current side (any side)
  undo                 take back your last move (and the agent's reply)
  help                 this text
  quit";

enum Flow {
    Continue,
    Quit,
}

pub struct Repl<W: Write> {
    out: W,
    spec: AgentSpec,
    agent: AnyAgent,
    state: BitState,
    human: Player,
    /// States before each applied move, for `undo`.
    history: Vec<BitState>,
}

/// Runs the REPL until `quit` or end of input.
pub fn run(input: impl BufRead, out: impl Write, spec: AgentSpec) -> io::Result<()> {
    let mut repl = Repl::new(out, spec)?;
    repl.show()?;
    for line in input.lines() {
        let line = line?;
        if let Flow::Quit = repl.handle(&line)? {
            break;
        }
        write!(repl.out, "> ")?;
        repl.out.flush()?;
    }
    Ok(())
}

impl<W: Write> Repl<W> {
    pub fn new(mut out: W, spec: AgentSpec) -> io::Result<Self> {
        writeln!(
            out,
            "Ultimate Tic Tac Toe. You are X against {spec}. Type `help` for commands."
        )?;
        write!(out, "> ")?;
        out.flush()?;
        Ok(Self {
            out,
            agent: spec.build(),
            spec,
            state: BitState::initial(),
            human: Player::X,
            history: Vec::new(),
        })
    }

    pub fn into_output(self) -> W {
        self.out
    }

    fn handle(&mut self, line: &str) -> io::Result<Flow> {
        let words: Vec<&str> = line.split_whitespace().collect();
        let Some(&cmd) = words.first() else {
            return Ok(Flow::Continue);
        };
        match cmd {
            "quit" | "exit" | "q" => return Ok(Flow::Quit),
            "help" | "?" => writeln!(self.out, "{HELP}")?,
            "new" => self.new_game(words.get(1).copied())?,
            "agent" => self.set_agent(&words[1..])?,
            "moves" => self.list_moves()?,
            "show" => self.show()?,
            "hint" => self.hint()?,
            "go" => self.agent_moves()?,
            "undo" => self.undo()?,
            _ => match parse_move(line) {
                Some(mv) => self.human_moves(mv)?,
                None => writeln!(
                    self.out,
                    "unknown command or malformed move {line:?}; type `help`"
                )?,
            },
        }
        Ok(Flow::Continue)
    }

    fn show(&mut self) -> io::Result<()> {
        let position: Position = self.state.to_position();
        writeln!(self.out, "\n{position}\n")
    }

    fn new_game(&mut self, side: Option<&str>) -> io::Result<()> {
        self.human = match side.map(str::to_ascii_lowercase).as_deref() {
            None | Some("x") => Player::X,
            Some("o") => Player::O,
            Some(other) => {
                return writeln!(self.out, "side must be x or o, got {other:?}");
            }
        };
        self.state = BitState::initial();
        self.history.clear();
        writeln!(self.out, "new game: you are {}", self.human)?;
        if self.state.to_move() != self.human {
            self.agent_moves()?;
        }
        self.show()
    }

    fn set_agent(&mut self, words: &[&str]) -> io::Result<()> {
        match AgentSpec::parse(words) {
            Ok(spec) => {
                self.agent = spec.build();
                self.spec = spec;
                writeln!(self.out, "opponent: {}", self.spec)
            }
            Err(e) => writeln!(self.out, "{e}"),
        }
    }

    fn list_moves(&mut self) -> io::Result<()> {
        if self.state.outcome().is_some() {
            return writeln!(self.out, "game over; `new` to start another");
        }
        let moves = self.state.legal_moves_vec();
        let text: Vec<String> = moves.iter().map(ToString::to_string).collect();
        writeln!(self.out, "{} legal moves: {}", moves.len(), text.join(" "))
    }

    fn hint(&mut self) -> io::Result<()> {
        if self.state.outcome().is_some() {
            return writeln!(self.out, "game over; `new` to start another");
        }
        let mv = self.agent.choose_move(&self.state);
        writeln!(self.out, "{} suggests {mv}", self.spec)
    }

    fn human_moves(&mut self, mv: Move) -> io::Result<()> {
        if self.state.outcome().is_some() {
            return writeln!(self.out, "game over; `new` to start another");
        }
        if !self.state.is_legal(mv) {
            return writeln!(self.out, "illegal move {mv}; `moves` lists the legal ones");
        }
        self.play(mv)?;
        if self.state.outcome().is_none() && self.state.to_move() != self.human {
            self.agent_moves()?;
        }
        self.show()
    }

    /// The agent plays for whichever side is to move.
    fn agent_moves(&mut self) -> io::Result<()> {
        if self.state.outcome().is_some() {
            return writeln!(self.out, "game over; `new` to start another");
        }
        let mv = self.agent.choose_move(&self.state);
        self.play(mv)
    }

    fn play(&mut self, mv: Move) -> io::Result<()> {
        let mover = self.state.to_move();
        self.history.push(self.state);
        self.state.apply(mv);
        writeln!(self.out, "{mover} plays {mv}")?;
        if let Some(outcome) = self.state.outcome() {
            writeln!(self.out, "game over: {outcome:?}")?;
        }
        Ok(())
    }

    /// Goes back to the most recent state in which the human was to move.
    fn undo(&mut self) -> io::Result<()> {
        let Some(at) = self.history.iter().rposition(|s| s.to_move() == self.human) else {
            return writeln!(self.out, "nothing to undo");
        };
        self.state = self.history[at];
        self.history.truncate(at);
        self.show()
    }
}

/// Accepts `r1c2/r0c0`, or four digits in any spacing: board row, board col, square row,
/// square col (`1 2 0 0`, `12 00`, `1200`).
pub fn parse_move(text: &str) -> Option<Move> {
    let text = text.trim();
    if let Ok(mv) = text.parse::<Move>() {
        return Some(mv);
    }
    if text
        .chars()
        .any(|c| !c.is_ascii_digit() && !c.is_whitespace())
    {
        return None;
    }
    let digits: Vec<u8> = text
        .chars()
        .filter(char::is_ascii_digit)
        .map(|c| c as u8 - b'0')
        .collect();
    let [br, bc, sr, sc] = digits[..] else {
        return None;
    };
    use crate::game::Square;
    Some(Move::new(Square::new(br, bc)?, Square::new(sr, sc)?))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::game::sq;

    fn session(spec: AgentSpec, script: &str) -> String {
        let mut out = Vec::new();
        run(Cursor::new(script), &mut out, spec).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn parses_moves_in_both_forms() {
        let mv = Move::new(sq(1, 2), sq(0, 0));
        assert_eq!(parse_move("r1c2/r0c0"), Some(mv));
        assert_eq!(parse_move("1 2 0 0"), Some(mv));
        assert_eq!(parse_move("12 00"), Some(mv));
        assert_eq!(parse_move(" 1200 "), Some(mv));
        assert_eq!(parse_move("1 2 0"), None);
        assert_eq!(parse_move("1 3 0 0"), None);
        assert_eq!(parse_move("moves"), None);
    }

    #[test]
    fn plays_a_move_and_gets_a_reply() {
        let out = session(AgentSpec::Random { seed: 1 }, "1 1 0 2\nmoves\nquit\n");
        assert!(out.contains("X plays r1c1/r0c2"));
        assert!(out.contains("O plays r0c2/"), "{out}");
        assert!(out.contains("legal moves:"));
        assert!(out.contains("to move: X"));
    }

    #[test]
    fn rejects_illegal_and_unknown_input() {
        let out = session(
            AgentSpec::Random { seed: 1 },
            "1 1 1 1\n1 1 1 1\nfrobnicate\nnew z\nquit\n",
        );
        assert!(out.contains("illegal move r1c1/r1c1"));
        assert!(out.contains("unknown command"));
        assert!(out.contains("side must be x or o"));
    }

    #[test]
    fn new_as_o_makes_agent_open_and_undo_restores() {
        let out = session(
            AgentSpec::Mcts {
                simulations: 10,
                seed: 2,
            },
            "new o\nundo\nquit\n",
        );
        assert!(out.contains("new game: you are O"));
        assert!(out.contains("X plays "));
        // Nothing of the human's to take back: the agent's opening stays on the board.
        assert!(out.contains("nothing to undo"));
        // The last board shown still has O (the human) to move.
        assert!(out.rfind("to move: O") > out.rfind("to move: X"), "{out}");
    }

    #[test]
    fn undo_takes_back_human_and_agent_move() {
        let mut out = Vec::new();
        let mut repl = Repl::new(&mut out, AgentSpec::Random { seed: 3 }).unwrap();
        repl.handle("1 1 0 2").unwrap();
        assert_eq!(repl.history.len(), 2);
        repl.handle("1 1 0 2").unwrap(); // illegal now: nothing changes
        assert_eq!(repl.history.len(), 2);
        repl.handle("undo").unwrap();
        assert!(repl.history.is_empty());
        assert_eq!(repl.state, BitState::initial());
        repl.handle("undo").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("nothing to undo"));
    }

    #[test]
    fn agent_can_be_switched_mid_game() {
        let out = session(
            AgentSpec::Random { seed: 1 },
            "agent mcts 5 1\nhint\nagent bogus\nquit\n",
        );
        assert!(out.contains("opponent: mcts (5 simulations"));
        assert!(out.contains("suggests r"));
        assert!(out.contains("unknown agent \"bogus\""));
    }
}
