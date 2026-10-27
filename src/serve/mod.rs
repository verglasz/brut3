//! A local web page for playing against agents: `serve "mcts 400" "mcts-nn models/x cuda"`
//! serves a board where each side is a human or one of the given agents.
//!
//! A small HTTP/1.1 server on `std::net`, GET only, one connection per request. Each
//! connection gets its own thread; requests that need an agent go to the thread that called
//! [`run`], which owns every agent and answers them one at a time, so agents need not be
//! `Send` and a search never runs twice at once. The page is stateless towards the server: every
//! request carries the whole game as a list of move indices, which the server replays.
//!
//! API (moves are [`Move::index`] values joined by `.`, e.g. `moves=40.36.5`):
//! - `/api/agents`: the agent specs, in order.
//! - `/api/state?moves=…`: cells, board results, legal moves, side to move, outcome.
//! - `/api/move?agent=i&moves=…`: agent `i` picks a move; also its move distribution.
//! - `/api/analyse?agent=i&moves=…`: agent `i`'s value and priors for the position.
//!
//! The page's board styles, colours and icons come from uttt.ai (`utttai/website`,
//! Apache-2.0).

use std::fmt::Write as _;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use crate::agent::{AgentSpec, AnyAgent, UtttAgent};
use crate::game::{BitState, BoardStatus, GameState, Move, Outcome, Player, Square};

const INDEX_HTML: &str = include_str!("index.html");
const FAVICON_SVG: &str = include_str!("favicon.svg");

struct Response {
    status: u16,
    content_type: &'static str,
    body: String,
}

impl Response {
    fn json(body: String) -> Self {
        Self {
            status: 200,
            content_type: "application/json",
            body,
        }
    }

    fn error(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "text/plain; charset=utf-8",
            body: message.into(),
        }
    }
}

/// What the agent thread is asked to do.
enum Task {
    Move,
    Analyse,
}

struct Job {
    task: Task,
    agent: usize,
    state: BitState,
    reply: Sender<Response>,
}

/// Builds the agents, binds `addr` and serves until the process is killed.
pub fn run(specs: &[AgentSpec], addr: &str) -> io::Result<()> {
    let mut agents = specs
        .iter()
        .map(|spec| spec.build())
        .collect::<Result<Vec<AnyAgent>, String>>()
        .map_err(io::Error::other)?;
    let listener = TcpListener::bind(addr)?;
    println!("serving on http://{}", listener.local_addr()?);
    for (i, spec) in specs.iter().enumerate() {
        println!("  agent {i}: {spec}");
    }

    let names: Vec<String> = specs.iter().map(ToString::to_string).collect();
    let agents_json = Arc::new(format!(
        "{{\"agents\":[{}]}}",
        join(names.iter().map(|n| json_string(n)))
    ));
    let (jobs, inbox) = mpsc::channel::<Job>();
    thread::spawn(move || {
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let jobs = jobs.clone();
                    let agents_json = Arc::clone(&agents_json);
                    thread::spawn(move || {
                        if let Err(e) = handle(stream, &jobs, &agents_json) {
                            eprintln!("request failed: {e}");
                        }
                    });
                }
                Err(e) => eprintln!("accept failed: {e}"),
            }
        }
    });
    work(&mut agents, specs, &inbox);
    Ok(())
}

/// Answers agent jobs in arrival order.
fn work(agents: &mut [AnyAgent], specs: &[AgentSpec], inbox: &Receiver<Job>) {
    for job in inbox {
        let Some(agent) = agents.get_mut(job.agent) else {
            let _ = job
                .reply
                .send(Response::error(404, format!("no agent {}", job.agent)));
            continue;
        };
        if job.state.is_terminal() {
            let _ = job.reply.send(Response::error(400, "the game is over"));
            continue;
        }
        let start = Instant::now();
        let response = match job.task {
            Task::Move => {
                let from_book = matches!(agent, AnyAgent::Booked(book, _) if book.best_move(&job.state).is_some());
                let (mv, policy) = agent.choose_move_with_policy(&job.state);
                let ms = start.elapsed().as_millis();
                println!(
                    "agent {} ({}) played {mv}{} in {ms} ms",
                    job.agent,
                    specs[job.agent],
                    if from_book { " from its book" } else { "" }
                );
                format!(
                    "{{\"move\":{},\"policy\":{},\"book\":{from_book},\"ms\":{ms}}}",
                    mv.index(),
                    policy_json(&policy)
                )
            }
            Task::Analyse => {
                let eval = agent.evaluate(&job.state);
                format!(
                    "{{\"value\":{},\"policy\":{},\"ms\":{}}}",
                    json_number(eval.value),
                    policy_json(&eval.priors),
                    start.elapsed().as_millis()
                )
            }
        };
        let _ = job.reply.send(Response::json(response));
    }
}

fn handle(stream: TcpStream, jobs: &Sender<Job>, agents_json: &str) -> io::Result<()> {
    // Browsers open connections ahead of need; one that never sends a request only holds its
    // own thread, until this times out.
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut reader = BufReader::new(&stream);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
            break;
        }
    }
    let mut words = request_line.split_whitespace();
    let (method, target) = (words.next().unwrap_or(""), words.next().unwrap_or(""));
    let response = if method == "GET" {
        route(target, jobs, agents_json)
    } else {
        Response::error(405, "only GET is supported")
    };
    write_response(&stream, &response)
}

fn route(target: &str, jobs: &Sender<Job>, agents_json: &str) -> Response {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let param = |name: &str| {
        query
            .split('&')
            .find_map(|kv| kv.strip_prefix(name)?.strip_prefix('='))
            .unwrap_or("")
    };
    match path {
        "/" | "/index.html" => Response {
            status: 200,
            content_type: "text/html; charset=utf-8",
            body: INDEX_HTML.to_string(),
        },
        "/favicon.svg" => Response {
            status: 200,
            content_type: "image/svg+xml",
            body: FAVICON_SVG.to_string(),
        },
        "/api/agents" => Response::json(agents_json.to_string()),
        "/api/state" => match replay(param("moves")) {
            Ok(state) => Response::json(state_json(&state)),
            Err(e) => Response::error(400, e),
        },
        "/api/move" | "/api/analyse" => {
            let state = match replay(param("moves")) {
                Ok(state) => state,
                Err(e) => return Response::error(400, e),
            };
            let Ok(agent) = param("agent").parse::<usize>() else {
                return Response::error(400, "expected agent=<index>");
            };
            let task = if path == "/api/move" {
                Task::Move
            } else {
                Task::Analyse
            };
            let (reply, answer) = mpsc::channel();
            if jobs
                .send(Job {
                    task,
                    agent,
                    state,
                    reply,
                })
                .is_err()
            {
                return Response::error(503, "the agents are gone");
            }
            answer
                .recv()
                .unwrap_or_else(|_| Response::error(500, "the agent failed"))
        }
        _ => Response::error(404, "not found"),
    }
}

fn write_response(mut stream: &TcpStream, response: &Response) -> io::Result<()> {
    let reason = match response.status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {} {reason}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        response.status,
        response.content_type,
        response.body.len()
    )?;
    stream.write_all(response.body.as_bytes())?;
    stream.flush()
}

/// The state after playing `moves` (move indices joined by `.`) from the initial position.
fn replay(moves: &str) -> Result<BitState, String> {
    let mut state = BitState::initial();
    for (ply, word) in moves.split('.').filter(|w| !w.is_empty()).enumerate() {
        let mv = match word.parse::<usize>() {
            Ok(i) if i < Move::COUNT => Move::from_index(i),
            _ => return Err(format!("move {ply}: {word:?} is not a move index")),
        };
        if state.is_terminal() || !state.is_legal(mv) {
            return Err(format!("move {ply}: {mv} is not legal"));
        }
        state.apply(mv.into());
    }
    Ok(state)
}

fn symbol(player: Player) -> char {
    match player {
        Player::X => 'x',
        Player::O => 'o',
    }
}

/// `cells`: 81 of `x`, `o` or `.` by move index; `boards`: 9 of `x`, `o`, `d` (full) or `.`.
fn state_json(state: &BitState) -> String {
    let cells: String = (0..Move::COUNT)
        .map(|i| state.cell(Move::from_index(i)).map_or('.', symbol))
        .collect();
    let boards: String = (0..9)
        .map(|b| match state.board_status(Square::from_index(b)) {
            BoardStatus::Open => '.',
            BoardStatus::Won(p) => symbol(p),
            BoardStatus::Full => 'd',
        })
        .collect();
    // BitState still lists the open squares once the big board is decided.
    let legal = if state.is_terminal() {
        String::new()
    } else {
        join(
            state
                .legal_moves_vec()
                .into_iter()
                .map(|mv| mv.index().to_string()),
        )
    };
    let outcome = match state.outcome() {
        None => "null".to_string(),
        Some(Outcome::Win(p)) => format!("\"{}\"", symbol(p)),
        Some(Outcome::Draw) => "\"draw\"".to_string(),
    };
    let last = state
        .last_move()
        .map_or("null".to_string(), |mv| mv.index().to_string());
    format!(
        "{{\"cells\":\"{cells}\",\"boards\":\"{boards}\",\"legal\":[{legal}],\"toMove\":\"{}\",\"outcome\":{outcome},\"last\":{last}}}",
        symbol(state.to_move())
    )
}

fn policy_json(policy: &[(Move, f32)]) -> String {
    let entries = policy
        .iter()
        .map(|(mv, p)| format!("[{},{}]", mv.index(), json_number(*p)));
    format!("[{}]", join(entries))
}

fn json_number(x: f32) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".to_string()
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c < ' ' => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn join(items: impl Iterator<Item = impl AsRef<str>>) -> String {
    let mut out = String::new();
    for (i, item) in items.enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(item.as_ref());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Position;

    #[test]
    fn replays_legal_games_and_rejects_illegal_moves() {
        let state = replay("40.36.4").expect("legal");
        assert_eq!(state.to_move(), Player::O);
        assert_eq!(state.last_move(), Some(Move::from_index(4)));
        assert!(replay("40.40").is_err());
        assert!(
            replay("40.0").is_err(),
            "move 0 is not on the forced board 4"
        );
        assert!(replay("81").is_err());
        assert!(replay("").expect("empty game").legal_moves_vec().len() == 81);
    }

    #[test]
    fn finished_games_have_no_legal_moves() {
        // X holds the top rows of boards 0, 1 and 2: the top row of the big board.
        let mut position = Position::initial();
        for board in 0..3 {
            for square in 0..3 {
                position.set_cell(Move::from_index(board * 9 + square), Some(Player::X));
            }
        }
        let state = BitState::from_position(&position);
        assert_eq!(state.outcome(), Some(Outcome::Win(Player::X)));
        let json = state_json(&state);
        assert!(json.contains("\"legal\":[]"), "{json}");
        assert!(json.contains("\"boards\":\"xxx......\""), "{json}");
        assert!(json.contains("\"outcome\":\"x\""), "{json}");
    }

    #[test]
    fn state_json_lists_cells_and_legal_moves() {
        let json = state_json(&replay("40").expect("legal"));
        assert!(json.contains("\"toMove\":\"o\""), "{json}");
        assert!(json.contains("\"last\":40"), "{json}");
        assert!(
            json.contains("\"legal\":[36,37,38,39,41,42,43,44]"),
            "{json}"
        );
        assert_eq!(json_string("a\"b\\c\n"), "\"a\\\"b\\\\c\\u000a\"");
    }
}
