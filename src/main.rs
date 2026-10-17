use std::io;
use std::process::ExitCode;

use ultimate_ttt_burn::agent::AgentSpec;
use ultimate_ttt_burn::repl;

/// Usage: `ultimate-ttt-burn [agent spec]`, e.g. `ultimate-ttt-burn mcts 800 3`.
/// The opponent can also be changed inside the REPL with `agent ...`.
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let spec = if words.is_empty() {
        AgentSpec::default()
    } else {
        match AgentSpec::parse(&words) {
            Ok(spec) => spec,
            Err(e) => {
                eprintln!(
                    "{e}\nusage: ultimate-ttt-burn [random [seed] | mcts [simulations] [seed]]"
                );
                return ExitCode::FAILURE;
            }
        }
    };
    let stdin = io::stdin();
    match repl::run(stdin.lock(), io::stdout().lock(), spec) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("io error: {e}");
            ExitCode::FAILURE
        }
    }
}
