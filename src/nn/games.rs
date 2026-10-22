//! Self-play games on disk, so that a generation's games can be reloaded instead of replayed.
//!
//! Little-endian binary: the magic `UTTTGAMES\x02`, a `u32` game count, then per game a `u16`
//! position count and per position 21 bytes of 2-bit cells by move index (0 empty, 1 X, 2 O),
//! one byte for the side to move (0 X, 1 O), one byte for the last move index (255 for none),
//! one byte for the verdict index, one byte saying whether a search win/draw/loss follows
//! (then three `u16` probabilities in units of 1/65535), one byte for the policy length, and
//! per policy entry one byte of move index and a `u16` probability. Version 1 files (magic
//! `\x01`) have no search win/draw/loss and still load.

use std::collections::BTreeMap;
use std::path::Path;

use super::model::Verdict;
use super::train::{Dataset, Sample};
use crate::game::{BitState, GameState, Move, Player, Position};

const MAGIC: &[u8; 10] = b"UTTTGAMES\x02";
const MAGIC_V1: &[u8; 10] = b"UTTTGAMES\x01";
const NO_MOVE: u8 = 255;
const CELL_BYTES: usize = 21;
const PROBABILITY_SCALE: f32 = 65535.0;

/// Writes every game of `data` to `path`, creating the parent directory.
pub fn save(path: &Path, data: &Dataset) -> Result<(), String> {
    let mut by_game: BTreeMap<usize, Vec<&Sample>> = BTreeMap::new();
    for s in &data.samples {
        by_game.entry(s.game).or_default().push(s);
    }
    let mut out = Vec::with_capacity(MAGIC.len() + 4 + data.len() * 64);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(by_game.len() as u32).to_le_bytes());
    for positions in by_game.values() {
        let n = u16::try_from(positions.len())
            .map_err(|_| format!("a game of {} positions does not fit", positions.len()))?;
        out.extend_from_slice(&n.to_le_bytes());
        for s in positions {
            write_sample(&mut out, s)?;
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn write_sample(out: &mut Vec<u8>, s: &Sample) -> Result<(), String> {
    let position = s.state.to_position();
    let mut cells = [0u8; CELL_BYTES];
    for i in 0..Move::COUNT {
        let code = match position.cell(Move::from_index(i)) {
            None => 0,
            Some(Player::X) => 1,
            Some(Player::O) => 2,
        };
        cells[i / 4] |= code << (2 * (i % 4));
    }
    out.extend_from_slice(&cells);
    out.push(match position.to_move() {
        Player::X => 0,
        Player::O => 1,
    });
    out.push(position.last_move().map_or(NO_MOVE, |m| m.index() as u8));
    out.push(s.verdict.index() as u8);
    match s.search_wdl {
        None => out.push(0),
        Some(wdl) => {
            out.push(1);
            for p in wdl {
                let q = (p.clamp(0.0, 1.0) * PROBABILITY_SCALE).round() as u16;
                out.extend_from_slice(&q.to_le_bytes());
            }
        }
    }
    let n = u8::try_from(s.policy.len())
        .map_err(|_| format!("a policy over {} moves does not fit", s.policy.len()))?;
    out.push(n);
    for (mv, p) in &s.policy {
        out.push(mv.index() as u8);
        let q = (p.clamp(0.0, 1.0) * PROBABILITY_SCALE).round() as u16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    Ok(())
}

/// Reads the games written by [`save`]; game ids are `0..`.
pub fn load(path: &Path) -> Result<Dataset, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Reads several files into one dataset with distinct game ids.
pub fn load_many(paths: &[impl AsRef<Path>]) -> Result<Dataset, String> {
    let mut all = Dataset::default();
    for path in paths {
        let data = load(path.as_ref())?;
        let offset = all.games();
        all.samples.extend(data.samples.into_iter().map(|s| Sample {
            game: s.game + offset,
            ..s
        }));
    }
    Ok(all)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at + n;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| format!("truncated at byte {}", self.at))?;
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

fn parse(bytes: &[u8]) -> Result<Dataset, String> {
    let mut r = Reader { bytes, at: 0 };
    let with_search = match r.take(MAGIC.len())? {
        m if m == MAGIC => true,
        m if m == MAGIC_V1 => false,
        _ => return Err("not a games file".into()),
    };
    let games = r.u32()? as usize;
    let mut data = Dataset::default();
    for game in 0..games {
        let positions = r.u16()? as usize;
        for _ in 0..positions {
            data.samples.push(read_sample(&mut r, game, with_search)?);
        }
    }
    if r.at != bytes.len() {
        return Err(format!("{} trailing bytes", bytes.len() - r.at));
    }
    Ok(data)
}

fn read_sample(r: &mut Reader, game: usize, with_search: bool) -> Result<Sample, String> {
    let cells = r.take(CELL_BYTES)?;
    let mut position = Position::initial();
    for i in 0..Move::COUNT {
        let cell = match (cells[i / 4] >> (2 * (i % 4))) & 3 {
            0 => None,
            1 => Some(Player::X),
            2 => Some(Player::O),
            _ => return Err(format!("bad cell code at move {i}")),
        };
        position.set_cell(Move::from_index(i), cell);
    }
    position.set_to_move(match r.u8()? {
        0 => Player::X,
        1 => Player::O,
        other => return Err(format!("bad side to move {other}")),
    });
    position.set_last_move(match r.u8()? {
        NO_MOVE => None,
        index => Some(move_at(index)?),
    });
    let verdict = match r.u8()? {
        v @ 0..=2 => Verdict::from_index(v as usize),
        other => return Err(format!("bad verdict {other}")),
    };
    let search_wdl = if !with_search {
        None
    } else {
        match r.u8()? {
            0 => None,
            1 => {
                let mut wdl = [0.0; 3];
                for p in &mut wdl {
                    *p = r.u16()? as f32 / PROBABILITY_SCALE;
                }
                Some(wdl)
            }
            other => return Err(format!("bad search flag {other}")),
        }
    };
    let n = r.u8()? as usize;
    let mut policy = Vec::with_capacity(n);
    for _ in 0..n {
        let mv = move_at(r.u8()?)?;
        policy.push((mv, r.u16()? as f32 / PROBABILITY_SCALE));
    }
    Ok(Sample {
        state: BitState::from_position(&position),
        verdict,
        policy,
        game,
        search_wdl,
    })
}

fn move_at(index: u8) -> Result<Move, String> {
    if (index as usize) < Move::COUNT {
        Ok(Move::from_index(index as usize))
    } else {
        Err(format!("bad move index {index}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentSpec;
    use crate::nn::train::generate;

    #[test]
    fn games_survive_a_round_trip() {
        let data = generate(&AgentSpec::Mcts { simulations: 4, seed: 3 }, 2, |_| {}).unwrap();
        assert!(data.len() > 10 && data.games() == 2);
        let path = std::env::temp_dir().join(format!("uttt-games-{}.games", std::process::id()));
        save(&path, &data).unwrap();
        let back = load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(back.len(), data.len());
        for (a, b) in data.samples.iter().zip(&back.samples) {
            assert_eq!((a.state, a.verdict, a.game), (b.state, b.verdict, b.game));
            assert_eq!(a.search_wdl.is_some(), b.search_wdl.is_some());
            if let (Some(x), Some(y)) = (a.search_wdl, b.search_wdl) {
                assert!(x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-4), "{x:?} {y:?}");
            }
            assert_eq!(a.policy.len(), b.policy.len());
            for ((ma, pa), (mb, pb)) in a.policy.iter().zip(&b.policy) {
                assert_eq!(ma, mb);
                assert!((pa - pb).abs() < 1e-4, "{pa} vs {pb}");
            }
        }
    }

    #[test]
    fn load_many_renumbers_games() {
        let data = generate(&AgentSpec::Random { seed: 5 }, 3, |_| {}).unwrap();
        let path = std::env::temp_dir().join(format!("uttt-games-many-{}.games", std::process::id()));
        save(&path, &data).unwrap();
        let both = load_many(&[&path, &path]).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!((both.games(), both.len()), (6, 2 * data.len()));
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse(b"nope").is_err());
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        assert!(parse(&bytes).unwrap_err().contains("truncated"));
    }
}
