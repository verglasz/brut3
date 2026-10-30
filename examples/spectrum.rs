//! Singular values of NNUE content tables: how much of each table a rank-r factorisation
//! (`content_r · P`) would keep, with every row counting the same, and with the rows weighted
//! by how often their content occurs in a games file (as the positions are, and over their
//! eight symmetries, as `--augment random` trains on them).
//!
//! `cargo run --release --features full --example spectrum -- [--data <games>] <model stem>...`

use brut3::game::{BitState, GameState, Symmetry};
use brut3::nn::AnyModel;
use brut3::nn::backend::BackendKind;
use brut3::nn::linalg::{gram, symmetric_eigen};
use brut3::nn::nnue::{IDS, IDS_PER_BOARD, OPEN_CONTENTS, TOKENS, nnue_ids};

fn main() {
    let mut args = std::env::args().skip(1).peekable();
    let mut weights = Vec::new();
    if args.peek().map(String::as_str) == Some("--data") {
        args.next();
        let path = args.next().expect("--data <games>");
        let data = brut3::nn::games::load(std::path::Path::new(&path)).expect("games");
        let states: Vec<BitState> = data.samples.iter().map(|s| s.state).collect();
        let raw = content_counts(&states, false);
        let sym = content_counts(&states, true);
        println!("{path}: {} positions", states.len());
        coverage("as played", &raw);
        coverage("8 symmetries", &sym);
        weights = vec![("weighted, as played", raw), ("weighted, 8 symmetries", sym)];
    }
    let device = BackendKind::preferred().device();
    for stem in args {
        let model = AnyModel::load(std::path::Path::new(&stem), &device).expect("model");
        let table = match &model {
            AnyModel::Nnue(m) => m.content.weight.val(),
            AnyModel::NnueV2(m) => m.content.weight.val(),
            other => panic!("{stem}: a {} model, not an NNUE", other.spec()),
        };
        let [_, d] = table.dims();
        let rows: Vec<f32> = table.into_data().try_to_vec::<f32>().expect("f32 table");
        // The open contents only: the rows a factorisation would replace.
        let rows = &rows[..OPEN_CONTENTS * d];
        println!("{stem} ({}): {OPEN_CONTENTS} open content rows x {d}", model.spec());
        report("every row alike", rows, d, &vec![1.0; OPEN_CONTENTS]);
        for (label, w) in &weights {
            report(label, rows, d, w);
        }
    }
}

/// How often each open content is read as a token (the nine boards and the forced board's
/// token) over `states`, or over their eight symmetries each.
fn content_counts(states: &[BitState], symmetries: bool) -> Vec<f64> {
    let mut counts = vec![0f64; OPEN_CONTENTS];
    let mut ids = [0i32; IDS];
    let syms: &[Symmetry] = if symmetries { &Symmetry::ALL } else { &Symmetry::ALL[..1] };
    for s in states {
        for &sym in syms {
            nnue_ids(&s.transform(sym), &mut ids);
            for k in 0..TOKENS {
                let c = ids[k * IDS_PER_BOARD] as usize;
                if c < OPEN_CONTENTS {
                    counts[c] += 1.0;
                }
            }
        }
    }
    counts
}

/// How concentrated the reads are: contents never read, and how many of the most read ones
/// make up half, 90% and 99% of the reads.
fn coverage(label: &str, counts: &[f64]) {
    let mut sorted = counts.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let total: f64 = sorted.iter().sum();
    let unseen = sorted.iter().filter(|&&c| c == 0.0).count();
    let mut line = format!("  {label:<14} {unseen} contents never read; the top");
    for share in [0.5, 0.9, 0.99] {
        let (mut acc, mut n) = (0.0, 0);
        while acc < share * total {
            acc += sorted[n];
            n += 1;
        }
        line += &format!(" {n} make {:.0}%,", share * 100.0);
    }
    println!("{}", line.trim_end_matches(','));
}

/// Prints the share of the table's squared norm (its "energy"), each row weighted by
/// `weights`, that the top r singular directions hold, and the weighted RMS error of the best
/// rank-r approximation relative to the weighted RMS of the table.
fn report(label: &str, rows: &[f32], d: usize, weights: &[f64]) {
    // The weighted Gram matrix: its eigenvalues are the squared singular values of the rows
    // scaled by the square roots of their weights.
    let (eig, _) = symmetric_eigen(gram(rows, d, Some(weights)), d);
    let total: f64 = eig.iter().sum();
    let mut line = format!("  {label:<24}");
    for r in [4, 8, 16, 24, 32, 48] {
        let kept: f64 = eig[..r].iter().sum();
        let rel_err = ((total - kept).max(0.0) / total).sqrt();
        line += &format!("  r{r}: {:5.1}% (err {:.2})", 100.0 * kept / total, rel_err);
    }
    println!("{line}");
    let top = eig[0].sqrt();
    let sv: Vec<String> = eig.iter().take(16).map(|e| format!("{:.2}", e.sqrt() / top)).collect();
    println!("  {:<24}  singular values / largest: {}", "", sv.join(" "));
}
