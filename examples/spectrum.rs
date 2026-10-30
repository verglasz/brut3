//! Singular values of NNUE content tables: how much of each table a rank-r factorisation
//! (`content_r · P`) would keep.
//!
//! `cargo run --release --features full --example spectrum -- <model stem>...`

use brut3::nn::backend::BackendKind;
use brut3::nn::nnue::OPEN_CONTENTS;
use brut3::nn::AnyModel;

fn main() {
    let device = BackendKind::preferred().device();
    for stem in std::env::args().skip(1) {
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
        report("as is", rows, d, false);
        report("mean row removed", rows, d, true);
    }
}

/// Prints the share of the table's squared norm (its "energy") the top r singular directions
/// hold, and the RMS error per entry of the best rank-r approximation relative to the table's
/// RMS.
fn report(label: &str, rows: &[f32], d: usize, centre: bool) {
    let n = rows.len() / d;
    let mut mean = vec![0f64; d];
    if centre {
        for row in rows.chunks_exact(d) {
            for (m, &x) in mean.iter_mut().zip(row) {
                *m += f64::from(x) / n as f64;
            }
        }
    }
    // The Gram matrix: its eigenvalues are the squared singular values.
    let mut gram = vec![0f64; d * d];
    for row in rows.chunks_exact(d) {
        let r: Vec<f64> = row.iter().zip(&mean).map(|(&x, m)| f64::from(x) - m).collect();
        for i in 0..d {
            for j in 0..d {
                gram[i * d + j] += r[i] * r[j];
            }
        }
    }
    let mut eig = jacobi_eigenvalues(gram, d);
    eig.sort_by(|a, b| b.total_cmp(a));
    let total: f64 = eig.iter().sum();
    let mut line = format!("  {label:<18}");
    for r in [4, 8, 16, 24, 32, 48] {
        let kept: f64 = eig[..r].iter().sum();
        let rel_err = ((total - kept).max(0.0) / total).sqrt();
        line += &format!("  r{r}: {:5.1}% (err {:.2})", 100.0 * kept / total, rel_err);
    }
    println!("{line}");
    let sv: Vec<String> = eig.iter().take(12).map(|e| format!("{:.1}", e.sqrt())).collect();
    println!("  {:<18}  top singular values: {}", "", sv.join(" "));
}

/// The eigenvalues of the symmetric `d x d` matrix `a` by cyclic Jacobi rotations.
fn jacobi_eigenvalues(mut a: Vec<f64>, d: usize) -> Vec<f64> {
    for _sweep in 0..100 {
        let off: f64 = (0..d)
            .flat_map(|i| (0..d).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * d + j] * a[i * d + j])
            .sum();
        if off < 1e-18 {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..d {
                    let (akp, akq) = (a[k * d + p], a[k * d + q]);
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let (apk, aqk) = (a[p * d + k], a[q * d + k]);
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..d).map(|i| a[i * d + i]).collect()
}
