//! Small dense linear algebra on the host, for analysing weights.

/// The eigenvalues and eigenvectors of the symmetric `d x d` matrix `a` (row-major), by cyclic
/// Jacobi rotations, largest eigenvalue first. Eigenvector `i` is column `i` of the returned
/// row-major `d x d` matrix.
pub fn symmetric_eigen(mut a: Vec<f64>, d: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), d * d, "a {d} x {d} matrix");
    let mut v = vec![0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    let scale: f64 = a.iter().map(|x| x * x).sum::<f64>().max(f64::MIN_POSITIVE);
    for _sweep in 0..100 {
        let off: f64 = (0..d)
            .flat_map(|i| (0..d).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * d + j] * a[i * d + j])
            .sum();
        if off <= 1e-24 * scale {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                let apq = a[p * d + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
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
                for k in 0..d {
                    let (vkp, vkq) = (v[k * d + p], v[k * d + q]);
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| a[j * d + j].total_cmp(&a[i * d + i]));
    let values = order.iter().map(|&i| a[i * d + i]).collect();
    let mut vectors = vec![0f64; d * d];
    for (col, &i) in order.iter().enumerate() {
        for k in 0..d {
            vectors[k * d + col] = v[k * d + i];
        }
    }
    (values, vectors)
}

/// The Gram matrix `Σ w_i row_i row_iᵀ` of the `d`-wide rows of `rows`, row `i` weighted by
/// `weights[i]` (all 1 when `None`).
pub fn gram(rows: &[f32], d: usize, weights: Option<&[f64]>) -> Vec<f64> {
    let mut g = vec![0f64; d * d];
    for (n, row) in rows.chunks_exact(d).enumerate() {
        let w = weights.map_or(1.0, |w| w[n]);
        if w == 0.0 {
            continue;
        }
        for i in 0..d {
            let x = w * f64::from(row[i]);
            for j in 0..d {
                g[i * d + j] += x * f64::from(row[j]);
            }
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_decomposes_a_symmetric_matrix() {
        let a = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let (values, vectors) = symmetric_eigen(a.clone(), 3);
        assert!(values.windows(2).all(|w| w[0] >= w[1]));
        // A v = λ v for every pair.
        for (col, &lambda) in values.iter().enumerate() {
            for row in 0..3 {
                let av: f64 = (0..3).map(|k| a[row * 3 + k] * vectors[k * 3 + col]).sum();
                assert!((av - lambda * vectors[row * 3 + col]).abs() < 1e-9);
            }
        }
    }
}
