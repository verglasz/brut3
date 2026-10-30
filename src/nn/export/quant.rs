//! Quantisation of the weights and their base64 text for `Gen/Weights.hs`.
//!
//! Every array of [`Weights::arrays`] is shipped on its own, row-major, base64-encoded with
//! the standard alphabet and padding, in one of two formats for the whole export:
//! - int16: little-endian `Int16` values with one scale for the array, `max |w| / 32767`;
//! - int8 (`--int8`, see [`quantise_int8`]): each row an exponent byte then its values as signed
//!   bytes, the row's scale the array's base scale (`max |w| / 127`) times `2^(-e / 8)`.
//!   About half the text of int16, and every row keeps its own precision.

use super::weights::{Array, Weights};

/// Largest quantised magnitude.
const Q_MAX: f32 = i16::MAX as f32;

/// One array's quantisation: the values and the scale that maps them back.
pub struct Quantised {
    pub values: Vec<i16>,
    pub scale: f32,
}

pub fn quantise(data: &[f32]) -> Quantised {
    let max = data.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if max > 0.0 { max / Q_MAX } else { 1.0 };
    let values = data
        .iter()
        .map(|&x| (x / scale).round().clamp(-Q_MAX, Q_MAX) as i16)
        .collect();
    Quantised { values, scale }
}

fn dequantise(data: &[f32]) -> Vec<f32> {
    let q = quantise(data);
    q.values.iter().map(|&v| v as f32 * q.scale).collect()
}

/// Which arrays an export ships as int8 (see [`quantise_int8`]), the others as int16: those
/// whose name starts with one of the prefixes (`""` takes every array).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Int8Arrays(pub Vec<String>);

impl Int8Arrays {
    pub fn all() -> Self {
        Self(vec![String::new()])
    }

    pub fn covers(&self, name: &str) -> bool {
        self.0
            .iter()
            .any(|prefix| name.starts_with(prefix.as_str()))
    }
}

/// The int8 rows' scales step by this factor's inverse: `2^(1/8)`, about 9%.
pub const INT8_STEPS_PER_OCTAVE: f32 = 8.0;

/// Int8 quantisation of `rows` rows of `cols` values: the bytes as shipped (each row its
/// exponent byte `e`, then its values as signed bytes), the base scale, and the values as the
/// decoder gets them back. Row `r`'s scale is `base * 2^(-e / 8)`, `2^(k / 8)` for the smallest
/// `k` that covers the row's largest value in 127 steps (so at least about 116 are used), the
/// base being the largest row's scale; `e = 255` for a row of zeros.
pub fn quantise_int8(data: &[f32], cols: usize) -> (Vec<u8>, f32, Vec<f32>) {
    quantise_int8_with(data, cols, None)
}

/// [`quantise_int8`], rounding each row GPTQ's way when `factor` is given: the upper Cholesky
/// factor `U` (`cols x cols`, `U^T U = H^-1`) of the matrix `H` that weighs a row's error
/// `e` as `e^T H e` (the error as the layers that read the row see it). The row's values are
/// rounded in order, each one's rounding error spread over the values not yet rounded so that
/// they make up for it under `H` (Frantar et al., "GPTQ").
pub fn quantise_int8_with(
    data: &[f32],
    cols: usize,
    factor: Option<&[f64]>,
) -> (Vec<u8>, f32, Vec<f32>) {
    if let Some(u) = factor {
        assert_eq!(u.len(), cols * cols, "a factor for rows of {cols}");
    }
    // Each row's scale is 2^(k / 8) for the smallest k that covers its largest value in 127
    // steps (an absolute ladder, as training fake-quantises: see nn::nnue::Qat); the base is
    // the largest row's, and a row's exponent byte says how many rungs below it the row is.
    let rung = |row: &[f32]| {
        let need = row.iter().fold(0.0f32, |m, &x| m.max(x.abs())) / 127.0;
        (need > 0.0).then(|| (need.log2() * INT8_STEPS_PER_OCTAVE).ceil() as i32)
    };
    let top = data.chunks(cols).filter_map(rung).max();
    let base = top.map_or(1.0, |k| 2f32.powf(k as f32 / INT8_STEPS_PER_OCTAVE));
    let mut bytes = Vec::with_capacity(data.len() + data.len() / cols.max(1));
    let mut values = Vec::with_capacity(data.len());
    for row in data.chunks(cols) {
        let e = match (rung(row), top) {
            (Some(k), Some(top)) => (top - k).clamp(0, 254) as u8,
            _ => 255,
        };
        let scale = int8_row_scale(base, e);
        bytes.push(e);
        let mut w: Vec<f64> = row.iter().map(|&x| f64::from(x)).collect();
        for j in 0..cols {
            let q = (w[j] / f64::from(scale)).round().clamp(-127.0, 127.0);
            if let Some(u) = factor {
                let error = (w[j] - q * f64::from(scale)) / u[j * cols + j];
                for k in j + 1..cols {
                    w[k] -= error * u[j * cols + k];
                }
            }
            let q = q as i8;
            bytes.push(q as u8);
            values.push(f32::from(q) * scale);
        }
    }
    (bytes, base, values)
}

/// The lower Cholesky factor `L` of a symmetric positive definite `n x n` matrix
/// (`L L^T = a`), row-major.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            l[i * n + j] = match i == j {
                true => (a[i * n + i] - dot).max(f64::MIN_POSITIVE).sqrt(),
                false => (a[i * n + j] - dot) / l[j * n + j],
            };
        }
    }
    l
}

/// GPTQ's factor for rows weighed by `h` (`n x n`, see [`quantise_int8_with`]): the upper
/// Cholesky factor of `(h + damping)^-1`, the damping 1% of `h`'s mean diagonal.
pub fn gptq_factor(h: &[f64], n: usize) -> Vec<f64> {
    let damping = 0.01 * (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let mut h = h.to_vec();
    for i in 0..n {
        h[i * n + i] += damping;
    }
    // The inverse from the Cholesky factor: L Y = I, then L^T X = Y.
    let l = cholesky(&h, n);
    let mut inverse = vec![0.0; n * n];
    for c in 0..n {
        let mut y = vec![0.0; n];
        for i in 0..n {
            let target = if i == c { 1.0 } else { 0.0 };
            let dot: f64 = (0..i).map(|k| l[i * n + k] * y[k]).sum();
            y[i] = (target - dot) / l[i * n + i];
        }
        for i in (0..n).rev() {
            let dot: f64 = (i + 1..n).map(|k| l[k * n + i] * inverse[k * n + c]).sum();
            inverse[i * n + c] = (y[i] - dot) / l[i * n + i];
        }
    }
    // The upper factor U of the inverse (U^T U = inverse) is the lower one transposed.
    let lower = cholesky(&inverse, n);
    let mut upper = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            upper[i * n + j] = lower[j * n + i];
        }
    }
    upper
}

/// The scale of an int8 row with exponent byte `e`, as the Haskell decoder computes it.
pub fn int8_row_scale(base: f32, e: u8) -> f32 {
    base * 2f32.powf(-f32::from(e) / INT8_STEPS_PER_OCTAVE)
}

/// One array as the Haskell decoder sees it: name, shape, scale, and its bytes: little-endian
/// `Int16` values, or for int8 each row's exponent byte and values (see [`quantise_int8`]),
/// `scale` then being the base scale.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantisedArray {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub scale: f32,
    pub bytes: Vec<u8>,
}

impl Weights {
    /// Rebuilds the weights with `f` applied to every array (the same arrays as
    /// [`Weights::arrays`]).
    pub fn map_arrays(&self, mut f: impl FnMut(&Array<'_>) -> Vec<f32>) -> Self {
        let mut stream = Vec::with_capacity(self.parameters());
        for a in self.arrays() {
            let mapped = f(&a);
            assert_eq!(
                mapped.len(),
                a.data.len(),
                "{} mapped to a different size",
                a.name
            );
            stream.extend(mapped);
        }
        // The arrays are the buffers cut into consecutive pieces, in the same order.
        let mut out = self.clone();
        let mut rest = stream.as_slice();
        for buffer in out.buffers_mut() {
            let (head, tail) = rest.split_at(buffer.len());
            buffer.copy_from_slice(head);
            rest = tail;
        }
        assert!(rest.is_empty(), "every array mapped");
        // A low-rank content table's tokens follow its mapped tensors.
        if let Some(f) = &out.content_factors {
            out.content = f.tokens();
        }
        out
    }

    /// The weights after the round trip of the export's quantisation (int8 for the arrays
    /// `int8` covers, int16 for the others): what the Haskell evaluator computes with.
    pub fn dequantised(&self, int8: &Int8Arrays, gptq: bool) -> Self {
        let factor = self.content_factors();
        self.map_arrays(|a| match int8.covers(&a.name) {
            true => {
                let rounding = gptq.then(|| self.rounding(&a.name, &factor)).flatten();
                quantise_int8_with(a.data, a.cols, rounding).2
            }
            false => dequantise(a.data),
        })
    }

    /// GPTQ's factor for the content arrays (see [`quantise_int8_with`]): a token's error
    /// reaches the accumulator through every board's slice `W` of the first layer and the
    /// forced board's, so `H` is the sum of their `W^T W`. (Weighing the accumulator's
    /// dimensions further by the first hidden layer and the value's skip that read them
    /// measured no better: the largest output deviations over 256 reference positions went
    /// from 0.012 / 0.015 / 0.016 for win/draw/loss, value and policy to 0.016 / 0.016 /
    /// 0.016, against 0.021 / 0.022 / 0.016 rounding to nearest.)
    ///
    /// A low-rank table's `content_low` rows reach the tokens through `proj` (`P`), so theirs
    /// is from `P H Pᵀ`; the second factor, for models that have one.
    fn content_factors(&self) -> (Vec<f64>, Option<Vec<f64>>) {
        let n = self.d_embed;
        let mut h = vec![0.0f64; n * n];
        for w in self.mix.iter().chain(std::iter::once(&self.forced)) {
            for r in 0..w.rows {
                let row = w.row(r);
                for i in 0..n {
                    let x = f64::from(row[i]);
                    for j in 0..n {
                        h[i * n + j] += x * f64::from(row[j]);
                    }
                }
            }
        }
        let low = self.content_factors.as_ref().map(|f| {
            let (r, p) = (f.proj.rows, &f.proj);
            let mut low = vec![0.0f64; r * r];
            for a in 0..r {
                for b in 0..r {
                    let mut sum = 0.0;
                    for i in 0..n {
                        let hp: f64 = (0..n).map(|j| h[i * n + j] * f64::from(p.row(b)[j])).sum();
                        sum += f64::from(p.row(a)[i]) * hp;
                    }
                    low[a * r + b] = sum;
                }
            }
            gptq_factor(&low, r)
        });
        (gptq_factor(&h, n), low)
    }

    /// How array `name` is rounded to int8: GPTQ's way for the content arrays.
    fn rounding<'f>(
        &self,
        name: &str,
        (content, low): &'f (Vec<f64>, Option<Vec<f64>>),
    ) -> Option<&'f [f64]> {
        match name {
            "content_low" => low.as_deref(),
            _ => name.starts_with("content").then_some(content.as_slice()),
        }
    }

    /// Every array quantised, in [`Weights::arrays`] order.
    pub fn quantised(&self, int8: &Int8Arrays, gptq: bool) -> Vec<QuantisedArray> {
        let factor = self.content_factors();
        self.arrays()
            .into_iter()
            .map(|a| {
                let (scale, bytes) = match int8.covers(&a.name) {
                    true => {
                        let rounding = gptq.then(|| self.rounding(&a.name, &factor)).flatten();
                        let (bytes, base, _) = quantise_int8_with(a.data, a.cols, rounding);
                        (base, bytes)
                    }
                    false => {
                        let q = quantise(a.data);
                        (
                            q.scale,
                            q.values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        )
                    }
                };
                QuantisedArray {
                    name: a.name,
                    rows: a.rows,
                    cols: a.cols,
                    scale,
                    bytes,
                }
            })
            .collect()
    }
}

/// How one array fares under quantisation, from its own values (see [`Weights::quant_report`]).
#[derive(Clone, Debug)]
pub struct ArrayReport {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub max: f32,
    pub rms: f32,
    /// RMS of the int16 and the int8 error over the array's RMS.
    pub int16_error: f32,
    pub int8_error: f32,
    /// A row's largest value over its RMS: the median row's and the worst row's.
    pub row_peak_median: f32,
    pub row_peak_max: f32,
    /// The largest column maximum over the median one: well above 1 when a few columns
    /// (inputs) carry much larger weights than the rest and set every row's scale.
    pub column_outlier: f32,
    /// The row with the largest int8 error relative to its own RMS, and that error.
    pub worst_row: usize,
    pub worst_row_error: f32,
    /// The column with the largest int8 error relative to its own RMS, and that error.
    pub worst_column: usize,
    pub worst_column_error: f32,
}

fn rms(xs: impl Iterator<Item = f32>) -> f32 {
    let (sum, n) = xs.fold((0.0f64, 0usize), |(s, n), x| {
        (s + f64::from(x) * f64::from(x), n + 1)
    });
    if n == 0 {
        0.0
    } else {
        (sum / n as f64).sqrt() as f32
    }
}

fn median(mut xs: Vec<f32>) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(f32::total_cmp);
    xs[xs.len() / 2]
}

impl Weights {
    /// Every array's quantisation statistics (see [`ArrayReport`]), in [`Weights::arrays`]
    /// order.
    pub fn quant_report(&self) -> Vec<ArrayReport> {
        self.arrays()
            .into_iter()
            .map(|a| {
                let (rows, cols) = (a.rows, a.cols);
                let int16 = dequantise(a.data);
                let (_, _, int8) = quantise_int8(a.data, cols);
                let base = rms(a.data.iter().copied()).max(f32::MIN_POSITIVE);
                let error = |q: &[f32]| rms(a.data.iter().zip(q).map(|(x, y)| x - y)) / base;
                let row_rms: Vec<f32> = a
                    .data
                    .chunks(cols)
                    .map(|r| rms(r.iter().copied()))
                    .collect();
                let peaks: Vec<f32> = a
                    .data
                    .chunks(cols)
                    .zip(&row_rms)
                    .map(|(r, &m)| {
                        r.iter().fold(0.0f32, |p, &x| p.max(x.abs())) / m.max(f32::MIN_POSITIVE)
                    })
                    .collect();
                let column_max: Vec<f32> = (0..cols)
                    .map(|c| (0..rows).fold(0.0f32, |p, r| p.max(a.data[r * cols + c].abs())))
                    .collect();
                let top_column = column_max.iter().fold(0.0f32, |p, &x| p.max(x));
                let (worst_row, worst_row_error) = a
                    .data
                    .chunks(cols)
                    .zip(int8.chunks(cols))
                    .zip(&row_rms)
                    .map(|((r, q), &m)| {
                        rms(r.iter().zip(q).map(|(x, y)| x - y)) / m.max(f32::MIN_POSITIVE)
                    })
                    .enumerate()
                    .fold(
                        (0, 0.0f32),
                        |best, (i, e)| if e > best.1 { (i, e) } else { best },
                    );
                let (worst_column, worst_column_error) = (0..cols)
                    .map(|c| {
                        let column = (0..rows).map(|r| a.data[r * cols + c]);
                        let error = (0..rows).map(|r| a.data[r * cols + c] - int8[r * cols + c]);
                        rms(error) / rms(column).max(f32::MIN_POSITIVE)
                    })
                    .enumerate()
                    .fold(
                        (0, 0.0f32),
                        |best, (i, e)| if e > best.1 { (i, e) } else { best },
                    );
                ArrayReport {
                    name: a.name,
                    rows,
                    cols,
                    max: a.data.iter().fold(0.0f32, |p, &x| p.max(x.abs())),
                    rms: base,
                    int16_error: error(&int16),
                    int8_error: error(&int8),
                    row_peak_median: median(peaks.clone()),
                    row_peak_max: peaks.iter().fold(0.0f32, |p, &x| p.max(x)),
                    column_outlier: top_column / median(column_max).max(f32::MIN_POSITIVE),
                    worst_row,
                    worst_row_error,
                    worst_column,
                    worst_column_error,
                }
            })
            .collect()
    }
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
pub fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i)) as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64(&[0xff, 0xfe, 0xfd]), "//79");
    }

    /// Standard base64 decoding, for the round-trip test only.
    fn decode(text: &str) -> Vec<u8> {
        let value = |c: u8| ALPHABET.iter().position(|&a| a == c).expect("alphabet") as u32;
        let mut out = Vec::new();
        for quad in text.as_bytes().chunks(4) {
            let pad = quad.iter().filter(|&&c| c == b'=').count();
            let n = quad
                .iter()
                .map(|&c| if c == b'=' { 0 } else { value(c) })
                .fold(0u32, |acc, v| acc << 6 | v);
            let bytes = n.to_be_bytes();
            out.extend_from_slice(&bytes[1..4 - pad]);
        }
        out
    }

    #[test]
    fn stream_holds_every_array_and_survives_base64() {
        let _device = crate::nn::device_lock();
        let device = crate::nn::BackendKind::preferred().device();
        let model = crate::nn::NnueConfig::new()
            .with_d_embed(4)
            .with_d_acc(8)
            .with_d_hidden(4)
            .init(&device);
        let weights = Weights::from_model(&model);
        let arrays = weights.quantised(&Int8Arrays::default(), true);
        let values: usize = arrays.iter().map(|a| a.rows * a.cols).sum();
        assert_eq!(values, weights.parameters());
        for a in &arrays {
            assert_eq!(a.bytes.len(), 2 * a.rows * a.cols, "{}", a.name);
            assert_eq!(decode(&base64(&a.bytes)), a.bytes, "{}", a.name);
        }
        // The first array is the open part of the content table; its first row decodes to the weights.
        let first = &arrays[0];
        assert_eq!(first.name, "content_open");
        for (i, w) in weights.content.row(0).iter().enumerate() {
            let q = i16::from_le_bytes([first.bytes[2 * i], first.bytes[2 * i + 1]]);
            assert!((q as f32 * first.scale - w).abs() <= first.scale / 2.0 + 1e-9);
        }
        assert_eq!(
            weights.dequantised(&Int8Arrays::default(), true).spec,
            weights.spec
        );
    }

    /// `map_arrays` writes every array back where `arrays` read it, for every layer kind and
    /// the head gates.
    #[test]
    fn mapped_arrays_land_in_their_own_buffers() {
        let _device = crate::nn::device_lock();
        let device = crate::nn::BackendKind::preferred().device();
        let v2 = crate::nn::NnueV2Config::new()
            .with_d_embed(4)
            .with_d_acc(8)
            .with_d_hidden(4)
            .with_layers(3);
        let nnue = crate::nn::NnueConfig::new()
            .with_d_embed(4)
            .with_d_acc(8)
            .with_d_hidden(4)
            .with_layers(3)
            .with_reglu(Some(6));
        for weights in [
            Weights::from_v2(&v2.clone().init(&device)),
            Weights::from_v2(&v2.with_reglu(Some(6)).init(&device)),
            Weights::from_model(&nnue.init(&device)),
        ] {
            let mut index = 0.0;
            let tagged = weights.map_arrays(|a| {
                index += 1.0;
                vec![index; a.data.len()]
            });
            for (i, a) in tagged.arrays().iter().enumerate() {
                assert!(
                    a.data.iter().all(|&x| x == (i + 1) as f32),
                    "{}: {}",
                    weights.spec,
                    a.name
                );
            }
        }
    }

    #[test]
    fn int8_rows_round_trip_within_their_own_step() {
        // Rows of very different magnitudes keep their own precision.
        let data = [
            1.0f32, -0.5, 0.25, 0.0, 0.001, -0.002, 0.0005, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let (bytes, base, values) = quantise_int8(&data, 4);
        assert_eq!(bytes.len(), 3 * 5);
        // The largest row's scale: the ladder's rung at or just above 1 / 127.
        assert!(
            base >= 1.0 / 127.0 && base < 1.0 / 127.0 * 2f32.powf(0.125),
            "{base}"
        );
        for (r, row) in data.chunks(4).enumerate() {
            let e = bytes[r * 5];
            let scale = int8_row_scale(base, e);
            let need = row.iter().fold(0.0f32, |m, &x| m.max(x.abs())) / 127.0;
            // The scale covers the row, within one step of 2^(1/8).
            assert!(scale >= need && (need == 0.0 || scale <= need * 2f32.powf(0.125) * 1.0001));
            for (c, &x) in row.iter().enumerate() {
                let q = bytes[r * 5 + 1 + c] as i8;
                assert_eq!(values[r * 4 + c], f32::from(q) * scale);
                assert!((x - values[r * 4 + c]).abs() <= scale / 2.0 + 1e-9, "{x}");
            }
        }
        assert_eq!(bytes[10], 255, "a row of zeros");
    }

    #[test]
    fn quantisation_round_trips_within_a_step() {
        let data = [0.5f32, -1.0, 0.001, 0.0, 0.25];
        let q = quantise(&data);
        assert_eq!(q.values[1], -32767);
        for (x, &v) in data.iter().zip(&q.values) {
            assert!((x - v as f32 * q.scale).abs() <= q.scale / 2.0 + 1e-9);
        }
        assert_eq!(quantise(&[0.0, 0.0]).values, [0, 0]);
    }
}
