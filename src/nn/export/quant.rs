//! Int16 quantisation of the weights and their base64 text for `Gen/Weights.hs`.
//!
//! Each array gets one scale, `max |w| / 32767`; the Haskell decoder multiplies the `Int16`
//! back into `Float`. Every array of [`Weights::arrays`] is shipped on its own: row-major,
//! little-endian `Int16`, base64-encoded with the standard alphabet and padding.

use super::weights::{Array, Dense, Matrix, Weights};

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

/// One array as the Haskell decoder sees it: name, shape, scale, and its values as
/// little-endian `Int16` bytes.
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
        let mut arrays = self.arrays().into_iter();
        let mut next = |expect: &str| {
            let a = arrays.next().expect("array");
            assert_eq!(a.name, expect, "array order");
            (a.rows, a.cols, f(&a))
        };
        // A table shipped in parts: the parts' rows stacked back together.
        let mut stacked = |names: &[&str]| {
            let parts: Vec<_> = names.iter().map(|&name| next(name)).collect();
            let rows = parts.iter().map(|&(rows, _, _)| rows).sum();
            Matrix::new(
                rows,
                parts[0].1,
                parts.into_iter().flat_map(|p| p.2).collect(),
            )
        };
        let content = stacked(&[
            "content_open",
            "content_decided",
            "no_forced",
            "cells",
            "cell_pad",
        ]);
        let mut matrix = |name: &str| stacked(&[name]);
        let mix: Vec<Matrix> = (0..self.mix.len())
            .map(|k| matrix(&format!("mix{k}")))
            .collect();
        let mix_bias = matrix("mix_bias").data;
        let forced = matrix("forced");
        let features = matrix("features");
        let hidden = (0..self.hidden.len())
            .map(|i| Dense {
                weight: matrix(&format!("hidden{i}")),
                bias: matrix(&format!("hidden{i}_bias")).data,
            })
            .collect();
        let value = Dense {
            weight: matrix("value"),
            bias: matrix("value_bias").data,
        };
        let skip = matrix("skip");
        let policy = Dense {
            weight: matrix("policy"),
            bias: matrix("policy_bias").data,
        };
        let local = stacked(&["local_open", "local_decided"]);
        let send = stacked(&["send_open", "send_decided"]).data;
        assert!(arrays.next().is_none(), "every array mapped");
        Self {
            config: self.config.clone(),
            content,
            mix,
            mix_bias,
            forced,
            features,
            hidden,
            value,
            skip,
            policy,
            local,
            send,
        }
    }

    /// The weights after an int16 round trip: what the Haskell evaluator computes with.
    pub fn dequantised(&self) -> Self {
        self.map_arrays(|a| dequantise(a.data))
    }

    /// Every array quantised, in [`Weights::arrays`] order.
    pub fn quantised(&self) -> Vec<QuantisedArray> {
        self.arrays()
            .into_iter()
            .map(|a| {
                let q = quantise(a.data);
                QuantisedArray {
                    name: a.name,
                    rows: a.rows,
                    cols: a.cols,
                    scale: q.scale,
                    bytes: q.values.iter().flat_map(|v| v.to_le_bytes()).collect(),
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
        let arrays = weights.quantised();
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
        assert_eq!(weights.dequantised().config, weights.config);
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
