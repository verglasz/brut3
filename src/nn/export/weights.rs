//! The trained tensors on the host, in the layout the Haskell code uses.
//!
//! Burn's `Linear` stores `[d_in, d_out]` and applies `x · W`; here every matrix is stored
//! transposed, one row per output, so that a layer is one dot product per output row. The
//! lookup tables (contents, features, local and send policy) are rows per id already and are
//! kept as they are.

use std::ops::Range;

use burn::prelude::*;

use crate::nn::model::MOVES;
use crate::nn::nnue::{
    BOARDS,
    CELLS,
    CONTENT_ROWS,
    CONTENTS,
    FEATURE_ROWS,
    IDS_PER_BOARD,
    Nnue,
    NnueConfig,
    OPEN_CONTENTS,
};

/// A row-major matrix of `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data for {rows}x{cols}");
        Self { rows, cols, data }
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn transposed(&self) -> Self {
        let mut data = vec![0.0; self.data.len()];
        for r in 0..self.rows {
            for c in 0..self.cols {
                data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Self::new(self.cols, self.rows, data)
    }

    /// Rows `range` as their own matrix.
    pub fn rows(&self, range: std::ops::Range<usize>) -> Self {
        let data = self.data[range.start * self.cols..range.end * self.cols].to_vec();
        Self::new(range.len(), self.cols, data)
    }

    fn from_tensor(t: Tensor<2>) -> Self {
        let [rows, cols] = t.dims();
        Self::new(rows, cols, host(t.into_data()))
    }
}

/// A dense layer: `weight` has one row per output; `bias` is empty when the layer has none.
#[derive(Clone, Debug, PartialEq)]
pub struct Dense {
    pub weight: Matrix,
    pub bias: Vec<f32>,
}

impl Dense {
    fn from_linear(linear: &burn::nn::Linear) -> Self {
        Self {
            weight: Matrix::from_tensor(linear.weight.val()).transposed(),
            bias: linear
                .bias
                .as_ref()
                .map(|b| host(b.val().into_data()))
                .unwrap_or_default(),
        }
    }
}

fn host(data: TensorData) -> Vec<f32> {
    data.try_into_vec::<f32>().expect("f32 tensor")
}

/// Everything the Haskell evaluator needs, extracted from a trained model.
#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    pub config: NnueConfig,
    /// `[CONTENT_ROWS, d_embed]`: the content embedding, cell rows and padding.
    pub content: Matrix,
    /// Nine `[d_acc, d_embed]` matrices: board `k`'s slice of the first layer, transposed.
    pub mix: Vec<Matrix>,
    /// `[d_acc]`: the first layer's bias.
    pub mix_bias: Vec<f32>,
    /// `[d_acc, d_embed]`: the forced board's slice of the first layer, transposed.
    pub forced: Matrix,
    /// `[FEATURE_ROWS, d_acc]`: feature rows, added to the accumulator by id.
    pub features: Matrix,
    /// `[d_hidden, d_acc]` then `[d_hidden, d_hidden]` layers, each with a bias.
    pub hidden: Vec<Dense>,
    /// `[3, d_hidden]` with bias: the win/draw/loss head.
    pub value: Dense,
    /// `[3, d_acc]`: the value head's linear skip from the accumulator (no bias).
    pub skip: Matrix,
    /// `[81, d_hidden]` with bias: the global policy head.
    pub policy: Dense,
    /// `[CONTENTS, 9]`: each content's own cell logits.
    pub local: Matrix,
    /// `[CONTENTS]`: the logit of sending the opponent to a board with each content.
    pub send: Vec<f32>,
}

impl Weights {
    pub fn from_model(model: &Nnue) -> Self {
        let config = model.config();
        let d_embed = config.d_embed;
        let mix = Dense::from_linear(&model.mix);
        let mix_t = mix.weight.transposed(); // [9 * d_embed, d_acc] again
        let boards = (0..BOARDS)
            .map(|k| mix_t.rows(k * d_embed..(k + 1) * d_embed).transposed())
            .collect();
        let weights = Self {
            content: Matrix::from_tensor(model.content.weight.val()),
            mix: boards,
            mix_bias: mix.bias,
            forced: Dense::from_linear(&model.forced_mix).weight,
            features: Matrix::from_tensor(model.features.weight.val()),
            hidden: model.hidden.iter().map(Dense::from_linear).collect(),
            value: Dense::from_linear(&model.value),
            skip: Dense::from_linear(&model.value_skip).weight,
            policy: Dense::from_linear(&model.policy),
            local: Matrix::from_tensor(model.local_policy.weight.val()),
            send: host(model.send_policy.weight.val().into_data()),
            config,
        };
        weights.check();
        weights
    }

    /// Asserts every shape against the config.
    fn check(&self) {
        let c = &self.config;
        let shape = |m: &Matrix, rows: usize, cols: usize, name: &str| {
            assert_eq!((m.rows, m.cols), (rows, cols), "shape of {name}");
        };
        shape(&self.content, CONTENT_ROWS, c.d_embed, "content");
        assert_eq!(self.mix.len(), BOARDS);
        for (k, m) in self.mix.iter().enumerate() {
            shape(m, c.d_acc, c.d_embed, &format!("mix[{k}]"));
        }
        assert_eq!(self.mix_bias.len(), c.d_acc, "mix bias");
        shape(&self.forced, c.d_acc, c.d_embed, "forced");
        shape(&self.features, FEATURE_ROWS, c.d_acc, "features");
        assert_eq!(self.hidden.len(), c.layers, "hidden layers");
        for (i, layer) in self.hidden.iter().enumerate() {
            let d_in = if i == 0 { c.d_acc } else { c.d_hidden };
            shape(&layer.weight, c.d_hidden, d_in, &format!("hidden[{i}]"));
            assert_eq!(layer.bias.len(), c.d_hidden, "hidden[{i}] bias");
        }
        shape(&self.value.weight, 3, c.d_hidden, "value");
        assert_eq!(self.value.bias.len(), 3, "value bias");
        shape(&self.skip, 3, c.d_acc, "skip");
        shape(&self.policy.weight, MOVES, c.d_hidden, "policy");
        assert_eq!(self.policy.bias.len(), MOVES, "policy bias");
        shape(&self.local, CONTENTS, CELLS, "local policy");
        assert_eq!(self.send.len(), CONTENTS, "send policy");
        debug_assert_eq!(IDS_PER_BOARD, 1 + CELLS);
    }

    /// The spec string of the model, e.g. `nnue 64 128 32 2`.
    pub fn config_display(&self) -> String {
        let c = &self.config;
        format!("nnue {} {} {} {}", c.d_embed, c.d_acc, c.d_hidden, c.layers)
    }

    /// Number of trained parameters.
    pub fn parameters(&self) -> usize {
        self.arrays().iter().map(|a| a.data.len()).sum()
    }

    /// Every array in a fixed order: the matrices one row per output, the biases as one row.
    /// The per-content tables are cut after the open contents: `_open` holds those rows and
    /// `_rest` the few rows after them (the decided kinds, and for the content table also
    /// no-forced, the cell rows and the pad), so the decoder never splits a long list.
    pub fn arrays(&self) -> Vec<Array<'_>> {
        fn m<'a>(name: &str, m: &'a Matrix) -> Array<'a> {
            Array {
                name: name.to_string(),
                rows: m.rows,
                cols: m.cols,
                data: &m.data,
            }
        }
        fn v<'a>(name: &str, v: &'a [f32]) -> Array<'a> {
            Array {
                name: name.to_string(),
                rows: 1,
                cols: v.len(),
                data: v,
            }
        }
        fn part<'a>(name: &str, m: &'a Matrix, rows: Range<usize>) -> Array<'a> {
            Array {
                name: name.to_string(),
                rows: rows.len(),
                cols: m.cols,
                data: &m.data[rows.start * m.cols..rows.end * m.cols],
            }
        }
        let (open, decided) = (0..OPEN_CONTENTS, OPEN_CONTENTS..CONTENTS);
        let mut out = vec![
            part("content_open", &self.content, open.clone()),
            part("content_rest", &self.content, OPEN_CONTENTS..CONTENT_ROWS),
        ];
        for (k, w) in self.mix.iter().enumerate() {
            out.push(m(&format!("mix{k}"), w));
        }
        out.push(v("mix_bias", &self.mix_bias));
        out.push(m("forced", &self.forced));
        out.push(m("features", &self.features));
        for (i, layer) in self.hidden.iter().enumerate() {
            out.push(m(&format!("hidden{i}"), &layer.weight));
            out.push(v(&format!("hidden{i}_bias"), &layer.bias));
        }
        out.push(m("value", &self.value.weight));
        out.push(v("value_bias", &self.value.bias));
        out.push(m("skip", &self.skip));
        out.push(m("policy", &self.policy.weight));
        out.push(v("policy_bias", &self.policy.bias));
        out.push(part("local_open", &self.local, open.clone()));
        out.push(part("local_rest", &self.local, decided.clone()));
        // One value per content: a column, not a row.
        out.push(Array {
            name: "send_open".into(),
            rows: open.len(),
            cols: 1,
            data: &self.send[open],
        });
        out.push(Array {
            name: "send_rest".into(),
            rows: decided.len(),
            cols: 1,
            data: &self.send[decided],
        });
        out
    }
}

/// A named view of one weight array: `rows x cols`, row-major.
#[derive(Clone, Debug)]
pub struct Array<'a> {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [f32],
}
