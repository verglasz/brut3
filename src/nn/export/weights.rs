//! The trained tensors on the host, in the layout the Haskell code uses.
//!
//! Burn's `Linear` stores `[d_in, d_out]` and applies `x · W`; here every matrix is stored
//! transposed, one row per output, so that a layer is one dot product per output row. The
//! lookup tables (contents, features, local and send policy) are rows per id already and are
//! kept as they are. Both NNUE kinds export: their hidden layers as a list of [`Layer`]s,
//! and the head gates of [`NnueV2`] when present.

use std::ops::Range;

use burn::prelude::*;

use super::codegen::CHUNK;
use crate::game::LINES;
use crate::nn::AnyModel;
use crate::nn::model::MOVES;
use crate::nn::nnue::{
    BOARDS,
    CELL_BASE,
    CELL_PAD,
    CELLS,
    CONTENT_ROWS,
    CONTENTS,
    DECIDED_BASE,
    Ends,
    FEATURE_ROWS,
    FORCED_BASE,
    GATES,
    IDS_PER_BOARD,
    LINE_BASE,
    LINE_STATES,
    NO_FORCED,
    Nnue,
    NnueNet,
    OPEN_CONTENTS,
    OPEN_MARKS_BASE,
    ReGlu,
};
use crate::nn::nnue_v2::NnueV2;

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

impl Dense {
    /// The layer with weight and bias multiplied by `s` (a ReZero scale folded in).
    fn scaled(mut self, s: f32) -> Self {
        self.weight.data.iter_mut().for_each(|w| *w *= s);
        self.bias.iter_mut().for_each(|b| *b *= s);
        self
    }
}

/// One hidden layer, from its input `x`. ReZero scales are folded into the matrices: the
/// export computes the same function without them.
#[derive(Clone, Debug, PartialEq)]
pub enum Layer {
    /// `relu(W x + b)`: the NNUE's first layer, from the accumulator.
    Dense(Dense),
    /// `W x + b`: nnue-v2's projection from the accumulator.
    Linear(Dense),
    /// `relu(x + W x + b)`.
    Residual(Dense),
    /// `relu(x + W relu(x) + b)`: nnue-v2's first layer without ReGLU.
    ReluResidual(Dense),
    /// `x + D (relu(x) * (U x + b))`: nnue-v2's first layer with ReGLU, gated by its own
    /// input. `D` has no bias.
    SelfGated { up: Dense, down: Matrix },
    /// `x + D (relu(G x + b_G) * (U x + b_U))`. `D` has no bias.
    ReGlu {
        gate: Dense,
        up: Dense,
        down: Matrix,
    },
}

impl Layer {
    /// The layer's arrays as `(suffix, rows, cols, data)`, appended to `hidden{i}`.
    fn parts(&self) -> Vec<(&'static str, &Matrix, Option<&[f32]>)> {
        match self {
            Layer::Dense(d) | Layer::Linear(d) | Layer::Residual(d) | Layer::ReluResidual(d) => {
                vec![("", &d.weight, Some(&d.bias))]
            }
            Layer::SelfGated { up, down } => {
                vec![("_up", &up.weight, Some(&up.bias)), ("_down", down, None)]
            }
            Layer::ReGlu { gate, up, down } => vec![
                ("_gate", &gate.weight, Some(&gate.bias)),
                ("_up", &up.weight, Some(&up.bias)),
                ("_down", down, None),
            ],
        }
    }

    /// The layer's buffers, in [`Layer::parts`] order (each matrix, then its bias).
    fn buffers_mut(&mut self) -> Vec<&mut Vec<f32>> {
        match self {
            Layer::Dense(d) | Layer::Linear(d) | Layer::Residual(d) | Layer::ReluResidual(d) => {
                vec![&mut d.weight.data, &mut d.bias]
            }
            Layer::SelfGated { up, down } => {
                vec![&mut up.weight.data, &mut up.bias, &mut down.data]
            }
            Layer::ReGlu { gate, up, down } => vec![
                &mut gate.weight.data,
                &mut gate.bias,
                &mut up.weight.data,
                &mut up.bias,
                &mut down.data,
            ],
        }
    }

    /// Multiply-adds of one evaluation.
    pub fn macs(&self) -> usize {
        self.parts().iter().map(|(_, m, _)| m.rows * m.cols).sum()
    }
}

/// Everything the Haskell evaluator needs, extracted from a trained model.
#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    /// The model's spec, e.g. `nnue 64 128 32 2` or `nnue-v2 64 512 96 4 reglu 96`.
    pub spec: String,
    pub d_embed: usize,
    pub d_acc: usize,
    pub d_hidden: usize,
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
    /// The hidden layers, the first from the accumulator.
    pub trunk: Vec<Layer>,
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
    /// `[GATES, d_hidden]` with bias: the head gates (see [`GATES`]), for models that have
    /// them. Through a sigmoid, row 0 mixes the value, rows `1 + b` scale board `b`'s local
    /// logits and rows `10 + s` the send logit of board `s`.
    pub gates: Option<Dense>,
}

/// The layers every NNUE model has around its trunk, from the model's [`Ends`].
struct Shared {
    content: Matrix,
    mix: Vec<Matrix>,
    mix_bias: Vec<f32>,
    forced: Matrix,
    features: Matrix,
    value: Dense,
    skip: Matrix,
    policy: Dense,
    local: Matrix,
    send: Vec<f32>,
}

impl Shared {
    fn new(e: &Ends) -> Self {
        let [_, d_embed] = e.content.weight.dims();
        let mix = Dense::from_linear(e.mix);
        let mix_t = mix.weight.transposed(); // [9 * d_embed, d_acc] again
        Self {
            content: Matrix::from_tensor(e.content.weight.val()),
            mix: (0..BOARDS)
                .map(|k| mix_t.rows(k * d_embed..(k + 1) * d_embed).transposed())
                .collect(),
            mix_bias: mix.bias,
            forced: Dense::from_linear(e.forced_mix).weight,
            features: Matrix::from_tensor(e.features.weight.val()),
            value: Dense::from_linear(e.value),
            skip: Dense::from_linear(e.value_skip).weight,
            policy: Dense::from_linear(e.policy),
            local: Matrix::from_tensor(e.local_policy.weight.val()),
            send: host(e.send_policy.weight.val().into_data()),
        }
    }
}

impl Weights {
    fn assemble(
        spec: String,
        [d_embed, d_acc, d_hidden]: [usize; 3],
        e: &Ends,
        trunk: Vec<Layer>,
    ) -> Self {
        let s = Shared::new(e);
        let weights = Self {
            spec,
            d_embed,
            d_acc,
            d_hidden,
            content: s.content,
            mix: s.mix,
            mix_bias: s.mix_bias,
            forced: s.forced,
            features: s.features,
            trunk,
            value: s.value,
            skip: s.skip,
            policy: s.policy,
            local: s.local,
            send: s.send,
            gates: e.gates.map(Dense::from_linear),
        };
        weights.check();
        weights
    }

    pub fn from_model(model: &Nnue) -> Self {
        let c = model.config();
        let mut trunk = vec![Layer::Dense(Dense::from_linear(&model.hidden[0]))];
        trunk.extend(
            model.hidden[1..]
                .iter()
                .map(|l| Layer::Residual(Dense::from_linear(l))),
        );
        trunk.extend(model.glu.iter().map(|b| reglu(b, 1.0)));
        let mut spec = format!("nnue {} {} {} {}", c.d_embed, c.d_acc, c.d_hidden, c.layers);
        if let Some(d_ff) = c.reglu {
            spec += &format!(" reglu {d_ff}");
        }
        Self::assemble(spec, [c.d_embed, c.d_acc, c.d_hidden], &model.ends(), trunk)
    }

    pub fn from_v2(model: &NnueV2) -> Self {
        let c = model.config();
        // Layer `i` after the projection scales its branch by its ReZero scalar, if any.
        let alphas: Vec<f32> = model.rezero.iter().map(|a| a.val().into_scalar()).collect();
        let alpha = |i: usize| alphas.get(i).copied().unwrap_or(1.0);
        let mut trunk = vec![Layer::Linear(Dense::from_linear(&model.proj))];
        let first = Dense::from_linear(&model.first);
        trunk.push(match &model.first_down {
            Some(down) => Layer::SelfGated {
                up: first,
                down: Dense::from_linear(down).scaled(alpha(0)).weight,
            },
            None => Layer::ReluResidual(first.scaled(alpha(0))),
        });
        trunk.extend(
            model
                .hidden
                .iter()
                .enumerate()
                .map(|(i, l)| Layer::Residual(Dense::from_linear(l).scaled(alpha(1 + i)))),
        );
        trunk.extend(
            model
                .glu
                .iter()
                .enumerate()
                .map(|(i, b)| reglu(b, alpha(1 + i))),
        );
        let mut spec = format!(
            "nnue-v2 {} {} {} {}",
            c.d_embed, c.d_acc, c.d_hidden, c.layers
        );
        if let Some(d_ff) = c.reglu {
            spec += &format!(" reglu {d_ff}");
        }
        Self::assemble(spec, [c.d_embed, c.d_acc, c.d_hidden], &model.ends(), trunk)
    }

    /// The weights of an exportable model: an NNUE of either kind.
    pub fn from_any(model: &AnyModel) -> Result<Self, String> {
        match model {
            AnyModel::Nnue(m) => Ok(Self::from_model(m)),
            AnyModel::NnueV2(m) => Ok(Self::from_v2(m)),
            other => Err(format!("a {} model, not an nnue", other.spec())),
        }
    }

    /// Asserts every shape against the widths.
    fn check(&self) {
        let (e, a, h) = (self.d_embed, self.d_acc, self.d_hidden);
        let shape = |m: &Matrix, rows: usize, cols: usize, name: &str| {
            assert_eq!((m.rows, m.cols), (rows, cols), "shape of {name}");
        };
        let dense = |d: &Dense, rows: usize, cols: usize, name: &str| {
            shape(&d.weight, rows, cols, name);
            assert_eq!(d.bias.len(), rows, "{name} bias");
        };
        shape(&self.content, CONTENT_ROWS, e, "content");
        assert_eq!(self.mix.len(), BOARDS);
        for (k, m) in self.mix.iter().enumerate() {
            shape(m, a, e, &format!("mix[{k}]"));
        }
        assert_eq!(self.mix_bias.len(), a, "mix bias");
        shape(&self.forced, a, e, "forced");
        shape(&self.features, FEATURE_ROWS, a, "features");
        for (i, layer) in self.trunk.iter().enumerate() {
            let name = format!("hidden[{i}]");
            match layer {
                Layer::Dense(d) | Layer::Linear(d) => {
                    assert_eq!(i, 0, "{name} reads the accumulator");
                    dense(d, h, a, &name);
                }
                Layer::Residual(d) | Layer::ReluResidual(d) => dense(d, h, h, &name),
                Layer::SelfGated { up, down } => {
                    dense(up, h, h, &name);
                    shape(down, h, h, &name);
                }
                Layer::ReGlu { gate, up, down } => {
                    let d_ff = gate.weight.rows;
                    dense(gate, d_ff, h, &name);
                    dense(up, d_ff, h, &name);
                    shape(down, h, d_ff, &name);
                }
            }
            if i > 0 {
                assert!(
                    !matches!(layer, Layer::Dense(_) | Layer::Linear(_)),
                    "{name} reads the accumulator"
                );
            }
        }
        assert!(!self.trunk.is_empty(), "hidden layers");
        dense(&self.value, 3, h, "value");
        shape(&self.skip, 3, a, "skip");
        dense(&self.policy, MOVES, h, "policy");
        shape(&self.local, CONTENTS, CELLS, "local policy");
        assert_eq!(self.send.len(), CONTENTS, "send policy");
        if let Some(g) = &self.gates {
            dense(g, GATES, h, "gates");
        }
        debug_assert_eq!(IDS_PER_BOARD, 1 + CELLS);
    }

    /// The spec string of the model, e.g. `nnue 64 128 32 2`.
    pub fn config_display(&self) -> String {
        self.spec.clone()
    }

    /// The inner width of the ReGLU blocks, if there are any.
    pub fn d_ff(&self) -> Option<usize> {
        self.trunk.iter().find_map(|l| match l {
            Layer::ReGlu { gate, .. } => Some(gate.weight.rows),
            _ => None,
        })
    }

    /// Number of trained parameters (with ReZero, its scalars are folded away).
    pub fn parameters(&self) -> usize {
        self.arrays().iter().map(|a| a.data.len()).sum()
    }

    /// Multiply-adds of one evaluation once the first layer is folded into tables: the hidden
    /// layers, the value head and its skip, the policy over all 81 moves and the gates.
    pub fn folded_macs(&self) -> usize {
        let h = self.d_hidden;
        let gates = if self.gates.is_some() { GATES * h } else { 0 };
        self.trunk.iter().map(Layer::macs).sum::<usize>()
            + 3 * h
            + 3 * self.d_acc
            + MOVES * h
            + gates
    }

    /// Every array in a fixed order: the matrices one row per output, the biases as one row.
    /// The tables are cut into the pieces the Haskell record holds, so that its decoder never
    /// splits a long list: the per-content tables into `_open` (the open contents) and
    /// `_decided` (the three decided kinds), the content table also into its no-forced row,
    /// cell rows and pad; the first layer's slices into pieces of [`CHUNK`] rows
    /// (`mix{k}_{p}`, `forced_{p}`); the feature rows into their four tables, one per line.
    /// A hidden layer's arrays are `hidden{i}` and `hidden{i}_bias`, or for a gated one
    /// `hidden{i}_gate` (and bias), `hidden{i}_up` (and bias) and `hidden{i}_down`; the head
    /// gates, if any, come last.
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
        // Pieces of CHUNK rows (the last one shorter if the width is not a multiple).
        fn pieces<'a>(name: &str, m: &'a Matrix) -> Vec<Array<'a>> {
            (0..m.rows.div_ceil(CHUNK))
                .map(|p| {
                    let rows = p * CHUNK..((p + 1) * CHUNK).min(m.rows);
                    part(&format!("{name}_{p}"), m, rows)
                })
                .collect()
        }
        let (open, decided) = (0..OPEN_CONTENTS, OPEN_CONTENTS..CONTENTS);
        let mut out = vec![
            part("content_open", &self.content, open.clone()),
            part("content_decided", &self.content, decided.clone()),
            part("content_no_forced", &self.content, NO_FORCED..NO_FORCED + 1),
            part("content_cells", &self.content, CELL_BASE..CELL_PAD),
            part("content_pad", &self.content, CELL_PAD..CONTENT_ROWS),
        ];
        for (k, w) in self.mix.iter().enumerate() {
            out.extend(pieces(&format!("mix{k}"), w));
        }
        out.push(v("mix_bias", &self.mix_bias));
        out.extend(pieces("forced", &self.forced));
        let features = |name: &str, rows: Range<usize>| part(name, &self.features, rows);
        out.push(features("features_forced", FORCED_BASE..LINE_BASE));
        for l in 0..LINES.len() {
            let start = LINE_BASE + l * LINE_STATES;
            out.push(features(
                &format!("features_line{l}"),
                start..start + LINE_STATES,
            ));
        }
        out.push(features("features_decided", DECIDED_BASE..OPEN_MARKS_BASE));
        out.push(features("features_marks", OPEN_MARKS_BASE..FEATURE_ROWS));
        for (i, layer) in self.trunk.iter().enumerate() {
            for (suffix, weight, bias) in layer.parts() {
                out.push(m(&format!("hidden{i}{suffix}"), weight));
                if let Some(bias) = bias {
                    out.push(v(&format!("hidden{i}{suffix}_bias"), bias));
                }
            }
        }
        out.push(m("value", &self.value.weight));
        out.push(v("value_bias", &self.value.bias));
        out.push(m("skip", &self.skip));
        out.push(m("policy", &self.policy.weight));
        out.push(v("policy_bias", &self.policy.bias));
        out.push(part("local_open", &self.local, open.clone()));
        out.push(part("local_decided", &self.local, decided.clone()));
        // One value per content, shipped as one row (so int8 gives it one scale).
        out.push(v("send_open", &self.send[open]));
        out.push(v("send_decided", &self.send[decided]));
        if let Some(g) = &self.gates {
            out.push(m("gates", &g.weight));
            out.push(v("gates_bias", &g.bias));
        }
        out
    }

    /// Every buffer [`Weights::arrays`] reads, in its order: the arrays are these buffers cut
    /// into consecutive pieces.
    pub(super) fn buffers_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut out = vec![&mut self.content.data];
        out.extend(self.mix.iter_mut().map(|m| &mut m.data));
        out.extend([
            &mut self.mix_bias,
            &mut self.forced.data,
            &mut self.features.data,
        ]);
        for layer in &mut self.trunk {
            out.extend(layer.buffers_mut());
        }
        out.extend([
            &mut self.value.weight.data,
            &mut self.value.bias,
            &mut self.skip.data,
            &mut self.policy.weight.data,
            &mut self.policy.bias,
            &mut self.local.data,
            &mut self.send,
        ]);
        if let Some(g) = &mut self.gates {
            out.extend([&mut g.weight.data, &mut g.bias]);
        }
        out
    }
}

/// A ReGLU block, its down projection scaled by `alpha`.
fn reglu(block: &ReGlu, alpha: f32) -> Layer {
    Layer::ReGlu {
        gate: Dense::from_linear(&block.gate),
        up: Dense::from_linear(&block.up),
        down: Dense::from_linear(&block.down).scaled(alpha).weight,
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
