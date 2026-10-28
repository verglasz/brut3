//! [`NnueV2`]: the NNUE's first layer and heads (see [`super::nnue`]) around reworked hidden
//! layers. The accumulator's projection to the hidden width is left unrectified, `h0 = W acc +
//! b`, and the first hidden layer reads it both ways:
//!
//! - with ReGLU (`reglu = Some(d_ff)`), `h = h0 + W_down (ReLU(h0) * (W_up h0 + b))`: the
//!   projection is the gate itself, so this block has no gate matrix and its inner width is
//!   the hidden width; the later layers are ordinary [`ReGlu`] blocks of inner width `d_ff`;
//! - without, `h = ReLU(h0 + W ReLU(h0) + b)`, the later layers residual,
//!   `h <- ReLU(h + W h + b)`.
//!
//! Neither `W_down` has a bias: the stream is only ever read by affine maps next.
//!
//! With `rezero`, every layer after the projection scales its branch by a learned scalar
//! that starts at 0 (Bachlechner et al., "ReZero is all you need"): `h + a Block(h)`, or
//! `ReLU(h + a (W h + b))` for the residual layers, so that the network starts as the
//! projection alone and each layer's contribution grows as training wants it.
//!
//! The heads are the NNUE's with gates (see [`GATES`]): the stream mixes the value's own
//! logits with the accumulator's skip, `(1 − g) V h + g S acc`, and scales each board's local
//! policy table and each destination's send table by a gate of its own, so that no shortcut
//! reaches the output unconditionally. The gates start with zero weights and biases, at 1/2
//! for every position; weights saved before the gates load with them so and the gated terms
//! doubled, which computes what those weights did (see [`NnueV2::load_weights`]).

use std::path::Path;

use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::store::{ModuleRecord, RecordError};
use burn::tensor::activation::relu;

use super::model::{MOVES, Model, Prediction};
use super::nnue::{
    EndLayers,
    Ends,
    FoldedNnue,
    GATES,
    NnueInput,
    NnueNet,
    ReGlu,
    fold,
    nnue_input,
    predict,
};
use crate::game::BitState;

#[derive(Config, Debug, PartialEq)]
pub struct NnueV2Config {
    /// Width of the content embedding (the rank of the factored first layer).
    #[config(default = 64)]
    pub d_embed: usize,
    /// Width of the accumulator (the folded tables' rows).
    #[config(default = 256)]
    pub d_acc: usize,
    /// Width of the hidden state.
    #[config(default = 32)]
    pub d_hidden: usize,
    /// Hidden layers: the projection from the accumulator, the first layer, then the others.
    #[config(default = 2)]
    pub layers: usize,
    /// Inner width of the ReGLU blocks after the first layer; `None` for residual layers.
    pub reglu: Option<usize>,
    /// ReZero: each layer after the projection scales its branch by a learned scalar
    /// starting at 0.
    #[config(default = false)]
    pub rezero: bool,
}

impl NnueV2Config {
    pub fn init(&self, device: &Device) -> NnueV2 {
        assert!(
            self.layers >= 2,
            "the projection and at least one hidden layer"
        );
        let (d_h, later) = (self.d_hidden, self.layers - 2);
        let linear = |d_in, d_out| LinearConfig::new(d_in, d_out).init(device);
        let (first_down, hidden, glu) = match self.reglu {
            Some(d_ff) => (
                Some(LinearConfig::new(d_h, d_h).with_bias(false).init(device)),
                Vec::new(),
                (0..later).map(|_| ReGlu::new(d_h, d_ff, device)).collect(),
            ),
            None => (
                None,
                (0..later).map(|_| linear(d_h, d_h)).collect(),
                Vec::new(),
            ),
        };
        let rezero = match self.rezero {
            true => (1..self.layers)
                .map(|_| Param::from_tensor(Tensor::zeros([1], device)))
                .collect(),
            false => Vec::new(),
        };
        let gates = LinearConfig::new(d_h, GATES)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let e = EndLayers::new(self.d_embed, self.d_acc, d_h, device);
        NnueV2 {
            config: self.clone(),
            content: e.content,
            mix: e.mix,
            forced_mix: e.forced_mix,
            features: e.features,
            proj: linear(self.d_acc, d_h),
            first: linear(d_h, d_h),
            first_down,
            hidden,
            glu,
            rezero,
            value: e.value,
            value_skip: e.value_skip,
            policy: e.policy,
            local_policy: e.local_policy,
            send_policy: e.send_policy,
            gates,
        }
    }
}

#[derive(Module, Debug)]
pub struct NnueV2 {
    #[module(skip)]
    config: NnueV2Config,
    pub content: burn::nn::Embedding,
    pub mix: Linear,
    pub forced_mix: Linear,
    pub features: burn::nn::Embedding,
    /// `acc -> h0`, without a ReLU.
    pub proj: Linear,
    /// The first layer's matrix: `W_up` of the gated layer, or `W` of the residual one.
    pub first: Linear,
    /// The gated first layer's `W_down`; `None` for the residual one.
    pub first_down: Option<Linear>,
    /// The residual layers after the first.
    pub hidden: Vec<Linear>,
    /// The ReGLU blocks after the first layer.
    pub glu: Vec<ReGlu>,
    /// With `rezero`, the branch scale of each layer after the projection, the first layer's
    /// first; empty without.
    pub rezero: Vec<Param<Tensor<1>>>,
    pub value: Linear,
    pub value_skip: Linear,
    pub policy: Linear,
    pub local_policy: burn::nn::Embedding,
    pub send_policy: burn::nn::Embedding,
    /// `h -> GATES`, through a sigmoid in the heads.
    pub gates: Linear,
}

impl NnueV2 {
    pub fn config(&self) -> NnueV2Config {
        self.config.clone()
    }

    /// The same network for inference, with its first layer folded into one table (see
    /// [`FoldedNnue`]).
    pub fn fold(&self) -> FoldedNnue<NnueV2> {
        fold(self)
    }

    /// Multiply-adds of one evaluation once the first layer is folded into tables.
    pub fn folded_macs(&self) -> usize {
        let c = &self.config;
        let (d_h, later) = (c.d_hidden, c.layers - 2);
        let rest = match c.reglu {
            Some(d_ff) => 2 * d_h * d_h + later * 3 * d_h * d_ff,
            None => (1 + later) * d_h * d_h,
        };
        c.d_acc * d_h + rest + d_h * 3 + c.d_acc * 3 + d_h * MOVES + d_h * GATES
    }

    /// Loads the weights saved at `path`, strictly, except that weights saved before the
    /// heads had gates load too: the gates keep their initial values, every gate 1/2, and the
    /// value's two terms and the policy tables are doubled, so that the network computes what
    /// the saved one did.
    pub fn load_weights(self, path: &Path) -> Result<Self, RecordError> {
        let record = ModuleRecord::load(path)?;
        match self.clone().try_load_record(record.clone()) {
            Err(RecordError::Validation(e)) if only_gates_missing(&e) => {
                Ok(self.try_load_record(record.allow_partial(true))?.ungated())
            }
            loaded => loaded,
        }
    }

    /// Doubles everything the gates scale, which at their initial 1/2 undoes them.
    fn ungated(mut self) -> Self {
        let double = |p: Param<Tensor<2>>| p.map(|t| t * 2.0);
        self.value.weight = double(self.value.weight);
        self.value.bias = self.value.bias.map(|b| b.map(|t| t * 2.0));
        self.value_skip.weight = double(self.value_skip.weight);
        self.local_policy.weight = double(self.local_policy.weight);
        self.send_policy.weight = double(self.send_policy.weight);
        self
    }
}

/// Whether a validation error is about missing tensors, all of them the gates'. Burn reports
/// them only as a message, `Missing tensors: ["gates.weight", ...]`.
fn only_gates_missing(error: &str) -> bool {
    let Some(list) = error.strip_prefix("Missing tensors: ") else {
        return false;
    };
    // The quoted paths are the odd pieces between quotes.
    let mut paths = list.split('"').skip(1).step_by(2).peekable();
    paths.peek().is_some() && paths.all(|p| p.starts_with("gates."))
}

impl NnueNet for NnueV2 {
    fn ends(&self) -> Ends<'_> {
        Ends {
            content: &self.content,
            mix: &self.mix,
            forced_mix: &self.forced_mix,
            features: &self.features,
            value: &self.value,
            value_skip: &self.value_skip,
            policy: &self.policy,
            local_policy: &self.local_policy,
            send_policy: &self.send_policy,
            gates: Some(&self.gates),
        }
    }

    fn trunk(&self, acc: Tensor<2>) -> Tensor<2> {
        // Layer `i`'s branch, scaled by its ReZero scalar when there is one.
        let scaled = |i: usize, branch: Tensor<2>| match self.rezero.get(i) {
            Some(a) => branch * a.val().unsqueeze::<2>(),
            None => branch,
        };
        let h0 = self.proj.forward(acc);
        let mut h = match &self.first_down {
            Some(down) => {
                let branch = down.forward(relu(h0.clone()) * self.first.forward(h0.clone()));
                h0 + scaled(0, branch)
            }
            None => relu(h0.clone() + scaled(0, self.first.forward(relu(h0)))),
        };
        for (i, layer) in self.hidden.iter().enumerate() {
            h = relu(h.clone() + scaled(1 + i, layer.forward(h)));
        }
        for (i, block) in self.glu.iter().enumerate() {
            h = h.clone() + scaled(1 + i, block.forward(h));
        }
        h
    }
}

impl Model for NnueV2 {
    type Input = NnueInput;

    fn encode(states: &[BitState], device: &Device) -> NnueInput {
        nnue_input(states, device)
    }

    fn predict(&self, ids: NnueInput) -> Prediction {
        predict(self, ids)
    }
}

crate::nn::train::impl_train_steps!(NnueV2);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::BackendKind;
    use crate::nn::nnue::{assert_folds, heads};

    fn small() -> NnueV2Config {
        NnueV2Config::new()
            .with_d_embed(16)
            .with_d_acc(32)
            .with_d_hidden(16)
            .with_layers(3)
    }

    #[test]
    fn folds_with_residual_and_gated_layers() {
        let _lock = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        for config in [small(), small().with_reglu(Some(24))] {
            let model = config.init(&device);
            match config.reglu {
                Some(_) => assert_eq!((model.hidden.len(), model.glu.len()), (0, 1)),
                None => assert_eq!((model.hidden.len(), model.glu.len()), (1, 0)),
            }
            assert_folds(&model, &device);
        }
        // ReZero starts every layer as the identity on the stream: the gated network's trunk
        // is the projection alone, and folding still holds.
        let rezero = small().with_reglu(Some(24)).with_rezero(true).init(&device);
        assert_eq!(rezero.rezero.len(), 2);
        let acc = Tensor::<2>::random(
            [4, 32],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let diff: f32 = (rezero.trunk(acc.clone()) - rezero.proj.forward(acc))
            .abs()
            .max()
            .into_scalar();
        assert!(diff < 1e-6, "trunk differs from the projection by {diff}");
        assert_folds(&rezero, &device);
        let gated = small().with_reglu(Some(24)).init(&device);
        assert_eq!(
            gated.folded_macs(),
            32 * 16 + 2 * 16 * 16 + 3 * 16 * 24 + 16 * 3 + 32 * 3 + 16 * MOVES + 16 * GATES
        );
    }

    #[test]
    fn gates_start_at_a_half_and_doubling_undoes_them() {
        let _lock = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let model = small().init(&device);
        let weight: f32 = model.gates.weight.val().abs().max().into_scalar();
        let bias: f32 = model
            .gates
            .bias
            .as_ref()
            .unwrap()
            .val()
            .abs()
            .max()
            .into_scalar();
        assert_eq!((weight, bias), (0.0, 0.0));
        // Doubled, the gated heads compute what the ungated ones did.
        let (n, d_acc) = (4, 32);
        let random = |dims: [usize; 2]| {
            Tensor::<2>::random(dims, burn::tensor::Distribution::Normal(0.0, 1.0), &device)
        };
        let (h, acc) = (random([n, 16]), random([n, d_acc]));
        let contents = Tensor::<2, Int>::random(
            [n, 9],
            burn::tensor::Distribution::Uniform(0.0, 1000.0),
            &device,
        );
        let head = |m: &NnueV2, gated: bool| {
            let mut e = m.ends();
            if !gated {
                e.gates = None;
            }
            let p = heads(
                &e,
                h.clone(),
                acc.clone(),
                contents.clone(),
                m.local_policy.weight.val(),
                m.send_policy.weight.val(),
            );
            Tensor::cat(vec![p.wdl, p.policy], 1)
        };
        let ungated = model.clone().ungated();
        let diff: f32 = (head(&model, false) - head(&ungated, true))
            .abs()
            .max()
            .into_scalar();
        assert!(diff < 1e-5, "doubled gated heads differ by {diff}");
        assert!(only_gates_missing(
            r#"Missing tensors: ["gates.weight", "gates.bias"]"#
        ));
        assert!(!only_gates_missing(
            r#"Missing tensors: ["gates.weight", "policy.bias"]"#
        ));
        assert!(!only_gates_missing("Missing tensors: []"));
        assert!(!only_gates_missing(r#"Apply errors: ["gates.weight"]"#));
    }
}
