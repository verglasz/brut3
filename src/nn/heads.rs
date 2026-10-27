//! Pieces shared by the transformer models: the encoder body and the perceptron heads.

use burn::module::Param;
use burn::nn::transformer::{
    TransformerEncoder,
    TransformerEncoderConfig,
    TransformerEncoderInput,
};
use burn::nn::{
    Dropout,
    DropoutConfig,
    Embedding,
    EmbeddingConfig,
    Gelu,
    Initializer,
    LayerNorm,
    LayerNormConfig,
    Linear,
    LinearConfig,
};
use burn::prelude::*;

/// An embedding table of `n` rows of width `d`, initialised like the token embeddings of a
/// small transformer (normal, std 0.02).
pub fn embedding(n: usize, d: usize, device: &Device) -> Embedding {
    EmbeddingConfig::new(n, d)
        .with_initializer(Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        })
        .init(device)
}

/// A two-layer perceptron: `Linear -> GELU -> Dropout -> Linear`.
#[derive(Config, Debug, PartialEq)]
pub struct PerceptronConfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub d_output: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl PerceptronConfig {
    pub fn init(&self, device: &Device) -> Perceptron {
        Perceptron {
            hidden: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            activation: Gelu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            output: LinearConfig::new(self.d_hidden, self.d_output).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Perceptron {
    hidden: Linear,
    activation: Gelu,
    dropout: Dropout,
    output: Linear,
}

impl Perceptron {
    /// `[..., d_input] -> [..., d_output]`, applied to the last dimension.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let h = self
            .dropout
            .forward(self.activation.forward(self.hidden.forward(x)));
        self.output.forward(h)
    }
}

/// Shape of a transformer body: pre-norm encoder layers with full all-to-all attention.
#[derive(Config, Debug, PartialEq)]
pub struct BodyConfig {
    #[config(default = 64)]
    pub d_model: usize,
    #[config(default = 2)]
    pub n_layers: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    /// Feed-forward width inside each layer; `0` means `2 * d_model`.
    #[config(default = 0)]
    pub d_ff: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl BodyConfig {
    pub fn d_ff(&self) -> usize {
        if self.d_ff == 0 {
            2 * self.d_model
        } else {
            self.d_ff
        }
    }

    pub fn init(&self, device: &Device) -> TransformerEncoder {
        TransformerEncoderConfig::new(self.d_model, self.d_ff(), self.n_heads, self.n_layers)
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device)
    }
}

/// A class token for the per-token models: a learned token prepended to the encoder input, so
/// that it mixes with every position in every layer, and read off the end of the stream
/// through a final layer norm and a perceptron to `d_output`. The same final norm is applied
/// to the other tokens for the per-token heads, as in GPT-2's `ln_f`.
#[derive(Config, Debug, PartialEq)]
pub struct ClsReadoutConfig {
    pub d_model: usize,
    pub d_output: usize,
    /// Hidden width of the output perceptron.
    pub d_hidden: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl ClsReadoutConfig {
    pub fn init(&self, device: &Device) -> ClsReadout {
        ClsReadout {
            cls: Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init([1, 1, self.d_model], device),
            norm: LayerNormConfig::new(self.d_model).init(device),
            output: PerceptronConfig::new(self.d_model, self.d_hidden, self.d_output)
                .with_dropout(self.dropout)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct ClsReadout {
    /// The class token, `[1, 1, d_model]`.
    cls: Param<Tensor<3>>,
    norm: LayerNorm,
    output: Perceptron,
}

impl ClsReadout {
    /// `[batch, tokens, d_model] -> [batch, tokens + 1, d_model]`, the class token first.
    pub fn prepend(&self, tokens: Tensor<3>) -> Tensor<3> {
        let [n, _, d] = tokens.dims();
        Tensor::cat(vec![self.cls.val().expand([n, 1, d]), tokens], 1)
    }

    /// Splits the encoded stream `[batch, tokens + 1, d_model]` into the normalised class
    /// token `[batch, d_model]` (the global readout) and the normalised token stream
    /// `[batch, tokens, d_model]`.
    pub fn split(&self, stream: Tensor<3>) -> (Tensor<2>, Tensor<3>) {
        let [n, t1, d] = stream.dims();
        let stream = self.norm.forward(stream);
        let cls = stream.clone().slice([0..n, 0..1]).reshape([n, d]);
        (cls, stream.slice([0..n, 1..t1]))
    }

    /// The class output `[batch, d_output]` from the readout given by [`ClsReadout::split`].
    pub fn output(&self, cls: Tensor<2>) -> Tensor<2> {
        self.output.forward(cls)
    }

    /// Every token with the global readout appended: `[batch, tokens, 2 * d_model]`, the input
    /// of a per-token head that should see the whole position.
    pub fn with_global(cls: Tensor<2>, tokens: Tensor<3>) -> Tensor<3> {
        let [n, t, d] = tokens.dims();
        Tensor::cat(vec![tokens, cls.reshape([n, 1, d]).expand([n, t, d])], 2)
    }
}

/// Runs the encoder over `tokens` (`[batch, tokens, d_model]`) with every token attending to
/// every other: no padding or attention masks, the whole board is always visible.
pub fn encode_all_to_all(encoder: &TransformerEncoder, tokens: Tensor<3>) -> Tensor<3> {
    encoder.forward(TransformerEncoderInput::new(tokens))
}
