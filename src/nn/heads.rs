//! Pieces shared by the transformer models: the encoder body and the perceptron heads.

use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig};
use burn::prelude::*;

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
    pub fn init<B: Backend>(&self, device: &B::Device) -> Perceptron<B> {
        Perceptron {
            hidden: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            activation: Gelu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            output: LinearConfig::new(self.d_hidden, self.d_output).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Perceptron<B: Backend> {
    hidden: Linear<B>,
    activation: Gelu,
    dropout: Dropout,
    output: Linear<B>,
}

impl<B: Backend> Perceptron<B> {
    /// `[batch, d_input] -> [batch, d_output]`
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
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

    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerEncoder<B> {
        TransformerEncoderConfig::new(self.d_model, self.d_ff(), self.n_heads, self.n_layers)
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device)
    }
}

/// Runs the encoder over `tokens` (`[batch, tokens, d_model]`) with every token attending to
/// every other: no padding or attention masks, the whole board is always visible.
pub fn encode_all_to_all<B: Backend>(
    encoder: &TransformerEncoder<B>,
    tokens: Tensor<B, 3>,
) -> Tensor<B, 3> {
    encoder.forward(TransformerEncoderInput::new(tokens))
}

/// Position ids `0..n` as a `[1, n]` int tensor, for a positional embedding.
pub fn positions<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::<B, 1, Int>::arange(0..n as i64, device).reshape([1, n])
}
