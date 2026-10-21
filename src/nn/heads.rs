//! Pieces shared by the transformer models: the encoder body and the perceptron heads.

use burn::module::Param;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::transformer::{
    PositionWiseFeedForward, PositionWiseFeedForwardConfig, TransformerEncoder,
    TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, Initializer, LayerNorm,
    LayerNormConfig, Linear, LinearConfig,
};
use burn::prelude::*;

/// An embedding table of `n` rows of width `d`, initialised like the token embeddings of a
/// small transformer (normal, std 0.02).
pub fn embedding<B: Backend>(n: usize, d: usize, device: &B::Device) -> Embedding<B> {
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

/// Reads the whole residual stream through one learned query: a decoder layer with a single
/// query token that cross-attends to every encoder position (pre-norm, residual), then a
/// feed-forward block and a linear map to `d_output`. Unlike a perceptron over the flattened
/// stream, its size does not grow with the number of tokens.
#[derive(Config, Debug, PartialEq)]
pub struct AttentionReadoutConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub d_output: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl AttentionReadoutConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AttentionReadout<B> {
        AttentionReadout {
            query: Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init([1, 1, self.d_model], device),
            norm_stream: LayerNormConfig::new(self.d_model).init(device),
            attention: MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                .with_dropout(self.dropout)
                .init(device),
            norm_ff: LayerNormConfig::new(self.d_model).init(device),
            ff: PositionWiseFeedForwardConfig::new(self.d_model, self.d_ff)
                .with_dropout(self.dropout)
                .init(device),
            output: LinearConfig::new(self.d_model, self.d_output).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct AttentionReadout<B: Backend> {
    /// The single query token, `[1, 1, d_model]`.
    query: Param<Tensor<B, 3>>,
    norm_stream: LayerNorm<B>,
    attention: MultiHeadAttention<B>,
    norm_ff: LayerNorm<B>,
    ff: PositionWiseFeedForward<B>,
    output: Linear<B>,
}

impl<B: Backend> AttentionReadout<B> {
    /// `[batch, tokens, d_model] -> [batch, d_output]`
    pub fn forward(&self, stream: Tensor<B, 3>) -> Tensor<B, 2> {
        let [n, _, d] = stream.dims();
        let query = self.query.val().expand([n, 1, d]);
        let keys = self.norm_stream.forward(stream);
        let context = self
            .attention
            .forward(MhaInput::new(query.clone(), keys.clone(), keys))
            .context;
        let h = query + context;
        let h = h.clone() + self.ff.forward(self.norm_ff.forward(h));
        self.output.forward(h.reshape([n, d]))
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
