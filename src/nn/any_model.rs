//! [`ModelSpec`] names a model kind and its hyperparameters; [`AnyModel`] is the concrete
//! enum over the built models. The spec is saved as JSON next to the weights, so a weights
//! path alone is enough to rebuild the right model.

use std::fmt;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::prelude::Device;
use serde::{Deserialize, Serialize};

use super::board_transformer::{
    BoardTransformer,
    BoardTransformerConfig,
    BoardTransformerV2,
    BoardTransformerV2Config,
};
use super::cell_transformer::{
    CellTransformer,
    CellTransformerConfig,
    CellTransformerV2,
    CellTransformerV2Config,
};
use super::heads::BodyConfig;
use super::linear::{LinearNet, LinearNetConfig};
use super::model::{HostPrediction, predict_host, predict_logits, predict_logits_tensor};
use super::nnue::{Nnue, NnueConfig};
use super::nnue_v2::{NnueV2, NnueV2Config};
use crate::game::BitState;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ModelSpec {
    /// One linear layer per head over the flat features.
    Linear,
    /// Transformer over nine small-board tokens, perceptron heads over the flattened stream.
    BoardTransformer(BoardTransformerConfig),
    /// Transformer over 81 cell tokens, perceptron heads over the flattened stream.
    CellTransformer(CellTransformerConfig),
    /// Board tokens with a per-token policy head and an attention readout for the value.
    BoardTransformerV2(BoardTransformerV2Config),
    /// Cell tokens with a per-token policy head and an attention readout for the value.
    CellTransformerV2(CellTransformerV2Config),
    /// Table-driven: content embeddings mixed into an accumulator, a small MLP, table-assisted
    /// policy. The first layer folds into lookups at export.
    Nnue(NnueConfig),
    /// The NNUE's tables and heads around reworked hidden layers: an unrectified projection
    /// that gates or feeds the first layer (see [`super::nnue_v2`]).
    NnueV2(NnueV2Config),
}

impl ModelSpec {
    /// Whether weights of kind `from` can start a model of this kind: the same kind, or an
    /// NNUE without line features for the same one with them (the line table starts at zero,
    /// so the model computes what `from` did).
    pub fn initialisable_from(&self, from: &ModelSpec) -> bool {
        let gains_lines = |to: Option<bool>, from: Option<bool>| to == Some(true) && from != Some(true);
        match (self, from) {
            (ModelSpec::Nnue(to), ModelSpec::Nnue(f)) if gains_lines(to.lines, f.lines) => {
                to.clone().with_lines(f.lines) == *f
            }
            (ModelSpec::NnueV2(to), ModelSpec::NnueV2(f)) if gains_lines(to.lines, f.lines) => {
                to.clone().with_lines(f.lines) == *f
            }
            _ => self == from,
        }
    }

    pub const NAMES: &str = "`linear`; `board-transformer`, `cell-transformer`, `board-v2` or \
         `cell-v2`, each followed by `[d_model] [layers] [heads]`; or `nnue` followed by \
         `[d_embed] [d_acc] [d_hidden] [layers] [reglu D_FF] [lines]`; or `nnue-v2` likewise, with \
         at least 2 layers, and optionally `rezero` and `rank R` (open contents' rows at rank R)";

    /// Parses a model name with optional hyperparameters, e.g. `linear`,
    /// `board-transformer 64 2 4` (`d_model`, layers, heads).
    pub fn parse(words: &[&str]) -> Result<Self, String> {
        match words.first().copied() {
            Some("linear") => {
                if let Some(extra) = words.get(1) {
                    return Err(format!("`linear` takes no parameters, got {extra:?}"));
                }
                Ok(Self::Linear)
            }
            Some("board-transformer") => Ok(Self::BoardTransformer(BoardTransformerConfig::new(
                body(&words[1..])?,
            ))),
            Some("cell-transformer") => Ok(Self::CellTransformer(CellTransformerConfig::new(
                body(&words[1..])?,
            ))),
            Some("board-v2") => Ok(Self::BoardTransformerV2(BoardTransformerV2Config::new(
                body(&words[1..])?,
            ))),
            Some("cell-v2") => Ok(Self::CellTransformerV2(CellTransformerV2Config::new(body(
                &words[1..],
            )?))),
            Some("nnue") => Ok(Self::Nnue(nnue(&words[1..])?)),
            Some("nnue-v2") => Ok(Self::NnueV2(nnue_v2(&words[1..])?)),
            Some(other) => Err(format!("unknown model {other:?}; try {}", Self::NAMES)),
            None => Err(format!("missing model name; try {}", Self::NAMES)),
        }
    }

    pub fn init(&self, device: &Device) -> AnyModel {
        match self {
            ModelSpec::Linear => AnyModel::Linear(LinearNetConfig::default().init(device)),
            ModelSpec::BoardTransformer(cfg) => AnyModel::BoardTransformer(cfg.init(device)),
            ModelSpec::CellTransformer(cfg) => AnyModel::CellTransformer(cfg.init(device)),
            ModelSpec::BoardTransformerV2(cfg) => AnyModel::BoardTransformerV2(cfg.init(device)),
            ModelSpec::CellTransformerV2(cfg) => AnyModel::CellTransformerV2(cfg.init(device)),
            ModelSpec::Nnue(cfg) => AnyModel::Nnue(cfg.init(device)),
            ModelSpec::NnueV2(cfg) => AnyModel::NnueV2(cfg.init(device)),
        }
    }

    fn json_path(weights: &Path) -> PathBuf {
        weights.with_extension("json")
    }

    /// Where the weights of a model saved at `path` live: burnpack, `<path>.bpk`.
    pub fn weights_path(path: &Path) -> PathBuf {
        path.with_extension("bpk")
    }
}

fn num(words: &[&str], i: usize, what: &str, default: usize) -> Result<usize, String> {
    match words.get(i) {
        None => Ok(default),
        Some(w) => w
            .parse()
            .map_err(|_| format!("{what} must be a number, got {w:?}")),
    }
}

/// [`nnue`]'s words for an [`NnueV2Config`], which needs at least 2 layers, and optionally
/// `rezero` and `rank R` (a low-rank content table).
fn nnue_v2(words: &[&str]) -> Result<NnueV2Config, String> {
    let rezero = words.contains(&"rezero");
    let mut words: Vec<&str> = words.iter().copied().filter(|w| *w != "rezero").collect();
    let content_rank = match words.iter().position(|w| *w == "rank") {
        Some(i) => {
            let rank = words
                .get(i + 1)
                .and_then(|w| w.parse::<usize>().ok())
                .filter(|&r| r > 0)
                .ok_or("rank takes a positive number, e.g. `rank 32`")?;
            words.drain(i..i + 2);
            Some(rank)
        }
        None => None,
    };
    let c = nnue(&words)?;
    if c.layers < 2 {
        return Err("nnue-v2 needs at least 2 layers: the projection and a hidden layer".into());
    }
    Ok(NnueV2Config::new()
        .with_d_embed(c.d_embed)
        .with_d_acc(c.d_acc)
        .with_d_hidden(c.d_hidden)
        .with_layers(c.layers)
        .with_reglu(c.reglu)
        .with_rezero(rezero)
        .with_lines(c.lines)
        .with_content_rank(content_rank))
}

/// `[d_embed] [d_acc] [d_hidden] [layers]`, each optional, then optionally `reglu D_FF` for
/// ReGLU blocks of inner width `D_FF` after the first hidden layer, and anywhere `lines` for
/// small-board line features.
fn nnue(words: &[&str]) -> Result<NnueConfig, String> {
    let lines = words.contains(&"lines");
    let words: Vec<&str> = words.iter().copied().filter(|w| *w != "lines").collect();
    let words = words.as_slice();
    let (words, reglu) = match words.iter().position(|w| *w == "reglu") {
        Some(i) => match &words[i + 1..] {
            [width] => (
                &words[..i],
                Some(
                    width
                        .parse::<usize>()
                        .ok()
                        .filter(|&w| w > 0)
                        .ok_or_else(|| format!("reglu needs a positive width, got {width:?}"))?,
                ),
            ),
            _ => return Err("reglu takes exactly one width, e.g. `reglu 176`".into()),
        },
        None => (words, None),
    };
    if let Some(extra) = words.get(4) {
        return Err(format!("unexpected argument {extra:?} after layers"));
    }
    let d = NnueConfig::new();
    let cfg = NnueConfig::new()
        .with_d_embed(num(words, 0, "d_embed", d.d_embed)?)
        .with_d_acc(num(words, 1, "d_acc", d.d_acc)?)
        .with_d_hidden(num(words, 2, "d_hidden", d.d_hidden)?)
        .with_layers(num(words, 3, "layers", d.layers)?)
        .with_reglu(reglu)
        .with_lines(lines.then_some(true));
    if cfg.d_embed == 0 || cfg.d_acc == 0 || cfg.d_hidden == 0 || cfg.layers == 0 {
        return Err("nnue sizes must be positive".into());
    }
    Ok(cfg)
}

/// `[d_model] [layers] [heads]`, each optional.
fn body(words: &[&str]) -> Result<BodyConfig, String> {
    if let Some(extra) = words.get(3) {
        return Err(format!("unexpected argument {extra:?} after heads"));
    }
    let defaults = BodyConfig::new();
    let cfg = BodyConfig::new()
        .with_d_model(num(words, 0, "d_model", defaults.d_model)?)
        .with_n_layers(num(words, 1, "layers", defaults.n_layers)?)
        .with_n_heads(num(words, 2, "heads", defaults.n_heads)?);
    if !cfg.d_model.is_multiple_of(cfg.n_heads) {
        return Err(format!(
            "d_model {} must be a multiple of heads {}",
            cfg.d_model, cfg.n_heads
        ));
    }
    Ok(cfg)
}

impl fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSpec::Linear => f.write_str("linear"),
            ModelSpec::BoardTransformer(cfg) => write!(
                f,
                "board-transformer (d_model {}, {} layers, {} heads)",
                cfg.body.d_model, cfg.body.n_layers, cfg.body.n_heads
            ),
            ModelSpec::CellTransformer(cfg) => write!(
                f,
                "cell-transformer (d_model {}, {} layers, {} heads)",
                cfg.body.d_model, cfg.body.n_layers, cfg.body.n_heads
            ),
            ModelSpec::BoardTransformerV2(cfg) => write!(
                f,
                "board-v2 (d_model {}, {} layers, {} heads)",
                cfg.body.d_model, cfg.body.n_layers, cfg.body.n_heads
            ),
            ModelSpec::CellTransformerV2(cfg) => write!(
                f,
                "cell-v2 (d_model {}, {} layers, {} heads)",
                cfg.body.d_model, cfg.body.n_layers, cfg.body.n_heads
            ),
            ModelSpec::Nnue(cfg) => {
                write!(
                    f,
                    "nnue (d_embed {}, d_acc {}, d_hidden {}, {} layers",
                    cfg.d_embed, cfg.d_acc, cfg.d_hidden, cfg.layers
                )?;
                if let Some(d_ff) = cfg.reglu {
                    write!(f, ", ReGLU {d_ff}")?;
                }
                if cfg.lines == Some(true) {
                    write!(f, ", lines")?;
                }
                write!(f, ")")
            }
            ModelSpec::NnueV2(cfg) => {
                write!(
                    f,
                    "nnue-v2 (d_embed {}, d_acc {}, d_hidden {}, {} layers",
                    cfg.d_embed, cfg.d_acc, cfg.d_hidden, cfg.layers
                )?;
                if let Some(d_ff) = cfg.reglu {
                    write!(f, ", ReGLU {d_ff}")?;
                }
                if cfg.rezero {
                    write!(f, ", ReZero")?;
                }
                if cfg.lines == Some(true) {
                    write!(f, ", lines")?;
                }
                if let Some(rank) = cfg.content_rank {
                    write!(f, ", content rank {rank}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Every model kind.
// A handful of long-lived values per process; variant size does not matter.
#[allow(clippy::large_enum_variant)]
pub enum AnyModel {
    Linear(LinearNet),
    BoardTransformer(BoardTransformer),
    CellTransformer(CellTransformer),
    BoardTransformerV2(BoardTransformerV2),
    CellTransformerV2(CellTransformerV2),
    Nnue(Nnue),
    NnueV2(NnueV2),
}

/// Runs `$body` with `$m` bound to the model inside `$any`, whatever its kind.
macro_rules! with_model {
    ($any:expr, |$m:ident| $body:expr) => {
        match $any {
            AnyModel::Linear($m) => $body,
            AnyModel::BoardTransformer($m) => $body,
            AnyModel::CellTransformer($m) => $body,
            AnyModel::BoardTransformerV2($m) => $body,
            AnyModel::CellTransformerV2($m) => $body,
            AnyModel::Nnue($m) => $body,
            AnyModel::NnueV2($m) => $body,
        }
    };
}

impl AnyModel {
    pub fn spec(&self) -> ModelSpec {
        match self {
            AnyModel::Linear(_) => ModelSpec::Linear,
            AnyModel::BoardTransformer(m) => ModelSpec::BoardTransformer(m.config()),
            AnyModel::CellTransformer(m) => ModelSpec::CellTransformer(m.config()),
            AnyModel::BoardTransformerV2(m) => ModelSpec::BoardTransformerV2(m.config()),
            AnyModel::CellTransformerV2(m) => ModelSpec::CellTransformerV2(m.config()),
            AnyModel::Nnue(m) => ModelSpec::Nnue(m.config()),
            AnyModel::NnueV2(m) => ModelSpec::NnueV2(m.config()),
        }
    }

    pub fn num_params(&self) -> usize {
        with_model!(self, |m| m.num_params())
    }

    /// Evaluates with the tables `int8` covers fake-quantised to the export's int8 grid (see
    /// [`super::model::Model::set_qat`]).
    pub fn set_qat(&mut self, int8: &super::export::Int8Arrays) {
        use super::model::Model;
        with_model!(self, |m| m.set_qat(int8))
    }

    /// Replaces an NNUE's open content rows by their best rank-`rank` approximation: each row
    /// projected onto the table's top `rank` singular directions (every row counting the
    /// same). Errors for other kinds.
    pub fn truncate_content(&mut self, rank: usize) -> Result<(), String> {
        use burn::prelude::*;

        let content = match self {
            AnyModel::Nnue(m) => &mut m.content,
            AnyModel::NnueV2(m) if m.content_low.is_some() => {
                return Err("its content table is low-rank already".into());
            }
            AnyModel::NnueV2(m) => &mut m.content,
            other => return Err(format!("a {} model has no content table", other.spec())),
        };
        let table = content.weight.val();
        let device = table.device();
        let [rows_n, d] = table.dims();
        if rank >= d {
            return Ok(());
        }
        let open = super::nnue::OPEN_CONTENTS;
        let rows: Vec<f32> = table
            .slice([0..open, 0..d])
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))?;
        let (_, vectors) =
            super::linalg::symmetric_eigen(super::linalg::gram(&rows, d, None), d);
        // The projection V_r V_rᵀ onto the top `rank` directions.
        let mut projection = vec![0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                projection[i * d + j] = (0..rank)
                    .map(|k| vectors[i * d + k] * vectors[j * d + k])
                    .sum::<f64>() as f32;
            }
        }
        let projection = Tensor::<2>::from_data(TensorData::new(projection, [d, d]), &device);
        content.weight = content.weight.clone().map(|w| {
            let open_rows = w.clone().slice([0..open, 0..d]).matmul(projection.clone());
            Tensor::cat(vec![open_rows, w.slice([open..rows_n, 0..d])], 0)
        });
        Ok(())
    }

    /// The model's quality over `samples` (see [`super::train::evaluate`]).
    pub fn evaluate(
        &self,
        device: &Device,
        samples: &[super::train::Sample],
    ) -> super::train::Evaluation {
        with_model!(self, |m| super::train::evaluate(m, device, samples))
    }

    /// How far the model is from respecting the board's symmetries over `samples` (see
    /// [`super::train::evaluate_symmetries`]).
    pub fn evaluate_symmetries(
        &self,
        device: &Device,
        samples: &[super::train::Sample],
    ) -> super::train::SymmetryEvaluation {
        with_model!(self, |m| super::train::evaluate_symmetries(
            m, device, samples
        ))
    }

    pub fn predict_host(&self, states: &[BitState], device: &Device) -> HostPrediction {
        with_model!(self, |m| predict_host(m, states, device))
    }

    /// Raw logits, one row of [`super::model::LOGITS`] per state; see
    /// [`super::model::predict_logits`].
    pub fn predict_logits(&self, states: &[BitState], device: &Device) -> Vec<f32> {
        with_model!(self, |m| predict_logits(m, states, device))
    }

    /// The logits still on the device; see [`super::model::predict_logits_tensor`].
    pub fn predict_logits_tensor(
        &self,
        states: &[BitState],
        device: &Device,
    ) -> burn::prelude::Tensor<2> {
        with_model!(self, |m| predict_logits_tensor(m, states, device))
    }

    /// Writes `<path>.bpk` (weights) and `<path>.json` (spec).
    pub fn save(self, path: &Path) -> Result<(), String> {
        let spec = self.spec();
        let json = serde_json::to_string_pretty(&spec).map_err(|e| e.to_string())?;
        let json_path = ModelSpec::json_path(path);
        std::fs::write(&json_path, json)
            .map_err(|e| format!("cannot write {}: {e}", json_path.display()))?;
        let result = with_model!(self, |m| m.save_file(ModelSpec::weights_path(path)));
        result.map_err(|e| format!("cannot save weights to {}: {e}", path.display()))
    }

    /// Reads the spec from `<path>.json` and the weights from `<path>.bpk` onto `device`, ready
    /// for inference (training flags off).
    pub fn load(path: &Path, device: &Device) -> Result<Self, String> {
        let json_path = ModelSpec::json_path(path);
        let json = std::fs::read_to_string(&json_path)
            .map_err(|e| format!("cannot read model spec {}: {e}", json_path.display()))?;
        let spec: ModelSpec = serde_json::from_str(&json)
            .map_err(|e| format!("bad model spec {}: {e}", json_path.display()))?;
        let load_err = |e| format!("cannot load weights from {}: {e}", path.display());
        Ok(match spec.init(device) {
            AnyModel::Linear(m) => AnyModel::Linear(
                m.try_load_file(ModelSpec::weights_path(path))
                    .map_err(load_err)?
                    .valid(),
            ),
            AnyModel::BoardTransformer(m) => AnyModel::BoardTransformer(
                m.try_load_file(ModelSpec::weights_path(path))
                    .map_err(load_err)?
                    .valid(),
            ),
            AnyModel::CellTransformer(m) => AnyModel::CellTransformer(
                m.try_load_file(ModelSpec::weights_path(path))
                    .map_err(load_err)?
                    .valid(),
            ),
            AnyModel::BoardTransformerV2(m) => AnyModel::BoardTransformerV2(
                m.try_load_file(ModelSpec::weights_path(path))
                    .map_err(load_err)?
                    .valid(),
            ),
            AnyModel::CellTransformerV2(m) => AnyModel::CellTransformerV2(
                m.try_load_file(ModelSpec::weights_path(path))
                    .map_err(load_err)?
                    .valid(),
            ),
            AnyModel::Nnue(m) => AnyModel::Nnue(
                m.load_weights(&ModelSpec::weights_path(path))
                    .map_err(load_err)?
                    .valid(),
            ),
            AnyModel::NnueV2(m) => AnyModel::NnueV2(
                m.load_weights(&ModelSpec::weights_path(path))
                    .map_err(load_err)?
                    .valid(),
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;
    use crate::nn::BackendKind;

    #[test]
    fn parse_specs() {
        assert_eq!(ModelSpec::parse(&["linear"]), Ok(ModelSpec::Linear));
        assert!(ModelSpec::parse(&["linear", "3"]).is_err());
        assert!(ModelSpec::parse(&["perceptron"]).is_err());
        assert!(ModelSpec::parse(&[]).is_err());
        let spec = ModelSpec::parse(&["board-transformer", "32", "3", "2"]).unwrap();
        match &spec {
            ModelSpec::BoardTransformer(cfg) => {
                assert_eq!(
                    (cfg.body.d_model, cfg.body.n_layers, cfg.body.n_heads),
                    (32, 3, 2)
                );
            }
            other => panic!("{other:?}"),
        }
        assert!(
            spec.to_string()
                .starts_with("board-transformer (d_model 32")
        );
        match ModelSpec::parse(&["cell-transformer"]).unwrap() {
            ModelSpec::CellTransformer(cfg) => assert_eq!(cfg.body.d_model, 64),
            other => panic!("{other:?}"),
        }
        assert!(
            ModelSpec::parse(&["cell-transformer", "30", "1", "4"]).is_err(),
            "not divisible"
        );
        assert!(ModelSpec::parse(&["cell-transformer", "32", "1", "2", "9"]).is_err());
        match ModelSpec::parse(&["board-v2", "48", "3", "4"]).unwrap() {
            ModelSpec::BoardTransformerV2(cfg) => assert_eq!(cfg.body.d_model, 48),
            other => panic!("{other:?}"),
        }
        assert!(
            ModelSpec::parse(&["cell-v2"])
                .unwrap()
                .to_string()
                .starts_with("cell-v2")
        );
        assert!(ModelSpec::parse(&["board-transformer", "big"]).is_err());
        match ModelSpec::parse(&["nnue"]).unwrap() {
            ModelSpec::Nnue(cfg) => assert_eq!(
                (cfg.d_embed, cfg.d_acc, cfg.d_hidden, cfg.layers),
                (64, 256, 32, 2)
            ),
            other => panic!("{other:?}"),
        }
        match ModelSpec::parse(&["nnue", "32", "512", "512", "3"]).unwrap() {
            ModelSpec::Nnue(cfg) => assert_eq!(
                (cfg.d_embed, cfg.d_acc, cfg.d_hidden, cfg.layers),
                (32, 512, 512, 3)
            ),
            other => panic!("{other:?}"),
        }
        assert!(ModelSpec::parse(&["nnue", "32", "512", "512", "0"]).is_err());
        assert!(ModelSpec::parse(&["nnue", "1", "2", "3", "4", "5"]).is_err());
        match ModelSpec::parse(&["nnue", "64", "512", "128", "4", "reglu", "176"]).unwrap() {
            ModelSpec::Nnue(cfg) => assert_eq!((cfg.layers, cfg.reglu), (4, Some(176))),
            other => panic!("parsed {other}"),
        }
        assert!(ModelSpec::parse(&["nnue", "reglu"]).is_err());
        assert!(ModelSpec::parse(&["nnue", "reglu", "0"]).is_err());
        match ModelSpec::parse(&["nnue-v2", "64", "512", "128", "4", "reglu", "176"]).unwrap() {
            ModelSpec::NnueV2(cfg) => assert_eq!((cfg.layers, cfg.reglu), (4, Some(176))),
            other => panic!("parsed {other}"),
        }
        assert!(ModelSpec::parse(&["nnue-v2", "64", "512", "128", "1"]).is_err());
        match ModelSpec::parse(&["nnue", "64", "512", "128", "4", "lines"]).unwrap() {
            ModelSpec::Nnue(cfg) => assert_eq!((cfg.layers, cfg.lines), (4, Some(true))),
            other => panic!("parsed {other}"),
        }
        match ModelSpec::parse(&["nnue-v2", "lines", "64", "512", "128", "4", "reglu", "96"])
            .unwrap()
        {
            ModelSpec::NnueV2(cfg) => assert_eq!((cfg.reglu, cfg.lines), (Some(96), Some(true))),
            other => panic!("parsed {other}"),
        }
        match ModelSpec::parse(&["nnue-v2", "64", "512", "128", "4", "rank", "32", "lines"])
            .unwrap()
        {
            ModelSpec::NnueV2(cfg) => {
                assert_eq!((cfg.content_rank, cfg.layers), (Some(32), 4));
                assert_eq!(
                    ModelSpec::NnueV2(cfg).to_string(),
                    "nnue-v2 (d_embed 64, d_acc 512, d_hidden 128, 4 layers, lines, content rank 32)"
                );
            }
            other => panic!("parsed {other:?}"),
        }
        assert!(ModelSpec::parse(&["nnue-v2", "rank"]).is_err());
        assert!(ModelSpec::parse(&["nnue-v2", "rank", "0"]).is_err());
        match ModelSpec::parse(&["nnue-v2", "64", "512", "128", "3", "rezero", "reglu", "96"])
            .unwrap()
        {
            ModelSpec::NnueV2(cfg) => assert_eq!((cfg.reglu, cfg.rezero), (Some(96), true)),
            other => panic!("parsed {other}"),
        }
    }

    #[test]
    fn transformer_specs_roundtrip_through_json_and_weights() {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let dir = std::env::temp_dir().join(format!("uttt-anymodel-tf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, words) in [
            ("board", vec!["board-transformer", "16", "1", "2"]),
            ("cell", vec!["cell-transformer", "16", "1", "2"]),
            ("board-v2", vec!["board-v2", "16", "1", "2"]),
            ("cell-v2", vec!["cell-v2", "16", "1", "2"]),
            ("nnue", vec!["nnue", "8", "16", "8", "2"]),
        ] {
            let spec = ModelSpec::parse(&words).unwrap();
            let model = spec.init(&device);
            let states = [BitState::initial()];
            let before = model.predict_host(&states, &device);
            let path = dir.join(name);
            model.save(&path).unwrap();
            let loaded = AnyModel::load(&path, &device).unwrap();
            assert_eq!(loaded.spec(), spec);
            let after = loaded.predict_host(&states, &device);
            for c in 0..3 {
                assert!((after.wdl[0][c] - before.wdl[0][c]).abs() < 1e-5);
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn save_and_load_roundtrip_with_spec() {
        let _device = crate::nn::device_lock();
        let device = BackendKind::preferred().device();
        let model = ModelSpec::Linear.init(&device);
        let states = [BitState::initial()];
        let before = model.predict_host(&states, &device);
        let dir = std::env::temp_dir().join(format!("uttt-anymodel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("net");
        model.save(&path).unwrap();
        assert!(path.with_extension("json").exists());
        assert!(path.with_extension("bpk").exists());
        let loaded = AnyModel::load(&path, &device).unwrap();
        assert_eq!(loaded.spec(), ModelSpec::Linear);
        assert_eq!(loaded.predict_host(&states, &device), before);
        assert!(AnyModel::load(&dir.join("missing"), &device).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
