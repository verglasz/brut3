//! [`ModelSpec`] names a model kind and its hyperparameters; [`AnyModel`] is the concrete
//! enum over the built models. The spec is saved as JSON next to the weights, so a weights
//! path alone is enough to rebuild the right model.

use std::fmt;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::{Dispatch, DispatchDevice};
use serde::{Deserialize, Serialize};

use super::board_transformer::{BoardTransformer, BoardTransformerConfig};
use super::cell_transformer::{CellTransformer, CellTransformerConfig};
use super::heads::BodyConfig;
use super::linear::{LinearNet, LinearNetConfig};
use super::model::{HostPrediction, predict_host, predict_logits};
use crate::game::BitState;

/// Weights are stored as named MessagePack; the recorder appends its own `.mpk` extension.
pub type Recorder = NamedMpkFileRecorder<FullPrecisionSettings>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ModelSpec {
    /// One linear layer per head over the flat features.
    Linear,
    /// Transformer over nine small-board tokens.
    BoardTransformer(BoardTransformerConfig),
    /// Transformer over 81 cell tokens.
    CellTransformer(CellTransformerConfig),
}

impl ModelSpec {
    pub const NAMES: &str = "`linear`, `board-transformer [d_model] [layers] [heads]` or \
         `cell-transformer [d_model] [layers] [heads]`";

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
            Some(other) => Err(format!("unknown model {other:?}; try {}", Self::NAMES)),
            None => Err(format!("missing model name; try {}", Self::NAMES)),
        }
    }

    pub fn init(&self, device: &DispatchDevice) -> AnyModel {
        match self {
            ModelSpec::Linear => AnyModel::Linear(LinearNetConfig::default().init(device)),
            ModelSpec::BoardTransformer(cfg) => AnyModel::BoardTransformer(cfg.init(device)),
            ModelSpec::CellTransformer(cfg) => AnyModel::CellTransformer(cfg.init(device)),
        }
    }

    fn json_path(weights: &Path) -> PathBuf {
        weights.with_extension("json")
    }
}

/// `[d_model] [layers] [heads]`, each optional.
fn body(words: &[&str]) -> Result<BodyConfig, String> {
    fn num(words: &[&str], i: usize, what: &str, default: usize) -> Result<usize, String> {
        match words.get(i) {
            None => Ok(default),
            Some(w) => w
                .parse()
                .map_err(|_| format!("{what} must be a number, got {w:?}")),
        }
    }
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
        }
    }
}

/// Every model kind, on the [`Dispatch`] backend.
// A handful of long-lived values per process; variant size does not matter.
#[allow(clippy::large_enum_variant)]
pub enum AnyModel {
    Linear(LinearNet<Dispatch>),
    BoardTransformer(BoardTransformer<Dispatch>),
    CellTransformer(CellTransformer<Dispatch>),
}

/// Runs `$body` with `$m` bound to the model inside `$any`, whatever its kind.
macro_rules! with_model {
    ($any:expr, |$m:ident| $body:expr) => {
        match $any {
            AnyModel::Linear($m) => $body,
            AnyModel::BoardTransformer($m) => $body,
            AnyModel::CellTransformer($m) => $body,
        }
    };
}

impl AnyModel {
    pub fn spec(&self) -> ModelSpec {
        match self {
            AnyModel::Linear(_) => ModelSpec::Linear,
            AnyModel::BoardTransformer(m) => ModelSpec::BoardTransformer(m.config()),
            AnyModel::CellTransformer(m) => ModelSpec::CellTransformer(m.config()),
        }
    }

    pub fn num_params(&self) -> usize {
        with_model!(self, |m| m.num_params())
    }

    pub fn predict_host(&self, states: &[BitState], device: &DispatchDevice) -> HostPrediction {
        with_model!(self, |m| predict_host(m, states, device))
    }

    /// Raw logits, one row of [`super::model::LOGITS`] per state; see
    /// [`super::model::predict_logits`].
    pub fn predict_logits(&self, states: &[BitState], device: &DispatchDevice) -> Vec<f32> {
        with_model!(self, |m| predict_logits(m, states, device))
    }

    /// Writes `<path>.mpk` (weights) and `<path>.json` (spec).
    pub fn save(self, path: &Path) -> Result<(), String> {
        let spec = self.spec();
        let json = serde_json::to_string_pretty(&spec).map_err(|e| e.to_string())?;
        let json_path = ModelSpec::json_path(path);
        std::fs::write(&json_path, json)
            .map_err(|e| format!("cannot write {}: {e}", json_path.display()))?;
        let result = with_model!(self, |m| m.save_file(path.to_path_buf(), &Recorder::new()));
        result.map_err(|e| format!("cannot save weights to {}: {e}", path.display()))
    }

    /// Reads the spec from `<path>.json` and the weights from `<path>.mpk` onto `device`.
    pub fn load(path: &Path, device: &DispatchDevice) -> Result<Self, String> {
        let json_path = ModelSpec::json_path(path);
        let json = std::fs::read_to_string(&json_path)
            .map_err(|e| format!("cannot read model spec {}: {e}", json_path.display()))?;
        let spec: ModelSpec = serde_json::from_str(&json)
            .map_err(|e| format!("bad model spec {}: {e}", json_path.display()))?;
        let load_err = |e| format!("cannot load weights from {}: {e}", path.display());
        Ok(match spec.init(device) {
            AnyModel::Linear(m) => AnyModel::Linear(
                m.load_file(path.to_path_buf(), &Recorder::new(), device)
                    .map_err(load_err)?,
            ),
            AnyModel::BoardTransformer(m) => AnyModel::BoardTransformer(
                m.load_file(path.to_path_buf(), &Recorder::new(), device)
                    .map_err(load_err)?,
            ),
            AnyModel::CellTransformer(m) => AnyModel::CellTransformer(
                m.load_file(path.to_path_buf(), &Recorder::new(), device)
                    .map_err(load_err)?,
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
        assert!(ModelSpec::parse(&["board-transformer", "big"]).is_err());
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
        assert!(path.with_extension("mpk").exists());
        let loaded = AnyModel::load(&path, &device).unwrap();
        assert_eq!(loaded.spec(), ModelSpec::Linear);
        assert_eq!(loaded.predict_host(&states, &device), before);
        assert!(AnyModel::load(&dir.join("missing"), &device).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
