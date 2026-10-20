//! [`ModelSpec`] names a model kind and its hyperparameters; [`AnyModel`] is the concrete
//! enum over the built models. The spec is saved as JSON next to the weights, so a weights
//! path alone is enough to rebuild the right model.

use std::fmt;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::{Dispatch, DispatchDevice};
use serde::{Deserialize, Serialize};

use super::linear::{LinearNet, LinearNetConfig};
use super::model::{HostPrediction, predict_host};
use crate::game::BitState;

/// Weights are stored as named MessagePack; the recorder appends its own `.mpk` extension.
pub type Recorder = NamedMpkFileRecorder<FullPrecisionSettings>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ModelSpec {
    /// One linear layer per head over the flat features.
    Linear,
}

impl ModelSpec {
    pub const NAMES: &str = "`linear`";

    /// Parses a model name with optional hyperparameters, e.g. `linear`.
    pub fn parse(words: &[&str]) -> Result<Self, String> {
        match words.first().copied() {
            Some("linear") => {
                if let Some(extra) = words.get(1) {
                    return Err(format!("`linear` takes no parameters, got {extra:?}"));
                }
                Ok(Self::Linear)
            }
            Some(other) => Err(format!("unknown model {other:?}; try {}", Self::NAMES)),
            None => Err(format!("missing model name; try {}", Self::NAMES)),
        }
    }

    pub fn init(&self, device: &DispatchDevice) -> AnyModel {
        match self {
            ModelSpec::Linear => AnyModel::Linear(LinearNetConfig::default().init(device)),
        }
    }

    fn json_path(weights: &Path) -> PathBuf {
        weights.with_extension("json")
    }
}

impl fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSpec::Linear => f.write_str("linear"),
        }
    }
}

/// Every model kind, on the [`Dispatch`] backend.
pub enum AnyModel {
    Linear(LinearNet<Dispatch>),
}

impl AnyModel {
    pub fn spec(&self) -> ModelSpec {
        match self {
            AnyModel::Linear(_) => ModelSpec::Linear,
        }
    }

    pub fn num_params(&self) -> usize {
        match self {
            AnyModel::Linear(m) => m.num_params(),
        }
    }

    pub fn predict_host(&self, states: &[BitState], device: &DispatchDevice) -> HostPrediction {
        match self {
            AnyModel::Linear(m) => predict_host(m, states, device),
        }
    }

    /// Writes `<path>.mpk` (weights) and `<path>.json` (spec).
    pub fn save(self, path: &Path) -> Result<(), String> {
        let spec = self.spec();
        let json = serde_json::to_string_pretty(&spec).map_err(|e| e.to_string())?;
        let json_path = ModelSpec::json_path(path);
        std::fs::write(&json_path, json)
            .map_err(|e| format!("cannot write {}: {e}", json_path.display()))?;
        let result = match self {
            AnyModel::Linear(m) => m.save_file(path.to_path_buf(), &Recorder::new()),
        };
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
