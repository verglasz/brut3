//! Which burn backend to run on, chosen at runtime among those compiled in.

use std::fmt;
use std::str::FromStr;

use burn::DispatchDevice;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// NVIDIA GPU (feature `cuda`).
    Cuda,
    /// Portable GPU via wgpu (feature `wgpu`).
    Wgpu,
    /// MLIR/LLVM CPU backend (feature `cpu-mlir`).
    CpuMlir,
    /// Pure-Rust CPU backend (feature `ndarray`).
    NdArray,
}

impl BackendKind {
    /// Every kind this crate knows about, fastest first. Not all are necessarily compiled in.
    pub const ALL: [BackendKind; 4] = [Self::Cuda, Self::Wgpu, Self::CpuMlir, Self::NdArray];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::CpuMlir => "cpu-mlir",
            Self::NdArray => "ndarray",
        }
    }

    pub const fn is_compiled(self) -> bool {
        match self {
            Self::Cuda => cfg!(feature = "cuda"),
            Self::Wgpu => cfg!(feature = "wgpu"),
            Self::CpuMlir => cfg!(feature = "cpu-mlir"),
            Self::NdArray => cfg!(feature = "ndarray"),
        }
    }

    /// The compiled-in kinds, fastest first.
    pub fn compiled() -> Vec<BackendKind> {
        Self::ALL.into_iter().filter(|k| k.is_compiled()).collect()
    }

    /// The fastest compiled-in kind. Panics only if no backend feature is enabled, which the
    /// `nn` feature set does not allow.
    pub fn preferred() -> BackendKind {
        Self::compiled()
            .first()
            .copied()
            .expect("at least one backend is compiled with the nn feature")
    }

    /// Default device of this backend. Panics if the backend is not compiled in; check
    /// [`BackendKind::is_compiled`] first (parsing already does).
    pub fn device(self) -> DispatchDevice {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => DispatchDevice::Cuda(Default::default()),
            #[cfg(feature = "wgpu")]
            Self::Wgpu => DispatchDevice::Wgpu(Default::default()),
            #[cfg(feature = "cpu-mlir")]
            Self::CpuMlir => DispatchDevice::Cpu(Default::default()),
            #[cfg(feature = "ndarray")]
            Self::NdArray => DispatchDevice::NdArray(Default::default()),
            #[allow(unreachable_patterns)]
            other => panic!("backend {} is not compiled in", other.name()),
        }
    }
}

impl BackendKind {
    /// The compiled-in kinds minus those named (comma-separated) in `UTTT_SKIP_BACKENDS`,
    /// for running the per-backend tests on a machine where one backend is unavailable.
    #[cfg(test)]
    pub(crate) fn for_tests() -> Vec<BackendKind> {
        let skip = std::env::var("UTTT_SKIP_BACKENDS").unwrap_or_default();
        let skipped: Vec<&str> = skip.split(',').map(str::trim).collect();
        Self::compiled()
            .into_iter()
            .filter(|k| !skipped.contains(&k.name()))
            .collect()
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for BackendKind {
    type Err = String;

    /// Accepts the names from [`BackendKind::name`] (plus `cpu` for `cpu-mlir`), and only
    /// compiled-in backends.
    fn from_str(s: &str) -> Result<Self, String> {
        let kind = match s {
            "cuda" => Self::Cuda,
            "wgpu" => Self::Wgpu,
            "cpu-mlir" | "cpu" => Self::CpuMlir,
            "ndarray" => Self::NdArray,
            other => {
                return Err(format!(
                    "unknown backend {other:?}; compiled in: {}",
                    names(&Self::compiled())
                ));
            }
        };
        if !kind.is_compiled() {
            return Err(format!(
                "backend {kind} is not compiled in (build with --features {}); compiled in: {}",
                match kind {
                    Self::CpuMlir => "cpu-mlir".to_string(),
                    k => k.name().to_string(),
                },
                names(&Self::compiled())
            ));
        }
        Ok(kind)
    }
}

fn names(kinds: &[BackendKind]) -> String {
    kinds
        .iter()
        .map(|k| k.name())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_compiled_kinds() {
        let _device = crate::nn::device_lock();
        for k in BackendKind::for_tests() {
            assert_eq!(k.name().parse::<BackendKind>(), Ok(k));
            let _ = k.device();
        }
        assert!("tpu".parse::<BackendKind>().is_err());
        assert!(BackendKind::preferred().is_compiled());
    }
}
