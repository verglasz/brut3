//! What the loss does not show about a training run: the learning rate, the gradient norm of
//! each part of the network, and how far each optimizer step moves that part's weights
//! relative to their size. All are logged as training metrics next to the loss.
//!
//! The parts are the [`GROUPS`], by the model's top-level field names (the NNUE names: the
//! first layer's tables, the heads, and everything in between). The statistics are computed on
//! the device as part of each step and read back on burn's metric thread, so the training
//! thread does not wait for them. A step's update is only known once the optimizer has run,
//! after the step's output was made, so each step reports the update of the step before it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use burn::module::{ModuleVisitor, Param, ParamId};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::{ExecutionError, TensorReadError};
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    AccuracyInput,
    Adaptor,
    LossInput,
    Metric,
    MetricAttributes,
    MetricMetadata,
    MetricName,
    Numeric,
    NumericAttributes,
    NumericEntry,
    SerializedEntry,
};
use burn::train::{ClassificationOutput, ItemLazy};

/// The parts of a network the statistics are kept for.
pub const GROUPS: [&str; 3] = ["tables", "trunk", "heads"];

/// The group of a model's top-level field: the NNUE first layer's tables and matrices, its
/// heads, or the trunk (everything else, including every field of the other model kinds).
fn group(field: &str) -> usize {
    match field {
        "content" | "mix" | "forced_mix" | "features" => 0,
        "value" | "value_skip" | "policy" | "local_policy" | "send_policy" | "gates" => 2,
        _ => 1,
    }
}

/// One float parameter: its group, its id and its values flattened, without autodiff.
struct Flat {
    group: usize,
    id: ParamId,
    values: Tensor<1>,
    /// Its gradient, flattened, when asked for.
    grad: Option<Tensor<1>>,
}

/// Collects every float parameter of a module as [`Flat`], tracking the field path.
struct Collect<'a> {
    path: Vec<String>,
    grads: Option<&'a GradientsParams>,
    out: Vec<Flat>,
}

impl ModuleVisitor for Collect<'_> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let flat = |t: Tensor<D>| t.flatten::<1>(0, D - 1);
        self.out.push(Flat {
            group: self.path.first().map_or(1, |f| group(f)),
            id: param.id,
            values: flat(param.val().without_autodiff()),
            grad: self.grads.and_then(|g| g.get::<D>(param.id)).map(flat),
        });
    }
}

fn collect<M: Module>(module: &M, grads: Option<&GradientsParams>) -> Vec<Flat> {
    let mut visitor = Collect {
        path: Vec::new(),
        grads,
        out: Vec::new(),
    };
    module.visit(&mut visitor);
    visitor.out
}

/// `[GROUPS]`: the sum of squares of `x` over each group's parameters.
fn group_squares<'a>(
    items: impl Iterator<Item = (usize, &'a Tensor<1>)>,
    device: &Device,
) -> Tensor<1> {
    let mut sums: [Option<Tensor<1>>; GROUPS.len()] = Default::default();
    for (g, x) in items {
        let s = (x.clone() * x.clone()).sum();
        sums[g] = Some(match sums[g].take() {
            Some(acc) => acc + s,
            None => s,
        });
    }
    Tensor::cat(
        sums.into_iter()
            .map(|s| s.unwrap_or_else(|| Tensor::zeros([1], device)))
            .collect(),
        0,
    )
}

thread_local! {
    /// The squared update of each group by the last optimizer step on this thread, for the
    /// next step's output (training steps and optimizer steps alternate on one thread).
    static LAST_UPDATE: RefCell<Option<Tensor<1>>> = const { RefCell::new(None) };
}

/// The optimizer step with the squared size of its update recorded per group (see
/// [`TrainStats`]); what `TrainStep::optimize` does for the models.
pub fn optimize_recorded<M: Module>(
    module: M,
    optim: &mut burn::optim::ModuleOptimizer,
    lr: burn::lr_scheduler::module_lr_scheduler::ModuleLearningRate,
    grads: GradientsParams,
) -> M {
    let before = collect(&module, None);
    let module = optim.step(lr, module, grads);
    let after: HashMap<ParamId, Tensor<1>> = collect(&module, None)
        .into_iter()
        .map(|p| (p.id, p.values))
        .collect();
    if let Some(device) = before.first().map(|p| p.values.device()) {
        let diffs: Vec<(usize, Tensor<1>)> = before
            .into_iter()
            .filter_map(|p| Some((p.group, after.get(&p.id)?.clone() - p.values)))
            .collect();
        let update = group_squares(diffs.iter().map(|(g, d)| (*g, d)), &device);
        LAST_UPDATE.with(|slot| *slot.borrow_mut() = Some(update));
    }
    module
}

/// One training step's statistics, on the device until a metric reads them.
pub struct TrainStats {
    /// `[3 GROUPS]`: per group, the squared norms of the gradient and of the weights, then
    /// of the previous step's update (zero, and flagged by `has_update`, before the first).
    squares: Tensor<1>,
    has_update: bool,
    host: OnceLock<Vec<f32>>,
}

impl TrainStats {
    /// The statistics of a step on `module` with `grads`.
    pub fn new<M: Module>(module: &M, grads: &GradientsParams) -> Option<Self> {
        let params = collect(module, Some(grads));
        let device = params.first()?.values.device();
        let grad = group_squares(
            params
                .iter()
                .filter_map(|p| Some((p.group, p.grad.as_ref()?))),
            &device,
        );
        let weight = group_squares(params.iter().map(|p| (p.group, &p.values)), &device);
        let update = LAST_UPDATE.with(|slot| slot.borrow_mut().take());
        let has_update = update.is_some();
        let update = update.unwrap_or_else(|| Tensor::zeros([GROUPS.len()], &device));
        Some(Self {
            squares: Tensor::cat(vec![grad, weight, update], 0),
            has_update,
            host: OnceLock::new(),
        })
    }

    /// The squares on the host, read once for every metric.
    fn host(&self) -> Result<&[f32], TensorReadError> {
        if let Some(values) = self.host.get() {
            return Ok(values);
        }
        let data = self.squares.clone().try_into_data()?;
        let values = data.try_into_vec::<f32>()?;
        Ok(self.host.get_or_init(|| values))
    }
}

/// Output of one training or validation step: the loss and verdicts that burn's loss and
/// accuracy metrics read, and for training steps the [`TrainStats`].
pub struct StepOutput {
    pub classification: ClassificationOutput,
    pub stats: Option<Arc<TrainStats>>,
}

impl ItemLazy for StepOutput {
    fn sync(self) -> Result<Self, ExecutionError> {
        Ok(Self {
            classification: self.classification.sync()?,
            stats: self.stats,
        })
    }
}

impl Adaptor<LossInput> for StepOutput {
    fn adapt(&self) -> LossInput {
        self.classification.adapt()
    }
}

impl Adaptor<AccuracyInput> for StepOutput {
    fn adapt(&self) -> AccuracyInput {
        self.classification.adapt()
    }
}

/// What a [`StatMetric`] reads.
pub struct StatsInput(Option<Arc<TrainStats>>);

impl Adaptor<StatsInput> for StepOutput {
    fn adapt(&self) -> StatsInput {
        StatsInput(self.stats.clone())
    }
}

/// Values this small or this varied read best in scientific notation; the entry burn logs
/// keeps the full value either way.
fn entry(state: &NumericMetricState, name: &MetricName, final_entry: bool) -> SerializedEntry {
    let format = FormatOptions::new(name.clone());
    let mut entry = match final_entry {
        true => state.compute_final(format),
        false => state.compute_update(format),
    };
    let (current, running) = (
        state.current_value().current(),
        state.running_value().current(),
    );
    entry.formatted = format!("epoch {running:.3e} - batch {current:.3e}");
    entry
}

/// Which statistic a [`StatMetric`] shows.
#[derive(Clone, Copy, Debug)]
pub enum Stat {
    /// The L2 norm of the group's gradient.
    GradNorm,
    /// The norm of the previous step's update over the norm of the group's weights.
    UpdateRatio,
}

/// One statistic of one group, as a numeric training metric. burn names its log file after the
/// metric, so the name must not contain a `/`.
#[derive(Clone)]
pub struct StatMetric {
    name: MetricName,
    stat: Stat,
    group: usize,
    state: NumericMetricState,
}

impl StatMetric {
    pub fn new(stat: Stat, group: usize) -> Self {
        let what = match stat {
            Stat::GradNorm => "Grad norm",
            Stat::UpdateRatio => "Update to weight",
        };
        Self {
            name: Arc::new(format!("{what} {}", GROUPS[group])),
            stat,
            group,
            state: NumericMetricState::new(),
        }
    }

    /// Every statistic of every group.
    pub fn all() -> Vec<Self> {
        [Stat::GradNorm, Stat::UpdateRatio]
            .into_iter()
            .flat_map(|stat| (0..GROUPS.len()).map(move |g| Self::new(stat, g)))
            .collect()
    }
}

impl Metric for StatMetric {
    type Input = StatsInput;

    fn update(
        &mut self,
        item: &StatsInput,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let n = GROUPS.len();
        let value = match &item.0 {
            None => None,
            Some(stats) => {
                let s = stats.host()?;
                match self.stat {
                    Stat::GradNorm => Some(s[self.group].sqrt()),
                    Stat::UpdateRatio if stats.has_update => {
                        Some((s[2 * n + self.group] / s[n + self.group]).sqrt())
                    }
                    Stat::UpdateRatio => None,
                }
            }
        };
        // A step without the value counts for nothing in the epoch's mean.
        match value {
            Some(v) => self.state.update(f64::from(v), 1),
            None => self.state.update(f64::NAN, 0),
        }
        Ok(entry(&self.state, &self.name, false))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(entry(&self.state, &self.name, true))
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for StatMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

/// The learning rate of each step, as burn's own metric but in scientific notation (burn's
/// shows two decimals, so the tail of a cosine schedule reads as 0.00).
#[derive(Clone)]
pub struct LrMetric {
    name: MetricName,
    state: NumericMetricState,
}

impl LrMetric {
    pub fn new() -> Self {
        Self {
            name: Arc::new("Learning rate".to_string()),
            state: NumericMetricState::new(),
        }
    }
}

impl Default for LrMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for LrMetric {
    type Input = ();

    fn update(
        &mut self,
        _item: &(),
        metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let lr = metadata.lr.as_ref().map_or(0.0, |lr| lr.base());
        self.state.update(lr, 1);
        Ok(entry(&self.state, &self.name, false))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(entry(&self.state, &self.name, true))
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for LrMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}
