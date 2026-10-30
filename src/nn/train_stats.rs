//! What the loss does not show about a training run: the learning rate, the gradient norm of
//! each part of the network, and how far each optimizer step moves that part's weights
//! relative to their size. All are logged as training metrics next to the loss.
//!
//! The parts are the [`GROUPS`], by the model's top-level field names (the NNUE names: the
//! first layer's tables, the heads, and everything in between). The statistics are sampled
//! every [`DEFAULT_EVERY`] steps (see [`set_every`]), computed on the device as part of the
//! step and read back on burn's metric thread. A step's update is only known once the
//! optimizer has run, after the step's output was made, so a sampled step reports the update
//! of the step before it.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use burn::module::{ModuleVisitor, Param};
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

/// One float parameter: its group and its values flattened, without autodiff.
struct Flat {
    group: usize,
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

/// Training steps between two samples of the statistics, 0 for none (see [`set_every`]).
static EVERY: AtomicUsize = AtomicUsize::new(DEFAULT_EVERY);

/// The default of [`set_every`]: the statistics cost about five times a step's own work
/// (measured on the medium NNUE at batch 2048: 45 ms against 7.5 ms per step), so they are
/// sampled rather than taken every step.
pub const DEFAULT_EVERY: usize = 64;

/// Samples the statistics every `every` training steps (the first step of a run and every
/// `every`-th after it), or never with 0. Steps in between log the last sampled value without
/// counting it toward the epoch's mean.
pub fn set_every(every: usize) {
    EVERY.store(every, Ordering::Relaxed);
}

fn sampled(step: usize) -> bool {
    match EVERY.load(Ordering::Relaxed) {
        0 => false,
        every => step % every == 0,
    }
}

thread_local! {
    /// Optimizer steps taken on this thread (training steps and optimizer steps alternate on
    /// the training thread).
    static STEPS: Cell<usize> = const { Cell::new(0) };
    /// The squared update of each group by the last optimizer step, when the next step is
    /// sampled.
    static LAST_UPDATE: RefCell<Option<Tensor<1>>> = const { RefCell::new(None) };
}

/// Each group's tensors concatenated into one, so that a group costs one concatenation, one
/// product and one sum rather than those per parameter.
fn by_group(items: impl IntoIterator<Item = (usize, Tensor<1>)>) -> [Option<Tensor<1>>; 3] {
    let mut parts: [Vec<Tensor<1>>; GROUPS.len()] = Default::default();
    for (g, x) in items {
        parts[g].push(x);
    }
    parts.map(|p| (!p.is_empty()).then(|| Tensor::cat(p, 0)))
}

/// `[GROUPS]`: each group's sum of squares, 0 for an empty group.
fn squares(groups: [Option<Tensor<1>>; 3], device: &Device) -> Tensor<1> {
    Tensor::cat(
        groups
            .into_iter()
            .map(|x| match x {
                Some(x) => (x.clone() * x).sum(),
                None => Tensor::zeros([1], device),
            })
            .collect(),
        0,
    )
}

/// The optimizer step, its update's squared size per group recorded when the next training
/// step is sampled (see [`TrainStats`]); what `TrainStep::optimize` does for the models.
pub fn optimize_recorded<M: Module>(
    module: M,
    optim: &mut burn::optim::ModuleOptimizer,
    lr: burn::lr_scheduler::module_lr_scheduler::ModuleLearningRate,
    grads: GradientsParams,
) -> M {
    let step = STEPS.with(|steps| steps.replace(steps.get() + 1));
    if !sampled(step + 1) {
        return optim.step(lr, module, grads);
    }
    let values = |m: &M| by_group(collect(m, None).into_iter().map(|p| (p.group, p.values)));
    let before = values(&module);
    let module = optim.step(lr, module, grads);
    // The visit order is the module's, so the groups line up parameter by parameter.
    let diffs = values(&module)
        .into_iter()
        .zip(before)
        .map(|(after, before)| Some(after? - before?));
    let device = collect(&module, None).first().map(|p| p.values.device());
    if let Some(device) = device {
        let update = squares(
            diffs.collect::<Vec<_>>().try_into().expect("three groups"),
            &device,
        );
        LAST_UPDATE.with(|slot| *slot.borrow_mut() = Some(update));
    }
    module
}

/// One sampled training step's statistics, on the device until a metric reads them.
pub struct TrainStats {
    /// `[3 GROUPS]`: per group, the squared norms of the gradient and of the weights, then
    /// of the previous step's update (zero, and flagged by `has_update`, before the first).
    squares: Tensor<1>,
    has_update: bool,
    host: OnceLock<Vec<f32>>,
}

impl TrainStats {
    /// The statistics of a step on `module` with `grads`, if this step is sampled.
    pub fn new<M: Module>(module: &M, grads: &GradientsParams) -> Option<Self> {
        if !sampled(STEPS.with(Cell::get)) {
            return None;
        }
        let params = collect(module, Some(grads));
        let device = params.first()?.values.device();
        let grad = squares(
            by_group(
                params
                    .iter()
                    .filter_map(|p| Some((p.group, p.grad.clone()?))),
            ),
            &device,
        );
        let weight = squares(
            by_group(params.into_iter().map(|p| (p.group, p.values))),
            &device,
        );
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
/// accuracy metrics read, the loss's terms, and for sampled training steps the
/// [`TrainStats`].
pub struct StepOutput {
    pub classification: ClassificationOutput,
    pub stats: Option<Arc<TrainStats>>,
    pub parts: Arc<LossParts>,
}

impl ItemLazy for StepOutput {
    fn sync(self) -> Result<Self, ExecutionError> {
        let parts = LossParts::new(self.parts.values.clone().without_autodiff(), self.parts.n);
        Ok(Self {
            classification: self.classification.sync()?,
            stats: self.stats,
            parts: Arc::new(parts),
        })
    }
}

/// The terms of a step's loss before their weights: the win/draw/loss cross-entropy (over the
/// positions with a target), the value's squared error and the policy's cross-entropy, and the
/// last less its targets' entropy (the KL divergence left to learn), on the device until a
/// metric reads them.
pub struct LossParts {
    values: Tensor<1>,
    /// Positions in the step.
    n: usize,
    host: OnceLock<Vec<f32>>,
}

impl LossParts {
    /// The metrics' names, in the order of the values.
    pub const NAMES: [&str; 4] = ["WDL loss", "Value loss", "Policy loss", "Policy KL"];

    /// `values` are `[4]`, in [`LossParts::NAMES`] order, over a step of `n` positions.
    pub fn new(values: Tensor<1>, n: usize) -> Self {
        Self {
            values,
            n,
            host: OnceLock::new(),
        }
    }

    /// The values on the host, read once for every metric.
    fn host(&self) -> Result<&[f32], TensorReadError> {
        if let Some(values) = self.host.get() {
            return Ok(values);
        }
        let values = self.values.clone().try_into_data()?.try_into_vec::<f32>()?;
        Ok(self.host.get_or_init(|| values))
    }
}

impl Adaptor<Arc<LossParts>> for StepOutput {
    fn adapt(&self) -> Arc<LossParts> {
        self.parts.clone()
    }
}

/// One term of the loss (see [`LossParts`]), as a numeric training and validation metric.
#[derive(Clone)]
pub struct LossPartMetric {
    name: MetricName,
    part: usize,
    state: NumericMetricState,
}

impl LossPartMetric {
    pub fn new(part: usize) -> Self {
        Self {
            name: Arc::new(LossParts::NAMES[part].to_string()),
            part,
            state: NumericMetricState::new(),
        }
    }
}

impl Metric for LossPartMetric {
    type Input = Arc<LossParts>;

    fn update(
        &mut self,
        item: &Arc<LossParts>,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let value = item.host()?[self.part];
        self.state.update(f64::from(value), item.n);
        Ok(self
            .state
            .compute_update(FormatOptions::new(self.name.clone()).precision(4)))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self
            .state
            .compute_final(FormatOptions::new(self.name.clone()).precision(4)))
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

impl Numeric for LossPartMetric {
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
    /// The last sampled value, shown on the steps in between.
    last: f64,
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
            last: f64::NAN,
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
        // A step without a sample shows the last one and counts for nothing in the epoch's
        // mean.
        match value {
            Some(v) => {
                self.last = f64::from(v);
                self.state.update(self.last, 1);
            }
            None => self.state.update(self.last, 0),
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
