use crate::groove::gradient::{ForwardFiniteDiff, CentralFiniteDiff, GradientFinder, ForwardFiniteDiffImmutable, CentralFiniteDiffImmutable, GradientFinderImmutable};
use crate::groove::vars::{RelaxedIKVars};
use optimization_engine::{constraints::*, panoc::*, *};
use crate::groove::objective_master::ObjectiveMaster;

pub struct OptimizationEngineOpen {
    dim: usize,
    cache: PANOCCache
}
impl OptimizationEngineOpen {
    pub fn new(dim: usize) -> Self {
        let mut cache = PANOCCache::new(dim, 1e-14, 10);
        OptimizationEngineOpen { dim, cache }
    }

    /// Run PANOC. The `_max_iter` argument is kept for backwards compat but
    /// the per-call iter limit and FPR tolerance now come from `vars.opt_opts`
    /// (which itself is read from the YAML's `panoc_*` fields, defaulting to
    /// the values in `OptimizerOptions::default()`).
    pub fn optimize(&mut self, x: &mut [f64], v: &RelaxedIKVars, om: &ObjectiveMaster, _max_iter: usize) {
        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let (my_obj, my_grad) = om.gradient(u, v);
            for i in 0..my_grad.len() {
                grad[i] = my_grad[i];
            }
            Ok(())
        };

        let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
            *c = om.call(u, v);
            Ok(())
        };

        let bounds = Rectangle::new(Option::from(v.robot.lower_joint_limits.as_slice()), Option::from(v.robot.upper_joint_limits.as_slice()));

        let problem = Problem::new(&bounds, df, f);
        let mut panoc = PANOCOptimizer::new(problem, &mut self.cache)
            .with_max_iter(v.opt_opts.panoc_max_iter)
            .with_tolerance(v.opt_opts.panoc_tolerance);

        let _status = panoc.solve(x);
    }
}
