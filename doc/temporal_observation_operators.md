# Temporal observation operators

A temporal model defines a quantity independently of how it is observed:

\[
q(t) = \sum_j m_j b_j(t).
\]

Choose that quantity and its units before fitting. For example, if `q` is
instantaneous component velocity in metres/year, samples and averages have units
metres/year, and integrals in years have units metres. If `q` is displacement in
metres, endpoint differences are pair displacements in metres. Integrating a
displacement basis is a different operation from subtracting its endpoints.

| `observation` | Matrix element | Observation units |
| --- | --- | --- |
| `sample` (default) | `b_j(t_i)` | quantity |
| `integral` | integral of `b_j` over `[t_i, t_end_i]` | quantity × `time_units` |
| `average` | integral divided by full elapsed duration | quantity |
| `difference` | `b_j(t_end_i) - b_j(t_i)` | quantity |

The operators do not convert physical units or infer whether the modeled quantity
is velocity or displacement. Fit vector components separately; speed magnitude
does not obey this linear observation model.

For a reproducible 12-day versus 30-day comparison of reconstruction accuracy,
see [the synthetic reconstruction example](temporal_operator_reconstruction_example.md).
It compares exact averages, duration-scaled integrals, and the approximation of
treating pair-average velocities as midpoint samples.

## Fit pair-average velocities and predict on another grid

```python
import datetime as dt
import numpy as np
import iceutils as ice

origin = dt.datetime(2020, 1, 1)
starts = np.array([origin + dt.timedelta(days=k) for k in range(0, 1100, 10)])
ends = starts + dt.timedelta(days=24)

# The modeled quantity is instantaneous velocity in metres/year.
model = ice.tseries.build_temporal_model(
    starts,
    poly=1,
    periods=[1.0],
    isplines=[8, 4],
    observation="average",
    t_end=ends,
    time_units="years",
)
solver_config = dict(
    solver_type="ridge", reg_indices=model.reg_indices, penalty=0.1, n_min=20
)
solver = ice.tseries.select_solver(**solver_config)

# Replace these synthetic observations and uncertainties with measured values.
truth = np.zeros(model.npar)
truth[0] = 100
velocity = model.G @ truth
sigma_velocity = np.ones(len(starts))
m = model.invert(solver, velocity, wgt=1 / sigma_velocity)

grid = np.array([origin + dt.timedelta(days=k) for k in range(1100)])
instantaneous = model.predict(m, t=grid, observation="sample")
fitted_pairs = model.predict(m)  # Same averages used for fitting.

# Accumulated displacement relative to a chosen epoch, in metres.
displacement = model.predict(
    m,
    t=np.full(len(grid) - 1, origin),
    t_end=grid[1:],
    observation="integral",
    time_units="years",
)
```

`Model.G` and every `design_matrix()` result have shape `(N, P)`, including a
single observation. `Model.design_matrix()` with no arguments returns `G`.
New epochs default to the model's stored operator; explicitly request `sample`
for instantaneous predictions from an interval model. New interval starts need
matching new ends. With no new starts, an operator change reuses stored endpoints.

`predict()` returns `full`, `secular`, `seasonal`, `transient`, and `step`, plus
`sigma` when a coefficient covariance is available. Basis types without a named
partition contribute to `full`. Prediction uncertainty uses the new observation
matrix: `sqrt(diag(G_prediction @ Cm @ G_prediction.T))`. This propagates the
covariance returned by the solver; sparse solvers' default identity matrices are
placeholders, not calibrated posterior uncertainty.

Each matrix retains the original coefficient ordering and `reg_indices`.
Prediction reuses the existing collection: phase origins, spline centers, scales,
and active bounds do not change with a new grid. A convenience builder spans all
interval endpoints. A userfile builder receives their sorted unique union once;
its `computeCm()` uses the same loaded module and collection. Sample userfiles
continue receiving the original observation epochs.

## Time, support, and coefficient units

Inputs accept naive Python dates/datetimes, NumPy datetimes (microsecond precision),
and legacy decimal years. Decimal years are converted with the existing calendar
conversion before subtracting timestamps; that legacy conversion truncates to
seconds. Prefer datetimes for short intervals. Timezone-aware dates, NaT, and
nonfinite numeric times are rejected. Interval arrays must be nonempty, one
dimensional, equal in shape, and strictly increasing within each pair. Scalars
become single rows; there is no endpoint broadcasting. Overlap and unordered rows
are allowed and row order is retained.

`time_units` controls elapsed integration time, not the basis coordinates.
The default year is 365.25 days (31,557,600 seconds), matching the existing Pint
basis convention. Automatic polynomials, harmonic periods, and spline scales
remain in years even when integrals are requested in days or seconds. A leap-year
pair covering 366 days therefore lasts `366 / 365.25` model years.

The shared integration wrapper clips to each basis's active `[tmin, tmax]` range.
An average divides that clipped integral by the **original full interval duration**,
so time outside the active range contributes zero. Endpoint differences evaluate
the existing masked basis at each endpoint, including boundary jumps. Averages
approach point values at continuous epochs; at a jump their limit depends on the
interval's placement. Fitting-window bounds include an interval only if both
endpoints lie inside the window; intervals are never shortened to fit it.

Analytic integration supports constants/steps, nonnegative integer polynomial
powers, sine/cosine, B-splines, and integrated B-splines. It preserves signed spline
scales and integrates the I-spline plateau beyond the transition. Other functions,
including periodic B-splines and noninteger powers, raise `NotImplementedError`
for integral/average matrices; their existing point evaluations and endpoint
differences remain available. Point evaluators themselves are unchanged.

Coefficients keep the units implied by the original basis:

- Constants, harmonics, and ordinary dimensionless B-splines have coefficients
  in quantity units.
- A polynomial written as `((t - tref) / norm / tau)**p` has numerical coefficients
  in quantity units, with the physical trend obtained by dividing by the stated
  time scale to power `p`. For the default `tau=1`, those trend units are
  quantity per basis-time-unit to power `p`.
- An iceutils I-spline includes its numerical `scale`: its derivative with
  respect to the basis time coordinate is a B-spline. Its amplitude is therefore
  quantity per basis-time-unit. In a displacement model it represents a velocity
  excursion; in a velocity model it represents a smooth speed transition.

## Weights, penalties, and null spaces

`wgt` means inverse standard deviation, **not** inverse variance. For interval
`Model.invert()`, nonfinite observations or weights and zero weights are masked;
negative weights are rejected. Existing sample masking is retained. Direct solver
calls remain responsible for masking and checking identifiability themselves.

Converting pair-average velocity to displacement requires transforming its
uncertainty too, with duration in units consistent with velocity:

```python
duration = model.durations  # years in the velocity example above
d_displacement = velocity * duration
w_displacement = (1 / sigma_velocity) / duration
G_displacement = model.design_matrix(observation="integral")
status, m_displacement, Cm = solver.invert(
    G_displacement,
    d_displacement,
    wgt=w_displacement,
)
assert status == ice.SUCCESS
np.testing.assert_allclose(m_displacement, m, atol=1e-8)
```

The weighted data objective is unchanged, so retain the same coefficient penalty.
Ridge adds `penalty` directly to the chosen normal-matrix diagonal entries; it
does not square it. An explicit `regMat` replaces that diagonal penalty and is a
parameter **precision/penalty matrix**, not a covariance. For ridge the normal
system is `G.T @ W.T @ W @ G + regMat`. Positive-semidefinite singular penalties
are permitted if observations constrain their null space. Penalties act in
coefficient coordinates and must be reconsidered if the basis normalization or
modeled quantity changes. Changing only the observation operator does not change
coefficient units. Changing observation units without transforming the uncertainty
changes the objective and generally changes the appropriate penalty strength.

Before interval fitting with built-in solvers, `Model.invert()` checks for null
directions left unconstrained by the weighted data and the actual penalty.
Endpoint differences remove a global constant displacement. Omit it explicitly,
for example with `min_poly=1`, or provide a constraint. Columns are never removed
automatically. Rank is assessed after column normalization to reduce dependence
on coefficient units; very poorly conditioned systems can still be unreliable.

Lasso permits overlapping penalized columns but rejects unpenalized null
directions. Individual coefficients in overlapping dictionaries can remain
prior-dependent even when predictions are stable. OMP treats the entire
dictionary as eligible for sparse selection and has no separate unpenalized
subset. Robust fitting can further reduce rank when it selects inliers; the
pre-fit check does not guarantee that every inlier subset is identifiable.
Custom solver objects retain their existing delegation behavior and supply their
own identifiability policy. Existing sparse solver objectives and uncertainty
approximations are unchanged by this feature.

## Stack inversion compatibility

`inversion()` and `inversion_points()` now select penalized columns from
`model.reg_indices` when it is nonempty, falling back to `model.itransient`
otherwise. Previously, stack inversion always selected `model.itransient`.
The built-in model partitions mark both B-splines and integrated B-splines for
regularization; ordinary B-splines belong to the seasonal partition and were
therefore excluded from the previous selection.

This changes existing results in the following cases, even for the default
`sample` observation operator:

- Ridge without an explicit prior precision matrix now penalizes B-splines.
  The fitted coefficients, component predictions, and uncertainty can change
  for the same data, basis, and `regParam`.
- Lasso now includes B-splines in its sparse penalty, including when a prior
  precision matrix is also supplied.
- A mixed model containing splines and logarithmic transient terms now selects
  the spline regularization indices; logarithmic terms are no longer penalized
  merely because they belong to the transient partition.

For ridge with an explicit precision matrix (`prior_cov=True` with a userfile
that defines `computeCm()`), that matrix still replaces the diagonal penalty,
so the index selection does not change its penalty. Least squares and OMP do
not use this index selection. Models containing only polynomial, harmonic,
step, and integrated B-spline terms retain the same penalized columns.

When upgrading an existing B-spline or mixed transient analysis, compare fits
and uncertainties and reconsider `regParam` before replacing published or
archived results. A direct matrix solver still accepts an explicit
`reg_indices` selection when reproducing an earlier penalty is necessary.

## Mixed observations and correlated errors

Each Model stores one observation kind. Mixed fits can concatenate matrices from
one fixed basis. Put data and covariance in units appropriate to each row; do not
rebuild separate automatic bases for each observation type.

```python
G_sample = model.design_matrix(grid[:3], observation="sample")
G_average = model.design_matrix()  # Stored pair averages.
G_mixed = np.vstack((G_sample, G_average))
# Concatenate the corresponding data and inverse standard deviations in this
# same row order, then pass them to solver.invert(G_mixed, d_mixed, wgt=w_mixed).
```

Shared acquisitions can correlate errors. No full-covariance solver API is added;
whiten observations externally with a Cholesky factor:

```python
from scipy.linalg import solve_triangular


# C is the positive-definite observation covariance, with invalid rows removed.
def invert_correlated(G, d, C, solver):
    L = np.linalg.cholesky(C)
    G_white = solve_triangular(L, G, lower=True)
    d_white = solve_triangular(L, d, lower=True)
    return solver.invert(G_white, d_white)
```

For duration scaling with `D = diag(duration)`, transform covariance as
`C_displacement = D @ C_velocity @ D.T`. Whiten each representation using its own
covariance; the coefficient estimate and covariance should agree under the same
penalty.

## Reproducibility and limits

`model.metadata()` returns a JSON-compatible snapshot containing observation kind,
normalized ISO endpoints, durations and integration units, fitting window, actual
per-column basis definitions (including normalization), partitions, and penalty
indices. Custom basis classes require their own serialization. The metadata is a
record, not a new automatic model-loading format. Save it with coefficients,
solver configuration (including explicit precision matrices), modeled quantity
and physical units, software version, source observation identifiers, and
uncertainties/covariance. For example, use `json.dump(model.metadata(), stream)`
and save numerical solver arrays and coefficients with `np.savez`.

This feature does not change Stack storage or stack inversion. `inversion()` and
`inversion_points()` reject interval Models; use `Model.invert()` or matrix
solvers directly. Downloaders, radar projections, nuisance models, tide
corrections, and weighted mosaic operators remain outside this API. Nominal
monthly/annual coverage dates alone do not define a uniform average. Interval
averaging models attenuation but cannot recover variability lost in the data.
