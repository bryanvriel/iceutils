# Does fitting average velocity smooth more than fitting displacement?

**The exact average operator and the integral operator give the same reconstructed
instantaneous velocity when data, uncertainties, basis, and penalty are consistent.**
Changing from velocity averages to displacement integrals cannot by itself improve
temporal resolution. Treating a pair-average velocity as an instantaneous sample
at the pair midpoint is a separate approximation and can suppress fast changes.

Run the reproducible example from the repository root:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/iceutils-mpl \
  /Users/briel/miniforge3/envs/ice/bin/python \
  doc/compare_temporal_observation_operators.py \
  --output /tmp/temporal_operator_comparison
```

Use the `ice` environment on this machine; elsewhere substitute an interpreter
with iceutils's dependencies. `PYTHONPATH=.` ensures this checkout is imported;
the script prints its imported source path. `--trials` controls the number of
noise realizations (default 100). Outputs are a PNG/PDF comparison figure, JSON
metrics, and NPZ arrays with truth, reconstructions, noise, and penalty sweeps.

## Why the two interval formulations are equivalent

Let `D = diag(duration)` with durations in years for velocities in m/year. The
matrices, measurements, and standard deviations obey

\[
G_\mathrm{integral}=D G_\mathrm{average},\qquad
d_\mathrm{displacement}=D d_\mathrm{velocity},\qquad
\sigma_\mathrm{displacement}=D\sigma_\mathrm{velocity}.
\]

With inverse-standard-deviation weights, both weighted residual vectors are
identical. Adding the same penalty on the same coefficient vector therefore gives
the same optimization problem, including its instantaneous prediction matrix.
This holds for fixed **or variable** durations. For correlated observations,
transform the full covariance as `C_displacement = D @ C_velocity @ D.T`.

For a fixed duration and **unweighted** ridge fits, multiplying both `G` and data
by duration instead multiplies the data objective by duration squared. Keeping
the numerical penalty unchanged then changes its strength relative to the data.
To preserve that unweighted objective, scale the penalty by duration squared as
well. This can otherwise produce an apparent smoothing difference that is due
to regularization scaling, rather than additional information in displacements.
This example uses uncertainty weighting throughout, so it keeps the penalty fixed.

## Experiment

The true velocity is defined independently of the fitting basis:

\[
v(t)=1000 + 80\sin(2\pi t/180) + 60\cos(2\pi t/60)
      +250\exp[-(t-180)^2/(2\cdot6^2)]\quad\mathrm{m/year},
\]

where `t` is elapsed days. The Gaussian pulse has a full width at half maximum of
14.13 days. Its exact interval averages are calculated analytically using `erf`,
independently of the spline integration being evaluated.

The experiment compares fixed 12-day and 30-day intervals. Here "one month" means
exactly 30 days, rather than calendar-month boundaries. Both experiments have 55
pairs centered every 6 days and use the same independent Gaussian velocity noise
draws with standard deviation 8 m/year. These deliberately overlapping intervals
isolate duration from cadence; real shared acquisitions can cause correlated
errors, which are not simulated here. Holding velocity noise fixed also excludes
the product-dependent benefit of reduced velocity noise for longer baselines.

Every fit uses the same constant, 180-/60-day harmonics, and cubic B-splines with
6-day center spacing and scale. The spline coefficients receive a ridge penalty
of 0.001; constants and harmonics are unpenalized. The three fitting methods are:

1. **Average operator:** fit average velocities with the exact interval operator.
2. **Integral operator:** fit duration-scaled displacements and standard deviations
   with the integral operator on the same velocity basis.
3. **Midpoint sample fit:** fit the average velocities as samples at pair midpoints.

All three are evaluated as **instantaneous velocity** against the same truth on
a quarter-day grid over days 30–330. This excludes boundary extrapolation from
the accuracy calculation. The figure shows ensemble-mean reconstructions and the
16th–84th percentile noise range for the equivalent average/integral fits. The
bottom panels show sensitivity to penalty for one paired noise realization.

## Results with the default settings

RMSE below is the square root of mean squared instantaneous-velocity error across
both evaluation dates and all 100 noise realizations, in m/year.

| Pair duration | Exact average operator | Integral operator | Midpoint sample fit |
| --- | ---: | ---: | ---: |
| 12 days | 8.69 | 8.69 | 10.97 |
| 30 days | 19.55 | 19.55 | 31.66 |

Across all trials, average and integral predictions differ by less than
`2.4e-11 m/year`. The script asserts their coefficient and prediction equivalence.
At day 180, the true instantaneous velocity is 1310 m/year:

| Pair duration | Mean average/integral reconstruction | Mean midpoint reconstruction |
| --- | ---: | ---: |
| 12 days | 1279.60 | 1258.36 |
| 30 days | 1218.49 | 1158.92 |

Even the exact interval formulations under-recover the narrow peak under this
basis, sampling, and regularization. Noise-free fits with the same penalty still
have RMSEs of 5.08 and 17.89 m/year for 12 and 30 days, respectively. That residual
reflects finite sampling, basis approximation, and penalty bias. It does not
establish a universal resolution limit for every sampling pattern or model.

The JSON also separates empirical ensemble bias and variability. Their squared
RMS values sum to squared ensemble RMSE; the noise-free penalized result is
reported separately. The penalty sweep shows that weaker regularization can
reduce peak attenuation while amplifying noise, particularly for the 30-day pairs.
The penalty here is an illustrative fixed choice, not an optimum guaranteed to
apply to another dataset.

## Interpreting smoothing

The **observations** are temporally smoothed by their acquisition interval.
For a harmonic of period `T` and a centered interval of duration `L`, its observed
amplitude is multiplied by

\[
\operatorname{sinc}(L/T)=\frac{\sin(\pi L/T)}{\pi L/T}.
\]

For the example's 60-day oscillation, 12-day averages retain 93.5% of its amplitude
and 30-day averages retain 63.7%. A midpoint-sample fit can attribute that reduced
observed amplitude directly to the underlying velocity. Both exact interval
operators account for the attenuation when fitting the underlying coefficients.

Accounting for attenuation does not remove the uncertainty it creates. Components
with weak interval response are harder to estimate and tend to be suppressed by
regularization. An unrestricted harmonic with period exactly equal to a fixed
pair duration has zero interval response; converting those averages into
displacements cannot recover it. Irregular durations can change that null-space
structure but do not change average/integral equivalence for the same pairs.

Use the average operator when your measurements and uncertainties are already
velocities, and the integral operator when they are displacements. For the same
pairs, compare temporal resolution through acquisition duration/cadence, basis
resolution, uncertainty/correlation, and regularization. The operator's physical
normalization alone does not add or remove reconstruction smoothing.
