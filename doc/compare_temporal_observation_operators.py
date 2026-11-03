"""Compare instantaneous velocity recovery from fixed-duration image pairs.

Run from a checkout with its source on PYTHONPATH, for example::

    PYTHONPATH=. python doc/compare_temporal_observation_operators.py \
        --output /tmp/temporal_operator_comparison

The modeled quantity is velocity in m/year. Average and integral fits describe
the same observations in velocity and displacement units, respectively. The
midpoint fit intentionally treats pair averages as instantaneous velocities.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

import iceutils as ice

ORIGIN = dt.datetime(2020, 1, 1)
METHODS = ("average", "integral", "midpoint")
COLORS = {"average": "#18856b", "integral": "#6359bb", "midpoint": "#d55e32"}
PENALTY = 0.001
SIGMA_VELOCITY = 8.0


def velocity(days):
    """Return independent analytic truth in m/year for elapsed days."""
    seasonal = 80 * np.sin(2 * np.pi * days / 180)
    oscillation = 60 * np.cos(2 * np.pi * days / 60)
    pulse = 250 * np.exp(-0.5 * ((days - 180) / 6) ** 2)
    return 1000 + seasonal + oscillation + pulse


def average_velocity(start, end):
    """Return exact pair-average truth, integrating the Gaussian analytically."""
    duration = end - start
    midpoint = (start + end) / 2
    seasonal = 80 * np.sin(2 * np.pi * midpoint / 180) * np.sinc(duration / 180)
    oscillation = 60 * np.cos(2 * np.pi * midpoint / 60) * np.sinc(duration / 60)
    pulse_integral = (
        250
        * 6
        * np.sqrt(np.pi / 2)
        * (erf((end - 180) / (6 * np.sqrt(2))) - erf((start - 180) / (6 * np.sqrt(2))))
    )
    return 1000 + seasonal + oscillation + pulse_integral / duration


def dates(days):
    """Convert elapsed days to naive datetime objects without decimal years."""
    return np.array([ORIGIN + dt.timedelta(days=float(day)) for day in days])


def compare(output, trials=100):
    """Run a paired-noise experiment and save figures, metrics, and raw arrays.

    Parameters
    ----------
    output : pathlib.Path
        Directory for PNG/PDF figures, JSON metrics, and NPZ numerical results.
    trials : int, optional
        Independent synthetic noise realizations, paired across methods and
        durations. Default: 100. Noise is independent between observations;
        real overlapping image pairs can instead have correlated errors.
    """
    output.mkdir(parents=True, exist_ok=True)
    print("iceutils source:", ice.__file__)

    # Hold cadence, basis, noise in velocity units, and penalty fixed. Only the
    # pair duration changes. Evaluate interior dates to avoid boundary effects.
    midpoint = np.arange(18.0, 343.0, 6.0)
    grid = np.linspace(0, 360, 1441)
    interior = (grid >= 30) & (grid <= 330)
    truth = velocity(grid)
    noise = np.random.default_rng(31).normal(
        0, SIGMA_VELOCITY, size=(trials, len(midpoint))
    )
    basis = ice.tseries.timefn.TimefnCollection()
    fnmap = ice.tseries.timefn.fnmap
    basis.append(fnmap["constant"]())
    for period in (180, 60):
        basis.append(fnmap["periodic"](tref=ORIGIN, units="days", period=period))
    for center in np.arange(-12.0, 373.0, 6.0):
        basis.append(
            fnmap["bspline"](tref=dates([center])[0], units="days", scale=6, order=3)
        )

    figure, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    metrics = {
        "settings": {
            "seed": 31,
            "trials": trials,
            "cadence_days": 6,
            "sigma_velocity_m_per_year": SIGMA_VELOCITY,
            "penalty": PENALTY,
            "spline_spacing_days": 6,
            "evaluation_days": [30, 330],
            "pulse_fwhm_days": float(12 * np.sqrt(2 * np.log(2))),
        },
        "intervals": {},
    }
    arrays = {
        "grid_days": grid,
        "truth": truth,
        "midpoint_days": midpoint,
        "noise": noise,
    }
    penalties = np.logspace(-5, -1, 9)

    for column, duration in enumerate((12, 30)):
        start, end = dates(midpoint - duration / 2), dates(midpoint + duration / 2)
        averages = average_velocity(midpoint - duration / 2, midpoint + duration / 2)
        models = {
            "average": ice.tseries.Model(
                start, basis, observation="average", t_end=end
            ),
            "integral": ice.tseries.Model(
                start, basis, observation="integral", t_end=end
            ),
            "midpoint": ice.tseries.Model(dates(midpoint), basis),
        }
        average_model = models["average"]
        integral_model = models["integral"]
        duration_years = average_model.durations
        G_sample = average_model.design_matrix(dates(grid), observation="sample")
        wgts = np.full(len(midpoint), 1 / SIGMA_VELOCITY)
        np.testing.assert_allclose(
            integral_model.G, average_model.G * duration_years[:, None], atol=1e-14
        )

        # Noise-free and noisy fits use identical numerical settings. Scaling
        # both the observations and sigma makes the integral objective equal
        # to the average objective, including the coefficient penalty.
        predictions = {}
        noiseless = {}
        coefficients = {}
        sweep = {}
        interval_metrics = {}
        for method in METHODS:
            model = models[method]
            scale = duration_years if method == "integral" else np.ones(len(midpoint))
            solver = ice.tseries.select_solver(
                "ridge", reg_indices=model.reg_indices, penalty=PENALTY, n_min=5
            )
            m = model.invert(solver, averages * scale, wgt=wgts / scale)
            noiseless[method] = G_sample @ m
            coefficients[method] = np.empty((trials, model.npar))
            for trial in range(trials):
                m = model.invert(
                    solver, (averages + noise[trial]) * scale, wgt=wgts / scale
                )
                coefficients[method][trial] = m
            predictions[method] = coefficients[method] @ G_sample.T

            # RMS over dates and realizations measures true instantaneous
            # recovery; observational residuals alone cannot answer this question.
            error = predictions[method][:, interior] - truth[interior]
            ensemble_mean = predictions[method].mean(axis=0)
            bias = ensemble_mean[interior] - truth[interior]
            variance = predictions[method][:, interior].var(axis=0)
            at_peak = np.argmin(np.abs(grid - 180))
            interval_metrics[method] = {
                "ensemble_rmse_m_per_year": float(np.sqrt(np.mean(error**2))),
                "ensemble_bias_rms_m_per_year": float(np.sqrt(np.mean(bias**2))),
                "ensemble_std_rms_m_per_year": float(np.sqrt(np.mean(variance))),
                "noiseless_rmse_m_per_year": float(
                    np.sqrt(
                        np.mean((noiseless[method][interior] - truth[interior]) ** 2)
                    )
                ),
                "mean_peak_day180_m_per_year": float(ensemble_mean[at_peak]),
                "true_peak_day180_m_per_year": float(truth[at_peak]),
            }

            # A fixed noise realization exposes the regularization tradeoff.
            # This sweep is diagnostic, not a tuning rule based on real truth.
            sweep[method] = []
            for penalty in penalties:
                solver = ice.tseries.select_solver(
                    "ridge", reg_indices=model.reg_indices, penalty=penalty, n_min=5
                )
                m = model.invert(
                    solver, (averages + noise[0]) * scale, wgt=wgts / scale
                )
                error = (G_sample @ m - truth)[interior]
                sweep[method].append(np.sqrt(np.mean(error**2)))

        max_difference = np.max(
            np.abs(predictions["average"] - predictions["integral"])
        )
        np.testing.assert_allclose(
            coefficients["average"], coefficients["integral"], rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            predictions["average"], predictions["integral"], rtol=1e-9, atol=1e-6
        )
        interval_metrics["max_average_integral_prediction_difference"] = float(
            max_difference
        )
        metrics["intervals"][str(duration)] = interval_metrics
        arrays[f"average_truth_{duration}d"] = averages

        for row in (0, 1):
            ax = axes[row, column]
            ax.plot(
                grid,
                truth,
                color="#202936",
                linewidth=2,
                label="True instantaneous velocity",
            )
            for method in METHODS:
                mean = predictions[method].mean(axis=0)
                style = "--" if method == "integral" else "-"
                ax.plot(
                    grid,
                    mean,
                    style,
                    color=COLORS[method],
                    linewidth=1.8,
                    label=method.capitalize()
                    + (" sample fit" if method == "midpoint" else " operator"),
                )
            ax.scatter(
                midpoint,
                averages,
                s=12,
                facecolors="none",
                edgecolors="#89939e",
                label="Exact pair-average observations",
                zorder=3,
            )
            lower, upper = np.percentile(predictions["average"], [16, 84], axis=0)
            ax.fill_between(
                grid,
                lower,
                upper,
                color=COLORS["average"],
                alpha=0.15,
                label="Average/integral 16–84% noise range",
            )
            ax.set_xlim((30, 330) if row == 0 else (145, 215))
            ax.set_ylabel("Velocity (m/year)")
            ax.set_xlabel("Elapsed days")
            ax.grid(alpha=0.15)
        axes[0, column].set_title(f"{duration}-day pairs · starts every 6 days")
        axes[1, column].set_title("14.1-day FWHM velocity pulse")

        ax = axes[2, column]
        for method in METHODS:
            style = "--" if method == "integral" else "-"
            ax.semilogx(
                penalties,
                sweep[method],
                style,
                color=COLORS[method],
                label=method.capitalize(),
            )
            arrays[f"prediction_{method}_{duration}d"] = predictions[method]
            arrays[f"noiseless_{method}_{duration}d"] = noiseless[method]
            arrays[f"sweep_rmse_{method}_{duration}d"] = np.array(sweep[method])
        ax.axvline(PENALTY, color="#89939e", linestyle=":", label="Penalty used above")
        ax.set_xlabel("Ridge penalty (same coefficient coordinates)")
        ax.set_ylabel("Instantaneous velocity RMSE (m/year)")
        ax.set_title("Penalty sensitivity · one paired noise realization")
        ax.grid(alpha=0.15)
        print(f"\n{duration}-day pairs:")
        for method in METHODS:
            result = interval_metrics[method]
            print(
                f"  {method:8s}  ensemble RMSE = {result['ensemble_rmse_m_per_year']:.3f}, "
                f"noiseless RMSE = {result['noiseless_rmse_m_per_year']:.3f} m/year"
            )
        print(
            f"  max average/integral prediction difference = {max_difference:.3g} m/year"
        )

    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[2, 0].legend(fontsize=8)
    figure.suptitle(
        "Interval averages and integrals recover the same velocity\n"
        "Treating a pair average as a midpoint sample can attenuate fast motion",
        fontsize=14,
    )
    figure.savefig(output / "velocity_reconstruction_comparison.png", dpi=180)
    figure.savefig(output / "velocity_reconstruction_comparison.pdf")
    plt.close(figure)
    arrays["penalties"] = penalties
    np.savez_compressed(output / "comparison_arrays.npz", **arrays)
    (output / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print("\nOutputs:", output.resolve())
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("temporal_operator_comparison")
    )
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    compare(args.output, args.trials)
