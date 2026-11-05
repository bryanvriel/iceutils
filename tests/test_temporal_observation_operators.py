"""Numerical contracts for fixed-basis temporal observation operators."""

import datetime as dt
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import solve_triangular

import iceutils as ice
from iceutils.tseries import Model, build_temporal_model, build_temporal_model_fromfile
from iceutils.tseries import select_solver
from iceutils.tseries.timefn import TimefnCollection, fnmap
from iceutils.tseries.timefn.CenteredBasisFn import ureg

ORIGIN = dt.datetime(2020, 1, 1)


def dates(days):
    return np.array(
        [ORIGIN + dt.timedelta(days=float(day)) for day in np.atleast_1d(days)]
    )


def collection(*functions):
    result = TimefnCollection()
    for function in functions:
        result.append(function)
    return result


def polynomial(order=1, minorder=0):
    return collection(
        fnmap["poly"](tref=ORIGIN, order=order, minorder=minorder, units="days")
    )


@pytest.mark.parametrize("operator", ["sample", "integral", "average", "difference"])
def test_polynomial_analytic(operator):
    basis = polynomial(order=4)
    a, b = np.array([0.0, 3.0, -2.0]), np.array([2.0, 4.0, 1.0])
    ends = None if operator == "sample" else dates(b)
    G = basis.design_matrix(dates(a), operator, ends, "days")
    for power in range(5):
        if operator == "sample":
            expected = a**power
        elif operator == "difference":
            expected = b**power - a**power
        else:
            expected = (b ** (power + 1) - a ** (power + 1)) / (power + 1)
            if operator == "average":
                expected /= b - a
        np.testing.assert_allclose(G[:, power], expected, atol=1e-12)


def test_polynomial_tau_and_mixed_basis_units():
    basis = collection(
        fnmap["linear"](tref=ORIGIN, units="hours", tau=2),
        fnmap["quadratic"](tref=ORIGIN, units="days", tau=3),
    )
    G = basis.design_matrix(dates([0]), "integral", dates([1]), "hours")
    np.testing.assert_allclose(G, [[144, 8 / 9]])


def test_harmonic_attenuation_and_fixed_phase():
    basis = collection(fnmap["periodic"](tref=ORIGIN, units="days", period=10))
    a, b = np.array([0.0, 1.0, 2.0]), np.array([10.0, 21.0, 7.0])
    midpoint = (a + b) / 2
    expected = np.column_stack(
        (np.cos(2 * np.pi * midpoint / 10), np.sin(2 * np.pi * midpoint / 10))
    )
    expected *= np.sinc((b - a) / 10)[:, None]
    average = basis.design_matrix(dates(a), "average", dates(b), "days")
    np.testing.assert_allclose(average, expected, atol=1e-15)
    np.testing.assert_allclose(average[:2], 0, atol=1e-15)
    integral = basis.design_matrix(dates(a), "integral", dates(b), "hours")
    np.testing.assert_allclose(integral, average * ((b - a) * 24)[:, None], atol=1e-13)


@pytest.mark.parametrize("name", ["bspline", "ispline"])
@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize("scale", [2.0, -2.0])
@pytest.mark.parametrize("bounded", [False, True])
def test_spline_integrals_against_quadrature(name, order, scale, bounded):
    bounds = {}
    if bounded:
        bounds = {"tmin": dates([-2.5])[0], "tmax": dates([3.5])[0]}
    function = fnmap[name](
        tref=ORIGIN, units="days", scale=scale, order=order, **bounds
    )
    basis = collection(function)
    a = np.array([-12.0, -7.0, -3.0, 0.0, 3.0, 8.0])
    b = np.array([-8.0, -2.0, 2.0, 3.0, 7.0, 12.0])
    G = basis.design_matrix(dates(a), "integral", dates(b), "days")[:, 0]

    # Independently integrate the existing point evaluator, splitting at knots
    # and any imposed active boundaries rather than using its antiderivative.
    knots = (np.arange(order + 2) - (order + 1) / 2) * scale
    breaks = np.concatenate((knots, [-2.5, 3.5]))
    expected = []
    for start, end in zip(a, b):
        points = np.unique(breaks[(breaks > start) & (breaks < end)])
        value, error = quad(
            lambda x: float(function(dates([x]))[0]),
            start,
            end,
            points=points,
            epsabs=1e-7,
        )
        expected.append(value)
    np.testing.assert_allclose(G, expected, atol=2e-7, rtol=1e-7)


def test_active_support_and_full_average_duration():
    basis = collection(fnmap["constant"](tmin=dates([1])[0], tmax=dates([3])[0]))
    a, b = dates([0, 1, 3, 4]), dates([4, 3, 4, 5])
    np.testing.assert_allclose(
        basis.design_matrix(a, "integral", b, "days")[:, 0], [2, 2, 0, 0]
    )
    np.testing.assert_allclose(
        basis.design_matrix(a, "average", b, "days")[:, 0], [0.5, 1, 0, 0]
    )
    np.testing.assert_allclose(
        basis.design_matrix(a, "difference", b)[:, 0], [0, 0, -1, 0]
    )


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("power", {"exp": 0.5}),
        ("power", {"exp": -1}),
        ("arctan", {"period": 2}),
        ("exp", {"tau": 2}),
        ("pbspline", {"period": 10, "offset": 0, "scale": 2}),
    ],
)
def test_unsupported_integrals_are_explicit(name, kwargs):
    basis = collection(fnmap[name](tref=ORIGIN, units="days", **kwargs))
    a, b = dates([1, 2]), dates([2, 3])
    for operator in ("integral", "average"):
        with pytest.raises(NotImplementedError, match="Integration"):
            basis.design_matrix(a, operator, b)
    np.testing.assert_allclose(basis.design_matrix(a), np.reshape(basis(a), (2, 1)))
    np.testing.assert_allclose(
        basis.design_matrix(a, "difference", b)[:, 0], basis(b)[:, 0] - basis(a)[:, 0]
    )
    model = Model(a, basis, observation="difference", t_end=b)
    np.testing.assert_allclose(
        model.predict(np.ones(1))["full"], basis(b)[:, 0] - basis(a)[:, 0]
    )


def test_average_converges_to_sampling_and_short_polynomial_intervals():
    basis = collection(
        fnmap["poly"](tref=ORIGIN, units="days", order=3),
        fnmap["periodic"](tref=ORIGIN, units="days", period=10),
        fnmap["bspline"](tref=ORIGIN, units="days", scale=2),
        fnmap["ispline"](tref=ORIGIN, units="days", scale=2),
    )
    t = dates([0.5, 1.5])
    sample = basis.design_matrix(t)
    errors = []
    for width in [0.1, 0.001, 0.00001]:
        half = dt.timedelta(days=width / 2)
        average = basis.design_matrix(t - half, "average", t + half, "days")
        errors.append(np.max(np.abs(average - sample)))
    assert errors[1] < errors[0] / 100
    assert errors[2] < 1e-8

    # A factored power difference remains accurate far from the reference epoch.
    basis = polynomial(order=2)
    t = dates([10000])
    G = basis.design_matrix(t, "average", t + dt.timedelta(microseconds=1), "days")
    np.testing.assert_allclose(G, [[1, 10000, 1e8]], rtol=1e-12)


@pytest.mark.parametrize(
    "time_input",
    [
        [dt.date(2020, 1, 1), dt.date(2020, 1, 2)],
        np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        np.array(
            ["2020-01-01T00:00:00.123456", "2020-01-02T00:00:00.123456"],
            dtype="datetime64[us]",
        ),
        dates([0, 1]),
    ],
)
def test_time_normalization(time_input):
    model = Model(time_input)
    assert all(isinstance(t, dt.datetime) for t in model.t)
    assert model.G.shape == (2, 2)
    if np.asarray(time_input).dtype == np.dtype("datetime64[us]"):
        assert model.t[0].microsecond == 123456


def test_decimal_year_durations_use_calendar_dates():
    model = Model(
        [2019.0, 2020.0],
        observation="integral",
        t_end=[2020.0, 2021.0],
        time_units="days",
    )
    np.testing.assert_allclose(model.durations, [365, 366])
    np.testing.assert_allclose(model.G[:, 0], [365, 366])
    model = Model(2020.0, observation="integral", t_end=2021.0)
    np.testing.assert_allclose(model.durations, [366 / 365.25])
    assert ureg.Quantity(1, "years").to("seconds").magnitude == 31557600


@pytest.mark.parametrize(
    "t,operator,end,units",
    [
        ([], "sample", None, "days"),
        ([[2020.0]], "sample", None, "days"),
        ([np.nan], "sample", None, "days"),
        ([np.inf], "integral", [2021.0], "days"),
        ([np.datetime64("NaT")], "sample", None, "days"),
        ([ORIGIN.replace(tzinfo=dt.timezone.utc)], "sample", None, "days"),
        (dates([0]), "sample", dates([1]), "days"),
        (dates([0]), "average", None, "days"),
        (dates([0]), "difference", dates([0]), "days"),
        (dates([1]), "integral", dates([0]), "days"),
        (dates([0]), "integral", dates([1, 2]), "days"),
        (dates([0]), "integral", [np.datetime64("NaT")], "days"),
        (dates([0]), "other", None, "days"),
        (dates([0]), "sample", None, "metres"),
        (dates([0]), "integral", dates([1]), "not_a_unit"),
    ],
)
def test_invalid_observations(t, operator, end, units):
    with pytest.raises(ValueError):
        Model(t, observation=operator, t_end=end, time_units=units)


def test_single_row_shapes_and_legacy_collection_call():
    basis = polynomial()
    assert basis(dates([0])).shape == (2,)
    for operator in ("sample", "average", "integral", "difference"):
        end = None if operator == "sample" else dates([1])[0]
        G = basis.design_matrix(ORIGIN, operator, end)
        model = Model(ORIGIN, basis, observation=operator, t_end=end)
        assert G.shape == model.G.shape == (1, 2)
        assert model.predict(np.ones(2))["full"].shape == (1,)


def test_interval_window_contains_entire_pairs():
    model = Model(
        dates([3, 0, 1, 2]),
        observation="average",
        t_end=dates([4, 2, 3, 5]),
        t0="2020-01-02",
        tf=dates([4])[0],
    )
    np.testing.assert_array_equal(model.time_mask, [True, False, True, False])
    assert model.t[0] == dates([3])[0]


def test_builders_use_all_endpoints_and_keep_prediction_basis(tmp_path):
    a, b = dates([4, 0, 2, 0]), dates([8, 1, 5, 1])
    model = build_temporal_model(
        a, periods=[1], bsplines=[4], isplines=[4], observation="average", t_end=b
    )
    assert model.collection[1].tref == min(a)
    assert model.collection[2].tmax == max(b)
    splines = [fn for fn in model.collection if fn.fnname == "BSpline"]
    assert splines[0].tref == min(a)
    assert splines[-1].tref == max(b)
    before = model.metadata()
    m = np.arange(model.npar, dtype=float)
    model.Cm = np.eye(model.npar)
    grid = dates([-1, 1.5, 4, 10])
    prediction = model.predict(m, t=grid, observation="sample")
    G = model.collection.design_matrix(grid)
    np.testing.assert_allclose(prediction["full"], G @ m)
    np.testing.assert_allclose(
        prediction["sigma"], np.sqrt(np.diag(G @ model.Cm @ G.T))
    )
    assert model.metadata() == before
    assert model.design_matrix() is model.G
    np.testing.assert_allclose(model.predict(m)["full"], model.G @ m)
    np.testing.assert_allclose(
        model.design_matrix(observation="integral"), model.G * model.durations[:, None]
    )

    # A userfile asserts its build inputs, and the model must preserve pair rows.
    userfile = tmp_path / "model.py"
    userfile.write_text("""
import numpy as np
from iceutils.tseries.timefn import TimefnCollection, fnmap
calls = 0
def build(dates, marker=3):
    global calls
    calls += 1
    assert calls == 1
    assert marker == 3
    assert [date.day for date in dates] == [1, 2, 3, 5, 6, 9]
    result = TimefnCollection()
    result.append(fnmap['poly'](tref=dates[0], units='days', order=1))
    return result
def computeCm(collection, marker=None):
    assert calls == 1
    return np.eye(len(collection))
""")
    loaded, Cm = build_temporal_model_fromfile(
        a,
        str(userfile),
        cov=True,
        marker=3,
        observation="average",
        t_end=b,
        time_units="days",
    )
    np.testing.assert_allclose(loaded.G[:, 1], [6, 0.5, 3.5, 0.5])
    np.testing.assert_array_equal(Cm, np.eye(2))
    forwarded = build_temporal_model(
        a, userfile=str(userfile), observation="integral", t_end=b, time_units="days"
    )
    np.testing.assert_allclose(forwarded.G, loaded.G * np.array([4, 1, 3, 1])[:, None])


def test_metadata_is_json_snapshot_and_reconstructs_supported_basis():
    model = build_temporal_model(
        dates([0, 1, 2]),
        periods=[1],
        steps=[dates([1])[0]],
        bsplines=[3],
        isplines=[3],
        observation="average",
        t_end=dates([1, 2, 3]),
    )
    metadata = json.loads(json.dumps(model.metadata()))
    rebuilt = TimefnCollection()
    for definition in metadata["basis"]:
        params = definition["parameters"].copy()
        params.pop("norm", None)
        params.pop("radfactor", None)
        params.pop("efactor", None)
        if "exponent" in params:
            exponent = params.pop("exponent")
            if definition["type"] == "Power":
                params["exp"] = exponent
        for name in ("tref", "tmin", "tmax"):
            if name in params:
                params[name] = dt.datetime.fromisoformat(params[name])
        rebuilt.append(fnmap[definition["type"].lower()](**params))
    np.testing.assert_allclose(
        rebuilt.design_matrix(
            model.t, model.observation, model.t_end, model.time_units
        ),
        model.G,
    )
    metadata["basis"][0]["parameters"]["tmin"] = "changed"
    assert model.metadata()["basis"][0]["parameters"]["tmin"] != "changed"


def test_difference_identifiability_and_explicit_constraint():
    a, b = dates([0, 1, 3, 4]), dates([1, 3, 4, 6])
    model = Model(a, polynomial(), observation="difference", t_end=b)
    np.testing.assert_array_equal(model.G[:, 0], 0)
    d = model.G @ np.array([0.0, 2.0])
    with pytest.raises(ValueError, match="min_poly=1"):
        model.invert(select_solver("lsqr", n_min=1), d)
    with pytest.raises(ValueError, match="unconstrained"):
        model.invert(select_solver("ridge", reg_indices=[1], penalty=1, n_min=1), d)
    solver = select_solver("ridge", regMat=np.diag([1.0, 0]), n_min=1)
    np.testing.assert_allclose(model.invert(solver, d), [0, 2])
    reduced = Model(a, polynomial(minorder=1), observation="difference", t_end=b)
    np.testing.assert_allclose(reduced.invert(select_solver("lsqr", n_min=1), d), [2])


def test_nullspace_guard_uses_valid_weighted_rows_and_custom_solver():
    model = Model(
        dates([0, 1, 2]),
        polynomial(),
        observation="average",
        t_end=dates([1, 2, 3]),
        time_units="days",
    )
    d = model.G @ np.array([1.0, 2.0])
    solver = select_solver("lsqr", n_min=1)
    with pytest.raises(ValueError, match="unconstrained"):
        model.invert(solver, d, wgt=np.array([1.0, 0.0, np.nan]))
    with pytest.raises(ValueError, match="nonnegative"):
        model.invert(solver, d, wgt=np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="one value"):
        model.invert(solver, d[:, None])
    with pytest.raises(ValueError, match="one inverse"):
        model.invert(solver, d, wgt=np.ones(2))
    d[2] = np.nan
    np.testing.assert_allclose(model.invert(solver, d), [1.0, 2.0])

    class CustomSolver:
        def invert(self, G, d):
            return ice.SUCCESS, np.zeros(G.shape[1]), np.eye(G.shape[1])

    constant = Model(
        dates([0, 1]), polynomial(), observation="difference", t_end=dates([1, 2])
    )
    np.testing.assert_array_equal(constant.invert(CustomSolver(), np.ones(2)), 0)


@pytest.mark.parametrize(
    "regMat",
    [
        np.ones((3, 3)),
        np.array([[1, 1], [0, 1]]),
        np.diag([1.0, -1.0]),
        np.diag([np.nan, 1]),
    ],
)
def test_invalid_precision_rejected(regMat):
    model = Model(
        dates([0, 1]), polynomial(), observation="average", t_end=dates([1, 2])
    )
    with pytest.raises(ValueError, match="regMat"):
        model.invert(select_solver("ridge", regMat=regMat, n_min=1), np.ones(2))


def synthetic_pairs():
    rng = np.random.default_rng(82)
    start = np.sort(rng.uniform(0, 3 * 365.25, 180))
    start = start[(start < 420) | (start > 520)]
    end = start + rng.uniform(5, 90, len(start))
    basis = collection(
        fnmap["poly"](tref=ORIGIN, order=1, units="years"),
        fnmap["periodic"](tref=ORIGIN, period=1, units="years"),
        fnmap["bspline"](tref=dates([600])[0], scale=0.2, order=3, units="years"),
        fnmap["ispline"](tref=dates([800])[0], scale=0.15, order=3, units="years"),
    )
    return Model(dates(start), basis, observation="average", t_end=dates(end)), rng


def test_regularized_recovery_and_duration_scaled_displacements():
    model, rng = synthetic_pairs()
    truth = np.array([100.0, 4.0, 8.0, -3.0, 12.0, 20.0])
    sigma = rng.uniform(0.1, 0.3, len(model.t))
    d = model.G @ truth + rng.normal(size=len(model.t)) * sigma
    d[::17] = np.nan
    solver = select_solver("ridge", reg_indices=model.reg_indices, penalty=0.01)
    m = model.invert(solver, d, wgt=1 / sigma)
    assert np.all(np.abs(m - truth) < 3 * np.sqrt(np.diag(model.Cm)))
    covariance = model.Cm.copy()
    displacement = Model(
        model.t, model.collection, observation="integral", t_end=model.t_end
    )
    fitted = displacement.invert(
        solver, d * model.durations, wgt=1 / (sigma * model.durations)
    )
    np.testing.assert_allclose(fitted, m, atol=1e-10)
    np.testing.assert_allclose(displacement.Cm, covariance, atol=1e-12)
    grid = dates(np.linspace(0, 1100, 100))
    np.testing.assert_allclose(
        model.predict(m, t=grid, observation="sample")["full"],
        model.design_matrix(grid, observation="sample") @ truth,
        atol=0.3,
    )


def test_correlated_covariance_equivalence_with_external_whitening():
    model, rng = synthetic_pairs()
    n = len(model.t)
    covariance = 0.03 * 0.6 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    covariance += 0.01 * np.eye(n)
    d = model.G @ np.arange(model.npar) + rng.multivariate_normal(
        np.zeros(n), covariance
    )
    G_displacement = model.design_matrix(observation="integral")
    duration = model.durations
    C_displacement = duration[:, None] * covariance * duration[None, :]
    solver = select_solver("ridge", reg_indices=model.reg_indices, penalty=0.02)
    results = []
    for G, data, C in [
        (model.G, d, covariance),
        (G_displacement, d * duration, C_displacement),
    ]:
        L = np.linalg.cholesky(C)
        whitened_G = solve_triangular(L, G, lower=True)
        whitened_d = solve_triangular(L, data, lower=True)
        status, m, Cm = solver.invert(whitened_G, whitened_d)
        assert status == ice.SUCCESS
        results.append((m, Cm))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-10)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-12)


def test_sparse_overlap_allowed_but_unpenalized_null_rejected():
    # Different basis definitions can be duplicate columns over the data span.
    basis = collection(fnmap["constant"](), fnmap["constant"](tmin=dates([-1])[0]))
    model = Model(
        dates([0, 1, 2]), basis, observation="average", t_end=dates([1, 2, 3])
    )
    solver = select_solver(
        "lasso", reg_indices=[0, 1], penalty=0.01, rw_iter=1, n_min=1
    )
    m = model.invert(solver, np.ones(3))
    np.testing.assert_allclose(model.G @ m, 1, atol=0.01)
    solver = select_solver("lasso", reg_indices=[], penalty=0.01, rw_iter=1, n_min=1)
    with pytest.raises(ValueError, match="unconstrained"):
        model.invert(solver, np.ones(3))


def test_stack_workflows_reject_interval_models(tmp_path):
    from iceutils.tseries import inversion, inversion_points

    model = Model(dates([0, 1]), observation="average", t_end=dates([1, 2]))

    class DummyStack:
        tdec = np.array([2020.0, 2021.0])

    with pytest.raises(ValueError, match="Stack inversion does not support interval"):
        inversion(DummyStack(), model, str(tmp_path))
    with pytest.raises(ValueError, match="Stack inversion does not support interval"):
        inversion_points(DummyStack(), model, [0], [0])


@pytest.mark.parametrize("solver_type", ["lsqr", "ridge", "lasso", "omp", "ransac"])
def test_builtin_solvers_accept_interval_matrix(solver_type):
    model = Model(
        dates(np.arange(30)),
        polynomial(),
        observation="average",
        t_end=dates(np.arange(30) + 1),
        time_units="days",
    )
    d = model.G @ np.array([2.0, 3.0])
    solver = select_solver(
        solver_type,
        reg_indices=[1],
        penalty=0.001,
        rw_iter=1,
        n_nonzero_coefs=2,
        n_min=2,
    )
    m = model.invert(solver, d)
    np.testing.assert_allclose(model.predict(m)["full"], d, atol=0.01)


def test_legacy_sample_regression():
    t = np.linspace(2020.0, 2022.0, 50)
    model = build_temporal_model(t, periods=[1], bsplines=[4], isplines=[4])
    np.testing.assert_array_equal(model.G, model.collection(model.t))
    parts = ice.tseries.timefn.getFunctionTypes(model.collection)
    np.testing.assert_array_equal(model.reg_indices, parts["reg"])
    m = np.arange(model.npar, dtype=float)
    model.Cm = np.eye(model.npar)
    predicted = model.predict(m)
    for key in ("secular", "seasonal", "transient", "step"):
        indices = getattr(model, "i" + key)
        np.testing.assert_array_equal(predicted[key], model.G[:, indices] @ m[indices])
    np.testing.assert_allclose(predicted["full"], model.G @ m)
    np.testing.assert_allclose(
        predicted["sigma"], np.sqrt(np.diag(model.G @ model.G.T))
    )
    solver = select_solver("ridge", reg_indices=model.reg_indices, penalty=0.1)
    d = model.G @ m
    status, expected, Cm = solver.invert(model.G, d)
    np.testing.assert_array_equal(model.invert(solver, d), expected)
    np.testing.assert_array_equal(model.Cm, Cm)
