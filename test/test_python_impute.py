import numpy as np
import pandas as pd
import pytest

from geo_toolbox import GeoToolbox, GeochemSchema
from geo_toolbox.errors import BackendExecutionError, DataValidationError
from geo_toolbox.python_impute import impute_python


def fixture(mixed=False, missing=False):
    rng = np.random.default_rng(427)
    x = np.exp(rng.normal(size=(80, 4)) * 0.35 + [1., 2., 3., 4.])
    states = np.full(x.shape, "observed", dtype="U8")
    dl = np.zeros_like(x)
    for row, col in [(2, 1), (6, 2), (12, 3), (19, 2)]:
        states[row, col] = "missing" if missing else "censored"
        dl[row, col] = x[row, col] * 1.5
        x[row, col] = np.nan
    if mixed:
        states[35, 2] = "missing"
        x[35, 2] = np.nan
    return x, dl, states


def test_multiplicative_known_open_concentration_result():
    values = np.array([[10., np.nan, 30.], [20., 40., 50.]])
    states = np.array([["observed", "censored", "observed"], ["observed", "observed", "observed"]])
    dl = np.array([[0., 2., 0.], [0., 0., 0.]])
    output, diagnostics = impute_python(values, dl, states, "multRepl")
    assert output[0, 1] == pytest.approx(1.3 / (1 - 1.3 / 40))
    np.testing.assert_array_equal(output[states == "observed"], values[states == "observed"])
    assert diagnostics["converged"] is True
    assert np.isnan(values[0, 1])


@pytest.mark.parametrize("method,missing,mixed", [("multRepl", True, False), ("lrEM", True, False),
                                                ("lrEM", False, False), ("lrEMplus", False, True)])
def test_concentration_invariants(method, missing, mixed):
    values, dl, states = fixture(mixed=mixed, missing=missing)
    output, diagnostics = impute_python(values, dl, states, method, max_iter=500)
    assert np.isfinite(output).all() and (output > 0).all()
    observed = states == "observed"
    np.testing.assert_array_equal(output[observed], values[observed])
    censored = states == "censored"
    assert (output[censored] < dl[censored]).all()
    assert diagnostics["converged"]


@pytest.mark.parametrize("method", ["lrEM", "lrEMplus"])
def test_explicit_multiplicative_initialization(method):
    values, dl, states = fixture(mixed=method == "lrEMplus")
    dl[states == "censored"] = 0.1
    output, diagnostics = impute_python(values, dl, states, method, ini_cov="multRepl", max_iter=500)
    assert np.isfinite(output).all()
    assert diagnostics["ini_cov"] == "multRepl"


def test_mixed_missing_majority_branch():
    values, dl, states = fixture(mixed=True)
    states[36:41, 2] = "missing"
    values[36:41, 2] = np.nan
    output, diagnostics = impute_python(values, dl, states, "lrEMplus", max_iter=500)
    assert diagnostics["converged"]
    assert (output[states == "censored"] < dl[states == "censored"]).all()


def test_missing_and_censoring_are_not_interchangeable():
    values, dl, states = fixture(mixed=True)
    with pytest.raises(DataValidationError, match="Mixed"):
        impute_python(values, dl, states, "lrEM")
    values, dl, states = fixture()
    with pytest.raises(DataValidationError, match="requires both"):
        impute_python(values, dl, states, "lrEMplus")


def test_no_hidden_fallback_or_false_convergence():
    values, dl, states = fixture()
    with pytest.raises(BackendExecutionError, match="did not converge"):
        impute_python(values, dl, states, "lrEM", max_iter=1, tolerance=1e-16)
    states[0, 0] = "missing"
    values[0, 0] = np.nan
    with pytest.raises(DataValidationError, match="fully observed denominator"):
        all_states = states.copy()
        all_values = values.copy()
        for col in range(4):
            all_states[col + 40, col] = "missing"
            all_values[col + 40, col] = np.nan
        impute_python(all_values, dl, all_states, "lrEMplus")


@pytest.mark.parametrize("method,missing,mixed", [("multRepl", False, False), ("lrEM", True, False),
                                                ("lrEM", False, False), ("lrEMplus", False, True)])
def test_unit_scale_equivariance(method, missing, mixed):
    values, dl, states = fixture(mixed=mixed, missing=missing)
    if method == "multRepl":
        dl[states == "censored"] = 0.1
    expected, _ = impute_python(values, dl, states, method, max_iter=500)
    actual, _ = impute_python(values * 1000, dl * 1000, states, method, max_iter=500)
    np.testing.assert_allclose(actual, expected * 1000, rtol=1e-10, atol=1e-9)


def test_reject_unobservable_invalid_and_missing_detection_limit():
    values, dl, states = fixture()
    no_limits = dl.copy()
    no_limits[2, 1] = np.nan
    with pytest.raises(DataValidationError, match="positive explicit detection"):
        impute_python(values, no_limits, states, "lrEM")
    states[0, :] = "missing"
    with pytest.raises(DataValidationError, match="unobserved rows"):
        impute_python(values, dl, states, "lrEMplus")
    states[0, 0] = "invalid"
    with pytest.raises(DataValidationError, match="Only observed"):
        impute_python(values, dl, states, "lrEMplus")


def test_multiplicative_rejects_replacements_larger_than_row_total():
    values = np.array([[1., np.nan, 2.], [20., 20., 30.]])
    dl = np.array([[0., 20., 0.], [0., 0., 0.]])
    states = np.where(np.isfinite(values), "observed", "censored")
    with pytest.raises(DataValidationError, match="exceed observed row totals"):
        impute_python(values, dl, states, "multRepl")


def test_closed_compositions_explicitly_unsupported():
    values = np.array([[10., np.nan, 30.], [20., 20., np.nan]])
    dl = np.ones_like(values)
    states = np.where(np.isfinite(values), "observed", "censored")
    with pytest.raises(DataValidationError, match="closed compositions"):
        impute_python(values, dl, states, "multRepl")


def test_python_api_provenance_and_original_status():
    values, dl, states = fixture()
    frame = pd.DataFrame(values, columns=["A", "B", "C", "D"])
    frame["id"] = [str(i) for i in range(len(frame))]
    frame["x"], frame["y"] = np.arange(len(frame)), 0
    for row, col in np.argwhere(states == "censored"):
        name = ["A", "B", "C", "D"][col]
        frame[name] = frame[name].astype(object)
        frame.loc[row, name] = f"<{dl[row, col]}"
    schema = GeochemSchema("id", "x", "y", None, ["A", "B", "C", "D"], {e: "ppm" for e in ["A", "B", "C", "D"]})
    api = GeoToolbox().geochem
    data = api.read(frame, schema=schema)
    result = api.impute(data, backend="python", method="lrEM", max_iter=500)
    assert result.diagnostics["algorithm"]["converged"]
    assert result.data.history[-1]["backend"] == "python"
    assert result.data.status.equals(data.status)
    assert result.data.imputed.to_numpy().sum() == 4
    assert not data.imputed.to_numpy().any()


def test_default_backend_is_literature_python():
    values, _, _ = fixture(missing=True)
    frame = pd.DataFrame(values, columns=["A", "B", "C", "D"])
    frame["id"] = [str(i) for i in range(len(frame))]
    frame["x"], frame["y"] = np.arange(len(frame)), 0
    schema = GeochemSchema("id", "x", "y", None, ["A", "B", "C", "D"], {e: "ppm" for e in ["A", "B", "C", "D"]})
    api = GeoToolbox().geochem
    result = api.impute(api.read(frame, schema=schema), max_iter=500)
    assert result.data.history[-1]["backend"] == "python"
    assert result.diagnostics["algorithm"]["algorithm"] == "literature_alr_em_v1"
    assert result.diagnostics["algorithm"]["r_equivalence"] == "intentionally_not_equivalent"


def test_joint_multiple_censoring_rejected_but_multrepl_remains_explicit():
    values, dl, states = fixture()
    values[2, 2], dl[2, 2], states[2, 2] = np.nan, .1, "censored"
    with pytest.raises(DataValidationError, match="at most one censored"):
        impute_python(values, dl, states, "lrEM")
    dl[states == "censored"] = .1
    output, _ = impute_python(values, dl, states, "multRepl")
    assert np.isfinite(output).all()


def test_conditional_joint_moments_match_independent_quadrature():
    from scipy.integrate import quad
    from scipy.stats import norm
    from geo_toolbox.python_impute import _conditional_moments

    mean = np.array([.2, -.4, .7])
    covariance = np.array([[1.5, .3, -.2], [.3, 1.2, .5], [-.2, .5, 1.4]])
    value, limit = .6, -.5
    conditional_mean = mean[1:] + covariance[1:, 0] / covariance[0, 0] * (value - mean[0])
    conditional_cov = covariance[1:, 1:] - np.outer(covariance[1:, 0], covariance[0, 1:]) / covariance[0, 0]
    sigma = np.sqrt(conditional_cov[0, 0])
    probability = norm.cdf((limit - conditional_mean[0]) / sigma)
    beta = conditional_cov[1, 0] / conditional_cov[0, 0]
    remaining_variance = conditional_cov[1, 1] - conditional_cov[1, 0] ** 2 / conditional_cov[0, 0]

    def integrate(function):
        return quad(lambda z: function(z) * norm.pdf(z, conditional_mean[0], sigma) / probability,
                    -np.inf, limit, epsabs=1e-11)[0]

    def other_mean(z):
        return conditional_mean[1] + beta * (z - conditional_mean[0])

    expected_mean = np.array([integrate(lambda z: z), integrate(other_mean)])
    second = np.array([[integrate(lambda z: z * z), integrate(lambda z: z * other_mean(z))],
                       [integrate(lambda z: z * other_mean(z)), integrate(lambda z: other_mean(z) ** 2 + remaining_variance)]])
    actual_mean, actual_cov, _ = _conditional_moments(mean, covariance, np.array([0]), np.array([[value]]), 1, np.array([limit]))
    np.testing.assert_allclose(actual_mean[0], expected_mean, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(actual_cov[0], second - np.outer(expected_mean, expected_mean), rtol=1e-10, atol=1e-11)
    assert actual_cov[0, 0, 0] < conditional_cov[0, 0]


def test_extreme_censoring_moments_do_not_use_clamped_cdf():
    from geo_toolbox.python_impute import _conditional_moments

    mean, covariance, _ = _conditional_moments(np.zeros(2), np.eye(2), np.array([0]), np.zeros((1, 1)), 1, np.array([-40.]))
    assert -40.1 < mean[0, 0] < -40
    assert 0 < covariance[0, 0, 0] < .001


def test_em_likelihood_monotonic_and_mixed_row_supported():
    values, dl, states = fixture(mixed=True)
    values[2, 2], states[2, 2] = np.nan, "missing"
    output, diagnostic = impute_python(values, dl, states, "lrEMplus", tolerance=1e-9, max_iter=500)
    assert np.all(np.diff(diagnostic["log_likelihood_history"]) >= -1e-9)
    assert diagnostic["covariance_estimator"] == "maximum_likelihood_N"
    assert np.isfinite(output[2, 1:3]).all()
    assert output[2, 1] < dl[2, 1]


def test_em_parameters_match_independent_observed_likelihood_optimizer():
    from scipy.optimize import minimize
    from scipy.stats import norm, multivariate_normal

    rng = np.random.default_rng(228)
    ratios = rng.multivariate_normal([.5, -.2], [[.5, .18], [.18, .7]], size=120)
    denominator = np.exp(rng.normal(size=120))
    values = np.column_stack([denominator, denominator[:, None] * np.exp(ratios)])
    limits = np.zeros_like(values)
    states = np.full(values.shape, "observed", dtype="U8")
    censored_rows = ratios[:, 0] < -.1
    states[censored_rows, 1] = "censored"
    limits[censored_rows, 1] = denominator[censored_rows] * np.exp(-.1)
    values[censored_rows, 1] = np.nan
    missing_rows = np.flatnonzero(~censored_rows)[::11]
    states[missing_rows, 2], values[missing_rows, 2] = "missing", np.nan
    _, diagnostic = impute_python(values, limits, states, "lrEMplus", max_iter=500, tolerance=1e-10)

    def unpack(theta):
        factor = np.array([[np.exp(theta[2]), 0], [theta[3], np.exp(theta[4])]])
        return theta[:2], factor @ factor.T

    def objective(theta):
        mean, covariance = unpack(theta)
        total = 0.
        for i, row in enumerate(ratios):
            known = np.flatnonzero(states[i, 1:] == "observed")
            if len(known):
                total += multivariate_normal.logpdf(row[known], mean=mean[known], cov=covariance[np.ix_(known, known)])
            if censored_rows[i]:
                if len(known):
                    center = mean[0] + covariance[0, 1] / covariance[1, 1] * (row[1] - mean[1])
                    variance = covariance[0, 0] - covariance[0, 1] ** 2 / covariance[1, 1]
                else:
                    center, variance = mean[0], covariance[0, 0]
                total += norm.logcdf((-.1 - center) / np.sqrt(variance))
        return -float(total)

    optimum = minimize(objective, [.4, -.1, -.3, .2, -.2], method="BFGS", options={"gtol": 1e-6})
    assert optimum.success or np.linalg.norm(optimum.jac) < 1e-4
    expected_mean, expected_covariance = unpack(optimum.x)
    np.testing.assert_allclose(diagnostic["alr_mean"], expected_mean, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(diagnostic["alr_covariance"], expected_covariance, atol=2e-5, rtol=2e-5)
    assert abs(diagnostic["log_likelihood_history"][-1] + optimum.fun) < 1e-7
