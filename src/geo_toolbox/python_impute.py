"""Modified Python adaptation of zCompositions, 2026-11-10.

Upstream authors: Javier Palarea-Albaladejo and Josep Antoni Martin-Fernandez.
SPDX-License-Identifier: GPL-2.0-or-later
Distributed without warranty; full terms and provenance are in LICENSES/.
Only nonclosed concentrations and nonrobust estimates are supported.
Numerical equivalence with the upstream R package has not been verified.
"""

from __future__ import annotations

import numpy as np

from .dependencies import require
from .errors import BackendExecutionError, DataValidationError


SOURCE = {"package": "zCompositions", "commit": "a44822d6", "license": "GPL-2.0-or-later",
          "authors": ["Javier Palarea-Albaladejo", "Josep Antoni Martin-Fernandez"],
          "scope": "Nonclosed concentrations; nonrobust estimates only; no automatic sample deletion or fallback"}
EM_REFERENCES = ["Monahan (2006), doi:10.18637/jss.v016.i08",
                 "Martin-Fernandez et al. (2011), Boletin Geologico y Minero 122(4):459-468, equations 2-3",
                 "Greene, Econometric Analysis, Chapter 19, Theorem 19.2",
                 "Dempster, Laird and Rubin (1977), doi:10.1111/j.2517-6161.1977.tb01600.x"]


def _validate(values, limits, status, method, frac, tolerance, max_iter, ini_cov):
    x = np.array(values, dtype=float, copy=True)
    states = np.asarray(status)
    dl = np.asarray(limits, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2 or states.shape != x.shape or dl.shape != x.shape:
        raise DataValidationError("Values, detection limits and status must have the same N by D shape, D >= 2")
    if method not in {"multRepl", "lrEM", "lrEMplus"} or ini_cov not in {"complete.obs", "multRepl"}:
        raise DataValidationError("Choose multRepl/lrEM/lrEMplus and complete.obs/multRepl initialization")
    if not np.isfinite(frac) or not 0 < frac < 1:
        raise DataValidationError("frac must be strictly between zero and one")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise DataValidationError("tolerance must be finite and positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise DataValidationError("max_iter must be a positive integer")
    if not np.isin(states, ["observed", "missing", "censored"]).all():
        raise DataValidationError("Only observed, missing and censored states can be imputed")
    observed, missing, censored = states == "observed", states == "missing", states == "censored"
    if not np.all(np.isfinite(x[observed]) & (x[observed] > 0)):
        raise DataValidationError("Observed concentrations must be positive and finite")
    if not np.all(np.isfinite(dl[censored]) & (dl[censored] > 0)):
        raise DataValidationError("Every censored cell requires a positive explicit detection limit")
    if method == "lrEMplus" and not (missing.any() and censored.any()):
        raise DataValidationError("lrEMplus requires both missing and censored observations")
    if method != "lrEMplus" and missing.any() and censored.any():
        raise DataValidationError("Mixed missing and censored observations require method='lrEMplus'")
    if method != "multRepl" and (censored.sum(axis=1) > 1).any():
        raise DataValidationError("Literature-based EM supports at most one censored component per sample; joint multiple censoring is unsupported")
    x[~observed] = np.nan
    if not observed.any(axis=1).all() or not observed.any(axis=0).all():
        raise DataValidationError("Completely unobserved rows or columns cannot be inferred; review them explicitly")
    return x, dl, observed, missing, censored


def _open_concentrations(x):
    totals = np.nansum(x, axis=1)
    if len(x) > 1 and np.all(np.abs(totals - totals.mean()) < np.finfo(float).eps ** 0.3):
        raise DataValidationError("Constant observed row totals trigger upstream closure rescaling; closed compositions are not supported by this concentration backend")


def _multiplicative(x, dl, *, missing, frac):
    unresolved = ~np.isfinite(x)
    if not unresolved.any():
        return x.copy(), []
    totals = np.nansum(x, axis=1)
    if missing:
        logs = np.where(np.isfinite(x), np.log(np.where(np.isfinite(x), x, 1)), 0)
        count = np.isfinite(x).sum(axis=0)
        if (count == 0).any():
            raise DataValidationError("Cannot estimate a geometric mean without observations in each column")
        estimates = np.broadcast_to(np.exp(logs.sum(axis=0) / count), x.shape)
    else:
        estimates = frac * dl
    replacement_totals = np.where(unresolved, estimates, 0).sum(axis=1)
    excessive = replacement_totals >= totals
    if excessive.any() and not missing:
        raise DataValidationError("Replacement totals exceed observed row totals; multiplicative replacement is undefined")
    adjustment = 1 - replacement_totals / totals
    if missing:
        adjustment[excessive] = 1
    if not np.all(np.isfinite(adjustment) & (adjustment > 0)):
        raise BackendExecutionError("Nonpositive multiplicative adjustment")
    output = x.copy()
    output[unresolved] = (estimates / adjustment[:, None])[unresolved]
    return output, np.flatnonzero(excessive).tolist() if missing else []


def _covariance(x):
    covariance = np.atleast_2d(np.cov(x, rowvar=False, ddof=0))
    if not np.isfinite(covariance).all():
        raise DataValidationError("Initial log-ratio covariance is not finite")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise DataValidationError("Log-ratio covariance is singular; no hidden ridge regularization is applied") from error
    return covariance


def _conditional_moments(mean, covariance, observed_parts, observed_values, censored_part=None, upper=None):
    missing_parts = np.setdiff1d(np.arange(len(mean)), observed_parts)
    coo = covariance[np.ix_(observed_parts, observed_parts)]
    com = covariance[np.ix_(observed_parts, missing_parts)]
    try:
        coefficients = np.linalg.solve(coo, com)
    except np.linalg.LinAlgError as error:
        raise BackendExecutionError("Singular conditional covariance in lrEM") from error
    residual = covariance[np.ix_(missing_parts, missing_parts)] - com.T @ coefficients
    conditional = mean[missing_parts] + (observed_values - mean[observed_parts]) @ coefficients
    conditional_covariance = np.broadcast_to(residual, (len(observed_values), *residual.shape)).copy()
    log_probability = np.zeros(len(observed_values))
    if censored_part is not None:
        stats = require("scipy.stats", "scipy")
        position = int(np.flatnonzero(missing_parts == censored_part)[0])
        variance = residual[position, position]
        if not np.isfinite(variance) or variance <= 0:
            raise BackendExecutionError("Nonpositive conditional variance in censored lrEM")
        sigma = np.sqrt(variance)
        alpha = (upper - conditional[:, position]) / sigma
        truncated_mean, truncated_variance = stats.truncnorm.stats(-np.inf, alpha, moments="mv")
        if not (np.isfinite(truncated_mean).all() and np.isfinite(truncated_variance).all()
                and (truncated_variance > 0).all() and (truncated_variance <= 1).all()):
            raise BackendExecutionError("Invalid truncated conditional moments; no clipping was applied")
        direction = residual[:, position] / sigma
        conditional += truncated_mean[:, None] * direction
        conditional_covariance += (truncated_variance - 1)[:, None, None] * np.outer(direction, direction)
        log_probability = stats.norm.logcdf(alpha)
    return conditional, conditional_covariance, log_probability


def _expectation(alr, censored, thresholds, mean, covariance):
    stats = require("scipy.stats", "scipy")
    filled = alr.copy()
    correction = np.zeros_like(covariance)
    log_likelihood = 0.
    labels = np.where(censored, 2, np.where(np.isfinite(alr), 0, 1))
    for pattern in np.unique(labels, axis=0):
        rows = np.flatnonzero((labels == pattern).all(axis=1))
        observed_parts, missing_parts = np.flatnonzero(pattern == 0), np.flatnonzero(pattern != 0)
        observed_values = alr[np.ix_(rows, observed_parts)]
        log_likelihood += float(np.sum(stats.multivariate_normal.logpdf(
            observed_values, mean=mean[observed_parts], cov=covariance[np.ix_(observed_parts, observed_parts)])))
        if not len(missing_parts):
            continue
        censor_columns = np.flatnonzero(pattern == 2)
        censor = int(censor_columns[0]) if len(censor_columns) else None
        conditional, conditional_covariance, log_probability = _conditional_moments(
            mean, covariance, observed_parts, observed_values, censor,
            thresholds[rows, censor] if censor is not None else None)
        filled[np.ix_(rows, missing_parts)] = conditional
        correction[np.ix_(missing_parts, missing_parts)] += conditional_covariance.sum(axis=0)
        log_likelihood += float(log_probability.sum())
    return filled, correction, log_likelihood


def _em(x, dl, *, censored, frac, tolerance, max_iter, ini_cov):
    if len(x) <= x.shape[1]:
        raise DataValidationError("lrEM requires more rows than elemental parts")
    complete_columns = np.isfinite(x).all(axis=0)
    if not complete_columns.any():
        raise DataValidationError("lrEM requires at least one fully observed denominator column")
    denominator = int(np.flatnonzero(complete_columns)[0])
    parts = np.delete(np.arange(x.shape[1]), denominator)
    alr = np.log(x[:, parts]) - np.log(x[:, denominator, None])
    unobserved = ~np.isfinite(alr)
    if unobserved.all(axis=1).any():
        raise DataValidationError("lrEM needs at least two observed elemental parts per sample; no automatic multiplicative fallback")
    if ini_cov == "complete.obs":
        complete = alr[np.isfinite(alr).all(axis=1)]
        if len(complete) <= len(parts):
            raise DataValidationError("Too few complete rows to initialize log-ratio covariance; explicitly choose ini_cov='multRepl'")
        mean = complete.mean(axis=0)
        covariance = _covariance(complete)
    else:
        initial = x.copy()
        initial[censored] = frac * dl[censored]
        unresolved = ~np.isfinite(initial)
        means = np.exp(np.nanmean(np.log(x), axis=0))
        initial[unresolved] = np.broadcast_to(means, x.shape)[unresolved]
        initial_alr = np.log(initial[:, parts]) - np.log(initial[:, denominator, None])
        mean = initial_alr.mean(axis=0)
        covariance = _covariance(initial_alr)
    thresholds = np.full_like(alr, np.inf)
    censor_alr = censored[:, parts]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_limits = np.log(dl[:, parts]) - np.log(x[:, denominator, None])
    thresholds[censor_alr] = log_limits[censor_alr]
    log_likelihood_history = []
    final_delta = float("inf")
    for iteration in range(1, max_iter + 1):
        filled, correction, likelihood = _expectation(alr, censor_alr, thresholds, mean, covariance)
        log_likelihood_history.append(likelihood)
        new_mean = filled.mean(axis=0)
        deviations = filled - new_mean
        new_covariance = (deviations.T @ deviations + correction) / len(x)
        final_delta = max(float(np.max(np.abs(new_mean - mean))), float(np.max(np.abs(new_covariance - covariance))))
        if not np.isfinite(new_covariance).all() or not np.isfinite(new_mean).all():
            raise BackendExecutionError("Nonfinite EM state")
        try:
            np.linalg.cholesky(new_covariance)
        except np.linalg.LinAlgError as error:
            raise BackendExecutionError("Nonpositive-definite EM covariance; no hidden regularization") from error
        mean, covariance = new_mean, new_covariance
        if final_delta < tolerance:
            filled, _, likelihood = _expectation(alr, censor_alr, thresholds, mean, covariance)
            log_likelihood_history.append(likelihood)
            output = x.copy()
            with np.errstate(over="raise", invalid="raise"):
                try:
                    recovered = x[:, denominator, None] * np.exp(filled)
                except FloatingPointError as error:
                    raise BackendExecutionError("Overflow restoring concentration units") from error
            for position, part in enumerate(parts):
                rows = unobserved[:, position]
                output[rows, part] = recovered[rows, position]
            return output, {"iterations": iteration, "converged": True, "final_delta": final_delta,
                            "denominator_column": denominator, "alr_columns": parts.tolist(),
                            "alr_mean": mean.tolist(), "alr_covariance": covariance.tolist(),
                            "log_likelihood_history": log_likelihood_history}
    raise BackendExecutionError(f"lrEM did not converge after {max_iter} iterations; final delta={final_delta:.8g}")


def impute_python(values, limits, status, method, *, frac=0.65, tolerance=0.0001, max_iter=50, ini_cov="complete.obs"):
    x, dl, observed, missing, censored = _validate(values, limits, status, method, frac, tolerance, max_iter, ini_cov)
    diagnostics = {"source": dict(SOURCE), "method": method, "frac": frac, "tolerance": tolerance,
                   "max_iter": max_iter, "ini_cov": ini_cov, "fit_scope": "entire supplied batch; not inductive",
                   "converged": True, "iterations": 0, "r_equivalence": "not_verified"}
    if method != "multRepl":
        diagnostics.update(algorithm="literature_alr_em_v1", references=list(EM_REFERENCES),
                           r_equivalence="intentionally_not_equivalent", covariance_estimator="maximum_likelihood_N",
                           censoring_scope="at most one censored component per sample; ordinary missing values may coexist",
                           point_estimate="inverse ALR of conditional log-ratio mean; conditional geometric concentration, not arithmetic mean",
                           initialization="complete ALR observations" if ini_cov == "complete.obs" else "frac times limits for censoring; observed column geometric means for missing")
    if observed.all():
        return x, diagnostics
    _open_concentrations(x)
    if method == "multRepl":
        output, exceptional_rows = _multiplicative(x, dl, missing=missing.any(), frac=frac)
        diagnostics["geometric_mean_unadjusted_rows"] = exceptional_rows
    else:
        output, details = _em(x, dl, censored=censored, frac=frac, tolerance=tolerance, max_iter=max_iter, ini_cov=ini_cov)
        diagnostics.update(details)
    if not np.all(np.isfinite(output) & (output > 0)):
        raise BackendExecutionError("Imputation produced nonpositive or nonfinite concentrations")
    if not np.array_equal(output[observed], x[observed]):
        raise BackendExecutionError("Observed concentrations changed")
    if censored.any() and not np.all(output[censored] < dl[censored]):
        raise BackendExecutionError("Imputation exceeded a detection limit; no silent clipping was applied")
    return output, diagnostics
