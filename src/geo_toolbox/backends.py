from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .dependencies import require, version
from .errors import BackendExecutionError, DataValidationError, MissingBackendError


def impute_zcompositions(values, limits, status, method, rscript="Rscript", timeout=300):
    if method not in {"multRepl", "lrEM", "lrEMplus"}:
        raise DataValidationError("Supported zCompositions methods: multRepl, lrEM, lrEMplus")
    missing = status == "missing"
    censored = status == "censored"
    if method == "lrEMplus" and not (missing.any() and censored.any()):
        raise DataValidationError("lrEMplus requires both missing and censored observations; explicitly choose lrEM or multRepl for a single type")
    if method != "lrEMplus" and missing.any() and censored.any():
        raise DataValidationError("Mixed missing and censored observations require method='lrEMplus'")
    executable = shutil.which(rscript)
    if executable is None:
        raise MissingBackendError("Rscript not found; install R and zCompositions to use this optional backend")
    if censored.any() and not np.all(np.isfinite(limits[censored]) & (limits[censored] > 0)):
        raise DataValidationError("Every censored observation requires an explicit positive detection limit")
    matrix = values.copy()
    matrix[censored] = 0.0
    matrix[missing] = np.nan
    with tempfile.TemporaryDirectory(prefix="geo_toolbox_") as temporary:
        root = Path(temporary)
        pd.DataFrame(matrix).to_csv(root / "input.csv", index=False, na_rep="NA")
        pd.DataFrame(np.where(censored, limits, 0.0)).to_csv(root / "limits.csv", index=False)
        script = root / "run.R"
        script.write_text(
            'args <- commandArgs(trailingOnly=TRUE)\n'
            'if (!requireNamespace("zCompositions", quietly=TRUE)) stop("Install the zCompositions R package")\n'
            'x <- as.matrix(read.csv(args[1], check.names=FALSE, na.strings="NA"))\n'
            'dl <- as.matrix(read.csv(args[2], check.names=FALSE))\n'
            'fn <- getExportedValue("zCompositions", args[3])\n'
            'options(warn=2)\n'
            'if (args[3] == "lrEMplus") {\n'
            '  y <- fn(x, dl=dl, z.delete=FALSE)\n'
            '} else if (anyNA(x)) {\n'
            '  y <- fn(x, label=NA, imp.missing=TRUE, z.delete=FALSE)\n'
            '} else {\n'
            '  y <- fn(x, label=0, dl=dl, z.delete=FALSE)\n'
            '}\n'
            'write.csv(y, args[4], row.names=FALSE)\n'
            'writeLines(as.character(packageVersion("zCompositions")), args[5])\n',
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [executable, "--vanilla", str(script), str(root / "input.csv"), str(root / "limits.csv"),
                 method, str(root / "output.csv"), str(root / "version.txt")],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as error:
            raise BackendExecutionError(f"R backend could not complete: {error}") from error
        if completed.returncode:
            raise BackendExecutionError(f"zCompositions failed: {completed.stderr.strip() or completed.stdout.strip()}")
        output = pd.read_csv(root / "output.csv").to_numpy(dtype=float)
        backend_version = (root / "version.txt").read_text(encoding="utf-8").strip()
    if output.shape != matrix.shape or not np.all(np.isfinite(output) & (output > 0)):
        raise BackendExecutionError("zCompositions changed shape or returned nonpositive/nonfinite values")
    observed = status == "observed"
    if not np.allclose(output[observed], values[observed], rtol=1e-7, atol=0):
        raise BackendExecutionError("zCompositions changed observed concentrations; result rejected")
    if censored.any() and not np.all(output[censored] < limits[censored]):
        raise BackendExecutionError("Imputed censored concentrations must remain below their detection limits")
    return output, backend_version


def logratio(values, method, basis=None):
    codata = require("pyrolite.comp.codata", "pyrolite")
    if method == "clr":
        return np.asarray(codata.CLR(values)), None
    if method == "ilr":
        if basis is None:
            basis = require("pyrolite.util.math", "pyrolite").helmert_basis(D=values.shape[1])
        basis = np.asarray(basis, dtype=float)
        if basis.shape != (values.shape[1] - 1, values.shape[1]) or not np.isfinite(basis).all() or not np.allclose(basis @ basis.T, np.eye(values.shape[1] - 1)) or not np.allclose(basis.sum(axis=1), 0):
            raise DataValidationError("ILR basis must be an orthonormal, zero-sum (D-1) by D matrix")
        return np.asarray(codata.ILR(values, psi=basis)), basis
    raise DataValidationError("Transform method must be 'clr' or 'ilr'")


def kriging_parameters(parameters, model):
    keys = {"spherical": {"psill", "range", "nugget"},
            "exponential": {"psill", "range", "nugget"},
            "gaussian": {"psill", "range", "nugget"}}
    if model not in keys or not isinstance(parameters, dict) or set(parameters) != keys[model]:
        raise DataValidationError("Provide explicit named psill/range/nugget parameters for spherical/exponential/gaussian")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
           or not np.isfinite(value) for value in parameters.values()):
        raise DataValidationError("Variogram parameters must be finite numbers, not booleans")
    output = {key: float(value) for key, value in parameters.items()}
    if output["range"] <= 0 or min(output["psill"], output["nugget"]) < 0 or output["psill"] + output["nugget"] <= 0:
        raise DataValidationError("Variogram requires positive range, nonnegative partial sill/nugget and positive total sill")
    return output


def ordinary_kriging(x, y, z, pred_x, pred_y, *, variogram, variogram_parameters, nlags, exact_values, pseudo_inv, n_closest_points, chunk_size):
    parameters = kriging_parameters(variogram_parameters, variogram)
    ordinary = require("pykrige.ok", "PyKrige").OrdinaryKriging
    model = ordinary(x, y, z, variogram_model=variogram, variogram_parameters=parameters, nlags=nlags,
                     coordinates_type="euclidean", exact_values=exact_values, pseudo_inv=pseudo_inv,
                     verbose=False, enable_plotting=False)
    predictions, variances = [], []
    for start in range(0, len(pred_x), chunk_size):
        stop = start + chunk_size
        predicted, variance = model.execute("points", pred_x[start:stop], pred_y[start:stop],
                                           backend="loop", n_closest_points=n_closest_points)
        predictions.append(np.ma.filled(predicted, np.nan))
        variances.append(np.ma.filled(variance, np.nan))
    return np.concatenate(predictions), np.concatenate(variances), model.variogram_model_parameters


def variogram_gstools(x, y, z, *, model, estimator, bin_edges, direction):
    gs = require("gstools")
    constructors = {"spherical": gs.Spherical, "exponential": gs.Exponential, "gaussian": gs.Gaussian}
    if model not in constructors or estimator not in {"matheron", "cressie"}:
        raise DataValidationError("Choose spherical/exponential/gaussian and matheron/cressie")
    if direction is not None:
        distances, gamma, counts = gs.vario_estimate((x, y), z, bin_edges=bin_edges, estimator=estimator,
                                                    direction=direction, return_counts=True)
        return {"distance": distances, "semivariance": gamma, "pair_counts": counts,
                "parameters": None, "fit_r2": None, "fit_scope": "directional estimates only"}
    distances, gamma, counts = gs.vario_estimate((x, y), z, bin_edges=bin_edges, estimator=estimator, return_counts=True)
    keep = (counts > 0) & np.isfinite(gamma)
    if keep.sum() < 3:
        raise DataValidationError("At least three populated variogram bins are required for fitting")
    fitted = constructors[model](dim=2)
    parameters, _, r2 = fitted.fit_variogram(distances[keep], gamma[keep], return_r2=True)
    factors = {"spherical": 1.0, "exponential": 3.0, "gaussian": 7.0 / 4.0}
    explicit = kriging_parameters({"psill": fitted.var, "range": fitted.len_rescaled * factors[model],
                                   "nugget": fitted.nugget}, model)
    return {"distance": distances, "semivariance": gamma, "pair_counts": counts,
            "model": model, "parameters": parameters, "pykrige_parameters": explicit,
            "range_conversion": {"gstools_rescale": fitted.rescale, "pykrige_factor": factors[model]},
            "fit_r2": r2, "fit_scope": "isotropic empirical variogram fit, not predictive validation"}
