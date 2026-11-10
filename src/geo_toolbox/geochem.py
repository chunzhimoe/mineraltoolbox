from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS

from . import backends
from .core import (UNIT_TO_PPM, GeochemDataset, GeochemJob, GeochemSchema, GridSpec,
                   JobResult, MapResult, StepResult, record)
from .dependencies import require, version
from .errors import BackendExecutionError, DataValidationError


CENSORED_TOKEN = re.compile(r"^<\s*([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")


def _require_concentrations(data):
    if data.domain != "concentration":
        raise DataValidationError("This operation requires concentration data, not log-ratio features")


def _usable(data, element):
    if not np.isfinite(data.coordinates.to_numpy(dtype=float)).all():
        raise DataValidationError("All sample coordinates must be finite; correct them before spatial analysis")
    values = data.values[element].to_numpy(dtype=float)
    status = data.status[element].to_numpy()
    replaced = data.imputed[element].to_numpy(dtype=bool)
    if ((status != "observed") & ~replaced).any() or not np.all(np.isfinite(values) & (values > 0)):
        raise DataValidationError(f"{element} contains unresolved censored, missing or invalid observations")
    coordinates = data.coordinates.to_numpy(dtype=float)
    if pd.DataFrame(coordinates).duplicated().any():
        raise DataValidationError("Duplicate coordinates require explicit upstream review; no automatic averaging")
    if len(values) < 3:
        raise DataValidationError("Spatial analysis requires at least three distinct samples")
    return coordinates[:, 0], coordinates[:, 1], values


def _metric_crs(data):
    if data.crs is None:
        raise DataValidationError("CRS is unknown; confirm the source CRS before spatial analysis")
    crs = CRS.from_user_input(data.crs)
    if not crs.is_projected or not crs.axis_info or any(not np.isclose(axis.unit_conversion_factor, 1.0) for axis in crs.axis_info[:2]):
        raise DataValidationError("Spatial analysis requires a projected CRS with metre units; reproject explicitly")


def _limits(frame, schema):
    limits = pd.DataFrame(np.nan, index=frame.index, columns=schema.elements)
    for element, value in schema.detection_limits.items():
        limits[element] = float(value)
    for element, column in schema.detection_limit_columns.items():
        if column not in frame:
            raise DataValidationError(f"Detection-limit column not found: {column}")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values) & (values > 0)):
            raise DataValidationError(f"Detection-limit column {column} must be positive and finite")
        limits[element] = values
    return limits


class GeochemistryAPI:
    def read(self, source, *, schema: GeochemSchema, sheet_name=0, header=0, skiprows=None):
        if isinstance(source, pd.DataFrame):
            frame = source.copy(deep=True)
            source_name = "dataframe"
        else:
            path = Path(source)
            source_name = str(path)
            if path.suffix.lower() == ".csv":
                if isinstance(header, (list, tuple)):
                    raise DataValidationError("Multirow headers are currently supported for Excel inputs only")
                frame = pd.read_csv(path, dtype={schema.sample_id: "string"}, keep_default_na=False,
                                    header=header, skiprows=skiprows)
            elif path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
                engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
                require(engine)
                frame = pd.read_excel(path, dtype=str, keep_default_na=False, sheet_name=sheet_name,
                                      header=header, skiprows=skiprows, engine=engine)
                if isinstance(frame, dict):
                    raise DataValidationError("Read one worksheet at a time; merging quality-control samples must be explicit")
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = [next((str(part).strip() for part in reversed(parts)
                                           if not str(part).startswith("Unnamed:") and str(part).strip()), "")
                                     for parts in frame.columns]
            elif path.suffix.lower() == ".parquet":
                require("pyarrow")
                frame = pd.read_parquet(path)
            else:
                raise DataValidationError("Supported inputs: DataFrame, CSV, XLS/XLSX/XLSM, Parquet")
        required = [schema.sample_id, schema.x, schema.y, *schema.elements]
        if frame.empty or not frame.columns.is_unique or set(required) - set(frame.columns):
            raise DataValidationError("Input is empty, has duplicate columns or lacks required schema columns")
        identifiers = frame[schema.sample_id]
        if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any() or identifiers.duplicated().any():
            raise DataValidationError("Sample IDs must be present and unique")
        frame = frame.set_index(schema.sample_id, drop=False)
        limits = _limits(frame, schema)
        values = pd.DataFrame(np.nan, index=frame.index, columns=schema.elements)
        status = pd.DataFrame("missing", index=frame.index, columns=schema.elements)
        for element in schema.elements:
            factor = UNIT_TO_PPM[schema.units[element]]
            for row, raw_value in frame[element].items():
                if pd.isna(raw_value) or (isinstance(raw_value, str) and raw_value.strip() == ""):
                    continue
                token = CENSORED_TOKEN.fullmatch(str(raw_value).strip())
                if token is not None:
                    limit = float(token.group(1))
                    configured = limits.at[row, element]
                    if not np.isfinite(limit) or limit <= 0 or (np.isfinite(configured) and not np.isclose(limit, configured, rtol=1e-9, atol=0)):
                        raise DataValidationError(f"Invalid or contradictory detection limit at {row}/{element}")
                    limits.at[row, element] = limit
                    status.at[row, element] = "censored"
                    continue
                try:
                    numeric = float(raw_value)
                except (TypeError, ValueError):
                    status.at[row, element] = "invalid"
                    continue
                if not np.isfinite(numeric) or numeric < 0:
                    status.at[row, element] = "invalid"
                elif numeric == 0:
                    if schema.zero_policy == "censored" and np.isfinite(limits.at[row, element]):
                        status.at[row, element] = "censored"
                    else:
                        status.at[row, element] = "invalid"
                else:
                    values.at[row, element] = numeric * factor
                    status.at[row, element] = "observed"
            limits[element] *= factor
        coordinates = frame[[schema.x, schema.y]].apply(pd.to_numeric, errors="coerce")
        coordinates.columns = ["x", "y"]
        data = GeochemDataset(frame.copy(deep=True), values, coordinates, status, limits, schema,
                              None if schema.crs is None else CRS.from_user_input(schema.crs).to_string(), units={e: "ppm" for e in schema.elements})
        data.history.append(record("read", "geo_toolbox", {"source": source_name, "schema": vars(schema),
                                   "sheet_name": sheet_name, "header": header, "skiprows": skiprows},
                                   value_domain="concentration", normalized_unit="ppm"))
        return data

    def check(self, data: GeochemDataset):
        _require_concentrations(data)
        coordinates = data.coordinates.to_numpy(dtype=float)
        coordinate_valid = np.isfinite(coordinates).all(axis=1)
        duplicate = data.coordinates.duplicated(keep=False).to_numpy() & coordinate_valid
        counts = {element: data.status[element].value_counts().to_dict() for element in data.schema.elements}
        diagnostics = {
            "status_counts": counts,
            "invalid_coordinate_ids": data.values.index[~coordinate_valid].tolist(),
            "duplicate_coordinate_ids": data.values.index[duplicate].tolist(),
            "outlier_action": "none; mineralization-related high values are preserved",
            "unresolved_cells": int(((data.status != "observed") & ~data.imputed).to_numpy().sum()),
        }
        output = data.copy()
        output.history.append(record("quality_check", "geo_toolbox", {}, diagnostics=diagnostics))
        return StepResult(output, diagnostics)

    def impute(self, data, *, backend="python", method="lrEM", rscript="Rscript", timeout=300,
               frac=0.65, tolerance=0.0001, max_iter=50, ini_cov="complete.obs"):
        _require_concentrations(data)
        if backend not in {"zcompositions", "python"} or method not in {"multRepl", "lrEM", "lrEMplus"}:
            raise DataValidationError("Choose backend='zcompositions' or 'python' and multRepl/lrEM/lrEMplus")
        if backend == "zcompositions" and (frac != 0.65 or tolerance != 0.0001 or max_iter != 50
                                           or ini_cov != "complete.obs"):
            raise DataValidationError("Python-only parameters require backend=python")
        if (data.status == "invalid").to_numpy().any():
            raise DataValidationError("Invalid values require explicit correction; they are not ordinary missing data")
        if data.imputed.to_numpy().any():
            raise DataValidationError("Imputation has already been applied; restart from the unmodified input")
        needs_imputation = (data.status != "observed").to_numpy()
        output = data.copy()
        if not needs_imputation.any():
            output.history.append(record("impute", backend, {"method": method}, skipped="no unresolved cells"))
            return StepResult(output, {"imputed_cells": 0})
        if backend == "python":
            from .python_impute import impute_python
            matrix, algorithm_diagnostics = impute_python(
                data.values.to_numpy(dtype=float), data.detection_limits.to_numpy(dtype=float),
                data.status.to_numpy(), method,
                frac=frac, tolerance=tolerance, max_iter=max_iter, ini_cov=ini_cov,
            )
            backend_version = version("geo-toolbox")
            parameters = {"method": method, "frac": frac, "tolerance": tolerance,
                          "max_iter": max_iter, "ini_cov": ini_cov}
        else:
            matrix, backend_version = backends.impute_zcompositions(
                data.values.to_numpy(dtype=float), data.detection_limits.to_numpy(dtype=float),
                data.status.to_numpy(), method, rscript=rscript, timeout=timeout,
            )
            algorithm_diagnostics = {}
            parameters = {"method": method, "z.delete": False, "timeout": timeout}
        output.values = pd.DataFrame(matrix, index=data.values.index, columns=data.values.columns)
        output.imputed = pd.DataFrame(needs_imputation, index=data.values.index, columns=data.values.columns)
        diagnostics = {"imputed_cells": int(needs_imputation.sum()),
                       "fit_scope": "entire supplied batch; not a reusable inductive fit",
                       "algorithm": algorithm_diagnostics}
        output.history.append(record("impute", backend, parameters,
                                     backend_version=backend_version, diagnostics=diagnostics))
        return StepResult(output, diagnostics)

    def transform(self, data, *, backend="pyrolite", method="clr", basis=None):
        _require_concentrations(data)
        if backend != "pyrolite" or method not in {"clr", "ilr"}:
            raise DataValidationError("Choose backend='pyrolite' with method='clr' or 'ilr'")
        if basis is not None and method != "ilr":
            raise DataValidationError("An explicit basis is only meaningful for ILR")
        if len(data.schema.elements) < 2:
            raise DataValidationError("Log-ratio transforms require at least two elemental parts")
        if (((data.status != "observed") & ~data.imputed).to_numpy().any()
                or not np.all(np.isfinite(data.values) & (data.values > 0))):
            raise DataValidationError("Resolve all missing/censored/invalid observations before log-ratio transformation")
        transformed, fitted_basis = backends.logratio(data.values.to_numpy(dtype=float), method, basis)
        if not np.isfinite(transformed).all():
            raise BackendExecutionError("pyrolite returned nonfinite log-ratio coordinates")
        columns = [f"clr_{e}" for e in data.schema.elements] if method == "clr" else [f"ilr_{i + 1}" for i in range(transformed.shape[1])]
        output = data.copy(values=pd.DataFrame(transformed, index=data.values.index, columns=columns),
                           domain=method, units={column: "dimensionless" for column in columns})
        output.transform_state = {"method": method, "elements": list(data.schema.elements),
                                  "basis": None if fitted_basis is None else fitted_basis.tolist(),
                                  "input_unit": "ppm", "log_base": "e"}
        output.history.append(record("transform", backend, output.transform_state, backend_version=version("pyrolite"),
                                     note="Common multiplicative enrichment is removed; original concentrations remain a separate branch"))
        return StepResult(output, {"dimension": transformed.shape[1], "independent_dimensions": len(data.schema.elements) - 1})

    def variogram(self, data, *, element, backend="gstools", model="spherical", estimator="matheron", bin_edges=None, direction=None, nlags=None):
        _require_concentrations(data)
        _metric_crs(data)
        if backend != "gstools" or element not in data.schema.elements:
            raise DataValidationError("Choose backend='gstools' and a schema element")
        x, y, z = _usable(data, element)
        if nlags is not None and (isinstance(nlags, bool) or not isinstance(nlags, int) or nlags < 3):
            raise DataValidationError("nlags must be None or an explicit integer of at least three")
        if bin_edges is not None and nlags is not None:
            raise DataValidationError("Choose explicit bin_edges OR nlags, not both")
        edges = None
        if nlags is not None:
            edges = require("gstools").standard_bins((x, y), bin_no=nlags)
        elif bin_edges is not None:
            edges = np.asarray(bin_edges, dtype=float)
        if edges is not None and (edges.ndim != 1 or len(edges) < 4 or not np.isfinite(edges).all() or edges[0] < 0 or not (np.diff(edges) > 0).all()):
            raise DataValidationError("Provide at least four finite, nonnegative, strictly increasing variogram bin edges")
        result = backends.variogram_gstools(x, y, z, model=model, estimator=estimator, bin_edges=edges, direction=direction)
        result["history"] = data.history + [record("variogram", backend, {"element": element, "model": model,
            "estimator": estimator, "bin_edges": edges, "direction": direction, "nlags": nlags}, backend_version=version("gstools"))]
        return result

    def _kriging_variogram(self, data, element, model, nlags, fitted, estimator, bin_edges):
        if fitted is None:
            report = self.variogram(data, element=element, model=model, nlags=nlags,
                                    estimator=estimator, bin_edges=bin_edges)
            parameters = report["pykrige_parameters"]
            return parameters, report
        if not isinstance(fitted, dict) or element not in fitted:
            raise DataValidationError("Explicit variogram parameters must contain every schema element")
        parameters = backends.kriging_parameters(fitted[element], model)
        return parameters, {"model": model, "pykrige_parameters": parameters,
                            "fit_scope": "explicit caller-supplied parameters; fitting provenance must be verified by the caller"}

    def _support(self, grid, support_mask):
        if isinstance(support_mask, (str, Path)):
            rasterio = require("rasterio")
            with rasterio.open(support_mask) as source:
                if source.width != grid.width or source.height != grid.height or source.crs is None or CRS.from_user_input(source.crs) != CRS.from_user_input(grid.crs) or not np.allclose(tuple(source.transform)[:6], grid.transform, rtol=0, atol=1e-9):
                    raise DataValidationError("Support raster must match grid CRS, transform and dimensions exactly")
                array = source.read(1, masked=True)
                support = (~np.ma.getmaskarray(array)) & (array.filled(0) > 0)
        else:
            support = np.asarray(support_mask)
            if support.dtype != bool:
                raise DataValidationError("Array support_mask must be boolean, not implicitly cast from numeric values")
        if support.shape != (grid.height, grid.width) or not support.any():
            raise DataValidationError("Support mask must match the grid and contain at least one supported cell")
        return support.copy()

    def interpolate(self, data, *, grid: GridSpec, support_mask, backend="pykrige", method="ordinary",
                    variogram="spherical", nlags=None, exact_values=True, pseudo_inv=True, n_closest_points=None, chunk_size=10000,
                    variogram_parameters=None, estimator="matheron", bin_edges=None):
        _require_concentrations(data)
        _metric_crs(data)
        if CRS.from_user_input(data.crs) != CRS.from_user_input(grid.crs):
            raise DataValidationError("Sample and grid CRS differ; reproject explicitly before interpolation")
        if backend != "pykrige" or method != "ordinary":
            raise DataValidationError("First release supports backend='pykrige', method='ordinary'")
        if variogram not in {"spherical", "exponential", "gaussian"}:
            raise DataValidationError("Unsupported variogram model; choose spherical/exponential/gaussian")
        if nlags is not None and (isinstance(nlags, bool) or not isinstance(nlags, int) or nlags < 3):
            raise DataValidationError("nlags must be None or an explicit integer of at least three")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise DataValidationError("chunk_size must be a positive integer")
        if variogram_parameters is not None and (not isinstance(variogram_parameters, dict) or set(variogram_parameters) != set(data.schema.elements)):
            raise DataValidationError("Explicit variogram parameters must contain exactly the schema elements")
        if variogram_parameters is not None and (nlags is not None or bin_edges is not None or estimator != "matheron"):
            raise DataValidationError("Explicit variogram parameters cannot be combined with fitting options")
        if not isinstance(exact_values, bool) or not isinstance(pseudo_inv, bool):
            raise DataValidationError("exact_values and pseudo_inv must be boolean")
        if n_closest_points is not None and (isinstance(n_closest_points, bool) or not isinstance(n_closest_points, int) or n_closest_points < 3 or n_closest_points > len(data.values)):
            raise DataValidationError("n_closest_points must be an integer between 3 and the sample count")
        support = self._support(grid, support_mask)
        cx, cy = grid.centers()
        predictions, variances, fitted, variograms = {}, {}, {}, {}
        negative_counts = {}
        for element in data.schema.elements:
            x, y, z = _usable(data, element)
            explicit, report = self._kriging_variogram(data, element, variogram, nlags,
                                                       variogram_parameters, estimator, bin_edges)
            variograms[element] = report
            predicted, variance, parameters = backends.ordinary_kriging(x, y, z, cx[support], cy[support],
                variogram=variogram, variogram_parameters=explicit, nlags=nlags or 6, exact_values=exact_values, pseudo_inv=pseudo_inv,
                n_closest_points=n_closest_points, chunk_size=chunk_size)
            if not np.isfinite(predicted).all() or not np.isfinite(variance).all():
                raise BackendExecutionError(f"Nonfinite predictions or variances for {element}")
            tolerance = 1e-10 * max(float(np.var(z)), 1.0)
            if (variance < -tolerance).any():
                raise BackendExecutionError(f"Materially negative kriging variance for {element}")
            variance = np.maximum(variance, 0.0)
            predictions[element] = np.full(support.shape, np.nan)
            variances[element] = np.full(support.shape, np.nan)
            predictions[element][support] = predicted
            variances[element][support] = variance
            fitted[element] = np.asarray(parameters).tolist()
            negative_counts[element] = int((predicted < 0).sum())
            if negative_counts[element]:
                warnings.warn(f"{element}: {negative_counts[element]} negative concentration predictions retained, not censored or clipped", UserWarning)
        parameters = {"method": method, "variogram": variogram, "nlags": nlags, "exact_values": exact_values,
                      "pseudo_inv": pseudo_inv, "n_closest_points": n_closest_points, "chunk_size": chunk_size,
                      "coordinates_type": "euclidean", "grid": vars(grid), "estimator": estimator,
                      "bin_edges": bin_edges, "variogram_parameters": variogram_parameters}
        history = data.history + [record("interpolate", backend, parameters, backend_version=version("PyKrige"),
                    fitted_parameters=fitted, variograms=variograms, negative_prediction_counts=negative_counts,
                    support="explicit caller-supplied sampling support; no implicit extrapolation")]
        return MapResult(grid, predictions, variances, support, data.domain, dict(data.units), history)

    def validate(self, data, *, train_ids, test_ids, variogram="spherical", nlags=None, estimator="matheron", bin_edges=None):
        _require_concentrations(data)
        _metric_crs(data)
        if variogram not in {"spherical", "exponential", "gaussian"}:
            raise DataValidationError("Unsupported variogram model; choose spherical/exponential/gaussian")
        if nlags is not None and (isinstance(nlags, bool) or not isinstance(nlags, int) or nlags < 3):
            raise DataValidationError("nlags must be None or an explicit integer of at least three")
        if data.imputed.to_numpy().any():
            raise DataValidationError("Batch-imputed data cannot be evaluated as an independent holdout; validate from unfilled observations")
        train_ids, test_ids = list(train_ids), list(test_ids)
        if len(train_ids) < 3 or not test_ids or len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids) or set(train_ids) & set(test_ids):
            raise DataValidationError("Provide disjoint, unique train/test sample IDs and at least three training samples")
        if (set(train_ids) | set(test_ids)) - set(data.values.index):
            raise DataValidationError("Unknown train/test sample IDs")
        if data.coordinates.loc[train_ids + test_ids].duplicated().any():
            raise DataValidationError("Repeated coordinates across or within splits are not independent")
        training_data = data.copy(**{name: getattr(data, name).loc[train_ids].copy(deep=True)
                                    for name in ("raw", "values", "coordinates", "status", "detection_limits", "imputed")})
        metrics, predictions, variograms = {}, {}, {}
        for element in data.schema.elements:
            x, y, z = _usable(data, element)
            training = data.values.index.get_indexer(train_ids)
            testing = data.values.index.get_indexer(test_ids)
            explicit, report = self._kriging_variogram(training_data, element, variogram, nlags,
                                                       None, estimator, bin_edges)
            variograms[element] = report
            predicted, _, _ = backends.ordinary_kriging(x[training], y[training], z[training], x[testing], y[testing],
                variogram=variogram, variogram_parameters=explicit, nlags=nlags or 6, exact_values=True, pseudo_inv=True,
                n_closest_points=None, chunk_size=10000)
            residual = predicted - z[testing]
            if not np.isfinite(residual).all():
                raise BackendExecutionError("Holdout prediction returned nonfinite residuals")
            denominator = np.square(z[testing] - z[testing].mean()).sum()
            metrics[element] = {"n_test": len(test_ids), "mae": float(np.abs(residual).mean()),
                "rmse": float(np.sqrt(np.square(residual).mean())), "mean_error_prediction_minus_observed": float(residual.mean()),
                "r2": None if len(test_ids) < 2 or denominator == 0 else float(1 - np.square(residual).sum() / denominator), "unit": "ppm"}
            predictions[element] = predicted.tolist()
        return {"protocol": "caller-specified holdout; geographic separation must be assessed by the caller",
                "train_ids": train_ids, "test_ids": test_ids, "metrics": metrics, "predictions": predictions,
                "variograms": variograms,
                "history": data.history + [record("validate", "pykrige", {"variogram": variogram, "nlags": nlags,
                                                     "estimator": estimator, "bin_edges": bin_edges},
                                                     backend_version=version("PyKrige"), fit_scope="training IDs only")]}

    def run(self, job: GeochemJob):
        checked = self.check(self.read(job.source, schema=job.schema, **job.read_options))
        data = checked.data
        if job.imputation is not None:
            data = self.impute(data, **job.imputation).data
        features = self.transform(data, **job.transformation).data if job.transformation is not None else None
        maps = self.interpolate(data, **job.interpolation) if job.interpolation is not None else None
        history = data.history.copy()
        if features is not None:
            history.extend(features.history[len(data.history):])
        if maps is not None:
            history.extend(maps.history[len(data.history):])
        return JobResult(data, features, maps, {"status": "not_run", "reason": "Use geochem.validate with explicit held-out sample IDs"}, history)
