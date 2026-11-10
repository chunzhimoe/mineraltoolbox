from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS

from .core import GridSpec, record
from .dependencies import require, version
from .errors import BackendExecutionError, DataValidationError
from .magnetic_core import MagneticDataset, MagneticGrid, MagneticQC, read_magnetic


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not isinstance(value, (int, float, np.number)):
        raise DataValidationError(f"{name} must be a finite number")
    if not np.isfinite(value) or (positive and value <= 0):
        raise DataValidationError(f"{name} must be finite" + (" and positive" if positive else ""))
    return float(value)


def _metric(crs):
    if crs is None:
        raise DataValidationError("Confirm a projected CRS before spatial processing")
    parsed = CRS.from_user_input(crs)
    if not parsed.is_projected or len(parsed.axis_info) < 2 or any(
            not np.isclose(axis.unit_conversion_factor, 1.0) for axis in parsed.axis_info[:2]):
        raise DataValidationError("Magnetic spatial processing requires projected metre coordinates")
    return parsed


def _usable(data):
    _metric(data.schema.crs)
    selected = data.active.to_numpy(dtype=bool)
    if selected.sum() < 3:
        raise DataValidationError("At least three active observations are required")
    coordinates = data.coordinates.loc[selected].to_numpy(dtype=float)
    values = data.values.loc[selected].to_numpy(dtype=float)
    if not np.isfinite(coordinates[:, :2]).all() or not np.isfinite(values).all():
        raise DataValidationError("Active observations have invalid coordinates or values; select explicitly")
    if data.status.loc[selected].ne("observed").any():
        raise DataValidationError("Active observations contain unresolved invalid or missing values")
    if pd.DataFrame(coordinates[:, :2]).duplicated().any():
        raise DataValidationError("Duplicate active horizontal coordinates require explicit review")
    if np.linalg.matrix_rank(coordinates[:, :2] - coordinates[:, :2].mean(axis=0)) < 2:
        raise DataValidationError("Active coordinates must span a two-dimensional region")
    if data.stage not in {"anomaly", "rtp"}:
        raise DataValidationError("Grid confirmed magnetic anomalies, not uncorrected total-field observations")
    return coordinates, values


def _fit(data, method, *, depth, damping, block_size):
    coordinates, values = _usable(data)
    if damping is not None:
        damping = _number(damping, "damping", positive=True)
    if method == "equivalent_sources":
        depth = _number(depth, "depth", positive=True)
        if not np.isfinite(coordinates[:, 2]).all():
            raise DataValidationError("Equivalent sources require confirmed upward coordinates in metres")
        if block_size is not None:
            block_size = _number(block_size, "block_size", positive=True)
        hm = require("harmonica")
        estimator = hm.EquivalentSources(depth=depth, damping=damping, block_size=block_size, parallel=False)
        fit_coordinates = tuple(coordinates.T)
    elif method == "verde_spline":
        if depth is not None or block_size is not None:
            raise DataValidationError("depth and block_size are only supported by equivalent_sources")
        if not np.isfinite(coordinates[:, 2]).all() or not np.allclose(
                coordinates[:, 2], coordinates[0, 2], rtol=0, atol=1e-8):
            raise DataValidationError("A 2D spline cannot correct variable or unknown observation height")
        estimator = require("verde").Spline(damping=damping)
        fit_coordinates = tuple(coordinates[:, :2].T)
    else:
        raise DataValidationError("method must be equivalent_sources or verde_spline")
    estimator.fit(fit_coordinates, values)
    return estimator, coordinates


def _history(data, operation, backend, parameters, **details):
    return [*data.history, record(operation, backend, parameters, version=version(backend), **details)]


class MagneticAPI:
    def read(self, source, *, schema, sheet_name=0, header=0, skiprows=None):
        return read_magnetic(source, schema=schema, sheet_name=sheet_name, header=header, skiprows=skiprows)

    def check(self, data):
        xy = data.coordinates[["easting", "northing"]]
        finite_xy = np.isfinite(xy.to_numpy(dtype=float)).all(axis=1)
        duplicate = xy.duplicated(keep=False) & finite_xy
        return MagneticQC(data.copy(), {
            "rows": len(data.values), "active": int(data.active.sum()), "stage": data.stage,
            "status_counts": data.status.value_counts().to_dict(),
            "invalid_coordinate_ids": data.values.index[~finite_xy].tolist(),
            "duplicate_coordinate_ids": data.values.index[duplicate].tolist(),
            "unknown_upward_ids": data.values.index[~np.isfinite(data.coordinates.upward)].tolist(),
            "crs": data.schema.crs, "unit": "nT", "changed_values": False,
        })

    def select(self, data, *, sample_ids):
        ids = pd.Index(sample_ids)
        if ids.empty or not ids.is_unique or len(ids.difference(data.values.index)):
            raise DataValidationError("Select a nonempty, unique list of existing sample IDs")
        active = pd.Series(data.values.index.isin(ids), index=data.values.index)
        return data.copy(active=active, history=_history(data, "select", "pandas", {"sample_ids": ids.tolist()}))

    def flag_spikes(self, data, *, window, threshold_nT):
        if isinstance(window, bool) or not isinstance(window, int) or window < 3 or window % 2 != 1:
            raise DataValidationError("window must be an odd integer at least three")
        threshold = _number(threshold_nT, "threshold_nT", positive=True)
        if data.schema.line is None or data.schema.sequence is None:
            raise DataValidationError("Spike screening requires explicit line and along-line sequence columns")
        lines = data.raw[data.schema.line]
        order = pd.to_numeric(data.raw[data.schema.sequence], errors="coerce")
        if lines.isna().any() or lines.astype(str).str.strip().eq("").any() or not np.isfinite(order).all():
            raise DataValidationError("Every observation needs a line and finite sequence value")
        work = pd.DataFrame({"line": lines, "sequence": order, "value": data.values})
        if work.duplicated(["line", "sequence"]).any():
            raise DataValidationError("Along-line sequence must be unique within each line")
        if not np.isfinite(data.values).all():
            raise DataValidationError("Resolve invalid values explicitly before spike screening")
        flags = pd.Series(False, index=data.values.index)
        residual = pd.Series(np.nan, index=data.values.index)
        for _, group in work.loc[data.active].groupby("line", sort=False):
            group = group.sort_values("sequence")
            median = group.value.rolling(window, center=True, min_periods=window).median()
            residual.loc[group.index] = group.value - median
            flags.loc[group.index] = (group.value - median).abs().gt(threshold)
        return MagneticQC(data.copy(), {
            "method": "along_line_rolling_median_screen", "window": window,
            "threshold_nT": threshold, "flagged_ids": flags.index[flags].tolist(),
            "residual_nT": residual.to_dict(), "changed_values": False,
            "meaning": "Review candidates only, not certified instrument errors; incomplete windows are not screened",
        })

    def correct_diurnal(self, data, *, observation_times, base_times, base_values_nT,
                        reference_nT, max_gap_seconds):
        if data.stage != "raw_total_field":
            raise DataValidationError("Diurnal correction requires raw_total_field, preventing repeat correction")
        reference = _number(reference_nT, "reference_nT")
        max_gap = _number(max_gap_seconds, "max_gap_seconds", positive=True)
        observed_time = pd.DatetimeIndex(pd.to_datetime(observation_times, errors="raise"))
        base_time = pd.DatetimeIndex(pd.to_datetime(base_times, errors="raise"))
        if observed_time.tz is None or base_time.tz is None:
            raise DataValidationError("Observation and base times must have explicit timezones")
        observed_time, base_time = observed_time.tz_convert("UTC"), base_time.tz_convert("UTC")
        base_values = np.asarray(base_values_nT, dtype=float)
        if len(observed_time) != len(data.values) or len(base_time) < 2 or base_values.shape != (len(base_time),):
            raise DataValidationError("Times and base values must be one-dimensional and length-aligned")
        if observed_time.hasnans or base_time.hasnans or not base_time.is_monotonic_increasing or not base_time.is_unique:
            raise DataValidationError("Times must be valid; base times must be strictly increasing")
        if not np.isfinite(base_values).all() or not np.isfinite(data.values).all():
            raise DataValidationError("Correction requires finite magnetic values")
        t = (observed_time - base_time[0]).total_seconds().to_numpy()
        b = (base_time - base_time[0]).total_seconds().to_numpy()
        if (t < b[0]).any() or (t > b[-1]).any():
            raise DataValidationError("Base-station extrapolation is prohibited")
        right = np.searchsorted(b, t, side="left").clip(1, len(b) - 1)
        exact = np.isin(t, b)
        if ((b[right] - b[right - 1] > max_gap) & ~exact).any():
            raise DataValidationError("Base-station interpolation would bridge an excessive time gap")
        correction = np.interp(t, b, base_values) - reference
        values = data.values - correction
        return data.copy(values=values, stage="corrected_total_field", history=_history(
            data, "correct_diurnal", "numpy", {"reference_nT": reference, "max_gap_seconds": max_gap},
            correction_nT=correction.tolist(), time_convention="explicit timezone inputs converted to UTC",
            observation_times=observed_time.astype(str).tolist(), base_times=base_time.astype(str).tolist(),
            base_values_nT=base_values.tolist()))

    def normal_field(self, *, longitude, latitude, height_m, date, model="IGRF14"):
        lon, lat, height = np.broadcast_arrays(np.asarray(longitude, dtype=float),
                                              np.asarray(latitude, dtype=float), np.asarray(height_m, dtype=float))
        if not all(np.isfinite(a).all() for a in (lon, lat, height)) or (np.abs(lon) > 180).any() or (np.abs(lat) >= 90).any():
            raise DataValidationError("Provide finite WGS84 longitude/latitude/ellipsoidal height; poles are unsupported")
        if model not in {"IGRF13", "IGRF14"}:
            raise DataValidationError("Choose the coefficient model explicitly: IGRF13 or IGRF14")
        timestamp = pd.Timestamp(date)
        if pd.isna(timestamp) or timestamp.tz is not None:
            raise DataValidationError("date must be one timezone-naive model epoch")
        module = require("ppigrf")
        coefficient = Path(module.__file__).parent / f"{model}.shc"
        if not coefficient.is_file():
            raise BackendExecutionError(f"Missing installed coefficient file: {coefficient.name}")
        lines = coefficient.read_text(encoding="utf-8").splitlines()
        epoch_lines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        times = np.asarray(epoch_lines[1].split(), dtype=float)
        start = pd.Timestamp(year=int(times.min()), month=1, day=1)
        stop = pd.Timestamp(year=int(times.max()), month=1, day=1)
        if not start <= timestamp <= stop:
            raise DataValidationError(f"date is outside {model} coefficient coverage {start.date()} to {stop.date()}")
        east, north, up = [np.asarray(a, dtype=float).reshape(lon.shape) for a in module.igrf(
            lon, lat, height / 1000.0, timestamp.to_pydatetime(), coeff_fn=str(coefficient))]
        total = np.sqrt(east**2 + north**2 + up**2)
        inclination = np.degrees(np.arctan2(-up, np.hypot(east, north)))
        declination = np.degrees(np.arctan2(east, north))
        if not all(np.isfinite(a).all() for a in (east, north, up, total)):
            raise BackendExecutionError("IGRF returned nonfinite values")
        return {"east_nT": east, "north_nT": north, "up_nT": up, "total_nT": total,
                "inclination_deg": inclination, "declination_deg": declination,
                "metadata": {"backend": "ppigrf", "version": version("ppigrf"), "model": model,
                             "date": timestamp.isoformat(), "height_datum": "WGS84 ellipsoid", "height_unit": "m",
                             "coefficient_sha256": hashlib.sha256(coefficient.read_bytes()).hexdigest()}}

    def subtract_normal_field(self, data, *, longitude, latitude, height_m, date, model="IGRF14"):
        if data.stage != "corrected_total_field":
            raise DataValidationError("Normal-field subtraction requires explicitly corrected_total_field")
        field = self.normal_field(longitude=longitude, latitude=latitude, height_m=height_m, date=date, model=model)
        total = np.asarray(field["total_nT"])
        if total.shape not in {(), (len(data.values),)} or not np.isfinite(data.values).all():
            raise DataValidationError("Normal field must be scalar or aligned with every finite observation")
        return data.copy(values=data.values - total, stage="anomaly", history=_history(
            data, "subtract_normal_field", "ppigrf", field["metadata"], normal_field_nT=total.tolist(),
            longitude=np.asarray(longitude).tolist(), latitude=np.asarray(latitude).tolist(),
            height_m=np.asarray(height_m).tolist()))

    def interpolate(self, data, *, grid: GridSpec, method, upward, max_distance, depth=None,
                    damping=None, block_size=None):
        crs = _metric(data.schema.crs)
        if not crs.equals(_metric(grid.crs), ignore_axis_order=True):
            raise DataValidationError("Grid and sample CRS differ; reproject explicitly")
        target = _number(upward, "upward")
        distance = _number(max_distance, "max_distance", positive=True)
        estimator, coordinates = _fit(data, method, depth=depth, damping=damping, block_size=block_size)
        if method == "equivalent_sources" and target < coordinates[:, 2].max():
            raise DataValidationError("First release predicts only at or above all observation heights")
        if method == "verde_spline" and not np.isclose(target, coordinates[0, 2], rtol=0, atol=1e-8):
            raise DataValidationError("2D spline predictions must stay at the observation height")
        x, y = grid.centers()
        prediction_coordinates = (x, y, np.full(x.shape, target)) if method == "equivalent_sources" else (x, y)
        predicted = np.asarray(estimator.predict(prediction_coordinates), dtype=float)
        vd = require("verde")
        support = vd.distance_mask(tuple(coordinates[:, :2].T), maxdist=distance, coordinates=(x, y))
        support &= vd.convexhull_mask(tuple(coordinates[:, :2].T), coordinates=(x, y))
        if predicted.shape != x.shape or not np.isfinite(predicted).all():
            raise BackendExecutionError("Gridder returned invalid predictions")
        backend = "harmonica" if method == "equivalent_sources" else "verde"
        return MagneticGrid(grid, predicted, support, data.stage, "nT", target, _history(
            data, "interpolate", backend, {"method": method, "upward": target, "max_distance": distance,
                                          "depth": depth, "damping": damping, "block_size": block_size},
            support_meaning="convex hull AND nearest active sample distance; not prediction uncertainty"))

    def validate(self, data, *, train_ids, test_ids, method, buffer_distance,
                 depth=None, damping=None, block_size=None):
        train, test = pd.Index(train_ids), pd.Index(test_ids)
        if train.empty or test.empty or not train.is_unique or not test.is_unique or len(train.intersection(test)):
            raise DataValidationError("Provide nonempty disjoint unique training and held-out sample IDs")
        if len(train.union(test).difference(data.values.index)):
            raise DataValidationError("Validation IDs must exist")
        if not data.active.loc[train.union(test)].all():
            raise DataValidationError("Validation IDs must be active")
        buffer = _number(buffer_distance, "buffer_distance", positive=True)
        selected = self.select(data, sample_ids=train)
        estimator, coordinates = _fit(selected, method, depth=depth, damping=damping, block_size=block_size)
        target = data.coordinates.loc[test].to_numpy(dtype=float)
        actual = data.values.loc[test].to_numpy(dtype=float)
        if not np.isfinite(target).all() or not np.isfinite(actual).all() or data.status.loc[test].ne("observed").any():
            raise DataValidationError("Held-out values and all coordinates must be finite and observed")
        tree = require("scipy.spatial", "scipy").cKDTree(coordinates[:, :2])
        if (tree.query(target[:, :2])[0] < buffer).any():
            raise DataValidationError("Held-out samples violate the requested spatial buffer")
        if method == "verde_spline" and not np.allclose(target[:, 2], coordinates[0, 2], rtol=0, atol=1e-8):
            raise DataValidationError("2D validation requires identical observation heights")
        predicted = np.asarray(estimator.predict(tuple(target.T) if method == "equivalent_sources"
                                                else tuple(target[:, :2].T)), dtype=float)
        if predicted.shape != actual.shape or not np.isfinite(predicted).all():
            raise BackendExecutionError("Gridder returned invalid held-out predictions")
        residual = predicted - actual
        return {"train_ids": train.tolist(), "test_ids": test.tolist(), "method": method,
                "buffer_distance": buffer, "predicted_nT": predicted.tolist(), "observed_nT": actual.tolist(),
                "rmse_nT": float(np.sqrt(np.mean(residual**2))), "mae_nT": float(np.mean(np.abs(residual))),
                "bias_nT": float(np.mean(residual)), "selection": "explicit caller-provided spatial holdout; no tuning",
                "parameters": {"depth": depth, "damping": damping, "block_size": block_size}}

    def crop(self, data: MagneticGrid, *, rows, columns):
        if not isinstance(rows, (tuple, list)) or not isinstance(columns, (tuple, list)) or len(rows) != 2 or len(columns) != 2:
            raise DataValidationError("rows and columns must be explicit (start, stop) pixel ranges")
        r0, r1 = rows
        c0, c1 = columns
        if any(type(v) is not int for v in (r0, r1, c0, c1)) or not (0 <= r0 < r1 <= data.grid.height and 0 <= c0 < c1 <= data.grid.width):
            raise DataValidationError("Crop indices must define a nonempty in-bounds rectangle")
        a, b, c, d, e, f = data.grid.transform
        grid = GridSpec(c1 - c0, r1 - r0, (a, b, c + c0 * a, d, e, f + r0 * e), data.grid.crs)
        return MagneticGrid(grid, data.values[r0:r1, c0:c1], data.support[r0:r1, c0:c1],
                            data.stage, data.unit, data.upward,
                            _history(data, "crop", "numpy", {"rows": list(rows), "columns": list(columns)}))

    def transform(self, data: MagneticGrid, *, method, padding, trim, height_displacement=None,
                  azimuth=None, inclination=None, declination=None, magnetization_inclination=None,
                  magnetization_declination=None, assume_induced=False, max_gain=None):
        _metric(data.grid.crs)
        if data.stage not in {"anomaly", "rtp"} or data.unit != "nT" or data.upward is None:
            raise DataValidationError("Transform a confirmed anomaly field in nT on a known horizontal plane")
        if not np.isfinite(data.values).all() or not data.support.all():
            raise DataValidationError("FFT requires a fully finite supported rectangle; crop explicitly, never fill holes")
        if min(data.values.shape) < 4:
            raise DataValidationError("Transforms require at least four cells on each axis")
        if type(padding) is not int or padding < 0 or type(trim) is not int or trim < 0 or 2 * trim >= min(data.values.shape):
            raise DataValidationError("padding and trim must be nonnegative pixel counts retaining a nonempty grid")
        allowed = {"upward_continuation", "derivative_easting", "derivative_northing", "derivative_upward",
                   "directional_derivative", "total_gradient_amplitude", "tilt_angle", "reduction_to_pole"}
        if method not in allowed:
            raise DataValidationError(f"Unsupported magnetic transform: {method}")
        supplied = {"height_displacement": height_displacement, "azimuth": azimuth, "inclination": inclination,
                    "declination": declination, "magnetization_inclination": magnetization_inclination,
                    "magnetization_declination": magnetization_declination, "max_gain": max_gain}
        accepted = {"upward_continuation": {"height_displacement"}, "directional_derivative": {"azimuth"},
                    "reduction_to_pole": {"inclination", "declination", "magnetization_inclination",
                                          "magnetization_declination", "max_gain"}}.get(method, set())
        if any(value is not None and name not in accepted for name, value in supplied.items()):
            raise DataValidationError("Transform received irrelevant parameters")
        if not isinstance(assume_induced, bool) or (assume_induced and method != "reduction_to_pole"):
            raise DataValidationError("assume_induced applies only to RTP and must be boolean")
        hm, xr = require("harmonica"), require("xarray")
        x, y = data.grid.centers()
        values = xr.DataArray(data.values[::-1].copy(), dims=("northing", "easting"),
                              coords={"northing": y[::-1, 0], "easting": x[0]})
        if padding:
            north = y[::-1, 0][0] + np.arange(-padding, data.grid.height + padding) * abs(data.grid.transform[4])
            east = x[0, 0] + np.arange(-padding, data.grid.width + padding) * data.grid.transform[0]
            padded = xr.DataArray(np.pad(values.values, padding, mode="edge"),
                                  dims=values.dims, coords={"northing": north, "easting": east})
        else:
            padded = values
        params = {"method": method, "padding": padding, "padding_mode": "edge" if padding else "none",
                  "trim": trim, "azimuth_convention": "clockwise from north", "vertical_axis": "upward"}
        stage, unit, upward = data.stage, "nT", data.upward
        if method == "reduction_to_pole":
            if stage == "rtp":
                raise DataValidationError("RTP cannot be applied twice")
            inc = _number(inclination, "inclination")
            dec = _number(declination, "declination")
            if not -90 <= inc <= 90 or not -180 <= dec <= 180:
                raise DataValidationError("RTP inclination/declination angles are outside their degree ranges")
            if assume_induced:
                if magnetization_inclination is not None or magnetization_declination is not None:
                    raise DataValidationError("Choose explicit magnetization OR assume_induced")
                mi, md = inc, dec
            else:
                mi = _number(magnetization_inclination, "magnetization_inclination")
                md = _number(magnetization_declination, "magnetization_declination")
            if not -90 <= mi <= 90 or not -180 <= md <= 180:
                raise DataValidationError("Magnetization angles are outside their degree ranges")
            gain_limit = _number(max_gain, "max_gain", positive=True)
            ke, kn = np.meshgrid(np.fft.fftfreq(padded.shape[1]), np.fft.fftfreq(padded.shape[0]))
            ke = ke / data.grid.transform[0]
            kn = kn / abs(data.grid.transform[4])
            k = np.hypot(ke, kn)
            nonzero = k > 0
            def theta(i, d):
                i, d = np.radians([i, d])
                return np.sin(i) + 1j * np.cos(i) * (np.sin(d) * ke[nonzero] + np.cos(d) * kn[nonzero]) / k[nonzero]
            denominator = np.abs(theta(inc, dec) * theta(mi, md))
            gain = float(np.max(1 / np.maximum(denominator, np.finfo(float).tiny)))
            if gain > gain_limit:
                raise DataValidationError(f"Unstabilized RTP maximum gain {gain:g} exceeds explicit limit {gain_limit:g}")
            transformed = hm.reduction_to_pole(padded, inclination=inc, declination=dec,
                                              magnetization_inclination=mi, magnetization_declination=md)
            params.update(inclination=inc, declination=dec, magnetization_inclination=mi,
                          magnetization_declination=md, assume_induced=assume_induced,
                          maximum_filter_gain=gain, max_gain=gain_limit, dc_policy="upstream zero-frequency removed")
            stage = "rtp"
        elif method == "upward_continuation":
            displacement = _number(height_displacement, "height_displacement", positive=True)
            transformed = hm.upward_continuation(padded, height_displacement=displacement)
            upward += displacement
            params["height_displacement"] = displacement
        elif method == "directional_derivative":
            angle = _number(azimuth, "azimuth")
            east = hm.derivative_easting(padded, method="fft")
            north = hm.derivative_northing(padded, method="fft")
            transformed = np.sin(np.radians(angle)) * east + np.cos(np.radians(angle)) * north
            params["azimuth"] = angle
            unit = "nT/m"
        else:
            if method in {"derivative_easting", "derivative_northing"}:
                transformed = getattr(hm, method)(padded, method="fft")
            else:
                transformed = getattr(hm, method)(padded)
            if method == "tilt_angle":
                transformed = -transformed
                params.update(tilt_vertical_axis="downward",
                              tilt_formula="atan2(-dT/dupward, hypot(dT/deasting, dT/dnorthing))",
                              tilt_reference="Miller and Singh (1994), downward-positive vertical derivative")
            unit = "rad" if method == "tilt_angle" else "nT/m"
        result = transformed.isel(northing=slice(padding, -padding), easting=slice(padding, -padding)) if padding else transformed
        array = np.asarray(result.values, dtype=float)[::-1].copy()
        if array.shape != data.values.shape or not np.isfinite(array).all():
            raise BackendExecutionError("Transform produced nonfinite or shape-changing results")
        support = data.support.copy()
        if trim:
            support[:trim] = support[-trim:] = False
            support[:, :trim] = support[:, -trim:] = False
        return MagneticGrid(data.grid, array, support, stage, unit, upward,
                            _history(data, "transform", "harmonica", params))
