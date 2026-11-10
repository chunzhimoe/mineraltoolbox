import copy

import numpy as np
import pandas as pd
import pytest

from geo_toolbox import GeoToolbox, GridSpec, MagneticGrid, MagneticSchema
from geo_toolbox.errors import DataValidationError, MissingBackendError


def api():
    return GeoToolbox().magnetic


def samples(stage="anomaly"):
    x, y = np.meshgrid(np.arange(5) * 100.0, np.arange(5) * 100.0)
    frame = pd.DataFrame({"id": [f"p{i}" for i in range(x.size)], "x": x.ravel(), "y": y.ravel(),
                          "z": np.zeros(x.size), "v": (2 * x - y).ravel(),
                          "line": np.repeat(np.arange(5), 5), "sequence": np.tile(np.arange(5), 5)})
    schema = MagneticSchema("id", "x", "y", "v", "EPSG:21456", "nT", stage,
                            upward="z", line="line", sequence="sequence")
    return api().read(frame, schema=schema)


def wave_grid():
    spec = GridSpec(64, 48, (10.0, 0.0, 0.0, 0.0, -10.0, 480.0), "EPSG:21456")
    x, y = spec.centers()
    kx, ky = 2 * np.pi / 640 * 2, 2 * np.pi / 480 * 3
    phase = kx * x + ky * y
    grid = MagneticGrid(spec, np.cos(phase), np.ones(phase.shape, bool), "anomaly", "nT", 0.0)
    return grid, phase, kx, ky


@pytest.mark.parametrize("method,axis", [("derivative_easting", "e"), ("derivative_northing", "n"),
                                          ("derivative_upward", "u")])
def test_periodic_wave_derivative_sign_and_north_up_orientation(method, axis):
    grid, phase, kx, ky = wave_grid()
    result = api().transform(grid, method=method, padding=0, trim=0)
    expected = {"e": -kx * np.sin(phase), "n": -ky * np.sin(phase),
                "u": -np.hypot(kx, ky) * np.cos(phase)}[axis]
    np.testing.assert_allclose(result.values, expected, rtol=1e-10, atol=1e-12)
    assert result.unit == "nT/m" and result.grid == grid.grid
    np.testing.assert_array_equal(grid.values, np.cos(phase))


def test_directional_azimuth_clockwise_from_north():
    grid, phase, kx, ky = wave_grid()
    for angle in (0, 45, 90, 135):
        result = api().transform(grid, method="directional_derivative", azimuth=angle, padding=0, trim=0)
        expected = -(kx * np.sin(np.radians(angle)) + ky * np.cos(np.radians(angle))) * np.sin(phase)
        np.testing.assert_allclose(result.values, expected, atol=1e-12)


def test_continuation_exponential_and_metadata():
    grid, phase, kx, ky = wave_grid()
    result = api().transform(grid, method="upward_continuation", height_displacement=30, padding=0, trim=0)
    np.testing.assert_allclose(result.values, np.cos(phase) * np.exp(-np.hypot(kx, ky) * 30), atol=1e-12)
    assert result.upward == 30 and result.unit == "nT"
    assert result.history[-1]["parameters"]["vertical_axis"] == "upward"


def test_total_gradient_and_tilt_match_discrete_horizontal_derivatives():
    grid, phase, kx, ky = wave_grid()
    dx = np.gradient(grid.values, 10, axis=1)
    dn = np.gradient(grid.values, -10, axis=0)
    du = -np.hypot(kx, ky) * np.cos(phase)
    tga = api().transform(grid, method="total_gradient_amplitude", padding=0, trim=0)
    tilt = api().transform(grid, method="tilt_angle", padding=0, trim=0)
    np.testing.assert_allclose(tga.values, np.sqrt(dx**2 + dn**2 + du**2), atol=1e-12)
    np.testing.assert_allclose(tilt.values, np.arctan2(-du, np.hypot(dx, dn)), atol=1e-12)
    assert tilt.unit == "rad"
    parameters = tilt.history[-1]["parameters"]
    assert parameters["tilt_vertical_axis"] == "downward"
    assert parameters["tilt_formula"] == "atan2(-dT/dupward, hypot(dT/deasting, dT/dnorthing))"


def test_tilt_positive_directly_above_vertical_dipole():
    spec = GridSpec.from_bounds((-5025, -5025, 5025, 5025), 50, "EPSG:21456")
    east, north = spec.centers()
    depth = 500.0
    radius_squared = east**2 + north**2
    anomaly = 1e12 * (2 * depth**2 - radius_squared) / (radius_squared + depth**2)**2.5
    grid = MagneticGrid(spec, anomaly, np.ones(anomaly.shape, bool), "rtp", "nT", 0.0)
    tilt = api().transform(grid, method="tilt_angle", padding=30, trim=10)
    upward = api().transform(grid, method="derivative_upward", padding=30, trim=10)
    center = (spec.height // 2, spec.width // 2)
    assert east[center] == 0 and north[center] == 0
    assert upward.values[center] < 0
    assert tilt.support[center] and tilt.values[center] == pytest.approx(np.pi / 2, abs=1e-8)
    np.testing.assert_array_equal(grid.values, anomaly)
    assert tilt.history[-1]["parameters"]["tilt_reference"] == "Miller and Singh (1994), downward-positive vertical derivative"


def test_rtp_analytic_single_mode_and_gain_guard():
    grid, phase, kx, ky = wave_grid()
    inc, dec = np.radians([55.0, 15.0])
    theta = np.sin(inc) + 1j * np.cos(inc) * (np.sin(dec) * kx + np.cos(dec) * ky) / np.hypot(kx, ky)
    observed = grid.copy(values=np.real(theta**2 * np.exp(1j * phase)))
    result = api().transform(observed, method="reduction_to_pole", inclination=55, declination=15,
                             assume_induced=True, max_gain=10, padding=0, trim=0)
    np.testing.assert_allclose(result.values, np.cos(phase), atol=1e-11)
    assert result.stage == "rtp"
    with pytest.raises(DataValidationError):
        api().transform(result, method="reduction_to_pole", inclination=55, declination=15,
                        assume_induced=True, max_gain=10, padding=0, trim=0)
    with pytest.raises(DataValidationError):
        api().transform(grid, method="reduction_to_pole", inclination=0, declination=0,
                        assume_induced=True, max_gain=10, padding=0, trim=0)
    with pytest.raises(DataValidationError):
        api().transform(grid, method="reduction_to_pole", inclination=55, declination=15,
                        max_gain=10, padding=0, trim=0)


def test_transform_rejects_holes_unknown_planes_irrelevant_options_and_negative_height():
    grid, _, _, _ = wave_grid()
    bad_support = grid.support.copy(); bad_support[5, 6] = False
    bad_values = grid.values.copy(); bad_values[5, 6] = np.nan
    for bad in (grid.copy(support=bad_support), grid.copy(values=bad_values), grid.copy(upward=None),
                grid.copy(stage="raw_total_field"), grid.copy(unit="rad")):
        with pytest.raises(DataValidationError):
            api().transform(bad, method="derivative_upward", padding=0, trim=0)
    with pytest.raises(DataValidationError):
        api().transform(grid, method="derivative_easting", inclination=55, padding=0, trim=0)
    with pytest.raises(DataValidationError):
        api().transform(grid, method="upward_continuation", height_displacement=-1, padding=0, trim=0)
    result = api().transform(grid, method="derivative_upward", padding=8, trim=4)
    assert result.values.shape == grid.values.shape and result.support[4:-4, 4:-4].all()
    assert not result.support[:4].any() and not result.support[:, -4:].any()


def test_diurnal_time_interpolation_sign_stage_and_no_extrapolation():
    data = samples("raw_total_field")
    times = pd.date_range("2008-07-15", periods=len(data.values), freq="s", tz="Asia/Shanghai")
    base_times = [times[0], times[-1]]
    corrected = api().correct_diurnal(data, observation_times=times, base_times=base_times,
                                     base_values_nT=[50000, 50024], reference_nT=50000, max_gap_seconds=25)
    np.testing.assert_allclose(corrected.values, data.values - np.arange(25))
    assert corrected.stage == "corrected_total_field" and data.stage == "raw_total_field"
    for change in ({"max_gap_seconds": 10}, {"observation_times": times + pd.Timedelta(seconds=1)},
                   {"observation_times": times.tz_localize(None)}):
        options = dict(observation_times=times, base_times=base_times, base_values_nT=[50000, 50024],
                       reference_nT=50000, max_gap_seconds=25)
        options.update(change)
        with pytest.raises(DataValidationError):
            api().correct_diurnal(data, **options)
    with pytest.raises(DataValidationError):
        api().correct_diurnal(corrected, observation_times=times, base_times=base_times,
                              base_values_nT=[50000, 50024], reference_nT=50000, max_gap_seconds=25)


def test_igrf_independent_published_grid_reference_and_units():
    result = api().normal_field(longitude=-179, latitude=80, height_m=5000, date="2010-01-01", model="IGRF14")
    np.testing.assert_allclose([result["east_nT"], result["north_nT"], result["up_nT"]],
                               [594.3, 4095.0, -57440.4], rtol=0, atol=1.0)
    assert float(result["inclination_deg"]) > 0
    assert float(result["declination_deg"]) > 0
    assert len(result["metadata"]["coefficient_sha256"]) == 64
    with pytest.raises(DataValidationError):
        api().normal_field(longitude=93, latitude=36, height_m=0, date="2035-01-01")
    with pytest.raises(DataValidationError):
        api().normal_field(longitude=93, latitude=90, height_m=0, date="2008-01-01")


def test_normal_subtraction_explicit_stage_no_double_subtraction():
    data = samples("corrected_total_field")
    field = api().normal_field(longitude=93, latitude=36, height_m=3800, date="2008-07-15")
    result = api().subtract_normal_field(data, longitude=93, latitude=36, height_m=3800, date="2008-07-15")
    np.testing.assert_allclose(result.values, data.values - field["total_nT"])
    assert result.stage == "anomaly"
    with pytest.raises(DataValidationError):
        api().subtract_normal_field(result, longitude=93, latitude=36, height_m=3800, date="2008-07-15")


def test_spike_screen_is_within_lines_and_never_mutates():
    data = samples()
    values = data.values.copy(); values.loc["p2"] = 100000
    data = data.copy(values=values)
    before = copy.deepcopy(data)
    result = api().flag_spikes(data, window=3, threshold_nT=1000)
    assert result.diagnostics["flagged_ids"] == ["p2"]
    pd.testing.assert_series_equal(result.data.values, before.values)
    pd.testing.assert_series_equal(data.active, before.active)
    chosen = api().select(data, sample_ids=["p0", "p1", "p2"])
    assert len(chosen.values) == 25 and chosen.active.sum() == 3


@pytest.mark.parametrize("method", ["equivalent_sources", "verde_spline"])
def test_gridder_shape_support_and_raw_protection(method):
    data = samples()
    grid = GridSpec.from_bounds((0, 0, 400, 400), 50, "EPSG:21456")
    before = data.values.copy()
    options = {"depth": 100, "damping": 1e-9} if method == "equivalent_sources" else {"damping": 1e-9}
    result = api().interpolate(data, grid=grid, method=method, upward=0, max_distance=100, **options)
    assert result.values.shape == (8, 8) and result.support.all() and result.unit == "nT"
    pd.testing.assert_series_equal(data.values, before)
    assert "not prediction uncertainty" in result.history[-1]["support_meaning"]


def test_spline_height_gate_and_unconfirmed_coordinates():
    data = samples()
    grid = GridSpec.from_bounds((0, 0, 400, 400), 50, "EPSG:21456")
    uneven = data.coordinates.copy(); uneven.loc["p0", "upward"] = 10
    with pytest.raises(DataValidationError):
        api().interpolate(data.copy(coordinates=uneven), grid=grid, method="verde_spline", upward=0, max_distance=100)
    unknown = MagneticSchema("id", "x", "y", "v", None, "nT", "anomaly", upward="z")
    with pytest.raises(DataValidationError):
        api().interpolate(data.copy(schema=unknown), grid=grid, method="equivalent_sources", upward=0,
                          max_distance=100, depth=100)


def test_spatial_holdout_uses_only_train_and_enforces_buffer(monkeypatch):
    from geo_toolbox import magnetic
    data = samples()
    train = ["p0", "p4", "p20", "p24"]
    test = ["p12"]
    captured = []
    original = magnetic._fit
    def observed_fit(selected, method, **kwargs):
        captured.append(selected.values.index[selected.active].tolist())
        return original(selected, method, **kwargs)
    monkeypatch.setattr(magnetic, "_fit", observed_fit)
    result = api().validate(data, train_ids=train, test_ids=test, method="verde_spline", buffer_distance=100)
    assert captured == [train]
    residual = np.asarray(result["predicted_nT"]) - np.asarray(result["observed_nT"])
    assert result["rmse_nT"] == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert result["bias_nT"] == pytest.approx(np.mean(residual))
    with pytest.raises(DataValidationError):
        api().validate(data, train_ids=train, test_ids=test, method="verde_spline", buffer_distance=300)


def test_explicit_crop_updates_affine_without_filling_holes():
    grid, _, _, _ = wave_grid()
    cropped = api().crop(grid, rows=(4, 44), columns=(3, 60))
    assert cropped.grid.transform == (10.0, 0.0, 30.0, 0.0, -10.0, 440.0)
    np.testing.assert_array_equal(cropped.values, grid.values[4:44, 3:60])
    assert cropped.values.shape == (40, 57)
    with pytest.raises(DataValidationError):
        api().crop(grid, rows=(-1, 4), columns=(0, 3))


def test_optional_backend_missing_is_explicit_and_check_is_independent(monkeypatch):
    from geo_toolbox import magnetic
    data = samples()
    def missing(*args, **kwargs):
        raise MissingBackendError("missing")
    monkeypatch.setattr(magnetic, "require", missing)
    assert api().check(data).diagnostics["rows"] == 25
    with pytest.raises(MissingBackendError):
        api().normal_field(longitude=93, latitude=36, height_m=0, date="2008-01-01")
