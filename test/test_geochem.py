import json

import numpy as np
import pandas as pd
import pytest

from geo_toolbox import GeoToolbox, GeochemJob, GeochemSchema, GridSpec
from geo_toolbox.errors import DataValidationError, MissingBackendError


def schema(**kwargs):
    options = dict(sample_id="id", x="east", y="north", crs="EPSG:2380",
                   elements=["Au", "As"], units={"Au": "ppb", "As": "ppm"})
    options.update(kwargs)
    return GeochemSchema(**options)


def samples():
    return pd.DataFrame({"id": ["01", "02", "03", "04", "05", "06"],
                         "east": [0., 100., 0., 100., 25., 75.],
                         "north": [0., 0., 100., 100., 25., 75.],
                         "Au": [100., 200., 300., 400., 180., 320.],
                         "As": [10., 20., 30., 40., 18., 32.]})


def test_import_and_backend_discovery():
    tool = GeoToolbox()
    assert tool.geochem is not None
    report = tool.backends()
    assert "pyrolite" in report and "pykrige" in report


def test_read_normalizes_units_preserves_raw_and_ids(tmp_path):
    original = samples()
    path = tmp_path / "samples.csv"
    original.to_csv(path, index=False)
    result = GeoToolbox().geochem.read(path, schema=schema())
    assert result.values.index.tolist() == ["01", "02", "03", "04", "05", "06"]
    np.testing.assert_allclose(result.values.Au, original.Au * 0.001)
    np.testing.assert_allclose(result.raw.Au, original.Au)
    pd.testing.assert_frame_equal(original, samples())
    assert result.units == {"Au": "ppm", "As": "ppm"}
    assert (result.status == "observed").to_numpy().all()


def test_status_and_detection_limits_are_not_conflated():
    frame = samples()
    frame["Au"] = ["<0.5", "", "-3", "0", "bad", "2"]
    config = schema(detection_limits={"Au": 0.5}, zero_policy="censored")
    data = GeoToolbox().geochem.read(frame, schema=config)
    assert data.status.Au.tolist() == ["censored", "missing", "invalid", "censored", "invalid", "observed"]
    np.testing.assert_allclose(data.detection_limits.Au, 0.0005)
    assert data.values.loc["06", "Au"] == 0.002
    with pytest.raises(DataValidationError, match="Invalid values"):
        GeoToolbox().geochem.impute(data)


def test_variable_limits_and_contradictory_tokens():
    frame = samples()
    frame["dl"] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    frame["Au"] = frame.Au.astype(object)
    frame.loc[0, "Au"] = "<0.5"
    data = GeoToolbox().geochem.read(frame, schema=schema(detection_limit_columns={"Au": "dl"}))
    np.testing.assert_allclose(data.detection_limits.Au, frame.dl * 0.001)
    frame.loc[0, "Au"] = "<0.2"
    with pytest.raises(DataValidationError, match="contradictory"):
        GeoToolbox().geochem.read(frame, schema=schema(detection_limit_columns={"Au": "dl"}))


def test_zero_is_invalid_without_explicit_policy():
    frame = samples()
    frame.loc[0, "Au"] = 0
    data = GeoToolbox().geochem.read(frame, schema=schema(detection_limits={"Au": 0.5}))
    assert data.status.loc["01", "Au"] == "invalid"


def test_duplicate_ids_rejected_and_coordinates_reported():
    api = GeoToolbox().geochem
    frame = samples()
    frame.loc[1, "id"] = "01"
    with pytest.raises(DataValidationError, match="unique"):
        api.read(frame, schema=schema())
    frame = samples()
    frame.loc[1, ["east", "north"]] = [0., 0.]
    frame.loc[2, "east"] = np.nan
    data = api.read(frame, schema=schema())
    result = api.check(data)
    assert result.diagnostics["duplicate_coordinate_ids"] == ["01", "02"]
    assert result.diagnostics["invalid_coordinate_ids"] == ["03"]
    pd.testing.assert_frame_equal(result.data.values, data.values)
    assert result.diagnostics["outlier_action"].startswith("none")


def test_missing_r_backend_has_no_silent_fallback():
    api = GeoToolbox().geochem
    frame = samples()
    frame["Au"] = frame.Au.astype(object)
    frame.loc[0, "Au"] = "<0.5"
    data = api.read(frame, schema=schema())
    with pytest.raises(MissingBackendError, match="Rscript"):
        api.impute(data, backend="zcompositions", method="lrEM", rscript="geo_toolbox_nonexistent_Rscript_123456")
    assert not data.imputed.to_numpy().any()


def test_backend_imputation_dispatch_preserves_original_status(monkeypatch):
    from geo_toolbox import backends

    api = GeoToolbox().geochem
    frame = samples()
    frame["Au"] = frame.Au.astype(object)
    frame.loc[0, "Au"] = "<0.5"
    data = api.read(frame, schema=schema())

    def impute(values, limits, status, method, **kwargs):
        assert method == "lrEM"
        assert status[0, 0] == "censored"
        assert limits[0, 0] == 0.0005
        output = values.copy()
        output[0, 0] = 0.00025
        return output, "test-version"

    monkeypatch.setattr(backends, "impute_zcompositions", impute)
    result = api.impute(data, backend="zcompositions")
    assert result.data.values.loc["01", "Au"] == 0.00025
    assert result.data.imputed.loc["01", "Au"]
    assert result.data.status.loc["01", "Au"] == "censored"
    assert np.isnan(data.values.loc["01", "Au"])
    with pytest.raises(DataValidationError, match="already"):
        api.impute(result.data)


def test_grid_centers_and_alignment():
    grid = GridSpec.from_bounds((0, 0, 100, 100), 50, "EPSG:2380")
    x, y = grid.centers()
    np.testing.assert_array_equal(x, [[25, 75], [25, 75]])
    np.testing.assert_array_equal(y, [[75, 75], [25, 25]])
    with pytest.raises(DataValidationError, match="align"):
        GridSpec.from_bounds((0, 0, 100, 100), 30, "EPSG:2380")
    with pytest.raises(DataValidationError, match="north-up"):
        GridSpec(2, 2, (50, 1, 0, 0, -50, 100), "EPSG:2380")


def test_clr_ilr_match_upstream_and_roundtrip():
    codata = pytest.importorskip("pyrolite.comp.codata")
    api = GeoToolbox().geochem
    data = api.read(samples(), schema=schema())
    original = data.values.to_numpy().copy()
    clr = api.transform(data, method="clr").data
    ilr = api.transform(data, method="ilr").data
    np.testing.assert_allclose(clr.values, codata.CLR(original))
    np.testing.assert_allclose(clr.values.sum(axis=1), 0, atol=1e-12)
    assert ilr.values.shape == (6, 1)
    basis = np.array(ilr.transform_state["basis"])
    np.testing.assert_allclose(ilr.values, codata.ILR(original, psi=basis))
    closed = original / original.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(codata.inverse_ILR(ilr.values.to_numpy(), psi=basis), closed)
    np.testing.assert_array_equal(data.values, original)
    with pytest.raises(DataValidationError, match="requires concentration"):
        api.transform(clr)
    with pytest.raises(DataValidationError, match="orthonormal"):
        api.transform(data, method="ilr", basis=[[1., 1.]])


def test_unresolved_values_block_logratio():
    frame = samples()
    frame.loc[0, "Au"] = np.nan
    api = GeoToolbox().geochem
    with pytest.raises(DataValidationError, match="Resolve all"):
        api.transform(api.read(frame, schema=schema()))


def test_interpolation_matches_upstream_and_raster_orientation(tmp_path):
    ok = pytest.importorskip("pykrige.ok")
    rasterio = pytest.importorskip("rasterio")
    api = GeoToolbox().geochem
    data = api.read(samples(), schema=schema())
    grid = GridSpec.from_bounds((0, 0, 100, 100), 50, "EPSG:2380")
    mask = np.array([[True, False], [True, True]])
    parameters = {"Au": {"psill": .02, "range": 100., "nugget": .001},
                  "As": {"psill": 200., "range": 100., "nugget": .1}}
    result = api.interpolate(data, grid=grid, support_mask=mask, variogram_parameters=parameters, chunk_size=1)
    x, y = grid.centers()
    model = ok.OrdinaryKriging(data.coordinates.x, data.coordinates.y, data.values.Au,
        variogram_model="spherical", variogram_parameters=parameters["Au"], coordinates_type="euclidean", exact_values=True,
        pseudo_inv=True, verbose=False, enable_plotting=False)
    prediction, variance = model.execute("points", x[mask], y[mask], backend="loop")
    np.testing.assert_allclose(result.predictions["Au"][mask], prediction)
    np.testing.assert_allclose(result.variances["Au"][mask], np.maximum(variance, 0), atol=1e-12)
    assert np.isnan(result.predictions["Au"][~mask]).all()
    output = result.export(tmp_path / "maps")
    with rasterio.open(output / "01_prediction.tif") as source:
        np.testing.assert_allclose(source.read(1), result.predictions["Au"], equal_nan=True)
        assert source.xy(0, 0) == (25., 75.)
        assert source.crs.to_epsg() == 2380
    restored = GridSpec.from_reference(output / "01_prediction.tif")
    assert restored == grid
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["layers"][1]["unit"] == "(ppm)^2"
    with pytest.raises(FileExistsError):
        result.export(output)


def test_spatial_guards():
    api = GeoToolbox().geochem
    data = api.read(samples(), schema=schema())
    grid = GridSpec.from_bounds((0, 0, 100, 100), 50, "EPSG:2380")
    with pytest.raises(DataValidationError, match="boolean"):
        api.interpolate(data, grid=grid, support_mask=np.ones((2, 2)))
    wrong_grid = GridSpec.from_bounds((0, 0, 100, 100), 50, "EPSG:3857")
    with pytest.raises(DataValidationError, match="CRS differ"):
        api.interpolate(data, grid=wrong_grid, support_mask=np.ones((2, 2), dtype=bool))
    geographic = api.read(samples(), schema=schema(crs="EPSG:4326"))
    with pytest.raises(DataValidationError, match="projected CRS"):
        api.interpolate(geographic, grid=grid, support_mask=np.ones((2, 2), dtype=bool))


def test_variogram_matches_gstools():
    gs = pytest.importorskip("gstools")
    api = GeoToolbox().geochem
    data = api.read(samples(), schema=schema())
    bins = np.array([0, 40, 80, 120, 160], dtype=float)
    result = api.variogram(data, element="Au", bin_edges=bins)
    distance, gamma, counts = gs.vario_estimate((data.coordinates.x, data.coordinates.y), data.values.Au,
                                              bin_edges=bins, return_counts=True)
    np.testing.assert_allclose(result["distance"], distance)
    np.testing.assert_allclose(result["semivariance"], gamma)
    np.testing.assert_array_equal(result["pair_counts"], counts)
    assert "not predictive validation" in result["fit_scope"]


def test_holdout_trains_only_on_training_ids(monkeypatch):
    from geo_toolbox import backends

    api = GeoToolbox().geochem
    data = api.read(samples(), schema=schema())
    calls = []

    def fit(x, y, z, **kwargs):
        np.testing.assert_array_equal(x, [0, 100, 0, 100])
        assert len(z) == 4
        return {"model": "spherical", "pykrige_parameters": {"psill": 1., "range": 100., "nugget": 0.}}

    def predict(x, y, z, px, py, *, variogram_parameters, **kwargs):
        assert variogram_parameters == {"psill": 1., "range": 100., "nugget": 0.}
        calls.append((np.array(x), np.array(z), np.array(px)))
        return np.full(len(px), z.mean()), np.zeros(len(px)), np.array([1., 100., 0.])

    monkeypatch.setattr(backends, "variogram_gstools", fit)
    monkeypatch.setattr(backends, "ordinary_kriging", predict)
    result = api.validate(data, train_ids=["01", "02", "03", "04"], test_ids=["05", "06"])
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], [0, 100, 0, 100])
    np.testing.assert_allclose(calls[0][1], [0.1, 0.2, 0.3, 0.4])
    assert result["metrics"]["Au"]["mae"] == pytest.approx(0.07)
    assert result["metrics"]["Au"]["rmse"] == pytest.approx(0.07)
    data.imputed.iloc[0, 0] = True
    with pytest.raises(DataValidationError, match="Batch-imputed"):
        api.validate(data, train_ids=["01", "02", "03", "04"], test_ids=["05", "06"])


def test_r_backend_rejects_inapplicable_algorithm_without_fallback():
    from geo_toolbox import backends

    values = np.array([[1., np.nan], [2., 3.]])
    limits = np.ones((2, 2))
    only_missing = np.array([["observed", "missing"], ["observed", "observed"]])
    with pytest.raises(DataValidationError, match="requires both"):
        backends.impute_zcompositions(values, limits, only_missing, "lrEMplus")
    mixed = np.array([["observed", "missing"], ["censored", "observed"]])
    with pytest.raises(DataValidationError, match="Mixed missing"):
        backends.impute_zcompositions(values, limits, mixed, "lrEM")


def test_r_bridge_rejects_changed_observations_and_censoring_violations(monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from geo_toolbox import backends
    from geo_toolbox.errors import BackendExecutionError

    monkeypatch.setattr(backends.shutil, "which", lambda _: "Rscript")
    values = np.array([[np.nan, 10.], [2., 20.], [3., 30.]])
    limits = np.array([[1., 0.], [0., 0.], [0., 0.]])
    status = np.array([["censored", "observed"], ["observed", "observed"], ["observed", "observed"]])
    cases = [np.array([[0.5, 11.], [2., 20.], [3., 30.]]),
             np.array([[1.5, 10.], [2., 20.], [3., 30.]]),
             np.array([[0.5, 10.], [2., 20.], [3., 30.]])]

    def execute(arguments, **kwargs):
        assert arguments[1] == "--vanilla"
        assert arguments[5] == "lrEM"
        assert kwargs["check"] is False
        pd.DataFrame(cases.pop(0)).to_csv(arguments[6], index=False)
        Path(arguments[7]).write_text("test-version", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backends.subprocess, "run", execute)
    with pytest.raises(BackendExecutionError, match="changed observed"):
        backends.impute_zcompositions(values, limits, status, "lrEM")
    with pytest.raises(BackendExecutionError, match="below their detection"):
        backends.impute_zcompositions(values, limits, status, "lrEM")
    output, backend_version = backends.impute_zcompositions(values, limits, status, "lrEM")
    assert output[0, 0] == 0.5 and backend_version == "test-version"


def test_negative_predictions_remain_flagged_not_clipped(monkeypatch):
    from geo_toolbox import backends

    api = GeoToolbox().geochem
    data = api.read(samples(), schema=schema())
    grid = GridSpec.from_bounds((0, 0, 100, 100), 50, "EPSG:2380")

    def predict(x, y, z, px, py, **kwargs):
        return np.full(len(px), -0.1), np.ones(len(px)), np.array([1., 10., 0.])

    monkeypatch.setattr(backends, "ordinary_kriging", predict)
    with pytest.warns(UserWarning, match="negative concentration"):
        result = api.interpolate(data, grid=grid, support_mask=np.ones((2, 2), dtype=bool),
                                 variogram_parameters={e: {"psill": 1., "range": 10., "nugget": 0.}
                                                       for e in data.schema.elements})
    assert (result.predictions["Au"] == -0.1).all()
    assert result.history[-1]["negative_prediction_counts"]["Au"] == 4


def test_multilevel_excel_and_unknown_crs(tmp_path):
    pytest.importorskip("openpyxl")
    from openpyxl import Workbook

    path = tmp_path / "two_headers.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "samples"
    sheet.append(["id", "east", "north", "concentrations", ""])
    sheet.append(["", "", "", "Au", "As"])
    for row in samples().itertuples(index=False, name=None):
        sheet.append(list(row))
    workbook.save(path)
    api = GeoToolbox().geochem
    data = api.read(path, schema=schema(crs=None), sheet_name="samples", header=[0, 1])
    assert data.crs is None
    assert data.values.index.tolist() == samples().id.tolist()
    np.testing.assert_allclose(data.values.Au, samples().Au * 0.001)
    with pytest.raises(DataValidationError, match="CRS is unknown"):
        api.variogram(data, element="Au")


def test_raw_export_has_one_id_column(tmp_path):
    data = GeoToolbox().geochem.read(samples(), schema=schema())
    path = data.export(tmp_path / "samples")
    exported = pd.read_csv(path / "raw.csv", dtype={"id": str})
    assert exported.columns.tolist() == samples().columns.tolist()
    assert exported.id.tolist() == samples().id.tolist()


def test_job_no_hidden_imputation_or_validation(tmp_path):
    source = tmp_path / "samples.csv"
    samples().to_csv(source, index=False)
    result = GeoToolbox().geochem.run(GeochemJob(source=source, schema=schema()))
    assert result.features is None and result.maps is None
    assert result.validation["status"] == "not_run"
    result.export(tmp_path / "job")
    assert (tmp_path / "job" / "samples" / "status.csv").exists()
    assert (tmp_path / "job" / "run.json").exists()
