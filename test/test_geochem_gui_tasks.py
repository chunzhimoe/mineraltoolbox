import json

import numpy as np
import pandas as pd
import pytest

from geo_toolbox import GridSpec
from geo_toolbox.errors import DataValidationError
from geo_toolbox.geochem_gui_tasks import execute, export_result, import_source, preflight, preview_source


def loaded(tmp_path, *, crs="", censored=False):
    path = tmp_path / "geochem.csv"
    values = [100., 200., 300., 400., 180., 320.]
    if censored:
        values[0] = "<0.5"
    pd.DataFrame({"id": ["01", "02", "03", "04", "05", "06"],
                  "x": [0., 100., 0., 100., 25., 75.],
                  "y": [0., 0., 100., 100., 25., 75.],
                  "Au": values, "As": [10., 23., 31., 45., 18., 32.],
                  "Sb": [2., 4., 8., 11., 5., 7.]}).to_csv(path, index=False)
    preview = preview_source(path)
    mapping = dict(sample_id="id", x="x", y="y", crs=crs, elements=["Au", "As", "Sb"],
                   units={"Au": "ppb", "As": "ppm", "Sb": "ppm"}, zero_policy="invalid")
    return preview, mapping, import_source(preview, mapping=mapping)


def imputation(method="multRepl"):
    return dict(method=method, frac=.65, tolerance=.0001, max_iter=50, ini_cov="complete.obs")


def test_mapping_normalized_units_raw_and_provenance(tmp_path):
    preview, mapping, data = loaded(tmp_path)
    assert data.values.index.tolist() == ["01", "02", "03", "04", "05", "06"]
    assert data.values.loc["01", "Au"] == .1
    assert data.raw.loc["01", "Au"] == "100.0"
    assert data.units == {"Au": "ppm", "As": "ppm", "Sb": "ppm"}
    assert data.history[-1]["parameters"]["source_sha256"] == preview["source_sha256"]
    assert preflight(data, operation="check", parameters={})["ok"]
    assert execute(data, operation="check", parameters={})["report"]["unresolved_cells"] == 0
    mapping["units"]["Au"] = ""
    with pytest.raises(DataValidationError):
        import_source(preview, mapping=mapping)


def test_limits_units_tokens_zero_policy_and_audit_id(tmp_path):
    preview, mapping, _ = loaded(tmp_path)
    preview["frame"]["Au"] = ["<0.5", "", "0", "bad", "-1", "2"]
    preview["frame"].loc[2, "id"] = ""
    mapping.update(generated_ids=True, zero_policy="censored", detection_limits={"Au": "0.5"})
    data = import_source(preview, mapping=mapping)
    assert len(data.values) == 6 and data.raw.loc["row4", "id"] == ""
    assert data.status.Au.tolist() == ["censored", "missing", "censored", "invalid", "invalid", "observed"]
    np.testing.assert_allclose(data.detection_limits.Au, .0005)
    assert not preflight(data, operation="impute", parameters=imputation())["ok"]
    mapping["detection_limit_columns"] = {"Au": "Sb"}
    with pytest.raises(DataValidationError):
        import_source(preview, mapping=mapping)


def test_variable_limits_conflicting_tokens_and_mapping(tmp_path):
    preview, mapping, _ = loaded(tmp_path, censored=True)
    preview["frame"]["dl"] = [.5, .6, .7, .8, .9, 1.]
    mapping["detection_limit_columns"] = {"Au": "dl"}
    data = import_source(preview, mapping=mapping)
    np.testing.assert_allclose(data.detection_limits.Au, np.arange(.5, 1.01, .1) * .001)
    preview["frame"].loc[0, "Au"] = "<0.2"
    with pytest.raises(DataValidationError, match="contradictory"):
        import_source(preview, mapping=mapping)
    mapping["x"] = "Au"
    with pytest.raises(DataValidationError):
        import_source(preview, mapping=mapping)


def test_multrepl_dispatch_preserves_observed_raw_and_blocks_holdout(tmp_path):
    _, _, data = loaded(tmp_path, crs="EPSG:2380", censored=True)
    result = execute(data, operation="impute", parameters=imputation())
    output = result["data"]
    assert result["kind"] == "concentration"
    assert 0 < output.values.loc["01", "Au"] < .0005
    assert output.imputed.loc["01", "Au"]
    assert output.status.loc["01", "Au"] == "censored"
    assert np.isnan(data.values.loc["01", "Au"]) and not data.imputed.to_numpy().any()
    pd.testing.assert_frame_equal(data.raw, output.raw)
    observed = data.status == "observed"
    np.testing.assert_array_equal(output.values.to_numpy()[observed.to_numpy()], data.values.to_numpy()[observed.to_numpy()])
    assert not preflight(output, operation="impute", parameters=imputation())["ok"]
    heldout = preflight(output, operation="validate", parameters=dict(
        train_ids="01\n02\n03\n04", test_ids="05\n06", variogram="spherical", nlags=3))
    assert not heldout["ok"] and "填补" in heldout["errors"][0]


def test_em_mixed_missing_preflight_and_unresolved_transform(tmp_path):
    preview, mapping, _ = loaded(tmp_path, censored=True)
    preview["frame"].loc[1, "Au"] = ""
    data = import_source(preview, mapping=mapping)
    assert not preflight(data, operation="impute", parameters=imputation("lrEM"))["ok"]
    assert preflight(data, operation="impute", parameters=imputation("lrEMplus"))["ok"]
    assert not preflight(data, operation="transform", parameters={"method": "clr"})["ok"]


def test_transform_branch_does_not_replace_concentration(tmp_path):
    _, _, data = loaded(tmp_path)
    before = data.values.copy(deep=True)
    for method, dimension in (("clr", 3), ("ilr", 2)):
        result = execute(data, operation="transform", parameters={"method": method})
        assert result["kind"] == "features" and result["data"].domain == method
        assert result["data"].values.shape == (6, dimension)
        assert result["data"].units == {column: "dimensionless" for column in result["data"].values}
        assert not preflight(result["data"], operation="check", parameters={})["ok"]
    pd.testing.assert_frame_equal(data.values, before)
    assert data.domain == "concentration"


def test_spatial_unknown_crs_duplicate_and_holdout_guards(tmp_path):
    preview, mapping, data = loaded(tmp_path)
    options = dict(train_ids="01\n02\n03", test_ids="04", variogram="spherical", nlags=3)
    assert not preflight(data, operation="validate", parameters=options)["ok"]
    mapping["crs"] = "EPSG:2380"
    preview["frame"].loc[1, ["x", "y"]] = ["0", "0"]
    duplicate = import_source(preview, mapping=mapping)
    assert not preflight(duplicate, operation="validate", parameters=options)["ok"]
    preview["frame"].loc[1, ["x", "y"]] = ["100", "0"]
    good = import_source(preview, mapping=mapping)
    for train, test in (("01\n02", "03"), ("01\n02\n03", "03"), ("01\n02\n03", "unknown")):
        assert not preflight(good, operation="validate", parameters=dict(options, train_ids=train, test_ids=test))["ok"]


def test_validation_preserves_ids_and_uses_training_only(tmp_path, monkeypatch):
    from geo_toolbox import backends
    _, _, data = loaded(tmp_path, crs="EPSG:2380")
    calls = []

    def fit(x, y, z, **kwargs):
        np.testing.assert_array_equal(x, [0, 100, 0, 100])
        assert len(z) == 4
        return {"model": "spherical", "pykrige_parameters": {"psill": 1., "range": 100., "nugget": 0.}}

    def predict(x, y, z, px, py, *, variogram_parameters, **kwargs):
        assert variogram_parameters == {"psill": 1., "range": 100., "nugget": 0.}
        calls.append((x.copy(), z.copy(), px.copy()))
        return np.full(len(px), z.mean()), np.zeros(len(px)), np.array([1, 100, 0])

    monkeypatch.setattr(backends, "variogram_gstools", fit)
    monkeypatch.setattr(backends, "ordinary_kriging", predict)
    output = execute(data, operation="validate", parameters=dict(
        train_ids="01\n02\n03\n04", test_ids="05\n06", variogram="spherical", nlags=3))
    assert output["kind"] == "report"
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0][0], [0, 100, 0, 100])
    np.testing.assert_allclose(calls[0][1], [.1, .2, .3, .4])
    np.testing.assert_array_equal(calls[0][2], [25, 75])
    assert output["report"]["train_ids"] == ["01", "02", "03", "04"]
    assert output["report"]["metrics"]["Au"]["rmse"] == pytest.approx(.07)


def test_support_alignment_kriging_dispatch_and_exports(tmp_path):
    import rasterio
    from affine import Affine
    _, _, data = loaded(tmp_path, crs="EPSG:2380")
    spec = GridSpec.from_bounds((0, 0, 100, 100), 50, data.crs)
    support_path = tmp_path / "support.tif"
    with rasterio.open(support_path, "w", driver="GTiff", count=1, width=2, height=2,
                       dtype="uint8", crs=data.crs, transform=Affine(*spec.transform), nodata=0) as output:
        output.write(np.array([[1, 0], [1, 1]], dtype=np.uint8), 1)
    options = dict(xmin=0, ymin=0, xmax=100, ymax=100, resolution=50,
                   support_path=str(support_path), variogram="spherical", nlags="", bin_edges="0,40,80,120,160",
                   exact_values=True, pseudo_inv=True, n_closest_points="", chunk_size=1)
    assert preflight(data, operation="interpolate", parameters=options)["ok"]
    assert not preflight(data, operation="interpolate", parameters=dict(options, resolution=25))["ok"]
    result = execute(data, operation="interpolate", parameters=options)
    assert result["kind"] == "maps" and result["data"].support.sum() == 3
    assert np.isnan(result["data"].predictions["Au"][0, 1])
    assert np.isnan(result["data"].variances["Au"][0, 1])
    assert result["data"].history[-1]["parameters"]["source"] == str(support_path)
    directory = export_result(result["data"], directory=tmp_path / "maps", report=result["report"])
    assert (directory / "gui_report.json").is_file()
    assert (directory / "support.tif").is_file()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["layers"][1]["unit"] == "(ppm)^2"
    with pytest.raises(DataValidationError):
        export_result(result["data"], directory=directory, report={})
    report_dir = export_result({"test": True}, directory=tmp_path / "report", report={})
    assert json.loads((report_dir / "report.json").read_text(encoding="utf-8")) == {"test": True}


def test_variogram_dispatch_returns_backend_report(tmp_path):
    _, _, data = loaded(tmp_path, crs="EPSG:2380")
    p = dict(element="Au", model="spherical", estimator="matheron", bin_edges="0,40,80,120,160")
    output = execute(data, operation="variogram", parameters=p)
    assert output["kind"] == "report" and len(output["report"]["distance"]) == 4
    np.testing.assert_array_equal(output["report"]["pair_counts"], [2, 5, 6, 2])
    assert "not predictive validation" in output["report"]["fit_scope"]
    assert not preflight(data, operation="variogram", parameters=dict(p, bin_edges="0,40,30,120"))["ok"]


@pytest.mark.parametrize("mode,fixed,columns", [
    ("fixed", {}, {}), ("fixed", {"Au": ""}, {}),
    ("column", {}, {}), ("column", {}, {"Au": ""}),
    ("token_only", {"Au": .5}, {}), ("wrong", {}, {}),
])
def test_detection_mode_never_silently_falls_back(tmp_path, mode, fixed, columns):
    preview, mapping, _ = loaded(tmp_path, censored=True)
    mapping.update(detection_modes={"Au": mode}, detection_limits=fixed, detection_limit_columns=columns)
    with pytest.raises(DataValidationError):
        import_source(preview, mapping=mapping)


def test_literature_em_dispatch_uses_existing_algorithm_without_mutation(tmp_path):
    from geo_toolbox import GeochemSchema
    from geo_toolbox.geochem import GeochemistryAPI
    rng = np.random.default_rng(427)
    matrix = np.exp(rng.normal(size=(80, 4)) * .35 + [1., 2., 3., 4.])
    raw = pd.DataFrame(matrix, columns=["A", "B", "C", "D"]).astype(object)
    raw.insert(0, "id", [f"p{number:03d}" for number in range(80)])
    raw["x"], raw["y"] = np.arange(80), np.arange(80) * 2
    raw.loc[2, "B"] = f"<{matrix[2, 1] * 1.5}"
    raw.loc[35, "C"] = ""
    path = tmp_path / "em.csv"
    raw.to_csv(path, index=False)
    mapping = dict(sample_id="id", x="x", y="y", elements=["A", "B", "C", "D"],
                   units={name: "ppm" for name in ["A", "B", "C", "D"]})
    data = import_source(preview_source(path), mapping=mapping)
    params = dict(imputation("lrEMplus"), max_iter=500)
    wrapped = execute(data, operation="impute", parameters=params)
    direct = GeochemistryAPI().impute(data, method="lrEMplus", max_iter=500)
    pd.testing.assert_frame_equal(wrapped["data"].values, direct.data.values)
    algorithm = wrapped["report"]["algorithm"]
    assert algorithm["algorithm"] == "literature_alr_em_v1"
    assert algorithm["r_equivalence"] == "intentionally_not_equivalent"
    assert algorithm["converged"]
    assert not data.imputed.to_numpy().any()
    assert data.raw.loc["p002", "B"].startswith("<")
