import numpy as np
import pandas as pd
import pytest

from geo_toolbox import GeoToolbox, GeochemJob, GeochemSchema, GridSpec
from geo_toolbox.errors import DataValidationError


def dataset():
    rng = np.random.default_rng(71)
    coordinates = rng.uniform(0, 4000, (160, 2))
    values = np.exp(.3 * np.sin(coordinates[:, 0] / 300) + .2 * np.cos(coordinates[:, 1] / 400)
                    + rng.normal(0, .15, len(coordinates)))
    frame = pd.DataFrame({"id": [f"p{i:03d}" for i in range(len(values))],
                          "east": coordinates[:, 0], "north": coordinates[:, 1], "As": values})
    schema = GeochemSchema("id", "east", "north", "EPSG:2380", ["As"], {"As": "ppm"})
    return GeoToolbox().geochem.read(frame, schema=schema)


def test_default_binning_is_gstools_default_and_explicit_nlags_uses_third_extent():
    import gstools as gs
    api, data = GeoToolbox().geochem, dataset()
    result = api.variogram(data, element="As", model="exponential")
    xy = (data.coordinates.x, data.coordinates.y)
    distance, gamma, counts = gs.vario_estimate(xy, data.values.As, return_counts=True)
    np.testing.assert_array_equal(result["distance"], distance)
    np.testing.assert_array_equal(result["semivariance"], gamma)
    np.testing.assert_array_equal(result["pair_counts"], counts)
    assert result["history"][-1]["parameters"]["bin_edges"] is None
    explicit = api.variogram(data, element="As", nlags=6)
    edges = gs.standard_bins(xy, bin_no=6)
    np.testing.assert_allclose(explicit["distance"], (edges[:-1] + edges[1:]) / 2)
    assert edges[-1] == pytest.approx(np.hypot(np.ptp(data.coordinates.x), np.ptp(data.coordinates.y)) / 3)


@pytest.mark.parametrize("model,constructor", [("spherical", "Spherical"), ("exponential", "Exponential"), ("gaussian", "Gaussian")])
def test_gstools_to_pykrige_preserves_entire_variogram(model, constructor):
    import gstools as gs
    from pykrige import variogram_models
    result = GeoToolbox().geochem.variogram(dataset(), element="As", model=model)
    covariance = getattr(gs, constructor)(dim=2, **result["parameters"])
    p = result["pykrige_parameters"]
    distance = np.linspace(0, p["range"] * 2, 101)
    curve = getattr(variogram_models, model + "_variogram_model")([p["psill"], p["range"], p["nugget"]], distance)
    np.testing.assert_allclose(curve, covariance.variogram(distance), rtol=1e-12, atol=1e-12)


def test_real_interpolation_and_run_use_exact_report_parameters(monkeypatch):
    from pykrige.ok import OrdinaryKriging
    api, data = GeoToolbox().geochem, dataset()
    report = api.variogram(data, element="As", model="exponential")
    seen = []
    original = OrdinaryKriging.__init__

    def capture(self, *args, **kwargs):
        seen.append(kwargs["variogram_parameters"].copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(OrdinaryKriging, "__init__", capture)
    grid = GridSpec.from_bounds((1000, 1000, 1400, 1400), 100, data.crs)
    options = dict(grid=grid, support_mask=np.ones((4, 4), bool), variogram="exponential")
    result = api.interpolate(data, **options)
    assert seen == [report["pykrige_parameters"]]
    assert result.history[-1]["variograms"]["As"]["pykrige_parameters"] == report["pykrige_parameters"]
    direct = OrdinaryKriging(data.coordinates.x, data.coordinates.y, data.values.As,
        variogram_model="exponential", variogram_parameters=report["pykrige_parameters"],
        exact_values=True, pseudo_inv=True, coordinates_type="euclidean")
    x, y = grid.centers()
    predicted, variance = direct.execute("points", x.ravel(), y.ravel(), backend="loop")
    np.testing.assert_allclose(result.predictions["As"].ravel(), predicted)
    np.testing.assert_allclose(result.variances["As"].ravel(), variance, atol=1e-10)
    job = GeochemJob(source=data.raw.reset_index(drop=True), schema=data.schema, interpolation=options)
    pipeline = api.run(job)
    np.testing.assert_allclose(pipeline.maps.predictions["As"], result.predictions["As"])
    explicit = api.interpolate(data, **options, variogram_parameters={"As": report["pykrige_parameters"]})
    np.testing.assert_allclose(explicit.predictions["As"], result.predictions["As"])
    with pytest.raises(DataValidationError, match="cannot be combined"):
        api.interpolate(data, **options, variogram_parameters={"As": report["pykrige_parameters"]}, nlags=6)


def test_real_validation_fits_training_only_and_test_values_cannot_change_predictions(monkeypatch):
    from pykrige.ok import OrdinaryKriging
    api, data = GeoToolbox().geochem, dataset()
    train, test = list(data.values.index[:120]), list(data.values.index[120:])
    original = OrdinaryKriging.__init__
    seen = []

    def capture(self, *args, **kwargs):
        seen.append(kwargs["variogram_parameters"].copy())
        assert len(args[0]) == len(train)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(OrdinaryKriging, "__init__", capture)
    output = api.validate(data, train_ids=train, test_ids=test, variogram="exponential")
    changed = data.copy()
    changed.values.loc[test, "As"] *= 1000
    repeated = api.validate(changed, train_ids=train, test_ids=test, variogram="exponential")
    np.testing.assert_array_equal(output["predictions"]["As"], repeated["predictions"]["As"])
    assert seen[0] == seen[1] == output["variograms"]["As"]["pykrige_parameters"]
    fit_history = output["variograms"]["As"]["history"][-1]
    assert fit_history["operation"] == "variogram"
    assert output["history"][-1]["fit_scope"] == "training IDs only"


@pytest.mark.parametrize("model", ["linear", "power"])
def test_unsupported_models_rejected_before_fit(model, monkeypatch):
    from geo_toolbox import backends
    api, data = GeoToolbox().geochem, dataset()
    monkeypatch.setattr(backends, "ordinary_kriging", lambda *args, **kwargs: pytest.fail("backend must not run"))
    grid = GridSpec.from_bounds((0, 0, 100, 100), 50, data.crs)
    with pytest.raises(DataValidationError, match="Unsupported variogram"):
        api.interpolate(data, grid=grid, support_mask=np.ones((2, 2), bool), variogram=model)
    with pytest.raises(DataValidationError, match="Unsupported variogram"):
        api.validate(data, train_ids=list(data.values.index[:120]), test_ids=list(data.values.index[120:]), variogram=model)


def test_sparse_bins_fail_without_silent_autofit():
    api, data = GeoToolbox().geochem, dataset()
    ids = list(data.values.index[:3])
    with pytest.raises(DataValidationError, match="populated variogram bins"):
        api.validate(data, train_ids=ids, test_ids=[data.values.index[4]], bin_edges=[0, 1, 2, 3])
