import numpy as np
import pandas as pd
import pytest

from geo_toolbox import GridSpec, MagneticGrid
from geo_toolbox.errors import DataValidationError
from geo_toolbox.gui_tasks import describe_error, execute, export_result, import_source, preflight, preview_source


def loaded(tmp_path):
    path = tmp_path / "points.csv"
    pd.DataFrame({"id": ["a", "b", "c", "d"], "x": [0, 100, 0, 100],
                  "y": [0, 0, 100, 100], "z": [0] * 4, "v": [-1, 0, 1, 2]}).to_csv(path, index=False)
    preview = preview_source(path)
    mapping = dict(sample_id="id", x="x", y="y", upward="z", value="v", crs="", unit="nT", stage="anomaly")
    return preview, mapping, import_source(preview, mapping=mapping)


def test_preview_and_explicit_import_mapping(tmp_path):
    preview, mapping, data = loaded(tmp_path)
    assert preview["sheets"] == [] and len(preview["preview"]) == 4
    assert data.schema.crs is None and data.values.iloc[0] == -1
    qc = preflight(data, operation="check", parameters={})
    assert qc["ok"] and qc["warnings"]
    assert execute(data, operation="check", parameters={})["report"]["rows"] == 4
    mapping["unit"] = ""
    with pytest.raises(DataValidationError):
        import_source(preview, mapping=mapping)


def test_generated_ids_and_split_retain_blank_rows(tmp_path):
    preview, mapping, _ = loaded(tmp_path)
    preview["frame"]["id"] = ["1-1", "1-2", "", "1-4"]
    mapping.update(generated_ids=True, split_point_line=True, point_line="id")
    data = import_source(preview, mapping=mapping)
    assert len(data.values) == 4 and data.raw.loc["row4", "id"] == ""
    assert data.raw["__gui_line"].isna().sum() == 1
    assert data.schema.crs is None
    with pytest.raises(DataValidationError):
        execute(data, operation="spikes", parameters={"window": 3, "threshold_nT": 100})


def test_spatial_preflight_does_not_fit_unknown_crs(tmp_path):
    _, _, data = loaded(tmp_path)
    result = preflight(data, operation="interpolate", parameters=dict(
        xmin=0, ymin=0, xmax=100, ymax=100, resolution=50, upward=0,
        max_distance=100, method="equivalent_sources", depth=100))
    assert not result["ok"] and "CRS" in result["errors"][0]


def test_transform_repeat_rtp_and_holes_blocked():
    spec = GridSpec.from_bounds((0, 0, 80, 80), 10, "EPSG:21456")
    grid = MagneticGrid(spec, np.ones((8, 8)), np.ones((8, 8), bool), "rtp", "nT", 0)
    options = dict(method="reduction_to_pole", padding=0, trim=0,
                   inclination=55, declination=0, assume_induced=True, max_gain=10)
    assert not preflight(grid, operation="transform", parameters=options)["ok"]
    support = grid.support.copy(); support[3, 3] = False
    result = preflight(grid.copy(support=support), operation="transform", parameters=dict(method="derivative_upward", padding=0, trim=0))
    assert not result["ok"] and "FFT" in result["errors"][0]


def test_export_refuses_existing_and_error_details(tmp_path):
    _, _, data = loaded(tmp_path)
    output = export_result(data, directory=tmp_path / "result", report={"test": True})
    assert (output / "gui_report.json").exists()
    with pytest.raises(DataValidationError):
        export_result(data, directory=output, report={})
    described = describe_error(DataValidationError("CRS unknown"))
    assert described["kind"] == "validation" and "CRS unknown" in described["details"]


@pytest.mark.parametrize("options", [
    dict(method="not_a_method", padding=0, trim=0),
    dict(method="upward_continuation", padding=0, trim=0, height_displacement=-1),
    dict(method="directional_derivative", padding=0, trim=0),
    dict(method="derivative_upward", padding=0, trim=0, inclination=55),
    dict(method="reduction_to_pole", padding=0, trim=0, inclination=95, declination=0, max_gain=10, assume_induced=True),
    dict(method="reduction_to_pole", padding=0, trim=0, inclination=55, declination=0, max_gain=-1, assume_induced=True),
    dict(method="derivative_upward", padding=10000, trim=0),
])
def test_transform_preflight_rejects_bad_parameters(options):
    spec = GridSpec.from_bounds((0, 0, 80, 80), 10, "EPSG:21456")
    grid = MagneticGrid(spec, np.ones((8, 8)), np.ones((8, 8), bool), "anomaly", "nT", 0)
    assert not preflight(grid, operation="transform", parameters=options)["ok"]


def test_gui_grid_and_transform_dispatch_preserve_input(tmp_path):
    preview, mapping, _ = loaded(tmp_path)
    mapping["crs"] = "EPSG:21456"
    data = import_source(preview, mapping=mapping)
    params = dict(xmin=0, ymin=0, xmax=100, ymax=100, resolution=25, upward=0,
                  max_distance=150, method="verde_spline", damping=.1)
    assert preflight(data, operation="interpolate", parameters=params)["ok"]
    result = execute(data, operation="interpolate", parameters=params)
    assert result["data"].values.shape == (4, 4) and result["data"].support.all()
    transformed = execute(result["data"], operation="transform", parameters=dict(method="derivative_upward", padding=0, trim=0))
    assert transformed["data"].unit == "nT/m"
    assert result["data"].unit == "nT" and data.values.tolist() == [-1, 0, 1, 2]
    assert data.history[-1]["parameters"]["source_sha256"] == preview["source_sha256"]


def test_excel_header_sheet_and_leading_zero_ids(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"编号": ["00123", "00124"], "X": [1, 2], "Y": [3, 4], "值": [-1, 0]}).to_excel(
            writer, sheet_name="磁测", index=False, startrow=1)
    preview = preview_source(path, sheet_name="磁测", header=1)
    assert preview["sheets"] == ["磁测"] and preview["frame"]["编号"].tolist() == ["00123", "00124"]
    result = import_source(preview, mapping=dict(sample_id="编号", x="X", y="Y", value="值", unit="uT", stage="anomaly", crs=""))
    assert result.values.tolist() == [-1000, 0] and result.raw["值"].tolist() == ["-1", "0"]
    with pytest.raises(DataValidationError):
        preview_source(path, sheet_name=None, header=1)
    with pytest.raises(DataValidationError):
        preview_source(path, header=-1)


def test_import_duplicate_mapping_and_missing_source(tmp_path):
    preview, mapping, _ = loaded(tmp_path)
    mapping["value"] = "x"
    with pytest.raises(DataValidationError):
        import_source(preview, mapping=mapping)
    with pytest.raises(DataValidationError):
        preview_source(tmp_path / "absent.csv")
