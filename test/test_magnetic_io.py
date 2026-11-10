import json

import numpy as np
import pandas as pd
import pytest

from geo_toolbox import GeoToolbox
from geo_toolbox.core import GridSpec
from geo_toolbox.errors import DataValidationError
from geo_toolbox.magnetic_core import (MagneticDataset, MagneticGrid, MagneticSchema,
                                       read_magnetic)


def _schema(**overrides):
    base = dict(sample_id="点号", x="X", y="Y", value="△T", crs="EPSG:21456",
                unit="nT", stage="anomaly")
    base.update(overrides)
    return MagneticSchema(**base)


def _frame():
    return pd.DataFrame({
        "点号": ["A1", "A2", "A3", "A4"],
        "X": [14000.0, 14020.0, 14040.0, 14060.0],
        "Y": [80000.0, 80020.0, 80040.0, 80060.0],
        "△T": [-3.5, 0.0, "bad", None],
    })


def test_read_dataframe_preserves_signed_zero_and_units():
    data = read_magnetic(_frame(), schema=_schema())
    assert list(data.values)[:2] == [-3.5, 0.0]
    assert list(data.status) == ["observed", "observed", "invalid", "missing"]
    assert data.stage == "anomaly"
    assert data.active.all()
    assert data.history[0]["operation"] == "read"
    assert data.history[0]["parameters"]["original_unit"] == "nT"
    assert data.history[0]["parameters"]["output_unit"] == "nT"


def test_unit_conversion_uT_and_T():
    data = read_magnetic(_frame(), schema=_schema(unit="uT"))
    assert data.values.iloc[0] == pytest.approx(-3500.0)
    data = read_magnetic(_frame(), schema=_schema(unit="T"))
    assert data.values.iloc[0] == pytest.approx(-3.5e9)
    assert data.history[0]["parameters"]["original_unit"] == "T"


def test_raw_is_deep_copy_not_source():
    frame = _frame()
    data = read_magnetic(frame, schema=_schema())
    frame.loc[0, "△T"] = 99999.0
    assert data.raw.loc["A1", "△T"] == -3.5
    assert "点号" in data.raw.columns
    assert list(data.raw.index) == ["A1", "A2", "A3", "A4"]


def test_csv_leading_zero_ids(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("点号,X,Y,△T\n00123,14000,80000,1.5\n00456,14010,80010,2.5\n",
                    encoding="utf-8")
    data = read_magnetic(path, schema=_schema())
    assert list(data.values.index) == ["00123", "00456"]


def test_excel_chinese_header_and_sheet_choice(tmp_path):
    path = tmp_path / "测线.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"说明行": ["not a header"]}).to_excel(writer, sheet_name="数据", index=False)
        frame = _frame()
        frame.loc[0, "点号"] = "00123"
        frame.to_excel(writer, sheet_name="数据", index=False, startrow=1)
        pd.DataFrame({"other": [1]}).to_excel(writer, sheet_name="备注", index=False)
    schema = _schema()
    data = read_magnetic(path, schema=schema, sheet_name="数据", header=1)
    assert list(data.values.index) == ["00123", "A2", "A3", "A4"]
    with pytest.raises(DataValidationError):
        read_magnetic(path, schema=schema, sheet_name=None, header=1)
    with pytest.raises(DataValidationError):
        read_magnetic(path, schema=schema, sheet_name="备注", header=0)


def test_schema_rejections():
    with pytest.raises(DataValidationError):
        _schema(unit="gamma")
    with pytest.raises(DataValidationError):
        _schema(stage="reduction_to_pole")
    with pytest.raises(DataValidationError):
        _schema(crs="not a crs")
    with pytest.raises(DataValidationError):
        _schema(upward="X")
    with pytest.raises(DataValidationError):
        _schema(value="")
    with pytest.raises(DataValidationError):
        _schema(stage=["anomaly"])
    with pytest.raises(DataValidationError):
        _schema(sample_id=None)
    with pytest.raises(DataValidationError):
        _schema(sample_id="   ")
    with pytest.raises(DataValidationError):
        _schema(x=None)
    with pytest.raises(DataValidationError):
        _schema(upward="   ")
    with pytest.raises(DataValidationError):
        _schema(line=123)
    with pytest.raises(DataValidationError):
        _schema(crs=12345)


def test_id_and_column_rejections(tmp_path):
    schema = _schema()
    dup = _frame()
    dup.loc[1, "点号"] = "A1"
    with pytest.raises(DataValidationError):
        read_magnetic(dup, schema=schema)
    blank = _frame()
    blank.loc[2, "点号"] = "  "
    with pytest.raises(DataValidationError):
        read_magnetic(blank, schema=schema)
    with pytest.raises(DataValidationError):
        read_magnetic(_frame().drop(columns=["△T"]), schema=schema)
    with pytest.raises(DataValidationError):
        read_magnetic(pd.DataFrame(columns=["点号", "X", "Y", "△T"]), schema=schema)
    with pytest.raises(DataValidationError):
        read_magnetic(_frame(), schema="not a schema")


def test_csv_multirow_header_rejected(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("a,b,c,d\n点号,X,Y,△T\nA1,14000,80000,1.5\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        read_magnetic(path, schema=_schema(), header=[0, 1])


def test_excel_multirow_header_chinese(tmp_path):
    path = tmp_path / "multi.xlsx"
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["测线信息", "测线信息", "测线信息", "观测结果"])
    sheet.append(["点号", "X", "Y", "△T"])
    for row in _frame().itertuples(index=False):
        sheet.append(list(row))
    book.save(path)
    data = read_magnetic(path, schema=_schema(), header=[0, 1])
    assert list(data.values.index) == ["A1", "A2", "A3", "A4"]


def test_dataset_export_roundtrip_no_overwrite(tmp_path):
    frame = _frame()
    frame["L1"] = ["L1", "L1", "L1", "L1"]
    data = read_magnetic(frame, schema=_schema(line="L1"))
    directory = data.export(tmp_path / "mag")
    assert (directory / "raw.csv").exists()
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert meta["unit"] == "nT" and meta["stage"] == "anomaly" and meta["crs"] == "EPSG:21456"
    status = pd.read_csv(directory / "status.csv", index_col=0)
    assert list(status["status"]) == ["observed", "observed", "invalid", "missing"]
    active = pd.read_csv(directory / "active.csv", index_col=0)
    assert active.iloc[:, 0].all()
    with pytest.raises(FileExistsError):
        data.export(directory)


def test_dataset_invariants():
    schema = _schema()
    data = read_magnetic(_frame(), schema=schema)
    copied = data.copy(stage="rtp")
    assert copied.stage == "rtp" and data.stage == "anomaly"
    bad = data.values.copy()
    bad.index = ["x", "y", "z", "w"]
    with pytest.raises(DataValidationError):
        data.copy(values=bad)
    with pytest.raises(DataValidationError):
        data.copy(active=pd.Series([1, 0, 1, 0], index=data.values.index))
    with pytest.raises(DataValidationError):
        data.copy(stage="bogus")


def test_grid_export_raster_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    spec = GridSpec(width=3, height=2, transform=(10.0, 0.0, 100.0, 0.0, -10.0, 200.0), crs="EPSG:21456")
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    support = np.array([[True, True, False], [True, False, True]])
    grid = MagneticGrid(grid=spec, values=values, support=support, stage="rtp",
                        unit="nT", upward=50.0)
    directory = grid.export(tmp_path / "grid")
    with rasterio.open(directory / "values.tif") as source:
        band = source.read(1)
        assert source.transform.a == 10.0 and source.transform.f == 200.0
        assert np.isnan(band[0, 2]) and np.isnan(band[1, 1])
        assert band[0, 0] == 0.0 and band[1, 2] == 5.0
    with rasterio.open(directory / "support.tif") as source:
        assert source.read(1).tolist() == support.astype(np.uint8).tolist()
        assert source.nodata == 0
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert meta["stage"] == "rtp" and meta["unit"] == "nT" and meta["upward"] == 50.0
    assert "not statistical uncertainty" in meta["support_meaning"]


def test_grid_invariants():
    spec = GridSpec(width=2, height=2, transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0), crs="EPSG:21456")
    support = np.ones((2, 2), dtype=bool)
    with pytest.raises(DataValidationError):
        MagneticGrid(grid=spec, values=np.ones((2, 3)), support=support, stage="anomaly",
                     unit="nT", upward=None)
    with pytest.raises(DataValidationError):
        MagneticGrid(grid=spec, values=np.ones((2, 2)), support=np.ones((2, 2)), stage="anomaly",
                     unit="nT", upward=None)
    with pytest.raises(DataValidationError):
        MagneticGrid(grid=spec, values=np.full((2, 2), np.inf), support=support, stage="anomaly",
                     unit="nT", upward=None)
    with pytest.raises(DataValidationError):
        MagneticGrid(grid=spec, values=np.ones((2, 2)), support=support, stage="anomaly",
                     unit="gamma", upward=None)
    with pytest.raises(DataValidationError):
        MagneticGrid(grid=spec, values=np.ones((2, 2)), support=support, stage="bogus",
                     unit="nT", upward=None)
    with pytest.raises(DataValidationError):
        MagneticGrid(grid=spec, values=np.ones((2, 2)), support=support, stage="anomaly",
                     unit="nT", upward=float("inf"))
    grid = MagneticGrid(grid=spec, values=np.ones((2, 2)), support=support, stage="anomaly",
                        unit="nT", upward=None)
    assert np.isnan(grid.copy(values=np.full((2, 2), np.nan)).values).all()


def test_backend_reporting():
    report = GeoToolbox().backends()
    for name in ("harmonica", "verde", "ppigrf"):
        assert report[name]["available"] is True
        assert report[name]["distribution"] == name