import json

import pandas as pd
import pytest

from geo_toolbox import gui_tasks, geochem_gui_tasks
from geo_toolbox.errors import DataValidationError


@pytest.mark.parametrize("tasks", [gui_tasks, geochem_gui_tasks])
def test_explicit_skip_ids_preserves_first_coordinates_and_source(tmp_path, tasks):
    frame = pd.DataFrame({"id": ["001", "", "001", "002", "   ", None, "003"],
                          "x": [1, 2, 99, 1, 5, 6, 7], "y": [1, 2, 99, 1, 5, 6, 7],
                          "value": [10, 20, 999, 40, 50, 60, 70]})
    original = frame.copy(deep=True)
    preview = {"frame": frame, "source": "synthetic", "header": 1, "source_sha256": "test"}
    mapping = {"sample_id": "id", "x": "x", "y": "y", "crs": "", "id_policy": "skip"}
    if tasks is gui_tasks:
        mapping.update(value="value", unit="nT", stage="anomaly")
    else:
        mapping.update(elements=["value"], units={"value": "ppm"})
    data = tasks.import_source(preview, mapping=mapping)
    pd.testing.assert_frame_equal(frame, original)
    assert data.values.index.tolist() == ["001", "002", "003"]
    assert data.raw["value"].tolist() == [10, 40, 70]
    assert len(data.coordinates.drop_duplicates()) == 2
    audit = next(entry["parameters"] for entry in data.history if entry["operation"] == "import_id_filter")
    assert audit["input_rows"] == 7 and audit["retained_rows"] == 3
    assert audit["blank_rows"] == 3 and audit["duplicate_rows"] == 1
    assert [row["source_row"] for row in audit["skipped"]] == [4, 5, 7, 8]
    assert [row["reason"] for row in audit["skipped"]] == ["blank_id", "duplicate_id", "blank_id", "blank_id"]
    target = tasks.export_result(data, directory=tmp_path / "export", report={})
    skipped = pd.read_csv(target / "skipped_records.csv", keep_default_na=False, dtype=str)
    assert skipped["source_row"].tolist() == ["4", "5", "7", "8"]
    assert skipped["original:id"].tolist() == ["", "001", "   ", ""]
    assert json.loads((target / "import_id_audit.json").read_text(encoding="utf-8"))["retained_rows"] == 3


@pytest.mark.parametrize("tasks", [gui_tasks, geochem_gui_tasks])
def test_id_skip_does_not_weaken_default_or_generate_policy(tmp_path, tasks):
    preview = {"frame": pd.DataFrame({"id": ["", ""], "x": [1, 2], "y": [3, 4], "v": [5, 6]}),
               "source": "synthetic", "header": 0}
    mapping = {"sample_id": "id", "x": "x", "y": "y", "crs": ""}
    if tasks is gui_tasks:
        mapping.update(value="v", unit="nT", stage="anomaly")
    else:
        mapping.update(elements=["v"], units={"v": "ppm"})
    with pytest.raises(DataValidationError):
        tasks.import_source(preview, mapping=mapping)
    mapping["id_policy"] = "skip"
    with pytest.raises(DataValidationError, match="没有可导入"):
        tasks.import_source(preview, mapping=mapping)
    mapping["generated_ids"] = True
    assert len(tasks.import_source(preview, mapping=mapping).values) == 2
