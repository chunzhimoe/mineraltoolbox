import time
from pathlib import Path

import pandas as pd
import pytest

tk = pytest.importorskip("tkinter")

from geo_toolbox.gui import MagneticWorkbench
from geo_toolbox.geochem_gui import GeochemWorkbench
from geo_toolbox.gui_guidance import suggest_columns, raw_stage


class _Msg:
    def __init__(self, answer=True):
        self.calls = []
        self.answer = answer

    def showerror(self, t, m):
        self.calls.append(("error", t, str(m)))

    def showwarning(self, t, m):
        self.calls.append(("warning", t, str(m)))

    def showinfo(self, t, m):
        self.calls.append(("info", t, str(m)))

    def askokcancel(self, t, m):
        self.calls.append(("ask", t, str(m)))
        return self.answer


def _root(cls):
    root = tk.Tk()
    root.withdraw()
    wb = cls(root)
    wb.messagebox = _Msg()
    return root, wb


def _pump(root, wb, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if not wb._busy and wb._queue.empty():
            return True
        time.sleep(0.02)
    return False


def _csv(tmp_path):
    path = tmp_path / "mag.csv"
    pd.DataFrame({"点号": ["p1", "p2", "p3"], "X": ["0", "10", "20"],
                  "Y": ["0", "10", "20"], "△T": ["1.0", "-2.0", "3.0"]}).to_csv(
        path, index=False)
    return path


def test_initial_next_action():
    root, wb = _root(MagneticWorkbench)
    try:
        wb._update_step_label()
        assert "第1步" in wb.step_label.cget("text")
        wb.notebook.select(wb.tab_qc)
        wb._update_step_label()
    finally:
        wb._on_close()


def test_browse_auto_preview(tmp_path, monkeypatch):
    root, wb = _root(MagneticWorkbench)
    try:
        monkeypatch.setattr(wb.filedialog, "askopenfilename",
                            lambda **k: str(_csv(tmp_path)))
        wb._browse()
        assert _pump(root, wb)
        assert wb.preview is not None
        assert wb.map_vars["sample_id"].get() == "点号"
        assert wb.map_vars["value"].get() == "△T"
    finally:
        wb._on_close()


def test_ambiguous_not_suggested():
    assert "x" not in suggest_columns(["X1", "X2"], ["x"])
    assert suggest_columns(["点号", "X", "Y"], ["sample_id", "x", "y"]) == \
        {"sample_id": "点号", "x": "X", "y": "Y"}


def test_no_stage_unit_guess(tmp_path):
    root, wb = _root(MagneticWorkbench)
    try:
        wb.path_var.set(str(_csv(tmp_path)))
        wb._do_preview()
        assert _pump(root, wb)
        assert wb.stage_var.get() == "" and wb.unit_var.get() == "" \
            and wb.crs_var.get() == ""
    finally:
        wb._on_close()


def test_source_change_clears_mapping(tmp_path):
    root, wb = _root(MagneticWorkbench)
    try:
        wb.path_var.set(str(_csv(tmp_path)))
        wb._do_preview()
        assert _pump(root, wb)
        assert wb.map_vars["x"].get() == "X"
        wb.path_var.set(str(tmp_path / "other.csv"))
        wb._source_changed()
        root.update()
        assert wb.map_vars["x"].get() == "" and wb.stage_var.get() == ""
    finally:
        wb._on_close()


def test_missing_fields_prompt_before_submit(tmp_path):
    root, wb = _root(MagneticWorkbench)
    try:
        wb.path_var.set(str(_csv(tmp_path)))
        wb._do_preview()
        assert _pump(root, wb)
        wb.stage_var.set("")
        wb._do_import()
        assert any(call[0] == "warning" and "缺少" in call[1]
                   for call in wb.messagebox.calls)
        assert wb.dataset is None
    finally:
        wb._on_close()


def test_qc_readable_summary_unknown_crs(tmp_path):
    root, wb = _root(MagneticWorkbench)
    try:
        wb.path_var.set(str(_csv(tmp_path)))
        wb._do_preview()
        assert _pump(root, wb)
        wb.stage_var.set("anomaly")
        wb.unit_var.set("nT")
        wb._do_import()
        assert _pump(root, wb)
        assert wb.dataset is not None
        text = wb.report_text.get("1.0", "end")
        assert "行" in text and "未被修改" in text and "未知" in text
        assert "invalid_coordinate_ids" not in text
    finally:
        wb._on_close()


def test_raw_stage_fallback():
    assert raw_stage("anomaly") == "anomaly"
    assert raw_stage("磁异常 anomaly") == "anomaly"


def test_busy_disables_nav(tmp_path):
    root, wb = _root(MagneticWorkbench)
    try:
        assert wb._submit("x", lambda: __import__("time").sleep(0.4) or 1,
                          lambda r: None)
        assert str(wb.next_btn.cget("state")) == "disabled"
        assert str(wb.prev_btn.cget("state")) == "disabled"
        _pump(root, wb)
    finally:
        wb._on_close()


def test_grid_task_panels(tmp_path):
    root, wb = _root(MagneticWorkbench)
    try:
        wb.grid_task.set("裁掉网格边缘")
        wb._grid_task_changed()
        assert str(wb._crop_frame.winfo_manager()) == "pack"
        assert wb._interp_frame.winfo_manager() == ""
        wb.grid_task.set("把测点生成网格图")
        wb._grid_task_changed()
        assert str(wb._interp_frame.winfo_manager()) == "pack"
        assert wb._trans_frame.winfo_manager() == ""
    finally:
        wb._on_close()


def test_geochem_suggestions_and_tasks(tmp_path):
    root, wb = _root(GeochemWorkbench)
    try:
        path = tmp_path / "geo.csv"
        pd.DataFrame({"点号": ["p1"], "X": ["0"], "Y": ["0"], "Au": ["1.0"],
                      "As": ["2.0"]}).to_csv(path, index=False)
        wb.path_var.set(str(path))
        wb._do_preview()
        assert _pump(root, wb)
        assert wb.map_vars["sample_id"].get() == "点号"
        assert wb.map_vars["x"].get() == "X"
        wb.spatial_task.set("验证预测误差（留出验证）")
        wb._spatial_task_changed()
        assert str(wb._validate_frame.winfo_manager()) == "pack"
        assert wb._variogram_frame.winfo_manager() == ""
    finally:
        wb._on_close()


def test_shp_cancel_no_crash(tmp_path, monkeypatch):
    root, wb = _root(MagneticWorkbench)
    try:
        monkeypatch.setattr(wb.filedialog, "askopenfilename", lambda **kw: "")
        wb._choose_reference_overlay()
        assert not wb._busy
        assert not any(call[0] == "error" for call in wb.messagebox.calls)
    finally:
        wb._on_close()


def _fill_geo_elements(wb, elements=("Au", "As", "Sb")):
    for col, row in wb.element_rows.items():
        element = col in elements
        row["use"].set(element)
        if element:
            row["unit"].set("ppm")


def test_step_navigation_magnetic(tmp_path):
    root, wb = _root(MagneticWorkbench)
    try:
        wb.path_var.set(str(_csv(tmp_path)))
        wb._do_preview()
        assert _pump(root, wb)
        wb.next_btn.invoke()
        assert _pump(root, wb)
        assert wb.notebook.index("current") == wb.notebook.index(wb.tab_confirm)
        wb.unit_var.set("nT")
        wb.stage_var.set("anomaly")
        wb.next_btn.invoke()
        assert _pump(root, wb)
        assert wb.dataset is not None
        assert wb.notebook.index("current") == wb.notebook.index(wb.tab_qc)
        wb.next_btn.invoke()
        assert wb.notebook.index("current") == wb.notebook.index(wb.tab_log)
    finally:
        wb._on_close()


def test_step_navigation_geochem(tmp_path):
    root, wb = _root(GeochemWorkbench)
    try:
        path = tmp_path / "geo.csv"
        pd.DataFrame({"点号": ["p1", "p2", "p3"], "X": ["0", "10", "20"],
                      "Y": ["0", "10", "20"], "Au": ["1.0", "<0.5", "2.0"],
                      "As": ["10", "20", "30"], "Sb": ["2", "4", "6"]}).to_csv(
            path, index=False)
        wb.path_var.set(str(path))
        wb._do_preview()
        assert _pump(root, wb)
        _fill_geo_elements(wb)
        wb.next_btn.invoke()
        assert _pump(root, wb)
        assert wb.notebook.index("current") == wb.notebook.index(wb.tab_confirm)
        wb.next_btn.invoke()
        assert _pump(root, wb)
        assert wb.dataset is not None
        assert wb.notebook.index("current") == wb.notebook.index(wb.tab_qc)
    finally:
        wb._on_close()


def test_step1_has_no_mapping_controls(tmp_path):
    root, wb = _root(MagneticWorkbench)
    try:
        assert wb.tab_confirm is not None
        combo = getattr(wb, "map_combos", None)
        if combo:
            path = str(wb.map_combos["x"])
            assert not path.startswith(str(wb.tab_import) + "."), \
                "mapping combos must not live on step-1"
        assert wb.map_vars["x"] is not None
    finally:
        wb._on_close()


def test_combobox_human_labels(tmp_path):
    root, wb = _root(MagneticWorkbench)
    try:
        values = list(wb.stage_combo.cget("values"))
        assert values, "stage combo should have display values"
        assert raw_stage(values[-1]) in ("anomaly", "rtp", "raw_total_field",
                                        "corrected_total_field")
        wb.stage_var.set("anomaly")
        assert raw_stage(wb.stage_var.get()) == "anomaly"
    finally:
        wb._on_close()


def test_coordinate_outside_area_review_not_block():
    import numpy as np
    from geo_toolbox.coordinate_check import check_coordinates
    points = np.array([[999999.0, 4000000.0], [999998.0, 4000100.0],
                       [999997.0, 4000200.0]])
    before = points.copy()
    report = check_coordinates(points, crs="EPSG:32648")
    assert report["status"] == "review"
    assert report["authority"] == ("EPSG", "32648")
    assert report["invalid_rows"] == 0
    assert report.get("outside_area_rows", 0) > 0
    np.testing.assert_array_equal(points, before)


def test_overlay_missing_companions_and_prj(tmp_path):
    shapefile = pytest.importorskip("shapefile")
    from geo_toolbox.errors import DataValidationError
    from geo_toolbox.reference_overlay import read_overlay
    shp = tmp_path / "bare"
    with shapefile.Writer(str(shp), shapeType=1) as writer:
        writer.field("id", "C", size=10)
        writer.point(1.0, 2.0)
        writer.record("a")
    for suffix in (".dbf", ".shx"):
        shp.with_suffix(suffix).unlink()
    with pytest.raises(DataValidationError):
        read_overlay(str(shp.with_suffix(".shp")))


def test_overlay_malformed_prj_json(tmp_path):
    shapefile = pytest.importorskip("shapefile")
    from geo_toolbox.errors import DataValidationError
    from geo_toolbox.reference_overlay import read_overlay
    shp = tmp_path / "badprj"
    with shapefile.Writer(str(shp), shapeType=1) as writer:
        writer.field("id", "C", size=10)
        writer.point(1.0, 2.0)
        writer.record("a")
    target = shp.with_suffix(".prj")
    target.write_text("NOT-A-PROJ", encoding="utf-8")
    with pytest.raises(DataValidationError):
        read_overlay(str(shp.with_suffix(".shp")))
    from pyproj import CRS
    target.write_text(CRS.from_epsg(32648).to_wkt(), encoding="utf-8")
    bad_json = tmp_path / "ref.json"
    bad_json.write_text("{not valid", encoding="utf-8")
    with pytest.raises((DataValidationError, ValueError, OSError)):
        read_overlay(str(shp.with_suffix(".shp")), reference=str(bad_json))


def test_shp_overlay_synthetic(tmp_path, monkeypatch):
    shapefile = pytest.importorskip("shapefile")
    shp = tmp_path / "points"
    with shapefile.Writer(str(shp), shapeType=1) as writer:
        writer.field("id", "C", size=10)
        writer.point(5.0, 5.0)
        writer.record("a")
        writer.point(15.0, 15.0)
        writer.record("b")
    root, wb = _root(MagneticWorkbench)
    try:
        answers = iter([str(shp.with_suffix(".shp")), ""])
        monkeypatch.setattr(wb.filedialog, "askopenfilename",
                            lambda **kw: next(answers, ""))
        wb._choose_reference_overlay()
        assert _pump(root, wb, timeout=15.0), "overlay worker did not finish"
        assert wb.reference_overlay is not None or wb._overlay_status is not None
        assert not any(call[0] == "error" for call in wb.messagebox.calls)
    finally:
        wb._on_close()
