import time

import pandas as pd
import pytest

tk = pytest.importorskip("tkinter")

from geo_toolbox.geochem_gui import GeochemWorkbench
from geo_toolbox.gui import MagneticWorkbench


class _Msg:
    def __init__(self):
        self.calls = []

    def showinfo(self, title, message):
        self.calls.append(("info", title, str(message)))

    def showwarning(self, title, message):
        self.calls.append(("warn", title, str(message)))

    def showerror(self, title, message):
        self.calls.append(("error", title, str(message)))

    def askokcancel(self, title, message):
        self.calls.append(("ask", title, str(message)))
        return True


def _workbench(cls):
    from tk_test_support import create_tk_root
    root = create_tk_root()
    root.withdraw()
    workbench = cls(root)
    workbench.messagebox = _Msg()
    return root, workbench


def _pump(root, workbench, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if not workbench._busy and workbench._queue.empty():
            return True
        time.sleep(0.02)
    return False


def _bounds(workbench):
    return [workbench.interp_vars[name].get()
            for name in ("xmin", "ymin", "xmax", "ymax")]


def _states(workbench):
    return [str(workbench.interp_entries[name].cget("state"))
            for name in ("xmin", "ymin", "xmax", "ymax")]


def _magnetic_csv(tmp_path):
    path = tmp_path / "points.csv"
    pd.DataFrame({
        "id": ["01-1", "01-2", "01-3"],
        "X": ["1", "109", "1"],
        "Y": ["2", "2", "111"],
        "value": ["1.0", "2.0", "3.0"],
    }).to_csv(path, index=False)
    return path


def _import_magnetic(root, workbench, tmp_path):
    workbench.path_var.set(str(_magnetic_csv(tmp_path)))
    workbench._do_preview()
    assert _pump(root, workbench)
    for key, column in (("sample_id", "id"), ("x", "X"), ("y", "Y"),
                        ("value", "value")):
        workbench.map_combos[key].set(column)
        workbench.map_combos[key].event_generate("<<ComboboxSelected>>")
    workbench.unit_combo.set("nT")
    workbench.unit_combo.event_generate("<<ComboboxSelected>>")
    workbench.stage_combo.set("anomaly")
    workbench.stage_combo.event_generate("<<ComboboxSelected>>")
    workbench._do_import()
    assert _pump(root, workbench)
    assert workbench.dataset is not None


def _geochem_csv(tmp_path):
    path = tmp_path / "points.csv"
    pd.DataFrame({
        "id": ["001", "002", "003"],
        "X": ["1", "109", "1"],
        "Y": ["2", "2", "111"],
        "Au": ["1.2", "3.4", "0.8"],
    }).to_csv(path, index=False)
    return path


def _import_geochem(root, workbench, tmp_path):
    workbench.path_var.set(str(_geochem_csv(tmp_path)))
    workbench._do_preview()
    assert _pump(root, workbench)
    workbench.map_combos["sample_id"].set("id")
    workbench.map_combos["sample_id"].event_generate("<<ComboboxSelected>>")
    workbench.map_combos["x"].set("X")
    workbench.map_combos["x"].event_generate("<<ComboboxSelected>>")
    workbench.map_combos["y"].set("Y")
    workbench.map_combos["y"].event_generate("<<ComboboxSelected>>")
    row = workbench.element_rows["Au"]
    row["use"].set(True)
    row["unit"].set("ppm")
    workbench._do_import()
    assert _pump(root, workbench)
    assert workbench.dataset is not None


def test_magnetic_auto_bounds_populates_after_import(tmp_path):
    root, workbench = _workbench(MagneticWorkbench)
    try:
        _import_magnetic(root, workbench, tmp_path)
        assert _bounds(workbench) == ["1", "2", "109", "111"]
        assert _states(workbench) == ["readonly"] * 4
        workbench.interp_vars["resolution"].set("50")
        assert _bounds(workbench) == ["0", "0", "150", "150"]
    finally:
        workbench._on_close()


def test_magnetic_manual_mode_and_invalid_resolution(tmp_path):
    root, workbench = _workbench(MagneticWorkbench)
    try:
        _import_magnetic(root, workbench, tmp_path)
        workbench.auto_bounds_var.set(False)
        workbench._auto_bounds_changed()
        assert _states(workbench) == ["normal"] * 4
        workbench.interp_vars["xmin"].set("5")
        workbench.interp_vars["resolution"].set("50")
        assert workbench.interp_vars["xmin"].get() == "5"
        workbench.auto_bounds_var.set(True)
        workbench._auto_bounds_changed()
        workbench.interp_vars["resolution"].set("bad")
        assert _bounds(workbench) == ["1", "2", "109", "111"]
        workbench.interp_vars["resolution"].set("")
        assert _bounds(workbench) == ["1", "2", "109", "111"]
        dialogs = [call for call in workbench.messagebox.calls
                   if call[0] in ("error", "warn")]
        assert not dialogs
    finally:
        workbench._on_close()


def test_magnetic_no_dataset_clears_bounds(tmp_path):
    root, workbench = _workbench(MagneticWorkbench)
    try:
        _import_magnetic(root, workbench, tmp_path)
        workbench._invalidate()
        assert _bounds(workbench) == ["", "", "", ""]
        workbench.interp_vars["resolution"].set("50")
        assert _bounds(workbench) == ["", "", "", ""]
    finally:
        workbench._on_close()


def test_magnetic_interpolate_passes_auto_bounds(tmp_path):
    root, workbench = _workbench(MagneticWorkbench)
    try:
        _import_magnetic(root, workbench, tmp_path)
        captured = []
        workbench._run_operation = lambda op, params, data, cb: captured.append(params)
        workbench.interp_method.set("verde_spline")
        workbench._do_interpolate()
        assert captured and captured[0]["auto_bounds"] is True
        workbench.auto_bounds_var.set(False)
        captured.clear()
        workbench._do_interpolate()
        assert captured[0]["auto_bounds"] is False
    finally:
        workbench._on_close()


def test_geochem_auto_bounds_populates_after_import(tmp_path):
    root, workbench = _workbench(GeochemWorkbench)
    try:
        _import_geochem(root, workbench, tmp_path)
        assert _bounds(workbench) == ["1", "2", "109", "111"]
        assert _states(workbench) == ["readonly"] * 4
        workbench.interp_vars["resolution"].set("50")
        assert _bounds(workbench) == ["0", "0", "150", "150"]
    finally:
        workbench._on_close()


def test_geochem_manual_mode_no_dataset_and_auto_bounds_param(tmp_path):
    root, workbench = _workbench(GeochemWorkbench)
    try:
        _import_geochem(root, workbench, tmp_path)
        workbench.auto_bounds_var.set(False)
        workbench._auto_bounds_changed()
        assert _states(workbench) == ["normal"] * 4
        workbench.interp_vars["xmin"].set("5")
        workbench.interp_vars["resolution"].set("50")
        assert workbench.interp_vars["xmin"].get() == "5"
        captured = []
        workbench._run_operation = lambda op, params, data, cb: captured.append(params)
        workbench.kriging_variogram.set("spherical")
        workbench._do_interpolate()
        assert captured and captured[0]["auto_bounds"] is False
        workbench.auto_bounds_var.set(True)
        workbench._invalidate()
        assert _bounds(workbench) == ["", "", "", ""]
    finally:
        workbench._on_close()
