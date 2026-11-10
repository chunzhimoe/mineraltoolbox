import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import pandas as pd
import pytest

tk = pytest.importorskip("tkinter")

from geo_toolbox.gui import MagneticWorkbench


class _Msg:
    def __init__(self, answer=True):
        self.calls = []
        self.answer = answer

    def showerror(self, title, message):
        self.calls.append(("error", title, str(message)))

    def showwarning(self, title, message):
        self.calls.append(("warning", title, str(message)))

    def showinfo(self, title, message):
        self.calls.append(("info", title, str(message)))

    def askokcancel(self, title, message):
        self.calls.append(("ask", title, str(message)))
        return self.answer


def _workbench():
    root = None
    for _ in range(5):
        try:
            root = tk.Tk()
            break
        except tk.TclError:
            time.sleep(0.3)
    assert root is not None
    root.withdraw()
    workbench = MagneticWorkbench(root)
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


def _csv(tmp_path, name="synthetic.csv"):
    path = tmp_path / name
    pd.DataFrame({
        "点号": ["01-1", "01-2", "01-3", "01-4"],
        "X": ["14000.0", "14020.0", "14040.0", "14060.0"],
        "Y": ["80000.0", "80020.0", "80040.0", "80060.0"],
        "△T": ["-3.5", "0.0", "bad", ""],
    }).to_csv(path, index=False)
    return path


def _xlsx(tmp_path):
    path = tmp_path / "two_sheets.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"点号": ["A"], "X": ["1"], "Y": ["2"], "△T": ["3"]}).to_excel(
            writer, sheet_name="第一表", index=False)
        pd.DataFrame({"点号": ["B1", "B2"], "X": ["10", "20"], "Y": ["30", "40"],
                      "△T": ["1.5", "2.5"]}).to_excel(writer, sheet_name="第二表",
                                                     index=False)
    return path


def _map_required(workbench):
    for key, column in (("sample_id", "点号"), ("x", "X"), ("y", "Y"), ("value", "△T")):
        workbench.map_combos[key].set(column)
        workbench.map_combos[key].event_generate("<<ComboboxSelected>>")
    workbench.unit_combo.set("nT")
    workbench.unit_combo.event_generate("<<ComboboxSelected>>")
    workbench.stage_combo.set("anomaly")
    workbench.stage_combo.event_generate("<<ComboboxSelected>>")


def _imported(root, workbench, tmp_path):
    workbench.path_var.set(str(_csv(tmp_path)))
    workbench._do_preview()
    assert _pump(root, workbench)
    _map_required(workbench)
    workbench._do_import()
    assert _pump(root, workbench)
    assert workbench.dataset is not None


def test_import_probe_clean():
    probe = Path(__file__).with_name("_gui_import_probe.py")
    output = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True,
                            timeout=60)
    assert output.returncode == 0, output.stderr
    assert output.stdout.strip() == "loaded: none"


def test_pyproject_metadata():
    root_dir = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root_dir / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert all(isinstance(value, list) for value in extras.values())
    assert extras["gui"] == ["matplotlib>=3.11,<3.12"]
    assert extras["parquet"] == ["pyarrow>=26,<27"]
    assert extras["test"] == ["pytest>=9.2,<10"]
    assert data["project"]["gui-scripts"] == {"geo-toolbox-gui": "geo_toolbox.gui:main"}


def test_window_constructs_and_no_state():
    root, workbench = _workbench()
    try:
        assert workbench.dataset is None and workbench.current_grid is None
        assert str(workbench.import_btn.cget("state")) == "disabled"
        assert str(workbench.check_btn.cget("state")) == "disabled"
    finally:
        workbench._on_close()


def test_combobox_events_preserve_preview_then_import(tmp_path):
    root, workbench = _workbench()
    try:
        workbench.path_var.set(str(_csv(tmp_path)))
        workbench._do_preview()
        assert _pump(root, workbench)
        assert workbench.preview is not None
        workbench.map_combos["sample_id"].set("点号")
        workbench.map_combos["sample_id"].event_generate("<<ComboboxSelected>>")
        root.update()
        assert workbench.preview is not None
        _map_required(workbench)
        workbench._do_import()
        assert _pump(root, workbench)
        assert workbench.dataset is not None
        assert len(workbench.dataset.values) == 4
        assert workbench.last_report is not None
        assert "observed" in workbench.report_text.get("1.0", "end")
    finally:
        workbench._on_close()


def test_two_sheet_switching(tmp_path):
    root, workbench = _workbench()
    try:
        workbench.path_var.set(str(_xlsx(tmp_path)))
        workbench._do_preview()
        assert _pump(root, workbench)
        assert workbench.sheet_var.get() == "第一表"
        workbench.sheet_var.set("第二表")
        workbench._do_preview()
        assert _pump(root, workbench)
        first = workbench.tree.get_children()[0]
        assert workbench.tree.item(first)["values"][0] == "B1"
    finally:
        workbench._on_close()


def test_source_fields_invalidate(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        workbench.header_var.set("2")
        root.update()
        assert workbench.preview is None and workbench.dataset is None
    finally:
        workbench._on_close()


def test_mapping_change_keeps_preview_clears_dataset(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        preview = workbench.preview
        workbench.map_combos["upward"].set("X")
        workbench.map_combos["upward"].event_generate("<<ComboboxSelected>>")
        root.update()
        assert workbench.preview is preview
        assert workbench.dataset is None
        assert str(workbench.check_btn.cget("state")) == "disabled"
    finally:
        workbench._on_close()


def test_busy_disables_inputs_and_restores(tmp_path):
    root, workbench = _workbench()
    try:
        workbench.path_var.set(str(_csv(tmp_path)))
        assert workbench._submit("first",
                                 lambda: __import__("time").sleep(0.4) or 1,
                                 lambda result: None)
        assert str(workbench.map_combos["x"].cget("state")) == "disabled"
        assert str(workbench.sheet_combo.cget("state")) == "disabled"
        assert workbench._submit("second", lambda: 2, lambda r: None) is False
        _pump(root, workbench)
        root.update()
        assert str(workbench.map_combos["x"].cget("state")) == "readonly"
    finally:
        workbench._on_close()


def test_failed_reimport_retains_prior_result(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        workbench.unit_var.set("")
        workbench._do_import()
        assert _pump(root, workbench)
        assert any(call[0] == "error" for call in workbench.messagebox.calls)
        assert workbench.dataset is None
        assert workbench.last_result is None
    finally:
        workbench._on_close()


def test_warning_cancel_skips_execute(tmp_path, monkeypatch):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        workbench.messagebox = _Msg(answer=False)
        calls = []
        tasks = __import__("geo_toolbox.gui_tasks", fromlist=["preflight", "execute"])
        original_execute = tasks.execute
        monkeypatch.setattr(tasks, "execute",
                            lambda data, **kwargs: calls.append(kwargs)
                            or original_execute(data, **kwargs))
        workbench._do_check()
        assert _pump(root, workbench)
        assert not calls
        assert any(call[0] == "ask" for call in workbench.messagebox.calls)
    finally:
        workbench._on_close()


def test_max_gain_not_passed_for_derivative(tmp_path, monkeypatch):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        workbench.current_grid = object()
        workbench._refresh_actions()
        captured = []
        tasks = __import__("geo_toolbox.gui_tasks", fromlist=["preflight"])
        original = tasks.preflight
        monkeypatch.setattr(tasks, "preflight",
                            lambda data, **kw: captured.append(kw["parameters"])
                            or {"ok": False, "errors": ["stop"], "warnings": []})
        workbench.transform_method.set("derivative_easting")
        workbench._transform_selected()
        workbench.transform_vars["max_gain"].set("99")
        workbench.transform_vars["padding"].set("4")
        workbench.transform_vars["trim"].set("2")
        workbench._do_transform()
        assert _pump(root, workbench)
        assert captured and "max_gain" not in captured[0]
        workbench.transform_method.set("reduction_to_pole")
        workbench._transform_selected()
        workbench.transform_vars["max_gain"].set("99")
        workbench.transform_vars["inclination"].set("60")
        workbench.transform_vars["declination"].set("-3")
        workbench.induced_var.set(True)
        workbench._transform_selected()
        workbench._do_transform()
        assert _pump(root, workbench)
        assert "max_gain" in captured[-1]
        assert "magnetization_inclination" not in captured[-1]
        assert captured[-1]["assume_induced"] is True
    finally:
        workbench._on_close()


def test_failed_operation_retains_last_result(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        prior = workbench.last_result
        workbench.window_var.set("4")
        workbench.threshold_var.set("100")
        workbench._do_spikes()
        assert _pump(root, workbench)
        assert workbench.last_result is prior
    finally:
        workbench._on_close()


def test_export_refusal_existing_dir(tmp_path):
    root, workbench = _workbench()
    try:
        existing = tmp_path / "out"
        existing.mkdir()
        workbench.last_result = object()
        workbench.last_report = {}
        workbench.export_var.set(str(existing))
        workbench._do_export()
        assert _pump(root, workbench)
        assert any(call[0] == "error" for call in workbench.messagebox.calls)
        assert workbench.last_result is not None
    finally:
        workbench._on_close()


def test_callback_error_routes_to_fail(tmp_path):
    root, workbench = _workbench()
    try:
        log_dir = Path(tempfile.gettempdir()) / "geo_toolbox_gui"
        before = set(log_dir.glob("error_*.txt")) if log_dir.is_dir() else set()
        workbench._on_callback_error(ValueError, ValueError("boom"), None)
        root.update()
        assert any(call[0] == "error" for call in workbench.messagebox.calls)
        assert set(log_dir.glob("error_*.txt")) - before
    finally:
        workbench._on_close()


def test_error_log_autosaved(tmp_path):
    root, workbench = _workbench()
    try:
        log_dir = Path(tempfile.gettempdir()) / "geo_toolbox_gui"
        before = set(log_dir.glob("error_*.txt")) if log_dir.is_dir() else set()
        workbench.path_var.set(str(tmp_path / "missing.csv"))
        workbench._do_preview()
        assert _pump(root, workbench)
        new = set(log_dir.glob("error_*.txt")) - before
        assert new and new.pop().read_text(encoding="utf-8")
    finally:
        workbench._on_close()
