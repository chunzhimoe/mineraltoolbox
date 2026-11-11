import time
from pathlib import Path

import pandas as pd
import pytest

tk = pytest.importorskip("tkinter")

from geo_toolbox.geochem_gui import GeochemWorkbench


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
    from tk_test_support import create_tk_root
    root = create_tk_root()
    root.withdraw()
    workbench = GeochemWorkbench(root)
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


def _csv(tmp_path, name="geochem.csv"):
    path = tmp_path / name
    pd.DataFrame({
        "点号": ["001", "002", "003", "004", "005", "006"],
        "X": ["14000", "14020", "14040", "14060", "14080", "14100"],
        "Y": ["80000", "80020", "80040", "80060", "80080", "80100"],
        "Au": ["<0.5", "1.2", "3.4", "0.8", "2.1", ""],
        "As": ["10.5", "20.1", "33.0", "15.2", "8.8", "12.4"],
        "Sb": ["2.1", "3.3", "5.5", "1.9", "4.2", "2.8"],
    }).to_csv(path, index=False)
    return path


def _configure(workbench, elements=("Au", "As", "Sb"), units=None):
    workbench.map_combos["sample_id"].set("点号")
    workbench.map_combos["sample_id"].event_generate("<<ComboboxSelected>>")
    workbench.map_combos["x"].set("X")
    workbench.map_combos["x"].event_generate("<<ComboboxSelected>>")
    workbench.map_combos["y"].set("Y")
    workbench.map_combos["y"].event_generate("<<ComboboxSelected>>")
    units = units or {"Au": "ppb", "As": "ppm", "Sb": "ppm"}
    for column in elements:
        row = workbench.element_rows[column]
        row["use"].set(True)
        if column in units:
            row["unit"].set(units[column])


def _previewed(root, workbench, tmp_path):
    workbench.path_var.set(str(_csv(tmp_path)))
    workbench._do_preview()
    assert _pump(root, workbench)
    assert workbench.preview is not None


def _imported(root, workbench, tmp_path):
    _previewed(root, workbench, tmp_path)
    _configure(workbench)
    workbench._do_import()
    assert _pump(root, workbench)
    assert workbench.dataset is not None


def test_window_constructs():
    root, workbench = _workbench()
    try:
        assert workbench.original_dataset is None and workbench.features is None
        assert workbench.maps is None and str(workbench.import_btn.cget("state")) == "disabled"
    finally:
        workbench._on_close()


def test_element_config_and_import(tmp_path):
    root, workbench = _workbench()
    try:
        _previewed(root, workbench, tmp_path)
        assert set(workbench.element_rows) == {"点号", "X", "Y", "Au", "As", "Sb"}
        _configure(workbench)
        assert workbench.preview is not None
        workbench._do_import()
        assert _pump(root, workbench)
        data = workbench.dataset
        assert list(data.schema.elements) == ["Au", "As", "Sb"]
        assert data.schema.units == {"Au": "ppb", "As": "ppm", "Sb": "ppm"}
        assert data.values.index.tolist()[0] == "001"
        assert workbench.original_dataset is not None
        assert workbench.last_result is data
    finally:
        workbench._on_close()


def test_source_and_unit_change_invalidate(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        workbench.element_rows["Au"]["unit"].set("ppm")
        root.update()
        assert workbench.dataset is None and workbench.original_dataset is None
        assert workbench.preview is not None and workbench.element_rows
    finally:
        workbench._on_close()


def test_busy_blocks_duplicate(tmp_path):
    root, workbench = _workbench()
    try:
        workbench.path_var.set(str(_csv(tmp_path)))
        assert workbench._submit("first",
                                 lambda: __import__("time").sleep(0.4) or 1,
                                 lambda result: None)
        assert workbench._submit("second", lambda: 2, lambda r: None) is False
        assert str(workbench.preview_btn.cget("state")) == "disabled"
        _pump(root, workbench)
    finally:
        workbench._on_close()


def test_warning_cancel_no_execute(tmp_path, monkeypatch):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        workbench.messagebox = _Msg(answer=False)
        calls = []
        tasks = __import__("geo_toolbox.geochem_gui_tasks", fromlist=["execute"])
        original = tasks.execute
        monkeypatch.setattr(tasks, "execute",
                            lambda data, **kw: calls.append(kw) or original(data, **kw))
        workbench._do_check()
        assert _pump(root, workbench)
        assert not calls
        assert any(call[0] == "ask" for call in workbench.messagebox.calls)
    finally:
        workbench._on_close()


def test_missing_crs_spatial_preflight(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        workbench.variogram_model.set("spherical")
        workbench.variogram_estimator.set("matheron")
        workbench._do_variogram()
        assert _pump(root, workbench)
        assert any(call[0] == "error" for call in workbench.messagebox.calls)
    finally:
        workbench._on_close()


def test_transform_stores_features_preserving_concentration(tmp_path):
    root, workbench = _workbench()
    try:
        _previewed(root, workbench, tmp_path)
        _configure(workbench, elements=("As", "Sb"), units={"As": "ppm", "Sb": "ppm"})
        workbench._do_import()
        assert _pump(root, workbench)
        dataset = workbench.dataset
        assert dataset is not None
        workbench.logratio_method.set("clr")
        workbench._do_transform()
        assert _pump(root, workbench)
        assert workbench.features is not None
        assert workbench.dataset is dataset
        assert workbench.features.domain == "clr"
        assert workbench._results["特征"] is workbench.features
    finally:
        workbench._on_close()


def test_impute_invalidates_derived(tmp_path, monkeypatch):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        workbench.features = object()
        workbench.maps = object()
        workbench._results["特征"] = workbench.features
        tasks = __import__("geo_toolbox.geochem_gui_tasks",
                           fromlist=["preflight", "execute"])
        monkeypatch.setattr(tasks, "preflight",
                            lambda data, **kw: {"ok": True, "errors": [], "warnings": []})
        monkeypatch.setattr(tasks, "execute",
                            lambda data, **kw: {"kind": "concentration",
                                                "data": data, "report": {}})
        workbench.impute_method.set("multRepl")
        workbench._do_impute()
        assert _pump(root, workbench)
        assert workbench.features is None and workbench.maps is None
        assert "特征" not in workbench._results
    finally:
        workbench._on_close()


def test_validate_selector_passes_original(tmp_path, monkeypatch):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        captured = []
        tasks = __import__("geo_toolbox.geochem_gui_tasks", fromlist=["preflight"])
        monkeypatch.setattr(tasks, "preflight",
                            lambda data, **kw: captured.append(data)
                            or {"ok": False, "errors": ["stop"], "warnings": []})
        workbench.validate_source.set("原始导入")
        workbench.validate_variogram.set("spherical")
        workbench.train_text.insert("end", "001\n002\n003")
        workbench.test_text.insert("end", "004")
        workbench._do_validate()
        assert _pump(root, workbench)
        assert captured and captured[0] is workbench.original_dataset
    finally:
        workbench._on_close()


def test_import_has_no_errors_and_report(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        assert not any(call[0] == "error" for call in workbench.messagebox.calls)
        assert workbench.last_report
        assert "正常观测" in workbench.report_text.get("1.0", "end")
    finally:
        workbench._on_close()


def test_fixed_blank_rejected(tmp_path):
    root, workbench = _workbench()
    try:
        _previewed(root, workbench, tmp_path)
        _configure(workbench, elements=("As", "Sb"), units={"As": "ppm", "Sb": "ppm"})
        row = workbench.element_rows["As"]
        row["mode"].set("fixed")
        workbench._dl_mode_changed("As")
        workbench._do_import()
        assert any(call[0] == "error" for call in workbench.messagebox.calls)
        assert workbench.dataset is None
    finally:
        workbench._on_close()


def test_column_blank_rejected(tmp_path):
    root, workbench = _workbench()
    try:
        _previewed(root, workbench, tmp_path)
        _configure(workbench, elements=("As", "Sb"), units={"As": "ppm", "Sb": "ppm"})
        row = workbench.element_rows["As"]
        row["mode"].set("column")
        workbench._dl_mode_changed("As")
        workbench._do_import()
        assert any(call[0] == "error" for call in workbench.messagebox.calls)
        assert workbench.dataset is None
    finally:
        workbench._on_close()


def test_edit_limit_after_import_invalidates(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        preview = workbench.preview
        workbench.element_rows["As"]["mode"].set("fixed")
        workbench._dl_mode_changed("As")
        workbench.element_rows["As"]["fixed"].set("0.5")
        root.update()
        assert workbench.dataset is None
        assert workbench.preview is preview
    finally:
        workbench._on_close()


def test_mode_flip_preserves_separate_vars(tmp_path):
    root, workbench = _workbench()
    try:
        _previewed(root, workbench, tmp_path)
        row = workbench.element_rows["As"]
        row["mode"].set("fixed")
        workbench._dl_mode_changed("As")
        row["fixed"].set("0.5")
        row["mode"].set("column")
        workbench._dl_mode_changed("As")
        assert row["fixed"].get() == "0.5" and row["limit_column"].get() == ""
        row["limit_column"].set("Sb")
        row["mode"].set("fixed")
        workbench._dl_mode_changed("As")
        assert row["fixed"].get() == "0.5" and row["limit_column"].get() == "Sb"
    finally:
        workbench._on_close()


def test_export_missing_branch_rejected(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        workbench.export_kind.set("特征")
        workbench.export_var.set(str(tmp_path / "new_dir"))
        workbench._do_export()
        assert workbench.features is None
        assert any("尚无结果" in call[2] for call in workbench.messagebox.calls)
    finally:
        workbench._on_close()


def test_launcher_help_no_tk():
    import subprocess
    import sys
    output = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "geo_toolbox.gui", "--help"],
        capture_output=True, text=True, timeout=60,
        cwd=Path(__file__).resolve().parents[1])
    assert output.returncode == 0 and "module" in output.stdout


def test_launcher_chooser_instantiates(monkeypatch):
    launched = []

    class _FakeRoot:
        def __getattr__(self, name):
            return lambda *a, **k: None

    import tkinter
    monkeypatch.setattr(tkinter, "Tk", _FakeRoot)
    monkeypatch.setattr("geo_toolbox.gui.MagneticWorkbench",
                        lambda root: launched.append("magnetic"))
    import geo_toolbox.geochem_gui as geochem_gui
    monkeypatch.setattr(geochem_gui, "GeochemWorkbench",
                        lambda root: launched.append("geochem"))
    from geo_toolbox import gui
    gui.main(["--module", "geochem"])
    gui.main(["--module", "magnetic"])
    assert launched == ["geochem", "magnetic"]


def _censored_csv(tmp_path):
    path = tmp_path / "censored.csv"
    pd.DataFrame({
        "点号": ["001", "002", "003", "004", "005", "006"],
        "X": ["0", "100", "0", "100", "25", "75"],
        "Y": ["0", "0", "100", "100", "25", "75"],
        "Au": ["<0.5", "200", "300", "400", "180", "320"],
        "As": ["10", "23", "31", "45", "18", "32"],
        "Sb": ["2", "4", "8", "11", "5", "7"],
    }).to_csv(path, index=False)
    return path


def test_report_branch_match(tmp_path, monkeypatch):
    root, workbench = _workbench()
    try:
        _previewed(root, workbench, tmp_path)
        _configure(workbench, elements=("As", "Sb"), units={"As": "ppm", "Sb": "ppm"})
        workbench._do_import()
        assert _pump(root, workbench)
        assert workbench.features is None
        workbench.logratio_method.set("clr")
        workbench._do_transform()
        assert _pump(root, workbench)
        features_report = workbench.last_report
        assert workbench._reports["特征"] is features_report
        workbench._do_check()
        assert _pump(root, workbench)
        assert workbench._reports["特征"] is features_report
        assert workbench._reports["当前浓度"] is workbench.last_report
        sent = []
        tasks = __import__("geo_toolbox.geochem_gui_tasks", fromlist=["export_result"])
        monkeypatch.setattr(tasks, "export_result",
                            lambda data, **kw: sent.append((data, kw["report"]))
                            or Path(kw["directory"]))
        workbench.export_kind.set("特征")
        workbench.export_var.set(str(tmp_path / "out_features"))
        workbench._do_export()
        assert _pump(root, workbench)
        assert sent[0][0] is workbench.features
        context_keys = ("coordinate_check", "reference_overlay_check", "import_id_filter")
        assert {k: v for k, v in sent[0][1].items()
                if k not in context_keys} == features_report
        assert "coordinate_check" in sent[0][1]
        assert "coordinate_check" not in features_report
        assert workbench._reports["特征"] is features_report
        assert workbench._reports["原导入"] == {}
    finally:
        workbench._on_close()


def test_variogram_report_draw(tmp_path, monkeypatch):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        report = {"distance": [10, 20, 30], "semivariance": [1.0, 2.0, 3.0],
                  "pair_counts": [4, 8, 12], "parameters": {}, "fit_r2": 0.9}
        tasks = __import__("geo_toolbox.geochem_gui_tasks",
                           fromlist=["preflight", "execute"])
        monkeypatch.setattr(tasks, "preflight",
                            lambda data, **kw: {"ok": True, "errors": [], "warnings": []})
        monkeypatch.setattr(tasks, "execute",
                            lambda data, **kw: {"kind": "report", "data": report,
                                                "report": report})
        workbench.variogram_model.set("spherical")
        workbench.variogram_estimator.set("matheron")
        workbench._do_variogram()
        assert _pump(root, workbench)
        assert workbench.variogram_report is report
        assert workbench.preview_kind.get() == "变异函数"
        assert workbench._grid_canvas is not None
        workbench.maps = object()
        workbench.preview_kind.set("网格")
        assert workbench.preview_kind.get() == "网格"
    finally:
        workbench._on_close()


def test_original_immutable_and_tree_ids(tmp_path):
    root, workbench = _workbench()
    try:
        _previewed(root, workbench, tmp_path)
        _configure(workbench, elements=("As", "Sb"), units={"As": "ppm", "Sb": "ppm"})
        workbench._do_import()
        assert _pump(root, workbench)
        original = workbench.original_dataset
        snapshot = original.values.copy(deep=True)
        workbench.logratio_method.set("clr")
        workbench._do_transform()
        assert _pump(root, workbench)
        assert workbench.original_dataset is original
        assert original.values.equals(snapshot)
        assert workbench.features is not None
        items = workbench.features_tree.get_children()
        assert items and workbench.features_tree.item(items[0])["text"] in \
            set(workbench.features.values.index.astype(str))
    finally:
        workbench._on_close()


def test_actual_multrepl_impute(tmp_path):
    root, workbench = _workbench()
    try:
        workbench.path_var.set(str(_censored_csv(tmp_path)))
        workbench._do_preview()
        assert _pump(root, workbench)
        _configure(workbench)
        workbench._do_import()
        assert _pump(root, workbench)
        assert not workbench.original_dataset.imputed.to_numpy().any()
        workbench.impute_method.set("multRepl")
        workbench._do_impute()
        assert _pump(root, workbench, timeout=120.0)
        assert workbench.dataset.imputed.to_numpy().any()
        assert not workbench.original_dataset.imputed.to_numpy().any()
        assert workbench.last_result is workbench.dataset
    finally:
        workbench._on_close()


def test_export_actual_dir_and_existing_rejected(tmp_path):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        target = tmp_path / "export_new"
        workbench.export_var.set(str(target))
        workbench._do_export()
        assert _pump(root, workbench)
        assert (target / "gui_report.json").is_file()
        prior = workbench.last_result
        workbench.export_var.set(str(target))
        workbench._do_export()
        assert _pump(root, workbench)
        assert any(call[0] == "error" for call in workbench.messagebox.calls)
        assert workbench.last_result is prior
    finally:
        workbench._on_close()


def test_export_kind_selection(tmp_path, monkeypatch):
    root, workbench = _workbench()
    try:
        _imported(root, workbench, tmp_path)
        sent = []
        tasks = __import__("geo_toolbox.geochem_gui_tasks", fromlist=["export_result"])
        monkeypatch.setattr(tasks, "export_result",
                            lambda data, **kw: sent.append(data) or Path(kw["directory"]))
        workbench.export_kind.set("原导入")
        workbench.export_var.set(str(tmp_path / "new_dir"))
        workbench._do_export()
        assert _pump(root, workbench)
        assert sent and sent[0] is workbench.original_dataset
    finally:
        workbench._on_close()
