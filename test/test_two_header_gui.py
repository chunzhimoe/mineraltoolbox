import time

import pandas as pd
import pytest

tk = pytest.importorskip("tkinter")

from geo_toolbox.geochem_gui import GeochemWorkbench


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

    def askyesno(self, title, message):
        self.calls.append(("ask", title, str(message)))
        return True


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


def _two_header_xlsx(tmp_path):
    path = tmp_path / "two_header.xlsx"
    rows = pd.DataFrame([
        ["化探样品记录", "", "", "", ""],
        ["样品编号", "x", "y", "计量单位(Au:ppb)", ""],
        ["", "", "", "Au", "As"],
        ["001", "1", "2", "1.2", "10.5"],
        ["002", "3", "4", "3.4", "20.1"],
        ["003", "5", "6", "0.8", "8.8"],
    ])
    rows.to_excel(path, sheet_name="Sheet1", header=False, index=False,
                  engine="openpyxl")
    return path


def _preview(root, workbench, tmp_path):
    workbench.path_var.set(str(_two_header_xlsx(tmp_path)))
    workbench.header_var.set("2")
    workbench.header_rows_var.set("2")
    workbench._do_preview()
    assert _pump(root, workbench)
    assert workbench.preview is not None


def _configure(workbench):
    for key, column in (("sample_id", "样品编号"), ("x", "x"), ("y", "y")):
        workbench.map_combos[key].set(column)
        workbench.map_combos[key].event_generate("<<ComboboxSelected>>")
    for column in ("Au", "As"):
        row = workbench.element_rows[column]
        row["use"].set(True)
    workbench.element_rows["Au"]["unit"].set("ppb")
    workbench.element_rows["As"]["unit"].set("ppm")


def test_two_header_preview_merges_names_and_imports(tmp_path):
    root, workbench = _workbench()
    try:
        _preview(root, workbench, tmp_path)
        assert list(workbench.preview["columns"]) == [
            "样品编号", "x", "y", "Au", "As"]
        assert len(workbench.preview["frame"]) == 3
        _configure(workbench)
        workbench._do_import()
        assert _pump(root, workbench)
        assert workbench.dataset is not None
        assert len(workbench.dataset.values) == 3
        workbench.header_rows_var.set("1")
        assert workbench.dataset is None
    finally:
        workbench._on_close()


def test_two_header_generated_ids_start_at_data_row(tmp_path):
    root, workbench = _workbench()
    try:
        _preview(root, workbench, tmp_path)
        _configure(workbench)
        workbench.generated_var.set(True)
        workbench._do_import()
        assert _pump(root, workbench)
        assert workbench.dataset is not None
        assert len(workbench.dataset.values) == 3
        assert str(workbench.dataset.values.index[0]) == "row4"
    finally:
        workbench._on_close()
