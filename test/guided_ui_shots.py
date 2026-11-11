import time
from pathlib import Path

import pandas as pd


class _Msg:
    def __init__(self):
        self.calls = []

    def showerror(self, *a):
        self.calls.append(("error", *a))

    def showwarning(self, *a):
        self.calls.append(("warning", *a))

    def showinfo(self, *a):
        self.calls.append(("info", *a))

    def askokcancel(self, *a):
        return True


def _pump(root, wb, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if not wb._busy and wb._queue.empty():
            return True
        time.sleep(0.02)
    return False


def _shot(root, out, name):
    root.update()
    time.sleep(0.25)
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    from PIL import ImageGrab
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(out / name)


def _csv(tmp, name, frame):
    path = tmp / name
    frame.to_csv(path, index=False)
    return path


def _mag(tmp):
    return _csv(tmp, "mag.csv", pd.DataFrame(
        {"点号": ["p1", "p2", "p3"], "X": ["0", "10", "20"],
         "Y": ["0", "10", "20"], "△T": ["1.0", "-2.0", "3.0"]}))


def _geo(tmp):
    return _csv(tmp, "geo.csv", pd.DataFrame(
        {"点号": ["p1", "p2", "p3"], "X": ["0", "10", "20"],
         "Y": ["0", "10", "20"], "Au": ["1.0", "<0.5", "2.0"],
         "As": ["10", "20", "30"], "Sb": ["2", "4", "6"]}))


def _mag_flow(root, wb, csv):
    wb.path_var.set(str(csv))
    wb._do_preview()
    assert _pump(root, wb)
    wb._check_source_coordinates()
    assert _pump(root, wb)
    return {"unit": "nT", "stage": "anomaly"}


def _geo_flow(root, wb, csv):
    wb.path_var.set(str(csv))
    wb._do_preview()
    assert _pump(root, wb)
    wb._check_source_coordinates()
    assert _pump(root, wb)
    for col, row in wb.element_rows.items():
        element = col in ("Au", "As", "Sb")
        row["use"].set(element)
        if element:
            row["unit"].set("ppm")
    return {}


def _run(cls, tag, csv_maker, flow, apply_map, tmp, out):
    import tkinter as tk
    for size in ("1024x768", "900x640"):
        root = tk.Tk()
        wb = cls(root)
        wb.messagebox = _Msg()
        root.geometry(size)
        root.deiconify()
        root.update()
        csv = csv_maker(tmp)
        meta = flow(root, wb, csv)
        _shot(root, out, f"{tag}_step1_{size}.png")
        wb.notebook.select(wb.tab_confirm)
        apply_map(wb, meta)
        root.update()
        _shot(root, out, f"{tag}_step2_{size}.png")
        wb._do_import()
        assert _pump(root, wb), f"{tag} {size} import did not complete"
        _shot(root, out, f"{tag}_step3_{size}.png")
        if hasattr(wb, "_open_processing"):
            wb._open_processing()
            _shot(root, out, f"{tag}_task_{size}.png")
        wb.notebook.select(wb.tab_log)
        _shot(root, out, f"{tag}_step4_{size}.png")
        errors = [c for c in wb.messagebox.calls if c[0] == "error"]
        assert not errors, f"{tag} {size} error dialogs: {errors}"
        wb._on_close()


def _mag_map(wb, meta):
    wb.unit_var.set(meta["unit"])
    wb.stage_var.set(meta["stage"])


def _geo_map(wb, meta):
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    out.mkdir(parents=True, exist_ok=False)
    tmp = out / "_data"
    tmp.mkdir(parents=True)
    from geo_toolbox.gui import MagneticWorkbench
    from geo_toolbox.geochem_gui import GeochemWorkbench
    _run(MagneticWorkbench, "mag", _mag, _mag_flow, _mag_map, tmp, out)
    _run(GeochemWorkbench, "geo", _geo, _geo_flow, _geo_map, tmp, out)
    print("saved", out)


if __name__ == "__main__":
    main()
