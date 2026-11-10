from __future__ import annotations

import json
import queue
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path

TRANSFORMS = ("upward_continuation", "derivative_easting", "derivative_northing",
              "derivative_upward", "directional_derivative", "total_gradient_amplitude",
              "tilt_angle", "reduction_to_pole")
_TRANSFORM_PARAMS = {
    "upward_continuation": {"height_displacement"},
    "directional_derivative": {"azimuth"},
    "reduction_to_pole": {"inclination", "declination", "magnetization_inclination",
                          "magnetization_declination", "max_gain"},
}
_TRANSFORM_ALL = {"padding", "trim"}
_INTERPOLATE_METHODS = ("", "equivalent_sources", "verde_spline")
_EQS_ONLY = {"depth", "block_size"}
_UNITS = ("", "nT", "uT", "T")
_STAGES = ("", "raw_total_field", "corrected_total_field", "anomaly", "rtp")


def _tasks():
    from . import gui_tasks
    return gui_tasks


def _validation_error(text):
    from .errors import DataValidationError
    return DataValidationError(text)


def _describe(exc):
    try:
        return _tasks().describe_error(exc)
    except Exception:
        return {"kind": "unexpected", "title": "任务未完成", "message": str(exc),
                "hint": "请保留详细日志用于定位；之前的成功结果保持不变。",
                "details": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


class MagneticWorkbench:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        self.root = root
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        root.title("GeoToolbox – 磁测工作台")
        root.geometry("1200x850")
        root.minsize(900, 640)
        root.report_callback_exception = self._on_callback_error
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.preview = None
        self.dataset = None
        self.current_grid = None
        self.last_result = None
        self.last_report = None
        self._busy = False
        self._queue = queue.Queue()
        self._after = None
        self._populating = False
        self._saved_states = {}
        self._inputs = []
        self._poll_id = None
        self._scatter_canvas = self._grid_canvas = None

        self.notebook = notebook = ttk.Notebook(root)
        self.tab_import = ttk.Frame(notebook)
        self.tab_qc = ttk.Frame(notebook)
        self.tab_grid = ttk.Frame(notebook)
        self.tab_log = ttk.Frame(notebook)
        notebook.add(self.tab_import, text="导入与映射")
        notebook.add(self.tab_qc, text="质控与预览")
        notebook.add(self.tab_grid, text="网格与场变换")
        notebook.add(self.tab_log, text="日志与导出")
        notebook.pack(fill="both", expand=True, padx=6, pady=6)
        status = ttk.Frame(root)
        status.pack(fill="x", padx=6, pady=(0, 6))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=160)
        self.status_label = ttk.Label(status, text="就绪")
        self.progress.pack(side="left")
        self.status_label.pack(side="left", padx=8)
        self._build_import()
        self._build_qc()
        self._build_grid()
        self._build_log()
        self._transform_selected()
        self._interp_selected()
        self._refresh_actions()
        self._poll_id = root.after(0, self._poll)

    def _task_api(self):
        return _tasks()

    def _input(self, widget):
        self._inputs.append(widget)
        return widget

    def _build_import(self):
        tk, ttk = self.tk, self.ttk
        frame = self.tab_import
        src = ttk.LabelFrame(frame, text="源文件")
        src.pack(fill="x", padx=6, pady=4)
        self.path_var = tk.StringVar()
        self.sheet_var = tk.StringVar()
        self.header_var = tk.StringVar(value="1")
        self.path_var.trace_add("write", self._path_changed)
        for var in (self.sheet_var, self.header_var):
            var.trace_add("write", self._source_changed)
        ttk.Label(src, text="路径").grid(row=0, column=0, sticky="e")
        self._input(ttk.Entry(src, textvariable=self.path_var)).grid(
            row=0, column=1, columnspan=5, sticky="we", padx=4)
        self._input(ttk.Button(src, text="浏览…", command=self._browse)).grid(row=0, column=6)
        self.preview_btn = self._input(ttk.Button(src, text="预览", command=self._do_preview))
        self.preview_btn.grid(row=0, column=7, padx=4)
        ttk.Label(src, text="工作表").grid(row=1, column=0, sticky="e")
        self.sheet_combo = self._input(
            ttk.Combobox(src, textvariable=self.sheet_var, width=18, state="readonly"))
        self.sheet_combo.grid(row=1, column=1, sticky="w")
        ttk.Label(src, text="表头行(从1计)").grid(row=1, column=2, sticky="e")
        self._input(ttk.Entry(src, textvariable=self.header_var, width=6)).grid(row=1, column=3)
        src.columnconfigure(1, weight=1)

        mapping = ttk.LabelFrame(frame, text="列映射")
        mapping.pack(fill="x", padx=6, pady=4)
        self.map_vars = {}
        self.map_combos = {}
        fields = [("sample_id", "点号列*"), ("x", "X列*"), ("y", "Y列*"), ("value", "数值列*"),
                  ("upward", "高程列"), ("line", "测线列"), ("sequence", "点序列"),
                  ("point_line", "点线号列")]
        for i, (key, label) in enumerate(fields):
            row, col = divmod(i, 4)
            ttk.Label(mapping, text=label).grid(row=row * 2, column=col * 2, sticky="e")
            var = tk.StringVar()
            var.trace_add("write", self._mapping_changed)
            combo = self._input(ttk.Combobox(mapping, textvariable=var, width=12,
                                             state="readonly"))
            combo.grid(row=row * 2, column=col * 2 + 1, padx=2)
            combo.bind("<<ComboboxSelected>>", self._mapping_changed)
            self.map_vars[key] = var
            self.map_combos[key] = combo
        ttk.Label(mapping, text="单位*").grid(row=4, column=0, sticky="e")
        self.unit_var = tk.StringVar()
        self.unit_var.trace_add("write", self._mapping_changed)
        self.unit_combo = self._input(
            ttk.Combobox(mapping, textvariable=self.unit_var, values=_UNITS, width=8,
                         state="readonly"))
        self.unit_combo.grid(row=4, column=1, padx=2, sticky="w")
        self.unit_combo.bind("<<ComboboxSelected>>", self._mapping_changed)
        ttk.Label(mapping, text="阶段*").grid(row=4, column=2, sticky="e")
        self.stage_var = tk.StringVar()
        self.stage_var.trace_add("write", self._mapping_changed)
        self.stage_combo = self._input(
            ttk.Combobox(mapping, textvariable=self.stage_var, values=_STAGES, width=14,
                         state="readonly"))
        self.stage_combo.grid(row=4, column=3, padx=2, sticky="w")
        self.stage_combo.bind("<<ComboboxSelected>>", self._mapping_changed)
        ttk.Label(mapping, text="CRS").grid(row=4, column=4, sticky="e")
        self.crs_var = tk.StringVar()
        self.crs_var.trace_add("write", self._mapping_changed)
        self._input(ttk.Entry(mapping, textvariable=self.crs_var, width=14)).grid(
            row=4, column=5, padx=2, sticky="w")
        self.generated_var = tk.BooleanVar(value=False)
        self._input(ttk.Checkbutton(mapping, text="生成审计ID", variable=self.generated_var,
                                    command=self._mapping_changed)).grid(row=5, column=0,
                                                                        columnspan=2, padx=4,
                                                                        sticky="w")
        self.split_var = tk.BooleanVar(value=False)
        self._input(ttk.Checkbutton(mapping, text="拆分点线号", variable=self.split_var,
                                    command=self._split_toggled)).grid(row=5, column=2,
                                                                      columnspan=2, padx=4,
                                                                      sticky="w")
        self.map_combos["point_line"].configure(state="disabled")
        self.import_btn = self._input(
            ttk.Button(frame, text="导入并执行质控检查", command=self._do_import))
        self.import_btn.pack(anchor="w", padx=6, pady=4)

        table = ttk.Frame(frame)
        table.pack(fill="both", expand=True, padx=6, pady=4)
        self.tree = ttk.Treeview(table, show="headings", height=12)
        vbar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        hbar = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="we")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)

    def _build_qc(self):
        tk, ttk = self.tk, self.ttk
        frame = self.tab_qc
        info = ttk.Frame(frame)
        info.pack(fill="x", padx=6, pady=4)
        self.dataset_label = ttk.Label(info, text="尚未导入数据")
        self.dataset_label.pack(side="left")
        run = ttk.LabelFrame(frame, text="质控")
        run.pack(fill="x", padx=6, pady=4)
        self.check_btn = self._input(ttk.Button(run, text="质控检查", command=self._do_check))
        self.check_btn.grid(row=0, column=0, padx=4, pady=4)
        ttk.Label(run, text="窗口点数").grid(row=0, column=1)
        self.window_var = tk.StringVar()
        self._input(ttk.Entry(run, textvariable=self.window_var, width=8)).grid(row=0, column=2)
        ttk.Label(run, text="筛查阈值nT").grid(row=0, column=3)
        self.threshold_var = tk.StringVar()
        self._input(ttk.Entry(run, textvariable=self.threshold_var, width=10)).grid(row=0,
                                                                                   column=4)
        self.spikes_btn = self._input(ttk.Button(run, text="尖峰标记", command=self._do_spikes))
        self.spikes_btn.grid(row=0, column=5, padx=4)
        pane = ttk.PanedWindow(frame, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.LabelFrame(pane, text="检查报告")
        right = ttk.LabelFrame(pane, text="源坐标预览")
        self.report_text = tk.Text(left, wrap="none", width=60)
        report_bar = ttk.Scrollbar(left, orient="vertical", command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_bar.set)
        self.report_text.pack(side="left", fill="both", expand=True)
        report_bar.pack(side="right", fill="y")
        self.scatter_holder = ttk.Frame(right)
        self.scatter_holder.pack(fill="both", expand=True)
        pane.add(left, weight=3)
        pane.add(right, weight=2)

    def _scroll_frame(self, parent, width):
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(parent)
        canvas = tk.Canvas(outer, highlightthickness=0, width=width)
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(
            window, width=canvas.winfo_width()))
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        return outer, inner

    def _build_grid(self):
        tk, ttk = self.tk, self.ttk
        frame = self.tab_grid
        pane = ttk.PanedWindow(frame, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=4)
        left_outer, left = self._scroll_frame(pane, 500)
        right = ttk.Frame(pane)
        pane.add(left_outer, weight=0)
        pane.add(right, weight=1)

        interp = ttk.LabelFrame(left, text="网格化（样点 → 网格）")
        interp.pack(fill="x", pady=4)
        self.interp_vars = {}
        self.interp_entries = {}
        interp_fields = [("xmin", "x最小"), ("ymin", "y最小"), ("xmax", "x最大"),
                         ("ymax", "y最大"), ("resolution", "间距m*"),
                         ("upward", "目标面高程m*"), ("max_distance", "支持半径m*"),
                         ("depth", "源深度m"), ("damping", "阻尼"), ("block_size", "分块大小")]
        for i, (name, label) in enumerate(interp_fields):
            row, col = divmod(i, 2)
            ttk.Label(interp, text=label).grid(row=row, column=col * 2, sticky="e")
            var = tk.StringVar()
            entry = self._input(ttk.Entry(interp, textvariable=var, width=8))
            entry.grid(row=row, column=col * 2 + 1, padx=2, pady=1, sticky="w")
            self.interp_vars[name] = var
            self.interp_entries[name] = entry
        ttk.Label(interp, text="方法*").grid(row=5, column=0, sticky="e")
        self.interp_method = self._input(
            ttk.Combobox(interp, values=_INTERPOLATE_METHODS, width=18, state="readonly"))
        self.interp_method.grid(row=5, column=1, padx=2, sticky="w")
        self.interp_method.bind("<<ComboboxSelected>>", self._interp_selected)
        self.interp_btn = self._input(ttk.Button(interp, text="执行网格化",
                                               command=self._do_interpolate))
        self.interp_btn.grid(row=5, column=2, columnspan=2, padx=6)
        ttk.Label(interp, foreground="#8a5a00", wraplength=380, justify="left",
                  text="请确认X为东坐标、Y为北坐标、坐标/高程单位为米且高程基准已核实；"
                       "不要直接套用主图CRS").grid(row=6, column=0, columnspan=4, sticky="w")

        trans = ttk.LabelFrame(left, text="场变换（当前网格）")
        trans.pack(fill="x", pady=4)
        ttk.Label(trans, text="方法*").grid(row=0, column=0, sticky="e")
        self.transform_method = self._input(
            ttk.Combobox(trans, values=TRANSFORMS, width=18, state="readonly"))
        self.transform_method.grid(row=0, column=1, columnspan=3, padx=2, sticky="w")
        self.transform_method.bind("<<ComboboxSelected>>", self._transform_selected)
        self.transform_vars = {}
        self.transform_entries = {}
        transform_fields = [("padding", "扩展像元*"), ("trim", "裁边像元*"),
                            ("height_displacement", "延拓高度m"), ("azimuth", "方位角°"),
                            ("inclination", "磁倾角°"), ("declination", "磁偏角°"),
                            ("magnetization_inclination", "磁化倾角°"),
                            ("magnetization_declination", "磁化偏角°"), ("max_gain", "最大增益")]
        for i, (name, label) in enumerate(transform_fields):
            row, col = 1 + i // 2, (i % 2) * 2
            ttk.Label(trans, text=label).grid(row=row, column=col * 2, sticky="e")
            var = tk.StringVar()
            entry = self._input(ttk.Entry(trans, textvariable=var, width=8))
            entry.grid(row=row, column=col * 2 + 1, padx=2, pady=1, sticky="w")
            self.transform_vars[name] = var
            self.transform_entries[name] = entry
        self.induced_var = tk.BooleanVar(value=False)
        self.induced_check = self._input(
            ttk.Checkbutton(trans, text="感应磁化假设", variable=self.induced_var,
                            command=self._transform_selected))
        self.induced_check.grid(row=5, column=2, columnspan=2, sticky="w")
        self.transform_btn = self._input(ttk.Button(trans, text="执行变换",
                                                  command=self._do_transform))
        self.transform_btn.grid(row=7, column=0, columnspan=2, padx=6, pady=4)

        crop = ttk.LabelFrame(left, text="裁切（当前网格，零基索引 [起,止)）")
        crop.pack(fill="x", pady=4)
        self.crop_vars = {}
        for i, (name, label) in enumerate((("row_start", "行起"), ("row_stop", "行止"),
                                          ("column_start", "列起"), ("column_stop", "列止"))):
            row, col = divmod(i, 2)
            ttk.Label(crop, text=label).grid(row=row, column=col * 2, sticky="e")
            var = tk.StringVar()
            self._input(ttk.Entry(crop, textvariable=var, width=8)).grid(
                row=row, column=col * 2 + 1, padx=2, pady=1, sticky="w")
            self.crop_vars[name] = var
        self.crop_btn = self._input(ttk.Button(crop, text="裁切", command=self._do_crop))
        self.crop_btn.grid(row=2, column=0, columnspan=2, padx=6, pady=4)

        self.grid_label = ttk.Label(right, text="当前网格：无")
        self.grid_label.pack(anchor="w", pady=2)
        self.grid_holder = ttk.Frame(right)
        self.grid_holder.pack(fill="both", expand=True)

    def _build_log(self):
        tk, ttk = self.tk, self.ttk
        frame = self.tab_log
        self.log_text = tk.Text(frame, wrap="word", height=18)
        log_bar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_bar.set)
        self.log_text.pack(fill="both", expand=True, padx=6, pady=4)
        log_bar.pack(side="right", fill="y")
        actions = ttk.Frame(frame)
        actions.pack(fill="x", padx=6, pady=4)
        ttk.Label(actions, text="导出目录(新)").pack(side="left")
        self.export_var = tk.StringVar()
        self._input(ttk.Entry(actions, textvariable=self.export_var, width=50)).pack(
            side="left", padx=4)
        self._input(ttk.Button(actions, text="选择父目录…",
                               command=self._choose_export_parent)).pack(side="left")
        self.export_btn = self._input(ttk.Button(actions, text="导出当前结果",
                                                 command=self._do_export))
        self.export_btn.pack(side="left", padx=6)
        self._input(ttk.Button(actions, text="日志另存…", command=self._save_log)).pack(
            side="left", padx=4)
        self._input(ttk.Button(actions, text="清空日志显示",
                               command=lambda: self.log_text.delete("1.0", "end"))).pack(
            side="left")

    def _browse(self):
        name = self.filedialog.askopenfilename(
            filetypes=[("表格", ".csv .xls .xlsx .xlsm"), ("全部", "*.*")])
        if name:
            self.path_var.set(name)

    def _choose_export_parent(self):
        name = self.filedialog.askdirectory()
        if name:
            self.export_var.set(str(Path(name) / "export"))

    def _split_toggled(self):
        self.map_combos["point_line"].configure(
            state="readonly" if self.split_var.get() else "disabled")
        self._mapping_changed()

    def _interp_selected(self, event=None):
        enabled = "normal" if self.interp_method.get() == "equivalent_sources" else "disabled"
        for name in _EQS_ONLY:
            self.interp_entries[name].configure(state=enabled)

    def _transform_selected(self, event=None):
        method = self.transform_method.get()
        relevant = _TRANSFORM_PARAMS.get(method, set()) | _TRANSFORM_ALL
        if method == "reduction_to_pole" and self.induced_var.get():
            relevant -= {"magnetization_inclination", "magnetization_declination"}
        for name, entry in self.transform_entries.items():
            entry.configure(state="normal" if name in relevant else "disabled")
        self.induced_check.configure(
            state="normal" if method == "reduction_to_pole" else "disabled")

    def _path_changed(self, *args):
        if self._populating or self._busy:
            return
        self._populating = True
        try:
            self.sheet_var.set("")
            self.sheet_combo.configure(values=[])
        finally:
            self._populating = False
        self._source_changed()

    def _source_changed(self, *args):
        if self._populating or self._busy:
            return
        self.preview = None
        self.tree.delete(*self.tree.get_children())
        self._invalidate()
        self.dataset_label.configure(text="源配置已修改，请重新预览与导入")

    def _mapping_changed(self, *args):
        if self._populating or self._busy:
            return
        self._invalidate()

    def _invalidate(self):
        self.dataset = None
        self.current_grid = None
        self.last_result = None
        self.last_report = None
        self.grid_label.configure(text="当前网格：无")
        self.dataset_label.configure(text="尚未导入数据")
        self.report_text.delete("1.0", "end")
        self._dispose_canvas(self.scatter_holder, self._scatter_canvas)
        self._scatter_canvas = None
        self._dispose_canvas(self.grid_holder, self._grid_canvas)
        self._grid_canvas = None
        self._refresh_actions()

    def _refresh_actions(self):
        if self._busy:
            return
        self.import_btn.configure(state="normal" if self.preview is not None else "disabled")
        dataset = "normal" if self.dataset is not None else "disabled"
        for button in (self.check_btn, self.spikes_btn, self.interp_btn):
            button.configure(state=dataset)
        grid = "normal" if self.current_grid is not None else "disabled"
        for button in (self.transform_btn, self.crop_btn):
            button.configure(state=grid)
        self.export_btn.configure(
            state="normal" if self.last_result is not None else "disabled")

    def _set_busy(self, busy, text=""):
        self._busy = busy
        if busy:
            self._saved_states = {}
            for widget in self._inputs:
                try:
                    self._saved_states[widget] = widget.cget("state")
                    widget.configure(state="disabled")
                except Exception:
                    continue
            self.progress.start(12)
            self.status_label.configure(text=text or "运行中…")
        else:
            for widget, state in self._saved_states.items():
                try:
                    widget.configure(state=state)
                except Exception:
                    continue
            self._saved_states = {}
            self.progress.stop()
            self.status_label.configure(text="就绪")
            self._transform_selected()
            self._interp_selected()
            self._split_toggled_state()
            self._refresh_actions()

    def _split_toggled_state(self):
        self.map_combos["point_line"].configure(
            state="readonly" if self.split_var.get() else "disabled")

    def _submit(self, label, fn, on_success):
        if self._busy:
            return False
        self._set_busy(True, label)
        self._after = on_success

        def runner():
            try:
                self._queue.put(("ok", fn()))
            except Exception as exc:
                self._queue.put(("err", exc))

        threading.Thread(target=runner, daemon=True).start()
        return True

    def _poll(self):
        try:
            tag, payload = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._set_busy(False)
            callback, self._after = self._after, None
            if tag == "err":
                self._fail(payload)
            else:
                try:
                    callback(payload)
                except Exception as exc:
                    self._fail(exc)
        self._poll_id = self.root.after(80, self._poll)

    def _fail(self, exc):
        info = _describe(exc)
        self._log(f"[{info['kind']}] {info['title']}: {info['message']}\n{info['details']}")
        saved = self._autosave_error(info)
        hint = info["hint"] + (f"\n详细日志: {saved}" if saved
                               else "\n日志写入失败，请手动复制日志")
        try:
            self.messagebox.showerror(info["title"], f"{info['message']}\n\n{hint}")
        except Exception as exc:
            print(f"错误对话框显示失败: {exc}", file=sys.stderr)

    def _autosave_error(self, info):
        try:
            directory = Path(tempfile.gettempdir()) / "geo_toolbox_gui"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"error_{uuid.uuid4().hex}.txt"
            with path.open("x", encoding="utf-8") as stream:
                stream.write(f"{info['title']}\n{info['message']}\n\n{info['details']}")
            return str(path)
        except Exception as exc:
            self._log(f"错误日志写入失败: {exc}")
            return None

    def _log(self, message):
        from datetime import datetime
        self.log_text.insert("end", f"[{datetime.now():%H:%M:%S}] {message}\n")
        self.log_text.see("end")

    def _on_callback_error(self, exc, value, tb):
        self._fail(value.with_traceback(tb))

    def _dispose_canvas(self, holder, canvas):
        if canvas is not None:
            try:
                photo = getattr(canvas, "_tkphoto", None)
                if photo is not None:
                    canvas._tkcanvas.tk.call("image", "delete", photo.name)
                    canvas._tkphoto = None
            except Exception:
                pass
            try:
                canvas.figure.clear()
            except Exception:
                pass
        for widget in holder.winfo_children():
            widget.destroy()

    def _on_close(self):
        if self._busy:
            self.messagebox.showwarning("任务运行中",
                                        "任务运行中，请完成后关闭；无法中断底层计算")
            return
        if self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except Exception:
                pass
            self._poll_id = None
        self._dispose_canvas(self.scatter_holder, self._scatter_canvas)
        self._scatter_canvas = None
        self._dispose_canvas(self.grid_holder, self._grid_canvas)
        self._grid_canvas = None
        self.root.destroy()

    def _do_preview(self):
        path = self.path_var.get().strip()
        sheet_text = self.sheet_var.get().strip()
        sheet = sheet_text if sheet_text else 0
        try:
            header = int(self.header_var.get()) - 1
        except ValueError:
            header = -1
        if self._submit("预览…",
                        lambda: self._task_api().preview_source(
                            path, sheet_name=sheet, header=header),
                        self._preview_done):
            self._log(f"预览请求: {path}")

    def _preview_done(self, preview):
        self.preview = preview
        self._invalidate()
        self._populating = True
        try:
            self.tree.delete(*self.tree.get_children())
            columns = preview["columns"]
            self.tree.configure(columns=columns)
            for name in columns:
                self.tree.heading(name, text=name)
                self.tree.column(name, width=90, stretch=False)
            for _, row in preview["preview"].iterrows():
                self.tree.insert("", "end", values=[str(v) for v in row.tolist()])
            sheets = preview["sheets"]
            self.sheet_combo.configure(values=sheets,
                                       state="readonly" if sheets else "disabled")
            chosen = preview["sheet_name"]
            if sheets:
                self.sheet_var.set(str(sheets[chosen] if isinstance(chosen, int) else chosen))
            else:
                self.sheet_var.set("")
            for key, var in self.map_vars.items():
                if var.get() not in columns:
                    var.set("")
            self._apply_mapping_columns(columns)
        finally:
            self._populating = False
        self._refresh_actions()
        self._log(f"预览完成: {preview['source']} 列={len(columns)} "
                  f"全量行数={len(preview['frame'])}")

    def _apply_mapping_columns(self, columns):
        self._populating = True
        try:
            for key, combo in self.map_combos.items():
                values = [""] + list(columns) if key in ("upward", "line", "sequence",
                                                         "point_line") else list(columns)
                combo.configure(values=values)
        finally:
            self._populating = False

    def _do_import(self):
        if self.preview is None:
            return
        preview = self.preview
        mapping = {
            "sample_id": self.map_vars["sample_id"].get() or None,
            "x": self.map_vars["x"].get() or None,
            "y": self.map_vars["y"].get() or None,
            "value": self.map_vars["value"].get() or None,
            "upward": self.map_vars["upward"].get() or None,
            "line": self.map_vars["line"].get() or None,
            "sequence": self.map_vars["sequence"].get() or None,
            "unit": self.unit_var.get() or None,
            "stage": self.stage_var.get() or None,
            "crs": self.crs_var.get().strip() or None,
            "generated_ids": bool(self.generated_var.get()),
            "split_point_line": bool(self.split_var.get()),
            "point_line": self.map_vars["point_line"].get() or None,
        }
        if self._submit("导入…",
                        lambda: self._task_api().import_source(preview, mapping=mapping),
                        self._import_done):
            self._log("导入请求已提交")

    def _import_done(self, data):
        self.dataset = data
        self.current_grid = None
        self.last_result = data
        self.last_report = {}
        self.grid_label.configure(text="当前网格：无")
        self.report_text.delete("1.0", "end")
        self._refresh_actions()
        self._show_dataset_info()
        self._draw_scatter()
        self._log(f"导入完成: {len(data.values)}行 stage={data.stage} "
                  f"原始单位={data.schema.unit}(统一nT) crs={data.schema.crs or '未知'}")
        self._do_check()

    def _show_dataset_info(self):
        data = self.dataset
        if data is None:
            self.dataset_label.configure(text="尚未导入数据")
            return
        counts = data.status.value_counts().to_dict()
        self.dataset_label.configure(
            text=f"样点 {len(data.values)} 行 | stage={data.stage} | "
                 f"原单位={data.schema.unit}(统一nT) | CRS={data.schema.crs or '未知'} | "
                 f"status={counts}")

    def _do_check(self):
        if self.dataset is not None:
            self._run_operation("check", {}, self.dataset, self._qc_done)

    def _do_spikes(self):
        if self.dataset is not None:
            parameters = {"window": self.window_var.get(),
                          "threshold_nT": self.threshold_var.get()}
            self._run_operation("spikes", parameters, self.dataset, self._qc_done)

    def _run_operation(self, operation, parameters, data, on_success):
        self._submit(f"预检查 {operation}…",
                     lambda: self._task_api().preflight(data, operation=operation,
                                                        parameters=parameters),
                     lambda result: self._preflight_done(result, operation, parameters,
                                                         data, on_success))

    def _preflight_done(self, result, operation, parameters, data, on_success):
        if not result["ok"]:
            self._log(f"预检查拒绝({operation}): " + "; ".join(result["errors"]))
            self._fail(_validation_error("\n".join(result["errors"])))
            return
        if result["warnings"]:
            proceed = self.messagebox.askokcancel(
                "预检查警告", "\n".join(result["warnings"]) + "\n\n是否继续执行？")
            if not proceed:
                self._log(f"警告未确认，{operation} 已取消")
                return
        self._submit(f"执行 {operation}…",
                     lambda: self._task_api().execute(data, operation=operation,
                                                      parameters=parameters),
                     on_success)

    def _qc_done(self, result):
        self.dataset = result["data"]
        self.last_result = result["data"]
        self.last_report = result["report"]
        self._refresh_actions()
        self._show_dataset_info()
        self._draw_scatter()
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", json.dumps(result["report"], ensure_ascii=False,
                                                  indent=2, default=str))
        self._log("质控完成，报告已更新")

    def _draw_scatter(self):
        data = self.dataset
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as exc:
            self._log(f"预览绘图不可用: {exc}")
            return
        self._dispose_canvas(self.scatter_holder, self._scatter_canvas)
        self._scatter_canvas = None
        figure = Figure(figsize=(5, 4), dpi=100)
        axis = figure.add_subplot(111)
        if data is not None:
            import numpy as np
            xy = data.coordinates[["easting", "northing"]].to_numpy(dtype=float)
            values = data.values.to_numpy(dtype=float)
            mask = np.isfinite(xy).all(axis=1) & np.isfinite(values)
            axis.scatter(xy[mask, 0], xy[mask, 1], s=4, c=values[mask])
            axis.set_title(
                f"Source coordinates (finite shown); CRS={data.schema.crs or 'unknown'}")
        canvas = FigureCanvasTkAgg(figure, master=self.scatter_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._scatter_canvas = canvas

    def _draw_grid(self):
        grid = self.current_grid
        try:
            import numpy as np
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as exc:
            self._log(f"网格绘图不可用: {exc}")
            return
        self._dispose_canvas(self.grid_holder, self._grid_canvas)
        self._grid_canvas = None
        figure = Figure(figsize=(6, 4.5), dpi=100)
        axis = figure.add_subplot(111)
        if grid is not None:
            a, _, c, _, e, f = grid.grid.transform
            extent = (c, c + grid.grid.width * a, f + grid.grid.height * e, f)
            masked = np.ma.masked_where(~grid.support, grid.values)
            image = axis.imshow(masked, extent=extent, origin="upper", aspect="equal")
            figure.colorbar(image, ax=axis, label=grid.unit)
            axis.set_title(f"{grid.stage} @ upward={grid.upward}m")
        canvas = FigureCanvasTkAgg(figure, master=self.grid_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._grid_canvas = canvas

    def _do_interpolate(self):
        if self.dataset is None:
            return
        parameters = {name: var.get() for name, var in self.interp_vars.items()}
        parameters["method"] = self.interp_method.get()
        if self.interp_method.get() != "equivalent_sources":
            for name in _EQS_ONLY:
                parameters.pop(name, None)
        self._run_operation("interpolate", parameters, self.dataset, self._grid_done)

    def _do_transform(self):
        if self.current_grid is None:
            return
        method = self.transform_method.get()
        parameters = {"method": method,
                      "padding": self.transform_vars["padding"].get(),
                      "trim": self.transform_vars["trim"].get()}
        relevant = set(_TRANSFORM_PARAMS.get(method, set()))
        if method == "reduction_to_pole" and self.induced_var.get():
            relevant -= {"magnetization_inclination", "magnetization_declination"}
        for name in relevant:
            value = self.transform_vars[name].get()
            if value != "":
                parameters[name] = value
        if method == "reduction_to_pole":
            parameters["assume_induced"] = bool(self.induced_var.get())
        self._run_operation("transform", parameters, self.current_grid, self._grid_done)

    def _do_crop(self):
        if self.current_grid is not None:
            parameters = {name: var.get() for name, var in self.crop_vars.items()}
            self._run_operation("crop", parameters, self.current_grid, self._grid_done)

    def _grid_done(self, result):
        self.current_grid = result["data"]
        self.last_result = result["data"]
        self.last_report = result["report"]
        grid = self.current_grid
        self.grid_label.configure(
            text=f"当前网格：{grid.grid.width}×{grid.grid.height} stage={grid.stage} "
                 f"unit={grid.unit} upward={grid.upward}")
        self._draw_grid()
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", json.dumps(result["report"], ensure_ascii=False,
                                                  indent=2, default=str))
        self._refresh_actions()
        self._log("网格运算完成")

    def _do_export(self):
        target = self.export_var.get().strip()
        result, report = self.last_result, self.last_report
        if result is None or not target:
            self.messagebox.showwarning("导出", "请先生成结果并填写新的导出目录")
            return
        self._submit("导出…",
                     lambda: self._task_api().export_result(
                         result, directory=target, report=report or {}),
                     self._export_done)

    def _export_done(self, path):
        self._log(f"导出完成: {path}")
        self.messagebox.showinfo("导出", f"已导出至 {path}")

    def _save_log(self):
        name = self.filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("日志", ".txt")],
            title="保存日志（不覆盖已有文件）")
        if not name:
            return
        try:
            with Path(name).open("x", encoding="utf-8") as stream:
                stream.write(self.log_text.get("1.0", "end"))
            self._log(f"日志已保存: {name}")
        except Exception as exc:
            self._fail(exc)

    def start(self):
        self.root.mainloop()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["magnetic", "geochem"], default=None)
    args = parser.parse_args(argv)
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        print(f"tkinter 不可用，无法启动图形界面: {exc}", file=sys.stderr)
        raise SystemExit(1)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"无法初始化图形界面: {exc}", file=sys.stderr)
        raise SystemExit(1)

    def launch(kind):
        if kind == "geochem":
            from .geochem_gui import GeochemWorkbench
            GeochemWorkbench(root)
        else:
            MagneticWorkbench(root)

    if args.module is not None:
        launch(args.module)
        root.mainloop()
        return
    root.title("GeoToolbox")
    chooser = ttk.Frame(root)
    chooser.pack(fill="both", expand=True, padx=20, pady=20)
    ttk.Label(chooser, text="选择工作台", anchor="center").pack(pady=8)
    for text, kind in (("磁测工作台", "magnetic"), ("地球化学工作台", "geochem")):
        def pick(kind=kind):
            chooser.destroy()
            launch(kind)
        ttk.Button(chooser, text=text, command=pick).pack(pady=4)
    root.geometry("360x160")
    root.mainloop()


if __name__ == "__main__":
    main()
