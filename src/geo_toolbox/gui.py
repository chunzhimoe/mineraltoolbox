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
        self._populating = True
        self._saved_states = {}
        self._inputs = []
        self._poll_id = None
        self._scatter_canvas = self._grid_canvas = None
        self.reference_overlay = None
        self._overlay_status = None

        self.notebook = notebook = ttk.Notebook(root)
        self.tab_import = ttk.Frame(notebook)
        self.tab_confirm = ttk.Frame(notebook)
        self.tab_qc = ttk.Frame(notebook)
        self.tab_grid = ttk.Frame(notebook)
        self.tab_log = ttk.Frame(notebook)
        notebook.add(self.tab_import, text="1 选择文件")
        notebook.add(self.tab_confirm, text="2 确认数据")
        notebook.add(self.tab_qc, text="3 检查与看图")
        notebook.add(self.tab_log, text="4 保存结果")
        notebook.add(self.tab_grid, text="后续处理（可跳过）")
        notebook.hide(self.tab_grid)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)
        status = ttk.Frame(root)
        status.pack(fill="x", padx=6, pady=(0, 6))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=160)
        self.status_label = ttk.Label(status, text="就绪")
        self.progress.pack(side="left")
        self.status_label.pack(side="left", padx=8)
        navigation = ttk.Frame(root)
        navigation.pack(fill="x", padx=6, pady=(0, 6))
        self.prev_btn = self._input(ttk.Button(navigation, text="上一步",
                                             command=lambda: self._navigate(-1)))
        self.prev_btn.pack(side="left")
        self.step_label = ttk.Label(navigation, text="")
        self.step_label.pack(side="left", padx=10)
        self.next_btn = self._input(ttk.Button(navigation, text="继续 →",
                                               command=lambda: self._navigate(1)))
        self.next_btn.pack(side="right")
        notebook.bind("<<NotebookTabChanged>>", self._update_step_label)
        self._build_import()
        self._build_coordinate_check()
        self._build_qc()
        self._build_grid()
        self._build_log()
        self._translate_choices()
        self._transform_selected()
        self._interp_selected()
        self._populating = False
        self._refresh_actions()
        self._poll_id = root.after(0, self._poll)

    def _task_api(self):
        return _tasks()

    def _translate_choices(self):
        from .gui_guidance import labeled
        for name in ('stage_combo', 'interp_method', 'transform_method', 'impute_method',
                     'logratio_method', 'variogram_model', 'kriging_variogram', 'validate_variogram'):
            combo = getattr(self, name, None)
            if combo is not None:
                combo.configure(values=[labeled(value) for value in combo.cget('values')])

    def _draw_reference_overlay(self, axis):
        overlay = self.reference_overlay
        if overlay is None or self._overlay_status != 'compared':
            return
        from matplotlib.collections import LineCollection
        if overlay.geometry_kind == 'points':
            import numpy as np
            xy = np.concatenate(overlay.parts) if overlay.parts else np.empty((0, 2))
            axis.scatter(xy[:, 0], xy[:, 1], s=10, c='k', marker='x')
        else:
            axis.add_collection(LineCollection(overlay.parts, colors='k', linewidths=.6))
            axis.autoscale_view()

    def _navigate(self, delta):
        if self._busy:
            return
        step = self._current_step()
        if delta > 0 and step == 1:
            if self.preview is None:
                self._do_preview()
            else:
                self._check_source_coordinates(continue_after=True)
            return
        if delta > 0 and step == 2:
            self._do_import()
            return
        if delta > 0 and step == 4:
            self._do_export()
            return
        pages = (self.tab_import, self.tab_confirm, self.tab_qc, self.tab_log)
        self.notebook.select(pages[max(0, min(3, step - 1 + delta))])
        self._update_step_label()

    def _open_processing(self):
        if self.dataset is None or self._busy:
            return
        self.notebook.select(self.tab_grid)

    def _update_step_label(self, event=None):
        if not hasattr(self, 'step_label'):
            return
        step = self._current_step()
        names = {1: "第1步：导入后先核对坐标；未知可做非空间检查", 2: "第2步：确认数据列、单位与含义",
                 3: "第3步：查看检查结果；可以直接保存", 4: "第4步：另存结果，不修改原文件"}
        self.step_label.configure(text=names.get(step, "后续处理：按需要选择，可跳过"))
        if not self._busy:
            texts = {1: "下一步：确认数据" if self.preview is not None else "读取表格",
                     2: "开始检查并看图", 3: "下一步：保存结果", 4: "保存到新目录"}
            self.next_btn.configure(text=texts.get(step, "返回检查结果"))
            self.prev_btn.configure(state="disabled" if step == 1 else "normal")

    def _current_step(self):
        selected = self.notebook.select()
        for step, page in enumerate((self.tab_import, self.tab_confirm, self.tab_qc, self.tab_log), 1):
            if selected == str(page):
                return step
        return 3

    def _toggle_section(self, frame):
        if frame.winfo_manager():
            frame.pack_forget()
        else:
            frame.pack(fill="x", padx=6, pady=4)

    def _details_section(self, parent, title):
        outer = self.ttk.Frame(parent)
        outer.pack(fill="x", padx=6, pady=4)
        body = self.ttk.Frame(outer)
        self._input(self.ttk.Button(outer, text=title,
                                   command=lambda: self._toggle_section(body))).pack(anchor="w")
        return body

    def _input(self, widget):
        self._inputs.append(widget)
        return widget

    def _build_coordinate_check(self):
        ttk = self.ttk
        self.coordinate_report = None
        panel = ttk.LabelFrame(self.tab_import, text="导入第一步：先检查测点坐标（磁测／地化通用）")
        panel.pack(fill="x", padx=6, pady=4, before=self.tree.master)
        ttk.Label(panel, text="表格没有内置投影信息。请选择坐标列，并提供测点自己的坐标说明；不知道也能继续非空间检查。",
                  wraplength=790).grid(row=0, column=0, columnspan=5, sticky="w", padx=6, pady=4)
        for column, (role, label) in enumerate((("x", "东坐标／经度列"), ("y", "北坐标／纬度列"))):
            ttk.Label(panel, text=label).grid(row=1, column=column * 2, sticky="e")
            self._input(ttk.Combobox(panel, textvariable=self.map_vars[role], width=15,
                                     postcommand=self._coordinate_columns, state="readonly")).grid(
                row=1, column=column * 2 + 1, padx=4, pady=4)
        ttk.Label(panel, text="测点坐标系（未知留空）").grid(row=2, column=0, sticky="e")
        self._input(ttk.Entry(panel, textvariable=self.crs_var, width=22)).grid(row=2, column=1, sticky="w")
        self._input(ttk.Button(panel, text="读取 PRJ／WKT", command=self._choose_coordinate_definition)).grid(row=2, column=2, padx=4)
        self._input(ttk.Button(panel, text="检查坐标", command=self._check_source_coordinates)).grid(row=2, column=3, padx=4)
        self.coordinate_label = ttk.Label(panel, text="选文件后，先核对坐标；没有坐标说明可保留未知。",
                                          wraplength=790, justify="left")
        self.coordinate_label.grid(row=3, column=0, columnspan=5, sticky="w", padx=6, pady=6)
        self._input(ttk.Button(panel, text="核对参考 SHP／逐层 JSON（不套用到测点）",
                               command=self._choose_coordinate_reference)).grid(row=4, column=0, columnspan=4, sticky="w", padx=6, pady=4)

    def _choose_coordinate_reference(self):
        path = self.filedialog.askopenfilename(title="选择参考 SHP（读取声明，不改变测点）",
                                              filetypes=[("SHP", "*.shp")])
        if not path:
            return
        reference = self.filedialog.askopenfilename(title="可选：该图层的 spatial_reference JSON；没有可取消",
                                                   filetypes=[("JSON", "*.json")]) or None
        from .reference_overlay import read_overlay
        self._submit("核对参考图层声明…", lambda: read_overlay(path, reference=reference), self._coordinate_reference_done)

    def _coordinate_reference_done(self, overlay):
        from pyproj import CRS
        name = CRS.from_wkt(overlay.crs_wkt).name if overlay.crs_wkt else '未知'
        self.coordinate_label.configure(text=f'参考图层：{overlay.layer}；坐标声明：{name}。\n'
                                        f'原始范围：{overlay.bounds}。\n'
                                        '此声明只属于该图层，未套用到测点；请按测量报告确认测点自己的坐标系。\n'
                                        + '；'.join(overlay.warnings))

    def _coordinate_columns(self):
        columns = self.preview['columns'] if self.preview else []
        for child in self.coordinate_label.master.winfo_children():
            if isinstance(child, self.ttk.Combobox):
                child.configure(values=columns)

    def _choose_coordinate_definition(self):
        name = self.filedialog.askopenfilename(title="选择测点本身的坐标定义（不是替用底图坐标系）",
                                              filetypes=[("坐标定义", "*.prj *.wkt *.txt")])
        if name:
            try:
                self.crs_var.set(Path(name).read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError) as error:
                self._fail(error)

    def _check_source_coordinates(self, continue_after=False):
        if self.preview is None:
            self.coordinate_label.configure(text="请先选择并读取表格，再检查坐标。")
            return
        x, y = self.map_vars['x'].get(), self.map_vars['y'].get()
        if x not in self.preview['columns'] or y not in self.preview['columns'] or x == y:
            self.coordinate_label.configure(text="请分别选择两列坐标。若没有可用坐标，仍可保留未知，但当前数据接口需要明确坐标列。")
            return
        import pandas as pd
        points = self.preview['frame'][[x, y]].apply(pd.to_numeric, errors='coerce').to_numpy()
        crs = self.crs_var.get().strip() or None
        from .coordinate_check import check_coordinates
        self._submit("检查坐标声明与数值…", lambda: check_coordinates(points, crs=crs),
                     lambda report: self._coordinates_checked(report, continue_after))

    def _coordinates_checked(self, report, continue_after):
        from .coordinate_check import coordinate_summary
        self.coordinate_report = report
        self.coordinate_label.configure(text=coordinate_summary(report))
        if continue_after:
            if report['status'] == 'review':
                self.messagebox.showwarning('坐标需要复核，原值保留', coordinate_summary(report) + '\n\n可以先做非空间检查；空间处理会再次提示，并继续执行原有坐标／单位约束。')
            self.notebook.select(self.tab_confirm)
        self._update_step_label()

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

        ttk.Label(frame, text="先看下面的列名和数据。若第一行是标题，请把“列名在第几行”改成 2，再点读取。",
                  wraplength=760).pack(anchor="w", padx=8, pady=6)
        confirm_outer, confirm = self._scroll_frame(self.tab_confirm, 760)
        confirm_outer.pack(fill="both", expand=True)
        ttk.Label(confirm, text="请选择要检查的数据。列名只是候选；单位和数据类型请按成果说明确认。",
                  wraplength=740).pack(anchor="w", padx=6, pady=8)
        mapping = ttk.LabelFrame(confirm, text="这些列分别表示什么？")
        mapping.pack(fill="x", padx=6, pady=4)
        optional = self._details_section(confirm, "补充坐标／测线信息（不知道可跳过）")
        self.mapping_details = optional
        self.map_vars = {}
        self.map_combos = {}
        fields = [("sample_id", "点号列*"), ("x", "X列*"), ("y", "Y列*"), ("value", "数值列*"),
                  ("upward", "高程列"), ("line", "测线列"), ("sequence", "点序列"),
                  ("point_line", "点线号列")]
        for i, (key, label) in enumerate(fields):
            container = mapping if i < 4 else optional
            row, col = divmod(i % 4, 2)
            ttk.Label(container, text=label).grid(row=row, column=col * 2, sticky="e")
            var = tk.StringVar()
            var.trace_add("write", self._mapping_changed)
            combo = self._input(ttk.Combobox(container, textvariable=var, width=20,
                                             state="readonly"))
            combo.grid(row=row, column=col * 2 + 1, padx=6, pady=6)
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
        ttk.Label(mapping, text="这列是什么？*").grid(row=4, column=2, sticky="e")
        self.stage_var = tk.StringVar()
        self.stage_var.trace_add("write", self._mapping_changed)
        self.stage_combo = self._input(
            ttk.Combobox(mapping, textvariable=self.stage_var, values=_STAGES, width=30,
                         state="readonly"))
        self.stage_combo.grid(row=4, column=3, padx=2, sticky="w")
        self.stage_combo.bind("<<ComboboxSelected>>", self._mapping_changed)
        ttk.Label(optional, text="测点坐标系（未知留空）").grid(row=2, column=0, sticky="e")
        self.crs_var = tk.StringVar()
        self.crs_var.trace_add("write", self._mapping_changed)
        self._input(ttk.Entry(optional, textvariable=self.crs_var, width=22)).grid(
            row=2, column=1, padx=6, sticky="w")
        ttk.Label(optional, text="例如 EPSG:21456。底图坐标系不一定就是测点坐标系；不自动交换 X/Y 或去除带号。",
                  wraplength=680).grid(row=3, column=0, columnspan=4, sticky="w", pady=6)
        self.skip_ids_var = tk.BooleanVar(value=True)
        self._input(ttk.Checkbutton(mapping, text="跳过空点号；重复点号只保留首次（记录跳过明细）",
                                    variable=self.skip_ids_var, command=self._mapping_changed)).grid(
            row=5, column=0, columnspan=4, sticky="w", padx=4)
        self.generated_var = tk.BooleanVar(value=False)
        self._input(ttk.Checkbutton(optional, text="改用行号编号（优先于跳过选项，保留所有行及原点号）", variable=self.generated_var,
                                    command=self._mapping_changed)).grid(row=4, column=0,
                                                                        columnspan=2, padx=4,
                                                                        sticky="w")
        self.split_var = tk.BooleanVar(value=False)
        self._input(ttk.Checkbutton(optional, text="按“线号-点序”拆分（仅用于沿线检查）", variable=self.split_var,
                                    command=self._split_toggled)).grid(row=5, column=0,
                                                                      columnspan=2, padx=4,
                                                                      sticky="w")
        self.map_combos["point_line"].configure(state="disabled")
        self.import_btn = self._input(
            ttk.Button(confirm, text="确认这些列，开始检查", command=self._do_import))
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
        basic = ttk.Frame(frame)
        basic.pack(fill="x", padx=6, pady=4)
        self._input(ttk.Button(basic, text="按需要继续处理（可跳过）", command=self._open_processing)).pack(side="left", padx=4)
        self._input(ttk.Button(basic, text="导入 SHP 底图核对", command=self._choose_reference_overlay)).pack(side="left", padx=4)
        self._input(ttk.Button(basic, text="检查摘要", command=lambda: self._show_readable_report(self.last_report or {}))).pack(side="left", padx=4)
        self._input(ttk.Button(basic, text="详细记录", command=self._show_detailed_report)).pack(side="left", padx=4)
        run = self._details_section(frame, "重新检查／沿测线标记突变候选（不删除原值）")
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
        self.report_text = tk.Text(left, wrap="word", width=44)
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
        taskbar = ttk.Frame(frame)
        taskbar.pack(fill="x", pady=2)
        ttk.Label(taskbar, text="选择任务：").pack(side="left")
        self.grid_task = ttk.Combobox(
            taskbar, width=26, state="readonly",
            values=("把测点生成网格图", "突出变化/边界（导数/总梯度/倾角）",
                    "让浅部细节减弱（向上延拓）", "调整磁异常位置（化极）",
                    "裁掉网格边缘"))
        self.grid_task.set("把测点生成网格图")
        self.grid_task.pack(side="left", padx=4)
        self.grid_task.bind("<<ComboboxSelected>>", self._grid_task_changed)
        self.grid_task_hint = ttk.Label(taskbar, foreground="#8a5a00", wraplength=520)
        self.grid_task_hint.pack(side="left", padx=6)
        pane = ttk.PanedWindow(frame, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=4)
        left_outer, left = self._scroll_frame(pane, 500)
        right = ttk.Frame(pane)
        pane.add(left_outer, weight=0)
        pane.add(right, weight=1)

        interp = self._interp_frame = ttk.LabelFrame(left, text="网格化（样点 → 网格）")
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

        trans = self._trans_frame = ttk.LabelFrame(left, text="场变换（当前网格）")
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

        crop = self._crop_frame = ttk.LabelFrame(left, text="裁切（当前网格，零基索引 [起,止)）")
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
        self._grid_task_changed()

    def _grid_task_changed(self, event=None):
        task = self.grid_task.get()
        visibility = {"把测点生成网格图": (self._interp_frame,),
                      "突出变化/边界（导数/总梯度/倾角）": (self._trans_frame,),
                      "让浅部细节减弱（向上延拓）": (self._trans_frame,),
                      "调整磁异常位置（化极）": (self._trans_frame,),
                      "裁掉网格边缘": (self._crop_frame,)}
        from .gui_guidance import labeled
        if task == "让浅部细节减弱（向上延拓）":
            self.transform_method.set(labeled('upward_continuation'))
        elif task == "调整磁异常位置（化极）":
            self.transform_method.set(labeled('reduction_to_pole'))
        self._transform_selected()
        hints = {"突出变化/边界（导数/总梯度/倾角）": "显示变化强的位置；需先有完整网格，导数可能放大噪声。",
                 "让浅部细节减弱（向上延拓）": "填写要向上移动多少米；不改变原始数据。",
                 "调整磁异常位置（化极）": "需要已确认的磁场／磁化方向；低纬或增益过大时不宜直接使用。"}
        for frame in (self._interp_frame, self._trans_frame, self._crop_frame):
            frame.pack_forget()
        for frame in visibility.get(task, (self._interp_frame,)):
            frame.pack(fill="x", pady=4)
        self.grid_task_hint.configure(text=hints.get(task, ""))

    def _build_log(self):
        tk, ttk = self.tk, self.ttk
        frame = self.tab_log
        self.log_text = tk.Text(frame, wrap="word", height=18)
        log_bar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_bar.set)
        self.log_text.pack(fill="both", expand=True, padx=6, pady=4)
        log_bar.pack(side="right", fill="y")
        actions = ttk.LabelFrame(frame, text="另存当前检查结果（原文件不动）")
        actions.pack(fill="x", padx=6, pady=8, before=self.log_text)
        ttk.Label(actions, text="新目录").grid(row=0, column=0, sticky="e")
        self.export_var = tk.StringVar()
        self._input(ttk.Entry(actions, textvariable=self.export_var, width=45)).grid(
            row=0, column=1, padx=4, pady=6, sticky="we")
        self._input(ttk.Button(actions, text="选择保存位置…",
                               command=self._choose_export_parent)).grid(row=0, column=2)
        actions.columnconfigure(1, weight=1)
        self.export_btn = self._input(ttk.Button(actions, text="导出当前结果",
                                                 command=self._do_export))
        self.export_btn.grid(row=0, column=3, padx=6)
        self._input(ttk.Button(actions, text="日志另存…", command=self._save_log)).grid(
            row=1, column=1, padx=4, pady=6, sticky="w")
        self._input(ttk.Button(actions, text="清空日志显示",
                               command=lambda: self.log_text.delete("1.0", "end"))).grid(
            row=1, column=2, padx=4)
        ttk.Label(actions, text="下面是运行记录；导出包括原始记录、当前数据、检查报告和处理元数据。",
                  wraplength=720).grid(row=2, column=0, columnspan=4, sticky="w", padx=6, pady=6)

    def _choose_reference_overlay(self):
        path = self.filedialog.askopenfilename(
            title="选择SHP底图", filetypes=[("SHP", "*.shp"), ("全部", "*.*")])
        if not path:
            return
        reference = self.filedialog.askopenfilename(
            title="可选：参考JSON文件（没有直接取消）",
            filetypes=[("JSON", "*.json"), ("全部", "*.*")]) or None
        data = self.dataset
        self._submit("读取底图…",
                     lambda: self._load_overlay(path, reference, data),
                     self._overlay_done)

    def _load_overlay(self, path, reference, data):
        import numpy as np
        from .reference_overlay import compare_overlay, read_overlay
        overlay = read_overlay(path, reference=reference)
        points = None
        if data is not None:
            axes = ("easting", "northing") if "easting" in data.coordinates else ("x", "y")
            xy = data.coordinates[list(axes)].to_numpy(dtype=float)
            points = xy
        crs = getattr(data, "crs", None) if data is not None else None
        if crs is None and data is not None:
            crs = getattr(getattr(data, "schema", None), "crs", None)
        report = compare_overlay(points, overlay, source_crs=crs) \
            if points is not None else None
        return overlay, report

    def _overlay_done(self, result):
        overlay, report = result
        self.reference_overlay = overlay
        self.overlay_report = report
        self._overlay_status = (report or {}).get("status")
        if self._overlay_status == "compared":
            self._log("底图已叠加显示（仅按双方声明比较，不能认证定位）")
            self._draw_scatter()
            self.messagebox.showinfo('底图范围核对',
                f"总计 {report['sample_rows']} 行；底图外接矩形内 {report['rows_inside_layer_bbox']} 行，"
                f"矩形外 {report['rows_outside_layer_bbox']} 行，坐标缺失／非法 {report['invalid_coordinate_rows']} 行。\n"
                '这是外接范围检查，不是地质面相交统计，也不是坐标基准正确性证明。')
        else:
            self._draw_scatter()
            self.messagebox.showinfo(
                "底图核对",
                "底图与测点尚未确认对齐，不叠加绘制。\n"
                f"测点范围：{(report or {}).get('sample_bounds')}\n"
                f"底图范围：{overlay.bounds}\n"
                f"原因：{(report or {}).get('reason', '请先导入测点')}\n"
                '请在第一步核对测点自己的坐标说明；不自动套用底图投影或修改坐标。')
            self._log(f"底图未叠加: {(report or {}).get('reason', '坐标系未知')}")

    def _browse(self):
        name = self.filedialog.askopenfilename(
            filetypes=[("表格", ".csv .xls .xlsx .xlsm"), ("全部", "*.*")])
        if name:
            self.path_var.set(name)
            self._do_preview()

    def _choose_export_parent(self):
        name = self.filedialog.askdirectory()
        if name:
            self.export_var.set(str(Path(name) / "export"))

    def _split_toggled(self):
        self.map_combos["point_line"].configure(
            state="readonly" if self.split_var.get() else "disabled")
        self._mapping_changed()

    def _interp_selected(self, event=None):
        from .gui_guidance import raw_method
        enabled = "normal" if raw_method(self.interp_method.get()) == "equivalent_sources" else "disabled"
        for name in _EQS_ONLY:
            self.interp_entries[name].configure(state=enabled)

    def _transform_selected(self, event=None):
        from .gui_guidance import raw_transform
        method = raw_transform(self.transform_method.get())
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
            self._clear_mapping_choices()
        finally:
            self._populating = False
        self._source_changed()

    def _clear_mapping_choices(self):
        for var in self.map_vars.values():
            var.set("")
        for name in ("unit_var", "stage_var", "crs_var"):
            var = getattr(self, name, None)
            if var is not None:
                var.set("")

    def _source_changed(self, *args):
        if self._populating or self._busy:
            return
        self.preview = None
        self.tree.delete(*self.tree.get_children())
        self._populating = True
        try:
            self._clear_mapping_choices()
        finally:
            self._populating = False
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
        self.reference_overlay = None
        self._overlay_status = None
        self.overlay_report = None
        self.coordinate_report = None
        if hasattr(self, 'coordinate_label'):
            self.coordinate_label.configure(text="配置已更改，请重新检查坐标；不自动套用之前的坐标系。")
        if self.notebook.select() not in (str(self.tab_import), str(self.tab_confirm)):
            self.notebook.select(self.tab_import if self.preview is None else self.tab_confirm)
        self._refresh_actions()
        self._update_step_label()

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
        self._refresh_navigation()

    def _refresh_navigation(self):
        if self._busy:
            return
        for page, available in ((self.tab_confirm, self.preview is not None),
                                (self.tab_qc, self.dataset is not None),
                                (self.tab_log, self.last_result is not None)):
            self.notebook.tab(page, state='normal' if available else 'disabled')
        self._update_step_label()

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
            self._busy_tab = self.notebook.select()
            self.notebook.bind('<Button-1>', lambda event: 'break')
            self.notebook.bind('<Key>', lambda event: 'break')
            self.progress.start(12)
            self.status_label.configure(text=text or "运行中…")
        else:
            for widget, state in self._saved_states.items():
                try:
                    widget.configure(state=state)
                except Exception:
                    continue
            self._saved_states = {}
            self.notebook.unbind('<Button-1>')
            self.notebook.unbind('<Key>')
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
            from .gui_guidance import suggest_columns
            for role, column in suggest_columns(columns, self.map_vars).items():
                if not self.map_vars[role].get():
                    self.map_vars[role].set(column)
        finally:
            self._populating = False
        self._refresh_actions()
        self._coordinate_columns()
        self._update_step_label()
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
        from .gui_guidance import require_fields, raw_stage
        missing = require_fields(
            {"sample_id": self.map_vars["sample_id"].get(), "x": self.map_vars["x"].get(),
             "y": self.map_vars["y"].get(), "value": self.map_vars["value"].get(),
             "unit": self.unit_var.get(), "stage": self.stage_var.get()},
            [("sample_id", "点号列"), ("x", "X列"), ("y", "Y列"), ("value", "数值列"),
             ("unit", "单位"), ("stage", "数据阶段")])
        if missing:
            self.messagebox.showwarning("缺少信息", missing)
            self._log(f"导入未提交: {missing}")
            return
        mapping = {
            "sample_id": self.map_vars["sample_id"].get() or None,
            "x": self.map_vars["x"].get() or None,
            "y": self.map_vars["y"].get() or None,
            "value": self.map_vars["value"].get() or None,
            "upward": self.map_vars["upward"].get() or None,
            "line": self.map_vars["line"].get() or None,
            "sequence": self.map_vars["sequence"].get() or None,
            "unit": self.unit_var.get() or None,
            "stage": raw_stage(self.stage_var.get()) or None,
            "crs": self.crs_var.get().strip() or None,
            "generated_ids": bool(self.generated_var.get()),
            "id_policy": "skip" if self.skip_ids_var.get() else "strict",
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

    def _import_audit_summary(self):
        for entry in reversed(getattr(self.dataset, "history", [])):
            if entry["operation"] == "import_id_filter":
                audit = entry["parameters"]
                return (f"；导入跳过：空点号 {audit['blank_rows']} 行、后续重号 {audit['duplicate_rows']} 行"
                        "（重号保留首次，另存时附明细）")
        return ""

    def _show_dataset_info(self):
        data = self.dataset
        if data is None:
            self.dataset_label.configure(text="尚未导入数据")
            return
        from .gui_guidance import labeled
        self.dataset_label.configure(
            text=f"已保留 {len(data.values)} 行；{labeled(data.stage)}；"
                 f"原单位 {data.schema.unit}（当前统一为 nT）；坐标系{'已声明（见坐标检查）' if data.schema.crs else '未知，先做非空间检查'}" + self._import_audit_summary(),
            wraplength=790)

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
        for entry in reversed(self.dataset.history):
            if entry['operation'] == 'coordinate_check':
                self._coordinates_checked(entry['parameters'], False)
                break
        self._show_readable_report(result["report"])
        self._update_step_label()
        self.notebook.select(self.tab_qc)
        self._log("质控完成，报告已更新")

    def _show_readable_report(self, report):
        from .gui_guidance import magnetic_summary
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", magnetic_summary(self.dataset, report))

    def _show_detailed_report(self, report=None):
        report = self.last_report if report is None else report
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", json.dumps(report, ensure_ascii=False,
                                                  indent=2, default=str))

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
            self._draw_reference_overlay(axis)
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
        from .gui_guidance import raw_method
        parameters["method"] = raw_method(self.interp_method.get())
        if parameters["method"] != "equivalent_sources":
            for name in _EQS_ONLY:
                parameters.pop(name, None)
        self._run_operation("interpolate", parameters, self.dataset, self._grid_done)

    def _do_transform(self):
        if self.current_grid is None:
            return
        from .gui_guidance import raw_transform
        method = raw_transform(self.transform_method.get())
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

    def _report_with_coordinates(self, report):
        combined = dict(report or {})
        for entry in reversed(getattr(self.dataset, "history", [])):
            if entry["operation"] == "import_id_filter":
                combined['import_id_filter'] = entry['parameters']
                break
        if self.coordinate_report is not None:
            combined['coordinate_check'] = self.coordinate_report
        if getattr(self, 'overlay_report', None) is not None:
            combined['reference_overlay_check'] = self.overlay_report
        return combined

    def _do_export(self):
        target = self.export_var.get().strip()
        result, report = self.last_result, self._report_with_coordinates(self.last_report)
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
