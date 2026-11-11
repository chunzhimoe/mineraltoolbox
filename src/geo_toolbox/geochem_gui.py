from __future__ import annotations

import json

from .gui import MagneticWorkbench, _validation_error

_GEO_UNITS = ("", "ppm", "ppb", "%", "wt%", "mg/kg", "ug/g", "µg/g")
_DL_MODES = ("token_only", "fixed", "column")
_ZERO_POLICIES = ("invalid", "censored")
_IMPUTE_METHODS = ("", "multRepl", "lrEM", "lrEMplus")
_INI_COV = ("", "complete.obs", "multRepl")
_LOGRATIO_METHODS = ("", "clr", "ilr")
_VARIOGRAM_MODELS = ("", "spherical", "exponential", "gaussian")
_ESTIMATORS = ("", "matheron", "cressie")
_KRIGING_MODELS = ("", "spherical", "exponential", "gaussian")
_VALIDATE_SOURCES = ("原始导入", "当前浓度")


class GeochemWorkbench(MagneticWorkbench):
    def __init__(self, root):
        self.original_dataset = None
        self.features = None
        self.maps = None
        self.element_rows = {}
        self._results = {}
        super().__init__(root)
        root.title("GeoToolbox – 地球化学工作台")
        self.notebook.tab(self.tab_qc, text="3 检查与看图")

    def _task_api(self):
        from . import geochem_gui_tasks
        return geochem_gui_tasks

    def _transform_selected(self, event=None):
        pass

    def _interp_selected(self, event=None):
        pass

    def _split_toggled_state(self):
        pass

    def _build_import(self):
        tk, ttk = self.tk, self.ttk
        frame = self.tab_import
        src = ttk.LabelFrame(frame, text="源文件")
        src.pack(fill="x", padx=6, pady=4)
        self.path_var = tk.StringVar()
        self.sheet_var = tk.StringVar()
        self.header_var = tk.StringVar(value="1")
        self.header_rows_var = tk.StringVar(value="1")
        self.path_var.trace_add("write", self._path_changed)
        for var in (self.sheet_var, self.header_var, self.header_rows_var):
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
        ttk.Label(src, text="表头占几行").grid(row=1, column=4, sticky="e")
        self._input(ttk.Combobox(src, textvariable=self.header_rows_var,
                                 values=("1", "2"), width=4,
                                 state="readonly")).grid(row=1, column=5)
        src.columnconfigure(1, weight=1)

        ttk.Label(frame, text="先看列名是否读对。第一行若是标题，请调整列名行再读取。",
                  wraplength=740).pack(anchor="w", padx=8, pady=6)
        ttk.Label(frame, text="双行表头选2，逐列采用下行非空列名，否则采用上行；不自动识别单位。",
                  wraplength=740).pack(anchor="w", padx=8)
        confirm_outer, confirm = self._scroll_frame(self.tab_confirm, 760)
        confirm_outer.pack(fill="both", expand=True)
        ttk.Label(confirm, text="确认点号和坐标，再勾选要看的元素并填写各自单位。不会自动填补或删除原值。",
                  wraplength=740).pack(anchor="w", padx=6, pady=8)
        mapping = ttk.LabelFrame(confirm, text="确认点号与坐标")
        mapping.pack(fill="x", padx=6, pady=4)
        self.map_vars = {}
        self.map_combos = {}
        for i, (key, label) in enumerate((("sample_id", "点号列*"), ("x", "X列*"),
                                          ("y", "Y列*"))):
            ttk.Label(mapping, text=label).grid(row=0, column=i * 2, sticky="e")
            var = tk.StringVar()
            var.trace_add("write", self._mapping_changed)
            combo = self._input(ttk.Combobox(mapping, textvariable=var, width=10,
                                             state="readonly"))
            combo.grid(row=0, column=i * 2 + 1, padx=2)
            combo.bind("<<ComboboxSelected>>", self._mapping_changed)
            self.map_vars[key] = var
            self.map_combos[key] = combo
        self.skip_ids_var = tk.BooleanVar(value=True)
        self._input(ttk.Checkbutton(mapping, text="跳过空点号；重复点号只保留首次（记录跳过明细）",
                                    variable=self.skip_ids_var, command=self._mapping_changed)).grid(
            row=1, column=0, columnspan=6, sticky="w", padx=4)
        optional = self._details_section(confirm, "补充坐标与零值说明（不知道可跳过）")
        ttk.Label(optional, text="测点坐标系（未知留空）").grid(row=0, column=0, sticky="e")
        self.crs_var = tk.StringVar()
        self.crs_var.trace_add("write", self._mapping_changed)
        self._input(ttk.Entry(optional, textvariable=self.crs_var, width=22)).grid(
            row=0, column=1, padx=6, sticky="w")
        self.generated_var = tk.BooleanVar(value=False)
        self._input(ttk.Checkbutton(optional, text="改用行号编号（优先于跳过选项，保留所有行及原点号）", variable=self.generated_var,
                                    command=self._mapping_changed)).grid(row=1, column=0,
                                                                        columnspan=2,
                                                                        sticky="w")
        ttk.Label(optional, text="零值含义").grid(row=2, column=0, sticky="e")
        self.zero_var = tk.StringVar(value="待复核（不能直接填补）")
        self.zero_var.trace_add("write", self._mapping_changed)
        self.zero_combo = self._input(
            ttk.Combobox(optional, textvariable=self.zero_var, values=("待复核（不能直接填补）", "低于检出限（需明确限值）"),
                         width=30, state="readonly"))
        self.zero_combo.grid(row=2, column=1, sticky="w")
        self.zero_combo.bind("<<ComboboxSelected>>", self._mapping_changed)
        ttk.Label(optional, foreground="#8a5a00", wraplength=600, justify="left",
                  text="坐标系不知道可留空。零值先保留为待复核，只有报告明确检出限时才能按低于限值处理。").grid(
            row=3, column=0, columnspan=4, sticky="w")

        self.element_frame = ttk.LabelFrame(confirm, text="勾选要查看的元素，并确认单位")
        self.element_frame.pack(fill="x", padx=6, pady=4)

        self.import_btn = self._input(
            ttk.Button(confirm, text="确认这些列，开始检查", command=self._do_import))
        self.import_btn.pack(anchor="w", padx=6, pady=4)

        table = ttk.Frame(frame)
        table.pack(fill="both", expand=True, padx=6, pady=4)
        self.tree = ttk.Treeview(table, show="headings", height=8)
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
        self.dataset_label = ttk.Label(info, text="尚未导入数据", wraplength=900,
                                       justify="left")
        self.dataset_label.pack(side="left")

        controls = ttk.Frame(frame)
        controls.pack(fill="x", padx=6, pady=2)
        basic = ttk.Frame(controls)
        basic.pack(fill="x", pady=2)
        self.check_btn = self._input(ttk.Button(basic, text="质控检查",
                                                command=self._do_check))
        self.check_btn.pack(side="left", padx=4)
        ttk.Label(basic, text="散点元素").pack(side="left")
        self.scatter_element = self._input(
            ttk.Combobox(basic, values=[], width=10, state="readonly"))
        self.scatter_element.pack(side="left", padx=2)
        self.scatter_element.bind("<<ComboboxSelected>>",
                                  lambda event: self._draw_scatter())
        self.reset_btn = self._input(ttk.Button(basic, text="恢复原导入",
                                                command=self._restore_original))
        self.reset_btn.pack(side="left", padx=8)
        self._input(ttk.Button(basic, text="检查摘要",
                               command=lambda: self._show_readable_report(self.last_report or {}))).pack(side="left", padx=4)
        self._input(ttk.Button(basic, text="详细记录",
                               command=self._show_detailed_report)).pack(
            side="left", padx=4)
        self._input(ttk.Button(basic, text="导入SHP底图核对",
                               command=self._choose_reference_overlay)).pack(
            side="left", padx=4)

        self._input(ttk.Button(basic, text="后续处理（可跳过）", command=self._open_processing)).pack(side="left", padx=4)
        impute = self._details_section(controls, "需要时再处理低于检出限／缺失值（不会自动执行）")
        ttk.Label(impute, text="方法*").grid(row=0, column=0, sticky="e")
        self.impute_method = self._input(
            ttk.Combobox(impute, values=_IMPUTE_METHODS, width=10, state="readonly"))
        self.impute_method.grid(row=0, column=1, padx=2, sticky="w")
        self.impute_vars = {}
        for i, (name, label, default) in enumerate((("frac", "frac", "0.65"),
                                                    ("tolerance", "容差", "0.0001"))):
            ttk.Label(impute, text=label).grid(row=0, column=2 + i * 2, sticky="e")
            var = tk.StringVar(value=default)
            self._input(ttk.Entry(impute, textvariable=var, width=8)).grid(
                row=0, column=3 + i * 2, padx=2)
            self.impute_vars[name] = var
        ttk.Label(impute, text="最大迭代").grid(row=1, column=0, sticky="e")
        self.impute_vars["max_iter"] = tk.StringVar(value="50")
        self._input(ttk.Entry(impute, textvariable=self.impute_vars["max_iter"],
                              width=8)).grid(row=1, column=1, padx=2, sticky="w")
        ttk.Label(impute, text="ini_cov").grid(row=1, column=2, sticky="e")
        self.ini_cov = self._input(
            ttk.Combobox(impute, values=_INI_COV, width=12, state="readonly"))
        self.ini_cov.set("complete.obs")
        self.ini_cov.grid(row=1, column=3, padx=2, sticky="w")
        self.impute_btn = self._input(ttk.Button(impute, text="执行填补",
                                               command=self._do_impute))
        self.impute_btn.grid(row=1, column=4, padx=4)
        ttk.Label(impute, foreground="#8a5a00",
                  text="填补拟合整个批次，不作独立留出验证输入；失败不回退").grid(
            row=1, column=5, columnspan=3, sticky="w")

        transform = self._details_section(controls, "需要元素组合特征时再打开（保留原浓度）")
        ttk.Label(transform, text="方法*").grid(row=0, column=0, sticky="e")
        self.logratio_method = self._input(
            ttk.Combobox(transform, values=_LOGRATIO_METHODS, width=8, state="readonly"))
        self.logratio_method.grid(row=0, column=1, padx=2, sticky="w")
        self.transform_btn = self._input(ttk.Button(transform, text="执行变换",
                                                  command=self._do_transform))
        self.transform_btn.grid(row=0, column=2, padx=4)
        ttk.Label(transform, foreground="#8a5a00", wraplength=500, justify="left",
                  text="ILR默认正交基写入metadata，不提供自定义basis；特征保留，浓度不替换").grid(
            row=0, column=3, sticky="w")

        pane = ttk.PanedWindow(frame, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.LabelFrame(pane, text="报告 / 特征预览")
        right = ttk.LabelFrame(pane, text="浓度散点预览（ppm）")
        bottom = ttk.Frame(left)
        bottom.pack(fill="x", side="bottom")
        self.features_label = ttk.Label(bottom, text="特征：无")
        self.features_label.pack(anchor="w")
        self.features_tree = ttk.Treeview(bottom, show="tree headings", height=5)
        features_vbar = ttk.Scrollbar(bottom, orient="vertical",
                                    command=self.features_tree.yview)
        features_hbar = ttk.Scrollbar(bottom, orient="horizontal",
                                    command=self.features_tree.xview)
        self.features_tree.configure(yscrollcommand=features_vbar.set,
                                     xscrollcommand=features_hbar.set)
        features_vbar.pack(side="right", fill="y")
        features_hbar.pack(side="bottom", fill="x")
        self.features_tree.pack(fill="x")
        self.report_text = tk.Text(left, wrap="word", width=40)
        report_bar = ttk.Scrollbar(left, orient="vertical", command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_bar.set)
        report_bar.pack(side="right", fill="y")
        self.report_text.pack(side="left", fill="both", expand=True)
        self.scatter_holder = ttk.Frame(right)
        self.scatter_holder.pack(fill="both", expand=True)
        pane.add(left, weight=3)
        pane.add(right, weight=2)

    def _build_grid(self):
        tk, ttk = self.tk, self.ttk
        frame = self.tab_grid
        pane = ttk.PanedWindow(frame, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=4)
        left_outer, left = self._scroll_frame(pane, 500)
        right = ttk.Frame(pane)
        pane.add(left_outer, weight=0)
        pane.add(right, weight=1)

        taskbar = ttk.Frame(left)
        taskbar.pack(fill="x", pady=2)
        ttk.Label(taskbar, text="任务：").pack(side="left")
        self.spatial_task = ttk.Combobox(
            taskbar, width=22, state="readonly",
            values=("检查空间相关性（变异函数）", "生成元素分布网格",
                    "验证预测误差（留出验证）"))
        self.spatial_task.set("检查空间相关性（变异函数）")
        self.spatial_task.pack(side="left", padx=4)
        self.spatial_task.bind("<<ComboboxSelected>>", self._spatial_task_changed)

        variogram = self._variogram_frame = ttk.LabelFrame(left, text="变异函数（当前浓度）")
        variogram.pack(fill="x", pady=4)
        self.variogram_vars = {}
        ttk.Label(variogram, text="元素*").grid(row=0, column=0, sticky="e")
        self.variogram_element = self._input(
            ttk.Combobox(variogram, values=[], width=12, state="readonly"))
        self.variogram_element.grid(row=0, column=1, padx=2, sticky="w")
        ttk.Label(variogram, text="模型*").grid(row=0, column=2, sticky="e")
        self.variogram_model = self._input(
            ttk.Combobox(variogram, values=_VARIOGRAM_MODELS, width=12, state="readonly"))
        self.variogram_model.grid(row=0, column=3, padx=2, sticky="w")
        ttk.Label(variogram, text="估计*").grid(row=1, column=0, sticky="e")
        self.variogram_estimator = self._input(
            ttk.Combobox(variogram, values=_ESTIMATORS, width=12, state="readonly"))
        self.variogram_estimator.grid(row=1, column=1, padx=2, sticky="w")
        ttk.Label(variogram, text="分箱边界m").grid(row=1, column=2, sticky="e")
        self.bin_edges_var = tk.StringVar()
        self._input(ttk.Entry(variogram, textvariable=self.bin_edges_var, width=14)).grid(
            row=1, column=3, padx=2, sticky="w")
        self.variogram_btn = self._input(ttk.Button(variogram, text="执行变异函数",
                                                    command=self._do_variogram))
        self.variogram_btn.grid(row=2, column=0, columnspan=2, padx=4, pady=4)
        ttk.Label(variogram, text="估计/边界共用于插值和验证；填边界时分箱数留空",
                  wraplength=240).grid(row=2, column=2, columnspan=2, sticky="w")

        interp = self._interp_frame = ttk.LabelFrame(
            left, text="克里金插值（当前浓度，支持栅格必需）")
        interp.pack(fill="x", pady=4)
        self.interp_vars = {}
        self.interp_entries = {}
        for i, (name, label) in enumerate((("xmin", "x最小"), ("ymin", "y最小"),
                                          ("xmax", "x最大"), ("ymax", "y最大"),
                                          ("resolution", "间距m*"),
                                          ("n_closest_points", "邻近点数"))):
            row, col = divmod(i, 2)
            ttk.Label(interp, text=label).grid(row=row, column=col * 2, sticky="e")
            var = tk.StringVar()
            entry = self._input(ttk.Entry(interp, textvariable=var, width=9))
            entry.grid(row=row, column=col * 2 + 1, padx=2, pady=1, sticky="w")
            self.interp_vars[name] = var
            self.interp_entries[name] = entry
        ttk.Label(interp, text="支持GeoTIFF*").grid(row=3, column=0, sticky="e")
        self.support_var = tk.StringVar()
        self._input(ttk.Entry(interp, textvariable=self.support_var, width=24)).grid(
            row=3, column=1, columnspan=2, padx=2, sticky="w")
        self._input(ttk.Button(interp, text="…", width=3,
                               command=self._browse_support)).grid(row=3, column=3)
        ttk.Label(interp, text="克里金模型*").grid(row=4, column=0, sticky="e")
        self.kriging_variogram = self._input(
            ttk.Combobox(interp, values=_KRIGING_MODELS, width=12, state="readonly"))
        self.kriging_variogram.grid(row=4, column=1, padx=2, sticky="w")
        ttk.Label(interp, text="分箱数(空=自动)").grid(row=4, column=2, sticky="e")
        self.nlags_var = tk.StringVar()
        self._input(ttk.Entry(interp, textvariable=self.nlags_var, width=6)).grid(
            row=4, column=3, padx=2, sticky="w")
        self.exact_var = tk.BooleanVar(value=True)
        self._input(ttk.Checkbutton(interp, text="exact_values", variable=self.exact_var)).grid(
            row=5, column=0, columnspan=2, sticky="w")
        self.pseudo_var = tk.BooleanVar(value=True)
        self._input(ttk.Checkbutton(interp, text="pseudo_inv", variable=self.pseudo_var)).grid(
            row=5, column=2, columnspan=2, sticky="w")
        ttk.Label(interp, text="分批像元").grid(row=6, column=0, sticky="e")
        self.chunk_var = tk.StringVar(value="10000")
        self._input(ttk.Entry(interp, textvariable=self.chunk_var, width=9)).grid(
            row=6, column=1, padx=2, sticky="w")
        self.interp_btn = self._input(ttk.Button(interp, text="执行插值",
                                               command=self._do_interpolate))
        self.interp_btn.grid(row=6, column=2, columnspan=2, padx=4)
        self.auto_bounds_var = tk.BooleanVar(value=True)
        self._input(ttk.Checkbutton(
            interp, text="范围自动取自测点，并按间距向外对齐",
            variable=self.auto_bounds_var,
            command=self._auto_bounds_changed)).grid(
            row=7, column=0, columnspan=4, sticky="w", pady=2)
        self.interp_vars["resolution"].trace_add(
            "write", self._refresh_auto_bounds)
        self._auto_bounds_changed()

        validate = self._validate_frame = ttk.LabelFrame(left, text="空间留出验证")
        validate.pack(fill="x", pady=4)
        ttk.Label(validate, text="数据分支*").grid(row=0, column=0, sticky="e")
        self.validate_source = self._input(
            ttk.Combobox(validate, values=_VALIDATE_SOURCES, width=12, state="readonly"))
        self.validate_source.set("原始导入")
        self.validate_source.grid(row=0, column=1, padx=2, sticky="w")
        ttk.Label(validate, text="分箱数(空=自动)").grid(row=0, column=2, sticky="e")
        self.validate_nlags = tk.StringVar()
        self._input(ttk.Entry(validate, textvariable=self.validate_nlags, width=6)).grid(
            row=0, column=3, padx=2, sticky="w")
        ttk.Label(validate, text="克里金模型*").grid(row=1, column=0, sticky="e")
        self.validate_variogram = self._input(
            ttk.Combobox(validate, values=_KRIGING_MODELS, width=12, state="readonly"))
        self.validate_variogram.grid(row=1, column=1, columnspan=3, padx=2, sticky="w")
        ttk.Label(validate, text="训练ID（每行一个完整点号）").grid(
            row=2, column=0, columnspan=4, sticky="w")
        self.train_text = self._input(tk.Text(validate, width=35, height=4))
        self.train_text.grid(row=3, column=0, columnspan=4, padx=2, sticky="w")
        ttk.Label(validate, text="测试ID（每行一个完整点号）").grid(
            row=4, column=0, columnspan=4, sticky="w")
        self.test_text = self._input(tk.Text(validate, width=35, height=4))
        self.test_text.grid(row=5, column=0, columnspan=4, padx=2, sticky="w")
        self.validate_btn = self._input(ttk.Button(validate, text="执行验证",
                                                  command=self._do_validate))
        self.validate_btn.grid(row=6, column=0, padx=4, pady=4, sticky="w")

        self.grid_label = ttk.Label(right, text="当前网格：无", wraplength=420)
        self.grid_label.pack(anchor="w", pady=2)
        mapbar = ttk.Frame(right)
        mapbar.pack(fill="x")
        ttk.Label(mapbar, text="预览").pack(side="left")
        self.preview_kind = self._input(
            ttk.Combobox(mapbar, values=("网格", "变异函数"), width=8, state="readonly"))
        self.preview_kind.set("网格")
        self.preview_kind.pack(side="left", padx=2)
        self.preview_kind.bind("<<ComboboxSelected>>",
                               lambda event: self._draw_grid())
        ttk.Label(mapbar, text="元素").pack(side="left")
        self.map_element = self._input(
            ttk.Combobox(mapbar, values=[], width=9, state="readonly"))
        self.map_element.pack(side="left", padx=2)
        ttk.Label(mapbar, text="图层").pack(side="left")
        self.map_layer = self._input(
            ttk.Combobox(mapbar, values=("prediction", "variance"), width=10,
                         state="readonly"))
        self.map_layer.set("prediction")
        self.map_layer.pack(side="left", padx=2)
        self.map_element.bind("<<ComboboxSelected>>",
                              lambda event: self._draw_grid())
        self.map_layer.bind("<<ComboboxSelected>>",
                            lambda event: self._draw_grid())
        self.grid_holder = ttk.Frame(right)
        self.grid_holder.pack(fill="both", expand=True)
        self.variogram_report = None
        self._spatial_task_changed()

    def _spatial_task_changed(self, event=None):
        task = self.spatial_task.get()
        visibility = {"检查空间相关性（变异函数）": self._variogram_frame,
                      "生成元素分布网格": self._interp_frame,
                      "验证预测误差（留出验证）": self._validate_frame}
        for frame in (self._variogram_frame, self._interp_frame, self._validate_frame):
            frame.pack_forget()
        visibility.get(task, self._variogram_frame).pack(fill="x", pady=4)

    def _browse_support(self):
        name = self.filedialog.askopenfilename(
            filetypes=[("GeoTIFF", ".tif .tiff"), ("全部", "*.*")])
        if name:
            self.support_var.set(name)

    def _source_changed(self, *args):
        if self._populating or self._busy:
            return
        self._clear_element_rows()
        super()._source_changed()

    def _invalidate(self):
        self.original_dataset = None
        self.features = None
        self.maps = None
        self._results = {}
        self._reports = {}
        self.variogram_report = None
        if hasattr(self, "features_label"):
            self.features_label.configure(text="特征：无")
            self.features_tree.delete(*self.features_tree.get_children())
        if hasattr(self, "export_info"):
            self.export_info.configure(text="无")
        super()._invalidate()

    def _clear_element_rows(self):
        for row in self.element_rows.values():
            for widget in row["widgets"]:
                if widget in self._inputs:
                    self._inputs.remove(widget)
                widget.destroy()
        self.element_rows = {}

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
                self.map_combos[key].configure(values=list(columns))
            from .gui_guidance import suggest_columns
            for role, column in suggest_columns(columns, self.map_vars).items():
                if not self.map_vars[role].get():
                    self.map_vars[role].set(column)
            self._build_element_rows(columns)
        finally:
            self._populating = False
        self._refresh_actions()
        self._coordinate_columns()
        self._update_step_label()
        self._log(f"预览完成: {preview['source']} 列={len(columns)} "
                  f"全量行数={len(preview['frame'])}")

    def _build_element_rows(self, columns):
        tk, ttk = self.tk, self.ttk
        previous = self.element_rows
        self._clear_element_rows()
        for widget in self.element_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.element_frame, text="选用").grid(row=0, column=0)
        ttk.Label(self.element_frame, text="列名").grid(row=0, column=1)
        ttk.Label(self.element_frame, text="单位*").grid(row=0, column=2)
        ttk.Label(self.element_frame, text="检出限模式").grid(row=0, column=3)
        ttk.Label(self.element_frame, text="固定值 / 限值列").grid(row=0, column=4)
        ttk.Label(self.element_frame, text="顺序=预览列顺序，首个全观测元素为ALR分母",
                  foreground="#8a5a00").grid(row=0, column=5, sticky="w")
        for index, column in enumerate(columns, start=1):
            prior = previous.get(column)
            use_var = tk.BooleanVar(value=prior["use"].get() if prior else False)
            unit_var = tk.StringVar(value=prior["unit"].get() if prior else "")
            mode_var = tk.StringVar(value=prior["mode"].get() if prior else "token_only")
            fixed_var = tk.StringVar(value=prior["fixed"].get() if prior else "")
            column_var = tk.StringVar(value=prior["limit_column"].get() if prior else "")
            row = {"use": use_var, "unit": unit_var, "mode": mode_var,
                   "fixed": fixed_var, "limit_column": column_var, "widgets": []}
            use_check = self._input(ttk.Checkbutton(
                self.element_frame, variable=use_var, command=self._mapping_changed))
            use_check.grid(row=index, column=0)
            row["widgets"].append(use_check)
            ttk.Label(self.element_frame, text=column).grid(row=index, column=1,
                                                            sticky="w")
            unit_combo = self._input(ttk.Combobox(self.element_frame, textvariable=unit_var,
                                                  values=_GEO_UNITS, width=8,
                                                  state="readonly"))
            unit_combo.grid(row=index, column=2, padx=2)
            unit_combo.bind("<<ComboboxSelected>>", self._mapping_changed)
            unit_var.trace_add("write", self._mapping_changed)
            row["widgets"].append(unit_combo)
            mode_combo = self._input(ttk.Combobox(self.element_frame, textvariable=mode_var,
                                                  values=_DL_MODES, width=11,
                                                  state="readonly"))
            mode_combo.grid(row=index, column=3, padx=2)
            mode_combo.bind("<<ComboboxSelected>>",
                            lambda event, c=column: self._dl_mode_changed(c))
            row["widgets"].append(mode_combo)
            value_entry = self._input(ttk.Entry(self.element_frame, textvariable=fixed_var,
                                                width=8))
            column_combo = self._input(ttk.Combobox(self.element_frame,
                                                    textvariable=column_var,
                                                    values=list(columns), width=8,
                                                    state="readonly"))
            use_var.trace_add("write", self._mapping_changed)
            mode_var.trace_add("write", self._mapping_changed)
            fixed_var.trace_add("write", self._mapping_changed)
            column_var.trace_add("write", self._mapping_changed)
            column_combo.bind("<<ComboboxSelected>>", self._mapping_changed)
            row["value_entry"], row["column_combo"] = value_entry, column_combo
            row["widgets"] += [value_entry, column_combo]
            self.element_rows[column] = row
            self._dl_mode_changed(column)

    def _dl_mode_changed(self, column):
        row = self.element_rows[column]
        mode = row["mode"].get()
        row["value_entry"].grid_forget()
        row["column_combo"].grid_forget()
        index = list(self.element_rows).index(column) + 1
        if mode == "fixed":
            row["value_entry"].grid(row=index, column=4, padx=2)
        elif mode == "column":
            row["column_combo"].grid(row=index, column=4, padx=2)
        self._mapping_changed()

    def _do_import(self):
        if self.preview is None:
            return
        preview = self.preview
        from .gui_guidance import require_fields
        missing = require_fields(
            {"sample_id": self.map_vars["sample_id"].get(), "x": self.map_vars["x"].get(),
             "y": self.map_vars["y"].get()},
            [("sample_id", "点号列"), ("x", "X列"), ("y", "Y列")])
        if missing:
            self.messagebox.showwarning("缺少信息", missing)
            self._log(f"导入未提交: {missing}")
            return
        if not any(row["use"].get() for row in self.element_rows.values()):
            self.messagebox.showwarning("缺少信息", "请至少勾选一个元素列并填写单位")
            self._log("导入未提交: 未选择元素")
            return
        elements, units, limits, limit_columns, modes = [], {}, {}, {}, {}
        for column, row in self.element_rows.items():
            if not row["use"].get():
                continue
            elements.append(column)
            mode = row["mode"].get()
            modes[column] = mode
            if not row["unit"].get():
                self._fail(_validation_error(f"{column} 已选择为元素但未填写单位"))
                return
            units[column] = row["unit"].get()
            if mode == "fixed" and not row["fixed"].get().strip():
                self._fail(_validation_error(f"{column} 检出限模式fixed但固定值为空"))
                return
            if mode == "column" and not row["limit_column"].get():
                self._fail(_validation_error(f"{column} 检出限模式column但未选择限值列"))
                return
            if mode == "fixed":
                limits[column] = row["fixed"].get()
            elif mode == "column":
                limit_columns[column] = row["limit_column"].get()
        mapping = {
            "sample_id": self.map_vars["sample_id"].get() or None,
            "x": self.map_vars["x"].get() or None,
            "y": self.map_vars["y"].get() or None,
            "crs": self.crs_var.get().strip() or None,
            "generated_ids": bool(self.generated_var.get()),
            "id_policy": "skip" if self.skip_ids_var.get() else "strict",
            "elements": elements,
            "units": units,
            "detection_limits": limits,
            "detection_limit_columns": limit_columns,
            "detection_modes": modes,
            "zero_policy": {"待复核（不能直接填补）": "invalid", "低于检出限（需明确限值）": "censored"}.get(
                self.zero_var.get(), self.zero_var.get() or "invalid"),
        }
        if self._submit("导入…",
                        lambda: self._task_api().import_source(preview, mapping=mapping),
                        self._import_done):
            self._log("导入请求已提交")

    def _import_done(self, data):
        self._invalidate()
        self.original_dataset = data.copy()
        self.dataset = data
        for entry in reversed(data.history):
            if entry['operation'] == 'coordinate_check':
                self._coordinates_checked(entry['parameters'], False)
                break
        self.last_result = data
        self.last_report = {}
        self._results = {"原导入": self.original_dataset, "当前浓度": data}
        self._reports = {"原导入": {}, "当前浓度": {}}
        self.export_kind.set("最新结果")
        self._refresh_actions()
        self._show_dataset_info()
        self._populate_element_selectors()
        self._update_export_info()
        self._draw_scatter()
        self._log(f"导入完成: {len(data.values)}行 元素={list(data.schema.elements)} "
                  f"crs={data.crs or '未知'}")
        self._do_check()

    def _populate_element_selectors(self):
        elements = list(self.dataset.schema.elements) if self.dataset is not None else []
        for combo in (self.scatter_element, self.variogram_element, self.map_element):
            combo.configure(values=elements)
        if elements:
            if self.scatter_element.get() not in elements:
                self.scatter_element.set(elements[0])
            if self.variogram_element.get() not in elements:
                self.variogram_element.set(elements[0])
            if self.map_element.get() not in elements:
                self.map_element.set(elements[0])

    def _refresh_actions(self):
        if self._busy or not hasattr(self, "import_btn"):
            return
        self.import_btn.configure(state="normal" if self.preview is not None else "disabled")
        dataset = "normal" if self.dataset is not None else "disabled"
        for button in (self.check_btn, self.impute_btn, self.transform_btn,
                       self.variogram_btn, self.interp_btn, self.validate_btn,
                       self.reset_btn):
            button.configure(state=dataset)
        has_result = any(value is not None for value in self._results.values())
        self.export_btn.configure(state="normal" if has_result else "disabled")
        available = ["最新结果"] + [key for key in ("原导入", "当前浓度", "特征", "网格",
                                                  "最新报告") if self._results.get(key)]
        self.export_kind.configure(values=available)
        if self.export_kind.get() not in available:
            self.export_kind.set("最新结果")
        self.reset_btn.configure(
            state="normal" if self.original_dataset is not None else "disabled")
        self._refresh_navigation()

    def _show_dataset_info(self):
        data = self.dataset
        if data is None:
            self.dataset_label.configure(text="尚未导入数据")
            self._refresh_auto_bounds()
            return
        counts = {name: int((data.status == name).to_numpy().sum())
                  for name in ("observed", "missing", "censored", "invalid")}
        self.dataset_label.configure(
            text=f"保留 {len(data.values)} 行；元素 {','.join(data.schema.elements)}；"
                 f"缺失 {counts['missing']}、低于限值 {counts['censored']}、待复核 {counts['invalid']}；"
                 f"坐标系{'已声明（见坐标检查）' if data.crs else '未知，先做非空间检查'}" + self._import_audit_summary(),
            wraplength=790)
        self._refresh_auto_bounds()

    def _restore_original(self):
        if self.original_dataset is None:
            return
        proceed = self.messagebox.askokcancel(
            "恢复原导入", "将丢弃当前浓度、特征与网格等内存分支；不修改磁盘文件。继续？")
        if not proceed:
            return
        self.dataset = self.original_dataset.copy()
        self.features = None
        self.maps = None
        self.variogram_report = None
        self.preview_kind.set("网格")
        self._results = {"原导入": self.original_dataset, "当前浓度": self.dataset}
        self._reports = {"原导入": {}, "当前浓度": {}}
        self.export_kind.set("最新结果")
        self.last_result = self.dataset
        self.last_report = {}
        self.report_text.delete("1.0", "end")
        self._dispose_canvas(self.grid_holder, self._grid_canvas)
        self._grid_canvas = None
        self.grid_label.configure(text="当前网格：无")
        self.features_label.configure(text="特征：无")
        self.features_tree.delete(*self.features_tree.get_children())
        self._refresh_actions()
        self._update_export_info()
        self._show_dataset_info()
        self._draw_scatter()
        self._log("已恢复原导入分支")

    def _qc_done(self, result):
        self._operation_done(result)

    def _show_readable_report(self, report):
        from .gui_guidance import geochem_summary
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", geochem_summary(self.dataset, report))

    def _operation_done(self, result, operation=None):
        kind = result["kind"]
        data = result["data"]
        self.last_report = result["report"]
        self.last_result = data
        self._results["最新结果"] = data
        self._reports["最新结果"] = result["report"]
        if kind == "concentration":
            self.dataset = data
            self._results["当前浓度"] = data
            self._reports["当前浓度"] = result["report"]
            if operation == "impute":
                self.features = None
                self.maps = None
                self.variogram_report = None
                self.preview_kind.set("网格")
                for key in ("特征", "网格", "最新报告"):
                    self._results.pop(key, None)
                    self._reports.pop(key, None)
                self.grid_label.configure(text="当前网格：无")
                self._dispose_canvas(self.grid_holder, self._grid_canvas)
                self._grid_canvas = None
                self.features_label.configure(text="特征：无")
                self.features_tree.delete(*self.features_tree.get_children())
            self._show_dataset_info()
            self._draw_scatter()
        elif kind == "features":
            self.features = data
            self._results["特征"] = data
            self._reports["特征"] = result["report"]
            self._show_features(data)
        elif kind == "maps":
            self.maps = data
            self._results["网格"] = data
            self._reports["网格"] = result["report"]
            self.preview_kind.set("网格")
            self.grid_label.configure(
                text=f"当前网格：{data.grid.width}×{data.grid.height} "
                     f"元素={len(data.predictions)} 域={data.domain}")
            self._populate_element_selectors()
            self._draw_grid()
        else:
            if operation == "variogram":
                self.variogram_report = result["report"]
                self.preview_kind.set("变异函数")
                self._draw_grid()
            self._results["最新报告"] = data
            self._reports["最新报告"] = result["report"]
        if kind == 'concentration':
            self._show_readable_report(result['report'])
        else:
            descriptions = {'features': f'已生成组合特征 {data.values.shape[0]} 行 × {data.values.shape[1]} 列。' if kind == 'features' else '',
                            'maps': f'已生成 {len(data.predictions)} 个元素的 {data.grid.width} × {data.grid.height} 网格。' if kind == 'maps' else '',
                            'report': '已完成空间检查／预测验证。指标及参数请查看详细记录；不会替换当前浓度。'}
            self.report_text.delete('1.0', 'end')
            self.report_text.insert('end', descriptions.get(kind, descriptions['report']) + '\n原浓度和原文件保持不变；保存页可选择此次结果或原导入数据。')
        self._refresh_actions()
        self._update_export_info()
        self._update_step_label()
        if kind == 'concentration':
            self.notebook.select(self.tab_qc)
        filters = [entry["parameters"] for entry in
                   (result.get("report") or {}).get("history", [])
                   if entry.get("operation") == "spatial_coordinate_filter"]
        if filters:
            audit = filters[-1]
            self._log(f"空间输入过滤：无效XY {audit['invalid_rows']} 行、"
                      f"重复XY {audit['duplicate_rows']} 行，"
                      f"保留 {audit['retained_rows']} 行；原始数据未改")
        self._log(f"任务完成 kind={kind}")

    def _show_features(self, data):
        frame = data.values.head(100)
        self.features_tree.delete(*self.features_tree.get_children())
        columns = list(frame.columns)
        self.features_tree.configure(columns=columns)
        for name in columns:
            self.features_tree.heading(name, text=name)
            self.features_tree.column(name, width=80, stretch=False)
        self.features_tree.heading("#0", text="点号")
        self.features_tree.column("#0", width=90, stretch=False)
        for index, row in frame.iterrows():
            self.features_tree.insert("", "end", text=str(index),
                                      values=[f"{v:.4g}" for v in row])
        self.features_label.configure(
            text=f"特征：{len(data.values)}×{frame.shape[1]} domain={data.domain} "
                 f"(首100行)")

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
            element = self.scatter_element.get() or data.schema.elements[0]
            xy = data.coordinates[["x", "y"]].to_numpy(dtype=float)
            values = data.values[element].to_numpy(dtype=float)
            mask = np.isfinite(xy).all(axis=1) & np.isfinite(values)
            image = axis.scatter(xy[mask, 0], xy[mask, 1], s=8, c=values[mask])
            figure.colorbar(image, ax=axis, label="ppm")
            self._draw_reference_overlay(axis)
            axis.set_title(f"{element} (finite shown); CRS={data.crs or 'unknown'}")
        canvas = FigureCanvasTkAgg(figure, master=self.scatter_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._scatter_canvas = canvas

    def _draw_grid(self):
        maps = self.maps
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
        if self.preview_kind.get() == "变异函数" and self.variogram_report is not None:
            report = self.variogram_report
            distance = np.asarray(report.get("distance", []), dtype=float)
            gamma = np.asarray(report.get("semivariance", []), dtype=float)
            counts = np.asarray(report.get("pair_counts", []), dtype=float)
            keep = np.isfinite(distance) & np.isfinite(gamma)
            axis.scatter(distance[keep], gamma[keep], s=20)
            for x, y, n in zip(distance[keep], gamma[keep], counts[keep]):
                if np.isfinite(n):
                    axis.annotate(f"n={int(n)}", (x, y), textcoords="offset points",
                                  xytext=(4, 4), fontsize=7)
            axis.set_xlabel("distance")
            axis.set_ylabel("semivariance")
            axis.set_title("Empirical variogram (counts labeled)")
        elif maps is not None:
            element = self.map_element.get() or next(iter(maps.predictions), None)
            layer = self.map_layer.get() or "prediction"
            table = maps.predictions if layer == "prediction" else maps.variances
            if element in table:
                a, _, c, _, e, f = maps.grid.transform
                extent = (c, c + maps.grid.width * a,
                          f + maps.grid.height * e, f)
                masked = np.ma.masked_where(~maps.support, table[element])
                unit = maps.units.get(element, "") if layer == "prediction" else "ppm²"
                image = axis.imshow(masked, extent=extent, origin="upper", aspect="equal")
                figure.colorbar(image, ax=axis, label=unit)
                axis.set_title(f"{element} {layer} (masked outside support)")
        canvas = FigureCanvasTkAgg(figure, master=self.grid_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._grid_canvas = canvas

    def _do_impute(self):
        if self.dataset is None:
            return
        from .gui_guidance import raw_impute
        parameters = {"method": raw_impute(self.impute_method.get()),
                      "frac": self.impute_vars["frac"].get(),
                      "tolerance": self.impute_vars["tolerance"].get(),
                      "max_iter": self.impute_vars["max_iter"].get(),
                      "ini_cov": self.ini_cov.get()}
        self._run_operation("impute", parameters, self.dataset,
                            lambda result: self._operation_done(result, "impute"))

    def _do_transform(self):
        if self.dataset is not None:
            from .gui_guidance import raw_logratio
            parameters = {"method": raw_logratio(self.logratio_method.get())}
            self._run_operation("transform", parameters, self.dataset, self._operation_done)

    def _do_variogram(self):
        if self.dataset is None:
            return
        from .gui_guidance import raw_model
        parameters = {"element": self.variogram_element.get(),
                      "model": raw_model(self.variogram_model.get()),
                      "estimator": self.variogram_estimator.get(),
                      "bin_edges": self.bin_edges_var.get()}
        self._run_operation("variogram", parameters, self.dataset,
                            lambda result: self._operation_done(result, "variogram"))

    def _do_interpolate(self):
        if self.dataset is None:
            return
        from .gui_guidance import raw_model
        parameters = {"xmin": self.interp_vars["xmin"].get(),
                      "ymin": self.interp_vars["ymin"].get(),
                      "xmax": self.interp_vars["xmax"].get(),
                      "ymax": self.interp_vars["ymax"].get(),
                      "resolution": self.interp_vars["resolution"].get(),
                      "auto_bounds": bool(self.auto_bounds_var.get()),
                      "support_path": self.support_var.get(),
                      "variogram": raw_model(self.kriging_variogram.get()),
                      "nlags": self.nlags_var.get(),
                      "bin_edges": self.bin_edges_var.get(),
                      "estimator": self.variogram_estimator.get() or "matheron",
                      "exact_values": bool(self.exact_var.get()),
                      "pseudo_inv": bool(self.pseudo_var.get()),
                      "n_closest_points": self.interp_vars["n_closest_points"].get(),
                      "chunk_size": self.chunk_var.get()}
        self._run_operation("interpolate", parameters, self.dataset,
                            lambda result: self._operation_done(result, "interpolate"))

    def _do_validate(self):
        if self.dataset is None:
            return
        data = (self.original_dataset
                if self.validate_source.get() == "原始导入" else self.dataset)
        if data is None:
            self.messagebox.showwarning("验证", "所选分支不存在")
            return
        from .gui_guidance import raw_model
        parameters = {"variogram": raw_model(self.validate_variogram.get()),
                      "nlags": self.validate_nlags.get(),
                      "bin_edges": self.bin_edges_var.get(),
                      "estimator": self.variogram_estimator.get() or "matheron",
                      "train_ids": self.train_text.get("1.0", "end"),
                      "test_ids": self.test_text.get("1.0", "end")}
        self._run_operation("validate", parameters, data,
                            lambda result: self._operation_done(result, "validate"))

    def _update_export_info(self):
        selection = self.export_kind.get() or "最新结果"
        data = self.last_result if selection == "最新结果" else self._results.get(selection)
        if data is None:
            self.export_info.configure(text="无可用对象")
            return
        shape = getattr(getattr(data, "values", None), "shape", None)
        detail = f"{selection} domain={getattr(data, 'domain', 'report')}"
        if shape is not None:
            detail += f" {shape[0]}×{shape[1]}"
        self.export_info.configure(text=detail)

    def _do_export(self):
        target = self.export_var.get().strip()
        selection = self.export_kind.get() or "最新结果"
        if selection == "最新结果":
            data = self.last_result
            report = self.last_report
        else:
            data = self._results.get(selection)
            report = self._reports.get(selection, {})
        if data is None:
            self.messagebox.showerror("导出", f"{selection} 尚无结果，无法导出")
            return
        if not target:
            self.messagebox.showwarning("导出", "请填写新的导出目录")
            return
        self._submit("导出…",
                     lambda: self._task_api().export_result(
                         data, directory=target, report=self._report_with_coordinates(report)),
                     self._export_done)

    def _build_log(self):
        tk, ttk = self.tk, self.ttk
        frame = self.tab_log
        self.log_text = tk.Text(frame, wrap="word", height=18)
        log_bar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_bar.set)
        self.log_text.pack(fill="both", expand=True, padx=6, pady=4)
        log_bar.pack(side="right", fill="y")
        actions = ttk.LabelFrame(frame, text="另存检查结果／所选数据（原文件不动）")
        actions.pack(fill="x", padx=6, pady=8, before=self.log_text)
        ttk.Label(actions, text="保存哪份数据").grid(row=0, column=0, sticky="e")
        self.export_kind = self._input(
            ttk.Combobox(actions, values=("最新结果", "原导入", "当前浓度", "特征", "网格",
                                          "最新报告"),
                         width=10, state="readonly"))
        self.export_kind.set("最新结果")
        self.export_kind.grid(row=0, column=1, padx=2, sticky="w")
        self.export_kind.bind("<<ComboboxSelected>>",
                              lambda event: self._update_export_info())
        self.export_info = ttk.Label(actions, text="无")
        self.export_info.grid(row=0, column=2, columnspan=2, sticky="w", padx=4)
        ttk.Label(actions, text="导出目录(新)").grid(row=1, column=0, sticky="e")
        self.export_var = tk.StringVar()
        self._input(ttk.Entry(actions, textvariable=self.export_var, width=40)).grid(
            row=1, column=1, columnspan=3, padx=4, sticky="w")
        self._input(ttk.Button(actions, text="选择父目录…",
                               command=self._choose_export_parent)).grid(row=1, column=4)
        self.export_btn = self._input(ttk.Button(actions, text="导出当前结果",
                                                 command=self._do_export))
        self.export_btn.grid(row=1, column=5, padx=6)
        self._input(ttk.Button(actions, text="日志另存…", command=self._save_log)).grid(
            row=2, column=0, padx=4, pady=2)
        self._input(ttk.Button(actions, text="清空日志显示",
                               command=lambda: self.log_text.delete("1.0", "end"))).grid(
            row=2, column=1, sticky="w", pady=2)

    def start(self):
        super().start()


if __name__ == "__main__":
    from .gui import main
    main(["--module", "geochem"])
