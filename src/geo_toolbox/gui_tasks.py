from __future__ import annotations

import hashlib
import importlib.util
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj.exceptions import CRSError

from .core import GridSpec, record, write_json
from .dependencies import require
from .errors import BackendExecutionError, DataValidationError, MissingBackendError
from .magnetic import MagneticAPI, _metric, _usable
from .magnetic_core import MagneticDataset, MagneticGrid, MagneticSchema


def preview_source(source, *, sheet_name=0, header=0):
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise DataValidationError("输入文件不存在或不是普通文件")
    if type(header) is not int or header < 0:
        raise DataValidationError("表头行必须是正整数（界面从1开始计数）")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        sheets = []
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, header=header)
    elif suffix in {".xls", ".xlsx", ".xlsm"}:
        engine = "xlrd" if suffix == ".xls" else "openpyxl"
        require(engine)
        with pd.ExcelFile(path, engine=engine) as book:
            sheets = book.sheet_names
            if sheet_name is None or isinstance(sheet_name, (list, tuple, dict)):
                raise DataValidationError("请选择单个工作表")
            frame = pd.read_excel(book, sheet_name=sheet_name, header=header, dtype=str, keep_default_na=False)
    else:
        raise DataValidationError("仅支持 CSV / XLS / XLSX / XLSM")
    if frame.empty or not frame.columns.is_unique:
        raise DataValidationError("表格为空或列名重复")
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise DataValidationError("读取过程中源文件发生变化，请重新预览")
    return {"source": str(path), "source_sha256": source_hash, "sheets": sheets, "columns": frame.columns.tolist(),
            "frame": frame.copy(deep=True), "preview": frame.head(100).copy(deep=True),
            "header": header, "sheet_name": sheet_name}


def import_source(preview, *, mapping):
    frame = preview["frame"].copy(deep=True)
    generated = mapping.get("generated_ids", False)
    split = mapping.get("split_point_line", False)
    if type(generated) is not bool or type(split) is not bool:
        raise DataValidationError("审计ID和点线号拆分选项必须明确选择")
    sample_id = mapping.get("sample_id")
    derived = {}
    if generated:
        sample_id = "__gui_audit_id"
        if sample_id in frame:
            raise DataValidationError("保留列__gui_audit_id已存在，请更换源表列名")
        frame[sample_id] = [f"row{preview.get('header', 0) + 2 + i}" for i in range(len(frame))]
        derived["generated_id"] = "Excel/CSV row number; original identifiers retained, not silently repaired"
    line, sequence = mapping.get("line") or None, mapping.get("sequence") or None
    if split:
        column = mapping.get("point_line") or mapping.get("sample_id")
        if column not in frame:
            raise DataValidationError("拆分点线号需要选择原始点线号列")
        if "__gui_line" in frame or "__gui_sequence" in frame:
            raise DataValidationError("拆分点线号的保留列已存在")
        parsed = frame[column].astype(str).str.strip().str.extract(r"^(\d+)-(\d+)$")
        line, sequence = "__gui_line", "__gui_sequence"
        frame[line] = parsed[0]
        frame[sequence] = pd.to_numeric(parsed[1], errors="coerce")
        derived["point_line_split"] = {"column": column, "pattern": "digits-digits", "unparsed_rows_preserved": int(parsed.isna().any(axis=1).sum())}
    schema = MagneticSchema(sample_id=sample_id, x=mapping.get("x"), y=mapping.get("y"),
                            value=mapping.get("value"), crs=mapping.get("crs") or None,
                            unit=mapping.get("unit"), stage=mapping.get("stage"),
                            upward=mapping.get("upward") or None, line=line, sequence=sequence)
    data = MagneticAPI().read(frame, schema=schema)
    data.history.append(record("gui_import", "geo_toolbox", {"source": preview["source"],
                        "source_sha256": preview.get("source_sha256"),
                        "sheet_name": preview.get("sheet_name"), "header": preview.get("header"),
                        "mapping": dict(mapping), "derived_columns": derived}))
    return data


def _real(value, name, *, positive=False):
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        raise DataValidationError(f"请填写 {name}")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise DataValidationError(f"{name} 必须是数值") from error
    if not np.isfinite(number) or (positive and number <= 0):
        raise DataValidationError(f"{name} 必须有限" + ("且大于0" if positive else ""))
    return number


def _integer(value, name):
    number = _real(value, name)
    if number != int(number):
        raise DataValidationError(f"{name} 必须是整数")
    return int(number)


def _parameters(operation, parameters):
    p = dict(parameters)
    if operation == "check":
        return {}
    if operation == "spikes":
        return {"window": _integer(p.get("window"), "窗口点数"),
                "threshold_nT": _real(p.get("threshold_nT"), "筛查阈值nT", positive=True)}
    if operation == "crop":
        rows, columns = p.get("rows"), p.get("columns")
        if rows is None:
            rows = (p.get("row_start"), p.get("row_stop"))
        if columns is None:
            columns = (p.get("column_start"), p.get("column_stop"))
        return {"rows": tuple(_integer(v, "行索引") for v in rows),
                "columns": tuple(_integer(v, "列索引") for v in columns)}
    if operation == "interpolate":
        bounds = p.get("bounds") or [p.get(name) for name in ("xmin", "ymin", "xmax", "ymax")]
        result = {"bounds": [_real(v, "网格边界") for v in bounds],
                  "resolution": _real(p.get("resolution"), "网格间距m", positive=True),
                  "method": p.get("method"), "upward": _real(p.get("upward"), "目标平面高程m"),
                  "max_distance": _real(p.get("max_distance"), "支持半径m", positive=True)}
        for name in ("depth", "damping", "block_size"):
            result[name] = None if p.get(name) in (None, "") else _real(p[name], name, positive=True)
        return result
    if operation == "transform":
        result = {"method": p.get("method"), "padding": _integer(p.get("padding"), "扩展像元数"),
                  "trim": _integer(p.get("trim"), "裁边像元数")}
        for name in ("height_displacement", "azimuth", "inclination", "declination", "magnetization_inclination", "magnetization_declination", "max_gain"):
            if p.get(name) not in (None, ""):
                result[name] = _real(p[name], name)
        if "assume_induced" in p:
            result["assume_induced"] = p["assume_induced"]
        return result
    raise DataValidationError("不支持的界面任务")


def preflight(data, *, operation, parameters):
    errors, warnings = [], []
    try:
        if data is None:
            raise DataValidationError("请先导入样点或生成网格")
        p = _parameters(operation, parameters)
        if operation in {"check", "spikes", "interpolate"} and not isinstance(data, MagneticDataset):
            raise DataValidationError("该任务需要样点数据")
        if operation in {"crop", "transform"} and not isinstance(data, MagneticGrid):
            raise DataValidationError("该任务需要网格结果")
        if operation == "check":
            qc = MagneticAPI().check(data).diagnostics
            if qc["invalid_coordinate_ids"]:
                warnings.append("存在缺失/非法坐标；质控保留原行，空间处理将拒绝")
            if qc["duplicate_coordinate_ids"]:
                warnings.append("存在重复坐标；不自动平均或删除")
            if data.schema.crs is None:
                warnings.append("CRS未知：仅允许非空间质控，不生成定位成果")
        elif operation == "spikes":
            if p["window"] < 3 or p["window"] % 2 != 1:
                raise DataValidationError("窗口必须是大于等于3的奇数")
            if data.schema.line is None or data.schema.sequence is None:
                raise DataValidationError("筛查必须显式指定测线与点序列")
            lines = data.raw[data.schema.line]
            order = pd.to_numeric(data.raw[data.schema.sequence], errors="coerce")
            if lines.isna().any() or lines.astype(str).str.strip().eq("").any() or not np.isfinite(order).all():
                raise DataValidationError("存在空测线或无法解析的点序；原行保留，需明确复核后才能筛查")
            if pd.DataFrame({"line": lines, "sequence": order}).duplicated().any():
                raise DataValidationError("同一测线内点序重复，不能确定沿线顺序")
            if not np.isfinite(data.values).all():
                raise DataValidationError("存在缺失/非法磁值，请先复核")
            warnings.append("筛查只标记候选，不确认噪声，不自动去除异常")
        elif operation == "interpolate":
            coordinates, _ = _usable(data)
            grid = GridSpec.from_bounds(p["bounds"], p["resolution"], data.schema.crs)
            if grid.width * grid.height > 1_000_000:
                raise DataValidationError("本地GUI单次最多100万个网格像元，请减小范围或增大间距")
            if p["method"] not in {"equivalent_sources", "verde_spline"}:
                raise DataValidationError("请选择网格化方法")
            if not np.isfinite(coordinates[:, 2]).all():
                raise DataValidationError("空间网格化需要已确认的测量高程米坐标")
            if p["method"] == "equivalent_sources":
                if p["depth"] is None:
                    raise DataValidationError("等效源需要显式源深度")
                if p["upward"] < coordinates[:, 2].max():
                    raise DataValidationError("目标平面不得低于任一观测高程")
            else:
                if p["depth"] is not None or p["block_size"] is not None:
                    raise DataValidationError("二维样条不接受等效源深度或分块参数")
                if not np.allclose(coordinates[:, 2], p["upward"], rtol=0, atol=1e-8):
                    raise DataValidationError("二维样条要求所有观测与目标平面高程一致")
            if len(coordinates) ** 2 * 8 > 512 * 1024 ** 2:
                raise DataValidationError("本地GUI稠密拟合矩阵估算超过512MiB，请先设计分块/样点方案；不会自动抽样")
            warnings.append("请确认X为东坐标、Y为北坐标、坐标及高程单位为米，并已核实高程基准与现有高度改正；不能直接套用主图CRS")
            warnings.append("支持掩膜不是预测不确定度；大数据稠密拟合可能消耗大量内存")
        elif operation == "transform":
            methods = {"upward_continuation", "derivative_easting", "derivative_northing", "derivative_upward",
                       "directional_derivative", "total_gradient_amplitude", "tilt_angle", "reduction_to_pole"}
            if p["method"] not in methods:
                raise DataValidationError("请选择有效的场变换方法")
            if min(data.values.shape) < 4:
                raise DataValidationError("场变换每个轴至少需要4个像元")
            if p["method"] == "upward_continuation" and p.get("height_displacement", 0) <= 0:
                raise DataValidationError("向上延拓高度必须显式填写并大于0")
            if p["method"] == "directional_derivative" and "azimuth" not in p:
                raise DataValidationError("方向导数必须填写从北顺时针的方位角")
            if type(p.get("assume_induced", False)) is not bool:
                raise DataValidationError("感应磁化假设必须明确选择")
            if p.get("assume_induced", False) and p["method"] != "reduction_to_pole":
                raise DataValidationError("感应磁化选项仅适用于化极")
            accepted = {"upward_continuation": {"height_displacement"}, "directional_derivative": {"azimuth"},
                        "reduction_to_pole": {"inclination", "declination", "magnetization_inclination",
                                              "magnetization_declination", "max_gain"}}.get(p["method"], set())
            if set(p) - {"method", "padding", "trim", "assume_induced"} - accepted:
                raise DataValidationError("填写了与当前变换无关的参数")
            if (data.grid.width + 2 * p["padding"]) * (data.grid.height + 2 * p["padding"]) > 1_000_000:
                raise DataValidationError("本地GUI扩展后最多100万个像元，请减小padding或裁切")
            _metric(data.grid.crs)
            if data.unit != "nT" or data.stage not in {"anomaly", "rtp"} or data.upward is None:
                raise DataValidationError("需要nT异常场以及已知水平观测面")
            if not data.support.all() or not np.isfinite(data.values).all():
                raise DataValidationError("FFT要求完整有限支持矩形；请先显式裁切，不自动补洞")
            if p["padding"] < 0 or p["trim"] < 0 or 2 * p["trim"] >= min(data.values.shape):
                raise DataValidationError("扩展/裁边像元数无效")
            if p["method"] == "reduction_to_pole":
                if data.stage == "rtp":
                    raise DataValidationError("已化极场不能再次化极")
                for name in ("inclination", "declination", "max_gain"):
                    if name not in p:
                        raise DataValidationError(f"化极必须填写{name}")
                if not p.get("assume_induced", False) and any(name not in p for name in ("magnetization_inclination", "magnetization_declination")):
                    raise DataValidationError("请选择感应磁化假设或显式填写磁化方向")
                if p.get("assume_induced", False) and any(name in p for name in ("magnetization_inclination", "magnetization_declination")):
                    raise DataValidationError("感应磁化假设与显式磁化方向不能同时选择")
                for name, limit in (("inclination", 90), ("declination", 180),
                                    ("magnetization_inclination", 90), ("magnetization_declination", 180)):
                    if name in p and abs(p[name]) > limit:
                        raise DataValidationError(f"{name}超出角度范围±{limit}°")
                if p["max_gain"] <= 0:
                    raise DataValidationError("最大允许增益必须大于0")
            warnings.append("导数可能放大噪声；padding/trim不能保证消除边缘效应")
        elif operation == "crop":
            r0, r1 = p["rows"]
            c0, c1 = p["columns"]
            if not (0 <= r0 < r1 <= data.grid.height and 0 <= c0 < c1 <= data.grid.width):
                raise DataValidationError("裁切索引越界或范围为空")
        backend = "harmonica" if operation == "transform" or (operation == "interpolate" and p["method"] == "equivalent_sources") else "verde" if operation == "interpolate" else None
        if backend and importlib.util.find_spec(backend) is None:
            raise MissingBackendError(f"缺少可选后端{backend}，请安装magnetic extra")
    except (DataValidationError, CRSError, ValueError, TypeError) as error:
        errors.append(str(error))
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def execute(data, *, operation, parameters):
    result = preflight(data, operation=operation, parameters=parameters)
    if not result["ok"]:
        raise DataValidationError("\n".join(result["errors"]))
    p = _parameters(operation, parameters)
    api = MagneticAPI()
    if operation == "check":
        checked = api.check(data)
        return {"data": checked.data, "report": checked.diagnostics}
    if operation == "spikes":
        checked = api.flag_spikes(data, **p)
        return {"data": checked.data, "report": checked.diagnostics}
    if operation == "interpolate":
        grid = GridSpec.from_bounds(p.pop("bounds"), p.pop("resolution"), data.schema.crs)
        output = api.interpolate(data, grid=grid, **p)
    elif operation == "transform":
        output = api.transform(data, **p)
    else:
        output = api.crop(data, **p)
    return {"data": output, "report": {"stage": output.stage, "unit": output.unit,
                                        "grid": vars(output.grid), "upward": output.upward,
                                        "history": output.history, "warnings": result["warnings"]}}


def export_result(data, *, directory, report):
    if data is None:
        raise DataValidationError("没有可导出的结果")
    path = Path(directory).expanduser().resolve()
    if path.exists():
        raise DataValidationError("输出目录已存在：请选择新的子目录，不覆盖已有成果")
    if not path.parent.is_dir():
        raise DataValidationError("输出父目录不存在")
    output = data.export(path)
    write_json(output / "gui_report.json", report)
    return output


def describe_error(exc):
    if isinstance(exc, MissingBackendError):
        kind, title, hint = "dependency", "缺少可选依赖", "仅在tool-box隔离环境安装对应extra；不要修改全局Python。"
    elif isinstance(exc, (DataValidationError, CRSError, ValueError, KeyError, TypeError)):
        kind, title, hint = "validation", "输入或参数不符合要求", "检查列映射、单位、处理阶段、CRS和参数；不会自动删除或修复数据。"
    elif isinstance(exc, (PermissionError, FileNotFoundError, FileExistsError, OSError)):
        kind, title, hint = "io", "文件读取或写入失败", "确认文件可读、输出目录未占用且不覆盖原始数据。"
    elif isinstance(exc, BackendExecutionError):
        kind, title, hint = "backend", "计算后端执行失败", "保留日志并检查数据适用条件；不会静默切换算法。"
    else:
        kind, title, hint = "unexpected", "任务未完成", "请保留详细日志用于定位；之前的成功结果保持不变。"
    return {"kind": kind, "title": title, "message": str(exc), "hint": hint,
            "details": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}
