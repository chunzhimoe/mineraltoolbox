from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj.exceptions import CRSError

from .core import GeochemDataset, GeochemSchema, GridSpec, MapResult, record, write_json
from .errors import DataValidationError, MissingBackendError
from .geochem import GeochemistryAPI, _metric_crs, _require_concentrations, _usable
from .gui_tasks import _integer, _real, describe_error, preview_source


def import_source(preview, *, mapping):
    frame = preview["frame"].copy(deep=True)
    generated = mapping.get("generated_ids", False)
    if type(generated) is not bool:
        raise DataValidationError("生成审计ID必须明确选择")
    sample_id = mapping.get("sample_id")
    if generated:
        sample_id = "__gui_audit_id"
        if sample_id in frame:
            raise DataValidationError("保留列__gui_audit_id已存在")
        frame[sample_id] = [f"row{preview.get('header', 0) + 2 + i}" for i in range(len(frame))]
    elements = mapping.get("elements")
    if not isinstance(elements, (list, tuple)) or not elements:
        raise DataValidationError("至少选择一个元素列，顺序将用于填补与对数比变换")
    modes = mapping.get("detection_modes", {})
    if set(modes) - set(elements):
        raise DataValidationError("检出限模式包含未选择的元素")
    for element, mode in modes.items():
        if mode not in {"token_only", "fixed", "column"}:
            raise DataValidationError(f"{element}检出限模式无效")
        fixed = mapping.get("detection_limits", {})
        columns = mapping.get("detection_limit_columns", {})
        if mode == "fixed" and element not in fixed:
            raise DataValidationError(f"{element}必须填写固定检出限，不回退为从文本读取")
        if mode == "column" and (element not in columns or columns[element] not in frame):
            raise DataValidationError(f"{element}必须选择有效的检出限列，不自动回退")
        if (mode != "fixed" and element in fixed) or (mode != "column" and element in columns):
            raise DataValidationError(f"{element}检出限模式与配置不一致")
    limits = {name: _real(value, f"{name}检出限", positive=True)
              for name, value in mapping.get("detection_limits", {}).items()}
    schema = GeochemSchema(sample_id=sample_id, x=mapping.get("x"), y=mapping.get("y"),
                           crs=mapping.get("crs") or None, elements=elements,
                           units=mapping.get("units", {}), detection_limits=limits,
                           detection_limit_columns=mapping.get("detection_limit_columns", {}),
                           zero_policy=mapping.get("zero_policy", "invalid"))
    data = GeochemistryAPI().read(frame, schema=schema)
    data.history.append(record("geochem_gui_import", "geo_toolbox", {
        "source": preview["source"], "source_sha256": preview.get("source_sha256"),
        "sheet_name": preview.get("sheet_name"), "header": preview.get("header"),
        "mapping": dict(mapping), "element_order": list(schema.elements),
        "generated_ids": generated, "original_identifiers_preserved": True,
    }))
    return data


def _choice(value, choices, label):
    if value not in choices:
        raise DataValidationError(f"请选择{label}：{' / '.join(choices)}")
    return value


def _parameters(operation, parameters):
    p = dict(parameters)
    if operation == "check":
        return {}
    if operation == "impute":
        method = _choice(p.get("method"), ("multRepl", "lrEM", "lrEMplus"), "填补方法")
        frac = _real(p.get("frac"), "frac", positive=True)
        tolerance = _real(p.get("tolerance"), "收敛容差", positive=True)
        max_iter = _integer(p.get("max_iter"), "最大迭代次数")
        if frac >= 1 or not 1 <= max_iter <= 10000:
            raise DataValidationError("frac必须在(0,1)，本地GUI最大迭代次数为1至10000")
        return {"backend": "python", "method": method, "frac": frac, "tolerance": tolerance,
                "max_iter": max_iter,
                "ini_cov": _choice(p.get("ini_cov"), ("complete.obs", "multRepl"), "初始协方差方法")}
    if operation == "transform":
        return {"backend": "pyrolite", "method": _choice(p.get("method"), ("clr", "ilr"), "对数比方法")}
    if operation == "variogram":
        result = {"element": p.get("element"), "backend": "gstools",
                  "model": _choice(p.get("model"), ("spherical", "exponential", "gaussian"), "变异函数模型"),
                  "estimator": _choice(p.get("estimator"), ("matheron", "cressie"), "估计方法")}
        raw = p.get("bin_edges")
        if raw is not None and not (isinstance(raw, str) and not raw.strip()):
            parts = raw.split(",") if isinstance(raw, str) else raw
            edges = np.asarray([_real(v, "距离分箱边界m") for v in parts])
            if len(edges) < 4 or edges[0] < 0 or not (np.diff(edges) > 0).all():
                raise DataValidationError("距离分箱至少4个非负、严格递增边界（逗号分隔）")
            result["bin_edges"] = edges
        return result
    if operation in {"interpolate", "validate"}:
        result = {"variogram": _choice(p.get("variogram"), ("spherical", "exponential", "gaussian"), "克里金变异函数"),
                  "estimator": _choice(p.get("estimator") or "matheron", ("matheron", "cressie"), "估计方法"),
                  "nlags": None if p.get("nlags") in (None, "") else _integer(p.get("nlags"), "变异函数分箱数")}
        if result["nlags"] is not None and result["nlags"] < 3:
            raise DataValidationError("显式变异函数分箱数至少为3；留空使用GSTools默认")
        raw_edges = p.get("bin_edges")
        if raw_edges is not None and not (isinstance(raw_edges, str) and not raw_edges.strip()):
            parts = raw_edges.split(",") if isinstance(raw_edges, str) else raw_edges
            edges = np.asarray([_real(value, "距离分箱边界m") for value in parts])
            if len(edges) < 4 or edges[0] < 0 or not (np.diff(edges) > 0).all() or result["nlags"] is not None:
                raise DataValidationError("显式分箱边界需至少4个非负递增值，且分箱数应留空")
            result["bin_edges"] = edges
        if operation == "validate":
            for name in ("train_ids", "test_ids"):
                raw = p.get(name)
                if isinstance(raw, str):
                    raw = [line.strip() for line in raw.splitlines() if line.strip()]
                if not isinstance(raw, (list, tuple)):
                    raise DataValidationError("训练/测试ID必须逐行填写，每行一个完整点号")
                result[name] = list(raw)
            return result
        result["bounds"] = [_real(p.get(name), "网格边界m") for name in ("xmin", "ymin", "xmax", "ymax")]
        result["resolution"] = _real(p.get("resolution"), "网格间距m", positive=True)
        support = str(p.get("support_path") or "").strip()
        if not support:
            raise DataValidationError("必须显式提供与目标网格完全对齐的采样支持GeoTIFF，不能默认全域预测")
        result["support_path"] = str(Path(support).expanduser().resolve())
        result["chunk_size"] = _integer(p.get("chunk_size"), "预测分批像元数")
        if result["chunk_size"] < 1:
            raise DataValidationError("预测分批像元数必须为正整数")
        for name in ("exact_values", "pseudo_inv"):
            if type(p.get(name)) is not bool:
                raise DataValidationError(f"必须明确选择{name}")
            result[name] = p[name]
        raw = p.get("n_closest_points")
        result["n_closest_points"] = None if raw in (None, "") else _integer(raw, "邻近点数")
        return result
    raise DataValidationError("不支持的地化界面任务")


def _spatial(data):
    _metric_crs(data)
    for element in data.schema.elements:
        _usable(data, element)
    if len(data.values) ** 2 * 8 > 512 * 1024 ** 2:
        raise DataValidationError("本地GUI稠密矩阵估算超过512MiB；请先设计计算方案，不自动抽样")


def preflight(data, *, operation, parameters):
    errors, warnings = [], []
    try:
        if not isinstance(data, GeochemDataset):
            raise DataValidationError("该任务需要地化样点数据，请先导入")
        _require_concentrations(data)
        p = _parameters(operation, parameters)
        api = GeochemistryAPI()
        if operation == "check":
            qc = api.check(data).diagnostics
            if data.crs is None:
                warnings.append("CRS未知：可以非空间质控，不能变异函数/插值/空间留出验证")
            if qc["invalid_coordinate_ids"] or qc["duplicate_coordinate_ids"]:
                warnings.append("存在非法/缺失或重复坐标；保留原行，空间处理将拒绝")
            if qc["unresolved_cells"]:
                warnings.append("存在未解决的删失、缺失或非法值；不自动填补/删除/替换")
        elif operation == "impute":
            if data.imputed.to_numpy().any():
                raise DataValidationError("已执行填补，不能重复填补；请从原始导入重新开始")
            if (data.status == "invalid").to_numpy().any():
                raise DataValidationError("非法值不是普通缺失，必须先明确复核，不能直接填补")
            if (data.status != "observed").to_numpy().any():
                from .python_impute import _open_concentrations, _validate
                x, _, observed, _, censored = _validate(data.values.to_numpy(), data.detection_limits.to_numpy(),
                    data.status.to_numpy(), p["method"], p["frac"], p["tolerance"], p["max_iter"], p["ini_cov"])
                _open_concentrations(x)
                if p["method"] != "multRepl":
                    if len(x) <= x.shape[1] or not observed.all(axis=0).any() or (observed.sum(axis=1) < 2).any():
                        raise DataValidationError("EM需要样本数大于元素数、至少一个全观测分母列且每行至少两个观测元素")
                    if p["ini_cov"] == "complete.obs" and observed.all(axis=1).sum() < x.shape[1]:
                        raise DataValidationError("EM完整观测行不足以初始化协方差；不自动切换初始化")
            warnings.append("Python文献ALR-EM不宣称与原版R数值等价；只支持非闭合浓度、非稳健估计，失败不回退")
            warnings.append("填补拟合整个输入批次，不是可复用归纳模型；填补数据不能作为独立留出验证输入")
        elif operation == "transform":
            if len(data.schema.elements) < 2:
                raise DataValidationError("CLR/ILR至少需要两个元素")
            if ((data.status != "observed") & ~data.imputed).to_numpy().any() or not np.all(np.isfinite(data.values) & (data.values > 0)):
                raise DataValidationError("先解决全部缺失、删失及非法值，再执行对数比变换")
            warnings.append("CLR/ILR为无量纲特征，移除共同倍增富集；原浓度单独保留，不能把特征当浓度插值")
        else:
            _spatial(data)
            if operation == "variogram":
                if p["element"] not in data.schema.elements:
                    raise DataValidationError("请选择已导入的元素")
                warnings.append("变异函数拟合不是预测验证；首版GUI提供各向同性估计")
            elif operation == "interpolate":
                grid = GridSpec.from_bounds(p["bounds"], p["resolution"], data.crs)
                if grid.width * grid.height > 1_000_000:
                    raise DataValidationError("本地GUI单次最多100万个网格像元")
                if p["n_closest_points"] is not None and not 3 <= p["n_closest_points"] <= len(data.values):
                    raise DataValidationError("邻近点数必须在3至样本数之间")
                path = Path(p["support_path"])
                if not path.is_file():
                    raise DataValidationError("采样支持栅格不存在")
                api._support(grid, path)
                warnings.append("支持掩膜仅为调用方声明的采样范围，不是地质认证或找矿概率；请核实其依据")
                warnings.append("原浓度普通克里金可能产生负预测，保留并报告，不静默截零；方差是条件克里金方差而非全部勘查不确定度")
            else:
                if data.imputed.to_numpy().any():
                    raise DataValidationError("批次填补数据不能进行独立留出验证，请使用原始观测分支")
                train, test = p["train_ids"], p["test_ids"]
                if len(train) < 3 or not test or len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
                    raise DataValidationError("训练/测试ID必须唯一且不相交，训练至少3点，测试至少1点")
                if (set(train) | set(test)) - set(data.values.index):
                    raise DataValidationError("训练/测试包含未知ID；每行填写一个完整点号")
                warnings.append("划分由调用方指定，不自动随机抽样；应由用户评估地理隔离，不能将该结果直接声称为严格空间独立验证")
            warnings.append("空间步骤要求X为东坐标、Y为北坐标且单位为米；不自动套用主图CRS或修复坐标")
        backend = {"transform": "pyrolite", "variogram": "gstools", "interpolate": "pykrige", "validate": "pykrige"}.get(operation)
        if backend and importlib.util.find_spec(backend) is None:
            raise MissingBackendError(f"缺少可选后端{backend}，请在隔离环境安装对应extra")
    except (DataValidationError, CRSError, ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def execute(data, *, operation, parameters):
    checked = preflight(data, operation=operation, parameters=parameters)
    if not checked["ok"]:
        raise DataValidationError("\n".join(checked["errors"]))
    p = _parameters(operation, parameters)
    api = GeochemistryAPI()
    if operation in {"check", "impute", "transform"}:
        result = getattr(api, operation)(data, **p)
        report = dict(result.diagnostics, gui_warnings=checked["warnings"])
        return {"kind": "features" if operation == "transform" else "concentration", "data": result.data, "report": report}
    if operation == "interpolate":
        grid = GridSpec.from_bounds(p.pop("bounds"), p.pop("resolution"), data.crs)
        path = Path(p.pop("support_path"))
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        support = api._support(grid, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise DataValidationError("读取过程中支持栅格发生变化，请重新确认")
        result = api.interpolate(data, grid=grid, support_mask=support, **p)
        result.history.append(record("geochem_gui_support", "geo_toolbox", {"source": str(path), "source_sha256": before}))
        report = {"domain": result.domain, "units": result.units, "grid": vars(grid),
                  "history": result.history, "gui_warnings": checked["warnings"]}
        return {"kind": "maps", "data": result, "report": report}
    report = getattr(api, operation)(data, **p)
    report["gui_warnings"] = checked["warnings"]
    return {"kind": "report", "data": report, "report": report}


def export_result(data, *, directory, report):
    if not isinstance(data, (GeochemDataset, MapResult, dict)):
        raise DataValidationError("没有可导出的地化结果")
    target = Path(directory).expanduser().resolve()
    if target.exists():
        raise DataValidationError("输出目录已存在，必须选择新目录，不覆盖原成果")
    if not target.parent.is_dir():
        raise DataValidationError("输出父目录不存在")
    if isinstance(data, dict):
        target.mkdir(exist_ok=False)
        write_json(target / "report.json", data)
    else:
        data.export(target)
    write_json(target / "gui_report.json", report)
    return target
