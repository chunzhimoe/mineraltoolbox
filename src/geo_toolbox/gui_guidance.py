from __future__ import annotations

import re

_COLUMN_ALIASES = {
    "sample_id": ("点号", "编号", "样号", "id", "sample", "pointid", "station", "点线号"),
    "x": ("x", "easting", "东坐标"),
    "y": ("y", "northing", "北坐标"),
    "value": ("值", "磁场", "磁异常", "△t", "dt", "value", "field", "anomaly"),
    "upward": ("z", "高程", "高度", "upward", "elevation", "alt", "height"),
    "line": ("线号", "测线", "line", "测线号"),
    "sequence": ("点序", "序号", "sequence", "seq", "点号序"),
    "point_line": ("点线号", "点线", "pointline", "点号线号"),
}

_STAGE_LABELS = {"raw_total_field": "仪器测得的总磁场", "corrected_total_field": "已改正的总磁场（需核对改正内容）",
                 "anomaly": "磁异常", "rtp": "化极异常"}
_METHOD_LABELS = {"equivalent_sources": "等效源（需要高度）", "verde_spline": "二维样条（同高观测面）"}
_TRANSFORM_LABELS = {
    "upward_continuation": "向上延拓", "derivative_easting": "东向导数",
    "derivative_northing": "北向导数", "derivative_upward": "垂向导数",
    "directional_derivative": "方向导数", "total_gradient_amplitude": "总梯度模",
    "tilt_angle": "倾角", "reduction_to_pole": "化极"}
_IMPUTE_LABELS = {"multRepl": "按限值比例替换（不填普通缺失）",
                  "lrEM": "按元素间关系估计低于限值的值（有使用条件）",
                  "lrEMplus": "同时估计缺失与低于限值（有使用条件）"}
_MODEL_LABELS = {"spherical": "球状", "exponential": "指数", "gaussian": "高斯",
                 "linear": "线性", "power": "幂律"}
_ESTIMATOR_LABELS = {"matheron": "matheron", "cressie": "cressie-hawkins"}
_LOGRATIO_LABELS = {"clr": "各元素相对组合中心的特征", "ilr": "正交元素组合特征"}


def raw_value(text, table):
    text = (text or "").strip()
    if text in table:
        return table[text]
    for raw in table.values():
        if text.startswith(raw + " ") or text.endswith(" " + raw) or text == raw:
            return raw
    return text


def labeled(raw):
    label = (_STAGE_LABELS.get(raw) or _METHOD_LABELS.get(raw)
             or _TRANSFORM_LABELS.get(raw) or _IMPUTE_LABELS.get(raw)
             or _LOGRATIO_LABELS.get(raw) or _MODEL_LABELS.get(raw))
    return label or _ESTIMATOR_LABELS.get(raw) or raw


def raw_stage(text):
    return raw_value(text, {value: key for key, value in _STAGE_LABELS.items()})


def raw_method(text):
    return raw_value(text, {value: key for key, value in _METHOD_LABELS.items()})


def raw_transform(text):
    return raw_value(text, {value: key for key, value in _TRANSFORM_LABELS.items()})


def raw_impute(text):
    return raw_value(text, {value: key for key, value in _IMPUTE_LABELS.items()})


def raw_logratio(text):
    return raw_value(text, {value: key for key, value in _LOGRATIO_LABELS.items()})


def raw_model(text):
    return raw_value(text, {value: key for key, value in _MODEL_LABELS.items()})


def raw_estimator(text):
    return raw_value(text, {value: key for key, value in _ESTIMATOR_LABELS.items()})


def _normalize(name):
    text = str(name).strip().lower().replace("△t", "dt").replace("δt", "dt")
    return re.sub(r"[^a-z0-9一-鿿]+", "", text.lower())


def suggest_columns(columns, roles):
    normalized = {}
    for column in columns:
        normalized.setdefault(_normalize(column), []).append(column)
    suggestions = {}
    for role in roles:
        if role not in {"sample_id", "x", "y", "value"}:
            continue
        matches = set()
        for alias in _COLUMN_ALIASES.get(role, (role,)):
            key = _normalize(alias)
            if key in normalized:
                matches.update(normalized[key])
        if len(matches) == 1:
            suggestions[role] = matches.pop()
    return suggestions


def magnetic_step(preview, dataset, last_result):
    if preview is None:
        return 1
    if dataset is None:
        return 2
    if last_result is None:
        return 3
    return 4


_STATUS_NAMES = {"observed": "观测", "missing": "缺失", "invalid": "非法",
                 "censored": "低于检出限"}


def human_counts(series):
    counts = series.value_counts().to_dict() if hasattr(series, "value_counts") else {}
    return ", ".join(f"{_STATUS_NAMES.get(name, name)} {count}"
                     for name, count in counts.items())


def magnetic_summary(data, report):
    lines = [f"样点 {len(data.values)} 行；当前阶段 {_STAGE_LABELS.get(data.stage, data.stage)}；"
             "原始文件未被修改。"]
    lines.append("数值状态：" + "，".join(
        f"{_STATUS_NAMES.get(k, k)} {v} 行"
        for k, v in data.status.value_counts().to_dict().items()))
    invalid = len(report.get("invalid_coordinate_ids", []) or [])
    duplicate = len(report.get("duplicate_coordinate_ids", []) or [])
    if invalid or duplicate:
        lines.append(f"坐标：{invalid} 行缺失/非法，{duplicate} 行重复；原行保留，不自动修复。")
    else:
        lines.append("坐标：未发现缺失/非法或重复。")
    if data.schema.crs is None:
        lines.append("坐标系未知：可以做非空间质控与预览；网格化/空间运算需要确认投影CRS。")
    else:
        lines.append("坐标系已声明，仍需核对坐标检查结果与测量来源。")
    coordinate = next((entry['parameters'] for entry in reversed(data.history)
                       if entry['operation'] == 'coordinate_check'), None)
    if coordinate:
        lines.extend(coordinate['messages'])
    lines.append("下一步：可在本页执行质控，或到“后续处理”生成网格，或到“4 保存结果”导出。")
    return "\n".join(lines)


def geochem_summary(data, report):
    counts = {name: int((data.status == name).to_numpy().sum())
              for name in ("observed", "missing", "censored", "invalid")}
    lines = [f"样点 {len(data.values)} 行；元素：{', '.join(data.schema.elements)}；原数据未被修改。",
             f"状态：正常观测 {counts['observed']}，缺失 {counts['missing']}，"
             f"低于检出限 {counts['censored']}，非法 {counts['invalid']}。"]
    invalid = len(report.get("invalid_coordinate_ids", []) or [])
    duplicate = len(report.get("duplicate_coordinate_ids", []) or [])
    lines.append(f"坐标：{invalid} 行缺失/非法，{duplicate} 行重复。")
    if data.crs is None:
        lines.append("坐标系未知：可做非空间质控；空间步骤需要确认投影CRS。")
    coordinate = next((entry['parameters'] for entry in reversed(data.history)
                       if entry['operation'] == 'coordinate_check'), None)
    if coordinate:
        lines.extend(coordinate['messages'])
    if counts["invalid"]:
        lines.append("存在非法值：需先复核来源，不能直接填补或变换。")
    lines.append("下一步：可处理缺失/删失、生成特征或网格，也可直接导出。")
    return "\n".join(lines)


def require_fields(mapping, required):
    missing = [label for key, label in required if not (mapping.get(key) or "")]
    return "请先填写或选择： " + "、".join(missing) if missing else None
