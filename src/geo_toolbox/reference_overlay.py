from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS
from pyproj.exceptions import CRSError

from .dependencies import require
from .errors import DataValidationError
from .mapgis import MapGISAPI


@dataclass(frozen=True)
class ReferenceOverlay:
    source: str
    layer: str
    crs_wkt: str | None
    bounds: tuple[float, float, float, float] | None
    parts: tuple[np.ndarray, ...]
    geometry_kind: str
    feature_count: int
    file_hashes: dict[str, str]
    warnings: tuple[str, ...]
    reference_sha256: str | None = None


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar(source, extension):
    matches = [entry for entry in source.parent.iterdir()
               if entry.stem.casefold() == source.stem.casefold()
               and entry.suffix.casefold() == extension]
    if len(matches) > 1:
        raise DataValidationError(f'同名配套文件不唯一：{source.stem}{extension}')
    return matches[0] if matches else None


def _same_crs(first, second):
    if first.equals(second, ignore_axis_order=True):
        return True
    authority = first.to_authority(min_confidence=100)
    return authority is not None and authority == second.to_authority(min_confidence=100)


def reference_layers(reference):
    data = MapGISAPI().read_reference(reference)
    return {name: {'path': entry.get('path'), 'crs_status': entry['crs_status'],
                   'crs_id': entry.get('crs_id')}
            for name, entry in data['layers'].items()}


def read_overlay(source, *, reference=None, layer=None):
    path = Path(source).expanduser().resolve()
    if not path.is_file() or path.suffix.casefold() != '.shp':
        raise DataValidationError('请选择转换后的 .shp 文件；不会再次缩放 MapGIS 坐标')
    files = {'.shp': path}
    for extension in ('.shx', '.dbf', '.prj', '.cpg'):
        found = _sidecar(path, extension)
        if found is not None:
            files[extension] = found
    absent = {'.shx', '.dbf'} - files.keys()
    if absent:
        raise DataValidationError('SHP 配套文件缺失：' + '、'.join(sorted(absent)))
    hashes = {extension: _digest(file) for extension, file in files.items()}
    warnings = []
    crs = None
    if '.prj' in files:
        try:
            crs = CRS.from_wkt(files['.prj'].read_text(encoding='utf-8-sig'))
        except (CRSError, UnicodeError) as error:
            raise DataValidationError('底图 .prj 无法解析，请保留原文件并核对坐标说明') from error
    reference_hash = None
    selected = layer or path.stem
    if reference is not None:
        reference_path = Path(reference).expanduser().resolve()
        reference_hash = _digest(reference_path)
        data = MapGISAPI().read_reference(reference_path)
        if layer is None:
            matches = [name for name, entry in data['layers'].items()
                       if isinstance(entry.get('path'), str)
                       and Path(entry['path']).name.casefold() == path.name.casefold()]
            if len(matches) != 1:
                raise DataValidationError('JSON 中无法唯一匹配该 SHP，请明确选择对应图层')
            selected = matches[0]
        if selected not in data['layers']:
            raise DataValidationError('JSON 中不存在所选图层')
        entry = data['layers'][selected]
        declared_path = entry.get('path')
        if not isinstance(declared_path, str) or Path(declared_path).name.casefold() != path.name.casefold():
            raise DataValidationError('所选 SHP 文件名与 JSON 图层记录不一致')
        if entry['crs_status'] != 'defined':
            raise DataValidationError('此 JSON 图层没有已定义的坐标系，不能继承主图坐标系')
        linked = MapGISAPI().layer_crs(data, selected)
        expected_hash = entry.get('prj_sha256')
        if expected_hash and hashes.get('.prj') != expected_hash:
            raise DataValidationError('PRJ 哈希与 JSON 不一致；请选择对应批次的转换结果')
        if crs is not None and not _same_crs(crs, linked):
            raise DataValidationError('SHP 的 PRJ 与 JSON 中该图层的坐标系冲突')
        if crs is None:
            warnings.append('SHP 没有 PRJ；本次预览使用所选图层的 JSON 定义，不写入文件')
        crs = linked
        frame_warnings = (data.get('native_map_frame') or {}).get('warnings', [])
        warnings.extend(str(value) for value in frame_warnings)
        if _digest(reference_path) != reference_hash:
            raise DataValidationError('读取期间坐标说明 JSON 发生变化，请重新选择')
    if crs is None:
        warnings.append('底图坐标系未知：只能查看原坐标，不能判定与样点对齐')
    shp = require('shapefile', 'pyshp')
    parts = []
    vertices = 0
    count = 0
    with path.open('rb') as geometry, files['.shx'].open('rb') as index, \
            shp.Reader(shp=geometry, shx=index) as reader:
        shape_type = reader.shapeType
        if shape_type in (1, 8, 11, 18, 21, 28):
            kind = 'points'
        elif shape_type in (3, 13, 23):
            kind = 'lines'
        elif shape_type in (5, 15, 25):
            kind = 'polygons'
        elif shape_type == 0:
            kind = 'empty'
        else:
            raise DataValidationError('暂不支持此 SHP 几何类型，仅支持点、线和面底图')
        for shape in reader.iterShapes():
            count += 1
            if shape.shapeType == 0:
                continue
            xy = np.asarray(shape.points, dtype=float)
            if xy.size == 0:
                continue
            xy = xy[:, :2]
            if not np.isfinite(xy).all():
                raise DataValidationError('底图存在非有限坐标，不能静默忽略')
            vertices += len(xy)
            if vertices > 2_000_000:
                raise DataValidationError('底图超过本地预览的 200 万顶点资源上限；请另选较小图层，不自动抽样')
            starts = list(shape.parts) if kind != 'points' else [0]
            starts = starts or [0]
            for start, stop in zip(starts, starts[1:] + [len(xy)]):
                piece = xy[start:stop].copy()
                piece.setflags(write=False)
                parts.append(piece)
    if any(_digest(file) != hashes[extension] for extension, file in files.items()):
        raise DataValidationError('读取期间底图发生变化，请重新选择')
    bounds = None
    if parts:
        minimum = np.min([part.min(axis=0) for part in parts], axis=0)
        maximum = np.max([part.max(axis=0) for part in parts], axis=0)
        bounds = (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1]))
    return ReferenceOverlay(str(path), selected, crs.to_wkt() if crs is not None else None,
                            bounds, tuple(parts), kind, count, hashes, tuple(warnings), reference_hash)


def compare_overlay(points, overlay, *, source_crs=None):
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise DataValidationError('样点坐标必须为两列：东坐标、北坐标（SHP 存储 x,y 顺序）')
    result = {'status': 'blocked', 'layer': overlay.layer, 'sample_rows': len(xy),
              'overlay_bounds': overlay.bounds, 'changed_coordinates': False,
              'meaning': '范围重叠仅用于发现明显错位，不是位置或坐标基准正确性证明'}
    finite = np.isfinite(xy).all(axis=1)
    result.update(finite_sample_rows=int(finite.sum()), invalid_coordinate_rows=int((~finite).sum()),
                  sample_bounds=([float(value) for value in (*xy[finite].min(axis=0), *xy[finite].max(axis=0))]
                                 if finite.any() else None),
                  source_crs=str(source_crs) if source_crs else None, layer_crs_wkt=overlay.crs_wkt,
                  source_file_hashes=overlay.file_hashes, reference_sha256=overlay.reference_sha256)
    if source_crs is None or overlay.crs_wkt is None:
        result['reason'] = '请先确认样点和底图各自的坐标系；底图定义不会自动赋给 Excel'
        return result
    if not _same_crs(CRS.from_user_input(source_crs), CRS.from_wkt(overlay.crs_wkt)):
        result['reason'] = '样点与底图坐标系不同；本次不会自动重投影或采用未确认的基准转换'
        return result
    if overlay.bounds is None:
        result['reason'] = '所选底图为空，无法核对范围'
        return result
    finite = np.isfinite(xy).all(axis=1)
    x0, y0, x1, y1 = overlay.bounds
    inside = finite & (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    result.update(status='compared', finite_sample_rows=int(finite.sum()),
                  invalid_coordinate_rows=int((~finite).sum()),
                  rows_inside_layer_bbox=int(inside.sum()),
                  rows_outside_layer_bbox=int((finite & ~inside).sum()),
                  sample_bounds=([float(value) for value in (*xy[finite].min(axis=0), *xy[finite].max(axis=0))]
                                 if finite.any() else None))
    return result
