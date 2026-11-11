from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError, ProjError

from .errors import DataValidationError


def check_coordinates(points, *, crs=None):
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise DataValidationError('请选择两列坐标，按东坐标／经度、北坐标／纬度顺序核对')
    finite = np.isfinite(xy).all(axis=1)
    result = {'rows': len(xy), 'finite_rows': int(finite.sum()),
              'invalid_rows': int((~finite).sum()), 'status': 'unknown',
              'crs_wkt': None, 'changed_coordinates': False,
              'coordinate_order': 'xy: easting/longitude, northing/latitude',
              'meaning': '检查声明和数值是否矛盾，不认证测量基准；不猜坐标系、不自动交换轴或去带号'}
    values = xy[finite]
    result['bounds_xy'] = ([float(v) for v in (*values.min(axis=0), *values.max(axis=0))]
                           if len(values) else None)
    if crs is None or not str(crs).strip():
        result['messages'] = ['未提供测点坐标系；可以继续查看原值和做非空间检查，不能生成定位网格。']
        return result
    try:
        definition = CRS.from_user_input(crs)
    except CRSError as error:
        raise DataValidationError('坐标系无法解析。可填写 EPSG 编号或完整 WKT；不清楚时留空，继续非空间检查。') from error
    result.update(crs_wkt=definition.to_wkt(), name=definition.name,
                  authority=definition.to_authority(min_confidence=100),
                  is_projected=definition.is_projected, is_geographic=definition.is_geographic,
                  axes=[{'name': axis.name, 'direction': axis.direction,
                         'unit': axis.unit_name, 'factor': axis.unit_conversion_factor}
                        for axis in definition.axis_info])
    messages = ['已读取坐标系声明；坐标列仍需按实际记录确认，不按 EPSG 轴名自动交换。']
    if not (definition.is_projected or definition.is_geographic) or len(definition.axis_info) != 2:
        result.update(status='review', messages=messages + ['这不是二维平面／经纬度坐标系，请先核对水平坐标定义；不自动取复合坐标系的子系统。'])
        return result
    result['metric'] = definition.is_projected and all(
        axis.unit_conversion_factor == 1 for axis in definition.axis_info)
    if definition.is_geographic:
        messages.append('这是经纬度（角度），不能直接当米坐标用于网格或磁场运算。')
    elif not result['metric']:
        messages.append('平面单位不是米；本次不换算，公制空间运算需另行明确转换。')
    if not len(values):
        result.update(status='review', messages=messages + ['没有完整有限坐标，无法检查空间范围。'])
        return result
    geodetic = definition.geodetic_crs
    axes = {axis.direction: axis for axis in geodetic.axis_info} if geodetic else {}
    if not {'east', 'north'}.issubset(axes):
        result.update(status='review', messages=messages + ['此坐标系的经纬度轴方向需要专门核对，本次不推测轴的正负或顺序。'])
        return result
    try:
        if definition.is_geographic:
            angular = values.copy()
        else:
            converter = Transformer.from_crs(definition, geodetic, always_xy=True)
            lon, lat = converter.transform(values[:, 0], values[:, 1], errcheck=False)
            angular = np.column_stack((lon, lat))
        lonlat = angular * np.array([axes['east'].unit_conversion_factor,
                                    axes['north'].unit_conversion_factor]) * (180 / np.pi)
        native_longitude_valid = np.abs(lonlat[:, 0]) <= 180
        prime = geodetic.prime_meridian
        offset = prime.longitude * prime.unit_conversion_factor * 180 / np.pi
        if offset:
            lonlat[:, 0] = (lonlat[:, 0] + offset + 180) % 360 - 180
    except ProjError as error:
        result.update(status='review', messages=messages + [f'无法按此声明反算位置：{error}'])
        return result
    valid = np.isfinite(lonlat).all(axis=1) & native_longitude_valid & (np.abs(lonlat[:, 0]) <= 180) & (np.abs(lonlat[:, 1]) <= 90)
    result['invalid_inverse_rows'] = int((~valid).sum())
    area = definition.area_of_use
    outside = np.zeros(len(values), dtype=bool)
    if area is not None:
        west, south, east, north = area.bounds
        longitude_ok = ((lonlat[:, 0] >= west) & (lonlat[:, 0] <= east)) if west <= east else (
            (lonlat[:, 0] >= west) | (lonlat[:, 0] <= east))
        outside = valid & (~longitude_ok | (lonlat[:, 1] < south) | (lonlat[:, 1] > north))
        result['area_of_use'] = {'name': area.name, 'bounds_lonlat': list(area.bounds)}
        result['outside_area_rows'] = int(outside.sum())
        if outside.any():
            messages.append(f'{int(outside.sum())} 行反算位置在坐标系登记适用范围外；请复核带号、中央经线、轴顺序或来源说明，不自动修正。')
    if (~valid).any():
        messages.append(f'{int((~valid).sum())} 行不能反算为有效经纬度，请核对声明和坐标编码。')
    result['inverse_bounds_lonlat'] = ([float(v) for v in (*lonlat[valid].min(axis=0), *lonlat[valid].max(axis=0))]
                                      if valid.any() else None)
    result['status'] = 'review' if (~finite).any() or (~valid).any() or outside.any() else 'consistent'
    messages.append('数值检查未见明显矛盾不等于已认证坐标系；仍以测量技术报告和控制点为准。')
    result['messages'] = messages
    return result


def coordinate_summary(report):
    bounds = report.get('bounds_xy')
    lines = [f"坐标检查：共 {report['rows']} 行，完整坐标 {report['finite_rows']} 行，缺失／非法 {report['invalid_rows']} 行。"]
    if bounds:
        lines.append(f'第一列范围：{bounds[0]:.8g} ～ {bounds[2]:.8g}；第二列：{bounds[1]:.8g} ～ {bounds[3]:.8g}。')
    if report.get('name'):
        lines.append('声明的坐标系：' + report['name'])
    lines.extend(report['messages'])
    return '\n'.join(lines)
