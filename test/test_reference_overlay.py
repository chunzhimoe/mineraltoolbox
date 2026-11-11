import hashlib
import json

import numpy as np
import pytest
import shapefile
from pyproj import CRS

from geo_toolbox.errors import DataValidationError
from geo_toolbox.reference_overlay import compare_overlay, read_overlay, reference_layers


def _files(tmp_path, *, projection=True, status='defined'):
    path = tmp_path / '底图.shp'
    with shapefile.Writer(str(path), shapeType=shapefile.POLYLINE) as writer:
        writer.field('label', 'C')
        writer.line([[[660000, 3980000], [670000, 3980000]],
                     [[660000, 3990000], [670000, 3990000]]])
        writer.record('synthetic')
    crs = CRS.from_epsg(21456)
    prj = path.with_suffix('.prj')
    if projection:
        prj.write_text(crs.to_wkt('WKT1_ESRI'), encoding='utf-8')
    entry = {'path': 'annotations/layers/底图/底图.shp', 'crs_status': status,
             'crs_id': 'EPSG:21456' if status == 'defined' else None,
             'native_conversion': {'factor': 50.0}}
    if projection:
        entry['prj_sha256'] = hashlib.sha256(prj.read_bytes()).hexdigest()
    data = {'format_version': 1, 'kind': 'spatial_reference', 'geometry_coordinate_order': 'xy',
            'main_crs_id': 'EPSG:21456', 'data_crs_status': 'partial',
            'crs_definitions': {'EPSG:21456': {'status': 'defined', 'auth_id': 'EPSG:21456',
                                              'wkt': crs.to_wkt(), 'projjson': crs.to_json_dict()}},
            'layers': {'底图': entry},
            'native_map_frame': {'warnings': ['NO_REGIONAL_DATUM_SHIFT']}}
    reference = tmp_path / 'spatial_reference.json'
    reference.write_text(json.dumps(data), encoding='utf-8')
    return path, reference, data


def test_exported_xy_kept_without_rescaling_or_axis_swap(tmp_path):
    path, reference, _ = _files(tmp_path)
    before = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in tmp_path.iterdir()}
    overlay = read_overlay(path, reference=reference)
    assert overlay.bounds == (660000, 3980000, 670000, 3990000)
    assert len(overlay.parts) == 2
    np.testing.assert_array_equal(overlay.parts[0], [[660000, 3980000], [670000, 3980000]])
    assert 'NO_REGIONAL_DATUM_SHIFT' in overlay.warnings
    assert reference_layers(reference)['底图']['crs_id'] == 'EPSG:21456'
    result = compare_overlay([[665000, 3985000], [17000000, 3985000], [np.nan, 0]], overlay,
                             source_crs='EPSG:21456')
    assert result['status'] == 'compared'
    assert result['rows_inside_layer_bbox'] == 1
    assert result['rows_outside_layer_bbox'] == 1
    assert result['invalid_coordinate_rows'] == 1
    assert result['changed_coordinates'] is False
    assert before == {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in tmp_path.iterdir()}


def test_unknown_sample_crs_not_assigned_from_json(tmp_path):
    path, reference, _ = _files(tmp_path)
    overlay = read_overlay(path, reference=reference)
    assert compare_overlay([[665000, 3985000]], overlay)['status'] == 'blocked'
    assert compare_overlay([[665000, 3985000]], overlay, source_crs='EPSG:4326')['status'] == 'blocked'


def test_missing_layer_crs_never_inherits_main(tmp_path):
    path, reference, _ = _files(tmp_path, status='missing')
    with pytest.raises(DataValidationError, match='不能继承'):
        read_overlay(path, reference=reference)


def test_conflicting_or_modified_prj_rejected(tmp_path):
    path, reference, _ = _files(tmp_path)
    path.with_suffix('.prj').write_text(CRS.from_epsg(4326).to_wkt(), encoding='utf-8')
    with pytest.raises(DataValidationError, match='哈希'):
        read_overlay(path, reference=reference)


def test_json_without_prj_only_links_selected_layer(tmp_path):
    path, reference, _ = _files(tmp_path, projection=False)
    overlay = read_overlay(path, reference=reference)
    assert CRS.from_wkt(overlay.crs_wkt).to_epsg() == 21456
    assert any('没有 PRJ' in note for note in overlay.warnings)
    assert read_overlay(path).crs_wkt is None


def test_file_not_linked_by_filename_rejected(tmp_path):
    path, reference, data = _files(tmp_path)
    data['layers']['底图']['path'] = 'another.shp'
    reference.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(DataValidationError, match='文件名'):
        read_overlay(path, reference=reference, layer='底图')

def test_null_native_map_frame_kept_without_crs_guess(tmp_path):
    path, reference, data = _files(tmp_path)
    data['native_map_frame'] = None
    reference.write_text(json.dumps(data), encoding='utf-8')
    overlay = read_overlay(str(path), reference=str(reference))
    assert overlay.geometry_kind in ('points', 'lines', 'polygons')
    assert overlay.crs_wkt
