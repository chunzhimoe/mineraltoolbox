import numpy as np
import pandas as pd
import pytest

from geo_toolbox import GeochemSchema
from geo_toolbox.geochem import GeochemistryAPI
from geo_toolbox.geochem_gui_tasks import _prepare_spatial, execute, export_result, preflight
from geo_toolbox.errors import DataValidationError


def data():
    frame = pd.DataFrame({'id': ['a', 'bad', 'b', 'c', 'dup', 'd'],
                          'x': [0, np.nan, 100, 0, 0, 100], 'y': [0, 50, 0, 100, 0, 100],
                          'Au': [1., 2., 3., 4., 99., 6.]})
    return GeochemistryAPI().read(frame, schema=GeochemSchema('id', 'x', 'y', 'EPSG:32648', ['Au'], {'Au': 'ppm'}))


def test_spatial_copy_filters_coordinates_without_changing_import():
    original = data()
    before = original.copy()
    prepared = _prepare_spatial(original)
    assert prepared.values.index.tolist() == ['a', 'b', 'c', 'd']
    assert prepared.values.loc['a', 'Au'] == 1.
    for name in ('raw', 'values', 'coordinates', 'status', 'detection_limits', 'imputed'):
        pd.testing.assert_frame_equal(getattr(original, name), getattr(before, name))
        assert getattr(prepared, name).index.tolist() == ['a', 'b', 'c', 'd']
    assert original.history == before.history
    audit = prepared.history[-1]['parameters']
    assert audit['invalid_rows'] == audit['duplicate_rows'] == 1
    assert audit['skipped'][1]['retained_sample_id'] == 'a'
    assert not audit['values_averaged']


def test_variogram_preflight_dispatch_and_export_use_filtered_copy(tmp_path, monkeypatch):
    original = data()
    params = dict(element='Au', model='spherical', estimator='matheron')
    assert preflight(original, operation='variogram', parameters=params)['ok']
    received = []

    def capture(self, prepared, **kwargs):
        received.append(prepared)
        return {'history': prepared.history}

    monkeypatch.setattr(GeochemistryAPI, 'variogram', capture)
    result = execute(original, operation='variogram', parameters=params)
    assert received[0].values.index.tolist() == ['a', 'b', 'c', 'd']
    assert len(original.values) == 6
    output = export_result(result['data'], directory=tmp_path / 'result', report=result['report'])
    skipped = pd.read_csv(output / 'spatial_skipped_records.csv')
    assert skipped['sample_id'].tolist() == ['bad', 'dup']
    assert skipped['reason'].tolist() == ['invalid_xy', 'duplicate_xy']
    assert (output / 'spatial_coordinate_audit.json').exists()


def test_validation_remains_strict_and_too_few_points_is_readable():
    original = data()
    validation = dict(train_ids=['a', 'b', 'c'], test_ids=['d'], variogram='spherical')
    assert not preflight(original, operation='validate', parameters=validation)['ok']
    original.coordinates.loc[['b', 'c', 'd'], 'x'] = np.nan
    with pytest.raises(DataValidationError, match='至少需要3个'):
        _prepare_spatial(original)
