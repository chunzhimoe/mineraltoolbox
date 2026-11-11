import numpy as np
import pandas as pd
import pytest

from geo_toolbox import MagneticSchema
from geo_toolbox.magnetic import MagneticAPI
from geo_toolbox.errors import DataValidationError
from geo_toolbox.gui_tasks import _prepare_interpolation, execute, preflight, sample_bounds


def points():
    frame = pd.DataFrame({'id': ['a', 'b', 'c', 'repeat', 'inactive'],
                          'x': [1., 109., 1., 1., 999.], 'y': [2., 2., 111., 2., 999.],
                          'z': [0.] * 5, 'v': [10., 20., 30., 999., 5.]})
    data = MagneticAPI().read(frame, schema=MagneticSchema('id', 'x', 'y', 'v', 'EPSG:32648', 'nT', 'anomaly', upward='z'))
    return MagneticAPI().select(data, sample_ids=['a', 'b', 'c', 'repeat'])


def test_bounds_use_only_active_finite_coordinates_and_align_outward():
    data = points()
    assert sample_bounds(data) == [1., 2., 109., 111.]
    assert sample_bounds(data, 50) == [0., 0., 150., 150.]
    data.coordinates.loc['a', 'easting'] = np.nan
    assert sample_bounds(data) == [1., 2., 109., 111.]
    for spacing in (0, -1, np.nan, True):
        with pytest.raises(DataValidationError):
            sample_bounds(data, spacing)


def test_xy_filter_preserves_input_and_keeps_first_without_averaging():
    data = points()
    before = data.copy()
    prepared = _prepare_interpolation(data)
    assert prepared.active.tolist() == [True, True, True, False, False]
    assert prepared.values.loc['a'] == 10.
    pd.testing.assert_frame_equal(data.raw, before.raw)
    pd.testing.assert_series_equal(data.active, before.active)
    pd.testing.assert_series_equal(data.values, before.values)
    audit = prepared.history[-1]['parameters']
    assert audit['skipped_rows'] == 1
    assert audit['skipped'][0]['sample_id'] == 'repeat'
    assert audit['skipped'][0]['retained_sample_id'] == 'a'
    assert not audit['values_averaged']
    assert data.history == before.history


def test_interpolation_dispatch_prepares_duplicates_without_relaxing_api(monkeypatch):
    data = points()
    parameters = dict(auto_bounds=True, resolution=50, method='verde_spline', upward=0,
                      max_distance=200, damping=.1)
    assert preflight(data, operation='interpolate', parameters=parameters)['ok']
    received = []
    original = MagneticAPI.interpolate

    def capture(self, prepared, **kwargs):
        received.append(prepared)
        return original(self, prepared, **kwargs)

    monkeypatch.setattr(MagneticAPI, 'interpolate', capture)
    result = execute(data, operation='interpolate', parameters=parameters)
    assert result['data'].grid.width == result['data'].grid.height == 3
    assert result['data'].grid.transform == (50., 0., 0., 0., -50., 150.)
    assert received[0].active.sum() == 3
    assert data.active.sum() == 4
    assert any(entry['operation'] == 'interpolation_xy_filter' for entry in result['data'].history)
    with pytest.raises(DataValidationError, match='Duplicate active'):
        original(MagneticAPI(), data, grid=result['data'].grid, method='verde_spline', upward=0, max_distance=200)


def test_invalid_coordinates_still_block_interpolation():
    data = points()
    parameters = dict(auto_bounds=True, resolution=50, method='verde_spline', upward=0, max_distance=200)
    data.coordinates.loc['a', 'easting'] = np.nan
    assert not preflight(data, operation='interpolate', parameters=parameters)['ok']
