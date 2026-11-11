import numpy as np
import pytest

from geo_toolbox.coordinate_check import check_coordinates, coordinate_summary
from geo_toolbox.errors import DataValidationError


def test_unknown_keeps_values_and_allows_nonspatial():
    values = np.array([[600000., 4000000.], [np.nan, 2]])
    before = values.copy()
    result = check_coordinates(values)
    assert result['status'] == 'unknown' and result['invalid_rows'] == 1
    assert result['crs_wkt'] is None and not result['changed_coordinates']
    assert '非空间' in coordinate_summary(result)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize('crs, points', [
    ('EPSG:32632', [[500000, 5500000], [510000, 5510000]]),
    ('EPSG:21456', [[700000, 4000000]]),
    ('EPSG:4326', [[10, 50], [11, 51]]),
    ('EPSG:4807', [[0, 54]]),
    ('EPSG:27572', [[600000, 2200000]]),
])
def test_generic_supported_crs_not_site_specific(crs, points):
    result = check_coordinates(points, crs=crs)
    assert result['status'] == 'consistent'
    assert result['invalid_inverse_rows'] == 0
    assert not result['changed_coordinates']


def test_geographic_not_metres_and_swapped_axes_detected():
    assert not check_coordinates([[120, 35]], crs='EPSG:4326')['metric']
    result = check_coordinates([[35, 120]], crs='EPSG:4326')
    assert result['status'] == 'review' and result['invalid_inverse_rows'] == 1


def test_zone_encoded_not_auto_repaired():
    result = check_coordinates([[17230000, 3980000]], crs='EPSG:21456')
    assert result['status'] == 'review'
    assert result['bounds_xy'][0] == 17230000
    assert not result['changed_coordinates']


def test_prime_meridian_and_non_degree_units_not_silently_assumed():
    result = check_coordinates([[0, 54]], crs='EPSG:4807')
    assert result['inverse_bounds_lonlat'] == pytest.approx([2.33722917, 48.6, 2.33722917, 48.6])
    assert result['bounds_xy'] == [0, 54, 0, 54]


def test_invalid_crs_actionable_error():
    with pytest.raises(DataValidationError, match='留空'):
        check_coordinates([[1, 2]], crs='not-a-crs')
