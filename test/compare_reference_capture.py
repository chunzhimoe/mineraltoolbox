import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geo_toolbox.reference_overlay import ReferenceOverlay, compare_overlay


def compare(reference_capture, survey_capture, output):
    frozen = json.loads(reference_capture.read_text(encoding='utf-8'))
    survey = json.loads((survey_capture / 'reconciliation.json').read_text(encoding='utf-8'))
    assert frozen['survey_sha256'] == survey['source_sha256'] and survey['source_unchanged']
    reports = {'source_sha256': survey['source_sha256'], 'comparisons': {},
               'policy': 'Uses already captured layer metadata and GUI-imported coordinates. No source reopen, scaling, axis swap, zone removal or CRS assignment. Raw numerical rectangle coincidence is NOT spatial alignment.'}
    for layer, metadata in frozen['layers'].items():
        overlay = ReferenceOverlay(source=str(reference_capture), layer=layer, crs_wkt=metadata['crs_wkt'],
            bounds=tuple(metadata['bounds']), parts=(), geometry_kind=metadata['geometry_kind'],
            feature_count=metadata['features'], file_hashes=metadata['hashes'],
            warnings=tuple(metadata['warnings']), reference_sha256=metadata['reference_sha256'])
        reports['comparisons'][layer] = {}
        for index, sheet in enumerate(survey['sheets'], 1):
            xy = pd.read_csv(survey_capture / f'sheet_{index:02d}' / 'coordinates.csv', index_col=0)[['easting', 'northing']].to_numpy()
            report = compare_overlay(xy, overlay, source_crs=None)
            assert report['status'] == 'blocked'
            x0, y0, x1, y1 = overlay.bounds
            finite = np.isfinite(xy).all(axis=1)
            numerical = finite & (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
            report['raw_number_rectangle_coincidence_only'] = int(numerical.sum())
            reports['comparisons'][layer][sheet] = report
    with output.open('x', encoding='utf-8') as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    for layer, sheets in reports['comparisons'].items():
        print(layer, {name: {'rows': value['sample_rows'], 'invalid': value['invalid_coordinate_rows'],
                            'raw_numerical_bbox_coincidence_not_alignment': value['raw_number_rectangle_coincidence_only'],
                            'status': value['status']} for name, value in sheets.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference_capture', type=Path)
    parser.add_argument('survey_capture', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    compare(args.reference_capture, args.survey_capture, args.output)
