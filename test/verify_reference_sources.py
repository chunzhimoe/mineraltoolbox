import argparse
import json
from pathlib import Path

from geo_toolbox.reference_overlay import read_overlay


def verify(layers, reference, baseline, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'layers': {}, 'sample_ranges': {},
              'policy': 'Read-only exported SHP and frozen survey audit; no rescaling, coordinate edits or CRS assignment'}
    for name in ('5万地质矿产图_区', '5万地质矿产图_线', '5万边框_线', '昆中断裂_线'):
        overlay = read_overlay(layers / name / (name + '.shp'), reference=reference, layer=name)
        report['layers'][name] = {'bounds': overlay.bounds, 'features': overlay.feature_count,
                                  'geometry_kind': overlay.geometry_kind, 'hashes': overlay.file_hashes,
                                  'reference_sha256': overlay.reference_sha256,
                                  'warnings': overlay.warnings, 'crs_wkt': overlay.crs_wkt}
    frozen = json.loads((baseline / 'audit.json').read_text(encoding='utf-8'))
    report['survey_sha256'] = frozen['source_sha256_before']
    for sheet, info in frozen['sheets'].items():
        report['sample_ranges'][sheet] = {'rows': info['rows'],
            'X': {key: info['distributions']['X']['quantiles'][key] for key in ('min', 'max')},
            'Y': {key: info['distributions']['Y']['quantiles'][key] for key in ('min', 'max')},
            'Z': {key: info['distributions']['Z']['quantiles'][key] for key in ('min', 'max')},
            'crs': info['qc']['crs']}
    (output / 'reference_sources.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('layers', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    verify(args.layers, args.reference, args.baseline, args.output)
