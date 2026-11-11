import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from geo_toolbox.gui_tasks import import_source, preview_source

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--header', type=int, required=True)
parser.add_argument('--id', dest='identifier', required=True)
parser.add_argument('--x', required=True)
parser.add_argument('--y', required=True)
parser.add_argument('--values', nargs='+', required=True)
parser.add_argument('--unit', required=True)
parser.add_argument('--stage', required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
before_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
with pd.ExcelFile(args.source) as book:
    sheets = list(book.sheet_names)
report = {'source_sha256': before_hash, 'sheets': {}}
for sheet in sheets:
    preview = preview_source(args.source, sheet_name=sheet, header=args.header - 1)
    frame = preview['frame']
    value_column = next(name for name in args.values if name in frame)
    ids = frame[args.identifier]
    blank = ids.isna() | ids.astype(str).str.strip().eq('')
    duplicate = ids.duplicated(keep='first') & ~blank
    expected = frame.loc[~(blank | duplicate)].reset_index(drop=True)
    data = import_source(preview, mapping={
        'sample_id': args.identifier, 'x': args.x, 'y': args.y, 'value': value_column,
        'unit': args.unit, 'stage': args.stage, 'crs': '', 'id_policy': 'skip'})
    pd.testing.assert_frame_equal(data.raw.reset_index(drop=True), expected)
    assert data.schema.crs is None
    audit = next(entry['parameters'] for entry in data.history if entry['operation'] == 'import_id_filter')
    assert audit['blank_rows'] == int(blank.sum())
    assert audit['duplicate_rows'] == int(duplicate.sum())
    report['sheets'][sheet] = {key: audit[key] for key in
                               ('input_rows', 'retained_rows', 'blank_rows', 'duplicate_rows')}
    report['sheets'][sheet]['skipped_source_rows'] = [row['source_row'] for row in audit['skipped']]
    report['sheets'][sheet]['retained_raw_matches_source'] = True
assert hashlib.sha256(args.source.read_bytes()).hexdigest() == before_hash
report['source_unchanged'] = True
with args.output.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
