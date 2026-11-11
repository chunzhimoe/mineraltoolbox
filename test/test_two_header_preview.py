import pandas as pd
import pytest

from geo_toolbox import gui_tasks as magnetic_tasks
from geo_toolbox.errors import DataValidationError
from geo_toolbox.gui_tasks import preview_source
from geo_toolbox.geochem_gui_tasks import import_source


def workbook(tmp_path, duplicate=False):
    source = tmp_path / 'headers.xlsx'
    pd.DataFrame([
        ['Title', '', '', '', ''],
        ['id', 'x', 'y', 'Units Au=ppb', ''],
        ['', '', '', 'Au', 'Au' if duplicate else 'As'],
        ['001', '1', '2', '1.02', '4'],
        ['', '3', '4', '2', '5'],
        ['001', '6', '7', '3', '6'],
        ['002', '8', '9', '4', '7'],
    ]).to_excel(source, header=False, index=False)
    return source


def mapping(**kwargs):
    return dict(sample_id='id', x='x', y='y', elements=['Au', 'As'], units={'Au': 'ppb', 'As': 'ppm'}, crs='', **kwargs)


def test_two_headers_preserve_tokens_provenance_and_source_rows(tmp_path):
    source = workbook(tmp_path)
    preview = preview_source(source, header=1, header_rows=2)
    assert preview['columns'] == ['id', 'x', 'y', 'Au', 'As']
    assert preview['frame']['id'].tolist() == ['001', '', '001', '002']
    assert preview['frame']['Au'].tolist() == ['1.02', '2', '3', '4']
    assert preview['original_headers'][0][3] == 'Units Au=ppb'
    assert preview['data_start_row'] == 4
    data = import_source(preview, mapping=mapping(id_policy='skip'))
    assert data.values.index.tolist() == ['001', '002']
    assert data.values.loc['001', 'Au'] == pytest.approx(.00102)
    audit = next(entry['parameters'] for entry in data.history if entry['operation'] == 'import_id_filter')
    assert [row['source_row'] for row in audit['skipped']] == [5, 6]
    assert data.history[-1]['parameters']['original_headers'] == preview['original_headers']
    generated = import_source(preview, mapping=mapping(generated_ids=True))
    assert generated.values.index.tolist() == ['row4', 'row5', 'row6', 'row7']


def test_two_header_duplicate_names_rejected(tmp_path):
    with pytest.raises(DataValidationError, match='列名重复'):
        preview_source(workbook(tmp_path, duplicate=True), header=1, header_rows=2)


@pytest.mark.parametrize('count', [0, 3, True, '2'])
def test_header_count_must_be_explicit_integer(tmp_path, count):
    with pytest.raises(DataValidationError, match='表头行数'):
        preview_source(workbook(tmp_path), header_rows=count)


def test_two_header_csv_preserves_string_tokens(tmp_path):
    source = tmp_path / 'two_header.csv'
    source.write_text(
        'Title,,,,\n'
        'id,x,y,Au,As\n'
        'id,x,y,Au,As\n'
        '001,1.5,2.5,1.02,4\n'
        '007,3.5,4.5,0.50,5\n',
        encoding='utf-8')
    preview = preview_source(source, header=1, header_rows=2)
    assert preview['columns'] == ['id', 'x', 'y', 'Au', 'As']
    assert preview['data_start_row'] == 4
    assert preview['original_headers'][0] == ['id', 'x', 'y', 'Au', 'As']
    assert preview['frame']['id'].tolist() == ['001', '007']
    assert preview['frame']['Au'].tolist() == ['1.02', '0.50']
    assert preview['frame']['x'].tolist() == ['1.5', '3.5']


def test_two_header_blank_and_whitespace_names_fallback(tmp_path):
    source = tmp_path / 'blanks.xlsx'
    pd.DataFrame([
        ['id', 'x', 'y', 'Au', ''],
        ['id', 'x', '   ', 'As', ''],
        ['001', '1', '2', '1.02', '4'],
    ]).to_excel(source, header=False, index=False)
    preview = preview_source(source, header=0, header_rows=2)
    assert preview['columns'] == ['id', 'x', 'y', 'As', '未命名列5']
    assert preview['data_start_row'] == 3


def test_two_header_without_data_rejected(tmp_path):
    source = tmp_path / 'empty.xlsx'
    pd.DataFrame([
        ['id', 'x', 'y'],
        ['id', 'x', 'y'],
    ]).to_excel(source, header=False, index=False)
    with pytest.raises(DataValidationError, match='没有数据'):
        preview_source(source, header=0, header_rows=2)


def test_magnetic_import_two_header_generated_ids_and_provenance(tmp_path):
    source = workbook(tmp_path)
    preview = preview_source(source, header=1, header_rows=2)
    data = magnetic_tasks.import_source(preview, mapping=dict(
        sample_id=None, x='x', y='y', value='Au', unit='nT', stage='anomaly',
        crs='', id_policy='strict', generated_ids=True, split_point_line=False))
    assert data.values.index.tolist() == ['row4', 'row5', 'row6', 'row7']
    entry = next(item['parameters'] for item in data.history if item['operation'] == 'gui_import')
    assert entry['header_rows'] == 2
    assert entry['data_start_row'] == 4
    assert entry['original_headers'] == preview['original_headers']
    assert data.raw['Au'].tolist() == ['1.02', '2', '3', '4']
