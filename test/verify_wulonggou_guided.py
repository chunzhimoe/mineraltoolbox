import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


class Messages:
    def __init__(self):
        self.calls = []

    def showerror(self, title, message):
        self.calls.append(('error', title, str(message)))

    def showwarning(self, title, message):
        self.calls.append(('warning', title, str(message)))

    def showinfo(self, title, message):
        self.calls.append(('info', title, str(message)))

    def askokcancel(self, title, message):
        self.calls.append(('confirm', title, str(message)))
        return True


def pump(root, workbench, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if not workbench._busy and workbench._queue.empty():
            return
        time.sleep(.02)
    raise TimeoutError('GUI task did not finish')


def reconcile(source, baseline, output):
    import tkinter as tk
    from geo_toolbox.gui import MagneticWorkbench
    from geo_toolbox import gui_tasks

    output.mkdir(parents=True, exist_ok=False)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    historical = json.loads((baseline / 'audit.json').read_text(encoding='utf-8'))
    assert original_hash == historical['source_sha256_before']
    report = {'source_sha256': original_hash, 'sheets': {},
              'policy': 'GUI read-only import/QC; nT/anomaly provisional as in frozen audit; no CRS or height assignment'}
    with pd.ExcelFile(source, engine='openpyxl') as book:
        sheets = book.sheet_names
    for number, sheet in enumerate(sheets, 1):
        root = tk.Tk()
        root.withdraw()
        workbench = MagneticWorkbench(root)
        workbench.messagebox = Messages()
        try:
            workbench.path_var.set(str(source))
            workbench.header_var.set('2')
            workbench.sheet_var.set(sheet)
            workbench.preview_btn.invoke()
            pump(root, workbench)
            assert workbench.preview is not None
            final_columns = [name for name in ('dt', '△T', 'ΔT') if name in workbench.preview['columns']]
            assert len(final_columns) == 1
            for key, column in {'sample_id': '点线号', 'x': 'X', 'y': 'Y', 'value': final_columns[0]}.items():
                workbench.map_vars[key].set(column)
            workbench._check_source_coordinates()
            pump(root, workbench)
            assert workbench.coordinate_report['status'] == 'unknown'
            workbench.unit_var.set('nT')
            workbench.stage_var.set('anomaly')
            workbench.generated_var.set(True)
            workbench.crs_var.set('')
            workbench.import_btn.invoke()
            pump(root, workbench)
            assert not [call for call in workbench.messagebox.calls if call[0] == 'error'], workbench.messagebox.calls
            data = workbench.dataset
            assert data is not None
            historical_root = baseline / f'sheet_{number:02d}'
            values = pd.read_csv(historical_root / 'import/values.csv', index_col=0)
            coordinates = pd.read_csv(historical_root / 'import/coordinates.csv', index_col=0)
            statuses = pd.read_csv(historical_root / 'import/status.csv', index_col=0)
            qc = json.loads((historical_root / 'audit.json').read_text(encoding='utf-8'))['qc']
            assert len(data.values) == len(values) == historical['sheets'][sheet]['rows']
            np.testing.assert_allclose(data.values.to_numpy(dtype=float), values.iloc[:, 0].to_numpy(dtype=float), rtol=0, atol=1e-10, equal_nan=True)
            np.testing.assert_allclose(data.coordinates.to_numpy(dtype=float), coordinates.to_numpy(dtype=float), rtol=0, atol=1e-8, equal_nan=True)
            assert data.status.tolist() == statuses.iloc[:, 0].tolist()
            raw_baseline = pd.read_csv(historical_root / 'source_table.csv', dtype=str, keep_default_na=False)
            pd.testing.assert_frame_equal(data.raw[raw_baseline.columns].reset_index(drop=True), raw_baseline.reset_index(drop=True), check_dtype=False)
            current = workbench.last_report
            for field in ('rows', 'active', 'stage', 'status_counts', 'unit', 'changed_values', 'crs'):
                assert current[field] == qc[field], (sheet, field, current[field], qc[field])
            for field in ('invalid_coordinate_ids', 'duplicate_coordinate_ids', 'unknown_upward_ids'):
                assert [value.split(':')[-1] for value in qc[field]] == current[field]
            assert data.schema.crs is None and data.schema.upward is None
            blocked = gui_tasks.preflight(data, operation='interpolate', parameters={
                'bounds': [0, 0, 100, 100], 'resolution': 50, 'method': 'equivalent_sources',
                'upward': 0, 'max_distance': 100, 'depth': 100})
            assert not blocked['ok']
            sheet_output = output / f'sheet_{number:02d}'
            workbench.export_var.set(str(sheet_output))
            workbench.export_btn.invoke()
            pump(root, workbench)
            assert sheet_output.is_dir()
            assert not [call for call in workbench.messagebox.calls if call[0] == 'error']
            report['sheets'][sheet] = {'rows': len(data.values), 'values_match': True, 'coordinates_match': True,
                'raw_columns_match': True, 'status_match': True, 'qc_match': True,
                'coordinate_issue_rows': len(current['invalid_coordinate_ids']),
                'duplicate_coordinate_rows': len(current['duplicate_coordinate_ids']),
                'spatial_processing': 'blocked', 'messages': workbench.messagebox.calls}
        finally:
            workbench._on_close()
    report['total_rows'] = sum(value['rows'] for value in report['sheets'].values())
    assert report['total_rows'] == historical['total_rows']
    report['source_unchanged'] = hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert report['source_unchanged']
    (output / 'reconciliation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    reconcile(args.source, args.baseline, args.output)
