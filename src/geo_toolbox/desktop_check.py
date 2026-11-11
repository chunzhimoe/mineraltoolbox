import json
import tempfile
import time
import traceback
from pathlib import Path


def _write(report, result):
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')


def _workbench_checks():
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from .geochem_gui import GeochemWorkbench
    from .gui import MagneticWorkbench

    for workbench in (MagneticWorkbench, GeochemWorkbench):
        errors = []

        class CheckedWorkbench(workbench):
            def _on_callback_error(self, kind, value, tb):
                errors.append(''.join(traceback.format_exception(kind, value, tb)))

        root = tk.Tk()
        root.withdraw()
        app = None
        try:
            app = CheckedWorkbench(root)
            canvas = FigureCanvasTkAgg(Figure(figsize=(2, 2)), master=root)
            canvas.draw()
            canvas.get_tk_widget().destroy()
            root.update()
            if errors:
                raise RuntimeError('\n'.join(errors))
        finally:
            if app is not None:
                app._on_close()
            else:
                root.destroy()


def _science_checks(report):
    import numpy as np
    import pandas as pd
    import ppigrf
    import openpyxl
    import xlrd
    import rasterio
    from rasterio.transform import from_origin
    from . import GeoToolbox, GeochemSchema, GridSpec
    from .magnetic_core import MagneticGrid

    api = GeoToolbox().geochem
    frame = pd.DataFrame({'id': ['01', '02', '03', '04', '05', '06'],
                          'x': [0., 100., 0., 100., 25., 75.],
                          'y': [0., 0., 100., 100., 25., 75.],
                          'Au': ['<0.5', '200', '300', '400', '180', '320'],
                          'As': [10., 23., 31., 45., 18., 32.]})
    schema = GeochemSchema('id', 'x', 'y', 'EPSG:2380', ['Au', 'As'], {'Au': 'ppb', 'As': 'ppm'})
    data = api.impute(api.read(frame, schema=schema), method='multRepl').data
    assert np.isfinite(api.transform(data, method='clr').data.values.to_numpy()).all()
    grid = GridSpec.from_bounds((0, 0, 100, 100), 50, schema.crs)
    maps = api.interpolate(data, grid=grid, support_mask=np.ones((2, 2), bool), bin_edges=[0, 40, 80, 120, 160])
    assert all(np.isfinite(value).all() for value in maps.predictions.values())
    magnetic_spec = GridSpec.from_bounds((0, 0, 400, 400), 50, schema.crs)
    magnetic_values = np.arange(64, dtype=float).reshape(8, 8)
    magnetic = MagneticGrid(magnetic_spec, magnetic_values, np.ones((8, 8), bool), 'anomaly', 'nT', 0.)
    transformed = GeoToolbox().magnetic.transform(magnetic, method='derivative_upward', padding=2, trim=0)
    assert np.isfinite(transformed.values).all()
    import harmonica as hm
    source = hm.EquivalentSources(depth=100., parallel=False)
    source.fit((np.array([0., 100., 0., 100.]), np.array([0., 0., 100., 100.]), np.zeros(4)), np.array([1., 2., 3., 4.]))
    assert np.isfinite(source.predict((np.array([50.]), np.array([50.]), np.array([0.])))).all()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'projection.tif'
        with rasterio.open(path, 'w', driver='GTiff', width=2, height=2, count=1, dtype='float64',
                           crs=schema.crs, transform=from_origin(0, 100, 50, 50)) as stream:
            stream.write(maps.predictions['As'], 1)
        with rasterio.open(path) as stream:
            assert stream.crs.to_epsg() == 2380
            assert np.isfinite(stream.read(1)).all()
    report['checks'].append('GeoTIFF/CRS')


def _bundle_checks(result):
    import sys
    if not getattr(sys, 'frozen', False):
        return
    bundle = Path(sys._MEIPASS)
    licenses = list((bundle / 'LICENSES').iterdir()) if (bundle / 'LICENSES').is_dir() else []
    if not licenses:
        raise RuntimeError('Embedded LICENSES directory missing or empty')
    metadata_file = bundle / 'THIRD_PARTY_LICENSES' / 'affine' / 'METADATA.txt'
    embedded = metadata_file.read_text(encoding='utf-8').replace('\r\n', '\n')
    import importlib.metadata
    installed = importlib.metadata.distribution('affine').read_text('METADATA').replace('\r\n', '\n')
    if embedded != installed:
        raise RuntimeError('Embedded affine metadata differs from the installed distribution')
    license_files = [path for path in (bundle / 'THIRD_PARTY_LICENSES' / 'affine').rglob('*')
                     if path.is_file() and any(word in path.name.lower() for word in ('license', 'copying', 'notice'))]
    if not license_files:
        raise RuntimeError('Embedded affine license file missing')
    inventory = bundle / 'dependencies.txt'
    if not inventory.is_file() or not inventory.read_text(encoding='utf-8').strip():
        raise RuntimeError('Embedded dependency inventory missing or empty')
    result['checks'].append('Embedded licenses/dependency inventory')


def run(report):
    report.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'started', 'stage': 'startup', 'checks': [], 'started_at': time.time()}
    _write(report, result)
    try:
        result['stage'] = 'Tk/TkAgg workbenches'
        _write(report, result)
        _workbench_checks()
        result['checks'].extend(['Tk magnetic', 'Tk geochem', 'TkAgg'])
        result['stage'] = 'science algorithms'
        _write(report, result)
        _science_checks(result)
        result['stage'] = 'embedded resources'
        _write(report, result)
        _bundle_checks(result)
    except BaseException:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        _write(report, result)
        raise
    result['checks'][:0] = ['multRepl', 'CLR', 'GSTools/PyKrige', 'Harmonica FFT', 'Numba equivalent sources']
    result.update(status='passed', stage='finished', elapsed_seconds=time.time() - result['started_at'])
    _write(report, result)
