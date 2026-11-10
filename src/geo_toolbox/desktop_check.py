import json
import tempfile
from pathlib import Path


def run(report):
    import tkinter as tk
    import numpy as np
    import pandas as pd
    import ppigrf
    import openpyxl
    import xlrd
    import rasterio
    from rasterio.transform import from_origin
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from . import GeoToolbox, GeochemSchema, GridSpec
    from .geochem_gui import GeochemWorkbench
    from .gui import MagneticWorkbench
    from .magnetic_core import MagneticGrid

    for workbench in (MagneticWorkbench, GeochemWorkbench):
        root = tk.Tk()
        root.withdraw()
        app = workbench(root)
        canvas = FigureCanvasTkAgg(Figure(figsize=(2, 2)), master=root)
        canvas.draw()
        canvas.get_tk_widget().destroy()
        root.update()
        app._on_close()
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
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'passed', 'workbenches': ['magnetic', 'geochem'],
                   'checks': ['TkAgg', 'multRepl', 'CLR', 'GSTools/PyKrige', 'Harmonica FFT', 'Numba equivalent sources', 'GeoTIFF/CRS']}, stream, indent=2)
