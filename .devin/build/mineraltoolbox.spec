from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files, copy_metadata

root = Path(SPECPATH).resolve().parents[1]
datas = [(str(root / 'LICENSES'), 'LICENSES')]
binaries = []
hiddenimports = ['matplotlib.backends.backend_tkagg', 'rasterio.sample', 'rasterio.vrt']
for package in ('geo_toolbox', 'pyrolite', 'gstools', 'pykrige', 'ppigrf', 'xrft', 'openpyxl', 'xlrd', 'rasterio'):
    hiddenimports += collect_submodules(package, filter=lambda name: '.tests' not in name and '.test' not in name)
for package in ('pyrolite', 'ppigrf', 'rasterio'):
    datas += collect_data_files(package)
for package in ('numba', 'llvmlite', 'choclo', 'harmonica', 'verde'):
    data, binary, modules = collect_all(package, filter_submodules=lambda name: '.tests' not in name and '.test' not in name)
    datas += data
    binaries += binary
    hiddenimports += modules
for package in ('geo-toolbox', 'pyrolite', 'gstools', 'PyKrige', 'ppigrf', 'harmonica', 'verde', 'rasterio', 'pyproj', 'matplotlib'):
    datas += copy_metadata(package, recursive=True)

a = Analysis([str(root / '.devin/build/launcher.py')], pathex=[str(root / 'src')],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             hooksconfig={'matplotlib': {'backends': ['TkAgg']}},
             excludes=['PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'IPython', 'notebook', 'pytest'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='MineralToolbox',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False)
diagnostic = EXE(pyz, a.scripts, [], exclude_binaries=True, name='MineralToolbox-check',
                 debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, diagnostic, a.binaries, a.datas, strip=False, upx=False, name='MineralToolbox')
