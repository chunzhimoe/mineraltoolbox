import subprocess
import sys
import tempfile
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files, copy_metadata

root = Path(SPECPATH).resolve().parents[1]
datas = [(str(root / 'LICENSES'), 'LICENSES')]
binaries = []
hiddenimports = ['matplotlib.backends.backend_tkagg', 'rasterio.sample', 'rasterio.vrt', 'shapefile']
for package in ('geo_toolbox', 'pyrolite', 'gstools', 'pykrige', 'ppigrf', 'xrft', 'openpyxl', 'xlrd', 'rasterio'):
    hiddenimports += collect_submodules(package, filter=lambda name: '.tests' not in name and '.test' not in name)
for package in ('pyrolite', 'ppigrf', 'rasterio'):
    datas += collect_data_files(package)
for package in ('numba', 'llvmlite', 'choclo', 'harmonica', 'verde'):
    data, binary, modules = collect_all(package, filter_submodules=lambda name: '.tests' not in name and '.test' not in name)
    datas += data
    binaries += binary
    hiddenimports += modules
for package in ('geo-toolbox', 'pyrolite', 'gstools', 'PyKrige', 'ppigrf', 'harmonica', 'verde', 'rasterio', 'pyproj', 'matplotlib', 'pyshp'):
    datas += copy_metadata(package, recursive=True)

with tempfile.TemporaryDirectory(prefix='mineraltoolbox-build-') as resource_directory:
    resources = Path(resource_directory)
    licenses_dir = resources / 'THIRD_PARTY_LICENSES'
    subprocess.run([sys.executable, str(root / '.devin/build/collect_licenses.py'), str(licenses_dir)], check=True)
    if not (licenses_dir / 'affine' / 'METADATA.txt').is_file():
        raise RuntimeError('THIRD_PARTY_LICENSES collection produced no affine METADATA')
    frozen = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True,
                            capture_output=True, text=True)
    dependencies = resources / 'dependencies.txt'
    dependencies.write_text(frozen.stdout, encoding='utf-8')
    if 'numpy==' not in dependencies.read_text(encoding='utf-8'):
        raise RuntimeError('pip freeze inventory is unexpectedly empty')
    datas += [(str(licenses_dir), 'THIRD_PARTY_LICENSES'), (str(dependencies), '.')]

    a = Analysis([str(root / '.devin/build/launcher.py')], pathex=[str(root / 'src')],
                 binaries=binaries, datas=datas, hiddenimports=hiddenimports,
                 hooksconfig={'matplotlib': {'backends': ['TkAgg']}},
                 excludes=['PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'IPython', 'notebook', 'pytest'],
                 noarchive=False)
    pyz = PYZ(a.pure)
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, name='MineralToolbox',
              debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False)
    diagnostic = EXE(pyz, a.scripts, a.binaries, a.datas, name='MineralToolbox-check',
                     debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
