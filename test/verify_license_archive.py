import hashlib
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
release = root / 'release'
exe = release / 'MineralToolbox-windows-x64.exe'
assert exe.is_file() and exe.stat().st_size > 1_000_000, f'missing onefile release EXE: {exe}'
with zipfile.ZipFile(release / 'MineralToolbox-source.zip') as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    assert any(name.endswith('LICENSES/NOTICE-mapgis2qgis.txt') for name in names), \
        'source archive is missing bundled license notices'
    assert any(name.endswith('src/geo_toolbox/gui.py') for name in names)
    print('source ZIP CRC, license notices and package sources passed')
dependencies = (release / 'dependencies.txt').read_text(encoding='utf-8')
assert 'numpy==' in dependencies, 'dependency inventory missing or unexpected'
checksums = (release / 'SHA256SUMS.txt').read_text(encoding='ascii').strip().splitlines()
entries = {line.split('  ', 1)[1]: line.split('  ', 1)[0] for line in checksums}
for name in ('MineralToolbox-windows-x64.exe', 'MineralToolbox-source.zip', 'dependencies.txt'):
    expected = entries.get(name)
    actual = hashlib.sha256((release / name).read_bytes()).hexdigest()
    assert expected == actual, (name, expected, actual)
print('release EXE, source ZIP, dependency inventory and SHA256SUMS verified')
