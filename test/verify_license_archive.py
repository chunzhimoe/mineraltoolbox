import importlib.metadata
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root / 'release/MineralToolbox-windows-x64.zip') as archive:
    assert archive.testzip() is None
    metadata = archive.read('MineralToolbox/THIRD_PARTY_LICENSES/affine/METADATA.txt').decode('utf-8').replace('\r\n', '\n')
    original = importlib.metadata.distribution('affine').read_text('METADATA').replace('\r\n', '\n')
    assert metadata == original
    licenses = [name for name in archive.namelist() if name.startswith('MineralToolbox/THIRD_PARTY_LICENSES/affine/') and ('license' in Path(name).name.lower()) and not name.endswith('/')]
    assert licenses
    assert 'MineralToolbox/MineralToolbox.exe' in archive.namelist()
    print('ZIP CRC, affine original metadata, license presence and EXE presence passed')
