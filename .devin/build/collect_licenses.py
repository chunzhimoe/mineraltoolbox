import importlib.metadata
import re
import shutil
import sys
from pathlib import Path

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=False)
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get('Name', 'unknown')
    target = output / re.sub(r'[^A-Za-z0-9_.-]', '_', name)
    target.mkdir(exist_ok=True)
    metadata = distribution.read_text('METADATA') or distribution.read_text('PKG-INFO')
    if metadata is None:
        raise ValueError(f'{name}: neither METADATA nor PKG-INFO is available')
    (target / 'METADATA.txt').write_text(metadata, encoding='utf-8')
    for entry in distribution.files or []:
        if any('license' in part.lower() or 'copying' in part.lower() or 'notice' in part.lower() for part in entry.parts):
            source = Path(distribution.locate_file(entry))
            if source.is_file():
                destination = target / Path(*[p for p in entry.parts if p not in ('..', '.')])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
