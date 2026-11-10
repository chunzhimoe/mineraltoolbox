import importlib.metadata
import runpy
import sys
from email.errors import HeaderParseError
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / '.devin/build/collect_licenses.py'


@pytest.mark.parametrize('metadata_file', ['METADATA', 'PKG-INFO'])
def test_original_metadata_and_license_are_retained(tmp_path, monkeypatch, metadata_file):
    raw = 'Name: affine\nVersion: 3.0.1\nDescription: Affine\n======\n.. image:: example\n    :target: example\n\nUnicode: 地质\n'
    entry = Path('affine.dist-info/licenses/LICENSE.txt')
    source = tmp_path / 'installed' / entry
    source.parent.mkdir(parents=True)
    source.write_text('Copyright example\nFull license terms\n', encoding='utf-8')

    class Metadata(dict):
        def __str__(self):
            raise HeaderParseError('header value appears to contain an embedded header')

    class Distribution:
        metadata = Metadata(Name='affine')
        files = [entry]

        def read_text(self, name):
            return raw if name == metadata_file else None

        def locate_file(self, name):
            return tmp_path / 'installed' / name

    output = tmp_path / 'licenses'
    monkeypatch.setattr(importlib.metadata, 'distributions', lambda: [Distribution()])
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), str(output)])
    runpy.run_path(str(SCRIPT), run_name='__main__')
    assert (output / 'affine/METADATA.txt').read_text(encoding='utf-8') == raw
    assert (output / 'affine' / entry).read_bytes() == source.read_bytes()


def test_missing_original_metadata_is_not_silently_omitted(tmp_path, monkeypatch):
    class Distribution:
        metadata = {'Name': 'broken-package'}
        files = []

        def read_text(self, name):
            return None

    monkeypatch.setattr(importlib.metadata, 'distributions', lambda: [Distribution()])
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), str(tmp_path / 'licenses')])
    with pytest.raises(ValueError, match='broken-package.*METADATA.*PKG-INFO'):
        runpy.run_path(str(SCRIPT), run_name='__main__')
