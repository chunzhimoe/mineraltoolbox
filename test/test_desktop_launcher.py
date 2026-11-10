import json
import runpy
import sys
import tempfile
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / '.devin/build/launcher.py'


def test_self_test_failure_exits_without_modal_dialog(tmp_path, monkeypatch):
    from geo_toolbox import desktop_check
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['MineralToolbox.exe', '--self-test', str(report)])
    monkeypatch.setattr(sys, 'stderr', None)
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path))

    def fail(path):
        assert path == report
        assert json.loads(report.read_text(encoding='utf-8'))['stage'] == 'importing desktop_check'
        raise ModuleNotFoundError("No module named 'rasterio.serde'")

    monkeypatch.setattr(desktop_check, 'run', fail)
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(LAUNCHER), run_name='__main__')
    assert error.value.code == 1
    assert 'rasterio.serde' in (tmp_path / 'mineraltoolbox-startup-error.txt').read_text(encoding='utf-8')


def test_self_test_success_does_not_open_workbench_selector(tmp_path, monkeypatch):
    from geo_toolbox import desktop_check
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['MineralToolbox.exe', '--self-test', str(report)])
    calls = []
    monkeypatch.setattr(desktop_check, 'run', lambda path: calls.append(path))
    runpy.run_path(str(LAUNCHER), run_name='__main__')
    assert calls == [report]
    assert json.loads(report.read_text(encoding='utf-8'))['stage'] == 'importing desktop_check'
