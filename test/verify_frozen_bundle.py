import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[1]
environment = os.environ.copy()
environment.pop('PYTHONPATH', None)
environment.pop('PYTHONHOME', None)
for entry in ('MineralToolbox-check', 'MineralToolbox'):
    executable = root / 'dist' / (entry + '.exe')
    assert executable.is_file(), f'missing onefile artifact: {executable}'
    report = output / (entry + '-report.json')
    stdout_path = output / (entry + '-stdout.txt')
    stderr_path = output / (entry + '-stderr.txt')
    with tempfile.TemporaryDirectory(prefix=f'mineral-toolbox {entry} ü审-') as directory:
        isolated = Path(directory)
        isolated_exe = isolated / (entry + '.exe')
        shutil.copyfile(executable, isolated_exe)
        siblings = [p.name for p in isolated.iterdir()]
        assert siblings == [entry + '.exe'], siblings
        try:
            with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
                result = subprocess.run([str(isolated_exe), '--self-test', str(report)], cwd=isolated,
                                        env=environment, stdout=stdout, stderr=stderr, timeout=120)
            print(entry, 'exit=', result.returncode, flush=True)
            assert result.returncode == 0, (entry, result.returncode)
            payload = json.loads(report.read_text(encoding='utf-8'))
            assert payload['status'] == 'passed', payload
        finally:
            for log in (stdout_path, stderr_path, report):
                if log.exists():
                    print('---', log, '---', flush=True)
                    print(log.read_text(encoding='utf-8', errors='replace'), flush=True)
        invalid_report = output / (entry + '-invalid-report')
        invalid_report.mkdir()
        with (output / (entry + '-failure-stdout.txt')).open('w', encoding='utf-8') as stdout, (output / (entry + '-failure-stderr.txt')).open('w', encoding='utf-8') as stderr:
            failure = subprocess.run([str(isolated_exe), '--self-test', str(invalid_report)], cwd=isolated,
                                     env=environment, stdout=stdout, stderr=stderr, timeout=30)
        assert failure.returncode == 1, (entry, failure.returncode)
        print(entry, 'invalid-report failure exits without a dialog: code=1', flush=True)
