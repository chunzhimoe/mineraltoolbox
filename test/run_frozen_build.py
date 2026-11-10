import subprocess
import sys
from pathlib import Path

with Path(sys.argv[1]).open('x', encoding='utf-8') as log:
    result = subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '.devin/build/mineraltoolbox.spec'], stdout=log, stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
