import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geo_toolbox.desktop_check import run

run(Path(sys.argv[1]))
