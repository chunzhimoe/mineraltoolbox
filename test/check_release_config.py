from pathlib import Path
import ast
import yaml

root = Path(__file__).resolve().parents[1]
workflow = yaml.safe_load((root / '.github/workflows/windows-release.yml').read_text(encoding='utf-8'))
assert set(workflow['jobs']) == {'build', 'release'}
assert workflow['jobs']['build']['runs-on'] == 'windows-2022'
assert workflow['jobs']['release']['needs'] == 'build'
for path in (root / '.devin/build').glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'))
ast.parse((root / '.devin/build/mineraltoolbox.spec').read_text(encoding='utf-8'))
assert not list(root.glob('*.xlsx'))
print('Workflow YAML and packaging Python syntax passed')
