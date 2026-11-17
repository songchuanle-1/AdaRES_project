"""Repair only the unavailable template boundary; retain failed v1 unchanged."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
original = root / 'scripts/create_y88_execution_helpers.py'
source = original.read_text(encoding='utf-8')
old = "source = source[:source.index('\\ndef activate(token):')]"
new = "source = source[:source.index(\"\\nif __name__ == '__main__':\")]"
assert source.count(old) == 1
source = source.replace(old, new)
assert not (root / 'scripts/advance_y88.py').exists()
assert not (root / 'scripts/test_y88_admission.py').exists()
exec(compile(source, str(original), 'exec'), {'__file__': __file__, '__name__': '__main__'})
