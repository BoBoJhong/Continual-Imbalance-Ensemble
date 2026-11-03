"""One-time reader revision for authored tables; generated tables use exporters."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/thesis'))
from export_fixed_statistics import grouped_cells

changes = []
for filename in ('02_literature.tex', '03_method.tex', '04_experiments.tex', 'appendix.tex'):
    path = ROOT / 'thesis/latex/chapters' / filename
    text = path.read_text(encoding='utf-8')

    def credit(match):
        tail = match.string[match.end():]
        extra = '' if tail.startswith(('}', r'\par')) else r'\par '
        return '(資料來源：' + match.group(1) + ')' + extra

    text = re.sub(r'(?<!\()資料來源：([^。\n]+)。', credit, text)

    def table(match):
        whole = match.group(0)
        if r'\multirow' in whole:
            return whole
        marker = r'\endlastfoot'
        if marker not in whole:
            return whole
        head, body = whole.split(marker, 1)
        lines = body.splitlines()
        positions, rows = [], []
        for i, line in enumerate(lines):
            if ' & ' in line and re.search(r'\\\\\*?\s*$', line):
                positions.append(i)
                rows.append(re.sub(r'\s*\\\\\*?\s*$', '', line).split(' & '))
        if len(rows) < 2 or not any(rows[i][0] == rows[i+1][0] for i in range(len(rows)-1)):
            return whole
        cells, ends = grouped_cells(rows, (0,))
        for i, position in enumerate(positions):
            lines[position] = ' & '.join(cells[i]) + (r' \\' if i in ends else r' \\*')
        label = re.search(r'\\label\{([^}]+)\}', whole)
        changes.append((filename, label.group(1) if label else 'unnamed'))
        return head + marker + '\n'.join(lines)

    text = re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}', table, text, flags=re.S)
    path.write_text(text, encoding='utf-8')
print('Merged authored table groups:', changes)
