"""Audit local code and cleanup integrity without training or changing results."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def audit_code(root: Path) -> dict:
    counts = Counter()
    syntax_issues, missing_imports, top_level_io = [], [], []
    for folder in ('experiments', 'scripts', 'src', 'tests', 'examples'):
        for path in sorted((root / folder).rglob('*.py')):
            relative = path.relative_to(root).as_posix()
            if '__pycache__' in path.parts:
                continue
            counts['historical' if '/legacy/' in relative else folder] += 1
            try:
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            except (SyntaxError, UnicodeError) as exc:
                syntax_issues.append({'path': relative, 'error': str(exc)})
                continue
            for node in ast.walk(tree):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) and not node.level
                           else [entry.name for entry in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    if not module or module.split('.')[0] not in {'src', 'scripts', 'experiments'}:
                        continue
                    target = root.joinpath(*module.split('.'))
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        missing_imports.append({'path': relative, 'line': node.lineno, 'module': module})
            if '/legacy/' not in relative:
                for statement in tree.body:
                    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If)):
                        continue
                    for node in ast.walk(statement):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                            if node.func.attr in {'read_csv', 'to_csv', 'write_text', 'mkdir', 'savefig'}:
                                top_level_io.append({'path': relative, 'line': node.lineno,
                                                     'call': ast.unparse(node.func)})
    return {'python_files_by_scope': dict(counts), 'syntax_issues': syntax_issues,
            'missing_local_absolute_imports': missing_imports,
            'top_level_data_io_candidates': top_level_io,
            'limitations': 'Import targets checked statically; optional packages, relative imports, CLI execution and output validity require separate checks.'}


def audit_results(root: Path, ledger: dict) -> dict:
    baseline = json.loads((root / ledger['baseline_catalog']).read_text(encoding='utf-8'))
    removed = {entry['path'] for entry in ledger['files']}
    preserved, changed, resurrected = 0, [], []
    for record in baseline['files']:
        path = root / record['path']
        if record['path'] in removed:
            if path.exists():
                resurrected.append(record['path'])
            continue
        if not path.is_file():
            changed.append({'path': record['path'], 'reason': 'missing'})
        else:
            stat = path.stat()
            if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
                changed.append({'path': record['path'], 'reason': 'size or mtime changed'})
            else:
                preserved += 1
    hash_changes = [path for path, expected in ledger.get('protected_hashes', {}).items()
                    if not (root / path).is_file() or sha256(root / path) != expected]
    replacements = []
    for entry in ledger['files']:
        if entry['action'].startswith('duplicate'):
            destination = root / entry['replacement_path']
            replacements.append({'removed': entry['path'], 'retained': entry['replacement_path'],
                                 'matching_sha256': destination.is_file() and sha256(destination) == entry['sha256']})
    return {'preserved_result_files': preserved, 'changed_or_missing_retained_results': changed,
            'removed_paths_reappeared': resurrected, 'protected_hash_changes': hash_changes,
            'duplicate_replacements': replacements,
            'limitations': 'Preservation checks do not recompute scientific metrics or validate all past research protocols.'}


def audit_baseline_sources(root: Path) -> list[dict]:
    sources = [
        'results/phase1_baseline/xgb/tuned/bankruptcy_year_splits_xgb_raw.csv',
        'results/phase1_baseline/lgbm/bk_lgb_tuned48_20260421/tuned/bankruptcy_year_splits_lgb_raw.csv',
        'results/phase1_baseline/random_forest/bankruptcy_year_splits_rf_raw.csv',
        'results/phase1_baseline/logistic_regression/bk_lr_tuned4_20260421/tuned/bankruptcy_year_splits_lr_raw.csv',
        'results/phase1_baseline/tabm/tuned_light_full_20260410/tuned/bankruptcy_year_splits_tabm_raw.csv',
        'results/phase1_baseline/tabicl/tabicl_gpu_20260506_setA_temp075_ctx2048/bankruptcy_year_splits_tabicl_raw.csv',
    ]
    records = []
    for relative in sources:
        path = root / relative
        if not path.is_file():
            records.append({'path': relative, 'exists': False})
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        records.append({'path': relative, 'exists': True, 'rows': len(rows),
                        'splits': len({row['split'] for row in rows}), 'sha256': sha256(path),
                        'status': 'historical protocol, currently referenced by result figures'})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=ROOT / 'docs/REMOVED_RESULTS_20261012.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding='utf-8'))
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Maintenance audit; no training, numerical-result edits, or thesis edits',
              'ledger': args.ledger.resolve().relative_to(ROOT).as_posix(),
              'generator_sha256': sha256(Path(__file__)),
              'code': audit_code(ROOT), 'results': audit_results(ROOT, ledger),
              'current_chart_baseline_sources': audit_baseline_sources(ROOT)}
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT / 'docs/audits'):
            parser.error('--output must stay below docs/audits')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({'output': str(args.output) if args.output else None,
                      'python_files': sum(report['code']['python_files_by_scope'].values()),
                      'syntax_issues': report['code']['syntax_issues'],
                      'missing_local_imports': report['code']['missing_local_absolute_imports'],
                      'preserved_result_files': report['results']['preserved_result_files'],
                      'preservation_issues': report['results']['changed_or_missing_retained_results'],
                      'protected_hash_changes': report['results']['protected_hash_changes']}, ensure_ascii=False))
    if not args.output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report['code']['syntax_issues'] or report['code']['missing_local_absolute_imports']
                    or report['results']['changed_or_missing_retained_results']
                    or report['results']['protected_hash_changes'] or report['results']['removed_paths_reappeared']
                    or any(not row['matching_sha256'] for row in report['results']['duplicate_replacements'])))


if __name__ == '__main__':
    raise SystemExit(main())
