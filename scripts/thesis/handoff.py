"""Freeze reviewed local files and compare complete remote backups; never access the network."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = {'.tex', '.cls', '.bib', '.png', '.jpg', '.jpeg', '.pdf', '.svg', '.ttf', '.otf', '.json'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe_path(base: Path, name: str) -> Path:
    rel = Path(name)
    path = (base / rel).resolve()
    if rel.is_absolute() or rel.drive or '..' in rel.parts or not path.is_relative_to(base.resolve()):
        raise ValueError(f'Path escapes allowed directory: {name}')
    return path


def target(root: Path) -> tuple[dict, Path]:
    settings = load(root / 'config/thesis_overleaf.json')
    url = urlparse(settings['project_url'])
    project_id = settings['project_id']
    if (not re.fullmatch(r'[0-9a-f]{24}', project_id) or url.scheme != 'https'
            or url.netloc != 'www.overleaf.com' or url.path != '/project/' + project_id
            or url.query or url.fragment):
        raise ValueError('Invalid or inconsistent Overleaf target')
    return settings, safe_path(root, settings['local_source'])


def working_sources(source: Path) -> dict:
    return {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*')
            if p.is_file() and 'build' not in p.relative_to(source).parts}


def verify_data_sources(root: Path, source: Path) -> dict:
    pointer = source / 'generated/current.json'
    if not pointer.exists():
        return {}
    current = load(pointer)
    for name, expected in current['source_sha256'].items():
        if digest(safe_path(root, name)) != expected:
            raise ValueError(f'Result source changed; regenerate and recompile: {name}')
    for name, expected in current['generated_sha256'].items():
        if digest(safe_path(source / 'generated', name)) != expected:
            raise ValueError(f'Generated result changed: {name}')
    return current['source_sha256']


def import_backup(root: Path, archive: Path, output: Path, project_id: str) -> Path:
    """Import the full ZIP obtained after checking the project in the browser.

    The caller supplies observed identity; this is not browser authentication.
    A receipt binds the complete extracted inventory to the retained ZIP.
    """
    settings, _ = target(root)
    if project_id != settings['project_id']:
        raise ValueError('Backup project does not match configured target')
    if output.exists():
        raise ValueError('Use a new backup directory; existing backups are preserved')
    with zipfile.ZipFile(archive) as z:
        files = [i for i in z.infolist() if not i.is_dir()]
        names = [i.filename for i in files]
        if 'main.tex' not in names or len({n.casefold() for n in names}) != len(names):
            raise ValueError('Backup must contain main.tex at its root, without duplicate paths')
        for info in z.infolist():
            canonical = safe_path(output / 'source', info.filename).relative_to((output / 'source').resolve()).as_posix()
            if canonical != info.filename.rstrip('/'):
                raise ValueError('Noncanonical path in source ZIP')
            if '\\' in info.filename or ':' in info.filename or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Unsafe path in source ZIP')
        if sum(i.file_size for i in files) > 512 * 1024 * 1024:
            raise ValueError('Source ZIP exceeds the 512 MiB import limit')
        output.mkdir(parents=True)
        shutil.copyfile(archive, output / 'source.zip')
        source = output / 'source'
        source.mkdir()
        for info in files:
            dest = safe_path(source, info.filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(info))
    receipt = {'project_id': project_id, 'project_url': settings['project_url'],
               'imported_at_utc': datetime.now(timezone.utc).isoformat(),
               'archive_sha256': digest(output / 'source.zip'),
               'files_sha256': {p.relative_to(source).as_posix(): digest(p)
                                for p in source.rglob('*') if p.is_file()},
               'identity_evidence': 'Caller-provided browser observation; not independently authenticated.'}
    (output / 'source-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Imported full source ZIP:', source)
    return source


def verified_backup(remote_root: Path, project_id: str) -> dict:
    receipt = load(remote_root.parent / 'source-receipt.json')
    inventory = {p.relative_to(remote_root).as_posix(): digest(p)
                 for p in remote_root.rglob('*') if p.is_file()}
    if (remote_root.name != 'source' or receipt['project_id'] != project_id
            or 'main.tex' not in inventory or inventory != receipt['files_sha256']
            or digest(remote_root.parent / 'source.zip') != receipt['archive_sha256']):
        raise ValueError('Backup receipt, project, ZIP or complete source inventory differs')
    return receipt


def prepare(root: Path, files: list[str], document: str = 'main') -> Path:
    root = root.resolve()
    settings, source = target(root)
    if not files:
        raise ValueError('Specify the exact files to hand off')
    job = 'main' if document == 'main' else 'results-preview'
    report_name = 'build-report.json' if document == 'main' else 'results-preview-build-report.json'
    report_path = source / 'build' / report_name
    report = load(report_path)
    if report.get('status') != 'passed' or report.get('document') != document:
        raise ValueError('A successful build of the requested document is required')
    if any('Overfull' in w or 'Missing character:' in w for w in report.get('warnings', [])):
        raise ValueError('Resolve missing glyphs or overflowing content before handoff')
    state = working_sources(source)
    if state != report['source_sha256']:
        raise ValueError('Local source changed after compilation; rebuild first')
    pdf = source / 'build' / (job + '.pdf')
    if digest(pdf) != report['pdf_sha256']:
        raise ValueError('Build PDF differs from its report')
    data_sources = verify_data_sources(root, source)
    original_path = root / 'docs/thesis_environment/original-files.sha256.json'
    originals = load(original_path) if original_path.exists() else {}
    records = []
    seen = set()
    for name in files:
        path = safe_path(source, name)
        rel = path.relative_to(source).as_posix()
        if (rel in seen or not path.is_file() or path.suffix.lower() not in EXTENSIONS
                or any(part.startswith('.') for part in Path(rel).parts) or 'build' in Path(rel).parts):
            raise ValueError(f'Invalid, duplicate or excluded handoff file: {name}')
        seen.add(rel)
        records.append({'path': rel, 'sha256': state[rel],
                        'matches_original_template': originals.get(rel) == state[rel]})
    # Check again before freezing to avoid mixing a concurrent edit with the proof.
    if working_sources(source) != state or digest(pdf) != report['pdf_sha256']:
        raise ValueError('Working files changed during handoff preparation')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    package = root / 'thesis/handoffs' / stamp
    package.mkdir(parents=True, exist_ok=False)
    for record in records:
        dest = package / 'files' / record['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((source / record['path']).read_bytes())
        if digest(dest) != record['sha256']:
            raise ValueError('Source changed while copying the handoff')
    (package / 'compiled-preview.pdf').write_bytes(pdf.read_bytes())
    (package / 'build-report.json').write_bytes(report_path.read_bytes())
    manifest = {
        'schema_version': 1, 'status': 'prepared_not_uploaded',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'target': {k: settings[k] for k in ('project_url', 'project_id', 'local_source')},
        'document': document, 'font_mode': report['font_mode'], 'files': records,
        'compiled_source_sha256': state, 'result_source_sha256': data_sources,
        'pdf_sha256': report['pdf_sha256'], 'build_report_sha256': digest(package / 'build-report.json'),
        'authorization': 'A prepared package does not authorize online writing.'
    }
    (package / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Prepared local handoff:', package)
    return package


def check(root: Path, package: Path) -> dict:
    root = root.resolve()
    manifest = load(package / 'manifest.json')
    settings, source = target(root)
    if manifest['target'] != {k: settings[k] for k in ('project_url', 'project_id', 'local_source')}:
        raise ValueError('Handoff target changed; prepare a package for the intended project')
    if working_sources(source) != manifest['compiled_source_sha256']:
        raise ValueError('Working source changed; handoff is stale')
    verify_data_sources(root, source)
    for name, expected in manifest['result_source_sha256'].items():
        if digest(safe_path(root, name)) != expected:
            raise ValueError('Recorded result source changed: ' + name)
    records = manifest['files']
    if not records or len({r['path'] for r in records}) != len(records):
        raise ValueError('Missing or duplicate handoff file records')
    for record in records:
        if (manifest['compiled_source_sha256'].get(record['path']) != record['sha256']
                or digest(safe_path(package / 'files', record['path'])) != record['sha256']):
            raise ValueError('Handoff snapshot differs from compiled source: ' + record['path'])
    report = load(package / 'build-report.json')
    if (digest(package / 'build-report.json') != manifest['build_report_sha256']
            or report.get('status') != 'passed' or report.get('document') != manifest['document']
            or report['source_sha256'] != manifest['compiled_source_sha256']
            or report['pdf_sha256'] != manifest['pdf_sha256']
            or digest(package / 'compiled-preview.pdf') != manifest['pdf_sha256']):
        raise ValueError('Handoff build proof or PDF has changed')
    print('PASS: handoff snapshots, compilation proof, target and working sources match')
    return manifest


def baseline(root: Path, package: Path, remote_root: Path) -> dict:
    manifest = check(root, package)
    dest = package / 'remote-baseline.json'
    if dest.exists():
        raise ValueError('Remote baseline already exists; preserve it and prepare a new handoff if needed')
    if not remote_root.is_dir():
        raise ValueError('Remote root must be a complete, verified downloaded source snapshot')
    receipt = verified_backup(remote_root, manifest['target']['project_id'])
    state = {r['path']: digest(p) if p.is_file() else None for r in manifest['files']
             for p in [safe_path(remote_root, r['path'])]}
    value = {'project_id': manifest['target']['project_id'],
             'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'sha256': state,
             'backup_archive_sha256': receipt['archive_sha256'],
             'evidence': 'Local comparison of a provided full backup; not live browser verification.'}
    dest.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print('Remote baseline recorded locally; this does not authorize overwriting it')
    return value


def plan(root: Path, package: Path, remote_root: Path) -> list[dict]:
    manifest = check(root, package)
    observed = load(package / 'remote-baseline.json')
    if observed['project_id'] != manifest['target']['project_id']:
        raise ValueError('Remote baseline belongs to a different project')
    if not remote_root.is_dir():
        raise ValueError('Remote root must be a newly obtained complete source snapshot')
    verified_backup(remote_root, manifest['target']['project_id'])
    changes = []
    for record in manifest['files']:
        name = record['path']
        path = safe_path(remote_root, name)
        actual = digest(path) if path.is_file() else None
        before = observed['sha256'][name]
        if actual == record['sha256']:
            action = 'already-current'
        elif actual != before:
            action = 'conflict'
        else:
            action = 'create' if actual is None else 'update'
        changes.append({'path': name, 'action': action, 'remote_sha256': actual,
                        'baseline_sha256': before, 'local_sha256': record['sha256']})
    print(json.dumps(changes, indent=2))
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('import-backup')
    p.add_argument('--zip', type=Path, required=True, dest='archive')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--project-id', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--document', choices=('main', 'results'), default='main')
    p.add_argument('--file', action='append', required=True, dest='files')
    for action in ('check', 'baseline', 'plan'):
        p = sub.add_parser(action)
        p.add_argument('--package', type=Path, required=True)
        if action != 'check':
            p.add_argument('--remote-root', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'import-backup':
            import_backup(ROOT, args.archive.resolve(), args.output.resolve(), args.project_id)
        elif args.action == 'prepare':
            prepare(ROOT, args.files, args.document)
        elif args.action == 'check':
            check(ROOT, args.package.resolve())
        elif args.action == 'baseline':
            baseline(ROOT, args.package.resolve(), args.remote_root.resolve())
        else:
            actions = plan(ROOT, args.package.resolve(), args.remote_root.resolve())
            return 1 if any(r['action'] == 'conflict' for r in actions) else 0
    except (OSError, ValueError, KeyError) as exc:
        print('ERROR:', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
