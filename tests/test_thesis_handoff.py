"""Handoff version binding and remote conflict protection, without a network."""
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('thesis_handoff', Path(__file__).resolve().parents[1] / 'scripts/thesis/handoff.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def backup(root, directory, files):
    archive = directory.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w') as z:
        for name, text in {'main.tex': 'Complete project root', **files}.items():
            z.writestr(name, text)
    return MODULE.import_backup(root, archive, directory, '1' * 24)


@pytest.fixture
def built(tmp_path):
    root = tmp_path
    (root / 'config').mkdir()
    (root / 'config/thesis_overleaf.json').write_text(json.dumps({
        'project_url': 'https://www.overleaf.com/project/' + '1' * 24,
        'project_id': '1' * 24, 'local_source': 'thesis/latex'}))
    source = root / 'thesis/latex'
    (source / 'chapters').mkdir(parents=True)
    (source / 'chapters/03_method.tex').write_text('Reviewed local chapter', encoding='utf-8')
    (source / 'build').mkdir()
    pdf = source / 'build/main.pdf'
    pdf.write_bytes(b'%PDF-fixture-build-proof')
    report = {'status': 'passed', 'document': 'main', 'font_mode': 'strict',
              'source_sha256': MODULE.working_sources(source), 'pdf_sha256': MODULE.digest(pdf), 'warnings': []}
    (source / 'build/build-report.json').write_text(json.dumps(report))
    return root, source


def test_changed_local_file_invalidates_handoff(built):
    root, source = built
    package = MODULE.prepare(root, ['chapters/03_method.tex'])
    MODULE.check(root, package)
    (source / 'chapters/03_method.tex').write_text('Later unreviewed edit')
    with pytest.raises(ValueError, match='stale'):
        MODULE.check(root, package)
    assert (package / 'files/chapters/03_method.tex').read_text() == 'Reviewed local chapter'


@pytest.mark.parametrize('kind', ['target', 'snapshot', 'pdf'])
def test_reject_changed_target_or_build_proof(built, kind):
    root, _ = built
    package = MODULE.prepare(root, ['chapters/03_method.tex'])
    if kind == 'target':
        path = root / 'config/thesis_overleaf.json'
        d = json.loads(path.read_text())
        d.update(project_url='https://www.overleaf.com/project/' + '2' * 24, project_id='2' * 24)
        path.write_text(json.dumps(d))
    elif kind == 'snapshot':
        (package / 'files/chapters/03_method.tex').write_text('Changed frozen file')
    else:
        (package / 'compiled-preview.pdf').write_bytes(b'changed PDF')
    with pytest.raises(ValueError):
        MODULE.check(root, package)


def test_reject_stale_compile_and_path_escape(built):
    root, source = built
    with pytest.raises(ValueError, match='escapes'):
        MODULE.prepare(root, ['../outside.tex'])
    (source / 'chapters/03_method.tex').write_text('Changed after compile')
    with pytest.raises(ValueError, match='after compilation'):
        MODULE.prepare(root, ['chapters/03_method.tex'])


def test_remote_conflict_is_detected_without_writing_remote(built, tmp_path):
    root, _ = built
    package = MODULE.prepare(root, ['chapters/03_method.tex'])
    remote = backup(root, tmp_path / 'downloaded-remote', {'chapters/03_method.tex': 'Old remote chapter'})
    MODULE.baseline(root, package, remote)
    assert MODULE.plan(root, package, remote)[0]['action'] == 'update'
    remote = backup(root, tmp_path / 'downloaded-new', {'chapters/03_method.tex': 'Another collaborator edited this'})
    path = remote / 'chapters/03_method.tex'
    assert MODULE.plan(root, package, remote)[0]['action'] == 'conflict'
    assert path.read_text() == 'Another collaborator edited this'
    remote = backup(root, tmp_path / 'downloaded-current', {'chapters/03_method.tex': 'Reviewed local chapter'})
    assert MODULE.plan(root, package, remote)[0]['action'] == 'already-current'


def test_remote_new_file_cannot_replace_unknown_content(built, tmp_path):
    root, _ = built
    package = MODULE.prepare(root, ['chapters/03_method.tex'])
    remote = backup(root, tmp_path / 'complete-remote-without-chapter', {})
    MODULE.baseline(root, package, remote)
    assert MODULE.plan(root, package, remote)[0]['action'] == 'create'
    remote = backup(root, tmp_path / 'complete-remote-with-new-chapter', {'chapters/03_method.tex': 'New remote file'})
    assert MODULE.plan(root, package, remote)[0]['action'] == 'conflict'


def test_empty_or_modified_backup_cannot_prove_missing_files(built, tmp_path):
    root, _ = built
    package = MODULE.prepare(root, ['chapters/03_method.tex'])
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(OSError):
        MODULE.baseline(root, package, empty)
    remote = backup(root, tmp_path / 'intact', {'chapters/03_method.tex': 'Old chapter'})
    MODULE.baseline(root, package, remote)
    (remote / 'chapters/03_method.tex').unlink()
    with pytest.raises(ValueError, match='inventory differs'):
        MODULE.plan(root, package, remote)


@pytest.mark.parametrize('files', [{}, {'main.tex': 'main', '../escape.tex': 'escape'}])
def test_invalid_zip_is_rejected_before_extraction(built, tmp_path, files):
    root, _ = built
    archive = tmp_path / 'invalid.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name, text in files.items():
            z.writestr(name, text)
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        MODULE.import_backup(root, archive, output, '1' * 24)
    assert not output.exists()
