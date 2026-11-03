"""Unpack audited fixed TeX package archives into the private portable snapshot."""
from pathlib import Path
import hashlib, json, shutil, subprocess, tarfile, os

root=Path(__file__).resolve().parents[2]
runtime=root/'.tools/thesis/TinyTeX'
manifest=json.loads((root/'tools/thesis-tex/runtime-packages/manifest.json').read_text(encoding='utf-8'))
if not (runtime/'bin/windows/mktexlsr.exe').is_file():
    raise SystemExit('Extract the fixed Windows TinyTeX snapshot first.')
for record in manifest['packages']:
    archive=root/'tools/thesis-tex/runtime-packages'/record['file']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=record['sha256']:
        raise SystemExit('Package SHA256 mismatch: '+archive.name)
    with tarfile.open(archive,'r:xz') as tf:
        checked=[]
        for member in tf.getmembers():
            rel=Path(member.name)
            prefix=rel.parts[0] if rel.parts else ''
            # TeX Live package archives omit texmf-dist/ from ordinary file paths.
            base=runtime if prefix in ('texmf-dist','tlpkg') else runtime/'texmf-dist'
            dest=(base/rel).resolve()
            if not dest.is_relative_to(runtime.resolve()) or member.name.startswith('/'):
                raise SystemExit('Unsafe archive path: '+member.name)
            if member.isdir(): continue
            if not member.isfile() or prefix not in ('texmf-dist','tlpkg','tex','fonts','scripts','doc','source','bibtex','dvips','dvipdfmx','makeindex','metafont','web2c'):
                raise SystemExit('Unexpected archive member: '+member.name)
            checked.append((member,dest))
        for member,dest in checked:
            dest.parent.mkdir(parents=True,exist_ok=True)
            data=tf.extractfile(member).read()
            if dest.exists() and dest.read_bytes()!=data:
                raise SystemExit('Refusing to overwrite a different existing package: '+str(dest))
            dest.write_bytes(data)
    print('Prepared fixed package:',record['name'])
env=os.environ.copy()
env.update(LANG='C',LC_ALL='C',LC_CTYPE='C')
subprocess.run([str(runtime/'bin/windows/mktexlsr.exe')],env=env,check=True)
print('File database refreshed. Do not run tlmgr update on this frozen snapshot.')
