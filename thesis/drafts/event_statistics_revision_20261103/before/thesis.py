#!/usr/bin/env python3
"""NCU thesis workflow. Python 3.10+, standard library only."""
import argparse, hashlib, json, os, platform, re, shutil, subprocess, sys, zipfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / 'thesis' / 'latex'
BUILD = THESIS / 'build'
LOCK = json.loads((ROOT / 'config' / 'thesis_environment.lock.json').read_text(encoding='utf-8'))
TIMES = ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf')
PACKAGES = ('fontspec.sty','xeCJK.sty','zhnumber.sty','geometry.sty','setspace.sty',
            'amsmath.sty','amssymb.sty','amsthm.sty','booktabs.sty','array.sty',
            'longtable.sty','graphicx.sty','float.sty','caption.sty','pdfpages.sty',
            'chngcntr.sty','titlesec.sty','tocloft.sty','etoolbox.sty','fancyhdr.sty',
            'hyperref.sty','biblatex.sty')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def runtime_env():
    env = os.environ.copy()
    if env.get('NCU_TEX_BIN'):
        env['PATH'] = env['NCU_TEX_BIN'] + os.pathsep + env.get('PATH', '')
    if os.name == 'nt':
        b = ROOT / '.tools/thesis/TinyTeX/bin/windows'
        if b.is_dir(): env['PATH'] = str(b) + os.pathsep + env.get('PATH','')
        env.update(LANG='C', LC_ALL='C', LC_CTYPE='C')
    return env
ENV = runtime_env()
def run(cmd, cwd=THESIS, capture=False):
    result = subprocess.run(cmd, cwd=cwd, env=ENV, text=True, encoding='utf-8',
                            errors='replace', stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    if result.returncode:
        if capture: print(result.stdout[-8000:])
        raise RuntimeError(f'Command failed ({result.returncode}): {cmd[0]}')
    return result.stdout if capture else ''
def executable(name):
    p = shutil.which(name, path=ENV.get('PATH'))
    if not p: raise RuntimeError(f'Missing {name}. See docs/thesis_environment/README.md')
    return p
def verify():
    manifest = json.loads((ROOT/'docs/thesis_environment/original-files.sha256.json').read_text(encoding='utf-8'))
    different = [n for n,h in manifest.items() if not (THESIS/n).is_file() or sha(THESIS/n)!=h]
    if different: raise RuntimeError('Original files differ: '+', '.join(different))
    print(f'PASS: all {len(manifest)} original files match SHA256.')
def doctor(args):
    for p in ('main.tex','config.tex','ncuimthesis.cls','references.bib','fonts/kaiu.ttf'):
        if not (THESIS/p).is_file(): raise RuntimeError('Missing thesis/'+p)
    if args.backend=='docker':
        run([executable('docker'),'version'], cwd=ROOT)
        print('Pinned image:', LOCK['texlive_image'])
    else:
        for n in ('xelatex','biber','latexmk','kpsewhich'):
            print(n, run([executable(n),'--version'],capture=True).splitlines()[0])
        missing=[]
        for p in PACKAGES:
            r=subprocess.run([executable('kpsewhich'),p],env=ENV,cwd=THESIS,capture_output=True)
            if r.returncode or not r.stdout.strip(): missing.append(p)
        if missing: raise RuntimeError('Missing packages: '+', '.join(missing))
    if args.fonts=='strict':
        missing=[n for n in TIMES if not (THESIS/'fonts'/n).is_file()]
        if missing: raise RuntimeError('Strict mode needs licensed fonts: '+', '.join(missing)+'. Run scripts/thesis/prepare-fonts.ps1 on Windows, or supply them manually.')
    print('Doctor passed. This does not certify thesis submission requirements.')
def font_config(mode):
    if mode=='original': return '% Original class font selection is unchanged.\n'
    if mode=='strict':
        for n in ('kaiu.ttf',)+TIMES:
            if not (THESIS/'fonts'/n).is_file(): raise RuntimeError('Missing font: '+n)
        latin=r'\setmainfont[Path=fonts/,BoldFont=timesbd.ttf,ItalicFont=timesi.ttf,BoldItalicFont=timesbi.ttf]{times.ttf}'
    else:
        latin=r'\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,BoldItalicFont=texgyretermes-bolditalic.otf]{texgyretermes-regular.otf}'
    # Select file fonts before the class tries platform-specific family names.
    # Temporarily suppress only its English setmainfont; restore after loading.
    return '\n'.join([r'\AddToHook{package/fontspec/after}{',
        r'\let\ncuBuildOriginalSetMainFont\setmainfont',latin,
        r'\RenewDocumentCommand{\setmainfont}{O{}mO{}}{}',r'}',
        r'\AddToHook{class/ncuimthesis/after}{',
        r'\let\setmainfont\ncuBuildOriginalSetMainFont',
        r'\typeout{NCU BUILD FONT MODE: '+mode+r'}',r'}',''])
def build(args):
    BUILD.mkdir(exist_ok=True)
    job = 'main' if args.document == 'main' else 'results-preview'
    entry = 'environment-main.tex' if args.document == 'main' else 'results-preview.tex'
    report_name = 'build-report.json' if args.document == 'main' else 'results-preview-build-report.json'
    report={'status':'running','backend':args.backend,'font_mode':args.fonts,
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'font_sha256':{p.name:sha(p) for p in sorted((THESIS/'fonts').glob('*.ttf'))},
        'source_sha256':{p.relative_to(THESIS).as_posix():sha(p) for p in THESIS.rglob('*')
            if p.is_file() and 'build' not in p.relative_to(THESIS).parts},
        'source_date_epoch':os.environ.get('SOURCE_DATE_EPOCH','not fixed')}
    cmd=['latexmk','-norc','-r','.latexmkrc','-g','-jobname='+job,entry]
    try:
        if args.document == 'results':
            run([sys.executable, str(ROOT/'scripts/thesis/sync_results.py'), '--check'], capture=True)
        (BUILD/'environment.tex').write_text(font_config(args.fonts),encoding='utf-8')
        if args.backend=='docker':
            report['texlive_image']=LOCK['texlive_image']
            d=[executable('docker'),'run','--rm','--platform',LOCK['docker_platform'],
               '--network=none','--mount',f'type=bind,src={THESIS},dst=/workdir','-w','/workdir']
            if os.name!='nt': d+=['--user',f'{os.getuid()}:{os.getgid()}']
            if 'SOURCE_DATE_EPOCH' in os.environ: d+=['-e','SOURCE_DATE_EPOCH='+os.environ['SOURCE_DATE_EPOCH']]
            run(d+[LOCK['texlive_image']]+cmd)
        else:
            report['platform']=platform.platform()
            report['tool_versions']={n:run([executable(n),'--version'],capture=True).strip()
                                     for n in ('xelatex','biber','latexmk')}
            cmd[0]=executable('latexmk')
            run(cmd)
        if not (BUILD/(job+'.pdf')).is_file(): raise RuntimeError('Compiler did not produce '+job+'.pdf')
        log=(BUILD/(job+'.log')).read_text(encoding='utf-8',errors='replace')
        report['warnings']=[s for s in log.splitlines() if re.search(r'Warning|Missing character:|Overfull|NCU NOTICE',s)]
        report['font_messages']=[s for s in log.splitlines() if 'NCU FONT:' in s or 'NCU BUILD FONT' in s]
        if re.search(r'Citation .+ undefined|There were undefined references|Please \(re\)run Biber', log):
            raise RuntimeError('Unresolved references remain; inspect build/main.log')
        if args.document == 'main' and not (BUILD/'main.bbl').is_file(): raise RuntimeError('Biber output main.bbl is missing')
        report.update(status='passed', document=args.document, pdf_sha256=sha(BUILD/(job+'.pdf')))
        print('Built:',BUILD/(job+'.pdf'))
        print('Warnings:',len(report['warnings']), '(see '+report_name+')')
    except Exception as e:
        report.update(status='failed',error=str(e))
        raise
    finally:
        (BUILD/report_name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def export(args):
    out=ROOT/'dist'
    out.mkdir(exist_ok=True)
    target=out/'NCU_IM_Thesis_Overleaf.zip'
    # Original font files are preserved; newly staged Times fonts require opt-in.
    excluded={'build','legacy_reference','__pycache__'}
    current = THESIS/'generated/current.json'
    selected_snapshot = json.loads(current.read_text(encoding='utf-8'))['snapshot'] if current.is_file() else None
    paths=[p for p in THESIS.rglob('*') if p.is_file()
        and not any(x in excluded for x in p.relative_to(THESIS).parts)
        and not p.name.startswith('.') and p.name!='environment-main.tex'
        and (args.include_times or p.name not in TIMES)
        and not ('generated' in p.relative_to(THESIS).parts and p.suffix == '.csv')
        and (p.relative_to(THESIS).parts[0] != 'generated'
             or len(p.relative_to(THESIS).parts) == 2
             or p.relative_to(THESIS).parts[1] == selected_snapshot)]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths): z.write(p,p.relative_to(THESIS).as_posix())
    print('Exported:',target)
    print('Main document: main.tex; Compiler: XeLaTeX; references: Biber.')
    if not args.include_times: print('Newly staged Times font files excluded. Overleaf uses the original class selection.')
def sync(args):
    source=ROOT/'thesis/overleaf_exports'
    manifest_path=source/'manifest.json'
    if not manifest_path.is_file(): raise RuntimeError('Create overleaf_exports/manifest.json from the example first.')
    m=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(m.get('artifacts'),list): raise RuntimeError('manifest needs artifacts list')
    pending=[]; destinations=set()
    for a in m['artifacts']:
        for field in ('path','sha256','run_id','data_version','code_commit','config'):
            if not a.get(field): raise RuntimeError('Artifact missing '+field)
        rel=Path(a['path'])
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts)<2 or rel.parts[0] not in ('figures','tables'):
            raise RuntimeError('Artifact path must stay in figures/ or tables/')
        s=(source/rel).resolve(); t=(THESIS/rel).resolve()
        if not s.is_relative_to(source.resolve()) or not t.is_relative_to(THESIS.resolve()):
            raise RuntimeError('Resolved path escapes workspace')
        if t in destinations: raise RuntimeError('Duplicate artifact path')
        destinations.add(t)
        if not s.is_file() or sha(s)!=a['sha256']: raise RuntimeError('Artifact hash mismatch: '+str(rel))
        if s.suffix.lower() not in ('.pdf','.png','.jpg','.jpeg','.svg','.tex'): raise RuntimeError('Unsupported artifact type')
        if t.exists() and sha(t)!=sha(s): raise RuntimeError('Refusing to overwrite a different thesis file: '+str(rel))
        pending.append((s,t))
        print(('KEEP ' if t.exists() else 'COPY ')+rel.as_posix())
    if args.apply:
        for s,t in pending:
            if not t.exists(): t.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(s,t)
        print('Copied approved artifacts locally. Overleaf upload is a separate browser step.')
    else: print('Dry run only. Review and repeat with --apply to copy locally.')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('doctor','build'):
        p=sub.add_parser(name)
        p.add_argument('--backend',choices=('local','docker'),default='local')
        p.add_argument('--fonts',choices=('strict','original','preview'),default='strict')
        if name == 'build': p.add_argument('--document',choices=('main','results'),default='main')
    sub.add_parser('verify-original')
    p=sub.add_parser('export-overleaf'); p.add_argument('--include-times',action='store_true')
    p=sub.add_parser('sync-artifacts'); p.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    try:
        if args.action=='verify-original': verify()
        else: {'doctor':doctor,'build':build,'export-overleaf':export,'sync-artifacts':sync}[args.action](args)
    except (RuntimeError,OSError,ValueError,KeyError) as e:
        print('ERROR:',e,file=sys.stderr); return 1
    return 0
if __name__=='__main__': sys.exit(main())
