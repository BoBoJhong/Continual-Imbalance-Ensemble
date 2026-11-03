"""Verify final tuned reporting, local symbol explanations and grouped table PDF."""
from pathlib import Path
import hashlib
import json
import re

import fitz
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
LATEX = ROOT / 'thesis/latex'
OUT = ROOT / 'tmp/pdfs/reader-revision-final'
OUT.mkdir(parents=True, exist_ok=True)
REV = Path(__file__).parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

pdf = LATEX / 'build/main.pdf'
report = json.loads((LATEX / 'build/build-report.json').read_text(encoding='utf-8'))
assert report['status'] == 'passed' and sha(pdf) == report['pdf_sha256']
log = (LATEX / 'build/main.log').read_text(encoding='utf-8', errors='replace')
errors = [line for line in log.splitlines() if any(x in line for x in
          ['Overfull', 'Missing character', 'undefined', 'multiply defined', 'destination with the same identifier'])]
assert not errors, errors
doc = fitz.open(pdf)
texts = [p.get_text() for p in doc]
compact = [''.join(t.split()) for t in texts]
assert not any('符號說明' in t for t in compact)
assert not any(any(x in t for x in ['較早版本','擴充版本','標記版本','XGBoost（較早','XGBoost（擴充']) for t in compact)
toc = doc.get_toc()
assert not any('符號說明' in title for _,title,_ in toc)
starts = {}
for name in ['一、緒論','二、文獻探討','三、研究方法','四、實驗結果與討論','五、結論與未來研究']:
    pages = [i+1 for i,t in enumerate(texts) if t.splitlines() and ''.join(t.splitlines()[0].split()) == name]
    assert len(pages) == 1, (name,pages)
    starts[name] = pages[0]
figures = re.findall(r'\\numberline \{([^}]+)\}', (LATEX / 'build/main.lof').read_text(encoding='utf-8'))
tables = re.findall(r'\\numberline \{([^}]+)\}', (LATEX / 'build/main.lot').read_text(encoding='utf-8'))
assert figures == [f'3-{i}' for i in range(1,6)] + [f'4-{i}' for i in range(1,4)]
assert tables == ([f'2-{i}' for i in range(1,3)] + [f'3-{i}' for i in range(1,13)]
                  + [f'4-{i}' for i in range(1,11)] + [f'A-{i}' for i in range(1,13)])
credits = sum(t.count('(資料來源：') for t in compact)
assert credits == 44, credits
assert sum(t.count('資料來源：') for t in compact) == credits
data = pd.read_csv(LATEX / 'generated/chapter-results-20261027/baseline-summary.csv')
assert len(data) == 15 and data.candidate.nunique() == 5
before = pd.read_csv(REV / 'before/generated/chapter-results-20261027/baseline-summary.csv')
expected = before[~before.candidate.isin(['XGBoost（較早版本）','TabICL 標記版本'])].copy()
expected['candidate'] = expected.candidate.replace({'XGBoost（擴充版本）':'XGBoost'})
pd.testing.assert_frame_equal(expected.reset_index(drop=True), data.reset_index(drop=True))
method = (LATEX / 'chapters/03_method.tex').read_text(encoding='utf-8')
for label,count in [('tab:method-ml-search',4),('tab:method-tabm-search',1),('tab:method-fs-parameters',5)]:
    table = next(t for t in re.findall(r'\\begin\{longtable\}.*?\\end\{longtable\}',method,re.S) if '\\label{'+label+'}' in t)
    assert table.count(r'\multirow') == count, (label,table.count(r'\multirow'))
needles = ['表3-4Baseline','表3-5Baseline','表3-6Baseline','表3-12實驗二','表4-1超參數','表4-6實驗一',
           '圖4-1超參數','圖4-2未進行FS','圖4-3各FS','表A-5實驗一','表A-6實驗一','表A-7實驗二','表A-9實驗二']
locations = {needle:[i+1 for i,t in enumerate(compact) if needle in t] for needle in needles}
assert all(locations.values()), locations
pages = set(range(6,12)) | set(range(starts['一、緒論'],starts['三、研究方法']+2))
for group in locations.values():
    for page in group:
        if page > 11:
            pages.update([page, min(page+1,len(doc))])
pages.update(range(starts['四、實驗結果與討論'],starts['五、結論與未來研究']))
pages = sorted(pages)
thumbs = []
for n in pages:
    path = OUT / f'page-{n:02}.png'
    doc[n-1].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(path)
    im = Image.open(path).convert('RGB'); im.thumbnail((350,495)); thumbs.append((n,im))
for begin in range(0,len(thumbs),6):
    canvas=Image.new('RGB',(1120,1060),'#dddddd'); draw=ImageDraw.Draw(canvas)
    for i,(n,im) in enumerate(thumbs[begin:begin+6]):
        x,y=10+(i%3)*370,35+(i//3)*525;canvas.paste(im,(x,y));draw.text((x,y-23),f'PDF page {n}',fill='black')
    canvas.save(OUT/f'contact-{begin//6+1}.png')
proof = dict(status='passed', pages=len(doc), chapter_starts=starts, figure_numbers=figures, table_numbers=tables,
             source_credits_parenthesized=credits, symbol_page_removed=True, primary_tuned_classifiers=list(data.candidate.unique()),
             baseline_scores_match_original=True, table_locations=locations, rendered_pages=pages, log_defects=errors,
             pdf_sha256=sha(pdf), native_warnings=report['warnings'])
(REV / 'document-check.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'PASS: {len(doc)} pages, 44 parenthesized source credits, 15 tuned rows, grouped labels and chapter numbering')
print('Rendered',len(pages),'pages for visual review')
