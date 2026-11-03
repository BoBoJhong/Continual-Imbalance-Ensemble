"""Check the delivered PDF, numbering and manuscript source; render reviewed pages."""
import hashlib
import json
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
LATEX = ROOT / "thesis/latex"
OUT = ROOT / "tmp/pdfs/front-four-final"
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

report = json.loads((LATEX / "build/build-report.json").read_text(encoding="utf-8"))
assert report["status"] == "passed"
pdf = LATEX / "build/main.pdf"
assert sha(pdf) == report["pdf_sha256"]
log = (LATEX / "build/main.log").read_text(encoding="utf-8", errors="replace")
bad = [line for line in log.splitlines() if any(k in line for k in
       ("Overfull", "Missing character", "undefined", "multiply defined", "destination with the same identifier"))]
assert not bad, bad
lof = (LATEX / "build/main.lof").read_text(encoding="utf-8")
lot = (LATEX / "build/main.lot").read_text(encoding="utf-8")
figure_numbers = re.findall(r"\\numberline \{([^}]+)\}", lof)
table_numbers = re.findall(r"\\numberline \{([^}]+)\}", lot)
assert figure_numbers == [f"3-{i}" for i in range(1, 6)] + [f"4-{i}" for i in range(1, 4)]
assert table_numbers == ([f"2-{i}" for i in range(1, 3)] + [f"3-{i}" for i in range(1, 13)]
                         + [f"4-{i}" for i in range(1, 11)] + [f"A-{i}" for i in range(1, 13)])
bib = (LATEX / "references.bib").read_text(encoding="utf-8")
keys = re.findall(r"@\w+\{([^,]+),", bib)
assert len(set(keys)) == len(keys)
chapters = [LATEX / "chapters" / f for f in ["01_introduction.tex", "02_literature.tex", "03_method.tex", "04_experiments.tex"]]
text = "\n".join(p.read_text(encoding="utf-8") for p in chapters)
cited = {key.strip() for group in re.findall(r"\\(?:auto)?cite(?:\[[^]]*\])?\{([^}]+)\}", text) for key in group.split(",")}
assert not cited - set(keys), cited - set(keys)
assert not any(x in text for x in ["請自行取代", "引用示例", "研究目標請", "第一階段", "第二階段", "第三階段"])
doc = fitz.open(pdf)
pages_text = [p.get_text() for p in doc]
chapter_starts = {}
for label in ("一、緒論", "二、文獻探討", "三、研究方法", "四、實驗結果與討論", "五、結論與未來研究"):
    matches = [i+1 for i, text in enumerate(pages_text) if text.startswith(label)]
    assert len(matches) == 1, (label, matches)
    chapter_starts[label] = matches[0]
article_pages = [i+1 for i, t in enumerate(pages_text) if "Article 44" in t or "Article e0118432" in t]
assert article_pages
located = {}
captions = ["圖4-1 Baseline", "圖4-2 未進行FS", "圖4-3 各FS", "圖3-2 通用滑動"]
for caption in captions:
    compact = caption.replace(" ", "")
    matches = [i+1 for i,t in enumerate(pages_text) if compact in t.replace(" ", "").replace("\n", "")]
    located[caption] = matches
    assert len(matches) >= 2, (caption, matches)  # list and body
review_pages = sorted(set(list(range(6, chapter_starts["三、研究方法"] + 2))
                          + article_pages + [p for group in located.values() for p in group]
                          + list(range(chapter_starts["四、實驗結果與討論"], chapter_starts["五、結論與未來研究"]))
                          + [len(doc)-2, len(doc)-1, len(doc)]))
geometry_flags = []
for i, page in enumerate(doc):
    if not chapter_starts["一、緒論"] <= i+1 < chapter_starts["五、結論與未來研究"]:
        continue
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            if line["bbox"][0] < 35 or line["bbox"][2] > page.rect.width-25:
                geometry_flags.append({"page":i+1, "bbox":line["bbox"]})
assert not geometry_flags, geometry_flags
thumbs = []
for n in review_pages:
    path = OUT / f"page-{n:02}.png"
    doc[n-1].get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(path)
    im = Image.open(path).convert("RGB")
    im.thumbnail((350,495))
    thumbs.append((n, im))
for start in range(0, len(thumbs), 6):
    canvas = Image.new("RGB", (1120,1060), "#dddddd")
    draw = ImageDraw.Draw(canvas)
    for i, (n, im) in enumerate(thumbs[start:start+6]):
        x, y = 10+(i%3)*370, 35+(i//3)*525
        canvas.paste(im, (x,y))
        draw.text((x, y-23), f"PDF page {n}", fill="black")
    canvas.save(OUT / f"contact-{start//6+1}.png")
result = {"status":"passed", "pdf_sha256":sha(pdf), "pages":len(doc),
          "chapter_pdf_starts":chapter_starts, "figure_numbers":figure_numbers,
          "table_numbers":table_numbers, "cited_entries":len(cited), "bibliography_entries":len(keys),
          "new_figure_locations":located, "reviewed_page_renders":review_pages,
          "geometry_flags":geometry_flags, "log_defects":bad, "native_warnings":report["warnings"],
          "editor_compile":"Record final built-in compile separately; this check covers native PDF",
          "source_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in chapters}}
(Path(__file__).parent / "document-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(f"PASS: {len(doc)} pages; 8 figures; 36 chapter/appendix tables; {len(cited)} verified bibliography keys; {len(review_pages)} rendered pages")
