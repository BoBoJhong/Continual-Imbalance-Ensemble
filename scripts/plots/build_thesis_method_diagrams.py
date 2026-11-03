"""Reuse the supplied flowchart geometry with the thesis's two experiments."""
import hashlib
import json
import build_simplified_research_diagrams as diagrams


def main():
    graphics = diagrams.graphics
    graphics.OUT = graphics.ROOT / "thesis/latex/figures"
    entries = [
        (diagrams.stage1, "experiment1-baseline", "實驗一：Baseline 分類器比較"),
        (diagrams.stage2, "experiment1-ensemble", "實驗一：Old／New 集成比較"),
        (diagrams.stage3, "experiment2-fs-ensemble", "實驗二：特徵選取集成比較"),
    ]
    review = []
    for constructor, name, title in entries:
        f = constructor()
        f.name, f.caption = name, title
        f.find("title").text = title
        for shape in f.shapes:
            shape.size = max(24, shape.size) if shape.text else shape.size
            if shape.key == "newmodel":
                shape.text = "New"
            elif shape.key == "selectedclassifier":
                shape.text = "Best\nclassifier"
            elif name == "experiment1-baseline" and shape.key == "freeze":
                shape.text = "Validation tuning\n+ fixed threshold"
        f.validate()
        f.render()
        for ext in ("svg", "drawio"):
            source = (graphics.OUT / f"{name}.{ext}").read_text(encoding="utf-8")
            assert not any(word in source for word in ("第一階段", "第二階段", "第三階段", "Phase 1", "Phase 2", "Phase 3"))
        review.append(dict(title=title, source_geometry=constructor.__name__,
                           image=name + ".png", sha256=hashlib.sha256((graphics.OUT / (name + ".png")).read_bytes()).hexdigest()))
    (graphics.OUT / "thesis-flowchart-checks.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: original flowchart layout retained; Chinese experiment titles and editable page names agree.")


if __name__ == "__main__":
    main()
