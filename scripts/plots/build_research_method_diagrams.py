"""Shared native rendering primitives for the current three-stage diagrams.

The reference's visual grammar is used: cylinders, pastel method panels,
white method cards, orthogonal arrows, and a separate held-out test route.
New is the project's new-data / Fine-tune baseline; no Old-to-New weight
transfer is asserted. Experimental code and result artifacts are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import math
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/research/diagrams"
WIDTH, HEIGHT, SCALE = 1740, 1160, 2
INK = "#333333"
BLUE = "#D8E7FC"
CYAN = "#BCE3EF"
YELLOW = "#FFF1C9"
PEACH = "#FFE6CC"
GRAY = "#F3F3F3"
FONT_FAMILY = "Segoe UI"
FONT = Path("C:/Windows/Fonts/segoeui.ttf")
BOLD = Path("C:/Windows/Fonts/segoeuib.ttf")
CJK = Path("C:/Windows/Fonts/msjh.ttc")
CJK_BOLD = Path("C:/Windows/Fonts/msjhbd.ttc")


def font(size: int, bold: bool = False, text: str = ""):
    chinese = any(ord(ch) > 0x2FFF for ch in text)
    path = (CJK_BOLD if bold else CJK) if chinese else (BOLD if bold else FONT)
    return ImageFont.truetype(str(path), size * SCALE)


@dataclass
class Shape:
    key: str
    text: str
    x: float
    y: float
    w: float
    h: float
    kind: str = "box"
    fill: str = "#FFFFFF"
    size: int = 21
    bold: bool = False
    parent: str = "1"


class Figure:
    def __init__(self, name: str, title: str, subtitle: str, caption: str):
        self.name, self.caption = name, caption
        self.shapes: list[Shape] = []
        self.edges = []
        self.text("title", title, 40, 28, 1660, 48, 30, True)
        self.text("subtitle", subtitle, 40, 80, 1660, 36, 20)

    def add(self, key, text, x, y, w, h, kind="box", fill="#FFFFFF", size=21, bold=False, parent="1"):
        self.shapes.append(Shape(key, text, x, y, w, h, kind, fill, size, bold, parent))
        return key

    def text(self, key, text, x, y, w, h, size=20, bold=False):
        return self.add(key, text, x, y, w, h, "text", size=size, bold=bold)

    def cylinder(self, key, text, x, y, w=165, h=90, fill="#FFFFFF", size=21):
        return self.add(key, text, x, y, w, h, "cylinder", fill, size)

    def panel(self, key, title, x, y, w, h, fill, size=23):
        self.add(key, "", x, y, w, h, "panel", fill)
        self.add(key + "_title", title, x + 10, y + 16, w - 20, 66, "text", size=size, bold=True, parent=key)
        return key

    def card(self, key, text, panel, y, h=46, size=20):
        p = self.find(panel)
        return self.add(key, text, p.x + 18, y, p.w - 36, h, size=size, parent=panel)

    def find(self, key):
        return next(s for s in self.shapes if s.key == key)

    def point(self, key, side="right", fraction=.5):
        s = self.find(key)
        if side == "right": return (s.x + s.w, s.y + s.h * fraction)
        if side == "left": return (s.x, s.y + s.h * fraction)
        if side == "top": return (s.x + s.w * fraction, s.y)
        return (s.x + s.w * fraction, s.y + s.h)

    def edge(self, source, target, points=(), src="right", dst="left", sf=.5, tf=.5, dashed=False):
        self.edges.append((source, target, [self.point(source, src, sf), *points, self.point(target, dst, tf)], src, dst, sf, tf, dashed))

    def finish(self, note1: str, note2: str):
        self.text("note1", note1, 50, 1010, 1230, 40, 19)
        self.text("note2", note2, 50, 1054, 1230, 40, 18)
        self.text("caption", self.caption, 40, 1100, 1660, 45, 29, True)

    def validate(self):
        keys = [s.key for s in self.shapes]
        assert len(keys) == len(set(keys))
        dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        for s in self.shapes:
            assert 0 <= s.x and 0 <= s.y and s.x + s.w <= WIDTH and s.y + s.h <= HEIGHT, s
            if s.parent != "1":
                p = self.find(s.parent)
                assert p.x <= s.x and s.x + s.w <= p.x + p.w
                assert p.y <= s.y and s.y + s.h <= p.y + p.h
            ff = font(s.size, s.bold, s.text)
            margin = 14 if s.kind not in ("text", "panel") else 4
            for line in s.text.splitlines():
                assert dummy.textlength(line, font=ff) <= (s.w - margin) * SCALE, (s.key, line, s.w)
            assert len(s.text.splitlines()) * s.size * 1.25 <= s.h - (22 if s.kind == "cylinder" else 2), (s.key, s.text)
        for a, b, points, *_ in self.edges:
            assert a in keys and b in keys
            for point in points:
                assert 0 <= point[0] <= WIDTH and 0 <= point[1] <= HEIGHT
            for p, q in zip(points, points[1:]):
                assert p[0] == q[0] or p[1] == q[1], (a, b, p, q)

    def model(self):
        model = ET.Element("mxGraphModel", {"dx":str(WIDTH), "dy":str(HEIGHT), "grid":"1", "gridSize":"10", "guides":"1", "tooltips":"1", "connect":"1", "arrows":"1", "fold":"1", "page":"1", "pageScale":"1", "pageWidth":str(WIDTH), "pageHeight":str(HEIGHT), "math":"0", "shadow":"0"})
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id":"0"})
        ET.SubElement(root, "mxCell", {"id":"1", "parent":"0"})
        for s in self.shapes:
            family = "Microsoft JhengHei" if any(ord(ch) > 0x2FFF for ch in s.text) else FONT_FAMILY
            style = f"html=0;whiteSpace=wrap;fontFamily={family};fontSize={s.size};fontColor={INK};fontStyle={1 if s.bold else 0};align=center;verticalAlign=middle;spacing=6;"
            if s.kind == "text": style += "text;strokeColor=none;fillColor=none;"
            elif s.kind == "panel": style += f"rounded=1;arcSize=14;strokeColor=none;fillColor={s.fill};container=1;collapsible=0;"
            elif s.kind == "cylinder": style += f"shape=cylinder;size=15;boundedLbl=1;strokeColor={INK};strokeWidth=1.6;fillColor={s.fill};"
            else: style += f"rounded=0;strokeColor={INK};strokeWidth=1.5;fillColor={s.fill};"
            cell = ET.SubElement(root, "mxCell", {"id":s.key, "value":s.text, "style":style, "vertex":"1", "parent":s.parent})
            parent = self.find(s.parent) if s.parent != "1" else None
            ET.SubElement(cell, "mxGeometry", {"x":str(s.x - (parent.x if parent else 0)), "y":str(s.y - (parent.y if parent else 0)), "width":str(s.w), "height":str(s.h), "as":"geometry"})
        def port(side, frac):
            return {"right":(1, frac), "left":(0, frac), "top":(frac, 0), "bottom":(frac, 1)}[side]
        for i, (a, b, points, src, dst, sf, tf, dashed) in enumerate(self.edges):
            ex, ey = port(src, sf); ix, iy = port(dst, tf)
            style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=block;endFill=1;endSize=8;strokeColor={INK};strokeWidth=1.6;exitX={ex};exitY={ey};exitPerimeter=0;entryX={ix};entryY={iy};entryPerimeter=0;"
            if dashed: style += "dashed=1;dashPattern=6 4;"
            cell = ET.SubElement(root, "mxCell", {"id":f"edge{i}", "style":style, "edge":"1", "parent":"1", "source":a, "target":b})
            geom = ET.SubElement(cell, "mxGeometry", {"relative":"1", "as":"geometry"})
            if len(points) > 2:
                array = ET.SubElement(geom, "Array", {"as":"points"})
                for x, y in points[1:-1]: ET.SubElement(array, "mxPoint", {"x":str(x), "y":str(y)})
        return model

    def render(self):
        self.validate()
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">', '<rect width="100%" height="100%" fill="white"/>', '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#333333"/></marker></defs>']
        im = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "white")
        d = ImageDraw.Draw(im)
        def box(x, y, w, h): return (int(x*SCALE), int(y*SCALE), int((x+w)*SCALE), int((y+h)*SCALE))
        # Shape backgrounds first; arrows second; text last.
        for s in self.shapes:
            x,y,w,h = s.x,s.y,s.w,s.h
            if s.kind == "text": continue
            if s.kind == "panel":
                d.rounded_rectangle(box(x,y,w,h), radius=26*SCALE, fill=s.fill)
                svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="{s.fill}"/>')
            elif s.kind == "cylinder":
                r = 13
                d.ellipse(box(x,y+h-2*r,w,2*r),fill=s.fill,outline=INK,width=3)
                d.rectangle(box(x,y+r,w,h-2*r),fill=s.fill)
                d.line([(x*SCALE,(y+r)*SCALE),(x*SCALE,(y+h-r)*SCALE)],fill=INK,width=3)
                d.line([((x+w)*SCALE,(y+r)*SCALE),((x+w)*SCALE,(y+h-r)*SCALE)],fill=INK,width=3)
                d.ellipse(box(x,y,w,2*r),fill=s.fill,outline=INK,width=3)
                svg.append(f'<path d="M {x} {y+r} L {x} {y+h-r} A {w/2} {r} 0 0 0 {x+w} {y+h-r} L {x+w} {y+r}" fill="{s.fill}" stroke="{INK}" stroke-width="1.6"/>')
                svg.append(f'<ellipse cx="{x+w/2}" cy="{y+r}" rx="{w/2}" ry="{r}" fill="{s.fill}" stroke="{INK}" stroke-width="1.6"/>')
            else:
                d.rectangle(box(x,y,w,h),fill=s.fill,outline=INK,width=3)
                svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{s.fill}" stroke="{INK}" stroke-width="1.5"/>')
        for _, _, points, *rest in self.edges:
            dashed = rest[-1]
            for p,q in zip(points,points[1:]):
                if dashed:
                    length = math.dist(p,q)
                    if not length: continue
                    for step in range(0,int(length),12):
                        a,b = step/length,min(step+7,length)/length
                        d.line([((p[0]+(q[0]-p[0])*a)*SCALE,(p[1]+(q[1]-p[1])*a)*SCALE),((p[0]+(q[0]-p[0])*b)*SCALE,(p[1]+(q[1]-p[1])*b)*SCALE)],fill=INK,width=3)
                else: d.line([(p[0]*SCALE,p[1]*SCALE),(q[0]*SCALE,q[1]*SCALE)],fill=INK,width=3)
            p,q=points[-2:]
            angle=math.atan2(q[1]-p[1],q[0]-p[0])
            tip=(q[0]*SCALE,q[1]*SCALE)
            triangle=[tip]+[((q[0]-10*math.cos(angle)+sgn*4*math.sin(angle))*SCALE,(q[1]-10*math.sin(angle)-sgn*4*math.cos(angle))*SCALE) for sgn in (-1,1)]
            d.polygon(triangle,fill=INK)
            dash = ' stroke-dasharray="7 5"' if dashed else ''
            coords = " ".join(f"{x},{y}" for x,y in points)
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{INK}" stroke-width="1.6" marker-end="url(#arrow)"{dash}/>')
        for s in self.shapes:
            lines=s.text.splitlines()
            if not lines: continue
            ff=font(s.size,s.bold,s.text)
            center_y=s.y+s.h/2+(6 if s.kind=="cylinder" else 0)
            lineheight=s.size*1.25
            top=center_y-(len(lines)-1)*lineheight/2
            for j,line in enumerate(lines):
                yy=top+j*lineheight
                d.text(((s.x+s.w/2)*SCALE,yy*SCALE),line,font=ff,fill=INK,anchor="mm")
                svg.append(f'<text x="{s.x+s.w/2}" y="{yy}" text-anchor="middle" dominant-baseline="central" font-family="{FONT_FAMILY},Microsoft JhengHei,sans-serif" font-size="{s.size}" font-weight="{700 if s.bold else 400}" fill="{INK}">{escape(line)}</text>')
        svg.append('</svg>')
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/f"{self.name}.svg").write_text("\n".join(svg),encoding="utf-8")
        im.save(OUT/f"{self.name}.png",dpi=(300,300))
        mxfile=ET.Element("mxfile",{"host":"app.diagrams.net","type":"device"})
        page=ET.SubElement(mxfile,"diagram",{"id":self.name,"name":self.caption})
        page.append(self.model())
        ET.indent(mxfile,space="  ")
        path=OUT/f"{self.name}.drawio"
        ET.ElementTree(mxfile).write(path,encoding="utf-8",xml_declaration=True)
        ET.parse(path)
        print(f"{self.name}: {len(self.shapes)} editable objects, {len(self.edges)} connectors; SVG and PNG exported")



if __name__ == "__main__":
    from build_simplified_research_diagrams import main
    main()
