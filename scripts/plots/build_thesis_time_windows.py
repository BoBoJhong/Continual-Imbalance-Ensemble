"""Draw a generic rolling-window concept and the actual fixed-year case."""
from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'thesis/latex/figures'
FONT = FontProperties(fname=str(ROOT / 'thesis/latex/fonts/kaiu.ttf'))
COLORS = ['#d6e6f4', '#fce0bd', '#e8dcf3', '#d6eadb']


def label(ax, x, y, s, size=13, **kwargs):
    ax.text(x, y, s, fontproperties=FONT, fontsize=size, ha='center', va='center', **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={'height_ratios': [3.1, 2.1]})
    fig.subplots_adjust(left=.035, right=.985, top=.96, bottom=.04, hspace=.18)
    ax.set_xlim(-2, 11.1); ax.set_ylim(-.5, 3.8); ax.axis('off')
    label(ax, 4.5, 3.55, '（a）通用滑動視窗：每次向前移動一個時間單位', 16)
    for j in range(11):
        label(ax, j+.5, 2.98, 't' if j==8 else f't{j-8:+d}', 12)
    regions = [(0, 4, 'Old'), (4, 2, 'New'), (6, 1, 'Val.'), (7, 1, 'Test')]
    for row in range(3):
        y = 2.3-row*.88
        label(ax, -1.1, y+.25, f'視窗 {row+1}', 14)
        for k, (start, width, name) in enumerate(regions):
            x = start + row + 1
            ax.add_patch(Rectangle((x,y), width,.52,facecolor=COLORS[k],edgecolor='#67717a',linewidth=.8))
            label(ax, x+width/2, y+.26, name, 12 if name=='Val.' else 14)
    label(ax, 4.6, -.20, '各視窗各自擬合與驗證；Test 位於該視窗的訓練與驗證期間之後', 13)
    bx.set_xlim(0, 10); bx.set_ylim(0, 3.1); bx.axis('off')
    label(bx, 5, 2.85, '（b）本研究集成與 FS 實驗：固定年份切分', 16)
    widths=[3.1,2.0,2.4,2.2]; start=.15
    for k,(w,name,years) in enumerate(zip(widths,['Old-fit','New-fit','DSEL / Calibration','Test'],['1999–2011','2012–2013','2014','2015–2018'])):
        bx.add_patch(Rectangle((start,1.45),w,.88,facecolor=COLORS[k],edgecolor='#67717a',linewidth=.8))
        label(bx,start+w/2,1.99,name,13)
        label(bx,start+w/2,1.67,years,14)
        start+=w
    label(bx,5,1.03,'2014 年分為互不重疊的 DSEL 與 Calibration；兩者無年內先後順序',13)
    label(bx,5,.59,'Baseline：逐年移動 Old／New 分界，Test 同樣固定為 2015–2018 年',13)
    label(bx,5,.15,'（a）為框架概念；（b）為本次實際設定。區段寬度不代表相同年數。',12)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'time-window-concept.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    meta=dict(scope='generic sliding concept plus actual fixed-year application; no rolling evaluation claimed',
              actual=dict(old=[1999,2011],new=[2012,2013],validation=[2014,2014],test=[2015,2018]),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              png_sha256=hashlib.sha256((OUT/'time-window-concept.png').read_bytes()).hexdigest())
    (OUT/'time-window-concept.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Built generic time-window illustration and fixed-year application')


if __name__=='__main__':
    main()
