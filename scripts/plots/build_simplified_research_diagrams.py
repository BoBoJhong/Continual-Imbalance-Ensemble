"""Compact research overview figures; detailed protocol belongs in captions."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import build_research_method_diagrams as graphics

graphics.WIDTH,graphics.HEIGHT=1360,820
graphics.FONT_FAMILY='Arial'
graphics.FONT=Path('C:/Windows/Fonts/arial.ttf')
graphics.BOLD=Path('C:/Windows/Fonts/arialbd.ttf')
graphics.OUT=graphics.ROOT/'assets/research/diagrams'
BLUE,YELLOW,PEACH,GRAY=graphics.BLUE,graphics.YELLOW,graphics.PEACH,graphics.GRAY
METRIC_LABEL = 'AUC\nF1-score\nRecall'

class PaperFigure(graphics.Figure):
    def __init__(self,name,title,caption):
        self.name,self.caption=name,caption
        self.shapes,self.edges=[],[]
        self.text('title',title,35,24,1290,48,28,True)

def input_data(f,phase=2):
    f.cylinder('dataset','Original\nDataset',20,300,120,100,size=22)
    x=170 if phase==3 else 200
    w=135 if phase==3 else 150
    rows=[('old','Old',125),('new','New',340),('validation','Validation',540),('test','Test',690)]
    if phase==1: rows=[('old','Old',120),('new','New',270),('combined','Old + New\n(Retrain)',420),('validation','Validation',605),('test','Test',700)]
    for key,label,y in rows:
        if key=='validation' and phase==1:
            f.cylinder(key,label,410,y,210,100,size=24)
        else: f.cylinder(key,label,x,y,w,90 if phase==1 else 100,fill=graphics.CYAN if key=='validation' else '#FFFFFF',size=22)
        yy=f.point(key,'left')[1]
        trunk=155 if phase==3 else 165
        f.edge('dataset',key,[(trunk,350),(trunk,yy)])

def panel(f,key,title,x,y,w,h,fill): return f.panel(key,title,x,y,w,h,fill,24)

def prep_sampling(f,baseline=False):
    f.add('preprocess','Preprocess',410,105,210,75,fill=YELLOW,size=24)
    panel(f,'sampling','Sampling',410,205,210,335,BLUE)
    if baseline:
        for i,label in enumerate(['None','Under','Over','Hybrid']):f.card('sampling'+str(i),label,'sampling',300+i*55,42,24)
    else:
        for i,label in enumerate(['Under','Over','Hybrid']):f.card('sampling'+str(i),label,'sampling',300+i*75,55,24)
    for key in ['old','new']+(['combined'] if baseline else []):
        yy=f.point(key)[1]
        f.edge(key,'preprocess',[(385,yy),(385,142.5)])
    f.edge('preprocess','sampling',src='bottom',dst='top')

def pool(f,x=690,w=210):
    panel(f,'pool','Model pool',x,105,w,435,PEACH)
    f.card('old3','Old × 3','pool',220,80,24)
    f.card('new3','New × 3','pool',360,80,24)
    f.add('selectedclassifier','Selected\nclassifier',x+18,450,w-36,70,'text',size=24,parent='pool')

def ensembles(f,x=970,w=240):
    panel(f,'ensembles','Ensemble\nmethods',x,105,w,435,GRAY)
    f.card('static','Static\n49 mixed subsets' if w==240 else 'Static\n49 subsets','ensembles',210,95,22)
    f.card('dcs','DCS','ensembles',335,60,24)
    f.card('des','DES','ensembles',430,60,24)
    f.edge('pool','ensembles')
    xx=x-30
    for key in ['dcs','des']:
        yy=f.point(key,'left')[1]
        f.edge('validation',key,[(xx,590),(xx,yy)])
    f.text('dynamiclabel','Dynamic selection',410 if x==970 else 580,547,390,40,22)

def evaluation(f,x=970,w=240,phase=2):
    fx=650 if phase==3 else 690
    fw=270 if phase==3 else 240
    f.add('freeze','Validation\nchoices fixed',fx,605,fw,100,fill=YELLOW,size=24)
    yy=f.point('validation')[1]
    vx=330 if phase==3 else 385
    f.edge('validation','freeze',[(vx,yy),(vx,655)])
    if phase==3:
        f.edge('ensembles','freeze',[(1300,322.5),(1300,595),(fx+fw/2,595)],dst='top')
    else:
        f.edge('ensembles','freeze',[(1090,570),(955,570),(955,655)],src='bottom',dst='right')
    f.add('inference','Test inference',x,605,w,100,size=24)
    f.edge('freeze','inference')
    test_y=f.point('test')[1]
    tx=345 if phase==3 else 390
    stop=x-25
    f.edge('test','inference',[(tx,test_y),(tx,730),(stop,730),(stop,680)],tf=.75)
    f.text('metrics',METRIC_LABEL,1240,605,115,100,26,True)
    f.edge('inference','metrics')

def stage1():
    f=PaperFigure('phase1-baseline-flow','Phase 1 · Baseline comparison','第一階段：基準模型比較')
    input_data(f,1)
    prep_sampling(f,True)
    panel(f,'classifiers','Classifiers',650,105,280,435,GRAY)
    f.card('traditional','Traditional\nmachine learning','classifiers',235,110,24)
    f.card('deep','Deep learning','classifiers',380,110,24)
    f.edge('sampling','classifiers',sf=(322.5-205)/335)
    panel(f,'baselines','Training\nstrategies',970,105,240,435,PEACH)
    f.card('oldmodel','Old','baselines',220,80,24)
    f.card('newmodel','New / Fine-tune','baselines',345,80,22)
    f.card('retrain','Retrain','baselines',470,50,24)
    f.edge('classifiers','baselines')
    f.add('freeze','Classifier selection\n+ fixed threshold',690,605,240,100,fill=YELLOW,size=24)
    f.edge('validation','freeze')
    f.edge('baselines','freeze',[(1090,570),(810,570)],src='bottom',dst='top')
    f.add('inference','Test inference',970,605,240,100,size=24)
    f.edge('freeze','inference')
    f.edge('baselines','inference',[(1280,322.5),(1280,580),(1090,580)],dst='top')
    f.edge('test','inference',[(385,745),(385,730),(945,730),(945,680)],tf=.75)
    f.text('metrics',METRIC_LABEL,1240,605,115,100,26,True)
    f.edge('inference','metrics')
    f.text('matrix','Parallel settings: classifier × training × sampling',410,752,805,45,22)
    return f

def stage2():
    f=PaperFigure('phase2-ensemble-flow','Phase 2 · Old / New ensembles','第二階段：靜態與動態集成比較')
    input_data(f)
    prep_sampling(f)
    pool(f)
    f.edge('sampling','pool',sf=(322.5-205)/335)
    ensembles(f)
    evaluation(f)
    f.text('consistency','Apply the fitted preprocessing to held-out data',410,744,805,48,22)
    return f

def stage3():
    f=PaperFigure('phase3-fs-ensemble-flow','Phase 3 · Feature selection on ensembles','第三階段：特徵選取對集成分類器的影響')
    input_data(f,3)
    f.add('preprocess','Preprocess',350,105,175,75,fill=YELLOW,size=24)
    panel(f,'fs','Feature\nselection',350,205,175,335,YELLOW)
    for i,label in enumerate(['Filter','Wrapper','Embedded']):f.card('fs'+str(i),label,'fs',295+i*75,55,24)
    for key in ['old','new']:
        yy=f.point(key)[1]
        f.edge(key,'preprocess',[(330,yy),(330,142.5)])
    f.edge('preprocess','fs',src='bottom',dst='top')
    panel(f,'sampling','Sampling',575,105,165,435,BLUE)
    for i,label in enumerate(['Under','Over','Hybrid']):f.card('s'+str(i),label,'sampling',220+i*95,65,24)
    f.edge('fs','sampling',[(550,372.5),(550,322.5)])
    pool(f,790,190)
    f.edge('sampling','pool')
    ensembles(f,1030,200)
    evaluation(f,990,240,3)
    f.text('consistency','Same fitted\ntransforms + features',350,663,280,64,24)
    return f

def structural_check(path):
    tree=ET.parse(path)
    for page in tree.findall('diagram'):
        cells=page.findall('./mxGraphModel/root/mxCell')
        ids=[c.get('id') for c in cells]
        assert len(ids)==len(set(ids))
        assert len([c for c in cells if c.get('id')=='validation'])==1
        assert not any(c.get('id') in ('dsel','cal') for c in cells)
        for c in cells:
            assert 'image=' not in c.get('style','')
            for key in ('parent','source','target'):
                if c.get(key):assert c.get(key) in ids

def main():
    figures=[stage1(),stage2(),stage3()]
    result=ET.Element('mxfile',{'host':'app.diagrams.net','type':'device'})
    qa=[]
    for i,f in enumerate(figures,1):
        for shape in f.shapes:
            if shape.text:shape.size=max(24,shape.size)
        f.render()
        page=ET.SubElement(result,'diagram',{'id':f.name,'name':f'Phase {i}'})
        page.append(f.model())
        sizes=[s.size for s in f.shapes if s.text]
        minimum=min(sizes)*180/25.4*72/graphics.WIDTH
        qa.append({'phase':i,'objects':len(f.shapes),'edges':len(f.edges),'minimum_font_pt_at_180mm':round(minimum,2),'minimum_font_pt_at_90mm':round(minimum/2,2),'recommended_width_mm':180,'separate_validation_nodes':1})
    ET.indent(result,space='  ')
    target=graphics.OUT/'phase1-3-research-methods.drawio'
    ET.ElementTree(result).write(target,encoding='utf-8',xml_declaration=True)
    for path in [graphics.OUT/(f.name+'.drawio') for f in figures]+[target]:structural_check(path)
    (graphics.OUT/'phase1-3-quality-checks.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print('All editable objects and references validated; one Validation block per stage.')
    print(json.dumps(qa))

if __name__=='__main__':main()
