import pandas as pd, numpy as np, json, re, hashlib, urllib.parse, pathlib
p=pathlib.Path(__file__).resolve().parents[1]; f=p/'frontend'; out=f/'src'; pub=f/'public'; raw=pd.read_csv(pub/'raw.csv'); clean=pd.read_csv(pub/'cleaned.csv'); rules=json.loads((p/'backend/rules.json').read_text()); metrics=json.loads((p/'backend/metrics.json').read_text())
def counts(s):return {str(k):int(v) for k,v in s.value_counts().items()}
def ranges(d):return {c:[round(float(d[c].min()),3),round(float(d[c].median()),3),round(float(d[c].max()),3)] for c in ['Temp_C','Rain_mm','Fertilizer_kg_ha','Land_Hectares','Yield_per_Hectare']}
profiles={}
for name,g in clean.groupby('Crop'):
 profiles[name]={'n':len(g),'ranges':ranges(g),'meanYield':round(float(g.Yield_per_Hectare.mean()),3),'irrigation':[{'name':k,'mean':round(float(v.Yield_per_Hectare.mean()),3),'n':len(v)} for k,v in g.groupby('Irrigation_Type')],'categories':counts(g.Suitability_Category),'observedProvinces':sorted(g.Province.unique().tolist()),'zero':int((g.Yield_per_Hectare==0).sum())}
report=(pub/'research/reproduced.txt').read_text(); qual=report.split('STEP 2: Data quality')[1].split('dtype:')[0]; checks={k.strip():int(v) for k,v in re.findall(r'^(.+?)\s+(\d+)\s*$',qual,re.M)}
summary={'raw':len(raw),'clean':len(clean),'dirty':500,'recovered':50,'quarantined':450,'columns':list(clean.columns),'missing':{k:int(v) for k,v in raw.isna().sum().items() if v},'checks':checks,'categories':counts(clean.Suitability_Category),'province':counts(clean.Province),'crop':counts(clean.Crop),'year':counts(clean.Year),'irrigation':counts(clean.Irrigation_Type),'zero':int((clean.Yield_per_Hectare==0).sum()),'ranges':ranges(clean),'profiles':profiles,'ordinary':len(clean[clean.Crop!='Saffron']),'saffron':len(clean[clean.Crop=='Saffron'])}
cl=(pub/'research/classification.txt').read_text(); reg=(pub/'research/regression.txt').read_text(); cv=(pub/'research/cross-validation.txt').read_text(); opt=(pub/'research/optimization.txt').read_text()
classification=[dict(name=n,accuracy=float(a),f1=float(b)*100) for n,a,b in re.findall(r'\[\d+/13\] Training (.*?) \.\.\.\s*Training completed\.\s*Training accuracy: [\d.]+ %\s*Testing accuracy: ([\d.]+) %\s*Testing Macro F1: ([\d.]+)',cl)]
for name,a,b in [('Tuned Decision Tree',95.22,0.9481),('Tuned KNN',89.02,0.8520)]:
 if not any(x['name']==name for x in classification): classification.append(dict(name=name,accuracy=a,f1=b*100))
regsets={}
for k,t in [('ordinary',reg.split('ORDINARY CROP REGRESSION')[1].split('SAFFRON REGRESSION')[0]),('saffron',reg.split('SAFFRON REGRESSION')[1])]:
 regsets[k]=[dict(name=n,mae=float(a),rmse=float(b),r2=float(r)) for n,r,a,b in re.findall(r'\[\d+/11\] Training (.*?) \.\.\.\s*Training completed\.\s*Train R2: [-\d.]+\s*Test R2: ([-\d.]+)\s*Test MAE: ([\d.]+) .*?\s*Test RMSE: ([\d.]+)',t)]
cvsets={}
for key,mark in [('classification','--- CLASSIFICATION CV RESULTS ---'),('ordinary','--- ORDINARY CROP CV RESULTS ---'),('saffron','--- SAFFRON CV RESULTS ---')]:
 block=cv.split(mark)[1].split('\n\n')[0]; rows=[]
 for line in block.splitlines():
  match=re.match(r'^\s*(.*?)\s+(3|5|10)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',line)
  if not match:continue
  n,fold,a,b,c,d=match.groups();n=re.sub(r'^(Ordinary Crops|Saffron)\s+(t/ha|kg/ha)\s+','',n).strip(); rows.append(dict(name=n,fold=int(fold),score=float(c)*100 if key=='classification' else float(a)))
 cvsets[key]=rows
opts={}
for k,mark in [('classification','CLASSIFICATION'),('ordinary','ORDINARY REGRESSION'),('saffron','SAFFRON')]:
 block=opt.split('--- '+mark+' OPTIMIZATION RESULTS ---')[1].split('\n\n')[0]; rows=[]
 for line in block.splitlines()[2:]:
  m=re.match(r'^\s*(.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',line)
  if m:
   name,secs,v,w,x,y,z=m.groups();rows.append({'name':name,'score':float(w)*100 if k=='classification' else float(v),'secondary':float(v)*100 if k=='classification' else float(x)})
 opts[k]=rows
print('counts',len(classification),[(k,len(v)) for k,v in regsets.items()],[(k,len(v)) for k,v in cvsets.items()],[(k,len(v)) for k,v in opts.items()]); assert len(classification)==13 and all(len(v)==11 for v in regsets.values()) and all(len(v)==4 for v in opts.values()) and len(cvsets['classification'])==39 and len(cvsets['ordinary'])==33 and len(cvsets['saffron'])==33
(out/'researchData.json').write_text(json.dumps({'summary':summary,'rules':rules,'metrics':metrics,'classification':classification,**regsets,'cv':cvsets,'optimization':opts,'matrix':[[828,16,5],[2,241,4],[0,0,646]]},separators=(',',':')))

