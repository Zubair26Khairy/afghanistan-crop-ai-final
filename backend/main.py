from pathlib import Path
from functools import lru_cache
import json, math, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
BASE=Path(__file__).resolve().parent
RULES=json.loads((BASE/'rules.json').read_text())
METRICS=json.loads((BASE/'metrics.json').read_text())
app=FastAPI(title='Afghanistan Crop AI',version='3.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['*'])
class Input(BaseModel):
    Year:int=Field(ge=2010,le=2040)
    Month:int=Field(ge=1,le=12)
    Province:str
    Crop:str
    Temp_C:float=Field(ge=-40,le=55)
    Rain_mm:float=Field(ge=0,le=3000)
    Fertilizer_kg_ha:float=Field(ge=0,le=1000)
    Irrigation_Type:str
    Land_Hectares:float=Field(gt=0,le=100000)
@lru_cache(maxsize=3)
def load(name): return joblib.load(BASE/'models'/name)
def predict(row):
    for key,allowed in [('Province','valid_provinces'),('Crop','valid_crops'),('Irrigation_Type','valid_irrigation')]:
        if row[key] not in RULES[allowed]: raise HTTPException(422,'Invalid '+key)
    crop=row['Crop']; month=row['Month']; temp=row['Temp_C']; saffron=crop=='Saffron'
    frame=pd.DataFrame([{**row,'Month_sin':math.sin(2*math.pi*(month-1)/12),'Month_cos':math.cos(2*math.pi*(month-1)/12)}])
    reg=load('saffron_yield_model.joblib' if saffron else 'crop_yield_model.joblib')
    value=max(0.,float(reg.predict(frame)[0])); category='Not Evaluated'; raw=category; rule=category
    if crop not in ['Pine Nuts','Saffron']:
        r=RULES['crop_requirements'][crop]; months=r['planting_months']; allowed=crop in RULES['province_crops'][row['Province']] and r['tolerance_temp'][0]<=temp<=r['tolerance_temp'][1]
        distance=min(min(abs(month-m),12-abs(month-m)) for m in months)
        code=2 if allowed and r['optimal_temp'][0]<=temp<=r['optimal_temp'][1] and month in months else 1 if allowed and distance<=1 else 0
        ml=int(load('crop_suitability_model.joblib').predict(frame)[0]); labels=['Low','Medium','High']; raw=labels[ml]; rule=labels[code]; category=labels[min(ml,code)]
    return dict(category=category,ml_category=raw,rule_category=rule,yield_value=round(value,3),total_production=round(value*row['Land_Hectares'],3),unit='kg' if saffron else 't',classification_model=METRICS['classification'],regression_model=METRICS['saffron' if saffron else 'ordinary'],disclaimer='Synthetic-data prediction; not a field-validated forecast. Irrigation comparisons are not causal.')
@app.get('/api/health')
def health(): return {'status':'ok','model_count':3}
@app.get('/api/metadata')
def metadata(): return {**RULES,'provinces':RULES['valid_provinces'],'crops':RULES['valid_crops'],'irrigation_types':RULES['valid_irrigation'],'defaults':dict(Year=2027,Month=3,Province='Kandahar',Crop='Pomegranates',Temp_C=28,Rain_mm=300,Fertilizer_kg_ha=120,Irrigation_Type='Drip',Land_Hectares=5),'metrics':METRICS}
@app.post('/api/predict')
def prediction(data:Input): return predict(data.model_dump())
@app.post('/api/recommend')
def recommend(data:Input):
    rows=[{'crop':c,**predict({**data.model_dump(),'Crop':c})} for c in RULES['valid_crops']]; rows.sort(key=lambda x:({'High':0,'Medium':1,'Low':2,'Not Evaluated':3}[x['category']],x['crop']));return {'comparisons':rows}
@app.post('/api/irrigation')
def irrigation(data:Input):
    rows=[{'irrigation_type':i,**predict({**data.model_dump(),'Irrigation_Type':i})} for i in RULES['valid_irrigation']]; base=next(x['yield_value'] for x in rows if x['irrigation_type']=='Rainfed');return {'comparisons':[dict(x,difference=round(x['yield_value']-base,3)) for x in rows]}
@app.get('/api/models')
def models(): return METRICS
