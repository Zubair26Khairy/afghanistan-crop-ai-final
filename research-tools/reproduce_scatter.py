import pathlib,json,joblib,numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error,r2_score
p=pathlib.Path(__file__).resolve().parents[1]; dpath=p/'frontend/src/researchData.json'; d=json.loads(dpath.read_text()); g=pd.read_csv(p/'frontend/public/cleaned.csv'); g['Month_sin']=np.sin(2*np.pi*(g.Month-1)/12); g['Month_cos']=np.cos(2*np.pi*(g.Month-1)/12); d['scatter']={}
for key,file,df in [('ordinary','crop_yield_model.joblib',g[g.Crop!='Saffron']),('saffron','saffron_yield_model.joblib',g[g.Crop=='Saffron'])]:
 model=clone(joblib.load(p/'backend/models'/file)); cols=list(model.steps[0][1].transformers[0][2])+list(model.steps[0][1].transformers[1][2]); train=df.Year<=2024; model.fit(df.loc[train,cols],df.loc[train,'Yield_per_Hectare']); test=df.loc[~train]; pred=np.maximum(model.predict(test[cols]),0); actual=test.Yield_per_Hectare.to_numpy(); mae=mean_absolute_error(actual,pred); print(key,len(test),mae,r2_score(actual,pred)); assert abs(mae-d['metrics'][key]['mae'])<1e-6; d['scatter'][key]=[[round(float(a),3),round(float(b),3),c] for a,b,c in zip(actual,pred,test.Crop)]
# Yield histograms with distinct units.
d['summary']['histogram']={}
for name,df,bins in [('ordinary',g[g.Crop!='Saffron'],[0,5,10,15,20,25,30,35,40,45]),('saffron',g[g.Crop=='Saffron'],[0,2,4,6,8,10])]:
 v,edges=np.histogram(df.Yield_per_Hectare,bins=bins);d['summary']['histogram'][name]=[{'name':f'{a}–{b}','value':int(n)} for a,b,n in zip(edges[:-1],edges[1:],v)]
dpath.write_text(json.dumps(d,separators=(',',':')))
