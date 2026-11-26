# Afghanistan Crop AI — selected deployment models only
import calendar, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor, VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

# %% 1. Load data
# Load the CSV chosen in Spyder and inspect dimensions and missing values.
print("\nSTEP 1:", "Load data")
import os
file_path = os.environ.get('CROP_DATA_FILE')
if not file_path:
    from tkinter import Tk, filedialog
    root = Tk(); root.withdraw(); root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv')]); root.destroy()
if not file_path: raise SystemExit('No CSV selected')
df = pd.read_csv(file_path)
print('Shape:', df.shape, '\nMissing values:', df.isna().sum().to_string())
# %% 2. Data quality
# Build boolean flags for invalid categories, numeric ranges and production consistency.
print("\nSTEP 2:", "Data quality")

valid_provinces = ['Badakhshan', 'Badghis', 'Baghlan', 'Balkh', 'Bamyan', 'Daykundi', 'Farah', 'Faryab', 'Ghazni', 'Ghor', 'Helmand', 'Herat', 'Jawzjan', 'Kabul', 'Kandahar', 'Kapisa', 'Khost', 'Kunar', 'Kunduz', 'Laghman', 'Logar', 'Nangarhar', 'Nimroz', 'Nuristan', 'Paktia', 'Paktika', 'Panjshir', 'Parwan', 'Samangan', 'Sar-e-Pul', 'Takhar', 'Uruzgan', 'Wardak', 'Zabul']
valid_crops = ['Almonds', 'Apples', 'Apricots', 'Oranges', 'Figs', 'Grapes', 'Melons', 'Watermelons', 'Pine Nuts', 'Pomegranates', 'Pistachios', 'Saffron']
valid_irrigation = ['Drip', 'Canal', 'Rainfed']
valid_categories = ['High', 'Medium', 'Low']

checks = pd.DataFrame({
    'Missing': df.isna().any(axis=1),
    'Year': ~df.Year.between(2017, 2026),
    'Month': ~df.Month.between(1, 12),
    'Province': ~df.Province.isin(valid_provinces),
    'Crop': ~df.Crop.isin(valid_crops),
    'Temperature': ~df.Temp_C.between(-40, 55),
    'Rain': df.Rain_mm < 0, 'Fertilizer': df.Fertilizer_kg_ha < 0,
    'Irrigation': ~df.Irrigation_Type.isin(valid_irrigation),
    'Land': df.Land_Hectares <= 0, 'Yield': df.Yield_per_Hectare < 0,
    'Production': df.Total_Production < 0,
    'Category': ~df.Suitability_Category.isin(valid_categories),
    'Production formula': ~np.isclose(df.Total_Production,
        df.Land_Hectares * df.Yield_per_Hectare, rtol=.001, atol=.01)
}, index=df.index)
print(checks.sum())
print("Dirty rows:", checks.any(axis=1).sum())
# %% 3. Clean and label
# Correct recoverable spelling errors, quarantine remaining invalid rows and rebuild rule-based labels.
print("\nSTEP 3:", "Clean and label")
df_clean = df.copy()
corrected = df_clean.Crop.eq('Pomegrantes')
df_clean.loc[corrected, 'Crop'] = 'Pomegranates'
checks.loc[corrected, 'Crop'] = False
invalid = checks.any(axis=1)
print('Quarantined:', invalid.sum())
df_clean = df_clean.loc[~invalid].drop_duplicates().reset_index(drop=True)
df_clean['Month_Name'] = df_clean.Month.map(dict(enumerate(calendar.month_name)))
df_clean['Total_Production'] = (df_clean.Land_Hectares * df_clean.Yield_per_Hectare).round(4)

crop_requirements = {
    'Almonds': {'optimal_temp': (12, 35), 'tolerance_temp': (10, 40), 'planting_months': [2, 3, 11, 12]},
    'Apples': {'optimal_temp': (14, 27), 'tolerance_temp': (8, 33), 'planting_months': [2, 3, 11, 12]},
    'Apricots': {'optimal_temp': (14, 35), 'tolerance_temp': (7, 40), 'planting_months': [2, 3, 11, 12]},
    'Oranges': {'optimal_temp': (20, 30), 'tolerance_temp': (13, 38), 'planting_months': [3, 4]},
    'Figs': {'optimal_temp': (16, 26), 'tolerance_temp': (4, 38), 'planting_months': [2, 3, 4]},
    'Grapes': {'optimal_temp': (18, 30), 'tolerance_temp': (10, 38), 'planting_months': [2, 3]},
    'Melons': {'optimal_temp': (18, 30), 'tolerance_temp': (9, 35), 'planting_months': [4, 5]},
    'Watermelons': {'optimal_temp': (20, 35), 'tolerance_temp': (15, 40), 'planting_months': [4, 5]},
    'Pine Nuts': {'optimal_temp': (15, 25), 'tolerance_temp': (4, 38), 'planting_months': None},
    'Pomegranates': {'optimal_temp': (23, 32), 'tolerance_temp': (8, 40), 'planting_months': [2, 3]},
    'Pistachios': {'optimal_temp': (20, 35), 'tolerance_temp': (10, 40), 'planting_months': [2, 3]},
    'Saffron': {'optimal_temp': None, 'tolerance_temp': None, 'planting_months': [5, 6, 7, 8, 9, 10]},
}

province_crops = {
    'Badakhshan': ['Apples', 'Apricots', 'Almonds', 'Grapes'],
    'Badghis': ['Pistachios', 'Almonds', 'Pomegranates'],
    'Baghlan': ['Apples', 'Apricots', 'Almonds', 'Melons', 'Pistachios'],
    'Balkh': ['Grapes', 'Almonds', 'Pomegranates', 'Melons'],
    'Bamyan': ['Apples', 'Apricots', 'Almonds'],
    'Daykundi': ['Almonds', 'Apples', 'Apricots'],
    'Farah': ['Pomegranates', 'Grapes', 'Melons', 'Watermelons'],
    'Faryab': ['Grapes', 'Almonds', 'Melons'],
    'Ghazni': ['Apples', 'Grapes', 'Almonds', 'Apricots'],
    'Ghor': ['Apples', 'Apricots', 'Almonds'],
    'Helmand': ['Pomegranates', 'Grapes', 'Melons', 'Watermelons'],
    'Herat': ['Grapes', 'Saffron', 'Almonds', 'Pomegranates', 'Figs'],
    'Jawzjan': ['Grapes', 'Almonds', 'Melons'],
    'Kabul': ['Grapes', 'Apples', 'Apricots'],
    'Kandahar': ['Pomegranates', 'Grapes', 'Almonds', 'Figs', 'Melons'],
    'Kapisa': ['Pomegranates', 'Grapes', 'Apricots'],
    'Khost': ['Pine Nuts', 'Pomegranates', 'Grapes'],
    'Kunar': ['Oranges', 'Pomegranates', 'Pine Nuts'],
    'Kunduz': ['Melons', 'Apples', 'Grapes'],
    'Laghman': ['Oranges', 'Pomegranates', 'Grapes'],
    'Logar': ['Apples', 'Grapes', 'Almonds'],
    'Nangarhar': ['Oranges', 'Pomegranates', 'Melons'],
    'Nimroz': ['Melons', 'Watermelons', 'Grapes'],
    'Nuristan': ['Pine Nuts', 'Apricots'],
    'Paktia': ['Pine Nuts', 'Apples', 'Grapes'],
    'Paktika': ['Pine Nuts', 'Apples', 'Grapes'],
    'Panjshir': ['Apples', 'Apricots', 'Almonds'],
    'Parwan': ['Grapes', 'Apples', 'Almonds'],
    'Samangan': ['Pistachios', 'Almonds', 'Grapes'],
    'Sar-e-Pul': ['Grapes', 'Almonds', 'Pomegranates'],
    'Takhar': ['Pistachios', 'Apples', 'Almonds'],
    'Uruzgan': ['Almonds', 'Pomegranates', 'Grapes'],
    'Wardak': ['Apples', 'Apricots', 'Almonds'],
    'Zabul': ['Almonds', 'Grapes', 'Pomegranates'],
}

# Apply crop-specific rules directly, without a custom function.
for crop, rules in crop_requirements.items():
    if crop in ['Saffron', 'Pine Nuts']:
        continue
    rows = df_clean.Crop.eq(crop)
    t = df_clean.Temp_C
    m = df_clean.Month
    months = rules['planting_months']
    distance = np.minimum.reduce([np.minimum(abs(m-k), 12-abs(m-k)) for k in months])
    region = df_clean.Province.isin([p for p, crops in province_crops.items() if crop in crops])
    allowed = region & t.between(*rules['tolerance_temp'])
    high = allowed & t.between(*rules['optimal_temp']) & m.isin(months)
    medium = allowed & (distance <= 1)
    df_clean.loc[rows, 'Suitability_Category'] = np.select([high[rows], medium[rows]], ['High','Medium'], default='Low')
df_clean.to_csv(Path(file_path).parent / 'cleaned.csv', index=False)
print('Clean rows:', len(df_clean))
print(df_clean.Suitability_Category.value_counts())
# %% 4. EDA
# Compare crop distributions; saffron yield uses kg/ha rather than t/ha.
print("\nSTEP 4:", "Exploratory analysis")
print(df_clean.groupby('Crop').Yield_per_Hectare.agg(['count','min','mean','max']))
print('Zero yields:', (df_clean.Yield_per_Hectare == 0).sum())
pd.crosstab(df_clean.Crop, df_clean.Suitability_Category).plot.bar(stacked=True)
plt.tight_layout(); plt.show()
# %% 5. Features
# Encode seasonal month cyclically and exclude production to prevent target leakage.
print("\nSTEP 5:", "Feature engineering")
ml = df_clean.copy()
angle = 2*np.pi*(ml.Month-1)/12
ml['Month_sin'], ml['Month_cos'] = np.sin(angle), np.cos(angle)
classification_data = ml[~ml.Crop.isin(['Saffron','Pine Nuts'])].reset_index(drop=True)
fruit_data = ml[ml.Crop != 'Saffron'].reset_index(drop=True)
saffron_data = ml[ml.Crop == 'Saffron'].reset_index(drop=True)
classification_numeric = ['Month_sin','Month_cos','Temp_C']
classification_categorical = ['Province','Crop']
regression_numeric = ['Year','Month_sin','Month_cos','Temp_C','Rain_mm','Fertilizer_kg_ha','Land_Hectares']
regression_categorical = ['Province','Crop','Irrigation_Type']
saffron_numeric = regression_numeric.copy()
saffron_categorical = ['Province','Irrigation_Type']
classification_features = classification_numeric + classification_categorical
regression_features = regression_numeric + regression_categorical
saffron_features = saffron_numeric + saffron_categorical
X_class = classification_data[classification_features]
y_class = classification_data.Suitability_Category.map({'Low':0,'Medium':1,'High':2})
X_reg, y_reg = fruit_data[regression_features], fruit_data.Yield_per_Hectare
X_saffron, y_saffron = saffron_data[saffron_features], saffron_data.Yield_per_Hectare
print('Input shapes:', X_class.shape, X_reg.shape, X_saffron.shape)
# %% 6. Shared preprocessing
# Fit one-hot encoding and optional scaling inside each model pipeline.

tasks = {'Classification': (X_class,y_class,classification_data.Year,classification_numeric,classification_categorical),
         'Ordinary': (X_reg,y_reg,fruit_data.Year,regression_numeric,regression_categorical),
         'Saffron': (X_saffron,y_saffron,saffron_data.Year,saffron_numeric,saffron_categorical)}
preprocessors = {}
for task, (_, _, _, numeric, categorical) in tasks.items():
    for scale in [True, False]:
        preprocessors[task, scale] = ColumnTransformer([
            ('numeric', StandardScaler() if scale else 'passthrough', numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)])
# %% 7. Chronological split
# Use 2017–2024 for learning and 2025–2026 for chronological testing.
# Reserve later years for evaluation; do not fit preprocessing on test rows.
for task, (X, y, years, _, _) in tasks.items():
    train = years <= 2024
    print(task, "Train:", train.sum(), "Test:", (~train).sum())

# %% 8. Classification model (previously optimized)
# Average probabilities from HGB, RBF SVM and Decision Tree using soft voting.
hgb = Pipeline([('preprocessor', clone(preprocessors['Classification', False])),
    ('model', HistGradientBoostingClassifier(min_samples_leaf=20, max_leaf_nodes=63,
        max_iter=350, max_depth=None, learning_rate=.15, l2_regularization=0,
        early_stopping=False, random_state=42))])
svm = Pipeline([('preprocessor', clone(preprocessors['Classification', True])),
    ('model', SVC(kernel='rbf', C=50, gamma='scale', class_weight='balanced',
        probability=True, random_state=42, cache_size=500))])
tree = Pipeline([('preprocessor', clone(preprocessors['Classification', False])),
    ('model', DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1,
        max_depth=30, criterion='entropy', ccp_alpha=.0001, random_state=42))])

# Model 1: crop suitability
final_classifier = VotingClassifier([('hgb', hgb), ('svm', svm), ('tree', tree)],
    voting='soft', n_jobs=1)

# %% 9. Regression models
# Use boosted trees for ordinary crops and a shallow forest for the small saffron sample.
# Model 2: ordinary crop yield
final_regressor = Pipeline([('preprocessor', clone(preprocessors['Ordinary', False])),
    ('model', HistGradientBoostingRegressor(min_samples_leaf=10, max_leaf_nodes=63,
        max_iter=300, max_depth=8, loss='absolute_error', learning_rate=.03,
        l2_regularization=1, early_stopping=False, random_state=42))])

# Model 3: saffron yield
final_saffron = Pipeline([('preprocessor', clone(preprocessors['Saffron', False])),
    ('model', RandomForestRegressor(n_estimators=200, min_samples_split=5,
        min_samples_leaf=3, max_features=1.0, max_depth=2, random_state=42, n_jobs=1))])

# %% 10. Chronological evaluation and final training
# Train on earlier years, evaluate on later years, then refit on all eligible rows.
for task, model in [('Classification',final_classifier),('Ordinary',final_regressor),('Saffron',final_saffron)]:
    X,y,years,_,_ = tasks[task]
    train = years <= 2024
    # Learn only from the training years.
    model.fit(X.loc[train],y.loc[train])
    # Predict unseen later-year records.
    pred = model.predict(X.loc[~train])
    if task == 'Classification':
        print(task,'Accuracy:',accuracy_score(y.loc[~train],pred),'Macro F1:',f1_score(y.loc[~train],pred,average='macro'))
        print(confusion_matrix(y.loc[~train],pred))
    else:
        pred = np.maximum(pred,0)
        print(task,'MAE:',mean_absolute_error(y.loc[~train],pred),'RMSE:',np.sqrt(mean_squared_error(y.loc[~train],pred)),'R2:',r2_score(y.loc[~train],pred))
    # Refit for deployment after evaluation.
    model.fit(X,y)
# %% 11. Export and practical predictions
# Export the three selected pipelines and demonstrate irrigation and suitability predictions.
# Save the fitted pipelines, including preprocessing, for later inference.
output = Path(file_path).parent / 'selected_models'
output.mkdir(exist_ok=True)
joblib.dump(final_classifier,output/'crop_suitability_model.joblib')
joblib.dump(final_regressor,output/'crop_yield_model.joblib')
joblib.dump(final_saffron,output/'saffron_yield_model.joblib')
# Example yield and irrigation comparison
scenario = pd.DataFrame([dict(Year=2027,Month_sin=np.sin(2*np.pi*2/12),Month_cos=np.cos(2*np.pi*2/12),Province='Kandahar',Crop='Pomegranates',Temp_C=28,Rain_mm=300,Fertilizer_kg_ha=120,Land_Hectares=5,Irrigation_Type=method) for method in ['Drip','Canal','Rainfed']])
scenario['Predicted_Yield'] = np.maximum(final_regressor.predict(scenario[regression_features]),0)
scenario['Total_Production'] = scenario.Predicted_Yield * scenario.Land_Hectares
print(scenario[['Irrigation_Type','Predicted_Yield','Total_Production']])
# Raw ML recommendation, before agricultural guardrail
recommendations = pd.concat([scenario.iloc[[0]].assign(Crop=crop) for crop in sorted(X_class.Crop.unique())],ignore_index=True)
recommendations['ML_Category'] = pd.Series(final_classifier.predict(recommendations[classification_features])).map({0:'Low',1:'Medium',2:'High'})
print(recommendations[['Crop','ML_Category']])

# Agricultural guardrail for the example (March, Kandahar, 28 C)
rule_codes = []
for crop in recommendations.Crop:
    rules = crop_requirements[crop]
    months = rules['planting_months']
    distance = min(min(abs(3-m),12-abs(3-m)) for m in months)
    allowed = crop in province_crops['Kandahar'] and rules['tolerance_temp'][0] <= 28 <= rules['tolerance_temp'][1]
    high = allowed and rules['optimal_temp'][0] <= 28 <= rules['optimal_temp'][1] and 3 in months
    rule_codes.append(2 if high else 1 if allowed and distance <= 1 else 0)
recommendations['Rule_Category'] = pd.Series(rule_codes).map({0:'Low',1:'Medium',2:'High'})
recommendations['Final_Category'] = np.minimum(recommendations.ML_Category.map({'Low':0,'Medium':1,'High':2}),rule_codes).map({0:'Low',1:'Medium',2:'High'})
print(recommendations[['Crop','ML_Category','Rule_Category','Final_Category']])
