# This code is to construct interpretable machine learning models for EWAS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import pickle
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score

## process data
data_raw = pd.read_csv("data_covexper.csv")  # 565*23
column_discrete = ['Population Density', 'NPL_Prox', 'Traffic_Prox']  # quantile bin
for column in column_discrete:
    bin_cond = [-np.inf] + data_raw[column].quantile([0.2, 0.4, 0.6, 0.8]).tolist() + [np.inf]
    bin_lab = list(range(5))
    data_raw[column] = pd.cut(data_raw[column], bins=bin_cond, labels=bin_lab)
data_select = data_raw.sort_values(by = ['OBJECT_ID'])  # sort by object id

xname = list(data_raw.columns[:10])  # covariates of base model
xname_scale = data_select.columns.difference(["PM25 Average Conc.", "NO2", "Ozone Seasonal DM8HA", "Town_Death", "Population", "Centroid_X",
                                              "Centroid_Y", "County_Name", "Region_Name", "OBJECT_ID"])
data_select[xname_scale] = data_select[xname_scale].apply(scale)
data_select['Death_Rate'] = (data_select['Town_Death'] / data_select['Population'])
data_model = data_select[data_select['Town_Death'].notna() & (data_select['Population'] > 10) & (data_select['Death_Rate'] < 0.006)]  # 355*24
data_predict = data_select[~data_select['OBJECT_ID'].isin(data_model['OBJECT_ID'])]  # 210*69
data_model = data_model[['Death_Rate'] + xname]  # combine covariates and response, training
data_predict = data_predict[['Death_Rate'] + xname]  # prediction

## extreme gradient boosting

data_concat = pd.concat([data_model, data_predict], axis=0)
model = XGBRegressor(booster='gbtree', random_state=0, learning_rate=0.085, n_estimators=100, max_depth=6,
                     min_child_weight=6, gamma=0, subsample=1, colsample_bytree=0.78,
                     reg_alpha=5e-6, reg_lambda=1)
model.fit(data_model[xname], data_model['Death_Rate'])
pred = model.predict(data_concat[xname])
data_concat['XGBOOST'] = pred
data_predict = pd.merge(data_select, data_concat.XGBOOST, how='left', left_index=True, right_index=True)


# interpretation
# build up explainer
explainer = shap.Explainer(model.predict, data_predict[xname])  # shapley
shap_values = explainer(data_predict[xname])
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data_predict[xname])

# save shap plot
shap.plots.bar(shap_values[141], show=False)
plt.tight_layout()
plt.savefig('bar_plot_instance.png')  
plt.close()

shap.summary_plot(shap_values, data_concat[xname], show=False)
plt.tight_layout()
plt.savefig('summary_plot.png')  
plt.close()

shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig('bar_plot.png')  
plt.close()

# scale back to original data
pd_list = np.array([0,1,2,3,4])
mean = pd_list.mean()
std = pd_list.std()

data_map = data_predict[xname].copy()
data_map["Population Density"] = data_predict["Population Density"]*std+mean
relationship_decoding = {
    0: '0-20%',
    1: '20-40%',
    2: '40-60%',
    3: '60-80%',
    4: '80-100%',
}
data_map["Population Density"] = data_map["Population Density"].map(relationship_decoding)

mean=data_raw["Median Gross Rent"].mean()
std=data_raw["Median Gross Rent"].std()
data_map["Median Gross Rent"] = data_map["Median Gross Rent"]*std + mean 

shap.dependence_plot(
    ("Population Density", "Median Gross Rent"),
    shap_interaction_values, data_map[xname],
    display_features=data_map[xname], show=False)
plt.tight_layout()
plt.savefig('interaction_plot1.png')  
plt.close()

data_map = data_predict[xname].copy()
mean=data_raw["% Minority"].mean()
std=data_raw["% Minority"].std()
data_map["% Minority"] = data_map["% Minority"]*std + mean 

mean=data_raw["% Population(Age>64)"].mean()
std=data_raw["% Population(Age>64)"].std()
data_map["% Population(Age>64)"] = data_map["% Population(Age>64)"]*std + mean 

shap.dependence_plot(
    ("% Minority", "% Population(Age>64)"),
    shap_interaction_values, data_map[xname],
    display_features=data_map[xname], show=False)
plt.tight_layout()
plt.savefig('interaction_plot2.png')  
plt.close()


