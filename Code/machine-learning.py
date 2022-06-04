# This code is to construct interpretable machine learning models for association analysis

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
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

##########################
###### process data ######
##########################

data_raw = pd.read_csv("Data/data_covexper.csv")  # 565*23
column_discrete = ['PD', 'NPL_Prox', 'Traffic_Prox']  # quantile bin
for column in column_discrete:
    bin_cond = [-np.inf] + data_raw[column].quantile([0.2, 0.4, 0.6, 0.8]).tolist() + [np.inf]
    bin_lab = list(range(5))
    data_raw[column] = pd.cut(data_raw[column], bins=bin_cond, labels=bin_lab)
data_select = data_raw.sort_values(by = ['OBJECT_ID'])  # sort by object id
# data_select.to_csv("Data/Output/data_ref_unscaled.csv", index=False)

xname = list(data_raw.columns[:10])  # covariates of base model
xname_scale = data_select.columns.difference(["PM25", "NO2", "Ozone", "Town_Death", "Population", "Centroid_X",
                                              "Centroid_Y", "County_Name", "Region_Name", "OBJECT_ID"])
data_select[xname_scale] = data_select[xname_scale].apply(scale)
data_select['Death_Rate'] = data_select['Town_Death'] / data_select['Population']
# data_select.to_csv("Data/Output/data_select.csv", index=False)
data_model = data_select[data_select['Town_Death'].notna() & (data_select['Population'] > 10) & (data_select['Death_Rate'] < 0.006)]  # 352*24
data_predict = data_select[~data_select['OBJECT_ID'].isin(data_model['OBJECT_ID'])]  # 213*24

###########################
###### random forest ######
###########################

## hyperparameter tuning
random_forest = RandomForestRegressor(random_state=0)
param_grid = [{"n_estimators": np.arange(500, 701, 50), "min_samples_split": np.arange(17, 20, 1), "max_features": np.arange(2, 6, 1)}]
cv = KFold(n_splits=5, shuffle=True, random_state=0)
search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=cv,
                      scoring=["explained_variance", "neg_root_mean_squared_error"],
                      refit="explained_variance", n_jobs=-1, return_train_score=True)
search.fit(data_model[xname], data_model['Death_Rate'])
results_df = pd.DataFrame(search.cv_results_).sort_values(by=['rank_test_explained_variance'])
a = results_df[['params', 'rank_test_explained_variance', 'mean_test_explained_variance', 'mean_train_explained_variance',
                'rank_test_neg_root_mean_squared_error', 'mean_test_neg_root_mean_squared_error', 'mean_train_neg_root_mean_squared_error']]

## leave-one-out cross validation
X = data_model[xname]
y = data_model['Death_Rate']
cv = LeaveOneOut()
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(X):
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    model = RandomForestRegressor(n_estimators=600, min_samples_split=17, max_features=2, random_state=0)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    y_true.append(y_test.iloc[0])
    y_pred.append(yhat[0])
    print(test_ix)

loocv_rf = pd.DataFrame()  # save rf loocv results
loocv_rf['OBJECT_ID'] = data_model['OBJECT_ID']
loocv_rf['Death_Rate'] = y_true
loocv_rf['RF_Rate'] = y_pred
# loocv_rf.to_csv('Data/Output/loocv_rf.csv', index = False)

## prediction
data_concat = pd.concat([data_model, data_predict], axis=0)
model = RandomForestRegressor(n_estimators=600, min_samples_split=17, max_features=2, random_state=0)
model.fit(data_model[xname], data_model['Death_Rate'])
pred = model.predict(data_concat[xname])
data_concat['RF_Rate'] = pred
data_concat[['OBJECT_ID', 'Death_Rate', 'RF_Rate']].to_csv("Data/Output/predict_rf.csv", index = False)

# fig, axs = plt.subplots(1, 2)  # observed vs predicted
# axs[0].scatter(data_concat.Death_Rate, data_concat.RF_Rate, color="blue", alpha=0.2)
# axs[0].axline([0, 0], [0.02, 0.02], color="black", linewidth=0.6)
# axs[0].set(xlabel="Observed Death Rates", ylabel="Predicted Death Rates")
# axs[1].scatter(data_concat.Death_Rate*data_concat.Population, data_concat.RF_Rate*data_concat.Population, color="black", alpha=0.2)
# axs[1].axline([0, 0], [720, 720], color="black", linewidth=0.6)
# axs[1].set(xlabel="# Observed Deaths", ylabel="# Predicted Deaths")
# plt.show()

## interpretation
# shap.plots.partial_dependence(ind=xname[1], model=model.predict, data=data_concat[xname], ice=False)  # partial dependence plot
explainer = shap.Explainer(model.predict, data_concat[xname])  # shapley
shap_values = explainer(data_concat[xname])  # not json serializable, .values (565*10), .data (565*10), .base_values (565,)
shap.plots.scatter(shap_values[:, xname[1]])  # shap scatter plot
plt.scatter(shap_values.data[:,1], shap_values.values[:,1]+np.unique(shap_values.base_values), alpha=0.2)
plt.show()

# with open('Data/Output/shap_values_rf.pkl', 'wb') as wf:  # write binary mode
#     pickle.dump(obj=shap_values, file=wf)
# with open('Data/Output/shap_values_rf.pkl', 'rb') as rf:
#     shap_values = pickle.load(file=rf)

base_value = np.unique(shap_values.base_values)  # data for shap effects plot
mean_value = shap_values.values.mean(axis=0)
comp_value = np.array([sum(mean_value) - mean_value[i] for i in range(len(mean_value))]) + base_value
shap_df = pd.DataFrame(shap_values.values + comp_value, columns=[i+str("_ShapEffect") for i in shap_values.feature_names])
shap_df["OBJECT_ID"] = np.array(data_concat["OBJECT_ID"])
shap_df.to_csv("Data/Output/shap-effect_rf_basemodel.csv", index=False)

#######################################
###### extreme gradient boosting ######
#######################################

## hyperparameter tuning
xg_boost = XGBRegressor(booster='gbtree', random_state=0)
param_grid = {'learning_rate': [0.084, 0.0845, 0.085], 'n_estimator': [50, 100],
               'max_depth': [4, 6], 'min_child_weight': [4.2, 5, 6], 'gamma': [0],
               'subsample': [0.9, 1], 'colsample_bytree': [0.7, 0.78, 0.86],
               'reg_alpha': [5e-6], 'reg_lambda': [1]}
cv = KFold(n_splits=5, shuffle=True, random_state=0)  # cross validation object, 5 fold
search = GridSearchCV(estimator=xg_boost, param_grid=param_grid, cv=cv,
                      scoring=["explained_variance", "neg_root_mean_squared_error"],
                      refit='explained_variance', n_jobs=1, return_train_score=True)
search.fit(data_model[xname], data_model['Death_Rate'])
results_df = pd.DataFrame(search.cv_results_).sort_values(by=['rank_test_explained_variance'])
a = results_df[['params', 'rank_test_explained_variance', 'mean_test_explained_variance', 'mean_train_explained_variance',
                'rank_test_neg_root_mean_squared_error', 'mean_test_neg_root_mean_squared_error', 'mean_train_neg_root_mean_squared_error']]

## leave-one-out cross validation
X = data_model[xname]
y = data_model['Death_Rate']
cv = LeaveOneOut()
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(X):
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    model = XGBRegressor(booster='gbtree', random_state=0, learning_rate=0.085, n_estimators=100, max_depth=6,
                         min_child_weight=6, gamma=0, subsample=1, colsample_bytree=0.78,
                         reg_alpha=5e-6, reg_lambda=1)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    y_true.append(y_test.iloc[0])
    y_pred.append(yhat[0])
    print(test_ix)

loocv_xgboost = pd.DataFrame()  # save xgboost loocv results
loocv_xgboost['OBJECT_ID'] = data_model['OBJECT_ID']
loocv_xgboost['Death_Rate'] = y_true
loocv_xgboost['XGBOOST_Rate'] = y_pred
loocv_xgboost.to_csv('Data/Output/loocv_xgboost.csv', index = False)

## prediction
data_concat = pd.concat([data_model, data_predict], axis=0)
model = XGBRegressor(booster='gbtree', random_state=0, learning_rate=0.085, n_estimators=100, max_depth=6,
                     min_child_weight=6, gamma=0, subsample=1, colsample_bytree=0.78,
                     reg_alpha=5e-6, reg_lambda=1)
model.fit(data_model[xname], data_model['Death_Rate'])
pred = model.predict(data_concat[xname])
data_concat['XGBOOST_Rate'] = pred
data_concat[['OBJECT_ID', 'Death_Rate', 'XGBOOST_Rate']].to_csv("Data/Output/predict_xgboost.csv", index = False)

# fig, axs = plt.subplots(1, 2)  # observed vs predicted
# axs[0].scatter(data_concat.Death_Rate, data_concat.XGBOOST_Rate, color="blue", alpha=0.2)
# axs[0].axline([0, 0], [0.02, 0.02], color="black", linewidth=0.6)
# axs[0].set(xlabel="Observed Death Rates", ylabel="Predicted Death Rates")
# axs[1].scatter(data_concat.Death_Rate*data_concat.Population, data_concat.XGBOOST_Rate*data_concat.Population, color="black", alpha=0.2)
# axs[1].axline([0, 0], [720, 720], color="black", linewidth=0.6)
# axs[1].set(xlabel="# Observed Deaths", ylabel="# Predicted Deaths")
# plt.show()

## interpretation
# shap.plots.partial_dependence(ind=xname[1], model=model.predict, data=data_concat[xname], ice=False)
explainer = shap.Explainer(model.predict, data_concat[xname])  # shapley
shap_values = explainer(data_concat[xname])
shap.plots.scatter(shap_values[:, xname[8]])  # shap scatter plot
plt.scatter(shap_values.data[:,8], shap_values.values[:,8]+np.unique(shap_values.base_values), alpha=0.2)
plt.show()

# with open('Data/Output/shap_values_xgboost.pkl', 'wb') as wf:  # write binary mode
#     pickle.dump(obj=shap_values, file=wf)
# with open('Data/Output/shap_values_xgboost.pkl', 'rb') as rf:
#     shap_values = pickle.load(file=rf)

base_value = np.unique(shap_values.base_values)  # data for shap effects plot
mean_value = shap_values.values.mean(axis=0)
comp_value = np.array([sum(mean_value) - mean_value[i] for i in range(len(mean_value))]) + base_value
shap_df = pd.DataFrame(shap_values.values + comp_value, columns=[i+str("_ShapEffect") for i in shap_values.feature_names])
shap_df["OBJECT_ID"] = np.array(data_concat["OBJECT_ID"])
shap_df.to_csv("Data/Output/shap-effect_xgboost_basemodel.csv", index=False)

##############################################################################################
###### calculate shap values for other covariates (not included in the reference model) ######
##############################################################################################

data_concat = pd.concat([data_model, data_predict], axis=0)
xref = ['Over65_Pct', 'Minority_Pct', 'BelowHS_Pct', 'Med_Gross_Rent', 'PD', 'Job_Pct_HRisk', 'PM25', 'Ozone', 'CrowdHouse_Pct', 'Unemployed_Pct']
xtest = ['NO2', 'Resp_Risk', 'NPL_Prox', 'Traffic_Prox', 'Noise_Level', 'SVI_Overall']
data_cor = data_select[xref + xtest].corr()

def cor_remove(xname_add, xname_ref):  # remove highly correlated variables
    cor_select = abs(data_cor.loc[xname_add, xname_ref])
    return [xname_add] + list(cor_select.index[cor_select <= 0.6])

shap_effect = pd.DataFrame({'OBJECT_ID': data_concat['OBJECT_ID']})
for j in len(xtest):
    xname = cor_remove(xname_add=xtest[j], xname_ref=xref)
    model = RandomForestRegressor(n_estimators=600, min_samples_split=17, max_features=2, random_state=0)
    # model = XGBRegressor(booster='gbtree', random_state=0, learning_rate=0.085, n_estimators=100, max_depth=6,
    #                      min_child_weight=6, gamma=0, subsample=1, colsample_bytree=0.78, reg_alpha=5e-6, reg_lambda=1)
    model.fit(data_model[xname], data_model['Death_Rate'])  # train
    explainer = shap.Explainer(model.predict, data_concat[xname])  # shapley
    shap_values = explainer(data_concat[xname])
    base_value = np.unique(shap_values.base_values)  # 1 reference value
    mean_value = shap_values.values.mean(axis=0)
    comp_value = np.array([sum(mean_value) - mean_value[i] for i in range(len(mean_value))]) + base_value
    shap_df = pd.DataFrame(shap_values.values + comp_value, columns=[i + str("_ShapEffect") for i in shap_values.feature_names])
    shap_effect[shap_df.columns[0]] = np.array(shap_df[shap_df.columns[0]])
    print(j)
shap_effect.to_csv("Data/Output/shap-effect_rf_other-covariates.csv", index=False)