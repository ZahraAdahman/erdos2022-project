# erdos2022-project


## Project Description

COVID-19 adverse health outcomes such as mortality rate are associated with multiple Demographic, Environmental and Socioeconomic (DES) factors. Precise estimation of the association patterns can help improve the understanding of social and environmental justice issues related to the impact of the pandemic. In this project, we extracted a subset from the COVID-19 socioexposomic data<sup>1</sup> and developed Interpretable Machine Learning<sup>2</sup> (IML) methods to identify important nonlinear health effects and interactions at local (municipality) scale across New Jersey. Our results show that IML can be an effective supplement to traditional statistical and geospatial models for uncovering underlying complex patterns even for small sample sets.


## Team

We have 4 Ph.Ds from diverse backgrounds as part of the Erdos Institute DataScience Bootcamp - 2022.  

* Xiang Ren (Process Systems Engineering)

* Xiaoran Hao (Math)

* Jun Li (Math)

* Zahra Adahman (Neuroscience)


## Data Challenge (Environmental Health Data)

* Small/moderate sample size (~500)<br/>

* Strong inter-correlation<br/>

* Spatial heterogeneity<br/>

* Nonlinear relationships<br/>

* Potential interactions<br/>
<br/>

![Heatmap](heatmap.png)


## Document Description

Data folder includes raw data and intermediate data used for visualiation; Code folder includes two python scripts and two R scripts:

* machine-learning.py: hyperparameter tuning, validation, prediction and interpretaion of two Machine Learning models, i.e., random forest and extreme gradient boosting<br/>

* shap_plot.py: interaction analysis and visualization for the xgboost model<br/>

* statistical-modeling.R: construction, validation, prediction and interpretaion of two statistical models, i.e., poisson regression and negative-binomial bym spatial model<br/>

* visualization.R: correlation heatmap, effects plots from four modeling approaches, etc.<br/>


## Several Results
TBD


## Final Remark: 
Interpretable Machine Learning (even for small datasets) played a complementary role to advanced geostatistical models: all four modeling approaches can capture similar associations when an underlying exponential relation holds, but Machine Learning can further “learn” non-exponential patterns in the data.


## Reference:
1. Georgopoulos, PG., Mi, Z., Ren, X*. Socioexposomics of COVID-19: The Case of New Jersey, ISES 2021</br>
2. Molnar, C., 2019. Interpretable Machine Learning: A Guide for Making Black Box Models Explainable
