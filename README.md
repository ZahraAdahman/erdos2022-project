# erdos2022-project

## Goal of the project
We handle COVID-19 Health Risk data in New Jersey. 
The goal is to use interpretable Machine Learning methods to capture important nonlinear health effects (particularly the complex interactions), and evaluate the consistency with those captured by linear models.

## Team
We ahve 4 Ph.Ds from diverse backgrounds as part of the Erdos Institute DataScience Bootcamp - 2022.  

 
Zahra Adahman 

Xiaoran Hao  

Jun Li  

Xiang Ren 

## Project summary

### The challenge 

1. Determine a set of features of interest for analyses.  We have multiple possible non-linearly interacted features that could affect Covid death rate. 
2. Figure out the most suitable interpretation tool.
3. Train interpretable Machine Learning models and compare with commonly used linear models in public health.

### Our approach
1.variable selection.  There are linear tools, PCA for example, to select features that are not correlated.  However, our case has non-linear interactions, which is indeed our focus.    Here we use the domain knowledge from health industries.  10 features are selected, and they all affect Covid death rate  (per 100,000 people).

2.We decided Shapley value, better for interpretation, among other quantities.

3.We trained ramdom forest, xgboost models and compare them with Possion and Neg-Binomial BYM models


### Conclusion: 


#### For predictions:

ML obtained slightly higher validation accuracy than statistical models.


#### When it comes to interactions:

Compare commonly used geostatistical models, ML models capture similar 
associations  when an underlying exponential 
relation holds, but ML can further “learn” non-
exponential patterns and interactions in the data, 
an issue that is important for precision health and 
knowledge discovery.
