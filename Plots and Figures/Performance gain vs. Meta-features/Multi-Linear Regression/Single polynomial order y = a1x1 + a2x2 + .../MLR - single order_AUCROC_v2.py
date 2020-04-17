import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Performance gain vs. Meta-features')
from retrieve_performance_gain import retrieve_best_ML_and_stats_model
import pandas as pd
import pickle
import statsmodels.api as sm    
import matplotlib.pyplot as plt

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle) 
    
#AUC-ROC   
performance_gain_roc = retrieve_best_ML_and_stats_model(nested_results_roc)[2]
performance_gain_roc_list = list(performance_gain_roc.values())

#===============================================================================
#                               ALL VARIABLES
#===============================================================================
#Independent variables
X1 = df_results[['Number of Instances', 'Number of Features', 'Number of Classes',
       'Mean mean', 'Mean std', 'Mean Coefficient of Variation',
       'Mean skewness', 'Mean kurtosis', 'Mean Correlation',
       'Mean Normalised Feature Entropy', 'Normalized Class Entropy',
       'Signal to Noise ratio']]
X1 = sm.add_constant(X1, has_constant='add')
#Target variable: performance gain in AUC-ROC
y = pd.Series(data = performance_gain_roc_list,
                 index = list(X1.index))

model1 = sm.OLS(y, X1).fit()
print(model1.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = -0.625
#       p-value of Coefficient of variation = 0.163
#===============================================================================
#Independent variables
X2 = df_results[['Number of Instances', 'Number of Features', 'Number of Classes',
       'Mean mean', 'Mean std','Mean skewness', 'Mean kurtosis', 'Mean Correlation',
       'Mean Normalised Feature Entropy', 'Normalized Class Entropy', 'Signal to Noise ratio']]
X2 = sm.add_constant(X2, has_constant='add')

model2 = sm.OLS(y, X2).fit()
print(model2.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.177
#       p-value of Number of Instances = 0.866
#===============================================================================
#Independent variables
X3 = df_results[[ 'Number of Features', 'Number of Classes',
       'Mean mean', 'Mean std','Mean skewness', 'Mean kurtosis','Mean Correlation',
       'Mean Normalised Feature Entropy', 'Normalized Class Entropy', 'Signal to Noise ratio']]
X3 = sm.add_constant(X3, has_constant='add')

#Target variable: performance gain in AUC-ROC
model3 = sm.OLS(y, X3).fit()
print(model3.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.177
#       p-value of Number of mean skewness = 0.553 
#===============================================================================
#Independent variables
X4 = df_results[[ 'Number of Features', 'Number of Classes',
       'Mean mean', 'Mean std', 'Mean kurtosis','Mean Correlation',
       'Mean Normalised Feature Entropy', 'Normalized Class Entropy', 'Signal to Noise ratio']]
X4 = sm.add_constant(X4, has_constant='add')

#Target variable: performance gain in AUC-ROC
model4 = sm.OLS(y, X4).fit()
print(model4.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.175
#       p-value of Number of Features = 0.570 
#===============================================================================
#Independent variables
X5 = df_results[[ 'Number of Classes','Mean mean', 'Mean std', 'Mean kurtosis','Mean Correlation',
       'Mean Normalised Feature Entropy', 'Normalized Class Entropy', 'Signal to Noise ratio']]
X5 = sm.add_constant(X5, has_constant='add')

#Target variable: performance gain in AUC-ROC
model5 = sm.OLS(y, X5).fit()
print(model5.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.173
#       p-value of Class entropy = 0.594
#===============================================================================
#Independent variables
X6 = df_results[[ 'Number of Classes','Mean mean', 'Mean std', 'Mean kurtosis','Mean Correlation',
       'Mean Normalised Feature Entropy', 'Signal to Noise ratio']]
X6 = sm.add_constant(X6, has_constant='add')

#Target variable: performance gain in AUC-ROC
model6 = sm.OLS(y, X6).fit()
print(model6.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')


#===============================================================================
#       R-squared = 0.171
#       p-value of Feature entropy = 0.179
#===============================================================================
#Independent variables
X7 = df_results[[ 'Number of Classes','Mean mean', 'Mean std', 'Mean kurtosis','Mean Correlation','Signal to Noise ratio']]
X7 = sm.add_constant(X7, has_constant='add')

#Target variable: performance gain in AUC-ROC
model7 = sm.OLS(y, X7).fit()
print(model7.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')


#===============================================================================
#       R-squared = 0.159
#      All p-value are small! (<0.06)
#===============================================================================

"""
Residual plot:
If the points are randomly dispersed around the horizontal axis, 
a linear regression model is usually appropriate for the data
"""
predictions = model7.predict(X7)
residual = y - predictions
plt.scatter(y, residual)
plt.xlabel("Performance gain", fontsize=14)
plt.ylabel("Residuals \n= Actual gain - predicted gain", fontsize=14)
plt.title("Residuals plot", fontsize=14)
plt.show()















