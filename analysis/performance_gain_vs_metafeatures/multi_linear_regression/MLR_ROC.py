import sys, os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[1]))
from retrieve_performance_gain import retrieve_best_ML_and_stats_model
import pandas as pd
import numpy as np
import statsmodels.api as sm    
import matplotlib.pyplot as plt

df_results = pd.read_pickle(str(p.parents[3])+'/test/df_results.plk')
with open(str(p.parents[3])+'/test/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle) 
    
#AUC-ROC   
performance_gain_roc = retrieve_best_ML_and_stats_model(nested_results_roc)[2]
performance_gain_roc_list = list(performance_gain_roc.values())

#===============================================================================
#                               ALL VARIABLES
#===============================================================================
#Independent variables
X1 = df_results[['Number of Instances', 'Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 'Mean Coefficient of Variation',
        'Mean skewness', 'Mean kurtosis', 'Mean Pearson Correlation', 'Mean Kendall Correlation',
        'Mean Spearman Correlation', 'Mean Median Absolute Deviation','Mean Inter-Quartile Range',
        'Proportion of Scaled Features with Outliers', 'Mean Canonical Correlations', 
        'Mean Normalised Feature Entropy', 'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X1 = sm.add_constant(X1, has_constant='add')
#Target variable: performance gain in AUC-ROC
y = pd.Series(data = performance_gain_roc_list,
                  index = list(X1.index))

model1 = sm.OLS(y, X1).fit()
print(model1.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = -0.665
#       p-value of Coefficient of variation = 0.932
#===============================================================================
#Independent variables
X2 = df_results[['Number of Instances', 'Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 
        'Mean skewness', 'Mean kurtosis', 'Mean Pearson Correlation', 'Mean Kendall Correlation',
        'Mean Spearman Correlation', 'Mean Median Absolute Deviation','Mean Inter-Quartile Range',
        'Proportion of Scaled Features with Outliers', 'Mean Canonical Correlations', 
        'Mean Normalised Feature Entropy', 'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X2 = sm.add_constant(X2, has_constant='add')

model2 = sm.OLS(y, X2).fit()
print(model2.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.188
#       p-value of Mean IQR = 0.890
#===============================================================================
#Independent variables
X3 = df_results[['Number of Instances', 'Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 
        'Mean skewness', 'Mean kurtosis', 'Mean Pearson Correlation', 'Mean Kendall Correlation',
        'Mean Spearman Correlation', 'Mean Median Absolute Deviation',
        'Proportion of Scaled Features with Outliers', 'Mean Canonical Correlations', 
        'Mean Normalised Feature Entropy', 'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X3 = sm.add_constant(X3, has_constant='add')

#Target variable: performance gain in AUC-ROC
model3 = sm.OLS(y, X3).fit()
print(model3.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.188
#       p-value of Pearson correlation = 0.830 
#===============================================================================
#Independent variables
X4 = df_results[['Number of Instances', 'Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 
        'Mean skewness', 'Mean kurtosis', 'Mean Kendall Correlation',
        'Mean Spearman Correlation', 'Mean Median Absolute Deviation',
        'Proportion of Scaled Features with Outliers', 'Mean Canonical Correlations', 
        'Mean Normalised Feature Entropy', 'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X4 = sm.add_constant(X4, has_constant='add')

#Target variable: performance gain in AUC-ROC
model4 = sm.OLS(y, X4).fit()
print(model4.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.188
#       p-value of Mean Kurtosis = 0.753 
#===============================================================================
#Independent variables
X5 = df_results[['Number of Instances', 'Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 
        'Mean skewness', 'Mean Kendall Correlation',
        'Mean Spearman Correlation', 'Mean Median Absolute Deviation',
        'Proportion of Scaled Features with Outliers', 'Mean Canonical Correlations', 
        'Mean Normalised Feature Entropy', 'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X5 = sm.add_constant(X5, has_constant='add')

#Target variable: performance gain in AUC-ROC
model5 = sm.OLS(y, X5).fit()
print(model5.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.187
#       p-value of Mean Normalised Feature Entropy = 0.673
#===============================================================================
#Independent variables
X6 = df_results[['Number of Instances', 'Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 
        'Mean skewness', 'Mean Kendall Correlation',
        'Mean Spearman Correlation', 'Mean Median Absolute Deviation',
        'Proportion of Scaled Features with Outliers', 'Mean Canonical Correlations', 
        'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X6 = sm.add_constant(X6, has_constant='add')

#Target variable: performance gain in AUC-ROC
model6 = sm.OLS(y, X6).fit()
print(model6.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.186
#       p-value of Kendall correlation = 0.620
#===============================================================================
#Independent variables
X7 = df_results[['Number of Instances', 'Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 
        'Mean skewness', 
        'Mean Spearman Correlation', 'Mean Median Absolute Deviation',
        'Proportion of Scaled Features with Outliers', 'Mean Canonical Correlations', 
        'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X7 = sm.add_constant(X7, has_constant='add')

#Target variable: performance gain in AUC-ROC
model7 = sm.OLS(y, X7).fit()
print(model7.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.185
#       p-value of Spearman correlation = 0.709
#===============================================================================
#Independent variables
X8 = df_results[['Number of Instances', 'Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 
        'Mean skewness', 
        'Mean Median Absolute Deviation',
        'Proportion of Scaled Features with Outliers', 'Mean Canonical Correlations', 
        'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X8 = sm.add_constant(X8, has_constant='add')

#Target variable: performance gain in AUC-ROC
model8 = sm.OLS(y, X8).fit()
print(model8.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.185
#       p-value of Proportion of scaled features with outliers = 0.416
#===============================================================================
#Independent variables
X9 = df_results[['Number of Instances', 'Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 'Mean skewness', 'Mean Median Absolute Deviation',
        'Mean Canonical Correlations', 'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X9 = sm.add_constant(X9, has_constant='add')

#Target variable: performance gain in AUC-ROC
model9 = sm.OLS(y, X9).fit()
print(model9.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.185
#       p-value of Number of instances= 0.310
#===============================================================================
#Independent variables
X10 = df_results[['Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Number of Classes', 'Mean mean', 'Mean std', 'Mean skewness',
        'Mean Median Absolute Deviation', 'Mean Canonical Correlations', 
        'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X10 = sm.add_constant(X10, has_constant='add')

#Target variable: performance gain in AUC-ROC
model10 = sm.OLS(y, X10).fit()
print(model10.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.180
#       p-value of Number of Classes= 0.254
#===============================================================================
#Independent variables
X11 = df_results[['Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Mean mean', 'Mean std', 'Mean skewness', 'Mean Median Absolute Deviation',
        'Mean Canonical Correlations', 'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X11 = sm.add_constant(X11, has_constant='add')

#Target variable: performance gain in AUC-ROC
model11 = sm.OLS(y, X11).fit()
print(model11.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.180
#       p-value of Mean of mean = 0.254
#===============================================================================
#Independent variables
X12 = df_results[['Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Mean std', 'Mean skewness', 'Mean Median Absolute Deviation',
        'Mean Canonical Correlations', 'Normalized Class Entropy', 'Signal to Noise ratio',
        'Max Mutual Information', 'Mean Mutual Information']]
X12 = sm.add_constant(X12, has_constant='add')

#Target variable: performance gain in AUC-ROC
model12 = sm.OLS(y, X12).fit()
print(model12.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.180
#       p-value of SNR = 0.247
#===============================================================================
#Independent variables
X13 = df_results[['Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Mean std', 'Mean skewness', 'Mean Median Absolute Deviation',
        'Mean Canonical Correlations','Normalized Class Entropy', 
        'Max Mutual Information', 'Mean Mutual Information']]
X13 = sm.add_constant(X13, has_constant='add')

#Target variable: performance gain in AUC-ROC
model13 = sm.OLS(y, X13).fit()
print(model13.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.180
#       p-value of constant = 0.254
#===============================================================================
#Independent variables
X14 = df_results[['Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Mean std', 'Mean skewness', 'Mean Median Absolute Deviation',
        'Mean Canonical Correlations','Normalized Class Entropy', 
        'Max Mutual Information', 'Mean Mutual Information']]

#Target variable: performance gain in AUC-ROC
model14 = sm.OLS(y, X14).fit()
print(model14.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.518
#       p-value of mean std = 0.643
#===============================================================================
#Independent variables
X15 = df_results[['Number of Features', 'Proportion of Numerical Features',
        'Number of Dummy Variables after pre-processing','Proportion of Less Frequent Class',
        'Mean skewness', 'Mean Median Absolute Deviation',
        'Mean Canonical Correlations','Normalized Class Entropy', 
        'Max Mutual Information', 'Mean Mutual Information']]

#Target variable: performance gain in AUC-ROC
model15 = sm.OLS(y, X15).fit()
print(model15.summary())
print('\n' + '-------------------------------------------------------------------------------------------------------'+'\n' + '-------------------------------------------------------------------------------------------------------'+'\n'+'\n')

#===============================================================================
#       R-squared = 0.517
#      All p-value are small! (<0.05)
#===============================================================================

"""
Residual plot:
If the points are randomly dispersed around the horizontal axis, 
a linear regression model is usually appropriate for the data
"""
predictions = model15.predict(X15)
residual = y - predictions
residuals = plt.scatter(y, residual, s = 20, c = 'red', alpha = 0.5)
plt.xlabel("Performance gain (AUC-ROC)", fontsize=14)
plt.ylabel("Residuals", fontsize=14)
plt.title("Residuals plot", fontsize=14)
m, b = np.polyfit(y, residual, 1)
plt.plot(y, m*y + b, c = 'blue', label = 'Line of best fit')
plt.legend(loc="upper left")
# plt.legend((residuals, line), ('Residuals', 'Line of best fit'),
#             ncol = 2,
#             scatterpoints=1)
plt.savefig('residuals_plot.png',  dpi=500)
plt.show()
















