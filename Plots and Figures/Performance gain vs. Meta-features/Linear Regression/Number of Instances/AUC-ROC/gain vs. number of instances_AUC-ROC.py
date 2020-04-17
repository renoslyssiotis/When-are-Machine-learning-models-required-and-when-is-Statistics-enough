import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Performance gain vs. Meta-features')
from retrieve_performance_gain import retrieve_best_ML_and_stats_model
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle)    

#AUC-ROC   
performance_gain_roc = retrieve_best_ML_and_stats_model(nested_results_roc)[2]
performance_gain_roc_list = list(performance_gain_roc.values())

number_of_instances = df_results['Number of Instances'].values.tolist()

# #==============================================================================
# #                              AUC-ROC
# #      Scatter plot:    Plot performance gain vs. number of instances
# #==============================================================================
x = np.array(number_of_instances)
y = np.array(performance_gain_roc_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Number of instances", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-ROC) vs. Number of instances", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 20000)
plt.show()

#==============================================================================
# 133 Datasets: Exp. fit: Gain = 0.27161164*exp(-0.01891008*Number_Instances)+0.06707043
# 202 Datasets: Exp. fit: Gain = -0.08227625*exp(-0.00028126*Number_Instances)+0.15881178
#==============================================================================
def func(x, a, c, d):
    return a*np.exp(-c*x)+d

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
popt, pcov = curve_fit(func, x, y, p0=(1, 1e-3, 1))

xx = np.linspace(0, 250000)
yy = func(xx, *popt)
plt.plot(x, y, 'ro', label = 'Data')
plt.plot(xx, yy, 'b-', ls='--', label = 'Exponential fit')
plt.legend(fontsize=18)
plt.xlabel("Number of instances", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-ROC) vs. Number of instances", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#==============================================================================
#=============================133 Datasets====================================
#             For datasets with less than 2,000 instances
# Exp. fit: Gain = 0.26630041*exp(-0.01216286*Number_Instances)+0.03183699

#             For datasets with less than 1,000 instances
# Exp. fit: Gain = 0.31857489*exp(-0.02898858*Number_Instances)+0.05755904

#=============================202 Datasets====================================
#             For datasets with less than 2,000 instances
# Exp. fit: Gain = 0.08719188*exp(-0.01162717*Number_Instances)+0.06117672

#             For datasets with less than 1,000 instances
# Exp. fit: Gain = 0.15852088*exp(0.018979.*Number_Instances)+0.06110912

#==============================================================================
number_of_instances_below_2000 = []
index_of_instances_below_2000 = []
for index, value in enumerate(number_of_instances):
    if value <= 2000:
        number_of_instances_below_2000.append(value)
        index_of_instances_below_2000.append(index)
    else:
        pass

performance_gain_below_2000 = []
for j in index_of_instances_below_2000:
    performance_gain_below_2000.append(performance_gain_roc_list[j])
    
    
x = np.array(number_of_instances_below_2000)
y = np.array(performance_gain_below_2000)
def func(x, a, c, d):
    return a*np.exp(-c*x)+d

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
popt_, pcov_ = curve_fit(func, x, y, p0=(1, 1e-3, 1))

xx = np.linspace(0, 2000)
yy = func(xx, *popt_)
plt.plot(x, y, 'ro', label = 'Data')
plt.plot(xx, yy, 'b-', ls='--', label = 'Exponential fit')
plt.legend(fontsize=18)
plt.xlabel("Number of instances", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-ROC) vs. Number of instances", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


#==============================================================================
#             Confidence intervals: Check Matlab script
#==============================================================================
