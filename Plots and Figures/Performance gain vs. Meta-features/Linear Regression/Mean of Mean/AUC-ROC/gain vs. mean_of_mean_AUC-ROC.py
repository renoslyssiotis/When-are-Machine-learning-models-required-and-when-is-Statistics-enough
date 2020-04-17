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
mean_of_mean = df_results['Mean mean'].values.tolist()

# #==============================================================================
# #                              AUC-ROC
# #      Scatter plot:    Plot performance gain vs. mean of mean value of features
# #==============================================================================
x = np.array(mean_of_mean)
y = np.array(performance_gain_roc_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Mean of mean value of features", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-ROC) vs. Mean of mean value of features", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
#plt.xlim(-9e-15, 1.5e-14)
plt.xlim(-0.1e-15, .05e-14)
plt.show()

#Consider the datasets whose mean value is between -0.1e-15 and 0.05e-14
mean_mean_between_range = []
index_of_mean_between_range = []
for index, value in enumerate(mean_of_mean):
    if -0.1e-15 <= value <= 0.05e-14:
        mean_mean_between_range.append(value)
        index_of_mean_between_range.append(index)
    else:
        pass

performance_gain_between_range = []
for j in index_of_mean_between_range:
    performance_gain_between_range.append(performance_gain_roc_list[j]) #90 elements

#Copy x = mean_mean_between_range and y = performance_gain_between_range in Matlab
