import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Performance gain vs. Meta-features')
from retrieve_performance_gain import retrieve_best_ML_and_stats_model
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_f1.pickle', 'rb') as handle:
    nested_results_f1 = pickle.load(handle)    

#F1 metric   
performance_gain_f1 = retrieve_best_ML_and_stats_model(nested_results_f1)[2]
performance_gain_f1_list = list(performance_gain_f1.values())

mean_kurtosis = df_results['Mean kurtosis'].values.tolist()

# #==============================================================================
# #                              F1 metric
# #   Scatter plot:    Plot performance gain vs. Mean Kurtosis
# #==============================================================================
x = np.array(mean_kurtosis)
y = np.array(performance_gain_f1_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Mean kurtosis", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (f1 metric) vs. Mean Kurtosis", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(-1, 20) #max_lim = 1200
plt.show()

#Consider the datasets whose kurtosis is less than 20:
mean_kurtosis_below_20 = []
index_of_kurtosis_below_20 = []
for index, value in enumerate(mean_kurtosis):
    if value <= 20:
        mean_kurtosis_below_20.append(value)
        index_of_kurtosis_below_20.append(index)
    else:
        pass

performance_gain_below_20 = []
for j in index_of_kurtosis_below_20:
    performance_gain_below_20.append(performance_gain_f1_list[j])

#Copy x = mean_kurtosis_below_20 and y =performance_gain_below_20 in matlab