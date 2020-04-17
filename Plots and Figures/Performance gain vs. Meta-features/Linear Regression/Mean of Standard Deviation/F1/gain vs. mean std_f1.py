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

#f1 metric
performance_gain_f1 = retrieve_best_ML_and_stats_model(nested_results_f1)[2]
performance_gain_f1_list = list(performance_gain_f1.values())

mean_std = df_results['Mean std'].values.tolist()

# #==============================================================================
# #                              f1 metric
# #      Scatter plot:    Plot performance gain vs. mean of std
# #==============================================================================
x = np.array(mean_std)
y = np.array(performance_gain_f1_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Mean of standard deviation", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (f1 metric) vs. Mean of standard deviation", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0.999, 1.035)
plt.show()
