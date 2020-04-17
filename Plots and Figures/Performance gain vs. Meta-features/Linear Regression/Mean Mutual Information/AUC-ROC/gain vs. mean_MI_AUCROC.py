import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Performance gain vs. Meta-features')
from retrieve_performance_gain import retrieve_best_ML_and_stats_model
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle)    

#AUC-ROC   
performance_gain_roc = retrieve_best_ML_and_stats_model(nested_results_roc)[2]
performance_gain_roc_list = list(performance_gain_roc.values())

mean_mutual_information = df_results['Mean Mutual Information'].values.tolist()

# #==============================================================================
# #                              AUC-ROC
# #   Scatter plot:    Plot performance gain vs. Mean skewness
# #==============================================================================
x = np.array(mean_mutual_information)
y = np.array(performance_gain_roc_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Mean Mutual Information", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-ROC) vs. Mean Mutual Information", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
# plt.xlim(0.3, 0.7)
plt.show()
