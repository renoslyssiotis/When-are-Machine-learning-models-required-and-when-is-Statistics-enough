import sys, os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[3]))
from retrieve_performance_gain import retrieve_best_ML_and_stats_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

df_results = pd.read_pickle(str(p.parents[5])+'/test/df_results.plk')
with open(str(p.parents[5])+'/test/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle)    

#AUC-ROC   
performance_gain_roc = retrieve_best_ML_and_stats_model(nested_results_roc)[2]
performance_gain_roc_list = list(performance_gain_roc.values())

mean_skewness = df_results['Mean skewness'].values.tolist()

# #==============================================================================
# #                              AUC-ROC
# #   Scatter plot:    Plot performance gain vs. Mean skewness
# #==============================================================================
x = np.array(mean_skewness)
y = np.array(performance_gain_roc_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Mean skewness", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-ROC) vs. Mean skewness", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(-1, 13.5)
plt.show()
