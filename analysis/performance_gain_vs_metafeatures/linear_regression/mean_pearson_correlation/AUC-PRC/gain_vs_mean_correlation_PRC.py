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
with open(str(p.parents[5])+'/test/nested_results_prc.pickle', 'rb') as handle:
    nested_results_prc = pickle.load(handle)    

#AUC-PRC   
performance_gain_prc = retrieve_best_ML_and_stats_model(nested_results_prc)[2]
performance_gain_prc_list = list(performance_gain_prc.values())

mean_correlation = df_results['Mean Pearson Correlation'].values.tolist()

# #==============================================================================
# #                              AUC-PRC
# #   Scatter plot:    Plot performance gain vs. Correlation
# #==============================================================================
x = np.array(mean_correlation)
y = np.array(performance_gain_prc_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Mean Pearson Correlation", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-PRC) vs. Mean Pearson Correlation", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(-0.5, 1.25)
plt.show()
