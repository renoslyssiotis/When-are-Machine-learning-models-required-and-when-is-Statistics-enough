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

mean_kurtosis = df_results['Mean kurtosis'].values.tolist()

# #==============================================================================
# #                              AUC-prc
# #   Scatter plot:    Plot performance gain vs. Mean Kurtosis
# #==============================================================================
x = np.array(mean_kurtosis)
y = np.array(performance_gain_prc_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Mean kurtosis", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-PRC) vs. Mean Kurtosis", fontsize=20)
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
    performance_gain_below_20.append(performance_gain_prc_list[j])

#Copy x = mean_kurtosis_below_20 and y =performance_gain_below_20 in matlab