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
with open(str(p.parents[5])+'/test/nested_results_f1.pickle', 'rb') as handle:
    nested_results_f1 = pickle.load(handle)  
     
#AUC-ROC   
performance_gain_f1 = retrieve_best_ML_and_stats_model(nested_results_f1)[2]
performance_gain_f1_list = list(performance_gain_f1.values())
mean_of_mean = df_results['Mean mean'].values.tolist()

# #==============================================================================
# #                              f1 metric
# #      Scatter plot:    Plot performance gain vs. mean of mean value of features
# #==============================================================================
x = np.array(mean_of_mean)
y = np.array(performance_gain_f1_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Mean of mean value of features", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (f1 metric) vs. Mean of mean value of features", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(-0.1e-15, .05e-14)
plt.show()
