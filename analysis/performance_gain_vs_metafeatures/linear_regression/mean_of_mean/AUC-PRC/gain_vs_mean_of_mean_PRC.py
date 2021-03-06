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
from scipy.optimize import curve_fit

df_results = pd.read_pickle(str(p.parents[5])+'/test/df_results.plk')
with open(str(p.parents[5])+'/test/nested_results_prc.pickle', 'rb') as handle:
    nested_results_prc = pickle.load(handle)  
     
#AUC-ROC   
performance_gain_prc = retrieve_best_ML_and_stats_model(nested_results_prc)[2]
performance_gain_prc_list = list(performance_gain_prc.values())
mean_of_mean = df_results['Mean mean'].values.tolist()

# #==============================================================================
# #                              AUC-PRC
# #      Scatter plot:    Plot performance gain vs. mean of mean value of features
# #==============================================================================
x = np.array(mean_of_mean)
y = np.array(performance_gain_prc_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Mean of mean value of features", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-PRC) vs. Mean of mean value of features", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(-0.1e-15, .05e-14)
plt.show()

# def func(x, a, c, d):
#     return a*np.exp(-c*x)+d

# figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
# popt, pcov = curve_fit(func, x, y, p0=(1, 1e-3, 1))

# xx = np.linspace(0, 0.05e-14)
# yy = func(xx, *popt)
# plt.plot(x, y, 'ro', label = 'Data')
# plt.plot(xx, yy, 'b-', ls='--', label = 'Exponential fit')
# plt.legend(fontsize=18)
# plt.xlabel("Number of instances", fontsize=18)
# plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
# plt.title("Performance gain (AUC-ROC) vs. Number of instances", fontsize=20)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.show()