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
with open(str(p.parents[5])+'/test/nested_results_f1.pickle', 'rb') as handle:
    nested_results_f1 = pickle.load(handle)  

#F1
performance_gain_f1 = retrieve_best_ML_and_stats_model(nested_results_f1)[2]
performance_gain_f1_list = list(performance_gain_f1.values())

number_of_features = df_results['Number of Features'].values.tolist()

# #==============================================================================
# #                              F1 metric
# #      Scatter plot:    Plot performance gain vs. number of features
# #==============================================================================
x = np.array(number_of_features)
y = np.array(performance_gain_f1_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Number of features", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-PRC) vs. Number of features", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 100)
plt.show()

def func(x, a, c, d):
    return a*np.exp(-c*x)+d

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
popt, pcov = curve_fit(func, x, y, p0=(1, 1e-3, 1))

xx = np.linspace(0, 100)
yy = func(xx, *popt)
plt.plot(x, y, 'ro', label = 'Data')
plt.plot(xx, yy, 'b-', ls='--', label = 'Exponential fit')
plt.legend(fontsize=18)
plt.xlabel("Number of features", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (f1) vs. Number of features", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 100)
plt.show()