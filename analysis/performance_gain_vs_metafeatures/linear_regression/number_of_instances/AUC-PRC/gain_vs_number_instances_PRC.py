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
    
#AUC-RPC   
performance_gain_prc = retrieve_best_ML_and_stats_model(nested_results_prc)[2]
performance_gain_prc_list = list(performance_gain_prc.values())

number_of_instances = df_results['Number of Instances'].values.tolist()

# #==============================================================================
# #                              AUC-PRC
# #      Scatter plot:    Plot performance gain vs. number of instances
# #==============================================================================
x = np.array(number_of_instances)
y = np.array(performance_gain_prc_list)

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
plt.plot(x, y, 'ro', label = 'Data')
plt.xlabel("Number of instances", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-PRC) vs. Number of instances", fontsize=20)
plt.xticks(fontsize=18)
plt.legend(fontsize=18)
plt.yticks(fontsize=18)
# plt.xlim(0, 20000)
plt.show()

#==============================================================================
# 202 Datasets: Exp. fit: Gain = -0.08658749*exp(-0.00025238*Number_Instances)+0.16777879
#==============================================================================
def func(x, a, c, d):
    return a*np.exp(-c*x)+d

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
popt, pcov = curve_fit(func, x, y, p0=(1, 1e-3, 1))

xx = np.linspace(0, 250000)
yy = func(xx, *popt)
plt.plot(x, y, 'ro', label = 'Data')
plt.plot(xx, yy, 'b-', ls='--', label = 'Exponential fit')
plt.legend(fontsize=18)
plt.xlabel("Number of instances", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-PRC) vs. Number of instances", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#==============================================================================
#             For datasets with less than 2,000 instances
# Exp. fit: Gain = 0.18633979*exp(-0.01420243*Number_Instances)+0.04919477
#==============================================================================
number_of_instances_below_2000 = []
index_of_instances_below_2000 = []
for index, value in enumerate(number_of_instances):
    if value <= 2000:
        number_of_instances_below_2000.append(value)
        index_of_instances_below_2000.append(index)
    else:
        pass

performance_gain_below_2000 = []
for j in index_of_instances_below_2000:
    performance_gain_below_2000.append(performance_gain_prc_list[j])
    
x = np.array(number_of_instances_below_2000)
y = np.array(performance_gain_below_2000)
def func(x, a, c, d):
    return a*np.exp(-c*x)+d

figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
popt_, pcov_ = curve_fit(func, x, y, p0=(1, 1e-3, 1))

xx = np.linspace(0, 2000)
yy = func(xx, *popt_)
plt.plot(x, y, 'ro', label = 'Data')
plt.plot(xx, yy, 'b-', ls='--', label = 'Exponential fit')
plt.legend(fontsize=18)
plt.xlabel("Number of instances", fontsize=18)
plt.ylabel("Performance gain of ML vs. Statistics", fontsize=18)
plt.title("Performance gain (AUC-PRC) vs. Number of instances", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


#==============================================================================
#             Confidence intervals: Check Matlab script
#==============================================================================
