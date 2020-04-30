import os
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 750

df_results = pd.read_pickle(str(p.parents[2])+'/test/df_results.plk')
#==============================================================================
#                        COUNT PLOTS FOR THREE METRICS
#==============================================================================
N = 12 #Number of models used
x_labels = ['Logit','RForest','MLP', 'DTree','ADABoost','GaussianNB','BernoulliNB','QDA','Bagging', 'LDA','LinearSVC','Perceptron']
ROC_results = df_results['Best model ROC'].value_counts()
PRC_results = df_results['Best model PRC'].value_counts()
F1_results = df_results['Best model f1'].value_counts()

ROC = [ROC_results['auc_roc_logit'], ROC_results['auc_roc_rforest'], ROC_results['auc_roc_MLP'], ROC_results['auc_roc_dtree'], ROC_results['auc_roc_ADAboost'],
        ROC_results['auc_roc_gaussianNB'], ROC_results['auc_roc_bernoulliNB'], ROC_results['auc_roc_QDA'], ROC_results['auc_roc_bagging'], ROC_results['auc_roc_LDA'],
        ROC_results['auc_roc_linearSVC'], ROC_results['auc_roc_perceptron']]
PRC = [PRC_results['auc_prc_logit'], PRC_results['auc_prc_rforest'], PRC_results['auc_prc_MLP'], PRC_results['auc_prc_dtree'], PRC_results['auc_prc_ADAboost'],
        PRC_results['auc_prc_gaussianNB'], PRC_results['auc_prc_bernoulliNB'], PRC_results['auc_prc_QDA'], PRC_results['auc_prc_bagging'], PRC_results['auc_prc_LDA'],
        PRC_results['auc_prc_linearSVC'], PRC_results['auc_prc_perceptron']]
F1 = [F1_results['auc_f1_logit'], F1_results['auc_f1_rforest'], F1_results['auc_f1_MLP'], F1_results['auc_f1_dtree'], F1_results['auc_f1_ADAboost'],
        F1_results['auc_f1_gaussianNB'], F1_results['auc_f1_bernoulliNB'], F1_results['auc_f1_QDA'], F1_results['auc_f1_bagging'], F1_results['auc_f1_LDA'],
        F1_results['auc_f1_linearSVC'], F1_results['auc_f1_perceptron']]

ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, ROC, width, color='deepskyblue')
rects2 = ax.bar(ind + width, PRC, width, color='blue')
rects3 = ax.bar(ind + 2*width, F1, width, color='navy')

# add some text for labels, title and axes ticks
plt.ylabel("Count", fontsize=14, fontweight='bold')
plt.xlabel("Classification models", fontsize=14, fontweight='bold')
ax.set_title('Best-performing models for each metric', fontsize=14, fontweight='bold')
ax.set_xticks(ind + width / 2)
plt.xticks(rotation=45, ha="right")
ax.set_xticklabels(x_labels)
ax.legend((rects1[0], rects2[0], rects3[0]), ('AUC-ROC', 'AUC-PRC', 'F1'))
plt.show()

    
#==============================================================================
#                        COUNT PLOTS FOR 'LARGE' DATASETS
#==============================================================================
x_labels = ["RForest","Bagging","MLP","Logit","QDA", "Perceptron"]
N = len(x_labels)

ROC_results_large = df_results[df_results["Number of Instances"] > 5000]['Best model ROC'].value_counts()
PRC_results_large = df_results[df_results["Number of Instances"] > 5000]['Best model PRC'].value_counts()
F1_results_large = df_results[df_results["Number of Instances"] > 5000]['Best model f1'].value_counts()

ROC_ = [ROC_results_large['auc_roc_rforest'], ROC_results_large['auc_roc_bagging'], 
        ROC_results_large['auc_roc_MLP'], 0,
        ROC_results_large['auc_roc_QDA'], ROC_results_large['auc_roc_perceptron']]

PRC_ = [PRC_results_large['auc_prc_rforest'], PRC_results_large['auc_prc_bagging'],
        PRC_results_large['auc_prc_MLP'], PRC_results_large['auc_prc_logit'], 0, 0]

F1_ = [F1_results_large['auc_f1_rforest'], F1_results_large['auc_f1_bagging'],
        F1_results_large['auc_f1_MLP'], F1_results_large['auc_f1_logit'],0, 0]

ind = np.arange(N)  # the x locations for the groups
width = 0.20       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, ROC_, width, color='deepskyblue')
rects2 = ax.bar(ind + width, PRC_, width, color='blue')
rects3 = ax.bar(ind + 2*width, F1_, width, color='navy')

# add some text for labels, title and axes ticks
plt.ylabel("Count", fontsize=14, fontweight='bold')
plt.xlabel("Classification models", fontsize=14, fontweight='bold')
ax.set_title('Best-performing models for "large" datasets', fontsize=14, fontweight='bold')
ax.set_xticks(ind + width / 2)
plt.xticks(rotation=45, ha="right")
ax.set_xticklabels(x_labels)
ax.legend((rects1[0], rects2[0], rects3[0]), ('AUC-ROC', 'AUC-PRC', 'F1'))
#ax.legend((rects1[0], rects2[0]), ('AUC-ROC', 'AUC-PRC'))

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()