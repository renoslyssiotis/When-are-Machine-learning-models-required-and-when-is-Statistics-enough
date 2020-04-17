import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#==============================CLUSTER 7=======================================
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Analysis/Dataset clustering/Two-level clustering/General Clustering standardised/Second level clustering/AUC-ROC/subclusters_of_cluster_7.pickle', 'rb') as handle:
    cluster_7 = pickle.load(handle)   
 
best_models_prc_7 = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}#, 6: []}#, 7: [], 8: [], 9: [], 10: [], 11: []}

for key in cluster_7.keys(): #0 - 5 (since 6 sub-clusters)
    
    for dataset in cluster_7[key]:
        best_models_prc_7[key].append(df_results['Best model PRC'][dataset])

#==============================================================================
#                       PRC - BAR CHART FOR EACH SUB-CLUSTER of cluster 7
#==============================================================================
for key in list(best_models_prc_7.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(best_models_prc_7[key])))), list(dict(Counter(best_models_prc_7[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(best_models_prc_7[key])))), list(dict(Counter(best_models_prc_7[key])).keys()), rotation=45, fontsize = 14)
    plt.yticks(list(np.arange(1, max(list(dict(Counter(best_models_prc_7[key])).values())) + 1)),
               list(np.arange(1, max(list(dict(Counter(best_models_prc_7[key])).values())) + 1)))
    plt.title("Sub-cluster {} of Cluster 7".format(key), weight = 'bold', fontsize = 14)
    plt.ylabel("Counts of best-performing model\n (AUC-PRC)", fontsize = 14)