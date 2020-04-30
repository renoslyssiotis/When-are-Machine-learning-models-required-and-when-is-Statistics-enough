import pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[6])+'/test/df_results.plk')

#==============================CLUSTER 12=======================================
with open('subclusters_of_cluster_12.pickle', 'rb') as handle:
    cluster_12 = pickle.load(handle)   
 
best_models_prc_12 = {0: [], 1: []}#, 2: [], 3: [], 4: [], 5: [], 6: []}#, 7: [], 8: [], 9: [], 10: [], 11: []}

for key in cluster_12.keys(): #0 (since 1 sub-clusters)
    
    for dataset in cluster_12[key]:
        best_models_prc_12[key].append(df_results['Best model PRC'][dataset])

#==============================================================================
#                       PRC - BAR CHART FOR EACH SUB-CLUSTER of cluster 12
#==============================================================================
for key in list(best_models_prc_12.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(best_models_prc_12[key])))), list(dict(Counter(best_models_prc_12[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(best_models_prc_12[key])))), list(dict(Counter(best_models_prc_12[key])).keys()), rotation=45, fontsize = 14)
    plt.yticks(list(np.arange(1, max(list(dict(Counter(best_models_prc_12[key])).values())) + 1)),
               list(np.arange(1, max(list(dict(Counter(best_models_prc_12[key])).values())) + 1)))
    plt.title("Sub-cluster {} of Cluster 12".format(key), weight = 'bold', fontsize = 14)
    plt.ylabel("Counts of best-performing model\n (AUC-PRC)", fontsize = 14)
