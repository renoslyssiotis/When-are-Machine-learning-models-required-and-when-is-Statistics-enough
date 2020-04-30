import pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[6])+'/test/df_results.plk')

#==============================CLUSTER 1=======================================
with open('subclusters_of_cluster_1.pickle', 'rb') as handle:
    cluster_1 = pickle.load(handle)   
 
best_models_f1_1 = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}#, 8: [], 9: [], 10: [], 11: []}

for key in cluster_1.keys(): #0 - 6 (since 7 sub-clusters)
    
    for dataset in cluster_1[key]:
        best_models_f1_1[key].append(df_results['Best model f1'][dataset])

#==============================================================================
#                       f1 - BAR CHART FOR EACH SUB-CLUSTER of cluster 1
#==============================================================================
for key in list(best_models_f1_1.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(best_models_f1_1[key])))), list(dict(Counter(best_models_f1_1[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(best_models_f1_1[key])))), list(dict(Counter(best_models_f1_1[key])).keys()), rotation=45, fontsize = 14)
    plt.yticks(list(np.arange(1, max(list(dict(Counter(best_models_f1_1[key])).values())) + 1)),
               list(np.arange(1, max(list(dict(Counter(best_models_f1_1[key])).values())) + 1)))
    plt.title("Sub-cluster {} of Cluster 1".format(key), weight = 'bold', fontsize = 14)
    plt.ylabel("Counts of best-performing model\n (F1)", fontsize = 14)
