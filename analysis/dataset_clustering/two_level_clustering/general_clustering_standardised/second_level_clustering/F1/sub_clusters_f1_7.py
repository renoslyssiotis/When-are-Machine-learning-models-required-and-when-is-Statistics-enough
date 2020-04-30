import pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[6])+'/test/df_results.plk')

#==============================CLUSTER 7=======================================
with open('subclusters_of_cluster_7.pickle', 'rb') as handle:
    cluster_7 = pickle.load(handle)   
 
best_models_f1_7 = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}#, 6: []}#, 7: [], 8: [], 9: [], 10: [], 11: []}

for key in cluster_7.keys(): #0 - 5 (since 6 sub-clusters)
    
    for dataset in cluster_7[key]:
        best_models_f1_7[key].append(df_results['Best model f1'][dataset])

#==============================================================================
#                       f1 - BAR CHART FOR EACH SUB-CLUSTER of cluster 7
#==============================================================================
for key in list(best_models_f1_7.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(best_models_f1_7[key])))), list(dict(Counter(best_models_f1_7[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(best_models_f1_7[key])))), list(dict(Counter(best_models_f1_7[key])).keys()), rotation=45, fontsize = 14)
    plt.yticks(list(np.arange(1, max(list(dict(Counter(best_models_f1_7[key])).values())) + 1)),
               list(np.arange(1, max(list(dict(Counter(best_models_f1_7[key])).values())) + 1)))
    plt.title("Sub-cluster {} of Cluster 7".format(key), weight = 'bold', fontsize = 14)
    plt.ylabel("Counts of best-performing model\n (F1)", fontsize = 14)
