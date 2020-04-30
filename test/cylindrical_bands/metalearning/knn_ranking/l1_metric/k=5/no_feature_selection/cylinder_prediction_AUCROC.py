import sys, os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[7])+'/metalearners/knn_ranking_method/l1_metric')
from KNN_ranking_k_5 import KNN_ranking
import pandas as pd

#=====================META-FEATURE EXTRACTION==================================
with open(str(p.parents[5])+'/actual/cylinder_metafeatures_202.pickle', 'rb') as handle:
    meta_features = pickle.load(handle)

#============Load the meta-dataset and model performance of each dataset=======
df_results = pd.read_pickle(str(p.parents[7])+'/test/df_results.plk')

#nested_results is a nested dictionary with all the AUC-ROC performances for each dataset and all models
with open(str(p.parents[6])+'/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle)

#========================META-LEARNING: RANKING================================    
#KNN Ranking Method
top1, top2, top3 = KNN_ranking(df_results, meta_features, nested_results_roc)
print("==========================================")
print("           AUC-ROC         ")
print("==========================================")
print("Top 1 predicted model:      " + top1)
print("Top 2 predicted model:      " + top2)
print("Top 3 predicted model:      " + top3)

#Actual results
with open(str(p.parents[5])+'/actual/cylinder_top_3_roc.pickle', 'rb') as handle:
    actual_results = pickle.load(handle)
print("==========================================")
print("Top 1 ACTUAL model:      " + actual_results[0])
print("Top 2 ACTUAL model:      " + actual_results[1])
print("Top 3 ACTUAL model:      " + actual_results[2])