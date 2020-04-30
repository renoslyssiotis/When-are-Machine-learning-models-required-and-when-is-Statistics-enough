import sys, os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[7])+'/metalearners/knn_ranking_method/l1_metric')
from KNN_ranking_k_5 import KNN_ranking

#Load the selected meta-dataset after performing ANOVA F-value method
with open(str(p.parents[7])+'/analysis/feature_selection/univariate_selection/ANOVA_X_f1_202.pickle', 'rb') as handle:
    metadataset_feature_selected = pickle.load(handle)
    
#=====================META-FEATURE EXTRACTION==================================
with open(str(p.parents[5])+'/actual/wine_metafeatures_202.pickle', 'rb') as handle:
    meta_features = pickle.load(handle)

#nested_results is a nested dictionary with all the f1 performances for each dataset and all models
with open(str(p.parents[6])+'/nested_results_f1.pickle', 'rb') as handle:
    nested_results_f1 = pickle.load(handle)    
    
"""    
Remove the meta-features which are not in the meta-dataset 
(i.e. the features which have not been selected in the feature selection process)
"""
metafeatures_to_be_removed = []

for metafeature in meta_features.keys():
    if metafeature in metadataset_feature_selected.columns:
        pass
    else:
        metafeatures_to_be_removed.append(metafeature)
     
[meta_features.pop(key) for key in metafeatures_to_be_removed] 

#========================META-LEARNING: RANKING================================    
#KNN Ranking Method
top1, top2, top3 = KNN_ranking(metadataset_feature_selected, meta_features, nested_results_f1)
print("==========================================")
print("           f1        ")
print("==========================================")
print("Top 1 predicted model:      " + top1)
print("Top 2 predicted model:      " + top2)
print("Top 3 predicted model:      " + top3)

#Actual results
with open(str(p.parents[5])+'/actual/wine_top_3_f1.pickle', 'rb') as handle:
    actual_results = pickle.load(handle)
print("==========================================")
print("Top 1 ACTUAL model:      " + actual_results[0])
print("Top 2 ACTUAL model:      " + actual_results[1])
print("Top 3 ACTUAL model:      " + actual_results[2])