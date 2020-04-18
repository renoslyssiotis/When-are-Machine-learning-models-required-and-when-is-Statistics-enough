import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Meta-learners/KNN Ranking Method/RMSE')
from KNN_ranking_k_5_RMSE import KNN_ranking
import pickle

#Load the selected meta-dataset after performing zero-variance threshold
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Analysis/Feature Selection/Variance Threshold/var_threshold_zeroVar_X_202.pickle', 'rb') as handle:
    metadataset_feature_selected = pickle.load(handle)
    
#=====================META-FEATURE EXTRACTION==================================
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Cylinder Bands test dataset/Actual results/cylinder_metafeatures_202.pickle', 'rb') as handle:
    meta_features = pickle.load(handle)

#nested_results is a nested dictionary with all the AUC-ROC performances for each dataset and all models
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle)    
    
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
top1, top2, top3 = KNN_ranking(metadataset_feature_selected, meta_features, nested_results_roc)
print("==========================================")
print("           AUC-ROC         ")
print("==========================================")
print("Top 1 predicted model:      " + top1)
print("Top 2 predicted model:      " + top2)
print("Top 3 predicted model:      " + top3)

#Actual results
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Cylinder Bands test dataset/Actual results/cylinder_top_3_roc.pickle', 'rb') as handle:
    actual_results = pickle.load(handle)
print("==========================================")
print("Top 1 ACTUAL model:      " + actual_results[0])
print("Top 2 ACTUAL model:      " + actual_results[1])
print("Top 3 ACTUAL model:      " + actual_results[2])