import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Meta-learners/KNN Ranking Method/RMSE')
from KNN_ranking_k_3_RMSE import KNN_ranking
import pickle
import time
import matplotlib.pyplot as plt

#nested_results is a nested dictionary with all the AUC-ROC performances for each dataset and all models
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle)    

#Load the selected meta-dataset after selecting the customised set of meta-features
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Analysis/Feature Selection/Customised set/customised_X_AUCROC_202.pickle', 'rb') as handle:
    metadataset_feature_selected = pickle.load(handle)

#=====================META-FEATURE EXTRACTION==================================
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Wine quality test dataset/Actual results/wine_metafeatures_202.pickle', 'rb') as handle:
    wine_meta_features = pickle.load(handle)

"""    
Remove the meta-features which are not in the meta-dataset 
(i.e. the features which have not been selected in the feature selection process)
"""
metafeatures_to_be_removed = []

for metafeature in wine_meta_features.keys():
    if metafeature in metadataset_feature_selected.columns:
        pass
    else:
        metafeatures_to_be_removed.append(metafeature)
     
[wine_meta_features.pop(key) for key in metafeatures_to_be_removed]  
 
times = []

for i in range(3,203):
    
    start_time = time.time()  
    temp_metadaset = metadataset_feature_selected.iloc[:i, :]
    top1, top2, top3 = KNN_ranking(temp_metadaset, wine_meta_features, nested_results_roc)
    times.append((time.time() - start_time))

training_datasets = list(range(3,203))
plt.plot(training_datasets, times)