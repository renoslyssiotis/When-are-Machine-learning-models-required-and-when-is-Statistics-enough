import os, sys, time, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[3])+'/metalearners/knn_ranking_method/RMSE')
from KNN_ranking_k_3_RMSE import KNN_ranking
import matplotlib.pyplot as plt

#nested_results is a nested dictionary with all the AUC-ROC performances for each dataset and all models
with open(str(p.parents[3])+'/test/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle)    

#Load the selected meta-dataset after selecting the customised set of meta-features
with open(str(p.parents[2])+'/feature_selection/customised_set/customised_X_AUCROC_202.pickle', 'rb') as handle:
    metadataset_feature_selected = pickle.load(handle)

#=====================META-FEATURE EXTRACTION==================================
with open(str(p.parents[3])+'/test/cylindrical_bands/actual/cylinder_metafeatures_202.pickle', 'rb') as handle:
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
plt.figure(dpi=1200)
plt.plot(training_datasets, times, color = 'blue', label = 'Cylindrical bands dataset')
plt.legend(loc="upper left", fontsize = 12)
plt.xlabel('Number of training datasets', fontsize = 14)
plt.ylabel('Execution time of meta-learning [s]', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()