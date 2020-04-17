import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Meta-learners/KNN Ranking Method')
import pickle
import pandas as pd

#Load the selected meta-dataset after performing zero-variance threshold
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Univariate Selection/ANOVA_X_AUCROC.pickle', 'rb') as handle:
    metadataset_feature_selected = pickle.load(handle)
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#=====================META-FEATURE EXTRACTION==================================
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Cylinder Bands test dataset/Actual results/cylinder_metafeatures.pickle', 'rb') as handle:
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
#############################################################################
#############################################################################
    
distance_count = []
meta_features_list = list(meta_features.keys())
meta_features_values = list(meta_features.values())

#Find the 'closeness' of the test dataset with the training datasets
#in the meta-feature space:

for dataset in list(metadataset_feature_selected.index): #Dataset 0, Dataset 1, ..., Dataset 202   
    
    summation = 0
    
    for index,value in enumerate(meta_features_list):  
        distance = abs(metadataset_feature_selected.loc[dataset][index] - meta_features_values[index])
        summation += distance
        
    distance_count.append(summation)
# print(distance_count)
# print(len(distance_count))
"""
These are the indeces of the meta-dataset that indicate which k=5 datasets are 
the closest to the test dataset:
"""    
min_index_1 = distance_count.index(sorted(distance_count)[0])
print("mind_index_1: %d" %min_index_1)
min_index_2 = distance_count.index(sorted(distance_count)[1])
min_index_3 = distance_count.index(sorted(distance_count)[2])    
min_index_4 = distance_count.index(sorted(distance_count)[3])
min_index_5 = distance_count.index(sorted(distance_count)[4])

#Create a ranking dataframe
ranking_df = pd.DataFrame(data = [list(nested_results_roc[min_index_1].values()),
                              list(nested_results_roc[min_index_2].values()),
                              list(nested_results_roc[min_index_3].values()),
                              list(nested_results_roc[min_index_4].values()),
                              list(nested_results_roc[min_index_5].values())], 
                          columns = list(nested_results_roc[1].keys()),
                          index = ['Closest dataset 1','Closest dataset 2', 'Closest dataset 3', 'Closest dataset 4', 'Closest dataset 5'])

closest_5_datasets = ranking_df.T                

#Transform the AUC-ROC metric into model rankings:
closest_5_datasets['Rank dataset 1'] = closest_5_datasets['Closest dataset 1'].rank(ascending=False)
closest_5_datasets['Rank dataset 2'] = closest_5_datasets['Closest dataset 2'].rank(ascending=False)
closest_5_datasets['Rank dataset 3'] = closest_5_datasets['Closest dataset 3'].rank(ascending=False)
closest_5_datasets['Rank dataset 4'] = closest_5_datasets['Closest dataset 4'].rank(ascending=False)
closest_5_datasets['Rank dataset 5'] = closest_5_datasets['Closest dataset 5'].rank(ascending=False)
    

#Compute the average rank of each model for the three closest datasets:
average_rank = []
for i in range(len(closest_5_datasets.index)):
    average_rank.append((closest_5_datasets.iloc[i][-5:].sum())/5)

closest_5_datasets['Average model rank'] = average_rank
closest_5_datasets = closest_5_datasets.sort_values(by = 'Average model rank')

#Once the average rank of each model is obtained, get a ranking recommendation:
top1 = closest_5_datasets.index[0]
top2 = closest_5_datasets.index[1]
top3 = closest_5_datasets.index[2]
print(top1, top2, top3)


