import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Models')
from pre_processing import preProcessor
from meta_features import metaFeatures
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Meta-learners/KNN Ranking Method/RMSE')
from KNN_ranking_k_3_RMSE import KNN_ranking
import pickle
import pandas as pd

'''
Obtain a predidiction for the best-performing model on your binary
classification dataset, using the following meta-learner

               KNN meta-learner, k = 3, RMSE distance metric
               customised set of meta-features for AUC-ROC

(i.e.  (1) Number of instances,
       (2) Mean standard deviation
       (3) Mean skewness
       (4) Mean Pearson correlation
       (5) Mean normalised feature entropy
       (6) Mean normalised class entropy
'''

#Import your dataset: Change the path to below to enter the path on your own computer:
df = pd.read_csv('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/test.csv')

#Pre-processing
df_processed = preProcessor(df)
[dummy_columns, features_columns, scaled_features, X, y] = df_processed.pre_processing()
df_processed = X.join(y)

#Meta-feature extraction
df_meta_features = metaFeatures(df_processed, df)
meta_features = df_meta_features.get_meta_features()

#Load the selected meta-dataset after selecting the customised set of meta-features
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Analysis/Feature Selection/Customised set/customised_X_AUCROC_202.pickle', 'rb') as handle:
    metadataset_feature_selected = pickle.load(handle)
  
#nested_results is a nested dictionary with all the AUC-ROC performances for each dataset and all models
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle)    

#Remove the meta-features which are not in the meta-dataset 

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

