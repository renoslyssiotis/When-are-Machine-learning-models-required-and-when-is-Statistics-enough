import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Meta-learners/KNN Ranking Method/RMSE')
from KNN_ranking_k_3_RMSE import KNN_ranking
import pickle
import pandas as pd

#=====================META-FEATURE EXTRACTION==================================
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Wine quality test dataset/Actual results/wine_metafeatures_202.pickle', 'rb') as handle:
    meta_features = pickle.load(handle)

#============Load the meta-dataset and model performance of each dataset=======
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#nested_results is a nested dictionary with all the AUC-ROC performances for each dataset and all models
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_f1.pickle', 'rb') as handle:
    nested_results_f1 = pickle.load(handle)

#========================META-LEARNING: RANKING================================    
#KNN Ranking Method
top1, top2, top3 = KNN_ranking(df_results, meta_features, nested_results_f1)
print("==========================================")
print("           F1 metric         ")
print("==========================================")
print("Top 1 predicted model:      " + top1)
print("Top 2 predicted model:      " + top2)
print("Top 3 predicted model:      " + top3)

#Actual results
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Wine quality test dataset/Actual results/wine_top_3_f1.pickle', 'rb') as handle:
    actual_results = pickle.load(handle)
print("==========================================")
print("Top 1 ACTUAL model:      " + actual_results[0])
print("Top 2 ACTUAL model:      " + actual_results[1])
print("Top 3 ACTUAL model:      " + actual_results[2])

