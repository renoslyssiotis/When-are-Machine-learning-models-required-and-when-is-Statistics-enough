import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Performance gain vs. Meta-features')
import pandas as pd
import pickle

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
#                     VARIANCE THRESHOLD:  Zero-variance
#===============================================================================
"""
Feature selector that removes all zero-variance features. This feature selection
algorithm looks only at the features (X), not the desired outputs (y)
"""
meta_features = df_results.iloc[:,:23]
meta_features_variance = list(meta_features.var(axis=0))

index_of_high_variance_features_1 = []
threshold_1 = 0        #can change threshold to include more/less meta-features
for index, value in enumerate(meta_features_variance):
    if value != threshold_1:
        index_of_high_variance_features_1.append(index)
    else:
        pass

#Only 'number of classes' removed
selected_X1 = df_results.iloc[:, index_of_high_variance_features_1] 

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Variance Threshold/var_threshold_zeroVar_X_202.pickle', 'wb') as handle:
    pickle.dump(selected_X1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#===============================================================================
#                     VARIANCE THRESHOLD:  Low-variance
#                   Threshold = 0.1
#===============================================================================
"""
Feature selector that removes all low-variance features. This feature selection
algorithm looks only at the features (X), not the desired outputs (y)
"""
index_of_high_variance_features_2 = []
threshold_2 = 0.1        #can change threshold to include more/less meta-features
for index, value in enumerate(meta_features_variance):
    if value > threshold_2:
        index_of_high_variance_features_2.append(index)
    else:
        pass

#Only 'number of classes' removed
selected_X2 = df_results.iloc[:, index_of_high_variance_features_2] 

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Variance Threshold/var_threshold_lowVar_X_202.pickle', 'wb') as handle:
    pickle.dump(selected_X2, handle, protocol=pickle.HIGHEST_PROTOCOL)