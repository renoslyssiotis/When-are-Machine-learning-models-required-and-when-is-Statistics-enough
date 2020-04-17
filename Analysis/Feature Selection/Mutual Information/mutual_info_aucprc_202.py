import pandas as pd
import pickle
from sklearn.feature_selection import mutual_info_classif
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
#                     MUTUAL INFORMATION
#===============================================================================
"""
Mutual information (MI) [1] between two random variables is a non-negative value, 
which measures the dependency between the variables. It is equal to zero if and
 only if two random variables are independent, and higher values mean higher dependency.
"""
X = df_results.iloc[:,:23]

model_dict = {"auc_prc_logit": 1,
              "auc_prc_rforest": 2,
              "auc_prc_MLP": 3,
              "auc_prc_bagging": 4,
              "auc_prc_ADAboost": 5,
              "auc_prc_perceptron": 6,
              "auc_prc_QDA": 7,
              "auc_prc_linearSVC": 8,
              "auc_prc_dtree": 9,
              "auc_prc_bernoulliNB": 10,
              "auc_prc_LDA": 11,
              "auc_prc_gaussianNB": 12}

df_results = df_results.replace({"Best model PRC": model_dict})
y = df_results["Best model PRC"]

mi = mutual_info_classif(X, y, discrete_features = [0,1,3,5,16])

highest_mi_index = []
for index, value in enumerate(mi):
    if value >= 0.2:           #threshold for mutual information
        highest_mi_index.append(index)
    else:
        pass

selected_X = df_results.iloc[:, highest_mi_index] 

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Mutual Information/mutual_info_X_AUCPRC_202.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)