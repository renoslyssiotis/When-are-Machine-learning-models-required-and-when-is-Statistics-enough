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
X = df_results.iloc[:,:12]

model_dict = {"auc_roc_logit": 1,
              "auc_roc_rforest": 2,
              "auc_roc_MLP": 3,
              "auc_roc_bagging": 4,
              "auc_roc_ADAboost": 5,
              "auc_roc_perceptron": 6,
              "auc_roc_QDA": 7,
              "auc_roc_linearSVC": 8,
              "auc_roc_dtree": 9,
              "auc_roc_bernoulliNB": 10,
              "auc_roc_LDA": 11,
              "auc_roc_gaussianNB": 12}

df_results = df_results.replace({"Best model ROC": model_dict})
y = df_results["Best model ROC"]

mi = mutual_info_classif(X, y, discrete_features = [0,1,2])

highest_mi_index = []
for index, value in enumerate(mi):
    if value >= 0.20:           #threshold for mutual information
        highest_mi_index.append(index)
    else:
        pass

selected_X = df_results.iloc[:, highest_mi_index] 

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Mutual Information/mutual_info_X_AUCROC.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)