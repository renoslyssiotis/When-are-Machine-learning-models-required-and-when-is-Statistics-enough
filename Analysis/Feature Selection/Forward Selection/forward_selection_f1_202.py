import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
#                     FORWARD SELECTION
#===============================================================================

X = df_results.iloc[:,:23]

model_dict = {"auc_f1_logit": 1,
              "auc_f1_rforest": 2,
              "auc_f1_MLP": 3,
              "auc_f1_bagging": 4,
              "auc_f1_ADAboost": 5,
              "auc_f1_perceptron": 6,
              "auc_f1_QDA": 7,
              "auc_f1_linearSVC": 8,
              "auc_f1_dtree": 9,
              "auc_f1_bernoulliNB": 10,
              "auc_f1_LDA": 11,
              "auc_f1_gaussianNB": 12}

df_results = df_results.replace({"Best model f1": model_dict})
y = df_results["Best model f1"]

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Build step forward feature selection
selected_metafeatures = []

for i in range(10):
    sfs1 = sfs(clf,
               k_features=5,
               forward=True,
               floating=False,
               verbose=2,
               scoring='accuracy',
               cv=2)
    
    # Perform SFFS
    sfs1 = sfs1.fit(X_train, y_train)

    # Which features?
    feat_cols = list(sfs1.k_feature_idx_)
    selected_metafeatures.append(feat_cols)    

from collections import Counter
flattened = [val for sublist in selected_metafeatures for val in sublist]
results = Counter(flattened)

#Find the 5 most common indeces:
results_index = [0, 1, 5, 6, 7, 11, 13]
selected_X = df_results.iloc[:, results_index] 

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Forward Selection/forward_X_f1_202.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)