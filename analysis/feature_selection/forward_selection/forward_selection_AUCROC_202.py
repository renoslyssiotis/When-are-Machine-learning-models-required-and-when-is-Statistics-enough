import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[3])+'/test/df_results.plk')

#===============================================================================
#                     FORWARD SELECTION
#===============================================================================

X = df_results.iloc[:,:23]

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
results_index = [0, 1, 3, 4, 6, 11, 14, 16, 17, 19, 20]
selected_X = df_results.iloc[:, results_index] 

with open('forward_X_AUCROC_202.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)