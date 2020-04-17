import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
#                     FORWARD IMPORTANCE: EXTRA TREES
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
param_grid_rforest = dict(n_estimators = [10,20,30,40,50,60,70,80,90,100],
                          criterion = ['gini', 'entropy'])

random_forest= RandomForestClassifier()

grid_rforest = GridSearchCV(estimator = random_forest,
                           param_grid = param_grid_rforest,
                           cv = 3,
                           n_jobs = -1,
                           error_score = 0.0,
                           iid = False)

grid_rforest.fit(X_train, y_train)

y_pred = grid_rforest.predict(X_test)

print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))





roc_auc_rforest = roc_auc_score(y_test, y_pred)
prc_auc_rforest = average_precision_score(y_test, y_pred, average = 'weighted')
f1_auc_rforest = f1_score(y_test, y_pred, average = 'weighted')