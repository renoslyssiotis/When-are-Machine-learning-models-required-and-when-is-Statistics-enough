import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[3])+'/test/df_results.plk')

#===============================================================================
#                     ANOVA F-VALUE
#===============================================================================

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

# feature extraction
test = SelectKBest(score_func=f_classif, k=6)
fit = test.fit(X, y)
# summarize scores
set_printoptions(precision=2)
scores = fit.scores_
print(scores)

best_features = [index for index, value in enumerate(scores) if value > 1.2]
print(best_features)

selected_X = df_results.iloc[:, best_features]

# with open('ANOVA_X_AUCPRC_202.pickle', 'wb') as handle:
#     pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)