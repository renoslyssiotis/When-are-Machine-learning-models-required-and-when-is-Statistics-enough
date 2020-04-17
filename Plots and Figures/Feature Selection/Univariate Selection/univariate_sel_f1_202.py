# Feature Selection with Univariate Statistical Tests
import pickle
import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
#                     ANOVA F-VALUE
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

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Univariate Selection/ANOVA_X_f1_202.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)