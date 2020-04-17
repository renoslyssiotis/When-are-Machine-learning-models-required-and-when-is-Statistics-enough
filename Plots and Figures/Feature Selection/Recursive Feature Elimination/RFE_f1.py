import pickle
import pandas as pd
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
#                     RECURSIVE FEATURE ELIMINATION: AUC-ROC
#===============================================================================

X = df_results.iloc[:,:12]

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
              "auc_f1_gaussianNB": 10}

df_results = df_results.replace({"Best model f1": model_dict})
y = df_results["Best model f1"]

cv = StratifiedKFold(2)
visualizer = RFECV(RandomForestClassifier(n_estimators=10),
                   cv=2, 
                   scoring='accuracy')

visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()

print("Optimal number of features : %d" % visualizer.n_features_)
print(visualizer.ranking_)
print(visualizer.estimator_.feature_importances_)

index_list =[]
for index, value in enumerate(visualizer.ranking_):
    if value == 1:
        index_list.append(index)
    else:
        pass

selected_X = df_results.iloc[:, index_list]

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Recursive Feature Elimination/RFE_X_f1.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)