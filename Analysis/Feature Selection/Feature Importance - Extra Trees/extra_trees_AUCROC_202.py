import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

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

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

labels = []
for i in list(indices):
    labels.append(list(X.columns)[i])
labels[0] = "# Instances"    
labels[1] = "Mean IQR"
labels[2] = "Mean MAD"
labels[3] = "Class balance"
labels[4] = "Max M.I"
labels[7] = "Mean Feature Entropy"
labels[8] = "Class Entropy"
labels[10] = "Mean M.I"
labels[11] = "Mean Kendall corr."
labels[-4] = "Mean Canonical corr."
labels[-5] = "% of features with outliers"

# Plot the feature importances of the forest
plt.figure()
plt.title("Extra-trees classifier feature importance: AUC-ROC", fontsize = 14, weight = 'bold')
plt.bar(range(X.shape[1]), importances[indices],
       color="b", align="center", width=0.5)
plt.xticks(range(X.shape[1]), labels, rotation=45, fontsize = 8)
plt.xlim([-1, X.shape[1]])
plt.ylabel("Feature importance", fontsize = 14)
plt.show()

important_features = [index for index,value in enumerate(list(importances)) if value > 0]

selected_X = df_results.iloc[:, important_features]

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Feature Importance - Extra Trees/extra_trees_X_AUCROC_202.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)