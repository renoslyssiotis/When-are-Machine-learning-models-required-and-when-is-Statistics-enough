import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[3])+'/test/df_results.plk')

#===============================================================================
#                     FORWARD IMPORTANCE: EXTRA TREES
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
labels[2] = "Mean IQR"
labels[-1] = "Mean Canonical corr."
labels[7] = "Mean MAD"
labels[9] = "Class balance"
labels[8] = "Max M.I"
labels[6] = "Mean Feature Entropy"
labels[11] = "Class Entropy"
labels[6] = "Mean M.I"
labels[-9] = "Mean Kendall corr."
labels[-5] = "% of features with outliers"

# Plot the feature importances of the forest
plt.figure()
plt.title("Extra-trees classifier feature importance: AUC-PRC", fontsize = 14, weight = 'bold')
plt.bar(range(X.shape[1]), importances[indices],
       color="b", align="center", width=0.5)
plt.xticks(range(X.shape[1]), labels, rotation=45, fontsize = 8)
plt.xlim([-1, X.shape[1]])
plt.ylabel("Feature importance", fontsize = 14)
plt.show()

important_features = [index for index,value in enumerate(list(importances)) if value > 0]

selected_X = df_results.iloc[:, important_features]

with open('extra_trees_X_AUCPRC_202.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)