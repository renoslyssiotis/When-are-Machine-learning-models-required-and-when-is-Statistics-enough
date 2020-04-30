import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[3])+'/test/df_results.plk')

#===============================================================================
#                     RECURSIVE FEATURE ELIMINATION: f1
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

cv = StratifiedKFold(2)
visualizer = RFECV(RandomForestClassifier(n_estimators=10),
                   cv=cv, 
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

with open('RFE_X_f1_202.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#Imbalaned dataset 
labels = "Logit RForest MLP Bagging ADABoost Perceptron QDA LinearSVC DTree BernoulliNB LDA GaussianNB".split(" ")
sizes = [df_results["Best model ROC"].value_counts()[model] for model in list(np.arange(1,13))]
explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
        shadow=True, startangle=45)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Best-performing models for the training datasets: AUC-ROC", bbox={'facecolor':'0.8', 'pad':5})
plt.show()