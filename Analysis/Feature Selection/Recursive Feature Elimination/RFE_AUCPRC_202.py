import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
#                     RECURSIVE FEATURE ELIMINATION: AUC-ROC
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

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Recursive Feature Elimination/RFE_X_AUCPRC_202.pickle', 'wb') as handle:
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