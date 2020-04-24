import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Models')
from classifiers_multiclass import classifier_multiclass
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, accuracy_score

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
#       Multi-class classification of the meta-dataset: AUC-ROC
#===============================================================================

X = df_results.iloc[:,:23]

model_dict = {"f1_logit": 0,
              "f1_rforest": 1,
              "f1_MLP": 2,
              "f1_bagging": 3,
              "f1_ADAboost": 4,
              "f1_perceptron": 5,
              "f1_QDA": 6,
              "f1_linearSVC": 7,
              "f1_dtree": 8,
              "f1_bernoulliNB": 9,
              "f1_LDA": 10,
              "f1_gaussianNB": 11}

df_results = df_results.replace({"Best model f1": model_dict})
y = df_results["Best model f1"]

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# new_classifier = classifier_multiclass(X_train, X_test, y_train, y_test)
# results_accuracy, results_f1 = new_classifier.evaluate_metrics()

#Use random forest and F1 metric

param_grid_rforest = dict(n_estimators = [10,20,30,40,50,60,70,80,90,100], #Number of trees
                          criterion = ['gini','entropy'])

random_forest = RandomForestClassifier()

grid_rforest = GridSearchCV(estimator=random_forest, 
                            param_grid=param_grid_rforest, 
                            cv = 3, 
                            n_jobs=-1,
                            error_score=0.0,
                            iid=False)

grid_rforest.fit(X_train, y_train)

#Predict the test set results 
y_pred = grid_rforest.predict(X_test)
y_pred_proba = grid_rforest.predict_proba(X_test)

#Performance: F1 Metric
f1_rforest = f1_score(y_test, y_pred, average='weighted')

# Make predictions on the three test datasets
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Wine quality test dataset/Actual results/wine_metafeatures_202.pickle', 'rb') as handle:
    wine_meta_features = pickle.load(handle)
    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Pump sensor test dataset/Actual results/sensor_metafeatures_202.pickle', 'rb') as handle:
    pump_meta_features = pickle.load(handle)
    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Cylinder Bands test dataset/Actual results/cylinder_metafeatures_202.pickle', 'rb') as handle:
    bands_meta_features = pickle.load(handle)

test_pd = pd.DataFrame(data = [wine_meta_features.values(), pump_meta_features.values(), bands_meta_features.values()],
                       index = ['Wine','Pump','Cylindrical bands'],
                       columns = wine_meta_features.keys())

y_pred_test = grid_rforest.predict(test_pd)
y_pred_proba_test = grid_rforest.predict_proba(test_pd)

#Wine dataset
y_pred_proba_wine = np.argsort(list(y_pred_proba_test[0]))[-3:]

for model, index in model_dict.items():  
    if index == y_pred_proba_wine[-1]:
        top1_wine = model
    elif index == y_pred_proba_wine[-2]:
        top2_wine = model
    elif index == y_pred_proba_wine[-3]:
        top3_wine = model
    
#Pump dataset
y_pred_proba_pump = np.argsort(list(y_pred_proba_test[1]))[-3:]

for model, index in model_dict.items():  s
    if index == y_pred_proba_pump[-1]:
        top1_pump = model
    elif index == y_pred_proba_pump[-2]:
        top2_pump = model
    elif index == y_pred_proba_pump[-3]:
        top3_pump = model
    
#Pump dataset
y_pred_proba_bands = np.argsort(list(y_pred_proba_test[2]))[-3:]

for model, index in model_dict.items():  
    if index == y_pred_proba_bands[-1]:
        top1_bands = model
    elif index == y_pred_proba_bands[-2]:
        top2_bands = model
    elif index == y_pred_proba_bands[-3]:
        top3_bands = model
    







