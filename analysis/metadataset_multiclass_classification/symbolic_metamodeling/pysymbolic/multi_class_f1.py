import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Models')
from classifiers_multiclass import classifier_multiclass
import symbolic_metamodeling
from symbolic_metamodeling import *
from special_functions import MeijerG
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, accuracy_score

#===============================================================================
#       Multi-class classification of the meta-dataset: f1 metric
#===============================================================================

data = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
feature_names = ['Number of Instances', 'Number of Features',
                'Proportion of Numerical Features',
                'Number of Dummy Variables after pre-processing',
                'Proportion of Less Frequent Class', 'Number of Classes', 'Mean mean',
                'Mean std', 'Mean Coefficient of Variation', 'Mean skewness',
                'Mean kurtosis', 'Mean Pearson Correlation', 'Mean Kendall Correlation',
                'Mean Spearman Correlation', 'Mean Median Absolute Deviation',
                'Mean Inter-Quartile Range',
                'Proportion of Scaled Features with Outliers',
                'Mean Canonical Correlations', 'Mean Normalised Feature Entropy',
                'Normalized Class Entropy', 'Signal to Noise ratio',
                'Max Mutual Information', 'Mean Mutual Information']

scaler = MinMaxScaler(feature_range = (0,1))
X = scaler.fit_transform(data[feature_names])

model_dict = {"auc_f1_logit": 0,
              "auc_f1_rforest": 1,
              "auc_f1_MLP": 2,
              "auc_f1_bagging": 3,
              "auc_f1_ADAboost": 4,
              "auc_f1_perceptron": 5,
              "auc_f1_QDA": 6,
              "auc_f1_linearSVC": 7,
              "auc_f1_dtree": 8,
              "auc_f1_bernoulliNB": 9,
              "auc_f1_LDA": 10,
              "auc_f1_gaussianNB": 11}

data = data.replace({"Best model f1": model_dict})
y = data["Best model f1"]

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#Use random forest and F1 metric
model = RandomForestClassifier()
model.fit(X_train, y_train)

# F1 performance of the fitted random forrest model on the test data
f1_rforest = f1_score(y_test, model.predict(X_test), average='weighted')

#===============================================================================
#                           Symbolic meta-modelling
#===============================================================================
metamodel = symbolic_metamodel(model, X_train)
metamodel.fit(num_iter=10, batch_size=X_train.shape[0], learning_rate=.01)

Y_metamodel = metamodel.evaluate(X_test)
f1_rforest_metamodel = f1_score(y_test, y_metamodel, average='weighted')

print(metamodel.exact_expression)


# # Make predictions on the three test datasets
# with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Wine quality test dataset/Actual results/wine_metafeatures_202.pickle', 'rb') as handle:
#     wine_meta_features = pickle.load(handle)
    
# with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Pump sensor test dataset/Actual results/sensor_metafeatures_202.pickle', 'rb') as handle:
#     pump_meta_features = pickle.load(handle)
    
# with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Cylinder Bands test dataset/Actual results/cylinder_metafeatures_202.pickle', 'rb') as handle:
#     bands_meta_features = pickle.load(handle)

# test_pd = pd.DataFrame(data = [wine_meta_features.values(), pump_meta_features.values(), bands_meta_features.values()],
#                        index = ['Wine','Pump','Cylindrical bands'],
#                        columns = wine_meta_features.keys())

# y_pred_test = grid_rforest.predict(test_pd)
# y_pred_proba_test = grid_rforest.predict_proba(test_pd)

# #Wine dataset
# y_pred_proba_wine = np.argsort(list(y_pred_proba_test[0]))[-3:]

# for model, index in model_dict.items():  
#     if index == y_pred_proba_wine[-1]:
#         top1_wine = model
#     elif index == y_pred_proba_wine[-2]:
#         top2_wine = model
#     elif index == y_pred_proba_wine[-3]:
#         top3_wine = model
    
# #Pump dataset
# y_pred_proba_pump = np.argsort(list(y_pred_proba_test[1]))[-3:]

# for model, index in model_dict.items():  
#     if index == y_pred_proba_pump[-1]:
#         top1_pump = model
#     elif index == y_pred_proba_pump[-2]:
#         top2_pump = model
#     elif index == y_pred_proba_pump[-3]:
#         top3_pump = model
    
# #Pump dataset
# y_pred_proba_bands = np.argsort(list(y_pred_proba_test[2]))[-3:]

# for model, index in model_dict.items():  
#     if index == y_pred_proba_bands[-1]:
#         top1_bands = model
#     elif index == y_pred_proba_bands[-2]:
#         top2_bands = model
#     elif index == y_pred_proba_bands[-3]:
#         top3_bands = model
    







