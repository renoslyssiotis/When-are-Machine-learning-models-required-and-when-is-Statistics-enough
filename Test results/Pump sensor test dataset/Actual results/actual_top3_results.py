import sys
import time
import pickle
import numpy as np
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Models')
from meta_features import metaFeatures
from classifiers import classifier
from pre_processing import preProcessor
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

start_time = time.time()

#=========================DATA PRE-PROCESSING==================================
#Import dataset and split into features and target variable
df = pd.read_csv('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Datasets/Classification/Numerical/Sensor/sensor.csv')

del df['Unnamed: 0']
del df['sensor_15']
del df['timestamp']

def binary_class(row):
    if row == "NORMAL":
        row = 0
    else:
        row = 1
    return row

df['machine_status'] = df['machine_status'].apply(lambda x: binary_class(x))

for col in list(df.columns):
    df = df[df[col].notna()]

df.index = list(np.arange(0,df.shape[0]))

#One-hot encoding & Feature scaling
df_processed = preProcessor(df)
[dummy_columns, features_columns, scaled_features, X, y] = df_processed.pre_processing()
df_processed = X.join(y)

#Split dataset into the Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=====================META-FEATURE EXTRACTION==================================
df_meta_features = metaFeatures(df_processed, df)
meta_features = df_meta_features.get_meta_features()

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Pump sensor test dataset/Actual results/sensor_metafeatures_202.pickle', 'wb') as handle:
    pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


# param_grid_rforest = dict(n_estimators = [10,20,30,40,50,60,70,80,90,100], #Number of trees
#                           criterion = ['gini','entropy'])

# random_forest = RandomForestClassifier()

# grid_rforest = GridSearchCV(estimator=random_forest, 
#                             param_grid=param_grid_rforest, 
#                             cv = 3, 
#                             n_jobs=-1,
#                             error_score=0.0,
#                             iid=False)

# grid_rforest.fit(X_train, y_train)

# #Predict the test set results 
# y_pred = grid_rforest.predict(X_test)

# #Performance: AUC-ROC
# roc_auc_rforest = roc_auc_score(y_test, y_pred)
# print('Random Forest completed: ROC-AUC: {}'.format(roc_auc_rforest)+'\n' + '------------------------------------------------------------------')
# print("--- %s seconds ---" % (time.time() - start_time))

# #==========================MODEL FITTING=======================================
new_classifier = classifier(X_train, X_test, y_train, y_test)
results_roc, results_prc, results_f1 = new_classifier.evaluate_metrics()

results_roc = {k: v for k, v in sorted(results_roc.items(), key=lambda item: item[1])}
results_prc = {k: v for k, v in sorted(results_prc.items(), key=lambda item: item[1])}
results_f1 = {k: v for k, v in sorted(results_f1.items(), key=lambda item: item[1])}

print("========================================")
print("Actual best-perfmorming models")
print("========================================")

print("AUC-ROC: 1) %s , 2) %s, 3) %s " % ( list(results_roc.keys())[-1], list(results_roc.keys())[-2],list(results_roc.keys())[-3]))
print("AUC-PRC: 1) %s , 2) %s, 3) %s " % ( list(results_prc.keys())[-1], list(results_prc.keys())[-2],list(results_prc.keys())[-3]))
print("f1 metric: 1) %s , 2) %s, 3) %s " % ( list(results_f1.keys())[-1], list(results_f1.keys())[-2],list(results_f1.keys())[-3]))

results_roc = list(results_roc.keys())[::-1][:3]
results_prc = list(results_prc.keys())[::-1][:3]
results_f1 = list(results_f1.keys())[::-1][:3]

print("--- %s seconds ---" % (time.time() - start_time))
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Pump sensor test dataset/Actual results/sensor_top_3_roc.pickle', 'wb') as handle:
    pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Pump sensor test dataset/Actual results/sensor_top_3_prc.pickle', 'wb') as handle:
    pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Pump sensor test dataset/Actual results/sensor_top_3_f1.pickle', 'wb') as handle:
    pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)