import sys, os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[3]))
from models.meta_features import metaFeatures
from models.classifiers import classifier
from models.pre_processing import preProcessor
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

import time
start_time = time.time()

#==============================================================================
#==============================================================================
#The model was run and results are different for the best performing models
# use the report to see which top-3 best models to use
#==============================================================================
#==============================================================================


#=========================DATA PRE-PROCESSING==================================
#Import dataset and split into features and target variable
df = pd.read_csv('bands.data',
                 header = None)

del df[0]
del df[1]
del df[2]
del df[3]
del df[5]
del df[8]

for col in list(df.columns):
    df = df[df[col].notna()]

df.index = list(np.arange(0,df.shape[0]))
df.columns = list(np.arange(0,df.shape[1]))

for col in list(df.columns):
    df = df[df[col] != '?']

df.index = list(np.arange(0,df.shape[0]))
df.columns = list(np.arange(0,df.shape[1]))

def binary_class(row):
    if row == 'band':
        row = 0
    else:
        row = 1
    return row

df[33] = df[33].apply(lambda x: binary_class(x))

#One-hot encoding & Feature scaling
df_processed = preProcessor(df)
[dummy_columns, features_columns, scaled_features, X, y] = df_processed.pre_processing()
df_processed = X.join(y)

#Split dataset into the Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=====================META-FEATURE EXTRACTION==================================
df_meta_features = metaFeatures(df_processed, df)
meta_features = df_meta_features.get_meta_features()

with open('cylinder_metafeatures_202.pickle', 'wb') as handle:
    pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

#==========================MODEL FITTING=======================================
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

with open('cylinder_top_3_roc.pickle', 'wb') as handle:
    pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('cylinder_top_3_prc.pickle', 'wb') as handle:
    pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)    
with open('cylinder_top_3_f1.pickle', 'wb') as handle:
    pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)

#==============================================================================
#       Run the best model (MLP) to check its execution time
#==============================================================================
# MLP = MLPClassifier(max_iter = 1000)      
# MLP.fit(X_train, y_train)

# #Predict the test set results 
# y_pred = MLP.predict(X_test)

# #Performance: AUC-ROC
# roc_auc_MLP = roc_auc_score(y_test, y_pred)
# print('MLP completed: ROC-AUC: {}'.format(roc_auc_MLP)+'\n' + '------------------------------------------------------------------')
# print("--- %s seconds ---" % (time.time() - start_time))

