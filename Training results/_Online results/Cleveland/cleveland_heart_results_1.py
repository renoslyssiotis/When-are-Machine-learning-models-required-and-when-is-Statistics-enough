import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Models')
from meta_features import metaFeatures
from classifiers import classifier
from pre_processing import preProcessor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#=========================DATA PRE-PROCESSING==================================
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                   names = list(range(1,15)), index_col = False) 
def func(row):
    if row == 0:
        pass
    else:
        row = 1
    return row
df[14] = df[14].apply(func)

#Randomly select 20% of rows
df_ = df.sample(frac=0.2)
df_.sort_index(inplace=True)
df_.index = list(np.arange(0, len(df_)))

#One-hot encoding & Feature scaling
df_processed = preProcessor(df_)
[dummy_columns, features_columns, scaled_features, X, y] = df_processed.pre_processing()
df_processed = X.join(y)

#Split dataset into the Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=====================META-FEATURE EXTRACTION==================================
df_meta_features = metaFeatures(df_processed, df_)
meta_features = df_meta_features.get_meta_features()

#==========================MODEL FITTING=======================================
new_classifier = classifier(X_train, X_test, y_train, y_test)
results_roc, results_prc, results_f1 = new_classifier.evaluate_metrics()    

import pickle
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Cleveland/results_roc_cleveland_1.pickle', 'wb') as handle:
    pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Cleveland/results_prc_cleveland_1.pickle', 'wb') as handle:
    pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Cleveland/results_f1_cleveland_1.pickle', 'wb') as handle:
    pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Cleveland/metafeatures_cleveland_1.pickle', 'wb') as handle:
    pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
