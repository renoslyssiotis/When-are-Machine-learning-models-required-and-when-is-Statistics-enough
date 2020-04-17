import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Models')
from meta_features import metaFeatures
from classifiers import classifier
from pre_processing import preProcessor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#=========================DATA PRE-PROCESSING==================================
#Import dataset and split into features and target variable
df = pd.read_csv('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Datasets/Classification/Numerical/Ecoli/ecoli.data',
                 header = None, sep = '\t')
df.drop(df.columns[0], axis=1, inplace = True)

def binary_class(row):
    if row in ['cp', 'omL', 'imL', 'imS']:
        row = 0
    else:
        row = 1
    return row
df[8] =df[8].apply(lambda x: binary_class(x))

#Randomly select 40% of rows
df_ = df.sample(frac=0.4)
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
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Ecoli/results_roc_ecoli_2.pickle', 'wb') as handle:
    pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Ecoli/results_prc_ecoli_2.pickle', 'wb') as handle:
    pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Ecoli/results_f1_ecoli_2.pickle', 'wb') as handle:
    pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Ecoli/metafeatures_ecoli_2.pickle', 'wb') as handle:
    pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
