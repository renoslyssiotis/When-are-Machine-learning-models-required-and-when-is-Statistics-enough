import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Models')
from meta_features import metaFeatures
from classifiers import classifier
from pre_processing import preProcessor
import pandas as pd
from sklearn.model_selection import train_test_split

#=========================DATA PRE-PROCESSING==================================
#Import dataset and split into features and target variable
df = pd.read_csv('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Datasets/Classification/Numerical/Parkinsons/parkinsons.data',
                 sep = '\t')

#One-hot encoding & Feature scaling
df_processed = preProcessor(df)
[dummy_columns, features_columns, scaled_features, X, y] = df_processed.pre_processing()
df_processed = X.join(y)

#Split dataset into the Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=====================META-FEATURE EXTRACTION==================================
df_meta_features = metaFeatures(df_processed, df)
meta_features = df_meta_features.get_meta_features()

#==========================MODEL FITTING=======================================
new_classifier = classifier(X_train, X_test, y_train, y_test)

results_roc, results_prc, results_f1 = new_classifier.evaluate_metrics()

import pickle
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Parkinsons/results_roc_parkinsons_0.pickle', 'wb') as handle:
    pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Parkinsons/results_prc_parkinsons_0.pickle', 'wb') as handle:
    pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Parkinsons/results_f1_parkinsons_0.pickle', 'wb') as handle:
    pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Parkinsons/metafeatures_parkinsons_0.pickle', 'wb') as handle:
    pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
