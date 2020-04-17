import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Models')
from meta_features import metaFeatures
from classifiers import classifier
from pre_processing import preProcessor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

#=========================DATA PRE-PROCESSING==================================
#Import dataset and split into features and target variable
df1 = pd.read_csv('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Datasets/Classification/Numerical/Banana/class0_banana.txt', 
                 sep="\t", header = None)
df2 = pd.read_csv('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Datasets/Classification/Numerical/Banana/class1_banana.txt', 
                 sep="\t", header = None)
df = pd.concat([df1, df2], axis=0)

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

#Save results for offline use as .pkl:
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Banana/results_roc_banana_1.pickle', 'wb') as handle:
    pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Banana/results_prc_banana_1.pickle', 'wb') as handle:
    pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Banana/results_f1_banana_1.pickle', 'wb') as handle:
    pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Banana/metafeatures_banana_1.pickle', 'wb') as handle:
    pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
