import sys, os
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[3]))

from models.meta_features import metaFeatures
from models.classifiers import classifier
from models.pre_processing import preProcessor

import pandas as pd
from sklearn.model_selection import train_test_split

#=========================DATA PRE-PROCESSING==================================
#Import dataset
df = pd.read_excel('2014 _and_2015_CSM_dataset.xlsx')
df.drop(['Movie', 'Year'], axis = 1, inplace = True)

def classify(row):
    if row >= 3038193.449:
        row = 0
    else:
        row = 1
    return row

df['Aggregate Followers'] = df['Aggregate Followers'].apply(lambda x: classify(x))

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

#Save results for offline use as .pkl:
# import pickle
# with open(str(p.parents[2])+'/offline/movies/results_roc_movies_0.pickle', 'wb') as handle:
#     pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(str(p.parents[2])+'/offline/movies/results_prc_movies_0.pickle', 'wb') as handle:
#     pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(str(p.parents[2])+'/offline/movies/results_f1_movies_0.pickle', 'wb') as handle:
#     pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(str(p.parents[2])+'/offline/movies/metafeatures_movies_0.pickle', 'wb') as handle:
#     pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


