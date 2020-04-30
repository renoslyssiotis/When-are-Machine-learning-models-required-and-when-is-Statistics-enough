import pandas as pd
import numpy as np

import sys, os
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[3]))

from models.meta_features import metaFeatures
from models.classifiers import classifier
from models.pre_processing import preProcessor

from sklearn.model_selection import train_test_split

#=========================DATA PRE-PROCESSING==================================
df = pd.read_excel('dermatology.data.xlsm',
                   names = list(np.linspace(1,35,35)), index_col = False)

df.columns = [*df.columns[:-1], 'Class']

def classify_into_types(row):
    row = int(row)
    if row == 1:
        pass
    else:
        row = 0
    return row

df['Class'] = df['Class'].apply(lambda x: classify_into_types(x))

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

# import pickle
# with open(str(p.parents[2])+'/offline/dermatology/results_roc_derma_1.pickle', 'wb') as handle:
#     pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(str(p.parents[2])+'/offline/dermatology/results_prc_derma_1.pickle', 'wb') as handle:
#     pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(str(p.parents[2])+'/offline/dermatology/results_f1_derma_1.pickle', 'wb') as handle:
#     pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(str(p.parents[2])+'/offline/dermatology/metafeatures_derma_1.pickle', 'wb') as handle:
#     pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
