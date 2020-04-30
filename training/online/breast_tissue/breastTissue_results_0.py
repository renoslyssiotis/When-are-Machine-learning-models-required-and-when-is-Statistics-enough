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
df = pd.read_excel('BreastTissue.xlsx')
df = df.drop(['Case #'], axis = 1)
df = df[['I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP', 'DR', 'P', 'Class']]

def rename_class(row):
    if row == "car":
        row = 0
    else:
        row = 1
    return row

df['Class'] = df['Class'].apply(lambda x: rename_class(x))

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

# import pickle
# with open(str(p.parents[2])+'/offline/breast_tissue/results_roc_breastTissue_0.pickle', 'wb') as handle:
#     pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(str(p.parents[2])+'/offline/breast_tissue/results_prc_breastTissue_0.pickle', 'wb') as handle:
#     pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(str(p.parents[2])+'/offline/breast_tissue/results_f1_breastTissue_0.pickle', 'wb') as handle:
#     pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(str(p.parents[2])+'/offline/breast_tissue/metafeatures_breastTissue_0.pickle', 'wb') as handle:
#     pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


