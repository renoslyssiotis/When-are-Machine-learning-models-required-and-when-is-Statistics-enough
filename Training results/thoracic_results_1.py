from scipy.io import arff
import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Models')
from meta_features import metaFeatures
from classifiers import classifier
from pre_processing import preProcessor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#=========================DATA PRE-PROCESSING==================================
#Import dataset
data = arff.loadarff('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Datasets/Classification/Categorical/Thoracic Surgery/ThoraricSurgery.arff')
df = pd.DataFrame(data[0])

def bytes2string(row):
    try:
        row = row.decode("utf-8")
    except:
        pass
    return row

def binary_class(row):
    if row == "F":
        row = 0
    else:
        row = 1
    return row

df.iloc[:,0] = df.iloc[:,0].apply(lambda x: bytes2string(x))
df.iloc[:,1] = df.iloc[:,1].apply(lambda x: bytes2string(x))
df.iloc[:,2] = df.iloc[:,2].apply(lambda x: bytes2string(x))
df.iloc[:,3] = df.iloc[:,3].apply(lambda x: bytes2string(x))
df.iloc[:,4] = df.iloc[:,4].apply(lambda x: bytes2string(x))
df.iloc[:,5] = df.iloc[:,5].apply(lambda x: bytes2string(x))
df.iloc[:,6] = df.iloc[:,6].apply(lambda x: bytes2string(x))
df.iloc[:,7] = df.iloc[:,7].apply(lambda x: bytes2string(x))
df.iloc[:,8] = df.iloc[:,8].apply(lambda x: bytes2string(x))
df.iloc[:,9] = df.iloc[:,9].apply(lambda x: bytes2string(x))
df.iloc[:,10] = df.iloc[:,10].apply(lambda x: bytes2string(x))
df.iloc[:,11] = df.iloc[:,11].apply(lambda x: bytes2string(x))
df.iloc[:,12] = df.iloc[:,12].apply(lambda x: bytes2string(x))
df.iloc[:,13] = df.iloc[:,13].apply(lambda x: bytes2string(x))
df.iloc[:,14] = df.iloc[:,14].apply(lambda x: bytes2string(x))
df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: bytes2string(x))
df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: binary_class(x))

#Randomly select 20% of rows
df_ = df.sample(frac=0.2)
df_.sort_index(inplace=True)
df_.index = list(np.arange(0, len(df_)))

#One-hot encoding & Feature scaling
df_processed = preProcessor(df_)
[dummy_columns, features_columns, scaled_features, X, y] = df_processed.pre_processing()
df_processed = X.join(y)

#Size of original tumor: PRE14 (0C11, ..., 0C13)
#OC11 (smallest) to OC14 (largest) (OC11,OC12,OC13,OC14)
df_processed.iloc[:,7] = df_processed.iloc[:,7]*1    #OC14
df_processed.iloc[:,6] = df_processed.iloc[:,6]*0.75 #OC13
df_processed.iloc[:,5] = df_processed.iloc[:,6]*0.5 #OC12

#Split dataset into the Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=====================META-FEATURE EXTRACTION==================================
df_meta_features = metaFeatures(df_processed, df_)
meta_features = df_meta_features.get_meta_features()

#==========================MODEL FITTING=======================================
new_classifier = classifier(X_train, X_test, y_train, y_test)
results_roc, results_prc, results_f1 = new_classifier.evaluate_metrics()

#Save results for offline use as .pkl:
import pickle
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Thoracic/results_roc_thoracic_1.pickle', 'wb') as handle:
    pickle.dump(results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Thoracic/results_prc_thoracic_1.pickle', 'wb') as handle:
    pickle.dump(results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Thoracic/results_f1_thoracic_1.pickle', 'wb') as handle:
    pickle.dump(results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Training results/_Offline results/Thoracic/metafeatures_thoracic_1.pickle', 'wb') as handle:
    pickle.dump(meta_features, handle, protocol=pickle.HIGHEST_PROTOCOL)