import sys, os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[2]))
from retrieve_performance_gain import retrieve_best_ML_and_stats_model
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
sys.path.append(str(p.parents[0]))
from pysymbolic.algorithms.symbolic_metamodeling import *

df_results = pd.read_pickle(str(p.parents[6])+'/test/df_results.plk')
with open(str(p.parents[6])+'/test/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle) 
    
#Performance gain in AUC-ROC   
performance_gain_roc = retrieve_best_ML_and_stats_model(nested_results_roc)[2]
performance_gain_roc_list = list(performance_gain_roc.values())

#Create the dataframe, where the features are the features with p-value < 0.05
feature_names =     ['Number of Features', 'Proportion of Numerical Features',
                    'Number of Dummy Variables after pre-processing',
                      'Proportion of Less Frequent Class',
                    'Mean skewness', 'Mean Median Absolute Deviation',
                    'Mean Canonical Correlations','Normalized Class Entropy', 
                    'Max Mutual Information', 'Mean Mutual Information']

# feature_names =     ['Number of Features', 'Proportion of Less Frequent Class',
#                     'Mean skewness']

scaler        =     MinMaxScaler(feature_range=(0,1))
X             =     scaler.fit_transform(df_results[feature_names])
X             =     X[35:55]
Y             =     pd.Series(performance_gain_roc_list)
Y             =     Y[35:55]

def binary(row):
    if row >0:
        row = 1
    else:
        row = 0
    return row

# Treat as a binary classification problem: 1 if gain > 0; 0 if gain < 0
Y = Y.apply(lambda x: binary(x))

#Use Random forest classifier 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.33, 
                                                    random_state=42)

model         = RandomForestClassifier(n_estimators = 100)

model.fit(X_train, Y_train)
print(average_precision_score(Y_test, model.predict_proba(X_test)[:, 1]))

# # Symbolic metamodelling
metamodel = symbolic_metamodel(model, X_train)
metamodel.fit(num_iter = 2, batch_size = X_train.shape[0], learning_rate = 0.1)

# Evaluate the performance of the meta-model
Y_metamodel = metamodel.evaluate(X_test)
print(average_precision_score(Y_test, Y_metamodel))

# Obtain the exact and approximate expressions of the meta-model
exact_Expression = metamodel.exact_expression
approx_Expression = metamodel.approx_expression






