import sys, os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[0]))
from pysymbolic.algorithms.symbolic_metamodeling import *

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


data          = pd.read_csv("data/absenteeism.csv", delimiter=';')

feature_names = ['Transportation expense', 'Distance from Residence to Work',
                 'Service time', 'Age', 'Work load Average/day ', 'Hit target',
                 'Disciplinary failure', 'Education', 'Son', 'Social drinker',
                 'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index']

scaler        = MinMaxScaler(feature_range=(0, 1))
X             = scaler.fit_transform(data[feature_names])
Y             = ((data['Absenteeism time in hours'] > 4) * 1) 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

model         = RandomForestClassifier()

model.fit(X_train, Y_train)

print(roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]))

model_L = LogisticRegression()

model_L.fit(X_train, Y_train)

print(roc_auc_score(Y_test, model_L.predict_proba(X_test)[:, 1]))

metamodel = symbolic_metamodel(model, X_train)
metamodel.fit(num_iter=2, batch_size=X_train.shape[0], learning_rate=0.1) #10, 0.01

Y_metamodel = metamodel.evaluate(X_test)

print(roc_auc_score(Y_test, Y_metamodel))
exact = metamodel.exact_expression