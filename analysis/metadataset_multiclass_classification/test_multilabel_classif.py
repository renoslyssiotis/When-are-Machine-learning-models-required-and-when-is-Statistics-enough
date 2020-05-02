import os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

with open(str(p.parents[2])+'/test/wine_quality/actual/wine_metafeatures_202.pickle', 'rb') as handle:
    wine_meta_features = pickle.load(handle)
    
with open(str(p.parents[2])+'/test/pump_sensor/actual/sensor_metafeatures_202.pickle', 'rb') as handle:
    pump_meta_features = pickle.load(handle)
    
with open(str(p.parents[2])+'/test/cylindrical_bands/actual/cylinder_metafeatures_202.pickle', 'rb') as handle:
    cylinder_meta_features = pickle.load(handle)
    
#Pre-processing
df_results = pd.read_pickle(str(p.parents[2])+'/test/df_results.plk')

#Standardise the features (i.e. metafeatures) of the dataset (i.e. metadataset)
X = df_results.iloc[:,:23]
scaler = StandardScaler()
X = pd.DataFrame(data = scaler.fit_transform(X),
                 columns = list(df_results.columns[:23]))

#Transform problem into a multi-label classification problem
models = {'auc_f1_logit': 1,
          'auc_f1_rforest': 2,
          'auc_f1_MLP': 3,
          'auc_f1_bagging': 4,
          'auc_f1_perceptron': 5,
          'auc_f1_ADAboost': 6,
          'auc_f1_dtree': 7,
          'auc_f1_bernoulliNB': 8,
          'auc_f1_LDA': 9,
          'auc_f1_QDA': 10,
          'auc_f1_linearSVC': 11,
          'auc_f1_gaussianNB': 12}

df_results = df_results.replace({'Best model f1': models})
y = df_results['Best model PRC']

#Split dataset into the Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#==========================Model Training and predictions =====================
param_grid_rforest = dict(n_estimators = [10,20,30,40,50,60,70,80,90,100], #Number of trees
                          criterion = ['gini','entropy'])

wine_pred = 12*[0]
pump_pred = 12*[0]
cylinder_pred = 12*[0]

for i in tqdm(range(10)):
    random_forest = RandomForestClassifier()        
    grid_rforest = GridSearchCV(estimator=random_forest, 
                                param_grid=param_grid_rforest, 
                                cv = 3, 
                                n_jobs=-1,
                                error_score=0.0,
                                iid=False)
    
    grid_rforest.fit(X_train, y_train)
      
    wine_predictions = grid_rforest.predict_proba(pd.DataFrame(wine_meta_features.values()).T)
    wine_pred = [sum(x) for x in zip(wine_pred, wine_predictions.tolist()[0])]
    
    pump_predictions = grid_rforest.predict_proba(pd.DataFrame(pump_meta_features.values()).T)
    pump_pred = [sum(x) for x in zip(pump_pred, pump_predictions.tolist()[0])]
    
    cylinder_predictions = grid_rforest.predict_proba(pd.DataFrame(cylinder_meta_features.values()).T)
    cylinder_pred = [sum(x) for x in zip(cylinder_pred, cylinder_predictions.tolist()[0])]        

wine_pred = [x/10 for x in wine_pred]
pump_pred = [x/10 for x in pump_pred]
cylinder_pred = [x/10 for x in cylinder_pred]


















