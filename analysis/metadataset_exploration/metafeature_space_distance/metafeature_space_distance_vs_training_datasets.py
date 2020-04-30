import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[3])+'/test/df_results.plk')

'''
Use the customised set of meta-features for the AUC-ROC metric
1. Number of instances, 2. Mean std., 3. Mean skewness, 4. Mean pearson
5. Feature entropy, 6. Class entropy
'''   

#Wine meta-features
with open(str(p.parents[3])+'/test/wine_quality/actual/wine_metafeatures_202.pickle', 'rb') as handle:
    wine_meta_features = pickle.load(handle)
    
wine_meta_features = [list(wine_meta_features.values())[0],
                      list(wine_meta_features.values())[7],
                      list(wine_meta_features.values())[9],
                      list(wine_meta_features.values())[11],
                      list(wine_meta_features.values())[18],
                      list(wine_meta_features.values())[19]]  

#Pump sensor meta-features
with open(str(p.parents[3])+'/test/pump_sensor/actual/sensor_metafeatures_202.pickle', 'rb') as handle:
    pump_meta_features = pickle.load(handle)
    
pump_meta_features = [list(pump_meta_features.values())[0],
                      list(pump_meta_features.values())[7],
                      list(pump_meta_features.values())[9],
                      list(pump_meta_features.values())[11],
                      list(pump_meta_features.values())[18],
                      list(pump_meta_features.values())[19]]  

#Cylindrical bands meta-features
with open(str(p.parents[3])+'/test/cylindrical_bands/actual/cylinder_metafeatures_202.pickle', 'rb') as handle:
    cylindrical_meta_features = pickle.load(handle)
    
cylindrical_meta_features = [list(cylindrical_meta_features.values())[0],
                              list(cylindrical_meta_features.values())[7],
                              list(cylindrical_meta_features.values())[9],
                              list(cylindrical_meta_features.values())[11],
                              list(cylindrical_meta_features.values())[18],
                              list(cylindrical_meta_features.values())[19]]  

with open(str(p.parents[3])+'/analysis/feature_selection/customised_set/customised_X_AUCROC_202.pickle', 'rb') as handle:
    metadataset_feature_selected = pickle.load(handle) 
    
metadataset_feature_selected = metadataset_feature_selected.sample(frac=1)

#===============================================================================
# Find how the distance of the closest training to the test dataset varies
# with the number of training datasets.
#===============================================================================
def minimumDistance(metadataset_feature_selected, test_meta_features):

    min_distances = []
    
    for dataset in list(metadataset_feature_selected.index):
        
        list_of_metafeatures = metadataset_feature_selected.loc[:dataset,:].values.tolist()
        
        distances = []
        
        for i in range(len(list_of_metafeatures)):
        
            summation = 0
            
            for j in range(6):
            
                distance = (list_of_metafeatures[i][j] - test_meta_features[j])**2
                summation += distance
    
            distances.append(np.sqrt(summation/len(test_meta_features)))   
    
        min_distances.append(min(distances))   
        
    return min_distances
#===============================================================================
    
wine_min_distances = minimumDistance(metadataset_feature_selected, wine_meta_features)
pump_min_distances = minimumDistance(metadataset_feature_selected, pump_meta_features)
cylindrical_min_distances = minimumDistance(metadataset_feature_selected, cylindrical_meta_features)

training_datasets = list(range(1,203))
#===============================================================================

fig = plt.figure(dpi = 1200)
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

host.set_xlim(0, 202)
host.set_ylim(0, 2000) # wine
par1.set_ylim(0, 50000) # pump
par2.set_ylim(0, 500)   # cylindrical

host.set_xlabel("Number of training datasets", fontsize = 14)
host.set_ylabel("Distance to Wine quality dataset", fontsize = 14)
par1.set_ylabel("Distance to Pump sensor dataset", fontsize = 14)
par2.set_ylabel("Distance to Cylindrical bands dataset", fontsize = 14)

p1, = host.plot(training_datasets, wine_min_distances, color='blue',label="Wine quality")
p2, = par1.plot(training_datasets, pump_min_distances, color='darkorange', label="Pump sensor")
p3, = par2.plot(training_datasets, cylindrical_min_distances, color='darkgreen', label="Cylindrical bands")

lns = [p1, p2, p3]
host.legend(handles=lns, loc='upper right', fontsize = 11)

par2.spines['right'].set_position(('outward', 70))      
par2.xaxis.set_ticks([0,50,100,150,200], [0,50,100,150,200])

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())
