import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys, os
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[2]))

#=====================META-FEATURE EXTRACTION==================================
df_results = pd.read_pickle(str(p.parents[2])+'/df_results.plk')
training_metafeatures = df_results.iloc[:, :-6]

with open(str(p.parents[1])+'/actual/sensor_metafeatures_202.pickle', 'rb') as handle:
    meta_features = pickle.load(handle)
    
#Add the test meta-features at the end of the dataframe to perform standardisation
training_metafeatures.loc['Test dataset'] = list(meta_features.values())

#Standardise the meta-features (incl. those of test dataset) and add to dataframe
training_metafeatures = StandardScaler().fit(training_metafeatures.to_numpy()).transform(training_metafeatures.to_numpy())
all_meta_features = pd.DataFrame(training_metafeatures)

#STANDARDISED general and statistical meta-features of TEST dataset
general_metafeatures = list(all_meta_features.iloc[-1][:6])
stat_and_info_metafeatures = list(all_meta_features.iloc[-1][6:])

#Load the cluster centres of the 13 clusters:
with open(str(p.parents[3])+'/analysis/dataset_clustering/two_level_clustering/general_clustering_standardised/cluster_centres.pickle', 'rb') as handle:
    cluster_centres = pickle.load(handle)

#==============================================================================
#                       FIRST LEVEL CLUSTERING    
#==============================================================================
#Find the cluster at which the test dataset is allocated to:
distances = []

for key in cluster_centres.keys():
    distance = 0
    for j in range(5): #Since there are 5 general meta-features (excl. # of classes (2), which is common to all)
        distance += abs( cluster_centres[key][j] - general_metafeatures[j])
    distances.append(distance)

cluster_allocation = distances.index(min(distances))    #---> CLUSTER 11 
    
#==============================================================================
#                       SECOND LEVEL CLUSTERING    
#==============================================================================  
with open(str(p.parents[3])+'/analysis/dataset_clustering/two_level_clustering/general_clustering_standardised/second_level_clustering/AUC-ROC/subclusters_of_cluster_11.pickle', 'rb') as handle:  
    sub_cluster_indices = pickle.load(handle)           # ---> 1 sub-clusters only


#===> CLUSTER 11, SUB-CLUSTER 1  --> RANDOM FOREST