import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#=====================META-FEATURE EXTRACTION==================================
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
training_metafeatures = df_results.iloc[:, :-6]

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Cylinder Bands test dataset/Actual results/cylinder_metafeatures_202.pickle', 'rb') as handle:
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
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/cluster_centres.pickle', 'rb') as handle:
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

cluster_allocation = distances.index(min(distances))    #---> CLUSTER 10
    
#==============================================================================
#                       SECOND LEVEL CLUSTERING    
#==============================================================================    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/Plots/Stat_InfoTherory_2nd level clustering_standardised/AUC-ROC/subclusters_of_cluster_10.pickle', 'rb') as handle:
    sub_cluster_indices = pickle.load(handle)           # ---> 2 sub-clusters

training = all_meta_features.iloc[:-1, :]
training.index = df_results.index

"""
training variable is a dataframe with all statistical+info-theoretic
meta-features of the 202 training datasets:
"""
training = training.iloc[:, 6:]

#==============================================================================  
#                        Find the sub-cluster centres
#==============================================================================  

#==============================================================================  
#                        Sub-cluster 0
#==============================================================================  
subcluster_0_indeces =  sub_cluster_indices[0]
subcluster_0 = training.loc[subcluster_0_indeces]

subcluster_0_centres = []
for col in subcluster_0.columns: # 6 to 22
    subcluster_0_centres.append(np.mean(subcluster_0[col])) #--> Find the sub-cluster centre mean
    
#==============================================================================  
#                        Sub-cluster 1
#==============================================================================  
subcluster_1_indeces =  sub_cluster_indices[1]
subcluster_1 = training.loc[subcluster_1_indeces]

subcluster_1_centres = []
for col in subcluster_1.columns: # 6 to 22
    subcluster_1_centres.append(np.mean(subcluster_1[col]))



#==============================================================================  
#                        Find the sub-cluster of the test dataset
#==============================================================================  
subcluster_centres_all = {0: subcluster_0_centres,
                          1: subcluster_1_centres}

#Find the cluster at which the test dataset is allocated to:
distances_sub = []

for key in subcluster_centres_all.keys():
    distance = 0
    for j in range(17):
        distance += abs( subcluster_centres_all[key][j] - stat_and_info_metafeatures[j])
    distances_sub.append(distance)

subcluster_allocation = distances_sub.index(min(distances_sub))    #---> SUBCLUSTER 1 

#===> CLUSTER 10, SUB-CLUSTER 1
#==> Check from cluster bars!

#Actual results
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Cylinder Bands test dataset/Actual results/cylinder_top_3_roc.pickle', 'rb') as handle:
    actual_results = pickle.load(handle)
print("==========================================")
print("Top 1 ACTUAL model:      " + actual_results[0])
print("Top 2 ACTUAL model:      " + actual_results[1])
print("Top 3 ACTUAL model:      " + actual_results[2])

