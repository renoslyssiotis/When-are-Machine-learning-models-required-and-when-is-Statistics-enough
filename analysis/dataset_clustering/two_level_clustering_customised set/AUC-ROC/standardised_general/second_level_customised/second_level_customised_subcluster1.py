import os, pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import Counter
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

#==============================================================================
df_results = pd.read_pickle(str(p.parents[6])+'/test/df_results.plk')

with open(str(p.parents[1])+'/dataset_per_cluster.pickle', 'rb') as handle:
    dataset_per_cluster = pickle.load(handle)
    
with open(str(p.parents[1])+'/datasetIndex_per_cluster.pickle', 'rb') as handle:
    cluster_indexes = pickle.load(handle)    
    
with open(str(p.parents[1])+'/models_per_cluster_ROC.pickle', 'rb') as handle:
    roc_indexes = pickle.load(handle)    
  
#==============================================================================
#                               CLUSTER 1
#==============================================================================
#For the datasets in cluster 1, create a dataframe with their statistical
#and information-theoretic meta-features to perform clustering on:
cluster_1_data = df_results.iloc[cluster_indexes[1]]
cluster_1_data.index = list(np.arange(0,len(cluster_1_data.index))) 

cluster_1_data_numpy = df_results.iloc[cluster_indexes[1], [7,9,11,18,19]].to_numpy()
cluster_1_data_numpy = StandardScaler().fit(cluster_1_data_numpy).transform(cluster_1_data_numpy)

#==============================================================================
#==========================SILHOUETTE METHOD===================================
#==============================================================================
range_n_clusters = [2, 3]
silhouette_averages = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(cluster_1_data_numpy)
    silhouette_avg = silhouette_score(cluster_1_data_numpy, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    silhouette_averages.append(silhouette_avg)
    sample_silhouette_values = silhouette_samples(cluster_1_data_numpy, cluster_labels)

#Silhouete graph
x = list(np.arange(2,4))
plt.xticks([2,3])
plt.xlabel('Number of clusters, $\it{k}$')
plt.ylabel("Average silhouette score")
plt.title("Silhouette analysis", weight = 'bold')
plt.axvline(x=2, linestyle='--', color='blue')
plt.plot(x, silhouette_averages)                    #=============> 2 sub-clusters 

#==============================================================================
#==========================2-ND LEVEL CLUSTERING===============================
#==============================================================================
kmeans = KMeans(n_clusters = 2, 
                init = "k-means++",
                n_init = 50,
                max_iter = 500).fit(cluster_1_data_numpy)

print(kmeans.labels_)
clusters_1 = list(kmeans.labels_)

cluster_indexes_1 = {0: [], 1: []}

for index, value in enumerate(clusters_1):
    if value == 0:
        cluster_indexes_1[0].append(index)
    elif value == 1:
        cluster_indexes_1[1].append(index)        

#==============================================================================
#                       BAR CHART FOR EACH CLUSTER - AUC-ROC
#==============================================================================
roc_indexes_1 = {0: [], 1: []}
for key in list(cluster_indexes_1.keys()):
    for j in cluster_indexes_1[key]:
        roc_indexes_1[key].append(cluster_1_data['Best model ROC'][j])

for key in list(roc_indexes_1.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(roc_indexes_1[key])))), list(dict(Counter(roc_indexes_1[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(roc_indexes_1[key])))), list(dict(Counter(roc_indexes_1[key])).keys()), rotation=45)
    plt.title("Sub-cluster {} of Cluster 1".format(key), weight = 'bold')
    plt.ylabel("Counts of best-performing model (AUC-ROC)")


#==============================================================================
#                       Save the datasets that lie within each sub-cluster
#==============================================================================
with open(str(p.parents[1])+'/datasetIndex_per_cluster.pickle', 'rb') as handle:
    indeces = pickle.load(handle)   
cluster_1_data_ = df_results.iloc[indeces[1], [7,9,11,18,19]]
    
datasets_in_each_sub_cluster_1 = {0: [], 1: []}
for key in list(datasets_in_each_sub_cluster_1.keys()):
    for j in cluster_indexes_1[key]:
        datasets_in_each_sub_cluster_1[key].append(cluster_1_data_.index[j])

# with open('subclusters_of_cluster_1.pickle', 'wb') as handle:
#     pickle.dump(datasets_in_each_sub_cluster_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    











