import pandas as pd
import pickle
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from collections import Counter
import collections

#==============================================================================
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#Standardise the "Number of instances" meta-feature
general_metadataset = df_results.iloc[:, 0].to_numpy().reshape(-1,1)
general_metadataset = StandardScaler().fit(general_metadataset).transform(general_metadataset)

#==============================================================================
#==========================SILHOUETTE METHOD===================================
#==============================================================================
range_n_clusters = [2, 3, 4, 5, 6, 7, 8 ,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
silhouette_averages = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(general_metadataset)
    silhouette_avg = silhouette_score(general_metadataset, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    silhouette_averages.append(silhouette_avg)
    sample_silhouette_values = silhouette_samples(general_metadataset, cluster_labels)

#Silhouete graph
x = list(np.arange(2,26))
plt.xticks([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
plt.xlabel('Number of clusters, $\it{k}$')
plt.ylabel("Average silhouette score")
plt.title("Silhouette analysis", weight = 'bold')
plt.axvline(x=2, linestyle='--', color='blue')
plt.plot(x, silhouette_averages)                    #=============> 2 clusters 

#===============        clustering to 2 clusters =============================#
kmeans = KMeans(n_clusters = 2, 
                init = "k-means++",
                n_init = 50,
                max_iter = 500).fit(general_metadataset)

print(kmeans.labels_)
clusters = list(kmeans.labels_)
cluster_centers = kmeans.cluster_centers_

d = dict(Counter(clusters))
dataset_per_cluster = dict(sorted(d.items()))

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/Two-level clustering_customised set/f1/Standardised general/dataset_per_cluster.pickle', 'wb') as handle:
    pickle.dump(dataset_per_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

#==============================================================================
#       Find the dataset indeces that correspond to each cluster
#==============================================================================

cluster_indexes = {0: [], 1: []}

for index, value in enumerate(clusters):
    if value == 0:
        cluster_indexes[0].append(index)
    elif value == 1:
        cluster_indexes[1].append(index)        

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/Two-level clustering_customised set/f1/Standardised general/datasetIndex_per_cluster.pickle', 'wb') as handle:
    pickle.dump(cluster_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)

#==============================================================================
#Get the performance of F1 of each dataset, in each cluster
#==============================================================================  
f1_indexes = {0: [], 1: []}
for key in list(cluster_indexes.keys()):
    for j in cluster_indexes[key]:
        f1_indexes[key].append(df_results['Best model f1'][j])

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/Two-level clustering_customised set/f1/Standardised general/models_per_cluster_f1.pickle', 'wb') as handle:
    pickle.dump(f1_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)

#==============================================================================
#                       BAR CHART FOR EACH CLUSTER - f1
#==============================================================================
for key in list(f1_indexes.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(f1_indexes[key])))), list(dict(Counter(f1_indexes[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(f1_indexes[key])))), list(dict(Counter(f1_indexes[key])).keys()), rotation=45)
    # plt.yticks(list(np.arange(1, max(list(dict(Counter(roc_indexes[key])).values())) + 1)),
    #            list(np.arange(1, max(list(dict(Counter(roc_indexes[key])).values())) + 1)))
    plt.title("Cluster {}".format(key), weight = 'bold')
    plt.ylabel("Counts of best-performing model (f1)")
