import pickle, os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import collections
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

#==============================================================================
df_results = pd.read_pickle(str(p.parents[4])+'/test/df_results.plk')
general_metadataset = df_results.iloc[:, :6].to_numpy() #6 general meta-features
general_metadataset = StandardScaler().fit(general_metadataset).transform(general_metadataset) #Standardise the general meta-features

#==============================================================================
#==========================SILHOUETTE METHOD===================================
#==============================================================================
print(__doc__)

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
plt.axvline(x=13, linestyle='--', color='blue')
plt.plot(x, silhouette_averages)                    #=====> k = 13 clusters

#==============================================================================
#==========================SILHOUETTE METHOD===================================
#==============================================================================
kmeans = KMeans(n_clusters = 13, 
                init = "k-means++",
                n_init = 50,
                max_iter = 500).fit(general_metadataset)

print(kmeans.labels_)
clusters = list(kmeans.labels_)
cluster_centers = kmeans.cluster_centers_

#==============================================================================
#=======================BLOB PLOT==============================================
#=============================================================================
d = dict(Counter(clusters))
dataset_per_cluster = dict(sorted(d.items()))
print(dataset_per_cluster)

# with open('dataset_per_cluster.pickle', 'wb') as handle:
#     pickle.dump(dataset_per_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

od = collections.OrderedDict(sorted(d.items()))
datasets_per_cluster = list(dict(od).values())
plt.scatter(x = 1, y = 0.9, s= 10*datasets_per_cluster[0], alpha = 0.5)
plt.annotate(datasets_per_cluster[0], (1,0.9+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = -1, y = -1, s= 10*datasets_per_cluster[1], alpha = 0.5)
plt.annotate(datasets_per_cluster[1], (-1,-1+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = -1, y = 1, s= 10*datasets_per_cluster[2], alpha = 0.5)
plt.annotate(datasets_per_cluster[2], (-1+0.1,1-0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = 1, y = -1, s= 10*datasets_per_cluster[3], alpha = 0.5)
plt.annotate(datasets_per_cluster[3], (1,-1+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = 0, y = 0, s= 10*datasets_per_cluster[4], alpha = 0.5)
plt.annotate(datasets_per_cluster[4], (0,0+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = -0.4, y = 0.5, s= 10*datasets_per_cluster[5], alpha = 0.5)
plt.annotate(datasets_per_cluster[5], (-0.4, 0.5+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = -0.5, y = 0.4, s= 10*datasets_per_cluster[6], alpha = 0.5)
plt.annotate(datasets_per_cluster[6], (-0.5-0.1, 0.4+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = -1, y = 0.4, s= 10*datasets_per_cluster[7], alpha = 0.5)
plt.annotate(datasets_per_cluster[7], (-1, 0.4+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = 0.5, y = 0.5, s= 10*datasets_per_cluster[8], alpha = 0.5)
plt.annotate(datasets_per_cluster[8], (0.5, 0.5+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = 0.1, y = 0.5, s= 10*datasets_per_cluster[9], alpha = 0.5)
plt.annotate(datasets_per_cluster[9], (0.1, 0.5+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = 0, y = -0.6, s= 10*datasets_per_cluster[10], alpha = 0.5)
plt.annotate(datasets_per_cluster[10], (0, -0.6+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = 0.5, y = -0.4, s= 10*datasets_per_cluster[11], alpha = 0.5)
plt.annotate(datasets_per_cluster[11], (0.5, -0.4+0.1), fontsize = 14, weight = 'bold')
plt.scatter(x = -0.5, y = -0.4, s= 10*datasets_per_cluster[12], alpha = 0.5)
plt.annotate(datasets_per_cluster[12], (-0.5, -0.4+0.1), fontsize = 14, weight = 'bold')
plt.yticks([])
plt.xticks([])
plt.title("Dataset distribution in each one of the 13 clusters", weight='bold')

#==============================================================================
#       Find the dataset indeces that correspond to each cluster
#==============================================================================

cluster_indexes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []}

for index, value in enumerate(clusters):
    if value == 0:
        cluster_indexes[0].append(index)
    elif value == 1:
        cluster_indexes[1].append(index)        
    elif value == 2:
        cluster_indexes[2].append(index)
    elif value == 3:
        cluster_indexes[3].append(index)
    elif value == 4:
        cluster_indexes[4].append(index)        
    elif value == 5:
        cluster_indexes[5].append(index)
    elif value == 6:
        cluster_indexes[6].append(index)
    elif value == 7:
        cluster_indexes[7].append(index)        
    elif value == 8:
        cluster_indexes[8].append(index)
    elif value == 9:
        cluster_indexes[9].append(index)
    elif value == 10:
        cluster_indexes[10].append(index)        
    elif value == 11:
        cluster_indexes[11].append(index)
    else:
        cluster_indexes[12].append(index)
        
# with open('datasetIndex_per_cluster.pickle', 'wb') as handle:
#     pickle.dump(cluster_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)

#==============================================================================
#Get the performance of ROC of each dataset, in each cluster
#==============================================================================
        
roc_indexes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []}
for key in list(cluster_indexes.keys()):
    for j in cluster_indexes[key]:
        roc_indexes[key].append(df_results['Best model ROC'][j])
        
prc_indexes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []}
for key in list(cluster_indexes.keys()):
    for j in cluster_indexes[key]:
        prc_indexes[key].append(df_results['Best model PRC'][j])
        
f1_indexes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []}
for key in list(cluster_indexes.keys()):
    for j in cluster_indexes[key]:
        f1_indexes[key].append(df_results['Best model f1'][j])

# with open('models_per_cluster_ROC.pickle', 'wb') as handle:
#     pickle.dump(roc_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('models_per_cluster_PRC.pickle', 'wb') as handle:
#     pickle.dump(prc_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('models_per_cluster_f1.pickle', 'wb') as handle:
#     pickle.dump(f1_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)

#==============================================================================
#                       BAR CHART FOR EACH CLUSTER - AUC-ROC
#==============================================================================

from collections import Counter
for key in list(roc_indexes.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(roc_indexes[key])))), list(dict(Counter(roc_indexes[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(roc_indexes[key])))), list(dict(Counter(roc_indexes[key])).keys()), rotation=45)
    plt.yticks(list(np.arange(1, max(list(dict(Counter(roc_indexes[key])).values())) + 1)),
               list(np.arange(1, max(list(dict(Counter(roc_indexes[key])).values())) + 1)))
    plt.title("Cluster {}".format(key), weight = 'bold')
    plt.ylabel("Counts of best-performing model (AUC-ROC)")

#==============================================================================
#                       BAR CHART FOR EACH CLUSTER - AUC-PRC
#==============================================================================
for key in list(prc_indexes.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(prc_indexes[key])))), list(dict(Counter(prc_indexes[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(prc_indexes[key])))), list(dict(Counter(prc_indexes[key])).keys()), rotation=45)
    plt.yticks(list(np.arange(1, max(list(dict(Counter(prc_indexes[key])).values())) + 1)),
               list(np.arange(1, max(list(dict(Counter(prc_indexes[key])).values())) + 1)))
    plt.title("Cluster {}".format(key), weight = 'bold')
    plt.ylabel("Counts of best-performing model (AUC-PRC)")
    
#==============================================================================
#                       BAR CHART FOR EACH CLUSTER - f1
#==============================================================================
for key in list(f1_indexes.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(f1_indexes[key])))), list(dict(Counter(f1_indexes[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(f1_indexes[key])))), list(dict(Counter(f1_indexes[key])).keys()), rotation=45)
    plt.yticks(list(np.arange(1, max(list(dict(Counter(f1_indexes[key])).values())) + 1)),
               list(np.arange(1, max(list(dict(Counter(f1_indexes[key])).values())) + 1)))
    plt.title("Cluster {}".format(key), weight = 'bold')
    plt.ylabel("Counts of best-performing model (f1)")




























