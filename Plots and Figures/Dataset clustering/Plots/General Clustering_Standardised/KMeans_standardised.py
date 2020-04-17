import pandas as pd
import pickle
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter

#==============================================================================
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
general_metadataset = df_results.iloc[:, :6].to_numpy()
general_metadataset = StandardScaler().fit(general_metadataset).transform(general_metadataset)

kmeans = KMeans(n_clusters = 13, 
                init = "k-means++",
                n_init = 50,
                max_iter = 500).fit(general_metadataset)

print(kmeans.labels_)
clusters = list(kmeans.labels_)
cluster_centers = kmeans.cluster_centers_

#==============================================================================
#=======================BLOB PLOT==============================================
#==============================================================================
import collections
d = dict(Counter(clusters))
dataset_per_cluster = dict(sorted(d.items()))
print(dataset_per_cluster)

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/dataset_per_cluster.pickle', 'wb') as handle:
    pickle.dump(dataset_per_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    if value == 1:
        cluster_indexes[1].append(index)        
    if value == 2:
        cluster_indexes[2].append(index)
    if value == 3:
        cluster_indexes[3].append(index)
    if value == 4:
        cluster_indexes[4].append(index)        
    if value == 5:
        cluster_indexes[5].append(index)
    if value == 6:
        cluster_indexes[6].append(index)
    if value == 7:
        cluster_indexes[7].append(index)        
    if value == 8:
        cluster_indexes[8].append(index)
    if value == 9:
        cluster_indexes[9].append(index)
    if value == 10:
        cluster_indexes[10].append(index)        
    if value == 11:
        cluster_indexes[11].append(index)
    if value == 12:
        cluster_indexes[12].append(index)
        
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/datasetIndex_per_cluster.pickle', 'wb') as handle:
    pickle.dump(cluster_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/models_per_cluster_ROC.pickle', 'wb') as handle:
    pickle.dump(roc_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/models_per_cluster_PRC.pickle', 'wb') as handle:
    pickle.dump(prc_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/models_per_cluster_f1.pickle', 'wb') as handle:
    pickle.dump(f1_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

#==============================================================================
#==========================ELBOW METHOD========================================
#==============================================================================
# k means determine k
inertia = []
distortions = []
K = range(1,25)
for k in K:
    kmeanModel = KMeans(n_clusters=k,
                        init = "k-means++",
                        n_init = 50,
                        max_iter = 1000)
    kmeanModel = kmeanModel.fit(general_metadataset)
    inertia.append(kmeanModel.inertia_)
    distortions.append(sum(np.min(cdist(general_metadataset, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / general_metadataset.shape[0])

#DISTORTION
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of clusters, $\it{k}$')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#INERTIA
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters, $\it{k}$')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()


#==============================================================================
#==========================SILHOUETTE METHOD===================================
#==============================================================================
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

print(__doc__)


range_n_clusters = [2, 3, 4, 5, 6, 7, 8 ,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
silhouette_averages = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots()
    fig.set_size_inches(15, 10)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(general_metadataset) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(general_metadataset)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(general_metadataset, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    silhouette_averages.append(silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(general_metadataset, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=24)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters", fontsize = 26, weight = 'bold')
    ax1.set_xlabel("The silhouette coefficient values", fontsize = 24)
    ax1.set_ylabel("Cluster label", fontsize = 24)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(fontsize = 24)

    # Labeling the clusters
    centers = clusterer.cluster_centers_

plt.show()

#Silhouete graph
x = list(np.arange(2,26))
plt.xticks([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
plt.xlabel('Number of clusters, $\it{k}$')
plt.ylabel("Average silhouette score")
plt.title("Silhouette analysis", weight = 'bold')
plt.axvline(x=13, linestyle='--', color='blue')
plt.plot(x, silhouette_averages)




























