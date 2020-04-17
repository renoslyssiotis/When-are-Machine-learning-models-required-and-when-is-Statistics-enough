import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter

#==============================================================================
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/dataset_per_cluster.pickle', 'rb') as handle:
    dataset_per_cluster = pickle.load(handle)
    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/datasetIndex_per_cluster.pickle', 'rb') as handle:
    cluster_indexes = pickle.load(handle)    
    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/models_per_cluster_ROC.pickle', 'rb') as handle:
    roc_indexes = pickle.load(handle)    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/models_per_cluster_PRC.pickle', 'rb') as handle:
    prc_indexes = pickle.load(handle)    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/models_per_cluster_f1.pickle', 'rb') as handle:
    f1_indexes = pickle.load(handle)    
    

#==============================================================================
#                               CLUSTER 11
#==============================================================================
#For the datasets in cluster 11, create a dataframe with their statistical
#and information-theoretic meta-features to perform clustering on:
cluster_11_data = df_results.iloc[cluster_indexes[11]]
cluster_11_data.index = list(np.arange(0,len(cluster_11_data.index))) 

cluster_11_data_numpy = df_results.iloc[cluster_indexes[11], 6:-6].to_numpy()
cluster_11_data_numpy = StandardScaler().fit(cluster_11_data_numpy).transform(cluster_11_data_numpy)

# #==============================================================================
# #==========================SILHOUETTE METHOD===================================
# #==============================================================================
# from sklearn.metrics import silhouette_samples, silhouette_score
# import matplotlib.cm as cm

# print(__doc__)


# range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
# silhouette_averages = []
# for n_clusters in range_n_clusters:
#     fig, ax1 = plt.subplots()
#     fig.set_size_inches(15, 10)

#     ax1.set_xlim([-0.1, 1])
#     ax1.set_ylim([0, len(cluster_11_data) + (n_clusters + 1) * 10])
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(cluster_11_data)


#     silhouette_avg = silhouette_score(cluster_11_data, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#     silhouette_averages.append(silhouette_avg)
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(cluster_11_data, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]

#         ith_cluster_silhouette_values.sort()

#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i

#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)

#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=24)

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples

#     ax1.set_title("The silhouette plot for the various clusters", fontsize = 26, weight = 'bold')
#     ax1.set_xlabel("The silhouette coefficient values", fontsize = 24)
#     ax1.set_ylabel("Cluster label", fontsize = 24)

#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#     plt.xticks(fontsize = 24)

#     # Labeling the clusters
#     centers = clusterer.cluster_centers_

# plt.show()

# #Silhouete graph
# x = list(np.arange(2, 9))
# plt.xticks([2, 3, 4, 5, 6, 7, 8])
# plt.xlabel('Number of clusters, $\it{k}$')
# plt.ylabel("Average silhouette score")
# plt.title("Silhouette analysis: 2nd level clustering for Cluster 11", weight = 'bold')
# plt.axvline(x=6, linestyle='--', color='blue')
# plt.plot(x, silhouette_averages)

#==============================================================================
#==============================================================================
#==============================================================================

kmeans = KMeans(n_clusters = 1, 
                init = "k-means++",
                n_init = 50,
                max_iter = 500).fit(cluster_11_data_numpy)

print(kmeans.labels_)
clusters_11 = list(kmeans.labels_)

#==============================================================================
#==============================================================================
#==============================================================================
cluster_indexes_11 = {0: []}

for index, value in enumerate(clusters_11):
    if value == 0:
        cluster_indexes_11[0].append(index)
    # if value == 1:
    #     cluster_indexes[1].append(index)       
    # if value == 2:
    #     cluster_indexes[2].append(index)
    # if value == 3:
    #     cluster_indexes[3].append(index)
    # if value == 4:
    #     cluster_indexes[4].append(index)
    # if value == 5:
    #     cluster_indexes[5].append(index)
    # if value == 6:
    #     cluster_indexes[6].append(index) 

#==============================================================================
#                       BAR CHART FOR EACH CLUSTER - AUC-ROC
#==============================================================================
roc_indexes_11 = {0: []}
for key in list(cluster_indexes_11.keys()):
    for j in cluster_indexes_11[key]:
        roc_indexes_11[key].append(cluster_11_data['Best model ROC'][j])

for key in list(roc_indexes_11.keys()):
    plt.figure()
    plt.bar(range(len(dict(Counter(roc_indexes_11[key])))), list(dict(Counter(roc_indexes_11[key])).values()), align='center', width = 0.25)
    plt.xticks(range(len(dict(Counter(roc_indexes_11[key])))), list(dict(Counter(roc_indexes_11[key])).keys()), rotation=45)
    plt.yticks(list(np.arange(1, max(list(dict(Counter(roc_indexes_11[key])).values())) + 1)),
               list(np.arange(1, max(list(dict(Counter(roc_indexes_11[key])).values())) + 1)))
    plt.title("Sub-cluster {} of Cluster 11".format(key), weight = 'bold')
    plt.ylabel("Counts of best-performing model (AUC-ROC)")


#==============================================================================
#                       Save the datasets that lie within each sub-cluster
#==============================================================================
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/datasetIndex_per_cluster.pickle', 'rb') as handle:
    indeces = pickle.load(handle)   
cluster_11_data_ = df_results.iloc[indeces[11], 6:-6]
    
datasets_in_each_sub_cluster_11 = {0: []}
for key in list(datasets_in_each_sub_cluster_11.keys()):
    for j in cluster_indexes_11[key]:
        datasets_in_each_sub_cluster_11[key].append(cluster_11_data_.index[j])

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Dataset clustering/Plots/Stat_InfoTherory_2nd level clustering_standardised/subclusters_of_cluster_11.pickle', 'wb') as handle:
    pickle.dump(datasets_in_each_sub_cluster_11, handle, protocol=pickle.HIGHEST_PROTOCOL)
    











