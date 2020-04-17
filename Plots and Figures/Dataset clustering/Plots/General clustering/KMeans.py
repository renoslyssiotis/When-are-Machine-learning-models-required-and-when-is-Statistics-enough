import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#==============================================================================
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
general_metadataset = df_results.iloc[:, :6]
general_metadataset = general_metadataset.to_numpy()
kmeans = KMeans(n_clusters = 2, 
                init = "k-means++",
                n_init = 50,
                max_iter = 500).fit(general_metadataset)

print(kmeans.labels_)
clusters = list(kmeans.labels_)
print(kmeans.cluster_centers_)
cluster_centre_1 = list(kmeans.cluster_centers_[0])
cluster_centre_2 = list(kmeans.cluster_centers_[1])
for i in cluster_centre_1:
    i = '{0:.3f}'.format(i)
    
for j in cluster_centre_2:
    j = '{0:.3f}'.format(j)

                
#==============================================================================
#==========================ELBOW METHOD========================================
#==============================================================================
# k means determine k
inertia = []
distortions = []
K = range(1,10)
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
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#INERTIA
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()


#==============================================================================
#==========================SILHOUETTE METHOD===================================
#==============================================================================
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

print(__doc__)


range_n_clusters = [2, 3, 4, 5, 6, 7]

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





























