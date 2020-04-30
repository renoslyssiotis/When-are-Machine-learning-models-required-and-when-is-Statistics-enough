import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
import os
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

#==============================================================================
df_results = pd.read_pickle( str(p.parents[4])+'/test/df_results.plk')
general_metadataset = df_results.iloc[:, :6] #6 general meta-features (non-standardised)
general_metadataset = general_metadataset.to_numpy()

#==============================================================================
#==========================ELBOW METHOD========================================
#==============================================================================
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
print(__doc__)
range_n_clusters = [2, 3, 4, 5, 6, 7]

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(general_metadataset)

    silhouette_avg = silhouette_score(general_metadataset, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(general_metadataset, cluster_labels) #===> 2 CLUSTERS

#==============================================================================
#==========================CLUSTERING =========================================
#==============================================================================

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

                
