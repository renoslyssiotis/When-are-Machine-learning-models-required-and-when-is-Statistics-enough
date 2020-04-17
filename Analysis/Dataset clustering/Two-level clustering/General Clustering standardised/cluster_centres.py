import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#=====================META-FEATURE EXTRACTION==================================
df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
general_metadataset = df_results.iloc[:, :6].to_numpy()
general_metadataset = StandardScaler().fit(general_metadataset).transform(general_metadataset)
general_metadataset = pd.DataFrame(general_metadataset)

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Analysis/Dataset clustering/Two-level clustering/General Clustering standardised/datasetIndex_per_cluster.pickle', 'rb') as handle:
    index_per_cluster = pickle.load(handle)
    
#===========================CLUSTER 0==========================================
cluster_0_indeces = index_per_cluster[0]
cluster_0 = general_metadataset.iloc[cluster_0_indeces, :6]
cluster_0_centres = [np.mean(cluster_0[0]),
                     np.mean(cluster_0[1]),
                     np.mean(cluster_0[2]),
                     np.mean(cluster_0[3]),
                     np.mean(cluster_0[4])]

#===========================CLUSTER 1==========================================
cluster_1_indeces = index_per_cluster[1]
cluster_1 = general_metadataset.iloc[cluster_1_indeces, :6]
cluster_1_centres = [np.mean(cluster_1[0]),
                     np.mean(cluster_1[1]),
                     np.mean(cluster_1[2]),
                     np.mean(cluster_1[3]),
                     np.mean(cluster_1[4])]

#===========================CLUSTER 2==========================================
cluster_2_indeces = index_per_cluster[2]
cluster_2 = general_metadataset.iloc[cluster_2_indeces, :6]
cluster_2_centres = [np.mean(cluster_2[0]),
                     np.mean(cluster_2[1]),
                     np.mean(cluster_2[2]),
                     np.mean(cluster_2[3]),
                     np.mean(cluster_2[4])]

#===========================CLUSTER 3==========================================
cluster_3_indeces = index_per_cluster[3]
cluster_3 = general_metadataset.iloc[cluster_3_indeces, :6]
cluster_3_centres = [np.mean(cluster_3[0]),
                     np.mean(cluster_3[1]),
                     np.mean(cluster_3[2]),
                     np.mean(cluster_3[3]),
                     np.mean(cluster_3[4])]

#===========================CLUSTER 4==========================================
cluster_4_indeces = index_per_cluster[4]
cluster_4 = general_metadataset.iloc[cluster_4_indeces, :6]
cluster_4_centres = [np.mean(cluster_4[0]),
                     np.mean(cluster_4[1]),
                     np.mean(cluster_4[2]),
                     np.mean(cluster_4[3]),
                     np.mean(cluster_4[4])]

#===========================CLUSTER 5==========================================
cluster_5_indeces = index_per_cluster[5]
cluster_5 = general_metadataset.iloc[cluster_5_indeces, :6]
cluster_5_centres = [np.mean(cluster_5[0]),
                     np.mean(cluster_5[1]),
                     np.mean(cluster_5[2]),
                     np.mean(cluster_5[3]),
                     np.mean(cluster_5[4])]

#===========================CLUSTER 6==========================================
cluster_6_indeces = index_per_cluster[6]
cluster_6 = general_metadataset.iloc[cluster_6_indeces, :6]
cluster_6_centres = [np.mean(cluster_6[0]),
                     np.mean(cluster_6[1]),
                     np.mean(cluster_6[2]),
                     np.mean(cluster_6[3]),
                     np.mean(cluster_6[4])]

#===========================CLUSTER 7==========================================
cluster_7_indeces = index_per_cluster[7]
cluster_7 = general_metadataset.iloc[cluster_7_indeces, :6]
cluster_7_centres = [np.mean(cluster_7[0]),
                     np.mean(cluster_7[1]),
                     np.mean(cluster_7[2]),
                     np.mean(cluster_7[3]),
                     np.mean(cluster_7[4])]

#===========================CLUSTER 8==========================================
cluster_8_indeces = index_per_cluster[8]
cluster_8 = general_metadataset.iloc[cluster_8_indeces, :6]
cluster_8_centres = [np.mean(cluster_8[0]),
                     np.mean(cluster_8[1]),
                     np.mean(cluster_8[2]),
                     np.mean(cluster_8[3]),
                     np.mean(cluster_8[4])]

#===========================CLUSTER 9==========================================
cluster_9_indeces = index_per_cluster[9]
cluster_9 = general_metadataset.iloc[cluster_9_indeces, :6]
cluster_9_centres = [np.mean(cluster_9[0]),
                     np.mean(cluster_9[1]),
                     np.mean(cluster_9[2]),
                     np.mean(cluster_9[3]),
                     np.mean(cluster_9[4])]

#===========================CLUSTER 10==========================================
cluster_10_indeces = index_per_cluster[10]
cluster_10 = general_metadataset.iloc[cluster_10_indeces, :6]
cluster_10_centres = [np.mean(cluster_10[0]),
                     np.mean(cluster_10[1]),
                     np.mean(cluster_10[2]),
                     np.mean(cluster_10[3]),
                     np.mean(cluster_10[4])]

#===========================CLUSTER 11==========================================
cluster_11_indeces = index_per_cluster[11]
cluster_11 = general_metadataset.iloc[cluster_11_indeces, :6]
cluster_11_centres = [np.mean(cluster_11[0]),
                     np.mean(cluster_11[1]),
                     np.mean(cluster_11[2]),
                     np.mean(cluster_11[3]),
                     np.mean(cluster_11[4])]

#===========================CLUSTER 12==========================================
cluster_12_indeces = index_per_cluster[12]
cluster_12 = general_metadataset.iloc[cluster_12_indeces, :6]
cluster_12_centres = [np.mean(cluster_12[0]),
                     np.mean(cluster_12[1]),
                     np.mean(cluster_12[2]),
                     np.mean(cluster_12[3]),
                     np.mean(cluster_12[4])]

cluster_centres = {0: cluster_0_centres,
                   1: cluster_1_centres,
                   2: cluster_2_centres,
                   3: cluster_3_centres,
                   4: cluster_4_centres,
                   5: cluster_5_centres,
                   6: cluster_6_centres,
                   7: cluster_7_centres,
                   8: cluster_8_centres,
                   9: cluster_9_centres,
                   10: cluster_10_centres,
                   11: cluster_11_centres,
                   12: cluster_12_centres}

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Analysis/Dataset clustering/Two-level clustering/General Clustering standardised/cluster_centres.pickle', 'wb') as handle:
    pickle.dump(cluster_centres, handle, protocol=pickle.HIGHEST_PROTOCOL)






