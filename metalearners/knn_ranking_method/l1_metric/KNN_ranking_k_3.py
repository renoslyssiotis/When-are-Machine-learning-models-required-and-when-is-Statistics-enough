import pandas as pd

def KNN_ranking(feature_selected_meta_dataset, meta_features, nested_results):
    
    distance_count = []
    meta_features_list = list(meta_features.keys())
    meta_features_values = list(meta_features.values())
    
    #Find the 'closeness' of the test dataset with the training datasets
    #in the meta-feature space:
    
    for dataset in list(feature_selected_meta_dataset.index): #Dataset 0, Dataset 1, ..., Dataset 203
        
        summation = 0
        
        for index,value in enumerate(meta_features_list):  
            distance = abs(feature_selected_meta_dataset.loc[dataset][index] - meta_features_values[index])
            summation += distance
            
        distance_count.append(summation)
    # print(distance_count)
    # print(len(distance_count))
    """
    These are the indeces of the meta-dataset that indicate which k=3 datasets are 
    the closest to the test dataset:
    """    
    min_index_1 = distance_count.index(sorted(distance_count)[0])
    print("mind_index_1: %d" %min_index_1)
    min_index_2 = distance_count.index(sorted(distance_count)[1])
    min_index_3 = distance_count.index(sorted(distance_count)[2])  
    print(sorted(distance_count[:3]))
    
    #Create a ranking dataframe
    ranking_df = pd.DataFrame(data = [list(nested_results[min_index_1].values()),
                                  list(nested_results[min_index_2].values()),
                                  list(nested_results[min_index_3].values())], 
                              columns = list(nested_results[1].keys()),
                              index = ['Closest dataset 1','Closest dataset 2', 'Closest dataset 3'])
    
    closest_3_datasets = ranking_df.T                

    #Transform the AUC-ROC metric into model rankings:
    closest_3_datasets['Rank dataset 1'] = closest_3_datasets['Closest dataset 1'].rank(ascending=False)
    closest_3_datasets['Rank dataset 2'] = closest_3_datasets['Closest dataset 2'].rank(ascending=False)
    closest_3_datasets['Rank dataset 3'] = closest_3_datasets['Closest dataset 3'].rank(ascending=False)
    
    #Compute the average rank of each model for the three closest datasets:
    average_rank = []
    for i in range(len(closest_3_datasets.index)):
        average_rank.append((closest_3_datasets.iloc[i][-3:].sum())/3)
    
    closest_3_datasets['Average model rank'] = average_rank
    closest_3_datasets = closest_3_datasets.sort_values(by = 'Average model rank')

    #Once the average rank of each model is obtained, get a ranking recommendation:
    top1 = closest_3_datasets.index[0]
    top2 = closest_3_datasets.index[1]
    top3 = closest_3_datasets.index[2]

    # #Once the average rank of each model is obtained, get a ranking recommendation:
    # top1 = closest_3_datasets.loc[closest_3_datasets['Average model rank'] == sorted(closest_3_datasets['Average model rank'])[0]].index[0]
    # top2 = closest_3_datasets.loc[closest_3_datasets['Average model rank'] == sorted(closest_3_datasets['Average model rank'])[1]].index[0]
    # top3 = closest_3_datasets.loc[closest_3_datasets['Average model rank'] == sorted(closest_3_datasets['Average model rank'])[2]].index[0]
    
    return top1, top2, top3
