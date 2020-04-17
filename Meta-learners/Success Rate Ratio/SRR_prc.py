import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Meta-learners/KNN Ranking Method')
import pickle
import numpy as np

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_prc.pickle', 'rb') as handle:
    nested_results_prc = pickle.load(handle)    

SRR_consolidated_table = np.zeros((len(nested_results_prc[0]), len(nested_results_prc[0])))


for i in range(len(nested_results_prc[0])):     #0 to 11 classification models
    for j in range(len(nested_results_prc[0])): #0 to 11 classification models
        
        sum = 0
        
        for dataset in range(len(nested_results_prc)): #0 to 201 datasets
            
            SRR = (list(nested_results_prc[dataset].values())[i])/(list(nested_results_prc[dataset].values())[j])
            sum += SRR
            
        SRR_consolidated_table[i][j] = sum/len(nested_results_prc)
        
            
        
overall_mean_success_ratio = SRR_consolidated_table.shape[0] * [0]

for i in range(SRR_consolidated_table.shape[0]):
    
    sum = 0
    
    for j in range(SRR_consolidated_table.shape[0]):    
        
        sum += SRR_consolidated_table[i][j]
        
    overall_mean_success_ratio[i] = sum/(SRR_consolidated_table.shape[0] - 1)

keys = ['logit', 'dtree', 'rforest', 'perceptron', 'MLP', 'ADAboost', 'linearSVC', 'LDA', 'QDA', 'gaussianNB', 'bernoulliNB', 'bagging']
values = overall_mean_success_ratio

SRR_ranking = dict(zip(keys, values))
SRR_ranking = {k: v for k, v in sorted(SRR_ranking.items(), key=lambda item: item[1])}
