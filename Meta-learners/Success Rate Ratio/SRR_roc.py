import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Meta-learners/KNN Ranking Method')
import pickle
import numpy as np

with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/nested_results_roc.pickle', 'rb') as handle:
    nested_results_roc = pickle.load(handle)    
    
"""
For each dataset i, find the Success Rate Ratio, SRR_j,k which is equal to:
    (1 - ER_j)/(1 - ER_k), where ER_j is the error-rate of model j
    --> if SRR_j,k < 1, then k performs better than j
"""    

SRR_consolidated_table = np.zeros((len(nested_results_roc[0]), len(nested_results_roc[0])))


for i in range(len(nested_results_roc[0])):     #0 to 11 classification models
    for j in range(len(nested_results_roc[0])): #0 to 11 classification models
        
        sum = 0
        
        for dataset in range(len(nested_results_roc)): #0 to 201 datasets
            
            SRR = (list(nested_results_roc[dataset].values())[i])/(list(nested_results_roc[dataset].values())[j])
            sum += SRR
            
        SRR_consolidated_table[i][j] = sum/len(nested_results_roc)
        
            
        
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
