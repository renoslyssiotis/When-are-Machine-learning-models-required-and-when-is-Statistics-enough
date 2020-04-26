import pandas as pd
import matplotlib.pyplot as plt

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
# Find the total variational distace between the empirical probability distributions
# vs. the number of training datasets
#===============================================================================
y = df_results["Best model ROC"]
y = y.sample(frac=1)

probability_distributions = []

#Find the variational distance between the empirical dns of each iteration
for i in range(len(y)): #0,1,...,201
    
    unique_models = {'auc_roc_bernoulliNB': 0,
                 'auc_roc_MLP': 0, 
                 'auc_roc_logit': 0,
                 'auc_roc_rforest': 0,
                 'auc_roc_ADAboost': 0,
                 'auc_roc_dtree': 0,
                 'auc_roc_bagging': 0,
                 'auc_roc_linearSVC': 0,
                 'auc_roc_perceptron': 0,
                 'auc_roc_LDA': 0,
                 'auc_roc_QDA': 0,
                 'auc_roc_gaussianNB': 0}
    
    pd_series = pd.Series(unique_models) #index = models, elements = 0 for all
    best_models = dict(y[:i+1].value_counts())
    
    for model, count in best_models.items():
        unique_models[model] += count
        
    model_counts = list(unique_models.values())
    empirical_prob = [x/sum(model_counts) for x in model_counts]
    probability_distributions.append(empirical_prob)

variational_distances = []

for i in range(1, len(y)):
    
    difference = [abs(x1 - x2) for (x1,x2) in
                  zip(probability_distributions[i],probability_distributions[i-1])]
    variational_distances.append(sum(difference))
    
    
# Plot the variational distances vs. the number of training datasets
training_datasets = list(range(2,203))

plt.figure(dpi = 1200)
plt.xlabel('Number of training datasets', fontsize = 16)
plt.ylabel("Variatioal distance", fontsize = 16)
plt.yticks(fontsize = 14)
# plt.ylim(0,3.5)
plt.xticks(fontsize = 14)
plt.title("Variational distance vs. Number of\n training datasets", fontsize = 16, fontweight = 'bold')
plt.plot(training_datasets, variational_distances, c = 'blue')        

