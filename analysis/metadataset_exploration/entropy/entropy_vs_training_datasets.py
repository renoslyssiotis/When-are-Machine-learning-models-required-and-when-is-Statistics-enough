from scipy.stats import entropy
import pandas as pd
from math import log2
import matplotlib.pyplot as plt

import os
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[3])+'/test/df_results.plk')

#===============================================================================
# Find the variation in the target variable, y with increasing number of training datasets
#===============================================================================
y = df_results["Best model ROC"]
y = y.sample(frac=1)

entropies = []

#Compute Shanon's entropy
for i in range(len(y)): #0,1,...,201
    
    pd_series = pd.Series(y[:i+1])
    count = pd_series.value_counts()
    probabilities = count/(i+1)
    
    entropy = []
    
    for prob in probabilities:
        entropy.append(prob * log2(prob))
    
    entropies.append(-sum(entropy))
        
# Plot the entropy vs. the number of training datasets
training_datasets = list(range(1,203))

plt.figure(dpi = 1200)
plt.xlabel('Number of training datasets', fontsize = 16)
plt.ylabel("Shannon's entropy [bits]", fontsize = 16)
plt.yticks([0,1,2,3,4], [0,1,2,3,4], fontsize = 14)
plt.ylim(0,3.5)
plt.xticks(fontsize = 14)
plt.title("Entropy vs. Number of training datasets", fontsize = 16, fontweight = 'bold')
plt.plot(training_datasets, entropies, c = 'blue')        

