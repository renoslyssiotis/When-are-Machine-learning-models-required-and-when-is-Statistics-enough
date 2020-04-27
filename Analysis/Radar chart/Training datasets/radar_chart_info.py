import pandas as pd
from math import pi
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 750

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')
    
dataset_0 = list(df_results.iloc[0,[-11, -10, -9, -7]])
dataset_1 = list(df_results.iloc[49,[-11, -10, -9, -7]])
dataset_2 = list(df_results.iloc[99,[-11, -10, -9, -7]])
dataset_3 = list(df_results.iloc[149,[-11, -10, -9, -7]])
dataset_4 = list(df_results.iloc[199,[-11, -10, -9, -7]])

data = [dataset_0, dataset_1, dataset_2, dataset_3, dataset_4]

df = pd.DataFrame(data, columns = ['Mean normalised feature entropy',
                                   'Mean normalised class entropy',
                                   'Mean SNR',
                                   'Mean Mutual Information'])

col_names = ['Mean normalised feature entropy',
                                   'Mean normalised class entropy',
                                   'Mean SNR',
                                   'Mean Mutual Information']

scaled_features = df.copy()
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features
category = ["Info1", "Info2", "Info3", "Info4"]
scaled_features.columns = category

# ------- PART 1: Create background
 
# number of variable
apoelara = category
N = len(apoelara)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], apoelara, size=13)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([-2,-1,0,1,2], ["-2","-1","0","1","2"], color="black", size=10)
plt.ylim(-2,2)
 
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable
 
values = list(scaled_features.iloc[0,:])
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Dataset 1")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values = list(scaled_features.iloc[1,:])
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Dataset 2")
ax.fill(angles, values, 'r', alpha=0.1)

# Ind2
values = list(scaled_features.iloc[2,:])
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Dataset 3")
ax.fill(angles, values, 'g', alpha=0.1)

# Ind2
values = list(scaled_features.iloc[3,:])
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Dataset 4")
ax.fill(angles, values, 'c', alpha=0.1)

plt.title("Information-theoretic meta-features", y = 1.1, fontweight='bold', size=13)
plt.legend(loc='lower right', bbox_to_anchor=(0.92, -0.3), ncol = 2)




