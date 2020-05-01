import os
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

import pandas as pd
import matplotlib.pyplot as plt


#Pre-processing
df_results = pd.read_pickle(str(p.parents[2])+'/test/df_results.plk')

model = {'auc_f1_bagging': 0,
         'auc_f1_logit': 0,
         'auc_f1_MLP': 0,
         'auc_f1_rforest': 0,
         'auc_f1_ADAboost': 0,
         'auc_f1_perceptron': 0,
         'auc_f1_dtree': 0,
         'auc_f1_linearSVC': 0,
         'auc_f1_QDA': 0,
         'auc_f1_bernoulliNB': 0,
         'auc_f1_gaussianNB': 0,
         'auc_f1_LDA': 0}

for i in df_results['Best model f1']:
    model[i] += 1
    
for key in model.keys():
    model[key.split('_')[-1]] = model.pop(key)
    
model['gaussianNB'] = model.pop('auc_f1_gaussianNB')    
model['LDA'] = model.pop('auc_f1_LDA')

color = 'snow whitesmoke gainsboro lightgrey lightgray silver darkgrey darkgray grey gray dimgrey dimgray'.split(' ')

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = list(model.keys())
sizes = list(model.values())
explode = (0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
plt.figure(dpi=1200)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=200, colors = color)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()