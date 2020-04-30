import os, pickle
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 750

from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

with open(str(p.parents[3])+'/test/wine_quality/actual/wine_metafeatures_202.pickle', 'rb') as handle:
    wine_meta_features = pickle.load(handle)
wine_stat_meta_features = [list(wine_meta_features.values())[i] for i in (6,7,10,11,14)]   

with open(str(p.parents[3])+'/test/pump_sensor/actual/sensor_metafeatures_202.pickle', 'rb') as handle:
    sensor_meta_features = pickle.load(handle)
sensor_stat_meta_features = [list(sensor_meta_features.values())[i] for i in (6,7,10,11,14)]    

with open(str(p.parents[3])+'/test/cylindrical_bands/actual/cylinder_metafeatures_202.pickle', 'rb') as handle:
    cylinder_meta_features = pickle.load(handle)
cylinder_stat_meta_features = [list(cylinder_meta_features.values())[i] for i in (6,7,10,11,14)]    

meta_features = [wine_stat_meta_features, sensor_stat_meta_features, cylinder_stat_meta_features]

df = pd.DataFrame(meta_features, columns = ['Mean of mean', 'Mean of standard deviation',
        'Mean kurtosis', "Mean Pearson correlation",
        'Mean Median Absolute Deviation (MAD)'])

col_names = ['Mean of mean', 'Mean of standard deviation',
        'Mean kurtosis', "Mean Pearson correlation",
        'Mean Median Absolute Deviation (MAD)']

scaled_features = df.copy()
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features
category = ["Stat1", "Stat2", "Stat3", "Stat4", "Stat5"]
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
 
#Wine quality
values = list(scaled_features.iloc[0,:])
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Wine quality")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Pump sensor
values = list(scaled_features.iloc[1,:])
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Pump sensor")
ax.fill(angles, values, 'r', alpha=0.1)

# Cylindrical bands
values = list(scaled_features.iloc[2,:])
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Cylindrical bands")
ax.fill(angles, values, 'g', alpha=0.1)


plt.title("Statistical meta-features", y = 1.1, fontweight='bold',size=13)
plt.legend(loc='lower right', bbox_to_anchor=(1.1, -0.2), ncol = 2)


