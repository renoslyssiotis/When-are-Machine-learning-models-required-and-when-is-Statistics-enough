import numpy as np
import matplotlib.pyplot as plt

labels = ("Number of Instances", "Number of Features", "Number of Classes",
          "Mean mean", "Mean std", "Coefficient of Variation", "Mean skeweness",
          "Mean kurtosis", "Mean correlation", "Feature Entropy",
          "Class Entropy", "Signal to Noise Ratio")
    
N = 12
ROC = (0.119243, 0.100386, 0, 0, 0.103307, 0.094955, 0.118678, 0.117103,
       0.117222, 0.111626, 0.117480, 0)
PRC = (0.124910, 0.090479, 0, 0, 0.112055, 0.098621, 0.115941, 0.115676,
       0.112361, 0.115390, 0.114567, 0)
f1 = (0.123737, 0.090398, 0, 0, 0.109970, 0.098820, 0.119997, 0.118767, 
      0.108480, 0.123227, 0.106605, 0)

ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, ROC, width, color='silver')
rects2 = ax.bar(ind + width, PRC, width, color='dimgray')
rects3 = ax.bar(ind + 2*width, f1, width, color='k')

# add some text for labels, title and axes ticks
ax.set_ylabel('Feature importance', fontsize = 14)
ax.set_title('Feature importance for each meta-feature (all 3 metrics)', fontsize = 14, weight = 'bold')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(labels, rotation = 50)

ax.legend((rects1[0], rects2[0], rects3[0]), ('AUC-ROC', 'AUC-PRC', 'f1'), loc="lower right")
plt.show()