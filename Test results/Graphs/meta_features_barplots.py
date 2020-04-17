import sys
sys.path.append('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Meta-learners/KNN Ranking Method')
import pickle
import numpy as np
import matplotlib.pyplot as plt

#=====================META-FEATURE EXTRACTION==================================
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Wine quality test dataset/Actual results/wine_metafeatures.pickle', 'rb') as handle:
    meta_features_wine = pickle.load(handle)
    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Cylinder Bands test dataset/Actual results/cylinder_metafeatures.pickle', 'rb') as handle:
    meta_features_cylinder = pickle.load(handle)
    
with open('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/Pump sensor test dataset/Actual results/sensor_metafeatures.pickle', 'rb') as handle:
    meta_features_pump = pickle.load(handle)
    
#Number of instances
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
x = np.arange(1)
rects1 = ax.bar(x - width/2, round(list(meta_features_wine.values())[0], 2), width, label='Wine quality', color='k')
rects2 = ax.bar(x + width/2, round(list(meta_features_cylinder.values())[0],2), width, label='Cylinder bands', color='dimgray')
rects3 = ax.bar(x + 3*width/2, round(list(meta_features_pump.values())[0],2), width, label='Pump sensor', color='silver')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of Instances', fontsize = 14)
ax.set_title('Number of Instances of each test dataset', fontsize = 14, fontweight = 'bold')
# ax.set_xticks(x)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax.legend(fontsize = 14)
plt.axis('off')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize = 14)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
fig.tight_layout()
plt.show()

