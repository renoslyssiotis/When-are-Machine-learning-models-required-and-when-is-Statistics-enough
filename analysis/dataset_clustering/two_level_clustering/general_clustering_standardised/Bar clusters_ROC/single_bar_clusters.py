import matplotlib.pyplot as plt
x = ["auc_roc_rforest"]
y = [2]
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(2.5, 4.5)
ax.bar(x, y, width=0.2)
ax.set_xlim(-0.5,0.5)
plt.yticks([0, 1, 2], [0, 1, 2])
plt.title("Cluster 11", weight='bold')
plt.ylabel("Counts of best-performing model (AUC-ROC)")
plt.show()

x = ["auc_roc_bernoulliNB"]
y = [2]
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(2.5, 4.5)
ax.bar(x, y, width=0.2)
ax.set_xlim(-0.5,0.5)
plt.yticks([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
plt.title("Cluster 11", weight='bold')
plt.ylabel("Counts of best-performing model (AUC-ROC)")
plt.show()


