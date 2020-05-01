import sys, os
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[2]))
from models.classifiers_multiclass import classifier_multiclass

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Pre-processing
df_results = pd.read_pickle(str(p.parents[2])+'/test/df_results.plk')

#Standardise the features (i.e. metafeatures) of the dataset (i.e. metadataset)
X = df_results.iloc[:,:23]
scaler = StandardScaler()
X = pd.DataFrame(data = scaler.fit_transform(X),
                 columns = list(df_results.columns[:23]))

#Transform problem into a multi-label classification problem
models = {'auc_f1_logit': 1,
          'auc_f1_rforest': 2,
          'auc_f1_MLP': 3,
          'auc_f1_bagging': 4,
          'auc_f1_perceptron': 5,
          'auc_f1_ADAboost': 6,
          'auc_f1_dtree': 7,
          'auc_f1_bernoulliNB': 8,
          'auc_f1_LDA': 9,
          'auc_f1_QDA': 10,
          'auc_f1_linearSVC': 11,
          'auc_f1_gaussianNB': 12}

df_results = df_results.replace({'Best model f1': models})
y = df_results['Best model PRC']

f1_performance = {'f1_logit': [],
                  'f1_dtree': [],
                  'f1_rforest': [],
                  'f1_perceptron': [],
                  'f1_MLP': [],
                  'f1_ADAboost': [],
                  'f1_linearSVC': [],
                  'f1_LDA': [],
                  'f1_QDA': [],
                  'f1_gaussianNB': [],
                  'f1_bernoulliNB': [],
                  'f1_bagging': []}

for i in range(1,31):

    #Split dataset into the Training and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #==========================MODEL FITTING=======================================
    new_classifier = classifier_multiclass(X_train, X_test, y_train, y_test)
    results_accuracy, results_f1  = new_classifier.evaluate_metrics()
   
    for k,v in results_f1.items():
        f1_performance[k].append(v)     

iterations = list(range(1,4))

plt.figure(dpi=1200)
plt.plot(iterations, [sum(f1_performance['f1_logit'])/len((f1_performance['f1_logit']))]*3, label='logit')
plt.plot(iterations, [sum(f1_performance['f1_dtree'])/len((f1_performance['f1_logit']))]*3, label='decision tree')
plt.plot(iterations, [sum(f1_performance['f1_rforest'])/len((f1_performance['f1_logit']))]*3, label='random forest')
plt.plot(iterations, [sum(f1_performance['f1_perceptron'])/len((f1_performance['f1_logit']))]*3, label='perceptron')
plt.plot(iterations, [sum(f1_performance['f1_MLP'])/len((f1_performance['f1_logit']))]*3, label='MLP')
plt.plot(iterations, [sum(f1_performance['f1_ADAboost'])/len((f1_performance['f1_logit']))]*3, label='Ada Boost')
plt.plot(iterations, [sum(f1_performance['f1_linearSVC'])/len((f1_performance['f1_logit']))]*3, label='linearSVC')
plt.plot(iterations, [sum(f1_performance['f1_LDA'])/len((f1_performance['f1_logit']))]*3, label='LDA')
plt.plot(iterations, [sum(f1_performance['f1_QDA'])/len((f1_performance['f1_logit']))]*3, label='QDA')
plt.plot(iterations, [sum(f1_performance['f1_gaussianNB'])/len((f1_performance['f1_logit']))]*3, label='gaussianNB')
plt.plot(iterations, [sum(f1_performance['f1_bernoulliNB'])/len((f1_performance['f1_logit']))]*3, label='bernoulliNB')
plt.plot(iterations, [sum(f1_performance['f1_bagging'])/len((f1_performance['f1_logit']))]*3, label='bagging')

plt.legend(loc='best', bbox_to_anchor=(1, 0.95))
plt.title("Average model performances", fontsize=14)
plt.ylabel("Model performance (F1)", fontsize=14)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()
