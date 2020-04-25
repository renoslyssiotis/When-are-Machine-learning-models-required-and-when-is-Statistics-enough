import pandas as pd
import matplotlib.pyplot as plt

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
# Find the range of a meta-feature as a function of number of training datasets
#===============================================================================
X = df_results.iloc[:,:23]
X = X.drop(['Dataset 75','Dataset 76','Dataset 77','Dataset 78','Dataset 79'])
X = X.sample(frac=1)

def sequential_range(meta_feature):
    
    meta_feature_list = []
    
    for i in range(2, len(meta_feature)):
        
        max_minus_min = max(list(meta_feature[:i])) - min(list(meta_feature[:i]))
        meta_feature_list.append(max_minus_min)
    
    return meta_feature_list

training_datasets = list(range(2,197))

#General: Number of instances
number_of_instances = X.iloc[:,0]
number_of_instaces_range = sequential_range(number_of_instances)

#Statistical
mean_of_std = X.iloc[:,7]
mean_of_std_range = sequential_range(mean_of_std)

#Information theoretic
mean_of_class_entropy = X.iloc[:,-4]
mean_of_classEntropy_range = sequential_range(mean_of_class_entropy) 
        
#===============================================================================
fig = plt.figure(dpi = 1200)
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

host.set_xlim(0, 200)
host.set_ylim(0, 20000) #number of instances
par1.set_ylim(0, 0.06) #std dev
par2.set_ylim(0, 0.5)

host.set_xlabel("Number of training datasets", fontsize = 14)
host.set_ylabel("Range of number of instances", fontsize = 14)
par1.set_ylabel("Range of mean of \n standard deviation", fontsize = 14)
par2.set_ylabel("Range of mean of \n normalised class entropy", fontsize = 14)

p1, = host.plot(training_datasets, number_of_instaces_range, color='blue',label="Number of instances")
p2, = par1.plot(training_datasets, mean_of_std_range, color='darkorange', label="Mean of standard deviation")
p3, = par2.plot(training_datasets, mean_of_classEntropy_range, color='darkgreen', label="Mean of class entropy")

lns = [p1, p2, p3]
host.legend(handles=lns, loc='lower right', fontsize = 11)

par2.spines['right'].set_position(('outward', 70))      
par2.xaxis.set_ticks([0,50,100,150,200], [0,50,100,150,200])

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())
