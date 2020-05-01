import numpy as np
from scipy import stats
from math import e
import sys, os
current_dir = os.path.realpath(__file__)
from pathlib import PurePath
p = PurePath(current_dir)
sys.path.append(str(p.parents[0]))
from pre_processing import preProcessor
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_decomposition import CCA

class metaFeatures():
    
    def __init__(self, df_processed, df):
        self.df_processed = df_processed
        self.df = df
        self.new_object = preProcessor(self.df_processed)
        [self.dummy_columns, self.features_columns, self.scaled_features, self.X, self.y]= self.new_object.pre_processing()
        self.NumberOfInstances = None
        self.NumberOfFeatures = None
        self.ProportionNumericalFeatures = None
        self.NumberOfDummyFeatures = None
        self.NumberOfClasses = None
        self.ProportionOfLessFrequentClass = None
        
        self.MeanMean = None
        self.MeanStandardDeviation = None
        self.MeanCoefficientOfVariation = None
        self.MeanSkewness = None
        self.MeanKurtosis = None
        self.MeanPearsonCorrelation = None
        self.MeanKendallCorrelation = None
        self.MeanSpearmanCorrelation = None
        self.MeanMedianAbsoluteDeviation = None
        self.InterQuantileRange = None
        self.ProportionFeaturesWithOutliers = None
        self.MeanCanonicalCorrelation = None
        
        self.MeanNormalizedFeatureEntropy = None
        self.NormalizedClassEntropy = None
        self.SignalToNoiseRatio = None
        self.MeanMutualInformation = None
        self.MaxMutualInformation = None
#==============================================================================      
#                           GENERAL        
#==============================================================================      
    def general(self):
        
        self.NumberOfInstances = self.df_processed.shape[0]
        self.NumberOfFeatures = self.df.shape[1] - 1
        self.NumberOfClasses =  self.y.nunique()
        
        def common(a,b): 
            c = [value for value in a if value in b] 
            return c
        
        #-1 because 'y' is always a common column name hence needs to be removed
        self.ProportionNumericalFeatures = ((len(common(list(self.df.columns), list(self.df_processed.columns))))-1)/self.NumberOfFeatures
        self.NumberOfDummyFeatures = (self.df_processed.shape[1]) - len(common(list(self.df.columns), list(self.df_processed.columns)))
        self.ProportionOfLessFrequentClass = (sorted(list(self.df_processed.iloc[:,-1].value_counts()))[0])/(sum(list(self.df_processed.iloc[:,-1].value_counts())))

#==============================================================================      
#                           STATISTICAL        
#==============================================================================      
    def statistical(self):
        
        self.MeanMean = np.mean(self.scaled_features.describe().values[1])
        self.MeanStandardDeviation = np.mean(self.scaled_features.describe().values[2])
        self.MeanCoefficientOfVariation = self.MeanStandardDeviation/self.MeanMean
        self.MeanSkewness = np.mean(self.scaled_features.skew().values)
        self.MeanKurtosis = np.mean(self.scaled_features.kurtosis().values)
        self.MeanPearsonCorrelation = np.mean(np.unique(list(self.scaled_features.corr(method = 'pearson').values)))
        self.MeanKendallCorrelation = np.mean(np.unique(list(self.scaled_features.corr(method = 'kendall').values)))
        self.MeanSpearmanCorrelation = np.mean(np.unique(list(self.scaled_features.corr(method = 'spearman').values)))
        
#==============================================================================        
        def mean_median_absolute_deviation(scaled_features):
            length = len(scaled_features.columns)
            MAD = length * [0]
            
            for col in list(scaled_features.columns):
                MAD.append(stats.median_absolute_deviation(scaled_features[col]))
                
            mean_MAD = np.mean(MAD)
            return mean_MAD
        
        def mean_IQR(scaled_features):
            length = len(scaled_features.columns)
            IQR = length * [0]
            
            for col in list(scaled_features.columns):
                IQR.append(stats.iqr(scaled_features[col]))
                
            mean_IQR = np.mean(IQR)
            return mean_IQR
        
        def proportion_of_features_with_outliers(scaled_features):
            length = len(scaled_features.columns)
            outliers = length * [0]
            
            for col in list(scaled_features.columns):
               Q1= list(scaled_features[col].quantile([0.25,0.75]))[0]
               Q3= list(scaled_features[col].quantile([0.25,0.75]))[1]
               IQR = stats.iqr(scaled_features[col])
               
               mask = scaled_features[col].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR, inclusive = True)
               
               #Check if number of rows is the same, if not, then there's an outlier removed:
               if mask.shape[0] != scaled_features.shape[0]: 
                   outliers.append("OUTLIER")
               else:
                   outliers.append("NO OUTLIER")
                   
            proportion = outliers.count("OUTLIER")/len(outliers)
            return proportion
        
        def mean_canonical_correlations(scaled_features, df):
            
            cca = CCA(1)
            cca.fit(scaled_features, df.iloc[:,-1])
            X_c, Y_c = cca.transform(scaled_features, df.iloc[:,-1])
            
            mean_canonical_correlation = np.mean(X_c)
            return mean_canonical_correlation
            
#==============================================================================      
        self.MeanMedianAbsoluteDeviation = mean_median_absolute_deviation(self.scaled_features)
        self.InterQuantileRange = mean_IQR(self.scaled_features)
        self.ProportionFeaturesWithOutliers = proportion_of_features_with_outliers(self.scaled_features)
        self.MeanCanonicalCorrelation = mean_canonical_correlations(self.scaled_features, self.df)

#==============================================================================      
#                           INFORMATION-THEOREITC
#==============================================================================              
    def informationTheory(self):
        
        def entropy(column, base=e):
          vc = self.df_processed[column].value_counts(normalize=True, sort=False)           
          return -(vc * np.log(vc)/np.log(base)).sum()
      
        entropy_list = []
        for column in list(self.scaled_features):
            entropy_list.append(entropy(column))
    
            
        self.MeanNormalizedFeatureEntropy = np.mean(entropy_list)
        self.NormalizedClassEntropy = entropy('y') 
        self.SignalToNoiseRatio = 1/self.MeanCoefficientOfVariation #reciprocal
        self.MaxMutualInformation = max(mutual_info_classif(np.array(self.scaled_features), np.array(self.y)))
        self.MeanMutualInformation = np.mean(mutual_info_classif(np.array(self.scaled_features), np.array(self.y)))
        
#==============================================================================      
#                           DECISION-TREE BASED
#==============================================================================                   
    def get_meta_features(self):
        self.general()
        self.statistical()
        self.informationTheory()
        
        meta_features = {'Number of Instances' : self.NumberOfInstances, 
                        'Number of Features' : self.NumberOfFeatures,
                        'Proportion of Numerical Features': self.ProportionNumericalFeatures,
                        'Number of Dummy Variables after pre-processing': self.NumberOfDummyFeatures,
                        'Proportion of Less Frequent Class': self.ProportionOfLessFrequentClass,
                        'Number of Classes' : self.NumberOfClasses,
                        'Mean mean' : self.MeanMean, 
                        'Mean std' : self.MeanStandardDeviation, 
                        'Mean Coefficient of Variation' : self.MeanCoefficientOfVariation, 
                        'Mean skewness' : self.MeanSkewness, 
                        'Mean kurtosis' : self.MeanKurtosis, 
                        'Mean Pearson Correlation' : self.MeanPearsonCorrelation,
                        'Mean Kendall Correlation' : self.MeanKendallCorrelation,
                        'Mean Spearman Correlation' : self.MeanSpearmanCorrelation,
                        'Mean Median Absolute Deviation': self.MeanMedianAbsoluteDeviation,
                        'Mean Inter-Quartile Range': self.InterQuantileRange,
                        'Proportion of Scaled Features with Outliers': self.ProportionFeaturesWithOutliers,
                        'Mean Canonical Correlations': self.MeanCanonicalCorrelation,
                        'Mean Normalised Feature Entropy' : self.MeanNormalizedFeatureEntropy,
                        'Normalized Class Entropy' : self.NormalizedClassEntropy,
                        'Signal to Noise ratio' : self.SignalToNoiseRatio,
                        'Max Mutual Information': self.MaxMutualInformation,
                        'Mean Mutual Information': self.MeanMutualInformation}
    
        return meta_features




    

