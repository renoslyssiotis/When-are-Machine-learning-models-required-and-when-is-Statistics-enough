import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class preProcessor():
    """
    1. Convert features into dummy variables when needed (drop first column)
    2. Feature scaling for columns that were not one-hot encoded
    """
    
    def __init__(self, df):
        self.df = df
        self.df_imputed = None
        self.df_dummy_ready = None
        self.one_hot_encoded_columns = None
        self.X = df.iloc[:,:-1]
        self.y = df.iloc[:,-1]    

#==========================Bytes --> String====================================
    def bytesToString(self):
        """
       Check if data type is 'bytes' --> turn into string
        """
        
        def decode(row):
            row = str(row, 'utf-8')
            return row
        
        try:
            self.df = self.df.applymap(lambda x: decode(x))
                    
        except:
            pass

#==============================Target Variable --> dummy=======================
#Binary classification: convert categorical values into 0 and 1        
    def targetVariableCheck(self):
        """
        Turn strings into binary variables 0 and 1
        """
      
        list_of_unique_values = list(self.y.unique()) #Gets unique values of y
        
        if list_of_unique_values.sort() == [0,1]: #If unique values [0,1] or [1,0]
            pass
        
        else: #If unique values not [0,1] (i.e. [1,2]), replace with [0,1]
            self.y = self.y.replace(to_replace=[list_of_unique_values[0], list_of_unique_values[1]],
                                value = [0,1])
        
#=========================Dummy Variables======================================
    def dummyVarCheck(self):
        """
        For feature columns:
        1. Remove columns that are empty or have only 1 type of value
        2. One-hot encoding for columns that have 2-4 unique values
        """
    
        self.df_dummy_ready=self.df                #create a copy of the original dataframe
        one_hot_encoded_columns = []
        
        for col in self.df.columns[:-1]:
            try:
                #Remove columns that are empty (0) or have only 1 type of value
                if  0 <= self.df[col].nunique() <= 1:
                    self.df_dummy_ready.drop(col, axis = 1)
                    
                #One-hot encoding   
                elif 2 <= self.df[col].nunique() <= 9:
                    one_hot_encoding = pd.get_dummies(self.df[col], 
                                                      prefix = col, 
                                                      drop_first = True)
                    one_hot_encoded_columns.append(list(one_hot_encoding.columns))
                    self.df_dummy_ready = self.df_dummy_ready.drop(col, axis = 1)
                    self.df_dummy_ready = one_hot_encoding.join(self.df_dummy_ready)
             
                else:
                    pass
        
            except:
                pass
            
        #Get a single list with all of the columns that have been one-hot encoded
        one_hot_encoded_columns = [i for j in one_hot_encoded_columns for i in j]
        
        self.df_dummy_ready = self.df_dummy_ready
        self.one_hot_encoded_columns = one_hot_encoded_columns
        self.y.name = 'y' #Replace the name of the target variable, into 'y'
                   
#=========================Feature Scaling======================================
    def featureScaling(self):   
        """
        1. Scale the feature columns that are continuous
        """
        #Select the feature columns that need to be scaled
        dummy_columns = self.df_dummy_ready.iloc[:, :len(self.one_hot_encoded_columns)]
        features_columns = self.df_dummy_ready.iloc[:, len(self.one_hot_encoded_columns):]
        features_columns = features_columns.iloc[:,:-1] #Remove the target variable, y
        
        #Scale the feature columns that have not been one-hot encoded
        scaler = StandardScaler().fit(features_columns)
        scaled_features = pd.DataFrame(scaler.transform(features_columns),
                                           columns = list(features_columns.columns))
            
            #Add back the dummy variable columns and target variable
        X = dummy_columns.join(scaled_features)
        
        return [dummy_columns, features_columns, scaled_features, X]

#================================EVALUATION====================================  
    def pre_processing(self):
        self.bytesToString()
        self.targetVariableCheck()
        self.dummyVarCheck()
        outputs = self.featureScaling()
        outputs.append(self.y)
        
        return outputs

