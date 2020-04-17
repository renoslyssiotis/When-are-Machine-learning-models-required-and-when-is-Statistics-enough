# Feature Selection with Univariate Statistical Tests
import pickle
import pandas as pd

df_results = pd.read_pickle('/Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Test results/df_results.plk')

#===============================================================================
#                     Customised meta-feature set
#===============================================================================

selected_X = df_results.iloc[:,[0, 7, 9, 11, 18,19]]

with open('//Users/renoslyssiotis/Desktop/When-are-ML-models-required-and-when-is-Statistics-enough-/Plots and Figures/Feature Selection/Customised set/customised_X_AUCROC_202.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)